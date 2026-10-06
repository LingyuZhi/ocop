import ocop.runtime.recovery as recovery
import asyncio
import copy
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from ocop.collection.benchmark import BenchmarkConfig, build_manifest, validate_manifest
from ocop.collection.pipeline import run_collection
from ocop.collection.verification import verify
from ocop.runtime.config import canonical_json
from ocop.collection.expansion import ExpansionConfig, development_graphs, run_expansion, verify_suite
from ocop.runtime.llm import RequestLimits, RequestRunner
from ocop.collection.replication import coverage, holm_adjust, paired_statistics, prepare_snapshot, replication_report, run_replication, select_pairs
from ocop.runtime.storage import ReadStore, RunStore, StoreConflict
from ocop.runtime.storage import digest
from tests.support import SOURCE, collect, local_environment, reply, collection_runtime as runtime


def prepared(tmp_path, required=2, maximum=3):
    config = runtime(tmp_path)
    task = {"task_id": "task", "question": "Question", "reference_answer": "#### 5", "split": "eval"}
    chain, empty = development_graphs()
    snapshot = prepare_snapshot(config, {"task": task}, [{"task_id": "task", "left": {"trajectory": chain},
        "right": {"trajectory": empty}}], seed=9, required=required, maximum=maximum, kind="development", source={})
    return config, snapshot


def execute(prepared, handler=reply, **kwargs):
    config, snapshot = prepared
    return asyncio.run(run_replication(config, snapshot, "replication", Path("absent.env"),
        transport=httpx.MockTransport(handler), **kwargs))


def path(prepared):
    return Path(prepared[0].artifacts_dir) / "runs/replication"


def test_excluded_holdout_and_old_manifest_compatibility(tmp_path):
    config = runtime(tmp_path, repeats=1, maximum=2)
    old = BenchmarkConfig.model_validate(config.benchmark)
    assert "holdout_tasks" not in old.model_dump() and "excluded_rows" not in old.model_dump()
    old_manifest = build_manifest(SOURCE, old, config.seed)
    validate_manifest(old_manifest, old, config.seed)
    config.benchmark.update(excluded_rows=[t["source_row"] for t in old_manifest["tasks"]], eval_tasks=0, holdout_tasks=3)
    new = BenchmarkConfig.model_validate(config.benchmark)
    manifest = build_manifest(SOURCE, new, config.seed)
    assert len(manifest["tasks"]) == 4
    assert not {t["source_row"] for t in manifest["tasks"]} & set(new.excluded_rows)
    validate_manifest(manifest, new, config.seed)
    messages = []

    def handler(request):
        messages.append(request.content.decode())
        return reply(request)

    report, code = collect(config, handler)
    assert code == 0 and report["candidate_count"] == 1 and report["request_count"] == 6
    assert verify(Path(config.artifacts_dir) / "runs/test")["passed"]
    holdout = {t["question"] for t in manifest["tasks"] if t["split"] == "holdout"}
    assert all(not any(q in message for q in holdout) for message in messages)


def test_block_execution_and_resume(tmp_path):
    setup = prepared(tmp_path)
    calls = []

    def handler(request):
        calls.append(request.content)
        return reply(request)

    initial, code = execute(setup, handler, prepare_only=True)
    assert code == 0 and not initial["finished"] and not calls
    result, code = execute(setup, handler)
    assert code == 0 and result["finished"] and result["integrity_passed"]
    assert len(calls) == result["request_count"] == 20
    assert result["comparisons"][0]["paired_blocks"] == [0, 1]
    assert result["comparisons"][0]["mcnemar_exact_p"] == 1
    assert execute(setup, handler)[1] == 0 and len(calls) == 20
    changed = copy.deepcopy(setup[1])
    changed["seed"] += 1
    with pytest.raises(StoreConflict):
        execute((setup[0], changed))


@pytest.mark.parametrize("kind", ["execution_result", "replication_repeat", "block_done", "label", "candidate_result"])
def test_block_persistence_interruption(tmp_path, monkeypatch, kind):
    setup = prepared(tmp_path)
    original = RunStore.put_record
    interrupted = False
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return reply(request)

    def wrapped(store, record_kind, *args, **kwargs):
        nonlocal interrupted
        result = original(store, record_kind, *args, **kwargs)
        if record_kind == kind and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted")
        return result

    monkeypatch.setattr(RunStore, "put_record", wrapped)
    with pytest.raises(RuntimeError, match="interrupted"):
        execute(setup, handler)
    result, code = execute(setup, handler)
    assert code == 0 and result["integrity_passed"] and calls == 20


def test_incomplete_and_missing_pairs(tmp_path):
    setup = prepared(tmp_path)
    failed = False

    def handler(request):
        nonlocal failed
        body = json.loads(request.content)
        if "workers" in json.loads(body["messages"][-1]["content"]) and not failed:
            failed = True
            return reply(request, finish="length")
        return reply(request)

    result, code = execute(setup, handler)
    assert code == 0 and result["integrity_passed"]
    assert result["repeat_statuses"] == {"incomplete": 1, "completed": 4}
    paired = result["comparisons"][0]
    assert paired["paired_count"] == 1
    assert len(paired["left_only_blocks"]) == len(paired["right_only_blocks"]) == 1


def test_all_failed_is_bounded(tmp_path):
    setup = prepared(tmp_path, required=1, maximum=2)
    result, code = execute(setup, lambda r: reply(r, finish="length"))
    assert code == 0 and result["finished"]
    assert all(c["label"]["status"] == "incomplete" for c in result["candidates"])
    assert result["comparisons"][0]["paired_count"] == 0
    assert result["comparisons"][0]["mcnemar_exact_p"] is None
    assert sum(result["repeat_statuses"].values()) == 4


def test_statistics_known_discordance_and_holm():
    result = paired_statistics({i: True for i in range(10)}, {i: False for i in range(10)}, 42)
    assert result["mcnemar_exact_p"] == pytest.approx(2 / 1024)
    assert result["difference"] == 1 and result["bootstrap_percentile_95_ci"] == [1, 1]
    assert result == paired_statistics({i: True for i in range(10)}, {i: False for i in range(10)}, 42)
    rows = [{"mcnemar_exact_p": p} for p in [0.03, None, 0.01]]
    holm_adjust(rows)
    assert [r["holm_adjusted_p"] for r in rows] == [0.06, None, 0.03]


def test_shared_concurrency_across_runs(tmp_path):
    config = runtime(tmp_path)
    config.requests.concurrency = 2
    active = peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.002)
        active -= 1
        return reply(request)

    async def work():
        limits = RequestLimits(2)
        with RunStore(tmp_path, {}, run_id="a") as a, RunStore(tmp_path, {}, run_id="b") as b:
            async with RequestRunner(a, config.requests, {"OPENAI_API_KEY": "key"}, shared_limits=limits,
                    transport=httpx.MockTransport(handler)) as ra, RequestRunner(b, config.requests,
                    {"OPENAI_API_KEY": "key"}, shared_limits=limits, transport=httpx.MockTransport(handler)) as rb:
                assert ra._cooldowns is rb._cooldowns
                await asyncio.gather(*(runner.call(config.worker_model, key=str(i), messages=[{"role": "user", "content": "q"}])
                                       for runner in (ra, rb) for i in range(5)))
    asyncio.run(work())
    assert peak == 2


def test_selection_pools_same_graph_and_excludes_holdout():
    class View:
        def record_items(self, kind):
            return data[kind]

        def get_record(self, kind, key):
            return {"tasks": [{"task_id": "q", "split": "train"}]}

    data = {"candidate": [], "label": [], "graph": []}
    for cid, graph, success in [("a", "chain", 5), ("b", "chain", 0), ("c", "empty", 3)]:
        data["candidate"].append({"id": cid, "payload": {"task_id": "q", "split": "train"}})
        data["label"].append({"key": cid, "payload": {"status": "complete", "success_count": success,
                                                      "complete_count": 5, "mean_outcome": success / 5}})
        data["graph"].append({"payload": {"candidate_id": cid, "fingerprint": graph, "workers": [], "edges": []}})
    selected = select_pairs(View(), 42, 20)
    assert len(selected) == 1 and selected[0]["gap"] == 0.1
    assert selected[0]["high"]["fingerprint"] == "empty"
    assert selected[0]["low"]["complete_count"] == 10
    assert coverage(View())["tasks_with_multiple_graphs"] == 1


def test_collection_transport_recovery_preserves_completed_data(tmp_path):
    config = runtime(tmp_path, candidates=3, repeats=1, maximum=3, concurrency=1)
    config.requests.consecutive_exhausted_request_limit = 2
    finalizers = 0
    failures = 0

    def offline(request):
        nonlocal finalizers, failures
        body = json.loads(request.content)
        if body["model"] == "deepseek-flash":
            return reply(request)
        if finalizers:
            errors = [TimeoutError, httpx.ReadError, httpx.ConnectError]
            error = errors[failures % len(errors)]
            failures += 1
            raise error("transport unavailable")
        if "workers" in json.loads(body["messages"][-1]["content"]):
            finalizers += 1
        return reply(request)

    report, status = collect(config, offline)
    assert status == 1 and report["halt_reason"] == "consecutive_request_failures"
    source = Path(config.artifacts_dir) / "runs/test"
    before = ReadStore(source)
    module = recovery

    def failed_probe(config, **kwargs):
        raise httpx.ConnectError("still unavailable")

    with pytest.raises(httpx.ConnectError):
        module.recover_connections(source, config, before.archive["config"], probe=failed_probe)
    assert not ReadStore(source).record_items("recovery")
    module.recover_connections(source, config, before.archive["config"],
        probe=lambda config, **kwargs: [{"http_status": 200, "models": [config.worker_model.model_id]}])
    recovered = ReadStore(source)
    assert recovered.rows("attempts") == before.rows("attempts")
    assert recovered.rows("requests") == before.rows("requests")
    assert recovered.record_items("label") == before.record_items("label")
    assert len(recovered.record_items("recovery")) == 1
    assert recovered.rows("run")[0]["halt_reason"] is None
    result, code = collect(config, reply)
    assert code == 0 and result["finished"] and verify(source)["passed"]
    assert all(len(ReadStore(source).attempts_for(r["id"])) <= config.requests.max_attempts
               for r in recovered.rows("requests"))


def test_collection_connection_recovery_rejects_http_failures(tmp_path):
    config = runtime(tmp_path, candidates=3, repeats=1, maximum=3, concurrency=1)
    config.requests.consecutive_exhausted_request_limit = 2

    def unavailable(request):
        if json.loads(request.content)["model"] == "deepseek-flash":
            return reply(request)
        return httpx.Response(503)

    assert collect(config, unavailable)[1] == 1
    source = Path(config.artifacts_dir) / "runs/test"
    view = ReadStore(source)
    with pytest.raises(StoreConflict):
        recovery.recover_connections(source, config, view.archive["config"],
            probe=lambda config, **kwargs: pytest.fail("HTTP failures must not probe"))
    assert not ReadStore(source).record_items("recovery")


def test_collection_recovery_probes_both_failed_providers(tmp_path):
    config = runtime(tmp_path, candidates=3, repeats=1, maximum=2, concurrency=1)
    config.requests.consecutive_exhausted_request_limit = 3
    calls = 0

    def offline(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return reply(request, graph=json.dumps(development_graphs()[0]))
        raise httpx.ConnectError("both providers unavailable")

    assert collect(config, offline)[1] == 1
    source = Path(config.artifacts_dir) / "runs/test"
    before = ReadStore(source)
    probed = []

    def probe(config, *, providers, credentials_path):
        probed.append(providers)
        return [{"http_status": 200, "models": [config.worker_model.model_id, config.strong_model.model_id]}]

    recovery.recover_connections(source, config, before.archive["config"], probe=probe)
    assert probed == [{"openrouter", "deepseek"}]
    after = ReadStore(source)
    assert after.rows("requests") == before.rows("requests")
    assert after.rows("attempts") == before.rows("attempts")
    assert after.record_items("label") == before.record_items("label")
    assert after.rows("run")[0]["halt_reason"] is None


def test_recovery_model_probes_authenticate_and_validate_both_services(tmp_path, monkeypatch):
    config = runtime(tmp_path)
    module = recovery
    client = httpx.Client
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        if request.url.host == "api.deepseek.com":
            assert request.headers["Authorization"] == "Bearer test-key"
            model_id = config.strong_model.model_id
        else:
            assert "Authorization" not in request.headers
            model_id = config.worker_model.model_id
        return httpx.Response(200, json={"data": [{"id": model_id}]})

    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    evidence = module.probe_models(config, providers={"openrouter", "deepseek"}, credentials_path=Path("absent.env"))
    assert set(hosts) == {"api.deepseek.com", "openrouter.ai"}
    assert {row["provider"] for row in evidence} == {"openrouter", "deepseek"}
    assert "test-key" not in json.dumps(evidence)


def test_report_detects_tampered_score(tmp_path):
    setup = prepared(tmp_path)
    execute(setup)
    with sqlite3.connect(path(setup) / "records.sqlite3") as db:
        rid, blob = db.execute("SELECT id,payload_blob FROM records WHERE kind='replication_repeat' LIMIT 1").fetchone()
        payload = json.loads((path(setup) / "blobs" / blob).read_bytes())
        payload["score"]["normalized_prediction"] = "fake"
        raw = canonical_json(payload)
        checksum = digest(payload)
        (path(setup) / "blobs" / checksum).write_bytes(raw)
        db.execute("UPDATE records SET payload_blob=? WHERE id=?", (checksum, rid))
    assert not replication_report(path(setup))["checks"]["scores"]


@pytest.mark.parametrize("different_graphs", [False, True])
def test_expansion_end_to_end(tmp_path, different_graphs):
    config = runtime(tmp_path, repeats=1, maximum=2)
    collect(config, reply)
    source = Path(config.artifacts_dir) / "runs/test"
    old = ReadStore(source).get_record("task_manifest", "tasks")
    settings = ExpansionConfig(version="ocop.expansion.v1", source_collection=str(source), seed=6,
        train_tasks=2, holdout_tasks=2, candidates_per_task=2, complete_repeats=1, max_repeats=2,
        development_task_ids=[t["task_id"] for t in old["tasks"] if t["split"] == "eval"],
        development_repeats=1, development_max_repeats=2, selection_limit=2, replication_repeats=2, replication_max_repeats=3,
        report_every=1)
    proposals = 0

    def handler(request):
        nonlocal proposals
        body = json.loads(request.content)
        if not different_graphs:
            return reply(request)
        if body["model"] == "deepseek-flash":
            graph = development_graphs()[proposals % 2]
            proposals += 1
            return reply(request, graph=json.dumps(graph))
        inputs = json.loads(body["messages"][-1]["content"])
        if "workers" in inputs:
            correct = inputs["workers"][1]["output"] == "#### 5"
        else:
            correct = bool(inputs["predecessors"])
        return reply(request, answer="#### 5" if correct else "#### 0")

    result, code = asyncio.run(run_expansion(settings, "suite", Path("absent.env"), transport=httpx.MockTransport(handler)))
    assert code == 0 and result["integrity_passed"] and result["execution_finished"]
    assert result["coverage"]["tasks_with_multiple_graphs"] == (2 if different_graphs else 0)
    assert result["replications"]["replication"]["planned"] == (4 if different_graphs else 0)
    root = Path(config.artifacts_dir) / "experiments/suite"
    assert verify_suite(root)["integrity_passed"]
    resumed, code = asyncio.run(run_expansion(settings, "suite", Path("absent.env"), transport=httpx.MockTransport(
        lambda r: pytest.fail("Duplicate call"))))
    assert code == 0 and resumed["integrity_passed"]
