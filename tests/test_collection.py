import asyncio
import copy
import hashlib
import json
import runpy
from pathlib import Path

import httpx
import pytest

import ocop.collection as collection
import ocop.llm as llm
from ocop.benchmark import BenchmarkConfig, build_manifest, validate_manifest
from ocop.cli import main
from ocop.config import canonical_json, load_config
from ocop.graph import replay
from ocop.labels import aggregate_label
from ocop.storage import RunStore, StoreConflict, read_database


ROOT = Path(__file__).resolve().parents[1]
VALID_GRAPH = (ROOT / "examples/graph/valid.json").read_text()
SOURCE = [{"question": f"Question {i}", "answer": "ReferenceSecret calculation\n#### 5"} for i in range(50)]


def runtime(tmp_path, *, candidates=1, repeats=5, maximum=8, concurrency=2):
    config = load_config(ROOT / "config/prototype.json")
    config.artifacts_dir = str(tmp_path / "artifacts")
    config.benchmark.update(train_tasks=1, eval_tasks=1)
    config.collection.update(candidates_per_task=candidates, complete_repeats=repeats,
                             max_repeats=maximum, candidate_concurrency=concurrency)
    config.requests.retry_backoff_seconds = 0.0
    return config


@pytest.fixture(autouse=True)
def local_environment(monkeypatch):
    monkeypatch.setattr(collection, "load_source", lambda config, artifacts: SOURCE)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def reply(request, *, graph=VALID_GRAPH, answer="#### 5", finish="stop", reasoning="Design reasoning", usage=True):
    body = json.loads(request.content)
    strong = body["model"] == "deepseek-flash"
    return httpx.Response(200, json={"model": body["model"], "provider": "OpenAI",
        "choices": [{"finish_reason": finish, "message": {"content": graph if strong else answer,
                                                          "reasoning_content": reasoning if strong else None}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5} if usage else None})


def collect(config, handler, *, manifest_only=False):
    return asyncio.run(collection.run_collection(config, Path("absent.env"), "test",
        manifest_only=manifest_only, transport=httpx.MockTransport(handler)))


def payloads(config, kind):
    path = Path(config.artifacts_dir) / "runs/test"
    with read_database(path) as db:
        return [json.loads((path / "blobs" / row[0]).read_bytes())
                for row in db.execute("SELECT payload_blob FROM records WHERE kind=? ORDER BY rowid", (kind,))]


def test_manifest_sampling_split_and_integrity():
    config = BenchmarkConfig.model_validate(load_config(ROOT / "config/prototype.json").benchmark)
    first = build_manifest(SOURCE, config, 42)
    assert first == build_manifest(SOURCE, config, 42)
    assert first != build_manifest(SOURCE, config, 43)
    assert len(first["tasks"]) == len({task["source_row"] for task in first["tasks"]}) == 30
    assert [task["split"] for task in first["tasks"]] == ["train"] * 24 + ["eval"] * 6
    validate_manifest(first, config, 42)
    changed = copy.deepcopy(first)
    changed["tasks"][0]["question"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        validate_manifest(changed, config, 42)
    changed = copy.deepcopy(first)
    changed["tasks"][0]["split"] = "eval"
    changed["hash"] = hashlib.sha256(canonical_json({k: v for k, v in changed.items() if k != "hash"})).hexdigest()
    with pytest.raises(ValueError, match="split"):
        validate_manifest(changed, config, 42)


def test_bad_reference_stops_before_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(collection, "load_source", lambda config, artifacts: [{"question": "q", "answer": "not numeric"}] * 50)
    with pytest.raises(ValueError, match="reference"):
        collect(runtime(tmp_path), lambda request: pytest.fail("Unexpected API call"))


def test_manifest_only_then_offline_manifest_restore(tmp_path, monkeypatch):
    config = runtime(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    report, status = collect(config, lambda request: pytest.fail("Unexpected API call"), manifest_only=True)
    assert status == 0 and report["phase"] == "manifest_ready"
    assert report["request_count"] == 0
    monkeypatch.setattr(collection, "load_source", lambda *args: pytest.fail("Source reloaded"))
    assert collect(config, lambda request: pytest.fail("Unexpected API call"), manifest_only=True)[0] == report
    changed = config.model_copy(deep=True)
    changed.seed = 43
    with pytest.raises(StoreConflict):
        collect(changed, reply, manifest_only=True)


def repeat(index, status="completed", reason="success"):
    return {"execution_id": f"e{index}", "repeat_id": str(index), "executor_hash": "same",
            "status": status, "score": {"success": reason == "success", "reason": reason} if status == "completed" else None}


def test_labels_use_only_complete_repeats_and_keep_zero():
    values = [repeat(0, "incomplete"), repeat(1, "infra_failed"), repeat(2, reason="format_error"),
              repeat(3, reason="wrong_answer"), repeat(4), repeat(5), repeat(6)]
    label = aggregate_label(values)
    assert label["status"] == "complete" and label["mean_outcome"] == 0.6
    assert label["complete_repeat_ids"] == ["2", "3", "4", "5", "6"]
    assert aggregate_label([repeat(i, reason="wrong_answer") for i in range(5)])["mean_outcome"] == 0.0
    assert aggregate_label([repeat(i, "incomplete") for i in range(8)])["mean_outcome"] is None
    assert aggregate_label([repeat(i) for i in range(4)])["status"] == "pending"
    assert aggregate_label([repeat(i) for i in range(4)] + [repeat(i, "incomplete") for i in range(4, 8)])["status"] == "incomplete"


def test_label_rejects_duplicate_or_mixed_results():
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_label([repeat(0), repeat(0)])
    with pytest.raises(ValueError, match="configurations"):
        aggregate_label([repeat(0), {**repeat(1), "executor_hash": "changed"}])
    with pytest.raises(ValueError, match="score"):
        aggregate_label([{**repeat(0), "score": None}])
    with pytest.raises(ValueError, match="Too many"):
        aggregate_label([repeat(i) for i in range(6)])


def test_full_collection_duplicate_graphs_split_usage_and_resume(tmp_path):
    config = runtime(tmp_path, candidates=3, repeats=2, maximum=3)
    captured = []
    def handler(request):
        captured.append(request)
        assert b"ReferenceSecret" not in request.content
        return reply(request)
    report, status = collect(config, handler)
    assert status == 0 and report["finished"]
    assert len(captured) == report["request_count"] == 66
    assert report["all_legal_candidates_labeled"] is True
    for split in ("train", "eval"):
        summary = report["splits"][split]
        assert summary["candidate_count"] == summary["complete_labels"] == 3
        assert summary["duplicate_candidates"] == 2
        assert summary["duplicate_rate_among_legal"] == 2 / 3
        assert summary["label_distribution"] == {"1.0": 3}
        assert summary["repeat_count"] == 6
    proposal_requests = [json.loads(request.content) for request in captured if json.loads(request.content)["model"] == "deepseek-flash"]
    assert len(proposal_requests) == 6
    for request in proposal_requests:
        assert request["messages"] == collection.proposal_messages(request["messages"][1]["content"])
    assert len({row["execution_id"] for row in payloads(config, "collection_repeat")}) == 12
    assert len(payloads(config, "graph")) == 6
    assert report["usage"]["proposal"]["known_usage_sum"]["total_tokens"] == 30
    assert report["usage"]["worker"]["known_usage_sum"]["total_tokens"] == 240
    assert report["usage"]["finalizer"]["known_usage_sum"]["total_tokens"] == 60
    resumed, status = collect(config, lambda request: pytest.fail("Unexpected API call"))
    assert status == 0 and resumed == report


@pytest.mark.parametrize("after_topup", [False, True])
def test_explicit_budget_recovery_preserves_history_and_repeat_budget(tmp_path, monkeypatch, after_topup):
    config = runtime(tmp_path, repeats=2, maximum=3, concurrency=1)
    original = llm.normalize_response
    def legacy(body, status_code, *, catalog):
        result = original(body, status_code, catalog=catalog)
        if result["error"] == "in_flight_budget_exhausted":
            result.update(status="fatal", error="http_402")
        return result
    monkeypatch.setattr(llm, "normalize_response", legacy)
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 17:
            return httpx.Response(402, headers={"Retry-After": "0"}, json={"error": {"code": 402,
                "metadata": {"reason": "in_flight_budget_exhausted",
                    "limit_source": "openrouter_credits" if after_topup else "openrouter_in_flight_budget"}}})
        return reply(request)
    report, status = collect(config, handler)
    assert status == 1 and report["terminal_candidates"] == 1
    labels = payloads(config, "label")
    repeats = payloads(config, "collection_repeat")
    path = Path(config.artifacts_dir) / "runs/test"
    with read_database(path) as db:
        before = dict(db.execute("SELECT id,result_json FROM attempts"))
        frozen = db.execute("SELECT config_hash,manifest_blob FROM run").fetchone()
    monkeypatch.setattr(llm, "normalize_response", original)
    report, status = asyncio.run(collection.run_collection(config, Path("absent.env"), "test",
        resume_inflight_budget=not after_topup, resume_after_topup=after_topup, transport=httpx.MockTransport(reply)))
    assert status == 0 and report["finished"]
    assert payloads(config, "label")[:1] == labels
    assert payloads(config, "collection_repeat")[:len(repeats)] == repeats
    assert sorted(label["repeat_count"] for label in payloads(config, "label")) == [2, 3]
    assert len(payloads(config, "recovery")) == 1
    with read_database(path) as db:
        after = dict(db.execute("SELECT id,result_json FROM attempts"))
        assert all(after[key] == value for key, value in before.items())
        assert tuple(db.execute("SELECT config_hash,manifest_blob FROM run").fetchone()) == tuple(frozen)
    _, status = asyncio.run(collection.run_collection(config, Path("absent.env"), "test",
        resume_inflight_budget=not after_topup, resume_after_topup=after_topup,
        transport=httpx.MockTransport(lambda request: pytest.fail("Unexpected API call"))))
    assert status == 0 and len(payloads(config, "recovery")) == 1


@pytest.mark.parametrize(("mode", "expected"), [("invalid", "proposal_invalid"), ("length", "proposal_incomplete"), ("infra", "proposal_infra_failed")])
def test_unusable_proposals_do_not_execute_or_regenerate(tmp_path, mode, expected):
    config = runtime(tmp_path)
    calls = []
    def handler(request):
        calls.append(request)
        assert json.loads(request.content)["model"] == "deepseek-flash"
        if mode == "infra":
            return httpx.Response(503)
        return reply(request, graph="{}" if mode == "invalid" else VALID_GRAPH, finish="length" if mode == "length" else "stop")
    report, status = collect(config, handler)
    assert status == 0 and report["finished"]
    assert {item["status"] for item in report["candidates"]} == {expected}
    assert not payloads(config, "execution")
    assert not payloads(config, "label")
    assert len(calls) == (6 if mode == "infra" else 2)
    assert collect(config, lambda request: pytest.fail("Unexpected API call"))[0] == report


@pytest.mark.parametrize("mode", ["auth", "model", "reasoning"])
def test_fatal_proposal_stops_run_and_preserves_response(tmp_path, mode):
    config = runtime(tmp_path, concurrency=1)
    def handler(request):
        if mode == "auth":
            return httpx.Response(401)
        result = reply(request, reasoning=None if mode == "reasoning" else "reasoning")
        if mode == "model":
            content = result.json()
            content["model"] = "wrong-model"
            return httpx.Response(200, json=content)
        return result
    report, status = collect(config, handler)
    assert status == 1 and report["halt_reason"]
    assert report["request_count"] == 1
    assert len(payloads(config, "proposal_result")) == 1
    assert not payloads(config, "execution")


def test_incomplete_repeats_exhaust_eight_without_label_mean(tmp_path):
    config = runtime(tmp_path)
    def handler(request):
        strong = json.loads(request.content)["model"] == "deepseek-flash"
        return reply(request, finish="stop" if strong else "length")
    report, status = collect(config, handler)
    assert status == 0 and report["finished"]
    assert report["all_legal_candidates_labeled"] is False
    for label in payloads(config, "label"):
        assert label["status"] == "incomplete"
        assert label["repeat_count"] == 8 and label["complete_count"] == 0
        assert label["mean_outcome"] is None
    assert len(payloads(config, "execution")) == 16


@pytest.mark.parametrize("answer", ["#### 6", "unformatted"])
def test_wrong_and_format_answers_count_as_complete_without_extra_repeats(tmp_path, answer):
    config = runtime(tmp_path)
    report, status = collect(config, lambda request: reply(request, answer=answer))
    assert status == 0 and report["request_count"] == 52
    for label in payloads(config, "label"):
        assert label["status"] == "complete" and label["mean_outcome"] == 0
        assert label["repeat_count"] == label["complete_count"] == 5


@pytest.mark.parametrize("kind", ["proposal_result", "score", "collection_repeat", "label", "candidate_result"])
def test_recovery_between_persistence_boundaries(tmp_path, monkeypatch, kind):
    config = runtime(tmp_path, repeats=1, maximum=2, concurrency=1)
    put = RunStore.put_record
    interrupted = False
    calls = []
    def interrupt(store, record_kind, *args, **kwargs):
        nonlocal interrupted
        if record_kind == kind and not interrupted:
            interrupted = True
            raise asyncio.CancelledError()
        return put(store, record_kind, *args, **kwargs)
    def handler(request):
        calls.append(request)
        return reply(request)
    monkeypatch.setattr(RunStore, "put_record", interrupt)
    with pytest.raises(asyncio.CancelledError):
        collect(config, handler)
    assert not payloads(config, "candidate_result")
    monkeypatch.setattr(RunStore, "put_record", put)
    report, status = collect(config, handler)
    assert status == 0 and report["finished"]
    assert len(calls) == report["request_count"] == 12
    assert len(payloads(config, "label")) == 2


def test_global_concurrency_and_candidate_limit(tmp_path, monkeypatch):
    config = runtime(tmp_path, candidates=2, repeats=1, maximum=1)
    config.requests.concurrency = 2
    active = maximum = 0
    active_candidates = maximum_candidates = 0
    actual_candidate = collection.collect_candidate
    async def wrapped_candidate(*args, **kwargs):
        nonlocal active_candidates, maximum_candidates
        active_candidates += 1
        maximum_candidates = max(maximum_candidates, active_candidates)
        try:
            return await actual_candidate(*args, **kwargs)
        finally:
            active_candidates -= 1
    monkeypatch.setattr(collection, "collect_candidate", wrapped_candidate)
    async def handler(request):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.001)
        active -= 1
        return reply(request)
    report, status = collect(config, handler)
    assert status == 0 and maximum == 2
    assert maximum_candidates == 2
    assert len(payloads(config, "label")) == 4
    assert len(report["usage"]["proposal"]["request_ids"]) == 4


def test_fingerprint_ignores_edge_addition_order():
    document = json.loads(VALID_GRAPH)
    document["steps"][4:6] = reversed(document["steps"][4:6])
    other = replay(json.dumps(document))
    assert other.valid
    assert collection.graph_fingerprint(other.final_graph) == collection.graph_fingerprint(replay(VALID_GRAPH).final_graph)


def test_collect_cli_manifest_only(tmp_path, capsys):
    config = runtime(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    assert main(["collect", "--config", str(path), "--manifest-only", "--run-id", "test"]) == 0
    assert json.loads(capsys.readouterr().out)["phase"] == "manifest_ready"


def test_source_loading_preserves_declared_dataset_splits(tmp_path, monkeypatch):
    import datasets
    from ocop.benchmark import load_source

    local = tmp_path / "dataset"
    (local / "main").mkdir(parents=True)
    for split, size in (("train", 2), ("test", 1)):
        datasets.Dataset.from_list(SOURCE[:size]).to_parquet(local / "main" / f"{split}-00000-of-00001.parquet")
    metadata = {"configs": [{"config_name": "main", "data_files": [
        {"split": split, "path": f"main/{split}-*.parquet"} for split in ("train", "test")]}],
        "dataset_info": [{"config_name": "main", "features": [
            {"name": name, "dtype": "string"} for name in ("question", "answer")],
            "splits": [{"name": "train", "num_examples": 2}, {"name": "test", "num_examples": 1}]}]}
    (local / "README.md").write_text("---\n" + json.dumps(metadata) + "\n---\n")
    actual_load = datasets.load_dataset
    monkeypatch.setattr(datasets, "load_dataset", lambda path, *args, **kwargs: actual_load(str(local), *args, **kwargs))
    config = BenchmarkConfig.model_validate(load_config(ROOT / "config/prototype.json").benchmark)
    source = load_source(config, tmp_path / "cache")
    assert len(source) == 2
    assert source[0] == SOURCE[0]


def test_collection_verifier_recomputes_labels_and_rejects_corruption(tmp_path):
    import sqlite3

    config = runtime(tmp_path, repeats=1, maximum=2)
    verify = runpy.run_path(str(ROOT / "scripts/verify_collection.py"))["verify"]
    collect(config, reply, manifest_only=True)
    path = Path(config.artifacts_dir) / "runs/test"
    assert verify(path, allow_pending=True)["passed"]
    assert not verify(path)["passed"]
    collect(config, reply)
    report = verify(path)
    assert report["passed"] and report["finished"]
    assert report["label_count"] == 2
    assert report["request_count"] == report["attempt_count"] == 12
    with sqlite3.connect(path / "records.sqlite3") as database:
        row = database.execute("SELECT id,payload_blob FROM records WHERE kind='label' LIMIT 1").fetchone()
        payload = json.loads((path / "blobs" / row[1]).read_bytes())
        payload["mean_outcome"] = 0.0
        raw = canonical_json(payload)
        digest = hashlib.sha256(raw).hexdigest()
        (path / "blobs" / digest).write_bytes(raw)
        database.execute("UPDATE records SET payload_blob=? WHERE id=?", (digest, row[0]))
    assert not verify(path)["checks"]["labels_recomputed"]


def test_failed_repeat_is_replaced_within_budget(tmp_path):
    config = runtime(tmp_path, concurrency=1)
    worker_failures = 0
    incomplete_finalizers = 0
    def handler(request):
        nonlocal worker_failures, incomplete_finalizers
        body = json.loads(request.content)
        if body["model"] == "deepseek-flash":
            return reply(request)
        inputs = json.loads(body["messages"][1]["content"])
        if body["messages"][0]["content"].startswith("Break") and worker_failures < 3:
            worker_failures += 1
            return httpx.Response(503)
        if "workers" in inputs and incomplete_finalizers == 0:
            incomplete_finalizers += 1
            return reply(request, finish="length")
        return reply(request)
    report, status = collect(config, handler)
    assert status == 0
    labels = payloads(config, "label")
    assert [label["repeat_count"] for label in labels] == [7, 5]
    assert all(label["complete_count"] == label["success_count"] == 5 for label in labels)
    assert report["splits"]["train"]["repeat_states"] == {"infra_failed": 1, "incomplete": 1, "completed": 5}


def test_malformed_proposal_content_is_preserved_as_incomplete(tmp_path):
    config = runtime(tmp_path)
    report, status = collect(config, lambda request: reply(request, graph=[{"text": "partial"}], finish="length"))
    assert status == 0
    assert all(item["status"] == "proposal_incomplete" for item in report["candidates"])
    assert all(item["content"] == [{"text": "partial"}] for item in payloads(config, "proposal_result"))


def test_collector_preserves_global_exhaustion_limit(tmp_path):
    config = runtime(tmp_path, candidates=3, concurrency=1)
    config.requests.consecutive_exhausted_request_limit = 2
    report, status = collect(config, lambda request: httpx.Response(503))
    assert status == 1 and report["halt_reason"] == "consecutive_request_failures"
    assert report["request_count"] == 2
    assert len(report["usage"]["proposal"]["attempt_ids"]) == 6
    assert report["terminal_candidates"] == 2
