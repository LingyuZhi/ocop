import asyncio
import json
from pathlib import Path

import httpx
import pytest
from filelock import FileLock, Timeout

from ocop.collection.expansion import ExpansionConfig, development_graphs, prepare, supervise_expansion
from ocop.runtime.llm import recover_inflight_budget
from ocop.runtime.recovery import AutoRecoveryConfig, automatic_recovery, apply_recovery, recovery_evidence
from ocop.runtime.storage import ReadStore, RunStore, StoreConflict
from test_collection import collect, local_environment, reply, runtime


def halted_collection(tmp_path, response=None):
    config = runtime(tmp_path, candidates=3, repeats=1, maximum=3, concurrency=1)
    config.requests.consecutive_exhausted_request_limit = 2

    def handler(request):
        if json.loads(request.content)["model"] == "deepseek-flash":
            return reply(request, graph=json.dumps(development_graphs()[0]))
        if response is not None:
            return httpx.Response(response)
        raise httpx.ConnectError("offline")

    assert collect(config, handler)[1] == 1
    path = Path(config.artifacts_dir) / "runs/test"
    return config, path, ReadStore(path).archive["config"]


def run_recovery(setup, probe, **kwargs):
    config, path, snapshot = setup
    policy = kwargs.pop("settings", AutoRecoveryConfig(initial_delay_seconds=0.0, max_delay_seconds=1.0, max_probe_attempts=3))
    return asyncio.run(automatic_recovery(path, config, snapshot, policy, Path("absent.env"), probe=probe, **kwargs))


def test_backoff_persisted_then_recovers_without_resetting_requests(tmp_path):
    setup = halted_collection(tmp_path)
    before = ReadStore(setup[1])
    now, sleeps, attempts = [100.0], [], []

    async def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    def probe(config, **kwargs):
        attempts.append(kwargs["providers"])
        if len(attempts) < 3:
            raise httpx.ConnectError("offline")
        return [{"http_status": 200, "models": [config.worker_model.model_id]}]

    result = run_recovery(setup, probe, sleep=sleep, clock=lambda: now[0],
        settings=AutoRecoveryConfig(initial_delay_seconds=2.0, max_delay_seconds=5.0, max_probe_attempts=3))
    assert sleeps == [2.0, 4.0, 5.0]
    assert result["policy"] == "ocop.auto_recovery.v1"
    after = ReadStore(setup[1])
    assert after.rows("requests") == before.rows("requests")
    assert after.rows("attempts") == before.rows("attempts")
    assert after.record_items("label") == before.record_items("label")
    assert len(after.record_items("recovery_probe_result")) == 3
    assert after.rows("run")[0]["halt_reason"] is None
    assert collect(setup[0], reply)[1] == 0


def test_failed_probes_remain_bounded_across_restart(tmp_path):
    setup = halted_collection(tmp_path)
    calls = []

    def offline(config, **kwargs):
        calls.append(1)
        raise httpx.ConnectError("still offline")

    for _ in range(2):
        with pytest.raises(StoreConflict, match="budget exhausted"):
            run_recovery(setup, offline)
    assert len(calls) == 3
    assert ReadStore(setup[1]).rows("run")[0]["halt_reason"] == "consecutive_request_failures"
    assert not ReadStore(setup[1]).record_items("recovery")


def test_later_outage_allows_historical_interruption_and_late_success(tmp_path):
    config = runtime(tmp_path)
    config.requests.consecutive_exhausted_request_limit = 2
    model = config.worker_model
    spec = {"provider": model.provider, "url": model.base_url + "/chat/completions",
            "body": {"model": model.model_id}}
    with RunStore(tmp_path, {}, run_id="episodes") as store:
        old = store.request("interrupted", spec, None)
        store.start_attempt(old["id"])
    with RunStore(tmp_path, {}, run_id="episodes") as store:
        store.finish_attempt(store.start_attempt(old["id"]), {"status": "completed", "elapsed_seconds": 0},
                             terminal=True, failure_limit=2)
        episodes = []
        for number in range(2):
            late = store.start_attempt(store.request(f"late-{number}", spec, None)["id"])
            for i in range(2):
                request = store.request(f"failure-{number}-{i}", spec, None)
                for attempt in range(config.requests.max_attempts):
                    store.finish_attempt(store.start_attempt(request["id"]),
                        {"status": "infra_failed", "error": "ConnectError", "elapsed_seconds": 0},
                        terminal=attempt == config.requests.max_attempts - 1, failure_limit=2)
            store.finish_attempt(late, {"status": "completed", "elapsed_seconds": 0}, terminal=True, failure_limit=2)
            assert store.rows("run")[0]["failure_streak"] == 0
            evidence, providers = recovery_evidence(store, config, allow_http=True)
            assert len(evidence) == 2 and providers == {model.provider}
            episodes.append({e["request_id"] for e in evidence})
            apply_recovery(store, evidence, [{"http_status": 200}], policy="test")
        assert not episodes[0] & episodes[1]
        assert len(store.record_items("recovery")) == 2


def test_later_transport_recovery_preserves_historical_topup_fatal(tmp_path):
    config = runtime(tmp_path)
    config.requests.consecutive_exhausted_request_limit = 2
    model = config.worker_model
    spec = {"provider": model.provider, "url": model.base_url + "/chat/completions",
            "body": {"model": model.model_id}}
    payload = {"error": {"message": "Insufficient Balance (request_id: 123e4567-e89b-12d3-a456-426614174000)",
        "type": "unknown_error", "code": "invalid_request_error"}}
    with RunStore(tmp_path, {}, run_id="historical-topup-fatal") as store:
        fatal_spec = {"provider": "deepseek", "url": "https://api.deepseek.com/chat/completions",
            "body": {"model": "deepseek-chat"}, "api_key_env": "DEEPSEEK_API_KEY"}
        fatal = store.request("deepseek-request", fatal_spec, None)
        attempt = store.start_attempt(fatal["id"])
        store.save_response(attempt["id"], body=json.dumps(payload).encode(), status=402,
                            headers={}, elapsed=0.1)
        attempt = store.attempts_for(fatal["id"])[-1]
        store.finish_attempt(attempt, {"status": "fatal", "error": "http_402", "elapsed_seconds": 0.1},
                             terminal=True, failure_limit=1)
        recover_inflight_budget(store, after_topup=True)

        for number in range(config.requests.consecutive_exhausted_request_limit):
            request = store.request(f"transport-failure-{number}", spec, None)
            for attempt_number in range(config.requests.max_attempts):
                store.finish_attempt(store.start_attempt(request["id"]),
                    {"status": "infra_failed", "error": "ConnectError", "elapsed_seconds": 0},
                    terminal=attempt_number == config.requests.max_attempts - 1,
                    failure_limit=config.requests.consecutive_exhausted_request_limit)

        evidence, providers = recovery_evidence(store, config, allow_http=True)
        assert providers == {model.provider}
        assert {item["request_id"] for item in evidence} == {
            store.request(f"transport-failure-{number}", spec, None)["id"]
            for number in range(config.requests.consecutive_exhausted_request_limit)}
        assert store.rows("requests")[0]["state"] == "fatal"


def test_restart_preserves_scheduled_wait(tmp_path):
    setup = halted_collection(tmp_path)
    now, sleeps = [100.0], []
    policy = AutoRecoveryConfig(initial_delay_seconds=120.0)

    async def interrupt(delay):
        now[0] += 20
        raise asyncio.CancelledError

    def probe(config, **kwargs):
        return [{"http_status": 200}]

    with pytest.raises(asyncio.CancelledError):
        run_recovery(setup, probe, sleep=interrupt, clock=lambda: now[0], settings=policy)

    async def resumed_sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    run_recovery(setup, probe, sleep=resumed_sleep, clock=lambda: now[0], settings=policy)
    assert sleeps == [100.0]


@pytest.mark.parametrize("status", [401, 402, 403])
def test_fatal_service_errors_do_not_auto_recover(tmp_path, status):
    setup = halted_collection(tmp_path, status)
    with pytest.raises(StoreConflict, match="consecutive request failure halt"):
        run_recovery(setup, lambda *a, **k: pytest.fail("Must not probe fatal service errors"))


def test_http_503_circuit_recovers_but_probe_401_stops(tmp_path):
    setup = halted_collection(tmp_path, 503)
    calls = []

    def unauthorized(config, **kwargs):
        calls.append(1)
        response = httpx.Response(401, request=httpx.Request("GET", "https://example.test/models"))
        response.raise_for_status()

    with pytest.raises(StoreConflict, match="non-transient"):
        run_recovery(setup, unauthorized)
    assert len(calls) == 1
    assert not ReadStore(setup[1]).record_items("recovery")


def suite_settings(tmp_path):
    config = runtime(tmp_path, repeats=1, maximum=3)
    config.requests.consecutive_exhausted_request_limit = 2
    collect(config, reply)
    path = Path(config.artifacts_dir) / "runs/test"
    tasks = ReadStore(path).get_record("task_manifest", "tasks")["tasks"]
    return config, ExpansionConfig(version="ocop.expansion.v1", source_collection=str(path), seed=6,
        train_tasks=2, holdout_tasks=2, candidates_per_task=3, complete_repeats=1, max_repeats=3,
        development_task_ids=[t["task_id"] for t in tasks if t["split"] == "eval"],
        development_repeats=1, development_max_repeats=3, selection_limit=2,
        replication_repeats=2, replication_max_repeats=3, report_every=1)


def test_supervisor_recovers_collection_and_replication_end_to_end(tmp_path):
    config, settings = suite_settings(tmp_path)
    online, probes = [False], []

    def handler(request):
        if json.loads(request.content)["model"] == "deepseek-flash":
            return reply(request, graph=json.dumps(development_graphs()[0]))
        if not online[0]:
            raise httpx.ConnectError("simulated outage")
        return reply(request)

    def probe(config, **kwargs):
        probes.append(kwargs["providers"])
        if len(probes) == 1:
            raise httpx.ConnectError("not yet online")
        online[0] = True
        return [{"http_status": 200}]

    result, code = asyncio.run(supervise_expansion(settings, "auto", Path("absent.env"),
        recovery_settings=AutoRecoveryConfig(initial_delay_seconds=0.0, max_delay_seconds=1.0, max_probe_attempts=3),
        probe=probe, transport=httpx.MockTransport(handler)))
    assert code == 0 and result["execution_finished"] and result["integrity_passed"]
    assert len(probes) >= 2
    for name in ("collection", "development"):
        view = ReadStore(Path(config.artifacts_dir) / "runs" / f"auto-{name}")
        assert view.record_items("recovery")
        assert all(len(view.attempts_for(r["id"])) <= config.requests.max_attempts for r in view.rows("requests"))
    assert ReadStore(Path(config.artifacts_dir) / "runs/auto-development").record_items("recovery_probe_result")


def test_supervisor_waits_for_active_experiment_and_rejects_duplicate(tmp_path):
    config, settings = suite_settings(tmp_path)
    _, root, _, _ = prepare(settings, "attach")
    lock = FileLock(root / "writer.lock", timeout=0)
    lock.acquire()
    waits, calls = [], []

    async def sleep(delay):
        waits.append(delay)
        assert not calls
        assert json.loads((root / "supervisor-state.json").read_text())["phase"] == "attached_to_running_experiment"
        with pytest.raises(Timeout):
            await supervise_expansion(settings, "attach", Path("absent.env"))
        lock.release()

    async def execute(*args, **kwargs):
        calls.append(1)
        return {"execution_finished": True}, 0

    try:
        result, code = asyncio.run(supervise_expansion(settings, "attach", Path("absent.env"),
                                  sleep=sleep, operation=execute))
    finally:
        lock.release()
    assert code == 0 and result["execution_finished"]
    assert waits == [15] and calls == [1]
