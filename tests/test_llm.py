import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from ocop.runtime.config import RequestConfig, load_config
from ocop.runtime.llm import RequestRunner, budget_retry_deadline, load_credentials, normalize_response, recover_inflight_budget
from ocop.runtime.storage import RunStore, StoreConflict


CONFIG = Path(__file__).resolve().parents[1] / "config/prototype.json"
MODEL = load_config(CONFIG).worker_model
LIMITS = RequestConfig(concurrency=2, timeout_seconds=2.0, max_attempts=3,
                       consecutive_exhausted_request_limit=2, retry_backoff_seconds=0.0)
MESSAGES = [{"role": "user", "content": "Test input"}]


def response(finish="stop", content="5", **extra):
    return {"id": "provider-response", "model": MODEL.model_id,
            "choices": [{"finish_reason": finish, "message": {"content": content, "reasoning_content": "A thought"}}], **extra}


def run_call(store, handler, key="request"):
    async def call():
        async with RequestRunner(store, LIMITS, {"OPENAI_API_KEY": "dummy-secret"}, transport=httpx.MockTransport(handler)) as runner:
            return await runner.call(MODEL, key=key, messages=MESSAGES)
    return asyncio.run(call())


def test_provider_parameters_are_explicit():
    config = load_config(CONFIG)
    strong = config.strong_model.completion_body(MESSAGES)
    assert "temperature" not in strong
    assert strong["thinking"] == {"type": "enabled"}
    assert strong["reasoning_effort"] == "high"
    assert strong["top_p"] == 1
    assert strong["stream"] is False
    assert "response_format" not in strong
    worker = config.worker_model.completion_body(MESSAGES)
    assert worker["temperature"] == 0.7
    assert worker["provider"] == {"allow_fallbacks": False, "require_parameters": True, "order": ["OpenAI"]}
    values = config.strong_model.model_dump()
    values["temperature"] = 1.0
    with pytest.raises(ValidationError):
        type(config.strong_model).model_validate(values)


def test_credentials_use_dotenv_without_interpolation(tmp_path, monkeypatch):
    path = tmp_path / "credentials.env"
    path.write_text('OPENAI_API_KEY="literal-${UNSET}"\n')
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert load_credentials(path, {"OPENAI_API_KEY"})["OPENAI_API_KEY"] == "literal-${UNSET}"
    monkeypatch.setenv("OPENAI_API_KEY", "environment")
    assert load_credentials(path, {"OPENAI_API_KEY"})["OPENAI_API_KEY"] == "environment"
    with pytest.raises(ValueError, match="MISSING_KEY"):
        load_credentials(path, {"MISSING_KEY"})


def test_retries_are_attempts_and_missing_usage_stays_unknown(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503 if len(calls) < 3 else 200, json=response())
    with RunStore(tmp_path, {}) as store:
        outcome = run_call(store, handler)
        assert outcome["status"] == "completed"
        assert outcome["usage"] is None
        assert len(store.rows("requests")) == 1
        assert [row["state"] for row in store.rows("attempts")] == ["infra_failed", "infra_failed", "completed"]
        assert run_call(store, handler) == outcome
        assert len(calls) == 3
        spec = store.read_blob(store.rows("requests")[0]["spec_blob"])
        assert b"dummy-secret" not in spec


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls", None])
def test_incomplete_generation_is_not_retried(tmp_path, finish):
    with RunStore(tmp_path, {}) as store:
        outcome = run_call(store, lambda request: httpx.Response(200, json=response(finish)))
        assert outcome["status"] == "incomplete"
        assert len(store.rows("attempts")) == 1


def test_wrong_or_unformatted_answer_is_a_completed_request(tmp_path):
    with RunStore(tmp_path, {}) as store:
        outcome = run_call(store, lambda request: httpx.Response(200, json=response(content="incorrect answer")))
        assert outcome["status"] == "completed"
        assert len(store.rows("attempts")) == 1


def test_auth_error_is_fatal_and_halts_new_requests(tmp_path):
    with RunStore(tmp_path, {}) as store:
        outcome = run_call(store, lambda request: httpx.Response(401, json={"error": "bad key"}))
        assert outcome["status"] == "fatal"
        assert len(store.rows("attempts")) == 1
        with pytest.raises(StoreConflict, match="halted"):
            run_call(store, lambda request: pytest.fail("Unexpected request"), key="next")


def test_exhausted_request_circuit_breaker(tmp_path):
    with RunStore(tmp_path, {}) as store:
        for key in ("a", "b"):
            outcome = run_call(store, lambda request: httpx.Response(503), key=key)
            assert outcome["status"] == "infra_failed"
        assert len(store.rows("attempts")) == 6
        with pytest.raises(StoreConflict, match="halted"):
            run_call(store, lambda request: pytest.fail("Unexpected request"), key="c")


def test_recover_durable_response_without_calling_provider(tmp_path):
    spec = {"method": "POST", "url": MODEL.base_url + "/chat/completions", "body": MODEL.completion_body(MESSAGES), "provider": MODEL.provider, "api_key_env": MODEL.api_key_env}
    with RunStore(tmp_path, {}, run_id="run") as store:
        request = store.request("request", spec, None)
        attempt = store.start_attempt(request["id"])
        store.save_response(attempt["id"], body=json.dumps(response()).encode(), status=200, headers={"x-request-id": "r1"}, elapsed=0.5)
    with RunStore(tmp_path, {}, run_id="run") as store:
        outcome = run_call(store, lambda request: pytest.fail("Unexpected remote request"))
        assert outcome["status"] == "completed"
        assert outcome["response_headers"]["x-request-id"] == "r1"
        assert outcome["content"] == "5"
        assert len(store.rows("attempts")) == 1


def test_cancelled_attempt_uses_remaining_budget_after_resume(tmp_path):
    def cancel(request):
        raise asyncio.CancelledError()
    with RunStore(tmp_path, {}, run_id="run") as store:
        with pytest.raises(asyncio.CancelledError):
            run_call(store, cancel)
    with RunStore(tmp_path, {}, run_id="run") as store:
        assert store.rows("attempts")[0]["state"] == "uncertain"
        outcome = run_call(store, lambda request: httpx.Response(200, json=response()))
        assert outcome["status"] == "completed"
        assert len(store.rows("attempts")) == 2


def test_interruption_cannot_reset_attempt_budget(tmp_path):
    def cancel(request):
        raise asyncio.CancelledError()
    for _ in range(LIMITS.max_attempts):
        with RunStore(tmp_path, {}, run_id="run") as store:
            with pytest.raises(asyncio.CancelledError):
                run_call(store, cancel)
    with RunStore(tmp_path, {}, run_id="run") as store:
        outcome = run_call(store, lambda request: pytest.fail("Budget must be exhausted"))
        assert outcome["status"] == "infra_failed"
        assert outcome["error"] == "attempt_budget_exhausted_after_interruption"
        assert len(store.rows("attempts")) == LIMITS.max_attempts


def test_transport_timeouts_are_recorded_and_bounded(tmp_path):
    def timeout(request):
        raise httpx.ReadTimeout("Timeout", request=request)
    with RunStore(tmp_path, {}) as store:
        outcome = run_call(store, timeout)
        assert outcome["status"] == "infra_failed"
        assert outcome["remote_outcome_unknown"] is True
        assert outcome["usage"] is None
        assert len(store.rows("attempts")) == 3


def test_concurrency_bound_and_duplicate_logical_requests(tmp_path):
    active = maximum = calls = 0
    async def handler(request):
        nonlocal active, maximum, calls
        active += 1
        calls += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json=response())
    async def collect(store):
        async with RequestRunner(store, LIMITS, {"OPENAI_API_KEY": "dummy-secret"}, transport=httpx.MockTransport(handler)) as runner:
            return await asyncio.gather(*(runner.call(MODEL, key=str(i % 4), messages=MESSAGES) for i in range(8)))
    with RunStore(tmp_path, {}) as store:
        results = asyncio.run(collect(store))
        assert calls == 4
        assert maximum == 2
        assert all(item["status"] == "completed" for item in results)


def test_response_secrets_are_redacted_before_persistence(tmp_path):
    with RunStore(tmp_path, {}) as store:
        result = run_call(store, lambda request: httpx.Response(200, json=response(content="echo dummy-secret")))
        assert result["content"] == "echo [REDACTED]"
        assert result["body_redacted"] is True
        assert all(b"dummy-secret" not in path.read_bytes() for path in (store.path / "blobs").iterdir())


@pytest.mark.parametrize(("payload", "status"), [(b"bad JSON", "infra_failed"), (b"[]", "infra_failed"), (b'{"error":{"code":402}}', "fatal")])
def test_malformed_and_in_band_errors(payload, status):
    assert normalize_response(payload, 200, catalog=False)["status"] == status


@pytest.mark.parametrize("http_status", [200, 402])
def test_inflight_budget_error_is_retryable(http_status):
    payload = {"error": {"code": 402, "metadata": {
        "reason": "in_flight_budget_exhausted", "limit_source": "openrouter_in_flight_budget"}}}
    outcome = normalize_response(json.dumps(payload).encode(), http_status, catalog=False)
    assert outcome["status"] == "infra_failed"
    assert outcome["error"] == "in_flight_budget_exhausted"


def test_other_payment_errors_remain_fatal():
    payload = {"error": {"code": 402, "message": "Insufficient credits"}}
    assert normalize_response(json.dumps(payload).encode(), 402, catalog=False)["status"] == "fatal"


@pytest.mark.parametrize(("header", "expected"), [("120", 1120), ("bad", 1120), ("999", 1300),
    ("-1", 1000), ("Thu, 01 Jan 1970 00:18:40 GMT", 1120)])
def test_budget_retry_deadline(header, expected):
    assert budget_retry_deadline({"finished": 1000, "headers_json": json.dumps({"retry-after": header})}) == expected


def test_budget_retry_wait_and_attempt_budget_survive_restart(tmp_path, monkeypatch):
    clock = [1000.0]
    waits = []
    monkeypatch.setattr("ocop.runtime.llm.time.time", lambda: clock[0])
    async def sleep(delay):
        waits.append(delay)
        clock[0] += delay
    monkeypatch.setattr("ocop.runtime.llm.asyncio.sleep", sleep)
    payload = {"error": {"code": 402, "metadata": {
        "reason": "in_flight_budget_exhausted", "limit_source": "openrouter_in_flight_budget"}}}
    calls = []
    def handler(request):
        calls.append(clock[0])
        return httpx.Response(402, headers={"Retry-After": "120"}, json=payload)
    with RunStore(tmp_path, {}, run_id="budget") as store:
        request = store.request("request", {"method": "POST", "url": MODEL.base_url + "/chat/completions",
            "body": MODEL.completion_body(MESSAGES), "provider": MODEL.provider, "api_key_env": MODEL.api_key_env}, None)
        attempt = store.start_attempt(request["id"])
        store.save_response(attempt["id"], body=json.dumps(payload).encode(), status=402,
                            headers={"retry-after": "120"}, elapsed=0.1)
    with RunStore(tmp_path, {}, run_id="budget") as store:
        result = run_call(store, handler)
        assert result["status"] == "infra_failed"
        assert calls == [1120, 1240]
        assert len(store.rows("attempts")) == 3
        assert store.rows("run")[0]["halt_reason"] is None
        assert run_call(store, handler) == result
        assert len(calls) == 2
        assert [delay for delay in waits if delay] == [120, 120]


@pytest.mark.parametrize("status", [401, 402])
@pytest.mark.parametrize("after_topup", [False, True])
def test_budget_recovery_refuses_other_fatal_responses(tmp_path, status, after_topup):
    with RunStore(tmp_path, {}) as store:
        run_call(store, lambda request: httpx.Response(status, json={"error": {"code": status}}))
        with pytest.raises(StoreConflict, match="not a verified"):
            recover_inflight_budget(store, after_topup=after_topup)
        assert store.rows("run")[0]["halt_reason"] == "fatal_request_error"
        assert store.record_items("recovery") == []


def test_deepseek_balance_recovery_requires_verified_topup_evidence(tmp_path):
    payload = {"error": {"message": "Insufficient Balance (request_id: 123e4567-e89b-12d3-a456-426614174000)",
        "type": "unknown_error", "code": "invalid_request_error"}}
    with RunStore(tmp_path, {}) as store:
        request = store.request("deepseek-request", {"method": "POST", "url": "https://api.deepseek.com/chat/completions",
            "body": {}, "provider": "deepseek", "api_key_env": "DEEPSEEK_API_KEY"}, None)
        attempt = store.start_attempt(request["id"])
        store.save_response(attempt["id"], body=json.dumps(payload).encode(), status=402,
                            headers={}, elapsed=0.1)
        attempt = store.attempts_for(request["id"])[-1]
        store.finish_attempt(attempt, {"status": "fatal", "error": "http_402", "elapsed_seconds": 0.1},
                             terminal=True, failure_limit=1)
        with pytest.raises(StoreConflict, match="not a verified"):
            recover_inflight_budget(store)
        assert store.rows("run")[0]["halt_reason"] == "fatal_request_error"
        recover_inflight_budget(store, after_topup=True)
        recovery = store.record_items("recovery")[0]["payload"]
        assert recovery["policy"] == "deepseek_topup.v1"
        assert recovery["historical_results"] == "preserved"
        assert recovery["budgets"] == "cumulative"
        assert store.rows("run")[0]["halt_reason"] is None
        assert store.rows("requests")[0]["state"] == "fatal"
        assert len(store.rows("attempts")) == 1


@pytest.mark.parametrize("error", [
    {"message": "Insufficient Balance", "type": "unknown_error", "code": "invalid_request_error"},
    {"message": "Insufficient Balance (request_id: test)", "type": "unknown_error", "code": "402"},
    {"message": "Account unavailable", "type": "unknown_error", "code": "invalid_request_error"},
])
def test_deepseek_topup_recovery_refuses_unverified_402(tmp_path, error):
    with RunStore(tmp_path, {}) as store:
        request = store.request("deepseek-request", {"method": "POST", "url": "https://api.deepseek.com/chat/completions",
            "body": {}, "provider": "deepseek", "api_key_env": "DEEPSEEK_API_KEY"}, None)
        attempt = store.start_attempt(request["id"])
        store.save_response(attempt["id"], body=json.dumps({"error": error}).encode(), status=402,
                            headers={}, elapsed=0.1)
        attempt = store.attempts_for(request["id"])[-1]
        store.finish_attempt(attempt, {"status": "fatal", "error": "http_402", "elapsed_seconds": 0.1},
                             terminal=True, failure_limit=1)
        with pytest.raises(StoreConflict, match="not a verified"):
            recover_inflight_budget(store, after_topup=True)
        assert store.rows("run")[0]["halt_reason"] == "fatal_request_error"
        assert store.record_items("recovery") == []


def test_budget_cooldown_applies_to_other_requests(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("ocop.runtime.llm.time.time", lambda: clock[0])
    async def sleep(delay):
        clock[0] += delay
    monkeypatch.setattr("ocop.runtime.llm.asyncio.sleep", sleep)
    calls = []
    def handler(request):
        calls.append(clock[0])
        if len(calls) == 1:
            return httpx.Response(402, headers={"Retry-After": "120"}, json={"error": {"code": 402,
                "metadata": {"reason": "in_flight_budget_exhausted", "limit_source": "openrouter_in_flight_budget"}}})
        return httpx.Response(200, json=response())
    async def run(store):
        limits = LIMITS.model_copy(update={"max_attempts": 1})
        async with RequestRunner(store, limits, {"OPENAI_API_KEY": "secret"}, transport=httpx.MockTransport(handler)) as runner:
            first = await runner.call(MODEL, key="first", messages=MESSAGES)
            second = await runner.call(MODEL, key="second", messages=MESSAGES)
            assert first["status"] == "infra_failed" and second["status"] == "completed"
    with RunStore(tmp_path, {}) as store:
        asyncio.run(run(store))
    assert calls == [1000, 1120]
