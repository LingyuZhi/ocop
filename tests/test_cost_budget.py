import asyncio
import json

import httpx
import pytest

from ocop.runtime.llm import RequestRunner
from ocop.runtime.storage import RunHalted, RunStore
from ocop.runtime.usage import CostBudget, CostBudgetGuard, cost_budget_summary
from test_llm import LIMITS, MESSAGES, MODEL, response


def budget(maximum=0.01):
    return CostBudget(max_usd=maximum, input_per_million=0.15, output_per_million=0.6)


def call(store, settings, handler, key="request"):
    async def invoke():
        async with RequestRunner(store, LIMITS, {"OPENAI_API_KEY": "test"},
                transport=httpx.MockTransport(handler), cost_budget=CostBudgetGuard(store, settings)) as runner:
            return await runner.call(MODEL, key=key, messages=MESSAGES)
    return asyncio.run(invoke())


def test_insufficient_reservation_prevents_remote_call(tmp_path):
    with RunStore(tmp_path, {}) as store:
        with pytest.raises(RunHalted, match="cost budget"):
            call(store, budget(0.0001), lambda request: pytest.fail("Unexpected paid request"))
        assert not store.rows("attempts")
        assert store.rows("run")[0]["halt_reason"] == "evaluation_cost_budget_exhausted"


def test_provider_cost_is_cumulative_and_saved_success_is_reused(tmp_path):
    calls = []
    settings = budget(0.003)
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response(usage={"cost": 0.0007}))
    with RunStore(tmp_path, {}, run_id="run") as store:
        assert call(store, settings, handler)["status"] == "completed"
    with RunStore(tmp_path, {}, run_id="run") as store:
        call(store, settings, handler)
        with pytest.raises(RunHalted):
            call(store, settings, handler, key="next")
        summary = cost_budget_summary(store, settings)
        assert len(calls) == 1
        assert summary["provider_cost_usd"] == pytest.approx(0.0007)
        assert summary["all_attempts_reserved"] and summary["cap_respected"]


def test_unknown_remote_outcome_keeps_reservation_across_restart(tmp_path):
    settings = budget(0.003)
    def disconnect(request):
        raise asyncio.CancelledError()
    with RunStore(tmp_path, {}, run_id="run") as store:
        with pytest.raises(asyncio.CancelledError):
            call(store, settings, disconnect)
    with RunStore(tmp_path, {}, run_id="run") as store:
        before = cost_budget_summary(store, settings)
        assert before["reserved_unknown_usd"] > 0 and before["unknown_attempts"] == 1
        with pytest.raises(RunHalted):
            call(store, settings, lambda request: pytest.fail("Unknown charge was forgotten"))
        assert cost_budget_summary(store, settings)["accounted_cost_usd"] == before["accounted_cost_usd"]


def test_saved_response_settles_without_new_remote_call(tmp_path):
    settings = budget(0.01)
    with RunStore(tmp_path, {}, run_id="run") as store:
        spec = {"method": "POST", "url": MODEL.base_url + "/chat/completions",
                "body": MODEL.completion_body(MESSAGES), "provider": MODEL.provider, "api_key_env": MODEL.api_key_env}
        request = store.request("request", spec, None)
        asyncio.run(CostBudgetGuard(store, settings).reserve(request, MODEL, MESSAGES))
        attempt = store.start_attempt(request["id"])
        store.save_response(attempt["id"], body=json.dumps(response(usage={"cost": 0.0004})).encode(),
                            status=200, headers={}, elapsed=0.1)
    with RunStore(tmp_path, {}, run_id="run") as store:
        call(store, settings, lambda request: pytest.fail("Unexpected duplicate request"))
        summary = cost_budget_summary(store, settings)
        assert summary["accounted_cost_usd"] == pytest.approx(0.0004)
        assert summary["unknown_attempts"] == 0


def test_concurrent_requests_wait_for_reservation_to_settle(tmp_path):
    settings = budget(0.003)
    calls = []
    async def handler(request):
        calls.append(request)
        await asyncio.sleep(0)
        return httpx.Response(200, json=response(usage={"cost": 0.0001}))
    async def invoke(store):
        async with RequestRunner(store, LIMITS, {"OPENAI_API_KEY": "test"},
                transport=httpx.MockTransport(handler), cost_budget=CostBudgetGuard(store, settings)) as runner:
            return await asyncio.gather(*(runner.call(MODEL, key=key, messages=MESSAGES) for key in ("a", "b")))
    with RunStore(tmp_path, {}) as store:
        outcomes = asyncio.run(invoke(store))
        assert all(item["status"] == "completed" for item in outcomes) and len(calls) == 2
        assert cost_budget_summary(store, settings)["accounted_cost_usd"] == pytest.approx(0.0002)
