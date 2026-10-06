import asyncio
import json
import math

from pydantic import Field

from ocop.runtime.config import StrictModel
from ocop.runtime.storage import RunHalted, RunStore


class CostBudget(StrictModel):
    max_usd: float = Field(gt=0, allow_inf_nan=False)
    input_per_million: float = Field(ge=0, allow_inf_nan=False)
    output_per_million: float = Field(ge=0, allow_inf_nan=False)


def outcome_cost(outcome, budget):
    usage = outcome.get("usage")
    if isinstance(usage, dict):
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0:
            return float(cost), "provider"
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if all(isinstance(value, int) and value >= 0 for value in (prompt, completion)):
            return (prompt * budget.input_per_million + completion * budget.output_per_million) / 1e6, "uncached_token_estimate"
    if outcome.get("http_status") in {400, 401, 402, 403, 404, 422, 429}:
        return 0.0, "rejected_request"
    return None, "reserved_unknown"


def cost_budget_summary(store, budget):
    attempts = {(row["request_id"], row["number"]): row for row in store.rows("attempts")}
    known, estimated, unknown = [], [], []
    reservations = store.record_items("cost_reservation")
    reserved_keys = set()
    for row in reservations:
        item = row["payload"]
        key = (item["request_id"], item["attempt_number"])
        reserved_keys.add(key)
        attempt = attempts.get(key)
        outcome = json.loads(attempt["result_json"] or "{}") if attempt else {}
        cost, source = outcome_cost(outcome, budget)
        if cost is None:
            unknown.append(item["max_usd"])
        elif source == "provider":
            known.append(cost)
        else:
            estimated.append(cost)
    accounted = math.fsum(known + estimated + unknown)
    return {"max_usd": budget.max_usd, "provider_cost_usd": math.fsum(known),
            "estimated_cost_usd": math.fsum(estimated), "reserved_unknown_usd": math.fsum(unknown),
            "accounted_cost_usd": accounted, "remaining_usd": max(0.0, budget.max_usd - accounted),
            "reservations": len(reservations), "unknown_attempts": len(unknown),
            "all_attempts_reserved": set(attempts).issubset(reserved_keys),
            "cap_respected": accounted <= budget.max_usd + 1e-12}


class CostBudgetGuard:
    def __init__(self, store, budget):
        self.store, self.budget = store, budget
        summary = cost_budget_summary(store, budget)
        if not summary["all_attempts_reserved"]:
            raise ValueError("Cost budget requires a reservation for every historical attempt")
        self.accounted = summary["accounted_cost_usd"]
        self.active = set()
        self.changed = asyncio.Condition()

    async def reserve(self, request, model, messages):
        if model.provider != "openrouter" or model.model_id != "openai/gpt-4o-mini":
            raise ValueError("This cost budget requires the verified GPT-4o-mini executor")
        # Byte length bounds BPE text tokens; overhead covers chat framing.
        prompt_bound = sum(len(message["content"].encode("utf-8")) for message in messages) + 1024
        maximum = (prompt_bound * self.budget.input_per_million
                   + model.max_output_tokens * self.budget.output_per_million) / 1e6
        number = len(self.store.attempts_for(request["id"])) + 1
        key = f"{request['id']}:{number}"
        async with self.changed:
            self.store.ensure_active()
            if self.store.get_record("cost_reservation", key) is None:
                while self.accounted + maximum > self.budget.max_usd + 1e-12:
                    if not self.active:
                        self.store.halt("evaluation_cost_budget_exhausted")
                        self.changed.notify_all()
                        raise RunHalted("Evaluation cost budget exhausted")
                    await self.changed.wait()
                    self.store.ensure_active()
                self.store.put_record("cost_reservation", key,
                    {"request_id": request["id"], "attempt_number": number, "max_usd": maximum,
                     "prompt_token_bound": prompt_bound, "output_token_bound": model.max_output_tokens})
                self.accounted += maximum
            self.active.add(key)
        return key

    async def settle(self, key, outcome):
        reservation = self.store.get_record("cost_reservation", key)
        cost, _ = outcome_cost(outcome, self.budget)
        async with self.changed:
            if cost is not None:
                self.accounted += cost - reservation["max_usd"]
            self.active.discard(key)
            self.changed.notify_all()


def summarize_usage(store: RunStore, requests: list[dict]) -> dict:
    result = {"request_ids": [row["id"] for row in requests], "attempt_ids": [],
              "unknown_usage_attempts": 0, "elapsed_seconds": 0.0, "unknown_elapsed_attempts": 0}
    totals = {}
    for request in requests:
        for attempt in store.attempts_for(request["id"]):
            result["attempt_ids"].append(attempt["id"])
            if attempt["elapsed"] is None:
                result["unknown_elapsed_attempts"] += 1
            else:
                result["elapsed_seconds"] += attempt["elapsed"]
            usage = json.loads(attempt["result_json"] or "{}").get("usage")
            if not isinstance(usage, dict) or not usage:
                result["unknown_usage_attempts"] += 1
            else:
                for key, value in usage.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        totals[key] = totals.get(key, 0) + value
    result["known_usage_sum"] = totals or None
    return result
