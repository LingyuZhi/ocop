import json

from ocop.storage import RunStore


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
