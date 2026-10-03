def aggregate_label(repeats: list[dict], *, required: int = 5, max_repeats: int = 8) -> dict:
    if not 0 < required <= max_repeats or len(repeats) > max_repeats:
        raise ValueError("Invalid repeat budget")
    identities = [(item["repeat_id"], item["execution_id"]) for item in repeats]
    if len({item[0] for item in identities}) != len(identities) or len({item[1] for item in identities}) != len(identities):
        raise ValueError("Duplicate repeat or execution")
    if len({item["executor_hash"] for item in repeats}) > 1:
        raise ValueError("Cannot mix executor configurations")
    if len({item.get("candidate_id") for item in repeats}) > 1:
        raise ValueError("Cannot mix candidates")
    complete = []
    for item in repeats:
        if item["status"] not in {"completed", "incomplete", "infra_failed", "fatal"}:
            raise ValueError("Unknown execution status")
        if item["status"] == "completed":
            score = item["score"]
            if score is None or score["reason"] not in {"success", "wrong_answer", "format_error"}:
                raise ValueError("Completed execution requires a score")
            if score["success"] is not (score["reason"] == "success"):
                raise ValueError("Inconsistent score")
            complete.append(item)
        elif item.get("score") is not None:
            raise ValueError("Incomplete execution must not have a score")
    if len(complete) > required:
        raise ValueError("Too many complete repeats")
    count = sum(item["score"]["success"] for item in complete)
    status = "complete" if len(complete) == required else "incomplete" if len(repeats) == max_repeats else "pending"
    return {"version": "ocop.label.v1", "status": status, "required_repeats": required,
            "repeat_count": len(repeats), "complete_count": len(complete), "success_count": count,
            "complete_repeat_ids": [item["repeat_id"] for item in complete],
            "complete_execution_ids": [item["execution_id"] for item in complete],
            "execution_ids": [item["execution_id"] for item in repeats],
            "executor_hash": repeats[0]["executor_hash"] if repeats else None,
            "mean_outcome": count / required if status == "complete" else None}
