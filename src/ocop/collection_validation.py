import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ocop.benchmark import BenchmarkConfig, validate_manifest
from ocop.collection import CollectionConfig, graph_fingerprint
from ocop.config import RuntimeConfig, canonical_json
from ocop.executor import executor_hash
from ocop.graph import replay
from ocop.labels import aggregate_label
from ocop.scoring import normalize_reference, score_answer
from ocop.storage import read_database


def verify(path: Path, *, allow_pending: bool = False) -> dict:
    with read_database(path) as database:
        run = dict(database.execute("SELECT * FROM run").fetchone())
        rows = [dict(row) for row in database.execute("SELECT * FROM records")]
        links = defaultdict(dict)
        for row in database.execute("SELECT * FROM links"):
            links[row["child_id"]][row["relation"]] = row["parent_id"]
        requests = [dict(row) for row in database.execute("SELECT * FROM requests")]
        attempts = [dict(row) for row in database.execute("SELECT * FROM attempts")]

    def blob(digest):
        data = (path / "blobs" / digest).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Blob checksum mismatch")
        return json.loads(data)

    archive = blob(run["manifest_blob"])
    stored_config = {key: value for key, value in archive.items() if key != "provenance"}
    config = RuntimeConfig.model_validate(archive["config"]["runtime"])
    benchmark = BenchmarkConfig.model_validate(config.benchmark)
    settings = CollectionConfig.model_validate(config.collection)
    by_kind = defaultdict(dict)
    by_id = {}
    for row in rows:
        payload = blob(row["payload_blob"])
        by_kind[row["kind"]][row["logical_key"]] = {"id": row["id"], "payload": payload}
        by_id[row["id"]] = payload
    manifest = by_kind["task_manifest"]["tasks"]["payload"]
    validate_manifest(manifest, benchmark, config.seed)
    checks = {"config_hash": hashlib.sha256(canonical_json(stored_config)).hexdigest() == run["config_hash"],
              "manifest_valid": True, "not_halted": run["halt_reason"] is None}
    expected = {(task["task_id"], slot) for task in manifest["tasks"] if task["split"] != "holdout"
                for slot in range(settings.candidates_per_task)}
    candidates = {row["id"]: row["payload"] for row in by_kind["candidate"].values()}
    checks["candidate_slots"] = {(row["task_id"], row["slot"]) for row in candidates.values()} == expected and len(candidates) == len(expected)
    manifest_tasks = {task["task_id"]: task for task in manifest["tasks"]}
    checks["split_links"] = all(by_id[links[cid]["task"]] == manifest_tasks[item["task_id"]]
                                and item["split"] == manifest_tasks[item["task_id"]]["split"] for cid, item in candidates.items())
    graph_ok = True
    for row in by_kind["graph"].values():
        trajectory = by_id[links[row["id"]]["trajectory"]]
        graph = replay(trajectory["raw_content"], reasoning=trajectory["raw_reasoning"])
        graph_ok &= graph.valid and trajectory["eligible_for_execution"]
        graph_ok &= graph_fingerprint(graph.final_graph) == row["payload"]["fingerprint"] if graph.valid else False
    checks["legal_graphs"] = bool(graph_ok)
    repeat_groups = defaultdict(list)
    scores_ok = associations_ok = True
    for row in by_kind["collection_repeat"].values():
        item = row["payload"]
        repeat_groups[item["candidate_id"]].append(item)
        execution = by_kind["execution_result"][item["execution_id"]]["payload"]
        task = manifest_tasks[candidates[item["candidate_id"]]["task_id"]]
        expected_score = score_answer(execution["answer"], normalize_reference(task["reference_answer"])) if execution["status"] == "completed" else None
        scores_ok &= item["score"] == expected_score and item["status"] == execution["status"]
        if expected_score is not None:
            scores_ok &= by_kind["score"][item["execution_id"]]["payload"] == expected_score
        associations_ok &= links[item["execution_id"]]["candidate"] == item["candidate_id"]
        associations_ok &= item["repeat_id"] == execution["repeat_id"] and item["executor_hash"] == executor_hash(config)
    checks["scores_recomputed"] = bool(scores_ok)
    checks["repeat_associations"] = bool(associations_ok)
    labels_ok = True
    for cid, row in by_kind["label"].items():
        label = row["payload"]
        repeats = sorted(repeat_groups[cid], key=lambda item: int(item["repeat_id"]))
        expected_label = aggregate_label(repeats, required=settings.complete_repeats, max_repeats=settings.max_repeats)
        labels_ok &= all(label[key] == value for key, value in expected_label.items())
        labels_ok &= expected_label["status"] in {"complete", "incomplete"}
        labels_ok &= label["split"] == candidates[cid]["split"] and label["task_id"] == candidates[cid]["task_id"]
    checks["labels_recomputed"] = bool(labels_ok)
    checks["repeat_budgets"] = all(len(items) <= settings.max_repeats for items in repeat_groups.values())
    attempt_counts = Counter(row["request_id"] for row in attempts)
    checks["attempt_budgets"] = all(attempt_counts[request["id"]] <= config.requests.max_attempts for request in requests)
    results = by_kind["candidate_result"]
    finished = set(results) == set(candidates)
    checks["finished_or_pending_allowed"] = finished or allow_pending
    checks["terminal_labels"] = all(item["payload"]["status"] not in {"complete", "incomplete"}
                                    or (cid in by_kind["label"] and by_kind["label"][cid]["payload"]["status"] == item["payload"]["status"])
                                    for cid, item in results.items())
    return {"run_id": run["id"], "passed": all(checks.values()), "finished": finished,
            "manifest_hash": manifest["hash"], "candidate_count": len(candidates), "terminal_candidates": len(results),
            "label_count": len(by_kind["label"]), "request_count": len(requests), "attempt_count": len(attempts), "checks": checks}
