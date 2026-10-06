import argparse
import json
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from ocop.graph import graph_fingerprint
from ocop.runtime.config import load_config
from ocop.evaluation.pipeline import archived_snapshot_matches, prepare_evaluation
from ocop.evaluation.report import EvaluationView, audit_generation_tokens, audit_metrics, build_report
from ocop.graph import replay
from ocop.runtime.storage import export_run
from ocop.runtime.storage import digest, file_hash
from ocop.training.data import load_prepared


def history_checks(before, after):
    old = {row["candidate_id"]: row for row in before["candidates"]}
    new = {row["candidate_id"]: row for row in after["candidates"]}
    if old.keys() != new.keys() or before["run_id"] != after["run_id"]:
        return {"candidate_history": False}
    return {
        "candidate_history": all(all(row[key] == new[cid][key] for key in
            ("task_id", "split", "model", "z", "seed", "slot")) for cid, row in old.items()),
        "generation_history": all(not row["generated"] or all(row[key] == new[cid][key]
            for key in ("raw_blob", "generation", "graph")) for cid, row in old.items()),
        "repeat_history": all(row["repeats"] == new[cid]["repeats"][:len(row["repeats"])] for cid, row in old.items()),
        "terminal_history": all(row["status"] == "pending" or
            (row["status"] == new[cid]["status"] and row["label"] == new[cid]["label"]) for cid, row in old.items()),
    }


def training_coverage(samples):
    by_task = defaultdict(list)
    for sample in samples:
        graph = replay(sample["raw_content"], reasoning=sample["raw_reasoning"])
        if not graph.valid:
            raise ValueError("Training trajectory is invalid")
        by_task[sample["task_id"]].append((sample["z"], graph_fingerprint(graph.final_graph)))
    task_graph_labels = defaultdict(set)
    for task, rows in by_task.items():
        for z, graph in rows:
            task_graph_labels[(task, graph)].add(z)
    return {"samples": len(samples), "tasks": len(by_task),
        "label_distribution": dict(sorted(Counter(str(s["z"]) for s in samples).items())),
        "distinct_labels_per_task": dict(Counter(len({z for z, _ in rows}) for rows in by_task.values())),
        "distinct_graphs_per_task": dict(Counter(len({g for _, g in rows}) for rows in by_task.values())),
        "task_graph_groups": len(task_graph_labels),
        "task_graph_groups_with_multiple_labels": sum(len(zs) > 1 for zs in task_graph_labels.values())}


def verify(path, baseline=None):
    if json.loads((path / "config.json").read_text()).get("purpose") == "fixed_graph_baseline":
        if baseline is not None:
            raise ValueError("--baseline is only used for policy evaluation recovery history")
        return verify_fixed_graph(path)
    with FileLock(path / "writer.lock", timeout=0):
        files = ["records.sqlite3", "config.json", "evaluation-manifest.json", "evaluation-report.json",
                 "evaluation-report.md", "records.jsonl"]
        hashes = {name: file_hash(path / name) for name in files}
        view = EvaluationView(path)
        saved = json.loads((path / "evaluation-report.json").read_text())
        report = build_report(view, saved["phase"])
        snapshot = view.archive["config"]
        config = load_config(path / "config.json")
        current, _ = prepare_evaluation(config, Path(snapshot["source"]["path"]), Path(snapshot["training"]["path"]))
        checks = {**report["checks"], "finished": report["finished"],
            "policy_executor_evidence": report["policy_executor_evidence"],
            "engineering_passed": report["engineering_passed"],
            "frozen_inputs": archived_snapshot_matches(current, snapshot),
            "archived_manifest": digest(json.loads((path / "evaluation-manifest.json").read_text())) == digest(snapshot),
            "report_recomputed": digest({key: saved[key] for key in report}) == digest(report),
            "generation_tokens": audit_generation_tokens(view)["passed"],
            "tensorboard": audit_metrics(path, report)["passed"],
            "settled_requests": all(r["state"] in {"completed", "incomplete", "infra_failed", "fatal"}
                                    for r in view.tables["requests"])}
        blobs = {row["payload_blob"] for row in view.tables["records"]}
        blobs.add(view.tables["run"][0]["manifest_blob"])
        blobs.update(r["spec_blob"] for r in view.tables["requests"])
        blobs.update(a["body_blob"] for a in view.tables["attempts"] if a["body_blob"])
        checks["all_blob_checksums"] = all(file_hash(path / "blobs" / name) == name for name in blobs)
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=".verify-evaluation-") as temporary:
            output = Path(temporary) / "records.jsonl"
            export_run(path, output)
            checks["jsonl_matches_database"] = file_hash(output) == hashes["records.jsonl"]
        if baseline:
            checks.update(history_checks(json.loads(baseline.read_text()), report))
        recovery = [row["payload"] for row in view.record_items("recovery")]
        attempts = {row["id"]: row for row in view.tables["attempts"]}
        checks["recovery_evidence"] = all(event["config_hash"] == view.tables["run"][0]["config_hash"]
            and event["historical_results"] == "preserved" and event["budgets"] == "cumulative"
            and bool(event["evidence"]) and all(
                attempt_id in attempts and attempts[attempt_id]["request_id"] == evidence["request_id"]
                for evidence in event["evidence"]
                for attempt_id in evidence.get("attempt_ids", [evidence.get("attempt_id")])) for event in recovery)
        training_run = Path(snapshot["training"]["path"])
        training_manifest = json.loads((training_run / "run-manifest.json").read_text())
        _, samples = load_prepared(Path(training_manifest["data"]))
        details = report["candidates"]
        generations = []
        for model in snapshot["settings"]["models"]:
            rows = [c for c in details if c["model"] == model]
            lengths = [c["lengths"]["output_tokens"] for c in rows]
            generations.append({"model": model, "count": len(rows), "length_min": min(lengths),
                "length_median": statistics.median(lengths), "length_max": max(lengths),
                "generation_minutes": sum(c["generation_seconds"] for c in rows) / 60})
        repeat_counts = Counter(r["status"] for c in details for r in c["repeats"])
        score_counts = Counter(r["score"]["reason"] for c in details for r in c["repeats"] if r["score"])
        finished = max(row["created"] for row in view.tables["records"] if row["kind"] == "candidate_result")
        checks["source_files_unchanged"] = hashes == {name: file_hash(path / name) for name in files}
        return {"passed": all(checks.values()), "checks": checks, "run": str(path.resolve()),
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "last_candidate_finished_at": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
            "planned": report["planned"], "terminal": report["terminal"], "generations": generations,
            "groups": report["groups"], "repeat_statuses": dict(repeat_counts), "score_reasons": dict(score_counts),
            "candidate_statuses": dict(Counter(c["status"] for c in details)),
            "request_count": report["request_count"], "attempt_count": report["attempt_count"],
            "usage": {k: v for k, v in report["usage"].items() if k not in {"request_ids", "attempt_ids"}},
            "training_coverage": training_coverage(samples), "recovery_events": recovery,
            "source_file_hashes": hashes, "baseline": str(baseline.resolve()) if baseline else None}


def verify_fixed_graph(path):
    from fixed_graph_baseline import ReadStore, Settings, prepare, report
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    with FileLock(path / "writer.lock", timeout=0):
        files = ["records.sqlite3", "config.json", "baseline-report.json", "baseline-report.md", "records.jsonl"]
        hashes = {name: file_hash(path / name) for name in files}
        view = ReadStore(path)
        snapshot = view.archive["config"]
        saved = json.loads((path / "baseline-report.json").read_text())
        result = report(path)
        _, current = prepare(Path(snapshot["source"]["path"]), Settings.model_validate(snapshot["settings"]))
        checks = {**result["checks"], "finished": result["finished"],
            "frozen_inputs": digest(current) == digest(snapshot),
            "archived_config": digest(json.loads((path / "config.json").read_text())) == digest(snapshot),
            "report_recomputed": digest({key: saved[key] for key in result}) == digest(result),
            "settled_requests": all(r["state"] in {"completed", "incomplete", "infra_failed", "fatal"} for r in view.rows("requests"))}
        blobs = {row["payload_blob"] for row in view.rows("records")}
        blobs.add(view.rows("run")[0]["manifest_blob"])
        blobs.update(row["spec_blob"] for row in view.rows("requests"))
        blobs.update(row["body_blob"] for row in view.rows("attempts") if row["body_blob"])
        checks["all_blob_checksums"] = all(file_hash(path / "blobs" / name) == name for name in blobs)
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=".verify-baseline-") as temporary:
            output = Path(temporary) / "records.jsonl"
            export_run(path, output)
            checks["jsonl_matches_database"] = file_hash(output) == hashes["records.jsonl"]
        events = EventAccumulator(str(path / "tensorboard"), size_guidance={"scalars": 0}).Reload()
        checks["tensorboard"] = all(events.Scalars(f"baseline/{key}")[-1].value == result[key] for key in
            ("planned", "terminal", "complete_labels", "incomplete_labels", "successes", "execution_denominator", "correct_coverage_count"))
        source = EvaluationView(Path(snapshot["source"]["path"]))
        checks["independent_requests"] = not ({r["id"] for r in view.rows("requests")} &
                                                {r["id"] for r in source.tables["requests"]})
        repeats = [i for c in result["candidates"] for i in c["repeats"]]
        checks["five_nodes_per_complete_repeat"] = all(len(view.get_record("execution_result", r["execution_id"])["nodes"]) == 5
                                                       for r in repeats if r["status"] == "completed")
        finished = max(r["created"] for r in view.rows("records") if r["kind"] == "candidate_result")
        checks["source_files_unchanged"] = hashes == {name: file_hash(path / name) for name in files}
        return {"passed": all(checks.values()), "checks": checks, "run": str(path.resolve()),
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "last_candidate_finished_at": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
            "planned": result["planned"], "terminal": result["terminal"],
            "complete_labels": result["complete_labels"], "incomplete_labels": result["incomplete_labels"],
            "successes": result["successes"], "execution_denominator": result["execution_denominator"],
            "success_rate": result["success_rate"], "comparisons": result["comparisons"],
            "repeat_statuses": result["repeat_statuses"],
            "score_reasons": dict(Counter(r["score"]["reason"] for r in repeats if r["score"])),
            "request_count": result["request_count"], "attempt_count": result["attempt_count"],
            "usage": {k: v for k, v in result["usage"].items() if k not in {"request_ids", "attempt_ids"}},
            "source_file_hashes": hashes}


def main():
    parser = argparse.ArgumentParser(description="Read-only final verification of policy evaluation or fixed graph baseline")
    parser.add_argument("path", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.path.resolve()):
        parser.error("Verification output must be outside the evaluation run")
    report = verify(args.path, args.baseline)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "last_candidate_finished_at": report["last_candidate_finished_at"]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
