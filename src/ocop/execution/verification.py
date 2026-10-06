import asyncio
import json
from pathlib import Path

from ocop.runtime.config import load_config
from ocop.execution.single import run_execution
from ocop.runtime.storage import export_run, inspect_run


async def verify(args) -> tuple[dict, int]:
    config = load_config(args.config)
    path = Path(config.artifacts_dir) / "runs" / args.run_id
    summary = {"run_id": args.run_id, "passed": False, "phases": [], "checks": {}}
    reports = []
    try:
        for phase, repeat_id in (("first", "0"), ("resume", "0"), ("independent", "1")):
            print(f"Starting {phase}, repeat={repeat_id}", flush=True)
            report, status = await run_execution(config, args.task, args.trajectory, args.credentials, args.run_id, repeat_id)
            reports.append(report)
            summary["phases"].append({"phase": phase, **{key: report.get(key) for key in (
                "execution_id", "repeat_id", "executor_hash", "status", "score", "request_count", "attempt_count", "usage")}})
            print(f"Finished {phase}: status={report['status']}, requests={report.get('request_count')}, attempts={report.get('attempt_count')}", flush=True)
            if status != 0:
                return summary, 1
        first, resumed, independent = reports
        first_requests = {node["request_id"] for node in first["nodes"].values()}
        next_requests = {node["request_id"] for node in independent["nodes"].values()}
        export_run(path, path / "verified-export.jsonl")
        summary["checks"] = {
            "five_requests_per_repeat": len(first_requests) == len(next_requests) == 5,
            "resume_same_execution": first["execution_id"] == resumed["execution_id"],
            "resume_same_outputs": first["nodes"] == resumed["nodes"],
            "resume_no_new_requests": first["request_count"] == resumed["request_count"],
            "resume_no_new_attempts": first["attempt_count"] == resumed["attempt_count"],
            "independent_execution": first["execution_id"] != independent["execution_id"],
            "independent_requests": len(first_requests) == len(next_requests) == 5 and first_requests.isdisjoint(next_requests),
            "ten_total_requests": independent["request_count"] == 10,
            "fixed_executor": first["executor_hash"] == independent["executor_hash"],
            "all_scored": all(report["score"] is not None for report in reports),
            "export_matches": (path / "verified-export.jsonl").read_bytes() == (path / "records.jsonl").read_bytes(),
        }
        summary["store"] = inspect_run(path)
        summary["passed"] = all(summary["checks"].values())
        return summary, 0 if summary["passed"] else 1
    finally:
        path.mkdir(parents=True, exist_ok=True)
        (path / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Verification passed={summary['passed']}; report={path / 'verification.json'}", flush=True)
