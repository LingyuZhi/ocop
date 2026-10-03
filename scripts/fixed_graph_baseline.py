import argparse
import asyncio
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from filelock import FileLock
from pydantic import Field, model_validator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from ocop.benchmark import BenchmarkConfig, validate_manifest
from ocop.collection import graph_fingerprint, write_json
from ocop.config import RuntimeConfig, StrictModel, canonical_json
from ocop.diagnostics import provenance
from ocop.evaluation_report import EvaluationView, build_report
from ocop.executor import execute_repeat, executor_hash
from ocop.graph import replay
from ocop.labels import aggregate_label
from ocop.llm import RequestRunner, load_credentials, recover_inflight_budget
from ocop.scoring import normalize_reference, score_answer
from ocop.storage import RunHalted, RunStore, export_run, read_database
from ocop.trajectories import digest, file_hash
from ocop.usage import summarize_usage


class Settings(StrictModel):
    version: Literal["ocop.fixed_graph_baseline.v1"]
    complete_repeats: int = Field(gt=0)
    max_repeats: int = Field(gt=0)
    candidate_concurrency: int = Field(gt=0)
    trajectory: dict

    @model_validator(mode="after")
    def validate_graph(self):
        if self.complete_repeats > self.max_repeats or not replay(canonical_json(self.trajectory).decode()).valid:
            raise ValueError("Invalid fixed graph or repeat budget")
        return self


class ReadStore:
    def __init__(self, path):
        self.path = path
        with read_database(path) as db:
            self.tables = {name: [dict(r) for r in db.execute(f"SELECT * FROM {name} ORDER BY rowid")]
                           for name in ("run", "records", "links", "requests", "attempts")}
        self.archive = json.loads(self.read_blob(self.tables["run"][0]["manifest_blob"]))
        if self.archive["config"]["purpose"] != "fixed_graph_baseline":
            raise ValueError("Expected a fixed graph baseline run")
        self.by_kind = defaultdict(dict)
        self.links = defaultdict(dict)
        for row in self.tables["records"]:
            self.by_kind[row["kind"]][row["logical_key"]] = {"id": row["id"], "key": row["logical_key"],
                "payload": json.loads(self.read_blob(row["payload_blob"]))}
        for row in self.tables["links"]:
            self.links[row["child_id"]][row["relation"]] = row["parent_id"]

    def read_blob(self, name):
        path = self.path / "blobs" / name
        if file_hash(path) != name:
            raise ValueError("Baseline blob checksum mismatch")
        return path.read_bytes()

    def rows(self, name):
        return self.tables[name]

    def record_items(self, kind):
        return list(self.by_kind[kind].values())

    def get_record(self, kind, key):
        row = self.by_kind[kind].get(key)
        return row["payload"] if row else None

    def attempts_for(self, request_id):
        return [r for r in self.tables["attempts"] if r["request_id"] == request_id]


def prepare(source, settings):
    view = EvaluationView(source)
    report = build_report(view, "baseline_source")
    if not report["finished"] or not report["integrity_passed"]:
        raise ValueError("Baseline requires a completed and verified policy evaluation")
    archived = view.archive["config"]
    runtime = RuntimeConfig.model_validate(archived["runtime"])
    if any(getattr(settings, key) != archived["settings"][key] for key in ("complete_repeats", "max_repeats")):
        raise ValueError("Baseline repeat budgets must match the source evaluation")
    origin = provenance()
    if (origin["source_sha256"] != archived["environment"]["source_sha256"] or any(
            version != archived["environment"]["dependencies"].get(name) for name, version in origin["dependencies"].items())):
        raise ValueError("Executor implementation differs from the source evaluation")
    if archived["settings"]["candidates_per_condition"] != 1:
        raise ValueError("Paired baseline comparison requires one policy candidate per condition")
    origin["script_sha256"] = file_hash(Path(__file__))
    graph = replay(canonical_json(settings.trajectory).decode())
    snapshot = {"purpose": "fixed_graph_baseline", "runtime": runtime.model_dump(mode="json"),
        "settings": settings.model_dump(mode="json"), "environment": origin,
        "task_manifest": archived["task_manifest"], "executor_hash": executor_hash(runtime),
        "graph_fingerprint": graph_fingerprint(graph.final_graph),
        "source": {"path": str(source.resolve()), "manifest_blob": view.tables["run"][0]["manifest_blob"],
                   "report_hash": digest(report)}, "policy_report": report}
    if snapshot["executor_hash"] != archived["executor_hash"]:
        raise ValueError("Baseline executor differs from policy evaluation")
    return runtime, snapshot


def install(store, snapshot):
    source = store.put_record("source_evaluation", "source", snapshot["source"])
    manifest = store.put_record("task_manifest", "tasks", snapshot["task_manifest"], parents={"source": source})
    parsed = replay(canonical_json(snapshot["settings"]["trajectory"]).decode())
    trajectory = store.put_record("fixed_trajectory", "fixed", asdict(parsed), parents={"source": source})
    graph = store.put_record("graph", "fixed", {**asdict(parsed.final_graph), "fingerprint": snapshot["graph_fingerprint"]},
                             parents={"trajectory": trajectory})
    jobs = []
    for task in snapshot["task_manifest"]["tasks"]:
        if task["split"] != "eval":
            continue
        tid = store.put_record("task", task["task_id"], task, parents={"manifest": manifest})
        cid = store.put_record("candidate", task["task_id"], {"task_id": task["task_id"], "split": "eval",
            "graph_fingerprint": snapshot["graph_fingerprint"]}, parents={"task": tid, "graph": graph})
        jobs.append((task, {"task": tid, "candidate": cid, "trajectory": trajectory, "graph": graph}))
    return parsed, jobs


async def execute_candidate(store, runner, runtime, settings, graph, job):
    task, parents = job
    cid = parents["candidate"]
    if store.get_record("candidate_result", cid):
        return
    repeats = []
    for number in range(settings.max_repeats):
        key = f"{cid}:{number}"
        item = store.get_record("baseline_repeat", key)
        if item is None:
            store.ensure_active()
            result = await execute_repeat(question=task["question"], graph=graph, parents=parents,
                repeat_id=str(number), config=runtime, runner=runner)
            score = score_answer(result["answer"], normalize_reference(task["reference_answer"])) if result["status"] == "completed" else None
            if score is not None:
                store.put_record("score", result["execution_id"], score, parents={"execution": result["execution_id"]})
            item = {"candidate_id": cid, "execution_id": result["execution_id"], "repeat_id": str(number),
                    "status": result["status"], "score": score, "executor_hash": result["executor_hash"]}
            store.put_record("baseline_repeat", key, item, parents={"candidate": cid, "execution": result["execution_id"]})
        repeats.append(item)
        label = aggregate_label(repeats, required=settings.complete_repeats, max_repeats=settings.max_repeats)
        if label["status"] != "pending":
            break
        store.ensure_active()
    label_id = store.put_record("label", cid, label, parents={"candidate": cid,
        **{f"execution_{i}": r["execution_id"] for i, r in enumerate(repeats)}})
    store.put_record("candidate_result", cid, {"status": label["status"], "label_id": label_id},
                     parents={"candidate": cid, "label": label_id})


def comparisons(details, policy):
    baseline = {r["task_id"]: r for r in details}
    groups = defaultdict(list)
    for row in policy["candidates"]:
        if row["split"] == "eval":
            groups[(row["model"], row["z"])].append(row)
    result = []
    for (model, z), rows in sorted(groups.items()):
        paired = [r for r in rows if r["label"] and r["label"]["status"] == "complete"
                  and baseline[r["task_id"]]["status"] == "complete"]
        denominator = sum(r["label"]["complete_count"] for r in paired)
        a = sum(r["label"]["success_count"] for r in paired)
        b = sum(baseline[r["task_id"]]["label"]["success_count"] for r in paired)
        result.append({"model": model, "z": z, "paired_tasks": sorted({r["task_id"] for r in paired}),
            "paired_candidates": len(paired), "execution_denominator_each": denominator,
            "policy_successes": a, "baseline_successes": b,
            "policy_success_rate": a / denominator if denominator else None,
            "baseline_success_rate": b / denominator if denominator else None,
            "policy_minus_baseline": (a - b) / denominator if denominator else None,
            "same_graph_pairs": sum(r["graph"]["fingerprint"] == baseline[r["task_id"]]["graph_fingerprint"] for r in paired),
            "policy_correct_coverage": sum(r["observed_successes"] > 0 for r in rows),
            "policy_planned_candidates": len(rows)})
    return result


def report(path):
    view = ReadStore(path)
    snapshot = view.archive["config"]
    runtime = RuntimeConfig.model_validate(snapshot["runtime"])
    settings = Settings.model_validate(snapshot["settings"])
    graph = replay(canonical_json(settings.trajectory).decode())
    fingerprint = graph_fingerprint(graph.final_graph)
    run = view.rows("run")[0]
    tasks = [t for t in snapshot["task_manifest"]["tasks"] if t["split"] == "eval"]
    validate_manifest(snapshot["task_manifest"], BenchmarkConfig.model_validate(runtime.benchmark), runtime.seed)
    candidates = view.record_items("candidate")
    by_task = {r["payload"]["task_id"]: r for r in candidates}
    checks = {"config_hash": digest({k: v for k, v in view.archive.items() if k != "provenance"}) == run["config_hash"],
        "candidate_plan": len(candidates) == len(tasks) and set(by_task) == {t["task_id"] for t in tasks},
        "executor": executor_hash(runtime) == snapshot["executor_hash"] == snapshot["policy_report"]["provenance"]["executor_hash"],
        "source_report": digest(snapshot["policy_report"]) == snapshot["source"]["report_hash"],
        "fixed_graph": fingerprint == snapshot["graph_fingerprint"] and digest(view.get_record("graph", "fixed")) == digest(
            {**asdict(graph.final_graph), "fingerprint": fingerprint}),
        "manifest": view.get_record("task_manifest", "tasks") == snapshot["task_manifest"],
        "task_links": True, "repeats": True, "scores": True, "labels": True, "terminal_results": True,
        "request_budgets": all(len(view.attempts_for(r["id"])) <= runtime.requests.max_attempts for r in view.rows("requests"))}
    details = []
    known_ids = {r["id"] for r in candidates}
    if any(r["key"] not in known_ids for k in ("label", "candidate_result") for r in view.record_items(k)):
        raise ValueError("Unknown baseline candidate")
    if any(r["payload"]["candidate_id"] not in known_ids for r in view.record_items("baseline_repeat")):
        raise ValueError("Unknown repeat candidate")
    for task in tasks:
        row = by_task[task["task_id"]]
        cid = row["id"]
        checks["task_links"] &= (view.get_record("task", task["task_id"]) == task
            and view.links[cid]["task"] == view.by_kind["task"][task["task_id"]]["id"]
            and row["payload"] == {"task_id": task["task_id"], "split": "eval", "graph_fingerprint": fingerprint})
        items = sorted([r["payload"] for r in view.record_items("baseline_repeat") if r["payload"]["candidate_id"] == cid],
                       key=lambda r: int(r["repeat_id"]))
        for item in items:
            result = view.get_record("execution_result", item["execution_id"])
            execution_row = next(r for r in view.record_items("execution") if r["id"] == item["execution_id"])
            checks["repeats"] &= (view.links[item["execution_id"]]["candidate"] == cid
                and item["executor_hash"] == result["executor_hash"] == snapshot["executor_hash"]
                and item["repeat_id"] == result["repeat_id"]
                and execution_row["payload"]["question"] == task["question"]
                and digest(execution_row["payload"]["graph"]) == digest(asdict(graph.final_graph)))
            score = score_answer(result["answer"], normalize_reference(task["reference_answer"])) if result["status"] == "completed" else None
            checks["scores"] &= item["score"] == score and item["status"] == result["status"]
            if score is not None:
                checks["scores"] &= view.get_record("score", item["execution_id"]) == score
        label = aggregate_label(items, required=settings.complete_repeats, max_repeats=settings.max_repeats)
        saved = view.get_record("label", cid)
        terminal = view.get_record("candidate_result", cid)
        checks["labels"] &= saved is None or saved == label
        checks["terminal_results"] &= terminal is None or bool(saved and label["status"] in {"complete", "incomplete"}
                                                                 and terminal["status"] == label["status"])
        details.append({**row["payload"], "candidate_id": cid, "status": terminal["status"] if terminal else "pending",
                        "label": label, "repeats": items})
    complete = [r for r in details if r["status"] == "complete"]
    denominator = len(complete) * settings.complete_repeats
    successes = sum(r["label"]["success_count"] for r in complete)
    terminal = sum(r["status"] != "pending" for r in details)
    return {"run_id": run["id"], "planned": len(tasks), "terminal": terminal,
        "finished": terminal == len(tasks) and run["halt_reason"] is None, "halt_reason": run["halt_reason"],
        "integrity_passed": all(checks.values()), "checks": checks, "snapshot_records": len(view.rows("records")),
        "graph_fingerprint": fingerprint, "executor_hash": snapshot["executor_hash"],
        "complete_labels": len(complete), "incomplete_labels": sum(r["status"] == "incomplete" for r in details),
        "successes": successes, "execution_denominator": denominator,
        "success_rate": successes / denominator if denominator else None,
        "correct_coverage_count": sum(r["label"]["success_count"] > 0 for r in details),
        "correct_coverage_denominator": len(tasks), "label_distribution": dict(Counter(str(r["label"]["mean_outcome"]) for r in complete)),
        "candidates": details, "comparisons": comparisons(details, snapshot["policy_report"]),
        "repeat_statuses": dict(Counter(i["status"] for r in details for i in r["repeats"])),
        "request_count": len(view.rows("requests")), "attempt_count": len(view.rows("attempts")),
        "usage": summarize_usage(view, view.rows("requests")), "source": snapshot["source"]}


def save_report(path, writer):
    result = report(path)
    for key in ("planned", "terminal", "complete_labels", "incomplete_labels", "successes", "execution_denominator", "correct_coverage_count"):
        writer.add_scalar(f"baseline/{key}", result[key], result["snapshot_records"])
    writer.flush()
    events = EventAccumulator(str(path / "tensorboard"), size_guidance={"scalars": 0}).Reload()
    result["tensorboard_passed"] = all(events.Scalars(f"baseline/{key}")[-1].value == result[key]
        for key in ("planned", "terminal", "complete_labels", "successes"))
    write_json(path / "baseline-report.json", result)
    lines = ["# 固定图基线", "", f"候选终态：{result['terminal']}/{result['planned']}；完成：{result['finished']}；一致性：{result['integrity_passed']}。",
        f"完整标签：{result['complete_labels']}；incomplete：{result['incomplete_labels']}；完整标签成功执行：{result['successes']}/{result['execution_denominator']}。",
        "", "各条件仅在双方均有完整标签的共同任务上比较；无配对数据时成功率为未测得。", "",
        "| 模型 | z | 配对候选 | 策略成功执行 | 基线成功执行 | 同图对数 |", "|---|---|---:|---|---|---:|"]
    for item in result["comparisons"]:
        n = item["execution_denominator_each"]
        a, b = (f"{item['policy_successes']}/{n}", f"{item['baseline_successes']}/{n}") if n else ("未测得", "未测得")
        lines.append(f"| {item['model']} | {item['z']} | {item['paired_candidates']} | {a} | {b} | {item['same_graph_pairs']} |")
    (path / "baseline-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    export_run(path, path / "records.jsonl.tmp")
    os.replace(path / "records.jsonl.tmp", path / "records.jsonl")
    return result


async def run(runtime, snapshot, run_id, credentials_path, *, prepare_only=False, after_topup=False, transport=None):
    if after_topup and (prepare_only or not (Path(runtime.artifacts_dir) / "runs" / run_id / "records.sqlite3").is_file()):
        raise ValueError("Credit recovery requires an existing execution run")
    settings = Settings.model_validate(snapshot["settings"])
    credentials = {} if prepare_only else load_credentials(credentials_path, {runtime.worker_model.api_key_env, runtime.finalizer.api_key_env})
    with RunStore(Path(runtime.artifacts_dir) / "runs", snapshot, run_id=run_id, provenance=snapshot["environment"]) as store:
        write_json(store.path / "config.json", snapshot)
        graph, jobs = install(store, snapshot)
        if after_topup:
            recover_inflight_budget(store, after_topup=True)
        with SummaryWriter(str(store.path / "tensorboard"), purge_step=0) as writer:
            try:
                if not save_report(store.path, writer)["integrity_passed"]:
                    raise ValueError("Baseline integrity checks failed")
                store.ensure_active()
                if not prepare_only:
                    async with RequestRunner(store, runtime.requests, credentials, transport=transport) as runner:
                        queue = asyncio.Queue()
                        for job in jobs:
                            if store.get_record("candidate_result", job[1]["candidate"]) is None:
                                queue.put_nowait(job)

                        async def worker():
                            while not queue.empty():
                                job = queue.get_nowait()
                                try:
                                    await execute_candidate(store, runner, runtime, settings, graph, job)
                                except RunHalted:
                                    return
                                finally:
                                    queue.task_done()
                                    state = save_report(store.path, writer)
                                    print(json.dumps({"terminal": state["terminal"], "planned": state["planned"]}), flush=True)

                        workers = [asyncio.create_task(worker()) for _ in range(min(settings.candidate_concurrency, queue.qsize()))]
                        try:
                            await asyncio.gather(*workers)
                        finally:
                            for task in workers:
                                task.cancel()
                            await asyncio.gather(*workers, return_exceptions=True)
            finally:
                result = save_report(store.path, writer)
    return result, 0 if (result["finished"] or prepare_only) and result["integrity_passed"] and result["tensorboard_passed"] and not result["halt_reason"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/fixed-graph-baseline.json"))
    parser.add_argument("--run-id")
    parser.add_argument("--credentials", type=Path, default=Path("my_docs/secrets/credentials.env"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-after-topup", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--verify-experiment", type=Path)
    parser.add_argument("--recovery-config", type=Path, default=Path("config/auto-recovery.json"))
    args = parser.parse_args()
    if args.experiment_config or args.verify_experiment:
        from ocop.expansion import ExpansionConfig, supervise_expansion, verify_suite
        from ocop.recovery import AutoRecoveryConfig

        if args.verify_experiment:
            with FileLock(args.verify_experiment / "writer.lock", timeout=0):
                result = verify_suite(args.verify_experiment)
            status = 0 if result["integrity_passed"] else 1
        else:
            if not args.run_id:
                parser.error("--run-id is required")
            settings = ExpansionConfig.model_validate_json(args.experiment_config.read_text())
            recovery = AutoRecoveryConfig.model_validate_json(args.recovery_config.read_text())
            result, status = asyncio.run(supervise_expansion(settings, args.run_id, args.credentials,
                recovery_settings=recovery, prepare_only=args.prepare_only, after_topup=args.resume_after_topup))
        print(json.dumps({k: v for k, v in result.items() if k in {
            "phase", "path", "integrity_passed", "execution_finished", "semantic_review_status"}}, indent=2))
        raise SystemExit(status)
    if args.report:
        with FileLock(args.report / "writer.lock", timeout=0), SummaryWriter(str(args.report / "tensorboard"), purge_step=0) as writer:
            result = save_report(args.report, writer)
        status = 0 if result["finished"] and result["integrity_passed"] and result["tensorboard_passed"] else 1
    else:
        if not args.source or not args.run_id:
            parser.error("--source and --run-id are required")
        settings = Settings.model_validate_json(args.config.read_text())
        runtime, snapshot = prepare(args.source, settings)
        result, status = asyncio.run(run(runtime, snapshot, args.run_id, args.credentials,
            prepare_only=args.prepare_only, after_topup=args.resume_after_topup))
    print(json.dumps({k: result[k] for k in ("run_id", "planned", "terminal", "finished", "integrity_passed", "halt_reason")}, indent=2))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
