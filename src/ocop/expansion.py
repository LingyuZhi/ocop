import asyncio
import copy
import json
from pathlib import Path
from typing import Literal

from filelock import FileLock, Timeout
from pydantic import Field, model_validator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from ocop.benchmark import BenchmarkConfig, validate_manifest
from ocop.collection import collection_report, run_collection, write_json
from ocop.collection_validation import verify as verify_collection
from ocop.config import RuntimeConfig, StrictModel
from ocop.executor import executor_config, executor_hash
from ocop.llm import RequestLimits
from ocop.replication import coverage, prepare_snapshot, replication_report, run_replication, select_pairs
from ocop.recovery import AutoRecoveryConfig, automatic_recovery
from ocop.storage import ReadStore, StoreConflict, read_database
from ocop.trajectories import digest, file_hash


class ExpansionConfig(StrictModel):
    version: Literal["ocop.expansion.v1"]
    source_collection: str
    seed: int
    train_tasks: int = Field(gt=0)
    holdout_tasks: int = Field(gt=0)
    candidates_per_task: int = Field(gt=0)
    complete_repeats: int = Field(gt=0)
    max_repeats: int = Field(gt=0)
    development_task_ids: list[str] = Field(min_length=1)
    development_repeats: int = Field(gt=0)
    development_max_repeats: int = Field(gt=0)
    selection_limit: int = Field(gt=0)
    replication_repeats: int = Field(gt=0)
    replication_max_repeats: int = Field(gt=0)
    report_every: int = Field(default=25, gt=0)

    @model_validator(mode="after")
    def budgets(self):
        if (self.complete_repeats > self.max_repeats or self.development_repeats > self.development_max_repeats
                or self.replication_repeats > self.replication_max_repeats):
            raise ValueError("Required repeats exceed maximum")
        if len(set(self.development_task_ids)) != len(self.development_task_ids):
            raise ValueError("Duplicate development task")
        return self


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise StoreConflict(f"Frozen experiment content changed: {path}")
    else:
        write_json(path, value)
    return value


def development_graphs():
    roles = ["Decomposer", "Solver", "Checker", "Reviser"]
    assignments = [{"explanation": f"Assign {role} to worker {i}.",
        "action": {"type": "ASSIGN_ROLE", "worker_id": f"worker_{i}", "role": role}} for i, role in enumerate(roles)]
    stop = {"explanation": "Complete the organization.", "action": {"type": "STOP"}}
    edges = [{"explanation": "Pass output to the next worker.", "action": {
        "type": "ADD_EDGE", "source": f"worker_{i}", "target": f"worker_{i + 1}"}} for i in range(3)]
    return ({"version": "ocop.graph.v1", "steps": assignments + edges + [stop]},
            {"version": "ocop.graph.v1", "steps": copy.deepcopy(assignments + [stop])})


def prepare(settings, run_id):
    if len(run_id) > 80 or not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("Invalid experiment ID")
    source = ReadStore(Path(settings.source_collection))
    archived = source.archive["config"]
    if archived["purpose"] != "collection":
        raise ValueError("Source must be a collection run")
    old_runtime = RuntimeConfig.model_validate(archived["runtime"])
    old_manifest = source.get_record("task_manifest", "tasks")
    validate_manifest(old_manifest, BenchmarkConfig.model_validate(old_runtime.benchmark), old_runtime.seed)
    if archived["executor"] != executor_config(old_runtime):
        raise ValueError("Executor differs from frozen source")
    old_tasks = {t["task_id"]: t for t in old_manifest["tasks"]}
    if any(t not in old_tasks or old_tasks[t]["split"] != "eval" for t in settings.development_task_ids):
        raise ValueError("Development tasks must come from old evaluation split")
    runtime = old_runtime.model_copy(deep=True)
    runtime.seed = settings.seed
    runtime.benchmark.update(train_tasks=settings.train_tasks, eval_tasks=0, holdout_tasks=settings.holdout_tasks,
                             excluded_rows=sorted(t["source_row"] for t in old_manifest["tasks"]))
    runtime.collection.update(candidates_per_task=settings.candidates_per_task, complete_repeats=settings.complete_repeats,
                              max_repeats=settings.max_repeats, candidate_concurrency=2)
    if executor_hash(runtime) != executor_hash(old_runtime):
        raise ValueError("Expansion changed executor")
    root = Path(runtime.artifacts_dir) / "experiments" / run_id
    root.mkdir(parents=True, exist_ok=True)
    origin = {"path": str(source.path.resolve()), "manifest_hash": old_manifest["hash"],
              "run_manifest_blob": source.rows("run")[0]["manifest_blob"]}
    frozen = {"settings": settings.model_dump(mode="json"), "runtime": runtime.model_dump(mode="json"),
        "source": origin, "executor_hash": executor_hash(runtime),
        "run_ids": {"collection": run_id + "-collection", "development": run_id + "-development", "replication": run_id + "-replication"}}
    freeze(root / "experiment.json", frozen)
    chain, empty = development_graphs()
    pairs = [{"task_id": tid, "left_name": "chain", "right_name": "empty",
              "left": {"trajectory": chain}, "right": {"trajectory": empty}} for tid in settings.development_task_ids]
    development = prepare_snapshot(runtime, old_tasks, pairs, seed=settings.seed,
        required=settings.development_repeats, maximum=settings.development_max_repeats,
        kind="development", source=origin)
    freeze(root / "development.json", development)
    return runtime, root, frozen, development


def selected_snapshot(runtime, settings, source, verification):
    selected = select_pairs(source, settings.seed, settings.selection_limit)
    tasks = {t["task_id"]: t for t in source.get_record("task_manifest", "tasks")["tasks"]}
    trajectories = {r["id"]: r["payload"] for r in source.record_items("trajectory")}
    results = {r["key"]: r["payload"] for r in source.record_items("candidate_result")}
    candidates = {r["id"]: r["payload"] for r in source.record_items("candidate")}
    pairs = []
    for item in selected:
        pair = {"task_id": item["task_id"], "initial_difference": item["gap"],
                "left_name": "initial_high", "right_name": "initial_low"}
        for arm, name in (("left", "high"), ("right", "low")):
            group = item[name]
            cid = min(group["complete_candidate_ids"], key=lambda c: candidates[c]["slot"])
            trajectory_id = results[cid]["trajectory_id"]
            pair[arm] = {"trajectory": json.loads(trajectories[trajectory_id]["raw_content"]),
                "source": {"candidate_id": cid, "trajectory_id": trajectory_id, "group": group}}
        pairs.append(pair)
    snapshot = prepare_snapshot(runtime, tasks, pairs, seed=settings.seed, required=settings.replication_repeats,
        maximum=settings.replication_max_repeats, kind="selected_training",
        source={"path": str(source.path.resolve()), "manifest_hash": source.get_record("task_manifest", "tasks")["hash"],
                "run_manifest_blob": source.rows("run")[0]["manifest_blob"], "verification": verification})
    return snapshot


def audit_bundle(path):
    view = ReadStore(path)
    outputs = {r["key"]: r["payload"] for r in view.record_items("execution_node_result")}
    candidates = {r["id"]: r["payload"] for r in view.record_items("candidate")}
    nodes = defaultdict_nodes(view)
    records = []
    for row in view.record_items("replication_repeat"):
        item = row["payload"]
        candidate = candidates[item["candidate_id"]]
        records.append({"execution_id": item["execution_id"], "task_id": candidate["task"]["task_id"],
            "arm": candidate["arm"], "block": item["block"], "status": item["status"], "score": item["score"],
            "question": candidate["task"]["question"], "reference_answer": candidate["task"]["reference_answer"],
            "nodes": [{"node": n["payload"]["node"], "messages": n["payload"]["messages"],
                       "outcome": outputs.get(n["id"])} for n in nodes.get(item["execution_id"], [])]})
    return {"run_id": view.run_id, "semantic_review_status": "pending_manual_review",
            "review_fields": ["first_error", "correction", "finalizer_choice", "supporting_text"], "executions": records}


def defaultdict_nodes(view):
    nodes = {}
    for row in view.record_items("execution_node"):
        nodes.setdefault(view.links[row["id"]]["execution"], []).append(row)
    for rows in nodes.values():
        rows.sort(key=lambda r: r["payload"]["node"])
    return nodes


def verify_suite(root):
    frozen = json.loads((root / "experiment.json").read_text())
    runtime = RuntimeConfig.model_validate(frozen["runtime"])
    settings = ExpansionConfig.model_validate(frozen["settings"])
    paths = {k: Path(runtime.artifacts_dir) / "runs" / v for k, v in frozen["run_ids"].items()}
    checked = verify_collection(paths["collection"], allow_pending=True)
    view = ReadStore(paths["collection"])
    manifest = view.get_record("task_manifest", "tasks")
    tasks = manifest["tasks"]
    active_ids = {r["payload"]["task_id"] for r in view.record_items("candidate")}
    checks = {"collection": checked["passed"], "task_isolation":
        not {t["source_row"] for t in tasks} & set(runtime.benchmark["excluded_rows"]),
        "holdout_zero_calls": not active_ids & {t["task_id"] for t in tasks if t["split"] == "holdout"},
        "task_counts": sum(t["split"] == "train" for t in tasks) == settings.train_tasks
            and sum(t["split"] == "holdout" for t in tasks) == settings.holdout_tasks,
        "executor": digest(view.archive["config"]["executor"]) == frozen["executor_hash"],
        "replication_source": True, "saved_reports": True, "request_ownership": True, "jsonl": True}
    owners = {r["id"] for r in view.record_items("proposal")} | {r["id"] for r in view.record_items("execution_node")}
    for request in view.rows("requests"):
        checks["request_ownership"] &= request["owner_id"] in owners
    reports = {}
    for name in ("development", "replication"):
        if (paths[name] / "records.sqlite3").exists():
            report = replication_report(paths[name])
            reports[name] = report
            checks[name] = report["integrity_passed"]
            saved = json.loads((paths[name] / "replication-report.json").read_text())
            checks["saved_reports"] &= saved == report
            if name == "replication":
                expected = selected_snapshot(runtime, settings, view, checked)
                checks["replication_source"] &= expected == ReadStore(paths[name]).archive["config"]
    usage = {"collection": collection_report(view, "verified")["usage"],
             **{k: v["usage"] for k, v in reports.items()}}
    attempts = [a for path in paths.values() if (path / "records.sqlite3").exists() for a in ReadStore(path).rows("attempts")]
    events = []
    for a in attempts:
        if a["finished"] is not None:
            events.extend([(a["started"], 1), (a["finished"], -1)])
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    checks["global_request_concurrency"] = peak <= runtime.requests.concurrency
    for name, path in paths.items():
        if (path / "records.sqlite3").exists():
            from ocop.storage import export_run

            exported = path / "verification-export.jsonl"
            export_run(path, exported)
            checks["jsonl"] &= file_hash(exported) == file_hash(path / "records.jsonl")
    summary = {"integrity_passed": all(checks.values()), "checks": checks,
        "execution_finished": checked["finished"] and checked["passed"] and all(
            name in reports and reports[name]["finished"] for name in ("development", "replication")),
        "collection": checked, "replications": {k: {f: v[f] for f in ("planned", "terminal", "finished", "comparisons", "repeat_statuses")}
                                                for k, v in reports.items()},
        "coverage": coverage(view), "peak_observed_request_concurrency": peak, "usage": usage,
        "semantic_review_status": "pending_manual_review"}
    write_json(root / "verification.json", summary)
    return summary


async def run_expansion(settings, run_id, credentials, *, prepare_only=False, after_topup=False, transport=None):
    runtime, root, frozen, development = prepare(settings, run_id)
    with FileLock(root / "writer.lock", timeout=0):
        write_json(root / "status.json", {"phase": "preparing"})
        limits = RequestLimits(runtime.requests.concurrency)
        with SummaryWriter(str(root / "tensorboard")) as writer:
            def progress(name):
                def update(report):
                    step = report["request_count"]
                    writer.add_scalar(f"{name}/requests", step, step)
                    writer.add_scalar(f"{name}/terminal", report.get("terminal", report.get("terminal_candidates", 0)), step)
                    writer.flush()
                return update

            await run_collection(runtime, credentials, frozen["run_ids"]["collection"], manifest_only=True)
            await run_replication(runtime, development, frozen["run_ids"]["development"], credentials, prepare_only=True)
            if prepare_only:
                write_json(root / "status.json", {"phase": "prepared", "frozen": frozen})
                return {"phase": "prepared", "path": str(root)}, 0
            write_json(root / "status.json", {"phase": "collection_and_development"})
            work = [asyncio.create_task(run_collection(runtime, credentials, frozen["run_ids"]["collection"],
                    shared_limits=limits, resume_after_topup=after_topup, report_every=settings.report_every,
                    on_report=progress("collection"), transport=transport)),
                asyncio.create_task(run_replication(runtime, development, frozen["run_ids"]["development"], credentials,
                    shared_limits=limits, after_topup=after_topup, on_report=progress("development"), transport=transport))]
            try:
                results = await asyncio.gather(*work)
            finally:
                for task in work:
                    task.cancel()
                await asyncio.gather(*work, return_exceptions=True)
            dev_path = Path(runtime.artifacts_dir) / "runs" / frozen["run_ids"]["development"]
            write_json(root / "development-audit.json", audit_bundle(dev_path))
            if any(status != 0 for _, status in results):
                write_json(root / "status.json", {"phase": "halted", "results": [r for r, _ in results]})
                return {"phase": "halted", "path": str(root)}, 1
            source_path = Path(runtime.artifacts_dir) / "runs" / frozen["run_ids"]["collection"]
            checked = verify_collection(source_path)
            if not checked["passed"] or not checked["finished"]:
                raise ValueError("Collection verification failed")
            source = ReadStore(source_path)
            snapshot = selected_snapshot(runtime, settings, source, checked)
            freeze(root / "selection.json", snapshot)
            write_json(root / "coverage.json", coverage(source))
            write_json(root / "status.json", {"phase": "selected_replication", "selected_tasks": len(snapshot["pairs"])})
            _, status = await run_replication(runtime, snapshot, frozen["run_ids"]["replication"], credentials,
                shared_limits=limits, after_topup=after_topup, on_report=progress("replication"), transport=transport)
            result = verify_suite(root)
            writer.flush()
            result["tensorboard_readable"] = bool(EventAccumulator(str(root / "tensorboard")).Reload().Tags()["scalars"])
            write_json(root / "verification.json", result)
            write_json(root / "status.json", {"phase": "executions_complete" if result["execution_finished"] else "halted",
                       "semantic_review_status": "pending_manual_review", "integrity_passed": result["integrity_passed"]})
            return result, status if status else 0 if result["integrity_passed"] and result["tensorboard_readable"] else 1


async def supervise_expansion(settings, run_id, credentials, *, recovery_settings=None,
                              prepare_only=False, after_topup=False, transport=None,
                              sleep=asyncio.sleep, probe=None, operation=None):
    if prepare_only:
        return await run_expansion(settings, run_id, credentials, prepare_only=True)
    policy = recovery_settings or AutoRecoveryConfig()
    runtime, root, frozen, _ = prepare(settings, run_id)
    execute = operation or run_expansion
    with FileLock(root / "supervisor.lock", timeout=0):
        freeze(root / "supervision-policy.json", policy.model_dump(mode="json"))

        def state(payload):
            write_json(root / "supervisor-state.json", payload)
            print("supervisor " + json.dumps(payload), flush=True)

        try:
            while True:
                lock = FileLock(root / "writer.lock", timeout=0)
                try:
                    lock.acquire()
                except Timeout:
                    state({"phase": "attached_to_running_experiment"})
                    await sleep(15)
                    continue
                try:
                    for name, rid in frozen["run_ids"].items():
                        path = Path(runtime.artifacts_dir) / "runs" / rid
                        if not (path / "records.sqlite3").exists():
                            continue
                        with read_database(path) as db:
                            halt = db.execute("SELECT halt_reason FROM run").fetchone()[0]
                        if halt == "consecutive_request_failures":
                            snapshot = ReadStore(path).archive["config"]
                            options = {"probe": probe} if probe is not None else {}
                            await automatic_recovery(path, runtime, snapshot, policy, credentials,
                                sleep=sleep, on_state=state, **options)
                        elif halt and not after_topup:
                            state({"phase": "halted", "run_id": rid, "reason": halt})
                            return {"phase": "halted", "path": str(root)}, 1
                finally:
                    lock.release()
                state({"phase": "running"})
                try:
                    result, code = await execute(settings, run_id, credentials, after_topup=after_topup, transport=transport)
                except Timeout:
                    await sleep(15)
                    continue
                after_topup = False
                if code == 0:
                    state({"phase": "executions_complete", "semantic_review_status": "pending_manual_review"})
                    return result, code
                halted = []
                for rid in frozen["run_ids"].values():
                    path = Path(runtime.artifacts_dir) / "runs" / rid
                    if (path / "records.sqlite3").exists():
                        with read_database(path) as db:
                            halted.append(db.execute("SELECT halt_reason FROM run").fetchone()[0])
                if "consecutive_request_failures" not in halted:
                    state({"phase": "halted", "reason": "experiment_did_not_complete"})
                    return result, code
        except Exception as exc:
            state({"phase": "halted", "reason": type(exc).__name__})
            raise
