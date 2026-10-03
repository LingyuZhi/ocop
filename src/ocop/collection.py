import asyncio
import hashlib
import importlib.metadata
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from ocop.benchmark import BenchmarkConfig, build_manifest, load_source, validate_manifest
from ocop.config import RuntimeConfig, StrictModel, canonical_json
from ocop.diagnostics import provenance
from ocop.executor import execute_repeat, executor_config
from ocop.graph import load_contract, replay, trajectory_schema
from ocop.labels import aggregate_label
from ocop.llm import RequestRunner, load_credentials, recover_inflight_budget
from ocop.scoring import score_answer
from ocop.storage import RunHalted, RunStore
from ocop.usage import summarize_usage


COLLECTION_VERSION = "ocop.collection.v1"
PROPOSAL_VERSION = "ocop.proposal.v1"


class CollectionConfig(StrictModel):
    candidates_per_task: int = Field(gt=0)
    complete_repeats: int = Field(gt=0)
    max_repeats: int = Field(gt=0)
    duplicate_graphs: Literal["execute_independently"]
    experience: list[Any] = Field(max_length=0)
    candidate_concurrency: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def check_budget(self):
        if self.complete_repeats > self.max_repeats:
            raise ValueError("Complete repeats cannot exceed the repeat budget")
        return self


def proposal_template() -> dict:
    return {"version": PROPOSAL_VERSION,
            "system": canonical_json({**load_contract(), "schema": trajectory_schema()}).decode("utf-8"),
            "user_format": "original_question", "experience": []}


def proposal_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": proposal_template()["system"]}, {"role": "user", "content": question}]


def graph_fingerprint(graph) -> str:
    payload = {"workers": sorted((worker.worker_id, worker.role) for worker in graph.workers),
               "edges": sorted(graph.edges)}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def write_json(path: Path, payload: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


async def collect_candidate(store: RunStore, runner: RequestRunner, runtime: RuntimeConfig,
                            settings: CollectionConfig, task: dict, task_id: str, candidate_id: str) -> dict:
    existing = store.get_record("candidate_result", candidate_id)
    if existing is not None:
        return existing
    store.ensure_active()
    proposal_id = store.put_record("proposal", candidate_id, {"version": PROPOSAL_VERSION,
        "messages": proposal_messages(task["question"])}, parents={"candidate": candidate_id, "task": task_id})
    outcome = await runner.call(runtime.strong_model, key=f"proposal:{candidate_id}",
                                messages=proposal_messages(task["question"]), owner_id=proposal_id)
    store.put_record("proposal_result", proposal_id, outcome, parents={"proposal": proposal_id})
    content = outcome.get("content")
    parsed = replay(content if isinstance(content, str) else "", reasoning=outcome.get("reasoning"))
    status, error = outcome["status"], outcome.get("error")
    if status == "completed":
        if outcome.get("response_model") != runtime.strong_model.verified_model_id:
            status, error = "fatal", "proposal_model_mismatch"
        elif not isinstance(outcome.get("reasoning"), str) or not outcome["reasoning"].strip():
            status, error = "fatal", "proposal_missing_reasoning"
        elif not parsed.valid:
            status, error = "invalid", parsed.error.code
        else:
            status = "valid"
    if status == "fatal":
        store.halt(error or "fatal_proposal")
    trajectory_id = store.put_record("trajectory", proposal_id, {**asdict(parsed), "valid": parsed.valid,
        "eligible_for_execution": status == "valid", "proposal_status": status}, parents={"proposal": proposal_id})
    if status != "valid":
        result = {"status": f"proposal_{status}", "error": error, "trajectory_id": trajectory_id}
        store.put_record("candidate_result", candidate_id, result, parents={"candidate": candidate_id})
        return result
    fingerprint = graph_fingerprint(parsed.final_graph)
    graph_id = store.put_record("graph", trajectory_id, {"candidate_id": candidate_id,
        "fingerprint": fingerprint, **asdict(parsed.final_graph)}, parents={"trajectory": trajectory_id})
    repeats = []
    for number in range(settings.max_repeats):
        repeat_key = f"{candidate_id}:{number}"
        item = store.get_record("collection_repeat", repeat_key)
        if item is None:
            store.ensure_active()
            execution = await execute_repeat(question=task["question"], graph=parsed,
                parents={"task": task_id, "candidate": candidate_id, "trajectory": trajectory_id, "graph": graph_id},
                repeat_id=str(number), config=runtime, runner=runner)
            score = score_answer(execution["answer"], Decimal(task["normalized_reference"])) if execution["status"] == "completed" else None
            if score is not None:
                store.put_record("score", execution["execution_id"], score,
                                 parents={"execution": execution["execution_id"], "task": task_id})
            item = {"candidate_id": candidate_id, "repeat_id": str(number), "execution_id": execution["execution_id"],
                    "executor_hash": execution["executor_hash"], "status": execution["status"], "score": score}
            store.put_record("collection_repeat", repeat_key, item,
                             parents={"candidate": candidate_id, "execution": execution["execution_id"]})
        repeats.append(item)
        label = aggregate_label(repeats, required=settings.complete_repeats, max_repeats=settings.max_repeats)
        if label["status"] in {"complete", "incomplete"}:
            break
        store.ensure_active()
    if label["status"] == "pending":
        raise ValueError("Candidate ended before its repeat budget was exhausted")
    label_id = store.put_record("label", candidate_id, {**label, "candidate_id": candidate_id,
        "task_id": task["task_id"], "split": task["split"], "graph_fingerprint": fingerprint},
        parents={"candidate": candidate_id, "task": task_id, "trajectory": trajectory_id, "graph": graph_id,
                 **{f"execution_{i}": item["execution_id"] for i, item in enumerate(repeats)}})
    result = {"status": label["status"], "label_id": label_id, "trajectory_id": trajectory_id,
              "graph_id": graph_id, "graph_fingerprint": fingerprint}
    store.put_record("candidate_result", candidate_id, result, parents={"candidate": candidate_id, "label": label_id})
    return result


def collection_report(store: RunStore, phase: str) -> dict:
    manifest = store.get_record("task_manifest", "tasks")
    candidates = store.record_items("candidate")
    results = {row["key"]: row["payload"] for row in store.record_items("candidate_result")}
    labels = {row["key"]: row["payload"] for row in store.record_items("label")}
    graphs = {row["payload"]["candidate_id"]: row["payload"]["fingerprint"] for row in store.record_items("graph")}
    repeats = [row["payload"] for row in store.record_items("collection_repeat")]
    split_reports = {}
    for split in ("train", "eval"):
        selected = [row for row in candidates if row["payload"]["split"] == split]
        ids = {row["id"] for row in selected}
        states = Counter(results.get(row["id"], {}).get("status", "pending") for row in selected)
        task_graphs = defaultdict(list)
        for row in selected:
            if row["id"] in graphs:
                task_graphs[row["payload"]["task_id"]].append(graphs[row["id"]])
        legal = sum(len(values) for values in task_graphs.values())
        duplicates = sum(len(values) - len(set(values)) for values in task_graphs.values())
        selected_repeats = [item for item in repeats if item["candidate_id"] in ids]
        repeat_states = Counter(item["status"] for item in selected_repeats)
        scores = Counter(item["score"]["reason"] for item in selected_repeats if item["score"] is not None)
        distribution = Counter(str(label["mean_outcome"]) for key, label in labels.items()
                               if key in ids and label["status"] == "complete")
        split_reports[split] = {"candidate_count": len(selected), "candidate_states": dict(states),
            "legal_candidates": legal, "legal_rate": legal / len(selected) if selected else None,
            "duplicate_candidates": duplicates, "duplicate_rate_among_legal": duplicates / legal if legal else None,
            "complete_labels": sum(distribution.values()), "label_distribution": dict(sorted(distribution.items())),
            "repeat_count": len(selected_repeats), "repeat_states": dict(repeat_states), "score_counts": dict(scores),
            "incomplete_rate": repeat_states["incomplete"] / len(selected_repeats) if selected_repeats else None,
            "infra_failed_rate": repeat_states["infra_failed"] / len(selected_repeats) if selected_repeats else None}
    groups = {name: [] for name in ("proposal", "worker", "finalizer")}
    for row in store.record_items("proposal"):
        groups["proposal"].extend(store.requests_for_owner(row["id"]))
    for row in store.record_items("execution_node"):
        group = "finalizer" if row["payload"]["node"] == "finalizer" else "worker"
        groups[group].extend(store.requests_for_owner(row["id"]))
    finished = bool(candidates) and len(results) == len(candidates)
    halt = store.rows("run")[0]["halt_reason"]
    return {"version": COLLECTION_VERSION, "run_id": store.run_id, "path": str(store.path),
        "phase": "halted" if halt else "finished" if finished else phase, "halt_reason": halt,
        "finished": finished, "manifest_hash": manifest["hash"],
        "candidate_count": len(candidates), "terminal_candidates": len(results),
        "all_legal_candidates_labeled": all(key in labels and labels[key]["status"] == "complete" for key in graphs) if graphs else None,
        "splits": split_reports, "usage": {name: summarize_usage(store, rows) for name, rows in groups.items()},
        "request_count": sum(len(rows) for rows in groups.values()),
        "candidates": [{**row["payload"], "candidate_id": row["id"], **results.get(row["id"], {"status": "pending"}),
                        "label": labels.get(row["id"])} for row in candidates]}


def save_report(store: RunStore, phase: str, *, export: bool = True) -> dict:
    report = collection_report(store, phase)
    write_json(store.path / "collection-report.json", report)
    if export:
        temporary = store.path / "records.jsonl.tmp"
        store.export_jsonl(temporary)
        os.replace(temporary, store.path / "records.jsonl")
    return report


async def run_collection(runtime: RuntimeConfig, credentials_path: Path, run_id: str | None,
                         *, manifest_only: bool = False, resume_inflight_budget: bool = False,
                         resume_after_topup: bool = False, transport=None, shared_limits=None,
                         report_every: int = 1, on_report=None) -> tuple[dict, int]:
    if report_every < 1:
        raise ValueError("Report interval must be positive")
    if resume_inflight_budget and resume_after_topup:
        raise ValueError("Choose one recovery mode")
    resume = resume_inflight_budget or resume_after_topup
    if resume and (not run_id or manifest_only):
        raise ValueError("Budget recovery requires an existing run ID and collection execution")
    if resume and not (Path(runtime.artifacts_dir) / "runs" / run_id / "records.sqlite3").is_file():
        raise ValueError("Budget recovery requires an existing run database")
    benchmark = BenchmarkConfig.model_validate(runtime.benchmark)
    settings = CollectionConfig.model_validate(runtime.collection)
    if runtime.strong_model.provider != "deepseek" or runtime.strong_model.verified_model_id != runtime.strong_model.model_id:
        raise ValueError("Collection requires the verified DeepSeek proposal model")
    template = proposal_template()
    snapshot = {"purpose": "collection", "version": COLLECTION_VERSION, "runtime": runtime.model_dump(mode="json"),
                "benchmark": benchmark.model_dump(mode="json"), "collection": settings.model_dump(mode="json"),
                "executor": executor_config(runtime), "proposal_template": template}
    origin = provenance()
    origin["dependencies"]["datasets"] = importlib.metadata.version("datasets")
    with RunStore(Path(runtime.artifacts_dir) / "runs", snapshot, run_id=run_id, provenance=origin) as store:
        write_json(store.path / "config.json", runtime.model_dump(mode="json"))
        manifest = store.get_record("task_manifest", "tasks")
        if manifest is None:
            manifest = build_manifest(load_source(benchmark, Path(runtime.artifacts_dir)), benchmark, runtime.seed)
        validate_manifest(manifest, benchmark, runtime.seed)
        manifest_id = store.put_record("task_manifest", "tasks", manifest)
        store.put_record("proposal_template", PROPOSAL_VERSION, {
            **template, "hash": hashlib.sha256(canonical_json(template)).hexdigest()})
        write_json(store.path / "task-manifest.json", manifest)
        jobs = []
        for task in manifest["tasks"]:
            task_id = store.put_record("task", task["task_id"], task, parents={"manifest": manifest_id})
            if task["split"] == "holdout":
                continue
            for slot in range(settings.candidates_per_task):
                candidate = store.put_record("candidate", f"{task['task_id']}:{slot}",
                    {"task_id": task["task_id"], "split": task["split"], "slot": slot}, parents={"task": task_id})
                jobs.append((task, task_id, candidate))
        if manifest_only:
            return save_report(store, "manifest_ready"), 0
        credentials = load_credentials(credentials_path, {runtime.strong_model.api_key_env,
            runtime.worker_model.api_key_env, runtime.finalizer.api_key_env})
        if resume:
            recover_inflight_budget(store, after_topup=resume_after_topup)
        phase = "interrupted"
        try:
            async with RequestRunner(store, runtime.requests, credentials, transport=transport, shared_limits=shared_limits) as runner:
                queue = asyncio.Queue()
                for job in jobs:
                    if store.get_record("candidate_result", job[2]) is None:
                        queue.put_nowait(job)
                save_report(store, "running")
                print(f"collection {store.run_id}: starting {queue.qsize()} pending candidates, concurrency={settings.candidate_concurrency}", flush=True)

                completed_since_report = 0

                async def worker():
                    nonlocal completed_since_report
                    while not queue.empty():
                        task, task_id, candidate = queue.get_nowait()
                        try:
                            await collect_candidate(store, runner, runtime, settings, task, task_id, candidate)
                        except RunHalted:
                            return
                        finally:
                            queue.task_done()
                            completed_since_report += 1
                            if completed_since_report % report_every == 0:
                                report = save_report(store, "running", export=report_every == 1)
                                if on_report:
                                    on_report(report)
                                print(f"collection {store.run_id}: {report['terminal_candidates']}/{len(jobs)} candidates terminal", flush=True)

                workers = [asyncio.create_task(worker()) for _ in range(min(settings.candidate_concurrency, len(jobs)))]
                try:
                    await asyncio.gather(*workers)
                finally:
                    for worker_task in workers:
                        worker_task.cancel()
                    await asyncio.gather(*workers, return_exceptions=True)
            phase = "finished"
        finally:
            report = save_report(store, phase)
            if on_report:
                on_report(report)
        return report, 0 if report["finished"] and not report["halt_reason"] else 1
