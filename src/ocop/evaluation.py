import asyncio
import importlib.metadata
import json
import time
import uuid
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, model_serializer, model_validator
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from ocop.benchmark import BenchmarkConfig, validate_manifest
from ocop.collection import graph_fingerprint, write_json
from ocop.collection_validation import verify as verify_collection
from ocop.config import StrictModel, load_config
from ocop.diagnostics import provenance
from ocop.evaluation_report import save_report
from ocop.executor import execute_repeat, executor_config, executor_hash
from ocop.full_training import audit_inputs, batch_schedule, run_identity, training_settings, validate_saved_checkpoint
from ocop.graph import replay
from ocop.inference.vllm_pool import VllmConfig, VllmPool, inference_environment
from ocop.labels import aggregate_label
from ocop.llm import RequestRunner, load_credentials, recover_inflight_budget
from ocop.policy_generation import TransformersGenerator, candidate_seed, generation_config, parse_generation
from ocop.scoring import score_answer
from ocop.storage import RunHalted, RunStore
from ocop.trajectories import digest, file_hash, model_identity, policy_messages
from ocop.usage import CostBudget, CostBudgetGuard


class EvaluationConfig(StrictModel):
    target_z: list[float] = Field(min_length=1)
    candidates_per_condition: int = Field(gt=0)
    complete_repeats: int = Field(gt=0)
    models: list[Literal["base", "last_checkpoint"]]
    backend: Literal["transformers", "vllm"]
    max_repeats: int = Field(gt=0)
    candidate_concurrency: int = Field(gt=0)
    training_smoke: Literal["first_training_candidate"]
    task_split: Literal["eval", "holdout"] = "eval"
    execution_task_limit: int | None = Field(default=None, gt=0)
    cost_budget: CostBudget | None = None
    vllm: VllmConfig | None = None

    @model_serializer(mode="wrap")
    def serialize_task_split(self, handler):
        result = handler(self)
        if self.task_split == "eval":
            result.pop("task_split", None)
        for name in ("execution_task_limit", "cost_budget", "vllm"):
            if getattr(self, name) is None:
                result.pop(name, None)
        return result

    @model_validator(mode="after")
    def check_conditions(self):
        if (not all(0 <= z <= 1 for z in self.target_z) or len(set(self.target_z)) != len(self.target_z)
                or self.models != ["base", "last_checkpoint"] or self.max_repeats < self.complete_repeats):
            raise ValueError("Invalid evaluation conditions or repeat budget")
        if (self.backend == "vllm") != (self.vllm is not None):
            raise ValueError("vLLM backend requires explicit inference runtime settings")
        return self


def candidate_plan(manifest, smoke, settings, seed):
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    if smoke["task_id"] not in tasks or tasks[smoke["task_id"]]["split"] != "train":
        raise ValueError("Training smoke must reference a training task")
    selected = [task for task in manifest["tasks"] if task["split"] == settings.task_split]
    if not selected:
        raise ValueError("Evaluation task split is empty")
    if settings.execution_task_limit is not None and settings.execution_task_limit > len(selected):
        raise ValueError("Execution task limit exceeds the evaluation split")
    execution_tasks = {task["task_id"] for task in sorted(selected,
        key=lambda task: digest({"seed": seed, "execution_task": task["task_id"]}))[:settings.execution_task_limit]}
    result = []
    for model in settings.models:
        selections = [(tasks[smoke["task_id"]], "train_smoke", float(smoke["z"]), 0)]
        selections += [(task, settings.task_split, z, slot) for task in selected
                       for z in settings.target_z for slot in range(settings.candidates_per_condition)]
        for task, split, z, slot in selections:
            item = {"task_id": task["task_id"], "split": split, "model": model, "z": z, "slot": slot,
                    "seed": candidate_seed(seed, task["task_id"], slot)}
            if settings.execution_task_limit is not None:
                item["execution_selected"] = split == "train_smoke" or task["task_id"] in execution_tasks
            result.append({**item, "key": digest(item), "ordinal": len(result) + 1})
    return result


def prepare_evaluation(runtime, source, training_run):
    settings = EvaluationConfig.model_validate(runtime.evaluation)
    source_report = verify_collection(source)
    if not source_report["passed"] or not source_report["finished"]:
        raise ValueError("Evaluation requires a completed, verified collection")
    manifest = json.loads((source / "task-manifest.json").read_text())
    validate_manifest(manifest, BenchmarkConfig.model_validate(runtime.benchmark), runtime.seed)
    if source_report["manifest_hash"] != manifest["hash"]:
        raise ValueError("Collection task manifest differs from its archive")
    source_config = load_config(source / "config.json")
    if executor_hash(source_config) != executor_hash(runtime):
        raise ValueError("Evaluation executor differs from the collection executor")
    training_config = load_config(training_run / "config.json")
    training_manifest = json.loads((training_run / "run-manifest.json").read_text())
    data, samples, tokenizer = audit_inputs(training_config, Path(training_manifest["data"]))
    if (Path(data["source"]["path"]).resolve() != source.resolve() or runtime.policy != training_config.policy
            or data["source"]["verification"]["manifest_hash"] != manifest["hash"]):
        raise ValueError("Evaluation policy or source differs from the training run")
    schedule = batch_schedule(samples, training_settings(training_config))
    identity = run_identity(training_config, data, schedule)
    report = json.loads((training_run / "training-report.json").read_text())
    checkpoint = training_run / f"checkpoint-step-{len(schedule)}"
    metadata = validate_saved_checkpoint(checkpoint, identity, schedule)
    if (not report["passed"] or report["identity"] != identity or training_manifest["identity"] != identity
            or report["last_checkpoint"] != checkpoint.name or metadata["step"] != len(schedule)
            or report["reload"]["checkpoint_hash"] != metadata["hash"] or not report["reload"]["passed"]):
        raise ValueError("Final training checkpoint has not passed verification")
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    if any(s["task_id"] not in tasks or tasks[s["task_id"]]["split"] != "train"
           or tasks[s["task_id"]]["question"] != s["question"] for s in samples):
        raise ValueError("Training samples disagree with the source split or questions")
    selected = min(samples, key=lambda sample: sample["candidate_id"])
    smoke = {key: selected[key] for key in ("candidate_id", "task_id", "z")}
    plan = candidate_plan(manifest, smoke, settings, runtime.seed)
    checkpoint_tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    for candidate in plan:
        messages = policy_messages(tasks[candidate["task_id"]]["question"], candidate["z"])
        options = {"tokenize": True, "return_dict": False, "add_generation_prompt": True, "enable_thinking": True}
        if tokenizer.apply_chat_template(messages, **options) != checkpoint_tokenizer.apply_chat_template(messages, **options):
            raise ValueError("Base and checkpoint prompt serialization differ")
    origin = provenance()
    origin["dependencies"].update({name: importlib.metadata.version(name) for name in
                                   ("torch", "transformers", "tokenizers", "tensorboard", "numpy")})
    snapshot = {"purpose": "evaluation", "version": "ocop.evaluation.v1", "runtime": runtime.model_dump(mode="json"),
        "settings": settings.model_dump(), "task_manifest": manifest, "training_smoke": smoke,
        "candidates": plan, "executor": executor_config(runtime), "executor_hash": executor_hash(runtime),
        "generation_config": generation_config(runtime.policy, tokenizer).to_dict(),
        "models": {"base": data["model"], "last_checkpoint": model_identity(checkpoint)},
        "source": {"path": str(source.resolve()), "manifest_hash": manifest["hash"]},
        "training": {"path": str(training_run.resolve()), "data_hash": data["hash"],
                     "checkpoint_hash": metadata["hash"], "report_hash": file_hash(training_run / "training-report.json")},
        "environment": origin}
    if settings.vllm is not None:
        maximum_prompt = max(len(tokenizer.apply_chat_template(policy_messages(tasks[c["task_id"]]["question"], c["z"]),
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)) for c in plan)
        if settings.vllm.max_model_len < maximum_prompt + runtime.policy["max_new_tokens"]:
            raise ValueError("vLLM context is too short for the frozen prompt and full generation budget")
        snapshot["inference_environment"] = inference_environment(settings.vllm)
    return snapshot, tokenizer


def install_candidates(store, snapshot):
    source_id = store.put_record("evaluation_source", "source", snapshot["source"])
    training_id = store.put_record("training_source", "training", snapshot["training"])
    manifest_id = store.put_record("task_manifest", "tasks", snapshot["task_manifest"], parents={"source": source_id})
    models = {name: store.put_record("policy_model", name, spec, parents={"training": training_id})
              for name, spec in snapshot["models"].items()}
    tasks = {task["task_id"]: task for task in snapshot["task_manifest"]["tasks"]}
    jobs = []
    for candidate in snapshot["candidates"]:
        task = tasks[candidate["task_id"]]
        tid = store.put_record("task", task["task_id"], task, parents={"manifest": manifest_id})
        cid = store.put_record("candidate", candidate["key"], candidate, parents={"task": tid, "model": models[candidate["model"]]})
        jobs.append((task, tid, cid, candidate))
    return jobs


def finalize_generation(store, cid, raw):
    parsed = parse_generation(raw)
    raw_row = next(row for row in store.record_items("generation") if row["key"] == cid)
    trajectory_id = store.put_record("trajectory", cid, parsed, parents={"candidate": cid, "generation": raw_row["id"]})
    result = {"trajectory_id": trajectory_id, **{key: parsed[key] for key in
        ("json_parsed", "valid_graph", "eligible_for_execution", "error")}}
    if parsed["eligible_for_execution"]:
        graph = replay(parsed["raw_content"], reasoning=parsed["raw_reasoning"])
        fingerprint = graph_fingerprint(graph.final_graph)
        result["graph_id"] = store.put_record("graph", cid,
            {"candidate_id": cid, "fingerprint": fingerprint, **asdict(graph.final_graph)},
            parents={"trajectory": trajectory_id, "candidate": cid})
        result["graph_fingerprint"] = fingerprint
    else:
        store.put_record("candidate_result", cid, {"status": "generation_invalid", "error": parsed["error"]},
                         parents={"candidate": cid, "trajectory": trajectory_id})
    if parsed["eligible_for_execution"]:
        candidate = next(row["payload"] for row in store.record_items("candidate") if row["id"] == cid)
        if not candidate.get("execution_selected", True):
            store.put_record("candidate_result", cid, {"status": "execution_not_selected"},
                             parents={"candidate": cid, "trajectory": trajectory_id})
    store.put_record("generation_result", cid, result, parents={"candidate": cid, "trajectory": trajectory_id})
    return result


def generate_vllm_candidates(store, snapshot, tokenizer, jobs, writer, pool_factory):
    settings = EvaluationConfig.model_validate(snapshot["settings"])
    pending, attempts = [], {}
    vocabulary = set(tokenizer.get_vocab().values())
    candidates = {cid: candidate for _, _, cid, candidate in jobs}
    for task, _, cid, candidate in jobs:
        if store.get_record("generation_result", cid) is not None:
            continue
        raw = store.get_record("generation", cid)
        if raw is not None:
            finalize_generation(store, cid, raw)
            save_report(store, "generating", writer)
            continue
        store.ensure_active()
        attempt = store.put_record("generation_attempt", uuid.uuid4().hex,
            {"candidate_id": cid, "seed": candidate["seed"], "started": time.time()}, parents={"candidate": cid})
        attempts[cid] = attempt
        ids = tokenizer.apply_chat_template(policy_messages(task["question"], candidate["z"]), tokenize=True,
            return_dict=False, add_generation_prompt=True, enable_thinking=True)
        pending.append({"key": cid, "model": candidate["model"], "ordinal": candidate["ordinal"],
                        "seed": candidate["seed"], "input_ids": ids})
    if not pending:
        return
    pool = pool_factory(settings.vllm, snapshot, store.path / "inference")
    try:
        for job, output in pool.generate(pending):
            cid, ids = job["key"], output["token_ids"]
            if (output["input_ids"] != job["input_ids"] or len(ids) > snapshot["generation_config"]["max_new_tokens"]
                    or any(type(token) is not int or token not in vocabulary for token in ids)):
                raise ValueError("vLLM output changed the frozen prompt or token budget")
            eos = bool(ids) and ids[-1] == tokenizer.eos_token_id
            close = tokenizer.convert_tokens_to_ids("</think>")
            boundary = ids.index(close) + 1 if close in ids else len(ids)
            raw = {"candidate_id": cid, "attempt_id": attempts[cid], "input_ids": job["input_ids"],
                "token_ids": ids, "text": tokenizer.decode(ids, skip_special_tokens=False),
                "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id, "reached_eos": eos,
                "finish_reason": "eos" if eos else "length" if output["finish_reason"] == "length" else "other",
                "input_tokens": len(job["input_ids"]), "output_tokens": len(ids),
                "reasoning_tokens": boundary, "content_tokens": len(ids) - boundary,
                "token_count_convention": "reasoning includes closing think; content includes message end",
                "seconds": output["seconds"], "tokens_per_second": len(ids) / output["seconds"] if output["seconds"] else None,
                "peak_allocated_bytes": None, "backend": "vllm",
                "backend_metrics": {key: output.get(key) for key in
                    ("gpu_uuid", "batch_size", "timing_source", "gpu_memory_used_bytes", "finish_reason", "stop_reason")}}
            store.put_record("generation", cid, raw, parents={"candidate": cid, "attempt": attempts[cid]})
            store.put_record("generation_attempt_result", attempts[cid], {"status": "completed"}, parents={"attempt": attempts[cid]})
            result = finalize_generation(store, cid, raw)
            report = save_report(store, "generating", writer)
            print(json.dumps({"generated": report["generated"], "planned": len(jobs), "model": candidates[cid]["model"],
                "backend": "vllm", "gpu_uuid": output["gpu_uuid"], "output_tokens": len(ids),
                "seconds": raw["seconds"], "eligible": result["eligible_for_execution"], "error": result["error"]}), flush=True)
    finally:
        pool.close()


def generate_candidates(store, snapshot, tokenizer, jobs, device, writer, generator_factory, pool_factory=VllmPool):
    for attempt in store.record_items("generation_attempt"):
        if store.get_record("generation_attempt_result", attempt["id"]) is None:
            raw = store.get_record("generation", attempt["payload"]["candidate_id"])
            status = "completed" if raw is not None and raw["attempt_id"] == attempt["id"] else "interrupted"
            store.put_record("generation_attempt_result", attempt["id"], {"status": status}, parents={"attempt": attempt["id"]})
    if snapshot["settings"]["backend"] == "vllm":
        return generate_vllm_candidates(store, snapshot, tokenizer, jobs, writer, pool_factory)
    for model in snapshot["settings"]["models"]:
        generator = None
        try:
            for task, tid, cid, candidate in jobs:
                if candidate["model"] != model or store.get_record("generation_result", cid) is not None:
                    continue
                raw = store.get_record("generation", cid)
                if raw is None:
                    store.ensure_active()
                    if generator is None:
                        generator = generator_factory(snapshot["models"][model]["path"], device,
                                                      snapshot["generation_config"], tokenizer)
                    attempt_id = store.put_record("generation_attempt", uuid.uuid4().hex,
                        {"candidate_id": cid, "seed": candidate["seed"], "started": time.time()}, parents={"candidate": cid})
                    print(json.dumps({"generating": candidate, "attempt_id": attempt_id}), flush=True)
                    try:
                        raw = {**generator.generate(task["question"], candidate["z"], candidate["seed"]),
                               "candidate_id": cid, "attempt_id": attempt_id}
                        store.put_record("generation", cid, raw, parents={"candidate": cid, "attempt": attempt_id})
                    except Exception as exc:
                        store.put_record("generation_attempt_result", attempt_id,
                            {"status": "runtime_error", "error_type": type(exc).__name__, "error": str(exc)},
                            parents={"attempt": attempt_id})
                        raise
                    store.put_record("generation_attempt_result", attempt_id, {"status": "completed"}, parents={"attempt": attempt_id})
                result = finalize_generation(store, cid, raw)
                report = save_report(store, "generating", writer)
                print(json.dumps({"generated": report["generated"], "planned": len(jobs), "model": model,
                    "split": candidate["split"], "output_tokens": raw["output_tokens"], "seconds": raw["seconds"],
                    "tokens_per_second": raw["tokens_per_second"], "eligible": result["eligible_for_execution"],
                    "error": result["error"]}), flush=True)
        finally:
            if generator is not None:
                generator.close()


async def evaluate_candidate(store, runner, runtime, settings, job):
    task, tid, cid, candidate = job
    if store.get_record("candidate_result", cid) is not None:
        return
    generated = store.get_record("generation_result", cid)
    if generated is None or not generated["eligible_for_execution"]:
        raise ValueError("Execution requires a persisted eligible generation")
    if not candidate.get("execution_selected", True):
        raise ValueError("Execution candidate was not selected in the frozen plan")
    trajectory = store.get_record("trajectory", cid)
    graph = replay(trajectory["raw_content"], reasoning=trajectory["raw_reasoning"])
    repeats = []
    for number in range(settings.max_repeats):
        key = f"{cid}:{number}"
        item = store.get_record("evaluation_repeat", key)
        if item is None:
            store.ensure_active()
            execution = await execute_repeat(question=task["question"], graph=graph,
                parents={"task": tid, "candidate": cid, "trajectory": generated["trajectory_id"], "graph": generated["graph_id"]},
                repeat_id=str(number), config=runtime, runner=runner)
            score = score_answer(execution["answer"], Decimal(task["normalized_reference"])) if execution["status"] == "completed" else None
            if score is not None:
                store.put_record("score", execution["execution_id"], score, parents={"execution": execution["execution_id"], "task": tid})
            item = {"candidate_id": cid, "repeat_id": str(number), "execution_id": execution["execution_id"],
                    "executor_hash": execution["executor_hash"], "status": execution["status"], "score": score}
            store.put_record("evaluation_repeat", key, item, parents={"candidate": cid, "execution": execution["execution_id"]})
        repeats.append(item)
        label = aggregate_label(repeats, required=settings.complete_repeats, max_repeats=settings.max_repeats)
        if label["status"] in {"complete", "incomplete"}:
            break
        store.ensure_active()
    label_id = store.put_record("label", cid, {**label, "candidate_id": cid,
        "task_id": task["task_id"], "split": candidate["split"], "graph_fingerprint": generated["graph_fingerprint"]},
        parents={"candidate": cid, "graph": generated["graph_id"], **{f"execution_{i}": r["execution_id"] for i, r in enumerate(repeats)}})
    store.put_record("candidate_result", cid, {"status": label["status"], "label_id": label_id},
                     parents={"candidate": cid, "label": label_id})


async def run_evaluation(runtime, source, training_run, credentials_path, run_id, device, *, phase="all",
                         resume_inflight_budget=False, resume_after_topup=False,
                         generator_factory=TransformersGenerator, transport=None, vllm_pool_factory=VllmPool):
    if resume_inflight_budget and resume_after_topup:
        raise ValueError("Choose one budget recovery mode")
    if phase not in {"all", "generate", "execute"}:
        raise ValueError("Unknown evaluation phase")
    if (resume_inflight_budget or resume_after_topup) and (phase == "generate" or not run_id
            or not (Path(runtime.artifacts_dir) / "runs" / run_id / "records.sqlite3").is_file()):
        raise ValueError("Budget recovery requires an existing execution run")
    snapshot, tokenizer = prepare_evaluation(runtime, source, training_run)
    settings = EvaluationConfig.model_validate(snapshot["settings"])
    credentials = load_credentials(credentials_path, {runtime.worker_model.api_key_env, runtime.finalizer.api_key_env}) if phase != "generate" else {}
    with RunStore(Path(runtime.artifacts_dir) / "runs", snapshot, run_id=run_id, provenance=snapshot["environment"]) as store:
        write_json(store.path / "config.json", runtime.model_dump(mode="json"))
        jobs = install_candidates(store, snapshot)
        write_json(store.path / "evaluation-manifest.json", snapshot)
        for _, _, cid, _ in jobs:
            raw = store.get_record("generation", cid)
            if raw is not None and store.get_record("generation_result", cid) is None:
                finalize_generation(store, cid, raw)
        if resume_inflight_budget or resume_after_topup:
            recover_inflight_budget(store, after_topup=resume_after_topup)
        phase_status = "interrupted"
        with SummaryWriter(str(store.path / "tensorboard"), purge_step=0) as writer:
            try:
                initial = save_report(store, "starting", writer)
                if not initial["integrity_passed"]:
                    raise ValueError("Saved evaluation records failed integrity checks")
                store.ensure_active()
                if phase in {"all", "generate"}:
                    generate_candidates(store, snapshot, tokenizer, jobs, device, writer, generator_factory, vllm_pool_factory)
                if phase in {"all", "execute"}:
                    if any(store.get_record("generation_result", cid) is None for _, _, cid, _ in jobs):
                        raise ValueError("Execution phase requires all planned generations to be saved")
                    guard = CostBudgetGuard(store, settings.cost_budget) if settings.cost_budget is not None else None
                    async with RequestRunner(store, runtime.requests, credentials, transport=transport, cost_budget=guard) as runner:
                        queue = asyncio.Queue()
                        execution_jobs = jobs
                        if settings.execution_task_limit is not None:
                            execution_jobs = sorted(jobs, key=lambda job: (job[3]["split"] != "train_smoke",
                                digest({"seed": runtime.seed, "execution_task": job[3]["task_id"]}),
                                job[3]["z"], job[3]["model"], job[3]["slot"]))
                        for job in execution_jobs:
                            if store.get_record("candidate_result", job[2]) is None:
                                queue.put_nowait(job)

                        async def worker():
                            while not queue.empty():
                                job = queue.get_nowait()
                                try:
                                    await evaluate_candidate(store, runner, runtime, settings, job)
                                except RunHalted:
                                    return
                                finally:
                                    queue.task_done()
                                    report = save_report(store, "executing", writer)
                                    print(json.dumps({"terminal": report["terminal"], "planned": len(jobs)}), flush=True)

                        workers = [asyncio.create_task(worker()) for _ in range(min(settings.candidate_concurrency, queue.qsize()))]
                        try:
                            await asyncio.gather(*workers)
                        finally:
                            for task in workers:
                                task.cancel()
                            await asyncio.gather(*workers, return_exceptions=True)
                phase_status = "generated" if phase == "generate" else "finished"
            except Exception as exc:
                phase_status = "failed"
                write_json(store.path / f"failure-{uuid.uuid4().hex}.json", {"error_type": type(exc).__name__,
                    "error": str(exc), "time": time.time()})
                raise
            finally:
                report = save_report(store, phase_status, writer)
        ok = report["generated"] == len(jobs) if phase == "generate" else report["finished"]
        return report, 0 if ok and report["integrity_passed"] and not report["halt_reason"] else 1
