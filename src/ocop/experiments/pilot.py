import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from filelock import FileLock

from ocop.collection.benchmark import BenchmarkConfig, validate_manifest
from ocop.runtime.storage import write_json
from ocop.runtime.config import load_config
from ocop.evaluation.pipeline import EvaluationConfig, candidate_plan
from ocop.execution.executor import executor_hash
from ocop.training.engine import audit_inputs, batch_schedule, training_settings
from ocop.runtime.storage import digest


def preflight(args):
    config = load_config(args.config)
    data, samples, _ = audit_inputs(config, args.data)
    manifest = json.loads((args.source / "task-manifest.json").read_text())
    validate_manifest(manifest, BenchmarkConfig.model_validate(config.benchmark), config.seed)
    if (Path(data["source"]["path"]).resolve() != args.source.resolve()
            or data["source"]["verification"]["manifest_hash"] != manifest["hash"]
            or executor_hash(load_config(args.source / "config.json")) != executor_hash(config)):
        raise ValueError("Pilot data, task manifest or executor source mismatch")
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    if any(tasks[sample["task_id"]]["split"] != "train"
           or tasks[sample["task_id"]]["question"] != sample["question"] for sample in samples):
        raise ValueError("Pilot training samples differ from the frozen tasks")
    selected = min(samples, key=lambda sample: sample["candidate_id"])
    smoke = {key: selected[key] for key in ("candidate_id", "task_id", "z")}
    settings = EvaluationConfig.model_validate(config.evaluation)
    candidates = candidate_plan(manifest, smoke, settings, config.seed)
    evaluated = [candidate for candidate in candidates if candidate["split"] != "train_smoke"]
    schedule = batch_schedule(samples, training_settings(config))
    interval = config.training_runtime.checkpoint_every_steps
    checkpoints = sorted(set(range(interval, len(schedule) + 1, interval)) | {len(schedule)})
    return {"config_hash": digest(config.model_dump()), "data_hash": data["hash"],
        "task_manifest_hash": manifest["hash"], "executor_hash": executor_hash(config),
        "training_samples": len(samples), "training_tasks": data["task_count"],
        "mask_audit_samples": len(samples), "lengths": data["lengths"],
        "label_distribution": data["label_distribution"], "training_steps": len(schedule),
        "checkpoint_steps": checkpoints, "evaluation_task_split": settings.task_split,
        "evaluation_tasks": len({candidate["task_id"] for candidate in evaluated}),
        "evaluation_candidates": len(evaluated), "training_smoke_candidates": len(candidates) - len(evaluated),
        "complete_executions_if_all_legal": len(evaluated) * settings.complete_repeats,
        "candidate_counts": dict(Counter(candidate["split"] for candidate in candidates)),
        "candidates": candidates}


def status(args, stage, **details):
    payload = {"stage": stage, "updated": time.time(), **details}
    write_json(args.experiment_dir / "pipeline-status.json", payload)
    print(json.dumps(payload), flush=True)


def wait_for_gpu(args, stage, minimum_gib):
    if not torch.cuda.is_available():
        raise ValueError("Pilot requires CUDA")
    deadline = time.monotonic() + args.gpu_wait_hours * 3600
    limit = load_config(args.config).training["max_idle_gpus"]
    while True:
        devices = [{"device": f"cuda:{index}", "free_gib": torch.cuda.mem_get_info(index)[0] / 1024**3}
                   for index in range(min(torch.cuda.device_count(), limit))]
        available = [device for device in devices if device["free_gib"] >= minimum_gib]
        if available:
            selected = max(available, key=lambda device: device["free_gib"])
            status(args, "gpu_ready", next_stage=stage, **selected)
            return selected["device"]
        status(args, "waiting_for_gpu", next_stage=stage, minimum_free_gib=minimum_gib,
               devices=devices, wait_remaining_seconds=max(0, deadline - time.monotonic()))
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU availability wait budget exhausted")
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def run_stage(args, name, command):
    log = args.experiment_dir / f"{name}.log"
    status(args, name, command=command, log=str(log))
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_pilot(args):
    if not 0 < args.gpu_wait_hours < float('inf'):
        raise ValueError('GPU wait hours must be positive and finite')
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(args.experiment_dir / 'pipeline.lock', timeout=0):
        try:
            checked = preflight(args)
            frozen = args.experiment_dir / 'preflight.json'
            if frozen.exists() and digest(json.loads(frozen.read_text())) != digest(checked):
                raise ValueError('Pilot preflight differs from its frozen inputs')
            write_json(frozen, checked)
            runtime = load_config(args.config).model_dump()
            if digest(runtime) != checked['config_hash']:
                raise ValueError('Pilot configuration changed during preflight')
            args.config = args.experiment_dir / 'config.json'
            if args.config.exists() and digest(json.loads(args.config.read_text())) != checked['config_hash']:
                raise ValueError('Pilot configuration differs from its frozen snapshot')
            write_json(args.config, runtime)
            status(args, 'preflight_passed', training_samples=checked['training_samples'], training_steps=checked['training_steps'], evaluation_candidates=checked['evaluation_candidates'])
            if args.preflight_only:
                return {'path': str(args.experiment_dir), 'stage': 'preflight_passed'}
            verification = args.experiment_dir / 'sft-verification'
            common = [sys.executable, '-m', 'ocop']
            device = wait_for_gpu(args, 'verify_sft', 23)
            run_stage(args, 'verify_sft', common + ['verify-sft', '--config', str(args.config), '--data', str(args.data), '--output', str(verification), '--device', device])
            verified = json.loads((verification / 'verification.json').read_text())
            if not verified['passed'] or verified['data_hash'] != checked['data_hash']:
                raise ValueError('SFT update and reload verification failed')
            device = wait_for_gpu(args, 'train', 23)
            run_stage(args, 'train', common + ['train', '--config', str(args.config), '--data', str(args.data), '--output', str(args.output), '--device', device])
            run_stage(args, 'verify_training', common + ['verify', 'training', str(args.output), '--output', str(args.experiment_dir / 'training-verification.json')])
            device = wait_for_gpu(args, 'evaluate', 12)
            run_stage(args, 'evaluate', common + ['evaluate', '--config', str(args.config), '--source', str(args.source), '--training-run', str(args.output), '--run-id', args.run_id, '--device', device])
            evaluation = Path(load_config(args.config).artifacts_dir) / 'runs' / args.run_id
            run_stage(args, 'verify_evaluation', common + ['verify', 'evaluation', str(evaluation), '--output', str(args.experiment_dir / 'evaluation-verification.json')])
            status(args, 'completed', training_run=str(args.output), evaluation_run=str(evaluation))
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            previous = json.loads((args.experiment_dir / 'pipeline-status.json').read_text()) if (args.experiment_dir / 'pipeline-status.json').exists() else {}
            status(args, 'failed', failed_stage=previous.get('stage'), error_type=type(exc).__name__, error=str(exc))
            raise
    return {'path': str(args.experiment_dir), 'stage': 'completed'}
