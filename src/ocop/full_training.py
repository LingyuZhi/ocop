import gc
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForImageTextToText, AutoTokenizer, GenerationConfig

from ocop.collection import write_json
from ocop.config import load_config
from ocop.graph import replay
from ocop.training import optimizer_dtypes, parameter_hashes, probe, update
from ocop.trajectories import digest, encode_sample, file_hash, load_prepared, model_identity, policy_messages


def training_settings(config):
    required = {"representation": "raw_traj", "full_parameter": True, "precision": "bf16",
        "optimizer": "adamw", "gradient_checkpointing": True, "checkpoint_selection": "last",
        "overlength_policy": "report_and_wait"}
    if any(config.training.get(k) != v for k, v in required.items()) or not config.policy.get("thinking"):
        raise ValueError("Unsupported full SFT configuration")
    if config.training_runtime is None:
        raise ValueError("Full SFT requires explicit training_runtime settings")
    settings = config.training_runtime.model_dump()
    if settings["gradient_accumulation"] != config.training["effective_batch_size"]:
        raise ValueError("Effective batch size differs from microbatch times accumulation")
    if (type(config.training["epochs"]) is not int or config.training["epochs"] < 1
            or not 0 < config.training["learning_rate"] < float("inf")):
        raise ValueError("Epochs and learning rate must be positive and finite")
    return {**settings, "learning_rate": config.training["learning_rate"],
            "epochs": config.training["epochs"], "seed": config.seed}


def batch_schedule(samples, settings):
    ids = sorted(s["candidate_id"] for s in samples)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Training requires nonempty, unique candidate IDs")
    schedule = []
    size = settings["gradient_accumulation"]
    for epoch in range(settings["epochs"]):
        order = ids.copy()
        random.Random(settings["seed"] + epoch).shuffle(order)
        for offset in range(0, len(order), size):
            schedule.append({"epoch": epoch, "batch": offset // size,
                             "sample_ids": order[offset:offset + size]})
    return schedule


def next_position(schedule, step):
    if not 0 <= step <= len(schedule):
        raise ValueError("Invalid checkpoint step")
    return {"epoch": schedule[step]["epoch"], "batch": schedule[step]["batch"]} if step < len(schedule) else {
        "epoch": schedule[-1]["epoch"] + 1, "batch": 0}


def capture_rng(device):
    return {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state(device) if str(device).startswith("cuda:") else None}


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(output, network, tokenizer, optimizer, scheduler, identity, schedule, updates, sample, device):
    step = len(updates)
    destination = output / f"checkpoint-step-{step}"
    if destination.exists():
        raise ValueError("Checkpoint already exists")
    temporary = output / f".checkpoint-step-{step}-{uuid.uuid4().hex}.pending"
    temporary.mkdir()
    network.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    torch.save(probe(network, sample, device), temporary / "probe.pt")
    state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "rng": capture_rng(device), "step": step, "next_position": next_position(schedule, step),
             "updates": updates}
    torch.save(state, temporary / "training-state.pt")
    metadata = {"version": "ocop.sft_checkpoint.v1", "identity": identity, "step": step,
        "next_position": state["next_position"], "parameter_hashes": parameter_hashes(network),
        "optimizer_state_dtypes": optimizer_dtypes(optimizer),
        "files": {p.name: file_hash(p) for p in sorted(temporary.iterdir()) if p.is_file()}}
    metadata["hash"] = digest(metadata)
    write_json(temporary / "checkpoint.json", metadata)
    validate_saved_checkpoint(temporary, identity, schedule)
    os.rename(temporary, destination)
    write_json(output / "latest-checkpoint.json", {"path": destination.name, "step": step,
                                                 "hash": metadata["hash"]})
    return destination


def validate_saved_checkpoint(path, identity, schedule):
    metadata = json.loads((path / "checkpoint.json").read_text())
    if (metadata.get("version") != "ocop.sft_checkpoint.v1"
            or metadata.get("hash") != digest({k: v for k, v in metadata.items() if k != "hash"})):
        raise ValueError("Checkpoint metadata hash or version mismatch")
    if metadata["identity"] != identity:
        raise ValueError("Checkpoint data, configuration, code or schedule mismatch")
    if metadata["next_position"] != next_position(schedule, metadata["step"]):
        raise ValueError("Checkpoint cursor mismatch")
    files = metadata["files"]
    actual = {p.name for p in path.iterdir() if p.is_file() and p.name != "checkpoint.json"}
    if actual != set(files) or any(file_hash(path / name) != expected for name, expected in files.items()):
        raise ValueError("Checkpoint file checksum mismatch")
    return metadata


def latest_checkpoint(output):
    paths = list(output.glob("checkpoint-step-*"))
    if not paths:
        raise ValueError("No published checkpoint exists")
    checkpoints = []
    for path in paths:
        metadata = json.loads((path / "checkpoint.json").read_text())
        step = metadata["step"]
        if type(step) is not int or step < 1 or path.name != f"checkpoint-step-{step}":
            raise ValueError("Invalid published checkpoint name or step")
        checkpoints.append((step, path))
    return max(checkpoints, key=lambda item: item[0])[1]


def restore_checkpoint(path, network, optimizer, scheduler, metadata, schedule, device):
    if parameter_hashes(network) != metadata["parameter_hashes"]:
        raise ValueError("Reloaded parameters differ from checkpoint")
    state = torch.load(path / "training-state.pt", map_location="cpu", weights_only=False)
    step = metadata["step"]
    if (state["step"] != step or state["next_position"] != metadata["next_position"]
            or len(state["updates"]) != step):
        raise ValueError("Checkpoint training state mismatch")
    for index, metrics in enumerate(state["updates"]):
        if metrics["step"] != index + 1 or metrics["sample_ids"] != schedule[index]["sample_ids"]:
            raise ValueError("Checkpoint update history mismatch")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if (not optimizer.state or any(s["step"].item() != step for s in optimizer.state.values())
            or scheduler.last_epoch != step):
        raise ValueError("Optimizer or scheduler step mismatch")
    restore_rng(state["rng"], device)
    return state["updates"]


def audit_inputs(config, data):
    manifest, samples = load_prepared(data)
    if (manifest["training"] != config.training or manifest["policy"] != config.policy
            or manifest["seed"] != config.seed):
        raise ValueError("Prepared data and runtime configuration mismatch")
    if manifest["model"] != model_identity(Path(config.policy["model_path"])):
        raise ValueError("Base model or tokenizer files changed")
    tokenizer = AutoTokenizer.from_pretrained(config.policy["model_path"], local_files_only=True)
    for sample in samples:
        if sample["split"] != "train" or encode_sample(sample, tokenizer) != sample:
            raise ValueError("Prepared token mask audit failed")
    return manifest, samples, tokenizer


def run_identity(config, manifest, schedule):
    return {"data_hash": manifest["hash"], "config_hash": digest(config.model_dump()),
        "schedule_hash": digest(schedule), "model": manifest["model"],
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "tokenizers", "tensorboard", "numpy")},
        "implementation_hashes": {name: file_hash(Path(__file__).with_name(name)) for name in
                                  ("full_training.py", "training.py", "trajectories.py", "config.py")}}


def setup_device(device, seed, minimum_gib=23):
    if not torch.cuda.is_available() or not device.startswith("cuda:"):
        raise ValueError("Full SFT requires an available CUDA device")
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < minimum_gib * 1024**3:
        raise ValueError(f"Selected GPU requires at least {minimum_gib} GiB free")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)


def load_network(path, device):
    network = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    network.requires_grad_(True)
    network.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return network


def make_optimizer(network, settings):
    optimizer = torch.optim.AdamW(network.parameters(), lr=settings["learning_rate"],
        betas=tuple(settings["betas"]), eps=settings["epsilon"], weight_decay=settings["weight_decay"], foreach=False)
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def epoch_metrics(updates, epoch):
    rows = [row for row in updates if row["epoch"] == epoch]
    result = {}
    for loss, tokens in (("loss", "supervised_tokens"), ("reasoning_loss", "reasoning_tokens"),
                         ("content_loss", "content_tokens")):
        count = sum(row[tokens] for row in rows)
        result[loss] = sum(row[loss] * row[tokens] for row in rows) / count
        result[tokens] = count
    return result


def train_loop(output, network, tokenizer, optimizer, scheduler, settings, identity, schedule,
               samples, updates, device, writer, attempt):
    by_id = {sample["candidate_id"]: sample for sample in samples}
    probe_sample = by_id[sorted(by_id)[0]]
    for index in range(len(updates), len(schedule)):
        entry = schedule[index]
        metrics = update(network, optimizer, scheduler, [by_id[cid] for cid in entry["sample_ids"]],
                         tokenizer, device, settings, writer, index + 1, namespace="train")
        metrics.update(entry)
        updates.append(metrics)
        write_json(attempt / f"step-{index + 1}.json", metrics)
        writer.add_scalar("train/epoch", entry["epoch"] + (entry["batch"] + 1) /
                          math.ceil(len(samples) / settings["gradient_accumulation"]), index + 1)
        end_epoch = index + 1 == len(schedule) or schedule[index + 1]["epoch"] != entry["epoch"]
        if end_epoch:
            for key, value in epoch_metrics(updates, entry["epoch"]).items():
                writer.add_scalar(f"epoch/{key}", value, index + 1)
        writer.flush()
        if (index + 1) % settings["checkpoint_every_steps"] == 0 or index + 1 == len(schedule):
            started = time.monotonic()
            path = save_checkpoint(output, network, tokenizer, optimizer, scheduler, identity,
                                   schedule, updates, probe_sample, device)
            writer.add_scalar("checkpoint/save_seconds", time.monotonic() - started, index + 1)
            writer.flush()
            print(json.dumps({"checkpoint": str(path), "step": index + 1}), flush=True)
        write_json(output / "status.json", {"status": "training", "step": index + 1,
            "total_steps": len(schedule), "attempt": attempt.name, "updated": time.time()})
    return updates


def reload_final(config, data, output, device):
    settings = training_settings(config)
    manifest, samples, _ = audit_inputs(config, data)
    schedule = batch_schedule(samples, settings)
    identity = run_identity(config, manifest, schedule)
    path = latest_checkpoint(output)
    metadata = validate_saved_checkpoint(path, identity, schedule)
    if metadata["step"] != len(schedule):
        raise ValueError("Final checkpoint is not at the final training step")
    setup_device(device, config.seed, minimum_gib=8)
    network = load_network(path, device)
    if parameter_hashes(network) != metadata["parameter_hashes"]:
        raise ValueError("Final reload parameter hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    sample = sorted(samples, key=lambda item: item["candidate_id"])[0]
    if any(encode_sample(item, tokenizer) != item for item in samples):
        raise ValueError("Checkpoint tokenizer mask audit failed")
    saved = torch.load(path / "probe.pt", map_location="cpu", weights_only=True)
    actual = probe(network, sample, device)
    torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    network.eval()
    inputs = tokenizer.apply_chat_template(policy_messages(sample["question"], sample["z"]),
        tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=True).to(device)
    generation = GenerationConfig(do_sample=False, max_new_tokens=settings["generation_max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    with torch.inference_mode():
        generated = network.generate(**inputs, generation_config=generation)
    ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    reasoning, separator, content = raw.partition("</think>")
    content = content.removesuffix(tokenizer.eos_token).strip() if separator else ""
    parsed = replay(content, reasoning=reasoning)
    report = {"passed": True, "identity": identity, "checkpoint": path.name,
        "checkpoint_hash": metadata["hash"], "step": len(schedule), "parameter_hashes_match": True,
        "reload_logits_max_abs_diff": (actual - saved).abs().max().item(),
        "generation": {"purpose": "training_task_smoke", "candidate_id": sample["candidate_id"],
            "config": generation.to_dict(), "token_ids": ids, "raw": raw, "valid_graph": parsed.valid,
            "parse_error": None if parsed.valid else parsed.error.code,
            "reached_eos": bool(ids) and ids[-1] == tokenizer.eos_token_id}, "finished": time.time()}
    with SummaryWriter(str(output / "tensorboard")) as writer:
        writer.add_scalar("generation/legal", int(parsed.valid), len(schedule))
        writer.add_scalar("generation/tokens", len(ids), len(schedule))
        writer.add_scalar("generation/reached_eos", int(report["generation"]["reached_eos"]), len(schedule))
    write_json(output / "reload-report.json", report)
    return report


def audit_events(output, total_steps):
    events = EventAccumulator(str(output / "tensorboard"), size_guidance={"scalars": 0})
    events.Reload()
    tags = ("loss", "reasoning_loss", "content_loss", "supervised_tokens", "reasoning_tokens",
            "content_tokens", "gradient_norm", "changed_parameter_count", "learning_rate",
            "seconds", "peak_allocated_bytes", "epoch")
    expected = list(range(1, total_steps + 1))
    for tag in tags:
        values = events.Scalars(f"train/{tag}")
        if [event.step for event in values] != expected or any(not math.isfinite(event.value) for event in values):
            raise ValueError(f"TensorBoard training events are incomplete or invalid: {tag}")
    for tag in ("legal", "tokens", "reached_eos"):
        if not any(event.step == total_steps for event in events.Scalars(f"generation/{tag}")):
            raise ValueError("TensorBoard generation events are missing")
    return {"passed": True, "training_steps": total_steps, "tags": events.Tags()["scalars"]}


def finish_report(output, identity, schedule, samples, updates, settings):
    expected_steps = list(range(settings["checkpoint_every_steps"], len(schedule) + 1,
                                settings["checkpoint_every_steps"]))
    if len(schedule) not in expected_steps:
        expected_steps.append(len(schedule))
    checkpoints = []
    for step in expected_steps:
        path = output / f"checkpoint-step-{step}"
        metadata = validate_saved_checkpoint(path, identity, schedule)
        if metadata["step"] != step:
            raise ValueError("Checkpoint step does not match its directory")
        checkpoints.append({"path": path.name, "hash": metadata["hash"], "step": step})
    reload = json.loads((output / "reload-report.json").read_text())
    counts = Counter(cid for row in updates for cid in row["sample_ids"])
    if (len(updates) != len(schedule) or counts != Counter({s["candidate_id"]: settings["epochs"] for s in samples})
            or not reload["passed"] or reload["identity"] != identity
            or reload["checkpoint_hash"] != checkpoints[-1]["hash"]):
        raise ValueError("Training completion checks failed")
    for index, row in enumerate(updates):
        if (row["step"] != index + 1 or row["sample_ids"] != schedule[index]["sample_ids"]
                or row["epoch"] != schedule[index]["epoch"] or row["changed_parameter_count"] < 1
                or not row["gradient_norm"] > 0
                or any(not math.isfinite(row[key]) for key in ("loss", "reasoning_loss", "content_loss", "gradient_norm"))):
            raise ValueError("Invalid training update evidence")
    tensorboard = audit_events(output, len(schedule))
    report = {"passed": True, "identity": identity, "step": len(updates), "total_steps": len(schedule),
        "sample_exposures": dict(counts), "checkpoints": checkpoints, "last_checkpoint": checkpoints[-1]["path"],
        "epochs": [epoch_metrics(updates, epoch) for epoch in range(settings["epochs"])],
        "updates": updates, "reload": reload, "tensorboard": tensorboard, "finished": time.time()}
    write_json(output / "training-report.json", report)
    write_json(output / "status.json", {"status": "completed", "step": len(schedule),
                                       "total_steps": len(schedule), "updated": time.time()})
    return {k: report[k] for k in ("passed", "step", "total_steps", "last_checkpoint", "finished")}


def run_training(config, data, output, device, resume_checkpoint):
    settings = training_settings(config)
    manifest, samples, tokenizer = audit_inputs(config, data)
    schedule = batch_schedule(samples, settings)
    identity = run_identity(config, manifest, schedule)
    metadata = None
    if resume_checkpoint:
        saved = json.loads((output / "run-manifest.json").read_text())
        if saved["identity"] != identity or saved["schedule"] != schedule:
            raise ValueError("Saved run data, configuration, code or schedule mismatch")
        if resume_checkpoint.resolve() != latest_checkpoint(output).resolve():
            raise ValueError("Resume requires the latest published checkpoint of this run")
        metadata = validate_saved_checkpoint(resume_checkpoint, identity, schedule)
    else:
        write_json(output / "run-manifest.json", {"identity": identity, "schedule": schedule,
            "settings": settings, "sample_count": len(samples), "total_steps": len(schedule),
            "data": str(data.resolve()), "created": time.time()})
    attempt = output / "attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    write_json(attempt / "attempt.json", {"started": time.time(), "pid": os.getpid(), "device": device,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None})
    write_json(output / "status.json", {"status": "starting", "attempt": attempt.name,
        "step": metadata["step"] if metadata else 0, "total_steps": len(schedule), "updated": time.time()})
    setup_device(device, config.seed)
    network = load_network(resume_checkpoint or config.policy["model_path"], device)
    optimizer, scheduler = make_optimizer(network, settings)
    write_json(attempt / "model.json", {"parameter_count": sum(p.numel() for p in network.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in network.parameters() if p.requires_grad),
        "parameter_dtypes": dict(Counter(str(p.dtype) for p in network.parameters())),
        "device": torch.cuda.get_device_name(device)})
    updates = []
    if metadata:
        saved_probe = torch.load(resume_checkpoint / "probe.pt", map_location="cpu", weights_only=True)
        sample = sorted(samples, key=lambda item: item["candidate_id"])[0]
        torch.testing.assert_close(probe(network, sample, device), saved_probe, rtol=0, atol=0)
        updates = restore_checkpoint(resume_checkpoint, network, optimizer, scheduler, metadata, schedule, device)
    with SummaryWriter(str(output / "tensorboard"), purge_step=len(updates) + 1 if metadata else None) as writer:
        writer.add_scalar("data/training_samples", len(samples), len(updates))
        writer.add_scalar("data/training_tasks", len({s["task_id"] for s in samples}), len(updates))
        updates = train_loop(output, network, tokenizer, optimizer, scheduler, settings, identity, schedule,
                             samples, updates, device, writer, attempt)
    del network, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    write_json(output / "status.json", {"status": "verifying", "step": len(updates),
        "total_steps": len(schedule), "attempt": attempt.name, "updated": time.time()})
    result = subprocess.run([sys.executable, "-m", "ocop", "train", "--data", str(data.resolve()),
        "--output", str(output.resolve()), "--config", str((output / "config.json").resolve()),
        "--device", device, "--verify-final"], check=False)
    if result.returncode:
        raise ValueError("Final independent reload failed; inspect reload-failure.json and process log")
    return finish_report(output, identity, schedule, samples, updates, settings), 0


def train(config_path, data, output, device, resume_checkpoint=None, verify_final=False):
    if verify_final:
        try:
            return reload_final(load_config(config_path), data, output, device), 0
        except Exception as exc:
            write_json(output / "reload-failure.json", {"passed": False, "error_type": type(exc).__name__,
                "error": str(exc), "finished": time.time()})
            raise
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.resolve()) + ".lock", timeout=0):
        if resume_checkpoint:
            config = load_config(output / "config.json")
            if config_path is not None and load_config(config_path).model_dump() != config.model_dump():
                raise ValueError("Resume configuration differs from the archived run configuration")
        else:
            config = load_config(config_path or Path("config/prototype.json"))
            training_settings(config)
            output.mkdir(exist_ok=False)
            write_json(output / "config.json", config.model_dump())
        try:
            return run_training(config, data, output, device, resume_checkpoint)
        except Exception as exc:
            failure = {"passed": False, "error_type": type(exc).__name__, "error": str(exc), "finished": time.time()}
            write_json(output / f"failure-{uuid.uuid4().hex}.json", failure)
            write_json(output / "status.json", {"status": "failed", **failure})
            raise
