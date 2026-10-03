import gc
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForImageTextToText, AutoTokenizer, GenerationConfig

from ocop.collection import write_json
from ocop.graph import replay
from ocop.trajectories import digest, encode_sample, file_hash, load_prepared, model_identity, policy_messages


def verification_settings(config):
    settings = config.training
    required = {"representation": "raw_traj", "full_parameter": True, "precision": "bf16",
        "optimizer": "adamw", "effective_batch_size": 4, "gradient_checkpointing": True,
        "overlength_policy": "report_and_wait"}
    if any(settings.get(k) != v for k, v in required.items()) or not config.policy.get("thinking"):
        raise ValueError("Unsupported SFT verification configuration")
    verification = config.diagnostics["sft_verification"]
    fixed = {"microbatch_size": 1, "gradient_accumulation": 4, "optimizer_steps": 2,
             "resume_steps": 1, "scheduler": "constant", "attention": "sdpa",
             "precision": "bf16 parameters and Adam moments; fp32 cross entropy"}
    if any(verification.get(k) != v for k, v in fixed.items()):
        raise ValueError("Unsupported SFT verification settings")
    if verification["loss_chunk_tokens"] < 1 or verification["generation_max_new_tokens"] < 1:
        raise ValueError("Token budgets must be positive")
    return {**verification, "learning_rate": settings["learning_rate"], "seed": config.seed}


def select_samples(samples, seed):
    if len(samples) < 8:
        raise ValueError("Verification requires at least eight training candidates")
    selected = [max(samples, key=lambda s: (s["length"], s["candidate_id"]))]
    zeros = sorted((s for s in samples if s["z"] == 0), key=lambda s: s["candidate_id"])
    if zeros and zeros[0]["candidate_id"] != selected[0]["candidate_id"]:
        selected.append(zeros[0])
    remaining = sorted((s for s in samples if s["candidate_id"] not in {s["candidate_id"] for s in selected}),
                       key=lambda s: s["candidate_id"])
    random.Random(seed).shuffle(remaining)
    return selected + remaining[:8 - len(selected)]


def collate(samples, pad_token_id, device):
    length = max(s["length"] for s in samples)
    values = {"input_ids": [], "attention_mask": [], "labels": [], "loss_groups": []}
    for sample in samples:
        pad = length - sample["length"]
        values["input_ids"].append(sample["input_ids"] + [pad_token_id] * pad)
        values["attention_mask"].append([1] * sample["length"] + [0] * pad)
        values["labels"].append(sample["labels"] + [-100] * pad)
        values["loss_groups"].append(sample["loss_groups"] + [0] * pad)
    return {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in values.items()}


def token_losses(network, batch, chunk_size):
    hidden = network.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           use_cache=False).last_hidden_state[:, :-1]
    labels, groups = batch["labels"][:, 1:], batch["loss_groups"][:, 1:]
    mask = labels != -100
    hidden, labels, groups = hidden[mask], labels[mask], groups[mask]

    def loss_chunk(states, targets):
        return F.cross_entropy(network.lm_head(states).float(), targets, reduction="none")

    losses = torch.cat([checkpoint(loss_chunk, hidden[i:i + chunk_size], labels[i:i + chunk_size],
                                   use_reentrant=False) for i in range(0, len(labels), chunk_size)])
    return losses, groups


def parameter_hashes(network):
    return {name: hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, p in network.named_parameters()}


def optimizer_dtypes(optimizer):
    return dict(Counter(str(value.dtype) for state in optimizer.state.values()
                        for name, value in state.items() if isinstance(value, torch.Tensor) and name != "step"))


def update(network, optimizer, scheduler, samples, tokenizer, device, settings, writer, step,
           namespace="verification"):
    network.train()
    optimizer.zero_grad(set_to_none=True)
    before = parameter_hashes(network)
    total = sum(s["supervised_tokens"] for s in samples)
    sums, counts = {1: 0.0, 2: 0.0}, {1: 0, 2: 0}
    started = time.monotonic()
    for sample in samples:
        batch = collate([sample], tokenizer.pad_token_id, device)
        losses, groups = token_losses(network, batch, settings["loss_chunk_tokens"])
        if not torch.isfinite(losses).all():
            raise ValueError("Non-finite token loss")
        (losses.sum() / total).backward()
        for group in (1, 2):
            sums[group] += losses.detach()[groups == group].sum().item()
            counts[group] += (groups == group).sum().item()
        del losses, groups, batch
    gradients = [name for name, p in network.named_parameters() if p.grad is not None]
    missing_text = [name for name, p in network.named_parameters()
                    if "visual" not in name and p.requires_grad and p.grad is None]
    if missing_text:
        raise ValueError(f"Text parameters lack gradients: {missing_text}")
    grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), settings["max_grad_norm"],
                                              error_if_nonfinite=True).item()
    if not grad_norm > 0:
        raise ValueError("No nonzero gradient")
    optimizer.step()
    scheduler.step()
    after = parameter_hashes(network)
    changed = [name for name in before if before[name] != after[name]]
    if not changed:
        raise ValueError("Optimizer step did not change parameters")
    for p in network.parameters():
        if not torch.isfinite(p).all():
            raise ValueError("Non-finite parameter after optimizer step")
    metrics = {"step": step, "loss": sum(sums.values()) / total, "reasoning_loss": sums[1] / counts[1],
        "content_loss": sums[2] / counts[2], "supervised_tokens": total, "reasoning_tokens": counts[1],
        "content_tokens": counts[2], "gradient_norm": grad_norm, "seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda:") else 0,
        "learning_rate": optimizer.param_groups[0]["lr"], "changed_parameter_count": len(changed),
        "changed_parameters": changed,
        "parameters_with_gradients": gradients, "optimizer_state_dtypes": optimizer_dtypes(optimizer)}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{namespace}/{key}", value, step)
    writer.flush()
    optimizer.zero_grad(set_to_none=True)
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"changed_parameters", "parameters_with_gradients"}}), flush=True)
    return metrics


def probe(network, sample, device):
    network.eval()
    inputs = torch.tensor([sample["input_ids"][:sample["boundaries"]["prefix_end"]]], device=device)
    with torch.inference_mode():
        return network(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False,
                       logits_to_keep=1).logits.float().cpu()


def checkpoint_save(path, network, tokenizer, optimizer, scheduler, manifest, settings, samples, device):
    path.mkdir()
    network.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": 2,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
    torch.save(state, path / "training-state.pt")
    torch.save(probe(network, samples[0], device), path / "probe.pt")
    metadata = {"data_hash": manifest["hash"], "settings": settings,
        "sample_ids": [s["candidate_id"] for s in samples], "parameter_hashes": parameter_hashes(network),
        "optimizer_state_dtypes": optimizer_dtypes(optimizer),
        "files": {p.name: file_hash(p) for p in sorted(path.iterdir()) if p.is_file()}}
    write_json(path / "verification-state.json", metadata)


def validate_checkpoint(path, manifest, settings, samples):
    metadata = json.loads((path / "verification-state.json").read_text())
    if (metadata["data_hash"] != manifest["hash"] or metadata["settings"] != settings
            or metadata["sample_ids"] != [s["candidate_id"] for s in samples]):
        raise ValueError("Checkpoint data, settings or sample order mismatch")
    if any(file_hash(path / name) != value for name, value in metadata["files"].items()):
        raise ValueError("Checkpoint file checksum mismatch")
    return metadata


def run_verification(config, data, output, device, reload_checkpoint, config_path):
    manifest, samples = load_prepared(data)
    settings = verification_settings(config)
    if manifest["training"] != config.training or manifest["policy"] != config.policy or manifest["seed"] != config.seed:
        raise ValueError("Prepared data and runtime configuration mismatch")
    if manifest["model"] != model_identity(Path(config.policy["model_path"])):
        raise ValueError("Base model or tokenizer files changed")
    tokenizer = AutoTokenizer.from_pretrained(reload_checkpoint or config.policy["model_path"], local_files_only=True)
    for sample in samples:
        if sample["split"] != "train" or encode_sample(sample, tokenizer) != sample:
            raise ValueError("Prepared token mask audit failed")
    selected = select_samples(samples, config.seed)
    if not torch.cuda.is_available() or not device.startswith("cuda:"):
        raise ValueError("GPU verification requires an available CUDA device")
    torch.cuda.set_device(device)
    if not reload_checkpoint and torch.cuda.mem_get_info(device)[0] < 23 * 1024**3:
        raise ValueError("Selected GPU is not sufficiently idle (requires 23 GiB free)")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = validate_checkpoint(reload_checkpoint, manifest, settings, selected) if reload_checkpoint else None
    model_path = reload_checkpoint or Path(config.policy["model_path"])
    network = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    network.requires_grad_(True)
    network.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(network.parameters(), lr=settings["learning_rate"], betas=tuple(settings["betas"]),
        eps=settings["epsilon"], weight_decay=settings["weight_decay"], foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    report = {"passed": False, "data_hash": manifest["hash"], "settings": settings,
        "config_hash": digest(config.model_dump()),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "tensorboard")},
        "implementation_hashes": {name: file_hash(Path(__file__).with_name(name)) for name in ("training.py", "trajectories.py")},
        "sample_ids": [s["candidate_id"] for s in selected], "mask_audit_samples": len(samples),
        "parameter_count": sum(p.numel() for p in network.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in network.parameters() if p.requires_grad),
        "parameter_dtypes": dict(Counter(str(p.dtype) for p in network.parameters())),
        "device": torch.cuda.get_device_name(device), "updates": []}
    filename = "reload-report.json" if reload_checkpoint else "verification.json"
    with SummaryWriter(str(output / ("tensorboard-reload" if reload_checkpoint else "tensorboard"))) as writer:
        writer.add_scalar("data/training_samples", len(samples), 0)
        if reload_checkpoint:
            if parameter_hashes(network) != metadata["parameter_hashes"]:
                raise ValueError("Reloaded parameters differ from saved parameters")
            saved_probe = torch.load(reload_checkpoint / "probe.pt", map_location="cpu", weights_only=True)
            actual_probe = probe(network, selected[0], device)
            torch.testing.assert_close(actual_probe, saved_probe, rtol=0, atol=0)
            report["reload_logits_max_abs_diff"] = (actual_probe - saved_probe).abs().max().item()
            state = torch.load(reload_checkpoint / "training-state.pt", map_location="cpu", weights_only=False)
            if state["step"] != 2:
                raise ValueError("Unexpected checkpoint optimizer step")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"], device)
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            del state
            report["updates"].append(update(network, optimizer, scheduler, selected[:4], tokenizer,
                                             device, settings, writer, 3))
            if any(s["step"].item() != 3 for s in optimizer.state.values()):
                raise ValueError("Optimizer state did not resume at step three")
            network.eval()
            messages = policy_messages(selected[0]["question"], selected[0]["z"])
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=True).to(device)
            generation = GenerationConfig(do_sample=False, max_new_tokens=settings["generation_max_new_tokens"],
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
            with torch.inference_mode():
                generated = network.generate(**inputs, generation_config=generation)
            ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
            raw = tokenizer.decode(ids, skip_special_tokens=False)
            reasoning, separator, content = raw.partition("</think>")
            content = content.removesuffix(tokenizer.eos_token).strip() if separator else ""
            parsed = replay(content, reasoning=reasoning)
            report["generation"] = {"purpose": "training_task_smoke_after_resume_update", "candidate_id": selected[0]["candidate_id"],
                "config": generation.to_dict(), "token_ids": ids, "raw": raw, "valid_graph": parsed.valid,
                "parse_error": None if parsed.valid else parsed.error.code, "reached_eos": bool(ids) and ids[-1] == tokenizer.eos_token_id}
            writer.add_scalar("generation/legal", int(parsed.valid), 3)
            report["passed"] = True
            write_json(output / filename, report)
            return report, 0
        for offset in (0, 4):
            report["updates"].append(update(network, optimizer, scheduler, selected[offset:offset + 4],
                                             tokenizer, device, settings, writer, offset // 4 + 1))
            write_json(output / filename, report)
        checkpoint_path = output / "checkpoint-step-2"
        checkpoint_save(checkpoint_path, network, tokenizer, optimizer, scheduler, manifest, settings, selected, device)
    del network, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    result = subprocess.run([sys.executable, "-m", "ocop", "verify-sft", "--data", str(data.resolve()),
        "--output", str(output.resolve()), "--config", str(config_path.resolve()), "--device", device,
        "--reload-checkpoint", str(checkpoint_path.resolve())], check=False)
    if result.returncode != 0:
        raise ValueError("Independent checkpoint reload failed; inspect reload report and process log")
    report["reload"] = json.loads((output / "reload-report.json").read_text())
    report["checkpoint"] = str(checkpoint_path)
    report["passed"] = report["reload"]["passed"]
    write_json(output / filename, report)
    return report, 0 if report["passed"] else 1


def verify_sft(config, data, output, device, *, reload_checkpoint=None, config_path):
    if not reload_checkpoint:
        output.mkdir(parents=True, exist_ok=False)
        config_path = output / "config.json"
        write_json(config_path, config.model_dump())
    started = time.time()
    try:
        return run_verification(config, data, output, device, reload_checkpoint, config_path)
    except Exception as exc:
        name = "reload-failure.json" if reload_checkpoint else "failure.json"
        write_json(output / name, {"passed": False, "error_type": type(exc).__name__, "error": str(exc),
            "started": started, "finished": time.time(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else None})
        raise
