import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from ocop.runtime.config import load_config
from ocop.training.engine import audit_events, audit_inputs, batch_schedule, epoch_metrics, latest_checkpoint, next_position, training_settings, validate_saved_checkpoint, verified_producer_identity
from ocop.runtime.storage import file_hash


def read_json(path):
    return json.loads(path.read_text())


def verify(path):
    report_files = ["config.json", "run-manifest.json", "training-report.json", "reload-report.json", "status.json"]
    before = {name: file_hash(path / name) for name in report_files}
    config = load_config(path / "config.json")
    run = read_json(path / "run-manifest.json")
    report = read_json(path / "training-report.json")
    reload = read_json(path / "reload-report.json")
    status = read_json(path / "status.json")
    settings = training_settings(config)
    manifest, samples, _ = audit_inputs(config, Path(run["data"]))
    schedule = batch_schedule(samples, settings)
    identity, _ = verified_producer_identity(config, manifest, schedule, run["identity"])
    total = len(schedule)
    updates = report["updates"]
    counts = Counter(cid for row in updates for cid in row["sample_ids"])
    expected_steps = sorted(set(range(settings["checkpoint_every_steps"], total + 1,
                                      settings["checkpoint_every_steps"])) | {total})
    checkpoints = []
    for step in expected_steps:
        checkpoint = path / f"checkpoint-step-{step}"
        metadata = validate_saved_checkpoint(checkpoint, identity, schedule)
        if metadata["step"] != step:
            raise ValueError("Checkpoint directory and metadata step mismatch")
        checkpoints.append({"path": checkpoint.name, "hash": metadata["hash"], "step": step})
        print(f"Verified checkpoint {step}/{total}", file=sys.stderr, flush=True)
    final = path / checkpoints[-1]["path"]
    state = torch.load(final / "training-state.pt", map_location="cpu", weights_only=False, mmap=True)
    tensorboard = audit_events(path, total)
    epochs = [epoch_metrics(updates, epoch) for epoch in range(settings["epochs"])]
    checks = {
        "identity": run["identity"] == report["identity"] == reload["identity"] == identity,
        "schedule": run["schedule"] == schedule and run["settings"] == settings,
        "completed": report["passed"] and status["status"] == "completed"
            and report["step"] == report["total_steps"] == status["step"] == status["total_steps"] == total,
        "sample_coverage": counts == Counter({s["candidate_id"]: settings["epochs"] for s in samples})
            and dict(counts) == report["sample_exposures"],
        "update_history": len(updates) == total and all(row["step"] == i + 1
            and all(row[key] == schedule[i][key] for key in ("epoch", "batch", "sample_ids"))
            for i, row in enumerate(updates)),
        "finite_updates": all(row["gradient_norm"] > 0 and row["changed_parameter_count"] > 0
            and row["changed_parameter_count"] == len(row["changed_parameters"])
            and all(math.isfinite(row[key]) for key in ("loss", "reasoning_loss", "content_loss", "gradient_norm"))
            for row in updates),
        "epoch_aggregates": epochs == report["epochs"] and all(epoch["supervised_tokens"]
            == sum(s["supervised_tokens"] for s in samples) for epoch in epochs),
        "checkpoint_checksums": True,
        "checkpoint_inventory": report["checkpoints"] == checkpoints
            and {p.name for p in path.glob("checkpoint-step-*")} == {c["path"] for c in checkpoints},
        "last_checkpoint": latest_checkpoint(path) == final and report["last_checkpoint"] == final.name
            and read_json(path / "latest-checkpoint.json") == checkpoints[-1],
        "training_state": state["step"] == total and state["next_position"] == next_position(schedule, total)
            and state["updates"] == updates and state["scheduler"]["last_epoch"] == total
            and bool(state["optimizer"]["state"])
            and all(s["step"].item() == total for s in state["optimizer"]["state"].values())
            and all(state["rng"][key] is not None for key in ("torch", "python", "numpy", "cuda")),
        "independent_reload": reload == report["reload"] and reload["passed"]
            and reload["checkpoint"] == final.name and reload["step"] == total
            and reload["checkpoint_hash"] == checkpoints[-1]["hash"]
            and reload["parameter_hashes_match"] and reload["reload_logits_max_abs_diff"] == 0,
        "tensorboard": tensorboard == report["tensorboard"],
        "source_reports_unchanged": before == {name: file_hash(path / name) for name in report_files},
    }
    generation = reload["generation"]
    return {"passed": all(checks.values()), "checks": checks, "run": str(path.resolve()),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": datetime.fromtimestamp(report["finished"], timezone.utc).isoformat(),
        "elapsed_minutes": (report["finished"] - run["created"]) / 60,
        "steps": total, "sample_count": len(samples), "sample_exposures": dict(Counter(counts.values())),
        "checkpoint_count": len(checkpoints), "last_checkpoint": str(final.resolve()),
        "checkpoints_bytes": sum(f.stat().st_size for c in checkpoints for f in (path / c["path"]).iterdir() if f.is_file()),
        "epochs": epochs, "peak_allocated_gib": max(row["peak_allocated_bytes"] for row in updates) / 1024**3,
        "changed_parameters_range": [min(row["changed_parameter_count"] for row in updates),
                                     max(row["changed_parameter_count"] for row in updates)],
        "generation": {"tokens": len(generation["token_ids"]), "reached_eos": generation["reached_eos"],
                       "valid_graph": generation["valid_graph"], "parse_error": generation["parse_error"]},
        "source_report_hashes": before}


def main():
    parser = argparse.ArgumentParser(description="Read-only verification of completed full SFT artifacts")
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.resolve().is_relative_to(args.path.resolve()):
        parser.error("Verification output must be outside the training run directory")
    report = verify(args.path)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
