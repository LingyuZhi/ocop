import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from filelock import FileLock, Timeout
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForImageTextToText

import ocop.training.engine as training
from ocop.cli import main
from ocop.runtime.config import RuntimeConfig, load_config
from ocop.runtime.storage import digest
from test_sft import tiny_qwen


ROOT = Path(__file__).resolve().parents[1]


def producer_inputs():
    config = load_config(ROOT / "config/prototype.json")
    manifest = {"hash": "data", "model": {"path": "model", "files": {}}}
    schedule = [{"step": 1}]
    archived = training.run_identity(config, manifest, schedule)
    registry = json.loads((ROOT / "src/ocop/training/producers.json").read_text())
    archived["implementation_hashes"] = registry["producers"][0]["implementation_hashes"]
    return config, manifest, schedule, archived


def test_frozen_producer_is_accepted_for_read_only_checkpoint_consumption():
    args = producer_inputs()
    identity, producer = training.verified_producer_identity(*args)
    assert identity == args[-1]
    assert producer["revision"] == "d4377c83e4aa545ca6793d7abd40c9ff49fdda9d"
    assert identity != training.run_identity(*args[:3])


@pytest.mark.parametrize("field", ["data_hash", "config_hash", "schedule_hash", "model", "versions"])
def test_frozen_producer_does_not_accept_changed_training_inputs(field):
    config, manifest, schedule, archived = producer_inputs()
    archived[field] = "changed"
    with pytest.raises(ValueError, match="mismatch"):
        training.verified_producer_identity(config, manifest, schedule, archived)


def test_unknown_training_implementation_is_rejected():
    config, manifest, schedule, archived = producer_inputs()
    archived["implementation_hashes"] = {"unknown.py": "unknown"}
    with pytest.raises(ValueError, match="unknown production"):
        training.verified_producer_identity(config, manifest, schedule, archived)


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(19, 8)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, input_ids, attention_mask, use_cache):
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.lm_head = torch.nn.Linear(8, 19)

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        hidden = self.model(input_ids, attention_mask, use_cache).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden[:, -logits_to_keep:]))

    def save_pretrained(self, path, safe_serialization):
        torch.save(self.state_dict(), path / "model.pt")


class Tokenizer:
    pad_token_id = 0

    def save_pretrained(self, path):
        (path / "tokenizer.json").write_text(json.dumps({"pad_token_id": 0}))


def samples():
    return [{"candidate_id": str(i), "task_id": str(i), "split": "train", "length": 4 + i % 2,
        "input_ids": [1, 2, 3, 4] + ([5] if i % 2 else []),
        "labels": [-100, -100, 3, 4] + ([5] if i % 2 else []),
        "loss_groups": [0, 0, 1, 2] + ([2] if i % 2 else []),
        "supervised_tokens": 2 + i % 2, "boundaries": {"prefix_end": 2}}
        for i in range(5)]


def settings():
    result = training.training_settings(load_config(ROOT / "config/prototype.json"))
    return {**result, "epochs": 3, "checkpoint_every_steps": 1, "learning_rate": 0.01}


def test_schedule_covers_each_candidate_once_per_epoch_and_keeps_tail():
    schedule = training.batch_schedule(samples(), settings())
    assert schedule == training.batch_schedule(list(reversed(samples())), settings())
    assert len(schedule) == 6
    for epoch in range(3):
        rows = [row for row in schedule if row["epoch"] == epoch]
        assert [len(row["sample_ids"]) for row in rows] == [4, 1]
        assert sorted(cid for row in rows for cid in row["sample_ids"]) == list("01234")
    assert training.next_position(schedule, 2) == {"epoch": 1, "batch": 0}
    assert training.next_position(schedule, 6) == {"epoch": 3, "batch": 0}


@pytest.mark.parametrize(("stop", "save_every"), [(1, 1), (2, 1), (3, 1), (6, 1), (3, 2)])
def test_continuous_and_resumed_training_match_exactly(tmp_path, monkeypatch, stop, save_every):
    run_settings = {**settings(), "checkpoint_every_steps": save_every}
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    network = Network()
    initial = copy.deepcopy(network.state_dict())
    rng = training.capture_rng("cpu")
    schedule = training.batch_schedule(samples(), run_settings)
    identity = {"schedule_hash": digest(schedule)}
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "attempt").mkdir()
    optimizer, scheduler = training.make_optimizer(network, run_settings)
    with SummaryWriter(str(baseline / "tensorboard")) as writer:
        expected = training.train_loop(baseline, network, Tokenizer(), optimizer, scheduler,
            run_settings, identity, schedule, samples(), [], "cpu", writer, baseline / "attempt")
    final_parameters = copy.deepcopy(network.state_dict())
    final_optimizer = copy.deepcopy(optimizer.state_dict())
    final_scheduler = scheduler.state_dict()
    expected_random = (random.random(), np.random.rand(), torch.rand(3))

    output = tmp_path / "resumed"
    output.mkdir()
    (output / "first").mkdir()
    model = Network()
    model.load_state_dict(initial)
    optimizer, scheduler = training.make_optimizer(model, run_settings)
    training.restore_rng(rng, "cpu")
    actual_update = training.update

    def interrupt(*args, **kwargs):
        if args[8] == stop + 1:
            raise RuntimeError("simulated interruption")
        return actual_update(*args, **kwargs)

    monkeypatch.setattr(training, "update", interrupt)
    with SummaryWriter(str(output / "tensorboard")) as writer:
        if stop < len(schedule):
            with pytest.raises(RuntimeError, match="simulated interruption"):
                training.train_loop(output, model, Tokenizer(), optimizer, scheduler, run_settings, identity,
                    schedule, samples(), [], "cpu", writer, output / "first")
        else:
            training.train_loop(output, model, Tokenizer(), optimizer, scheduler, run_settings, identity,
                schedule, samples(), [], "cpu", writer, output / "first")
        writer.add_scalar("train/loss", 999, stop + 1)
    monkeypatch.setattr(training, "update", actual_update)
    path = training.latest_checkpoint(output)
    metadata = training.validate_saved_checkpoint(path, identity, schedule)
    assert metadata["step"] == stop // save_every * save_every
    model = Network()
    model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
    optimizer, scheduler = training.make_optimizer(model, run_settings)
    history = training.restore_checkpoint(path, model, optimizer, scheduler, metadata, schedule, "cpu")
    (output / "second").mkdir()
    with SummaryWriter(str(output / "tensorboard"), purge_step=metadata["step"] + 1) as writer:
        actual = training.train_loop(output, model, Tokenizer(), optimizer, scheduler, run_settings, identity,
            schedule, samples(), history, "cpu", writer, output / "second")
        for tag in ("legal", "tokens", "reached_eos"):
            writer.add_scalar(f"generation/{tag}", 0, len(schedule))
    assert [row["loss"] for row in actual] == [row["loss"] for row in expected]
    assert [row["sample_ids"] for row in actual] == [row["sample_ids"] for row in expected]
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, final_parameters[name], rtol=0, atol=0)
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, final_optimizer["state"][key][name], rtol=0, atol=0)
    assert scheduler.state_dict() == final_scheduler
    actual_random = (random.random(), np.random.rand(), torch.rand(3))
    assert actual_random[:2] == expected_random[:2]
    torch.testing.assert_close(actual_random[2], expected_random[2], rtol=0, atol=0)
    assert training.audit_events(output, len(schedule))["passed"]
    report = {"passed": True, "identity": identity,
              "checkpoint_hash": json.loads((training.latest_checkpoint(output) / "checkpoint.json").read_text())["hash"]}
    (output / "reload-report.json").write_text(json.dumps(report))
    assert training.finish_report(output, identity, schedule, samples(), actual, run_settings)["passed"]


def checkpoint_fixture(tmp_path):
    model = Network()
    optimizer, scheduler = training.make_optimizer(model, settings())
    schedule = training.batch_schedule(samples(), settings())
    entry = schedule[0]
    selected = {s["candidate_id"]: s for s in samples()}
    with SummaryWriter(str(tmp_path / "tensorboard")) as writer:
        metrics = training.update(model, optimizer, scheduler, [selected[cid] for cid in entry["sample_ids"]],
            Tokenizer(), "cpu", settings(), writer, 1)
    metrics.update(entry)
    path = training.save_checkpoint(tmp_path, model, Tokenizer(), optimizer, scheduler,
        {"data": "original"}, schedule, [metrics], samples()[0], "cpu")
    return path, model, optimizer, scheduler, schedule, metrics


def test_checkpoint_corruption_and_identity_changes_are_rejected(tmp_path):
    path, _, _, _, schedule, _ = checkpoint_fixture(tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        training.validate_saved_checkpoint(path, {"data": "changed"}, schedule)
    (path / "training-state.pt").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        training.validate_saved_checkpoint(path, {"data": "original"}, schedule)


def test_partial_save_is_not_published_or_selected(tmp_path, monkeypatch):
    path, model, optimizer, scheduler, schedule, metrics = checkpoint_fixture(tmp_path)
    def fail(path):
        raise OSError("simulated disk failure")
    tokenizer = Tokenizer()
    monkeypatch.setattr(tokenizer, "save_pretrained", fail)
    with pytest.raises(OSError, match="disk failure"):
        training.save_checkpoint(tmp_path, model, tokenizer, optimizer, scheduler,
            {"data": "original"}, schedule, [metrics, metrics], samples()[0], "cpu")
    assert training.latest_checkpoint(tmp_path) == path
    assert not (tmp_path / "checkpoint-step-2").exists()
    assert len(list(tmp_path.glob("*.pending"))) == 1
    assert json.loads((tmp_path / "latest-checkpoint.json").read_text())["step"] == 1
    with pytest.raises(ValueError, match="already exists"):
        training.save_checkpoint(tmp_path, model, Tokenizer(), optimizer, scheduler,
            {"data": "original"}, schedule, [metrics], samples()[0], "cpu")


def test_published_checkpoint_survives_pointer_write_interruption(tmp_path, monkeypatch):
    path, _, _, _, _, _ = checkpoint_fixture(tmp_path)
    (tmp_path / "latest-checkpoint.json").unlink()
    assert training.latest_checkpoint(tmp_path) == path


def test_qwen_checkpoint_reloads_weights_logits_and_optimizer(tmp_path):
    model = tiny_qwen()
    optimizer, scheduler = training.make_optimizer(model, settings())
    schedule = training.batch_schedule(samples(), settings())
    selected = {s["candidate_id"]: s for s in samples()}
    with SummaryWriter(str(tmp_path / "tensorboard")) as writer:
        metrics = training.update(model, optimizer, scheduler,
            [selected[cid] for cid in schedule[0]["sample_ids"]], Tokenizer(), "cpu", settings(), writer, 1)
    metrics.update(schedule[0])
    path = training.save_checkpoint(tmp_path, model, Tokenizer(), optimizer, scheduler,
        {}, schedule, [metrics], samples()[0], "cpu")
    metadata = training.validate_saved_checkpoint(path, {}, schedule)
    restored = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True, attn_implementation="sdpa")
    restored.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    saved = torch.load(path / "probe.pt", weights_only=True)
    torch.testing.assert_close(training.probe(restored, samples()[0], "cpu"), saved, rtol=0, atol=0)
    new_optimizer, new_scheduler = training.make_optimizer(restored, settings())
    assert training.restore_checkpoint(path, restored, new_optimizer, new_scheduler, metadata, schedule, "cpu") == [metrics]
    assert new_scheduler.last_epoch == 1
    assert all(state["step"].item() == 1 for state in new_optimizer.state.values())


def test_resume_rejects_older_published_checkpoint(tmp_path, monkeypatch):
    config = load_config(ROOT / "config/prototype.json")
    data = samples()
    schedule = training.batch_schedule(data, training.training_settings(config))
    identity = {"data": "fixed"}
    (tmp_path / "run-manifest.json").write_text(json.dumps({"identity": identity, "schedule": schedule}))
    for step in (1, 2):
        path = tmp_path / f"checkpoint-step-{step}"
        path.mkdir()
        (path / "checkpoint.json").write_text(json.dumps({"step": step}))
    monkeypatch.setattr(training, "audit_inputs", lambda *args: ({}, data, Tokenizer()))
    monkeypatch.setattr(training, "run_identity", lambda *args: identity)
    with pytest.raises(ValueError, match="latest published"):
        training.run_training(config, tmp_path, tmp_path, "cuda:0", tmp_path / "checkpoint-step-1")
    assert not (tmp_path / "attempts").exists()


def test_training_output_and_concurrent_writer_are_rejected(tmp_path):
    config = ROOT / "config/prototype.json"
    with pytest.raises(FileExistsError):
        training.train(config, tmp_path / "unused", tmp_path, "cuda:0")
    output = tmp_path / "run"
    with FileLock(str(output.resolve()) + ".lock", timeout=0):
        with pytest.raises(Timeout):
            training.train(config, tmp_path / "unused", output, "cuda:0")
    assert not output.exists()


def test_resume_uses_archived_config_and_rejects_override(tmp_path, monkeypatch):
    original = load_config(ROOT / "config/prototype.json")
    (tmp_path / "config.json").write_text(original.model_dump_json())
    changed = original.model_copy(deep=True)
    changed.training["epochs"] = 4
    override = tmp_path / "override.json"
    override.write_text(changed.model_dump_json())
    with pytest.raises(ValueError, match="configuration differs"):
        training.train(override, tmp_path, tmp_path, "cuda:0", tmp_path / "checkpoint-step-6")
    received = []
    monkeypatch.setattr(training, "run_training", lambda config, *args: received.append(config) or ({}, 0))
    training.train(None, tmp_path, tmp_path, "cuda:0", tmp_path / "checkpoint-step-6")
    assert received[0].model_dump() == original.model_dump()


def test_old_configuration_serialization_preserves_identity():
    config = load_config(ROOT / "config/prototype.json").model_dump()
    config.pop("training_runtime")
    parsed = RuntimeConfig.model_validate(config)
    assert parsed.training_runtime is None
    assert parsed.model_dump() == config
    assert digest(parsed.model_dump()) == digest(config)


def test_training_config_is_explicit_and_validates_batch_size():
    config = load_config(ROOT / "config/prototype.json")
    assert training.training_settings(config)["checkpoint_every_steps"] == 6
    assert len(training.batch_schedule([{"candidate_id": str(i)} for i in range(72)],
                                     training.training_settings(config))) == 54
    config.training_runtime.gradient_accumulation = 2
    with pytest.raises(ValueError, match="batch size"):
        training.training_settings(config)


def test_epoch_loss_is_token_weighted():
    rows = [{"epoch": 0, "loss": 1.0, "reasoning_loss": 2.0, "content_loss": 0.5,
             "supervised_tokens": 3, "reasoning_tokens": 1, "content_tokens": 2},
            {"epoch": 0, "loss": 3.0, "reasoning_loss": 4.0, "content_loss": 2.0,
             "supervised_tokens": 2, "reasoning_tokens": 1, "content_tokens": 1}]
    actual = training.epoch_metrics(rows, 0)
    assert actual["loss"] == 1.8
    assert actual["reasoning_loss"] == 3
    assert actual["content_loss"] == 1


def test_train_cli_arguments_and_error_reporting(tmp_path, monkeypatch, capsys):
    def run(config, data, output, device, resume, verify):
        assert config is None and device == "cuda:0" and not verify and resume is None
        raise ValueError("deliberate validation error")
    monkeypatch.setattr(training, "train", run)
    assert main(["train", "--data", str(tmp_path), "--output", str(tmp_path / "run")]) == 2
    assert "deliberate validation error" in capsys.readouterr().err
