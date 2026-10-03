import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen3_5Config

import ocop.trajectories as trajectories
from ocop.config import load_config
from ocop.training import collate, select_samples, token_losses, validate_checkpoint, verification_settings
from ocop.trajectories import digest, encode_sample, load_prepared, prepare_sft, source_samples
from test_collection import collect, local_environment, reply, runtime


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/zhilingyu/models/Qwen3.5-2B")


@pytest.fixture(scope="module")
def tokenizer():
    if not (MODEL / "tokenizer.json").exists():
        pytest.skip("Real Qwen tokenizer is required for template integration checks")
    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


def raw_sample():
    return {"candidate_id": "candidate", "task_id": "task", "split": "train", "question": "What is 2+3?",
            "z": 0.0, "raw_reasoning": "  Design an organization. 思考\n",
            "raw_content": (ROOT / "examples/graph/valid.json").read_text()}


def test_native_template_masks_reasoning_stop_and_end(tokenizer):
    sample = encode_sample(raw_sample(), tokenizer)
    start = sample["boundaries"]["prefix_end"]
    end = sample["boundaries"]["assistant_end"]
    assert sample["labels"][:start] == [-100] * start
    assert sample["labels"][start:end + 1] == sample["input_ids"][start:end + 1]
    assert sample["labels"][end] == tokenizer.eos_token_id
    target = tokenizer.decode([v for v in sample["labels"] if v != -100])
    assert "</think>" in target and "STOP" in target
    assert "<think>" not in target and "What is 2+3?" not in target
    assert sample["supervised_tokens"] == sample["reasoning_tokens"] + sample["content_tokens"]
    assert sample["raw_reasoning"] == raw_sample()["raw_reasoning"]
    assert encode_sample(sample, tokenizer) == sample
    groups = sample["loss_groups"]
    assert "</think>" in tokenizer.decode([v for v, g in zip(sample["input_ids"], groups) if g == 1])
    assert "STOP" in tokenizer.decode([v for v, g in zip(sample["input_ids"], groups) if g == 2])


@pytest.mark.parametrize("reasoning", [None, "", "   ", "bad <think> marker"])
def test_missing_or_reserved_reasoning_rejected(tokenizer, reasoning):
    with pytest.raises(ValueError):
        encode_sample({**raw_sample(), "raw_reasoning": reasoning}, tokenizer)


def test_padding_is_ignored(tokenizer):
    short = encode_sample(raw_sample(), tokenizer)
    long = encode_sample({**raw_sample(), "raw_reasoning": "Long reason. " * 30}, tokenizer)
    batch = collate([short, long], tokenizer.pad_token_id, "cpu")
    assert torch.all(batch["labels"][0, short["length"]:] == -100)
    assert torch.all(batch["attention_mask"][0, short["length"]:] == 0)
    assert torch.all(batch["loss_groups"][0, short["length"]:] == 0)
    assert batch["input_ids"].shape == batch["labels"].shape


def test_source_preserves_zero_duplicates_and_isolates_tasks(tmp_path):
    config = runtime(tmp_path, candidates=3)
    collect(config, lambda r: reply(r, answer="#### 6"))
    path = Path(config.artifacts_dir) / "runs/test"
    before = (path / "records.sqlite3").read_bytes()
    samples, info = source_samples(path)
    assert len(samples) == 3
    assert all(s["z"] == 0 and s["split"] == "train" for s in samples)
    assert len({s["source"]["graph_fingerprint"] for s in samples}) == 1
    assert len({s["candidate_id"] for s in samples}) == 3
    assert all(len(s["source"]["complete_execution_ids"]) == 5 for s in samples)
    assert len(info["excluded_labels"]) == 3
    assert all(s["split"] == "eval" for s in info["excluded_labels"])
    assert "ReferenceSecret" not in json.dumps(samples)
    assert (path / "records.sqlite3").read_bytes() == before


def test_prepare_reports_overlength_and_blocks_loading(tmp_path, tokenizer, monkeypatch):
    config = runtime(tmp_path)
    collect(config, reply)
    config.training["max_sequence_length"] = 100
    monkeypatch.setattr(trajectories, "model_identity", lambda path: {"path": str(path), "files": {}})
    output = tmp_path / "prepared"
    report, status = prepare_sft(config, Path(config.artifacts_dir) / "runs/test", output)
    assert status == 2 and report["overlength"] and not report["ready"]
    assert json.loads((output / "samples.json").read_text())[0]["length"] > 100
    with pytest.raises(ValueError, match="Overlength"):
        load_prepared(output)
    with pytest.raises(ValueError, match="already exists"):
        prepare_sft(config, Path(config.artifacts_dir) / "runs/test", output)


def test_data_tampering_rejected(tmp_path, tokenizer, monkeypatch):
    config = runtime(tmp_path)
    collect(config, reply)
    monkeypatch.setattr(trajectories, "model_identity", lambda path: {"path": str(path), "files": {}})
    output = tmp_path / "prepared"
    report, status = prepare_sft(config, Path(config.artifacts_dir) / "runs/test", output)
    assert status == 0 and report["sample_count"] == 1
    manifest, samples = load_prepared(output)
    samples[0]["labels"][0] = 42
    (output / "samples.json").write_text(json.dumps(samples))
    with pytest.raises(ValueError, match="data hash"):
        load_prepared(output)
    manifest["ready"] = False
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest hash"):
        load_prepared(output)


def test_selection_includes_longest_zero_and_is_deterministic():
    samples = [{"candidate_id": str(i), "length": i + 1, "z": float(i > 0)} for i in range(12)]
    selected = select_samples(samples, 42)
    assert selected == select_samples(list(reversed(samples)), 42)
    assert len({s["candidate_id"] for s in selected}) == 8
    assert selected[0]["candidate_id"] == "11" and selected[1]["z"] == 0


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(19, 8)

    def forward(self, input_ids, attention_mask, use_cache):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = torch.nn.Linear(8, 19)


def test_chunked_loss_and_accumulated_gradients_match_token_mean():
    torch.manual_seed(42)
    actual = TinyNetwork()
    expected = copy.deepcopy(actual)
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5], [2, 4, 7, 8, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, -100, 3, 4, 5], [-100, 4, 7, 8, -100]]),
        "loss_groups": torch.tensor([[0, 0, 1, 2, 2], [0, 1, 1, 2, 0]])}
    total = (batch["labels"][:, 1:] != -100).sum()
    sums = []
    for i in range(2):
        losses, groups = token_losses(actual, {k: v[i:i + 1] for k, v in batch.items()}, 2)
        sums.append(losses.detach().sum())
        assert (groups > 0).all()
        (losses.sum() / total).backward()
    hidden = expected.model(batch["input_ids"], batch["attention_mask"], False).last_hidden_state
    reference = F.cross_entropy(expected.lm_head(hidden[:, :-1]).reshape(-1, 19),
                                batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
    reference.backward()
    torch.testing.assert_close(sum(sums) / total, reference)
    for a, b in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize("field", ["data_hash", "settings", "sample_ids"])
def test_checkpoint_rejects_changed_inputs_before_loading(tmp_path, field):
    metadata = {"data_hash": "data", "settings": {"lr": 1e-5}, "sample_ids": ["c"], "files": {}}
    metadata[field] = "changed"
    (tmp_path / "verification-state.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mismatch"):
        validate_checkpoint(tmp_path, {"hash": "data"}, {"lr": 1e-5}, [{"candidate_id": "c"}])


def test_checkpoint_rejects_tampered_state(tmp_path):
    (tmp_path / "training-state.pt").write_bytes(b"modified")
    metadata = {"data_hash": "data", "settings": {}, "sample_ids": [], "files": {"training-state.pt": "incorrect"}}
    (tmp_path / "verification-state.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum"):
        validate_checkpoint(tmp_path, {"hash": "data"}, {}, [])


def test_verification_configuration_is_explicit():
    config = load_config(ROOT / "config/prototype.json")
    settings = verification_settings(config)
    assert settings["optimizer_steps"] == 2 and settings["resume_steps"] == 1
    assert settings["gradient_accumulation"] * settings["microbatch_size"] == 4
    config.training["full_parameter"] = False
    with pytest.raises(ValueError, match="Unsupported"):
        verification_settings(config)


def tiny_qwen():
    config = Qwen3_5Config(text_config={"vocab_size": 19, "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention"], "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_num_key_heads": 2,
        "linear_num_value_heads": 2, "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
            "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 16})
    network = AutoModelForImageTextToText.from_config(config, attn_implementation="sdpa")
    network.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return network


def test_qwen_native_loss_matches_training_loss_with_checkpointing():
    network = tiny_qwen()
    network.train()
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3, 4, 5]]), "loss_groups": torch.tensor([[0, 0, 1, 2, 2]])}
    losses, _ = token_losses(network, batch, 2)
    native = network(**{k: v for k, v in batch.items() if k != "loss_groups"}, use_cache=False).loss
    torch.testing.assert_close(losses.mean(), native)
    losses.mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for name, p in network.named_parameters() if "visual" not in name)
