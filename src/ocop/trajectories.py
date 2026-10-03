import hashlib
import importlib.metadata
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from ocop.collection import proposal_template, write_json
from ocop.collection_validation import verify
from ocop.config import canonical_json
from ocop.graph import contract_hash, replay
from ocop.storage import read_database


SFT_VERSION = "ocop.raw_sft.v1"


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_identity(path):
    names = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"}
    files = sorted(p for p in path.iterdir() if p.name in names or p.suffix == ".safetensors" or p.name.endswith(".index.json"))
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Model weights are missing")
    return {"path": str(path.resolve()), "files": {p.name: file_hash(p) for p in files}}


def policy_messages(question, z):
    return [{"role": "system", "content": proposal_template()["system"]},
            {"role": "user", "content": canonical_json({"question": question, "z": z}).decode()}]


def source_samples(path):
    verification = verify(path)
    if not verification["passed"] or not verification["finished"]:
        raise ValueError("Collection verification must pass and be finished")
    with read_database(path) as database:
        run = dict(database.execute("SELECT * FROM run").fetchone())
        rows = [dict(r) for r in database.execute("SELECT * FROM records ORDER BY rowid")]
        links = defaultdict(dict)
        for row in database.execute("SELECT * FROM links"):
            links[row["child_id"]][row["relation"]] = row["parent_id"]

    def blob(key):
        raw = (path / "blobs" / key).read_bytes()
        if hashlib.sha256(raw).hexdigest() != key:
            raise ValueError("Blob checksum mismatch")
        return json.loads(raw)

    archive = blob(run["manifest_blob"])
    if archive["contract_hash"] != contract_hash():
        raise ValueError("Collection graph contract differs from the current policy contract")
    records = {r["id"]: {**r, "payload": blob(r["payload_blob"])} for r in rows}
    kinds = defaultdict(dict)
    for row in records.values():
        kinds[row["kind"]][row["logical_key"]] = row
    samples, excluded = [], []
    for cid, label_row in sorted(kinds["label"].items()):
        label = label_row["payload"]
        if label["split"] != "train" or label["status"] != "complete":
            excluded.append({"candidate_id": cid, "split": label["split"], "status": label["status"]})
            continue
        parents = links[label_row["id"]]
        candidate = records[cid]["payload"]
        task = records[parents["task"]]["payload"]
        trajectory_row = records[parents["trajectory"]]
        trajectory = trajectory_row["payload"]
        graph = records[parents["graph"]]
        proposal_id = links[trajectory_row["id"]]["proposal"]
        if (parents["candidate"] != cid or task["split"] != "train" or candidate["split"] != "train"
                or task["task_id"] != label["task_id"] or task["task_id"] != candidate["task_id"]
                or links[proposal_id]["candidate"] != cid or graph["payload"]["candidate_id"] != cid
                or links[graph["id"]]["trajectory"] != trajectory_row["id"]
                or label["graph_fingerprint"] != graph["payload"]["fingerprint"]
                or not set(label["complete_execution_ids"]).issubset(set(parents.values()))):
            raise ValueError("SFT source associations mismatch")
        if records[proposal_id]["payload"]["messages"][0]["content"] != proposal_template()["system"]:
            raise ValueError("Collection proposal rules differ from the current policy rules")
        parsed = replay(trajectory["raw_content"], reasoning=trajectory["raw_reasoning"])
        if not parsed.valid or not trajectory["eligible_for_execution"]:
            raise ValueError("SFT requires a complete legal trajectory")
        samples.append({"candidate_id": cid, "task_id": task["task_id"], "split": "train",
            "question": task["question"], "z": label["mean_outcome"],
            "raw_reasoning": trajectory["raw_reasoning"], "raw_content": trajectory["raw_content"],
            "source": {"run_id": run["id"], "label_id": label_row["id"], "proposal_id": proposal_id,
                "trajectory_id": trajectory_row["id"], "graph_id": graph["id"],
                "graph_fingerprint": label["graph_fingerprint"], "executor_hash": label["executor_hash"],
                "complete_execution_ids": label["complete_execution_ids"],
                "record_hashes": {key: records[key]["payload_blob"] for key in
                    [cid, parents["task"], label_row["id"], trajectory_row["id"], graph["id"], proposal_id]}}})
    if not samples:
        raise ValueError("No complete training samples")
    return samples, {"path": str(path.resolve()), "verification": verification,
        "config_hash": run["config_hash"], "manifest_blob": run["manifest_blob"], "excluded_labels": excluded}


def encode_sample(sample, tokenizer):
    reasoning, content = sample["raw_reasoning"], sample["raw_content"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Raw SFT requires native reasoning")
    if any(marker in reasoning or marker in content for marker in [*tokenizer.all_special_tokens, "<think>", "</think>"]):
        raise ValueError("Raw trajectory contains reserved template tokens")
    messages = policy_messages(sample["question"], sample["z"])
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=True)
    complete = messages + [{"role": "assistant", "reasoning_content": reasoning, "content": content}]
    rendered = tokenizer.apply_chat_template(complete, tokenize=False, enable_thinking=True)
    ids = tokenizer.apply_chat_template(complete, tokenize=True, return_dict=False, enable_thinking=True)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != ids or ids[:len(prefix)] != prefix:
        raise ValueError("Template token prefix mismatch")
    target = tokenizer.decode(ids[len(prefix):], skip_special_tokens=False)
    expected = reasoning.strip() + "\n</think>\n\n" + content.strip() + "<|im_end|>\n"
    if target != expected:
        raise ValueError("Native thinking serialization mismatch")
    end = len(ids) - 1 - ids[::-1].index(tokenizer.eos_token_id)
    content_start = len(rendered) - len(content.strip() + "<|im_end|>\n")
    content_token = next(i for i, (_, stop) in enumerate(encoded["offset_mapping"]) if stop > content_start)
    labels = [-100] * len(prefix) + ids[len(prefix):end + 1] + [-100] * (len(ids) - end - 1)
    groups = [0 if label == -100 else 1 if i < content_token else 2 for i, label in enumerate(labels)]
    if not 0 < len(prefix) < content_token <= end or labels[end] != tokenizer.eos_token_id:
        raise ValueError("Invalid supervision boundaries")
    return {**sample, "input_ids": ids, "labels": labels, "loss_groups": groups,
        "boundaries": {"prefix_end": len(prefix), "content_start": content_token, "assistant_end": end},
        "length": len(ids), "supervised_tokens": sum(g > 0 for g in groups),
        "reasoning_tokens": groups.count(1), "content_tokens": groups.count(2)}


def prepare_sft(config, source, output):
    if output.exists():
        raise ValueError("SFT output already exists; use a new output directory")
    samples, source_info = source_samples(source)
    model_path = Path(config.policy["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    encoded = [encode_sample(item, tokenizer) for item in samples]
    limit = config.training["max_sequence_length"]
    overlength = [{"candidate_id": s["candidate_id"], "length": s["length"]} for s in encoded if s["length"] > limit]
    lengths = sorted(s["length"] for s in encoded)
    manifest = {"version": SFT_VERSION, "source": source_info, "contract_hash": contract_hash(),
        "model": model_identity(model_path), "policy": config.policy, "training": config.training, "seed": config.seed,
        "versions": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "torch")},
        "sample_count": len(encoded), "task_count": len({s["task_id"] for s in encoded}),
        "label_distribution": dict(Counter(str(s["z"]) for s in encoded)),
        "lengths": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]},
        "max_sequence_length": limit, "overlength": overlength, "ready": not overlength,
        "loss_groups": {"0": "ignored prefix, padding, or post-message separator",
            "1": "native reasoning and closing thinking delimiter", "2": "content explanations, actions and message end"},
        "samples_hash": digest(encoded)}
    manifest["hash"] = digest(manifest)
    output.mkdir(parents=True)
    write_json(output / "samples.json", encoded)
    write_json(output / "manifest.json", manifest)
    write_json(output / "length-report.json", {"ready": not overlength, "limit": limit,
        "summary": manifest["lengths"], "overlength": overlength,
        "samples": [{k: s[k] for k in ("candidate_id", "length", "supervised_tokens", "reasoning_tokens", "content_tokens")} for s in encoded]})
    return {"output": str(output), "ready": not overlength, "sample_count": len(encoded),
        "lengths": manifest["lengths"], "overlength": overlength}, 0 if not overlength else 2


def load_prepared(path):
    manifest = json.loads((path / "manifest.json").read_text())
    samples = json.loads((path / "samples.json").read_text())
    if manifest["hash"] != digest({k: v for k, v in manifest.items() if k != "hash"}):
        raise ValueError("SFT manifest hash mismatch")
    if manifest["version"] != SFT_VERSION or digest(samples) != manifest["samples_hash"]:
        raise ValueError("SFT data hash or version mismatch")
    if not manifest["ready"] or any(s["length"] > manifest["max_sequence_length"] for s in samples):
        raise ValueError("Overlength samples require an explicit user decision")
    if manifest["contract_hash"] != contract_hash():
        raise ValueError("SFT contract mismatch")
    return manifest, samples
