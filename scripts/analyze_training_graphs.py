import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ocop.config import canonical_json


def read_records(path):
    result = defaultdict(dict)
    with (path / "records.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["table"] != "records":
                continue
            payload = (path / "blobs" / row["payload_blob"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == row["payload_blob"]
            result[row["kind"]][row["id"]] = json.loads(payload)
    return result


def summary(rows):
    return {
        "count": len(rows),
        "tasks": len({r["task_id"] for r in rows}),
        "labels": dict(sorted(Counter(str(r["z"]) for r in rows).items())),
        "graphs": dict(Counter(r["graph"] for r in rows)),
        "roles": dict(Counter(r["roles"] for r in rows)),
        "edge_counts": dict(Counter(r["edge_count"] for r in rows)),
        "successes": sum(r["successes"] for r in rows),
        "complete_executions": sum(r["complete_count"] for r in rows),
        "label_statuses": dict(Counter(r["status"] for r in rows)),
        "successes_in_complete_labels": sum(r["successes"] for r in rows if r["status"] == "complete"),
        "executions_in_complete_labels": sum(r["complete_count"] for r in rows if r["status"] == "complete"),
        "topologies": dict(Counter(r["topology"] for r in rows)),
        "chain_backbone_count": sum(r["has_chain_backbone"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, default=Path("artifacts/runs/gsm8k-20260920"))
    parser.add_argument("--evaluation", type=Path, default=Path("artifacts/runs/gsm8k-20260920-eval"))
    parser.add_argument("--samples", type=Path, default=Path("artifacts/sft/gsm8k-20260920-raw/samples.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gsm8k-20260921-training-graph-analysis.json"))
    args = parser.parse_args()
    collected = read_records(args.collection)
    evaluated = read_records(args.evaluation)
    all_graphs = list(collected["graph"].values()) + list(evaluated["graph"].values())
    fingerprints = sorted({g["fingerprint"] for g in all_graphs})
    graph_names = {fp: f"G{index + 1}" for index, fp in enumerate(fingerprints)}
    graph_catalog = {}
    for graph in all_graphs:
        graph_catalog[graph_names[graph["fingerprint"]]] = {
            "fingerprint": graph["fingerprint"], "workers": graph["workers"], "edges": graph["edges"]}

    def rows(records):
        labels = {v["candidate_id"]: v for v in records["label"].values()}
        graphs = {v["candidate_id"]: v for v in records["graph"].values()}
        output = []
        for cid, candidate in records["candidate"].items():
            if cid not in graphs:
                continue
            graph, label = graphs[cid], labels[cid]
            output.append({
                "candidate_id": cid, "task_id": candidate["task_id"], "split": candidate["split"],
                "slot": candidate["slot"], "condition": candidate.get("z"),
                "model": candidate.get("model", "strong_llm"), "z": label["mean_outcome"],
                "status": label["status"], "successes": label["success_count"],
                "complete_count": label["complete_count"], "graph": graph_names[graph["fingerprint"]],
                "roles": "/".join(w["role"] for w in graph["workers"]), "edge_count": len(graph["edges"]),
                "topology": ",".join(f"{a[-1]}>{b[-1]}" for a, b in sorted(graph["edges"])),
                "has_chain_backbone": all([f"worker_{i}", f"worker_{i+1}"] in graph["edges"] for i in range(3)),
            })
        return output

    collection_rows, evaluation_rows = rows(collected), rows(evaluated)
    samples = json.loads(args.samples.read_text())
    sample_manifest = json.loads(args.samples.with_name("manifest.json").read_text())
    samples_hash = hashlib.sha256(canonical_json(samples)).hexdigest()
    manifest_hash = hashlib.sha256(canonical_json({k: v for k, v in sample_manifest.items() if k != "hash"})).hexdigest()
    assert samples_hash == sample_manifest["samples_hash"]
    assert manifest_hash == sample_manifest["hash"]
    sample_ids = {s["candidate_id"] for s in samples}
    training_rows = [r for r in collection_rows if r["candidate_id"] in sample_ids]
    assert len(training_rows) == len(samples)
    source_labels = {r["candidate_id"]: r["z"] for r in training_rows}
    assert all(source_labels[s["candidate_id"]] == s["z"] for s in samples)
    task_rows = {task: [r for r in training_rows if r["task_id"] == task]
                 for task in sorted({r["task_id"] for r in training_rows})}
    strong_eval = [r for r in collection_rows if r["split"] == "eval"]
    checkpoint_eval = [r for r in evaluation_rows if r["split"] == "eval"]
    comparisons = {}
    for task in sorted({r["task_id"] for r in strong_eval}):
        strong = [r for r in strong_eval if r["task_id"] == task]
        policy = [r for r in checkpoint_eval if r["task_id"] == task]
        comparisons[task] = {"strong_llm": strong, "checkpoint": policy,
                             "shared_graphs": sorted({r["graph"] for r in strong} & {r["graph"] for r in policy})}
    template = next(iter(collected["proposal_template"].values()))
    assert hashlib.sha256(canonical_json({k: v for k, v in template.items() if k != "hash"})).hexdigest() == template["hash"]
    proposals_by_task = defaultdict(list)
    for proposal in collected["proposal"].values():
        messages = proposal["messages"]
        assert len(messages) == 2 and messages[0] == {"role": "system", "content": template["system"]}
        assert messages[1]["role"] == "user"
        proposals_by_task[messages[1]["content"]].append(messages)
    request_bodies = []
    with (args.collection / "records.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["table"] != "requests" or not row["logical_key"].startswith("proposal:"):
                continue
            payload = (args.collection / "blobs" / row["spec_blob"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == row["spec_blob"]
            request_bodies.append(json.loads(payload)["body"])
    settings = [{k: v for k, v in body.items() if k != "messages"} for body in request_bodies]
    assert all(setting == settings[0] for setting in settings)
    report = {
        "integrity": {"samples_hash_verified": samples_hash, "manifest_hash_verified": manifest_hash,
                      "proposal_template_hash_verified": template["hash"], "record_blobs_verified": True},
        "teacher_proposal": {
            "template": template, "request_count": len(request_bodies), "request_settings": settings[0],
            "temperature_sent": any("temperature" in body for body in request_bodies),
            "seed_sent": any("seed" in body for body in request_bodies),
            "distinct_questions": len(proposals_by_task),
            "proposal_counts_per_question": dict(Counter(len(items) for items in proposals_by_task.values())),
            "same_messages_within_each_question": all(all(m == items[0] for m in items) for items in proposals_by_task.values()),
        },
        "sources": {"collection": str(args.collection), "evaluation": str(args.evaluation), "samples": str(args.samples)},
        "graphs": graph_catalog, "training": summary(training_rows),
        "training_by_z": {str(z): summary([r for r in training_rows if r["z"] == z])
                          for z in sorted({r["z"] for r in training_rows})},
        "training_by_graph": {g: summary([r for r in training_rows if r["graph"] == g])
                              for g in sorted({r["graph"] for r in training_rows})},
        "training_tasks": task_rows,
        "training_tasks_with_multiple_labels": {t: rs for t, rs in task_rows.items() if len({r["z"] for r in rs}) > 1},
        "training_tasks_with_any_low_label": {t: rs for t, rs in task_rows.items() if any(r["z"] < 1 for r in rs)},
        "strong_eval": summary(strong_eval), "checkpoint_eval": summary(checkpoint_eval),
        "eval_task_comparisons": comparisons,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "training": report["training"],
                      "strong_eval": report["strong_eval"], "checkpoint_eval": report["checkpoint_eval"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
