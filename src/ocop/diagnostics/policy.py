import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ocop.runtime.config import canonical_json
from ocop.execution.executor import finalizer_messages, worker_messages
from ocop.execution.scoring import normalize_reference, score_answer
from ocop.runtime.storage import read_database


def inspect_run(path):
    def blob(name):
        raw = (path / "blobs" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != name:
            raise ValueError(f"Blob checksum mismatch: {name}")
        return json.loads(raw)

    with read_database(path) as db:
        tables = {name: [dict(row) for row in db.execute(f"SELECT * FROM {name} ORDER BY rowid")]
                  for name in ("run", "records", "links", "requests", "attempts")}
    records = {row["id"]: {**row, "payload": blob(row["payload_blob"])} for row in tables["records"]}
    by_kind = defaultdict(dict)
    links = defaultdict(dict)
    for row in records.values():
        by_kind[row["kind"]][row["logical_key"]] = row
    for row in tables["links"]:
        links[row["child_id"]][row["relation"]] = row["parent_id"]
    checks = Counter()
    executions = []
    for row in by_kind["execution_result"].values():
        result = row["payload"]
        eid = result["execution_id"]
        execution = records[eid]["payload"]
        task = records[links[eid]["task"]]["payload"]
        candidate = records[links[eid]["candidate"]]["payload"]
        graph = execution["graph"]
        roles = {worker["worker_id"]: worker["role"] for worker in graph["workers"]}
        nodes = result["nodes"]
        for node, outcome in nodes.items():
            saved = by_kind["execution_node"][f"{eid}:{node}"]["payload"]
            if node == "finalizer":
                expected = finalizer_messages(task["question"], {key: nodes[key]["content"] for key in roles})
            else:
                expected = worker_messages(task["question"], roles[node], {
                    source: nodes[source]["content"] for source, target in graph["edges"] if target == node})
            if saved["messages"] != expected:
                raise ValueError(f"Message mismatch: {eid}:{node}")
            checks["node_messages_reconstructed"] += 1
        score = None
        if result["status"] == "completed":
            score = score_answer(result["answer"], normalize_reference(task["reference_answer"]))
            if by_kind["score"][eid]["payload"] != score:
                raise ValueError(f"Score mismatch: {eid}")
            checks["scores_recomputed"] += 1
        executions.append({"run": path.name, "execution_id": eid, "blob": row["payload_blob"],
            "task_id": task["task_id"], "question": task["question"], "reference": task["reference_answer"],
            "candidate": candidate, "repeat_id": result["repeat_id"], "status": result["status"],
            "graph": graph, "executor_hash": result["executor_hash"], "score": score, "nodes": nodes})
    request_parameters = Counter()
    root_requests = defaultdict(list)
    for request in tables["requests"]:
        body = blob(request["spec_blob"])["body"]
        request_parameters[canonical_json({k: v for k, v in body.items() if k != "messages"}).decode()] += 1
        owner = records[request["owner_id"]]
        if owner["payload"].get("node") == "worker_0":
            eid = links[owner["id"]]["execution"]
            task = records[links[eid]["task"]]["payload"]["task_id"]
            root_requests[task].append({"request_id": request["id"], "spec_blob": request["spec_blob"],
                "body_hash": hashlib.sha256(canonical_json(body)).hexdigest()})
    outcomes = [node for item in executions for node in item["nodes"].values() if node["status"] == "completed"]
    generation_inputs = defaultdict(list)
    for row in by_kind["generation"].values():
        candidate = records[links[row["id"]]["candidate"]]["payload"]
        if candidate["model"] == "last_checkpoint" and candidate["split"] in {"eval", "holdout"}:
            raw = row["payload"]
            generation_inputs[candidate["task_id"]].append({"z": candidate["z"], "seed": candidate["seed"],
                "input_ids": raw["input_ids"], "raw_blob": row["payload_blob"],
                "output_tokens": raw["output_tokens"], "reached_eos": raw["reached_eos"]})
    input_comparisons = {}
    for task, items in generation_inputs.items():
        items.sort(key=lambda x: x["z"])
        base = items[0]
        input_comparisons[task] = [{"z": item["z"], "seed": item["seed"], "raw_blob": item["raw_blob"],
            "input_length": len(item["input_ids"]), "output_tokens": item["output_tokens"],
            "reached_eos": item["reached_eos"], "length_matches_z0": len(item["input_ids"]) == len(base["input_ids"]),
            "differences_from_z0": [{"position": i, "z0_token": a, "token": b}
                for i, (a, b) in enumerate(zip(base["input_ids"], item["input_ids"])) if a != b]} for item in items]
    return {"run": path.name, "checks": dict(checks), "executions": executions,
        "request_parameters": [{"body_without_messages": json.loads(key), "requests": count}
                               for key, count in request_parameters.items()],
        "root_requests": dict(root_requests), "generation_input_comparisons": input_comparisons,
        "response_providers": dict(Counter(node.get("provider") for node in outcomes)),
        "response_models": dict(Counter(node.get("response_model") for node in outcomes)),
        "system_fingerprints": dict(Counter(node.get("system_fingerprint") for node in outcomes))}


def diagnose_policy(runs, output_path, tasks=()):
    reports = [inspect_run(path) for path in runs]
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'evidence.json').write_text(json.dumps(reports, indent=2, ensure_ascii=False) + '\n')
    audit = [{key: value for key, value in report.items() if key != 'executions'} for report in reports]
    (output_path / 'generation-config-audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    for task in tasks:
        for report in reports:
            lines = []
            excerpts = []
            for item in report['executions']:
                if item['task_id'] != task and item['task_id'].split('-')[-1] != task:
                    continue
                lines.extend([f"EXECUTION {item['execution_id']} REPEAT {item['repeat_id']}", f"CANDIDATE {item['candidate']}", f"QUESTION {item['question']}", f"REFERENCE {item['reference']}", f"GRAPH {item['graph']}", f"STATUS {item['status']} SCORE {item['score']}"])
                excerpts.extend([f"EXECUTION {item['execution_id']} REPEAT {item['repeat_id']}", f"CANDIDATE {item['candidate']}", f"STATUS {item['status']} SCORE {item['score']}"])
                for node, outcome in sorted(item['nodes'].items()):
                    lines.extend([f"NODE {node} STATUS {outcome['status']}", outcome.get('content', ''), ''])
                    content = outcome.get('content', '')
                    excerpts.extend([f'NODE {node}', content if node == 'worker_2' else content[-600:], ''])
            if lines:
                (output_path / f"{report['run']}-{task}.txt").write_text('\n'.join(lines))
                (output_path / f"{report['run']}-{task}-excerpts.txt").write_text('\n'.join(excerpts))
    return [{k: report[k] for k in ('run', 'checks')} for report in reports]
