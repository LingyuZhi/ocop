import asyncio
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from ocop.collection import graph_fingerprint, write_json
from ocop.config import RuntimeConfig, canonical_json
from ocop.diagnostics import provenance
from ocop.executor import execute_repeat, executor_hash, finalizer_messages, worker_messages
from ocop.graph import replay
from ocop.labels import aggregate_label
from ocop.llm import RequestRunner, load_credentials, recover_inflight_budget
from ocop.scoring import normalize_reference, score_answer
from ocop.storage import ReadStore, RunHalted, RunStore
from ocop.trajectories import digest
from ocop.usage import summarize_usage


def coverage(view):
    candidates = {r["id"]: r["payload"] for r in view.record_items("candidate")}
    labels = {r["key"]: r["payload"] for r in view.record_items("label")}
    groups = {}
    for row in view.record_items("graph"):
        graph = row["payload"]
        cid = graph["candidate_id"]
        task = candidates[cid]
        key = (task["task_id"], graph["fingerprint"])
        item = groups.setdefault(key, {"task_id": key[0], "fingerprint": key[1], "split": task["split"],
            "candidate_ids": [], "complete_candidate_ids": [], "success_count": 0, "complete_count": 0,
            "graph": {k: graph[k] for k in ("workers", "edges")}})
        item["candidate_ids"].append(cid)
        label = labels.get(cid)
        if label and label["status"] == "complete":
            item["complete_candidate_ids"].append(cid)
            item["success_count"] += label["success_count"]
            item["complete_count"] += label["complete_count"]
    tasks = []
    for task in view.get_record("task_manifest", "tasks")["tasks"]:
        if task["split"] == "holdout":
            continue
        graphs = sorted([g for (tid, _), g in groups.items() if tid == task["task_id"]], key=lambda g: g["fingerprint"])
        rates = [Fraction(g["success_count"], g["complete_count"]) for g in graphs if g["complete_count"]]
        task_labels = [labels[cid]["mean_outcome"] for cid, c in candidates.items()
                       if c["task_id"] == task["task_id"] and cid in labels and labels[cid]["status"] == "complete"]
        tasks.append({"task_id": task["task_id"], "unique_graphs": len(graphs), "graphs": graphs,
                      "label_distribution": dict(Counter(map(str, task_labels))),
                      "pooled_graph_rate_gap": float(max(rates) - min(rates)) if len(rates) > 1 else None})
    return {"tasks": tasks, "unique_graph_count": len(groups),
        "tasks_with_multiple_graphs": sum(t["unique_graphs"] > 1 for t in tasks),
        "tasks_with_positive_graph_gap": sum((t["pooled_graph_rate_gap"] or 0) > 0 for t in tasks),
        "edge_count_distribution": dict(Counter(str(len(g["graph"]["edges"])) for g in groups.values())),
        "role_assignment_distribution": dict(Counter(
            canonical_json(g["graph"]["workers"]).decode() for g in groups.values()))}


def select_pairs(view, seed, limit):
    rng = random.Random(seed)
    eligible = []
    for task in coverage(view)["tasks"]:
        graphs = [g for g in task["graphs"] if g["split"] == "train" and g["complete_count"]]
        if len(graphs) < 2:
            continue
        rng.shuffle(graphs)
        graphs.sort(key=lambda g: Fraction(g["success_count"], g["complete_count"]))
        low, high = graphs[0], graphs[-1]
        gap = Fraction(high["success_count"], high["complete_count"]) - Fraction(low["success_count"], low["complete_count"])
        if gap > 0:
            eligible.append({"task_id": task["task_id"], "high": high, "low": low, "gap": gap})
    rng.shuffle(eligible)
    eligible.sort(key=lambda p: p["gap"], reverse=True)
    return [{**p, "gap": float(p["gap"])} for p in eligible[:limit]]


def paired_statistics(left, right, seed):
    blocks = sorted(left.keys() & right.keys())
    a = np.asarray([left[b] for b in blocks], dtype=int)
    b = np.asarray([right[k] for k in blocks], dtype=int)
    n = len(blocks)
    plus, minus = int(np.sum(a > b)), int(np.sum(a < b))
    if n:
        diff = a - b
        rng = np.random.default_rng(seed)
        samples = diff[rng.integers(0, n, size=(10000, n))].mean(axis=1)
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
        p = float(binomtest(plus, plus + minus, 0.5).pvalue) if plus + minus else 1.0
    else:
        interval, p = None, None
    return {"paired_blocks": blocks, "paired_count": n, "left_only_blocks": sorted(left.keys() - right.keys()),
        "right_only_blocks": sorted(right.keys() - left.keys()), "left_successes": int(a.sum()),
        "right_successes": int(b.sum()), "difference": float((a - b).mean()) if n else None,
        "bootstrap_percentile_95_ci": interval, "bootstrap_resamples": 10000,
        "discordant_left_wins": plus, "discordant_right_wins": minus, "mcnemar_exact_p": p,
        "interval_note": "A degenerate empirical bootstrap interval does not establish population certainty."}


def holm_adjust(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["mcnemar_exact_p"] if rows[i]["mcnemar_exact_p"] is not None else 1)
    previous = 0.0
    for rank, i in enumerate(order):
        p = rows[i]["mcnemar_exact_p"]
        adjusted = min(1.0, max(previous, (len(rows) - rank) * (p if p is not None else 1)))
        rows[i]["holm_adjusted_p"] = adjusted if p is not None else None
        rows[i]["holm_reject_0_05"] = p is not None and adjusted <= 0.05
        previous = adjusted


def prepare_snapshot(runtime, tasks, pairs, *, seed, required, maximum, kind, source):
    if not 0 < required <= maximum:
        raise ValueError("Invalid replication budget")
    jobs = []
    for pair in pairs:
        for arm in ("left", "right"):
            parsed = replay(canonical_json(pair[arm]["trajectory"]).decode())
            if not parsed.valid:
                raise ValueError("Invalid replication graph")
            jobs.append({"key": f"{pair['task_id']}:{arm}", "task": tasks[pair["task_id"]], "arm": arm,
                "trajectory": pair[arm]["trajectory"], "fingerprint": graph_fingerprint(parsed.final_graph),
                "source": pair[arm].get("source")})
    if len({j["key"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate replication task")
    return {"purpose": "graph_replication", "version": "ocop.replication.v1", "kind": kind,
        "runtime": runtime.model_dump(mode="json"), "seed": seed, "required": required, "maximum": maximum,
        "candidate_concurrency": 2, "executor_hash": executor_hash(runtime), "source": source, "jobs": jobs,
        "pairs": [{k: v for k, v in p.items() if k not in {"left", "right"}} for p in pairs]}


def install(store, snapshot):
    jobs = {}
    source = store.put_record("source", "source", snapshot["source"])
    for item in snapshot["jobs"]:
        task = item["task"]
        tid = store.put_record("task", task["task_id"], task, parents={"source": source})
        parsed = replay(canonical_json(item["trajectory"]).decode())
        trajectory = store.put_record("fixed_trajectory", item["key"], asdict(parsed), parents={"source": source})
        graph = store.put_record("graph", item["key"], {**asdict(parsed.final_graph), "fingerprint": item["fingerprint"]},
                                 parents={"trajectory": trajectory})
        cid = store.put_record("candidate", item["key"], item, parents={"task": tid, "graph": graph})
        jobs[item["key"]] = (item, parsed, {"task": tid, "candidate": cid, "trajectory": trajectory, "graph": graph})
    return jobs


def repeat_groups(store):
    groups = defaultdict(list)
    for row in store.record_items("replication_repeat"):
        groups[row["payload"]["job_key"]].append(row["payload"])
    for items in groups.values():
        items.sort(key=lambda i: i["block"])
    return groups


def block_plan(snapshot, block, groups):
    active = [job["key"] for job in snapshot["jobs"]
              if sum(r["status"] == "completed" for r in groups.get(job["key"], [])) < snapshot["required"]]
    random.Random(snapshot["seed"] + block).shuffle(active)
    return {"block": block, "jobs": active}


async def execute_block(store, runner, runtime, snapshot, block, plan, jobs):
    queue = asyncio.Queue()
    for key in plan["jobs"]:
        queue.put_nowait(key)

    async def worker():
        while not queue.empty():
            key = queue.get_nowait()
            try:
                repeat_key = f"{key}:{block}"
                if store.get_record("replication_repeat", repeat_key) is not None:
                    continue
                store.ensure_active()
                item, parsed, parents = jobs[key]
                result = await execute_repeat(question=item["task"]["question"], graph=parsed, parents=parents,
                    repeat_id=str(block), config=runtime, runner=runner)
                score = score_answer(result["answer"], normalize_reference(item["task"]["reference_answer"])) if result["status"] == "completed" else None
                if score is not None:
                    store.put_record("score", result["execution_id"], score, parents={"execution": result["execution_id"]})
                store.put_record("replication_repeat", repeat_key, {"job_key": key, "block": block,
                    "candidate_id": parents["candidate"], "repeat_id": str(block), "execution_id": result["execution_id"],
                    "executor_hash": result["executor_hash"], "status": result["status"], "score": score},
                    parents={"candidate": parents["candidate"], "execution": result["execution_id"]})
            except RunHalted:
                return
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(min(snapshot["candidate_concurrency"], queue.qsize()))]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


def replication_report(path):
    view = ReadStore(path)
    snapshot = view.archive["config"]
    if snapshot["purpose"] != "graph_replication":
        raise ValueError("Expected graph replication")
    runtime = RuntimeConfig.model_validate(snapshot["runtime"])
    run = view.rows("run")[0]
    groups = repeat_groups(view)
    candidates = {r["key"]: r for r in view.record_items("candidate")}
    executions = {r["id"]: r for r in view.record_items("execution")}
    checks = {"config_hash": digest({k: v for k, v in view.archive.items() if k != "provenance"}) == run["config_hash"],
        "executor": executor_hash(runtime) == snapshot["executor_hash"],
        "candidate_plan": set(candidates) == {j["key"] for j in snapshot["jobs"]},
        "scores": True, "executions": True, "labels": True, "blocks": True, "budgets": True,
        "known_repeats": set(groups) <= set(candidates), "candidate_content": True, "node_messages": True,
        "request_ownership": True, "terminal_ids": True}
    known_cids = {r["id"] for r in candidates.values()}
    checks["terminal_ids"] = all(r["key"] in known_cids for kind in ("label", "candidate_result") for r in view.record_items(kind))
    nodes = {r["id"]: r for r in view.record_items("execution_node")}
    node_groups = defaultdict(list)
    for node in nodes.values():
        node_groups[view.links[node["id"]]["execution"]].append(node)
    for request in view.rows("requests"):
        checks["request_ownership"] &= request["owner_id"] in nodes
    details = []
    history = defaultdict(list)
    blocks = sorted(view.record_items("block"), key=lambda r: r["payload"]["block"])
    checks["blocks"] &= [r["payload"]["block"] for r in blocks] == list(range(len(blocks)))
    for row in blocks:
        b = row["payload"]["block"]
        expected = block_plan(snapshot, b, history)
        checks["blocks"] &= row["payload"] == expected and b < snapshot["maximum"]
        if b:
            checks["blocks"] &= view.get_record("block_done", str(b - 1)) is not None
        for key, repeats in groups.items():
            current = [r for r in repeats if r["block"] == b]
            checks["blocks"] &= len(current) <= 1 and (not current or key in expected["jobs"])
            history[key].extend(current)
        if view.get_record("block_done", str(b)):
            checks["blocks"] &= all(any(r["block"] == b for r in groups.get(key, [])) for key in expected["jobs"])
    checks["blocks"] &= all(r["block"] < len(blocks) for rs in groups.values() for r in rs)
    for job in snapshot["jobs"]:
        candidate = candidates.get(job["key"])
        if not candidate:
            continue
        cid = candidate["id"]
        parsed = replay(canonical_json(job["trajectory"]).decode())
        checks["candidate_content"] &= candidate["payload"] == job and graph_fingerprint(parsed.final_graph) == job["fingerprint"]
        checks["candidate_content"] &= view.get_record("task", job["task"]["task_id"]) == job["task"]
        repeats = groups.get(job["key"], [])
        for item in repeats:
            result = view.get_record("execution_result", item["execution_id"])
            execution = executions[item["execution_id"]]["payload"]
            checks["executions"] &= (item["candidate_id"] == cid and view.links[item["execution_id"]]["candidate"] == cid
                and item["repeat_id"] == str(item["block"]) == result["repeat_id"] == execution["repeat_id"]
                and item["executor_hash"] == result["executor_hash"] == execution["executor_hash"] == snapshot["executor_hash"]
                and execution["question"] == job["task"]["question"]
                and digest(execution["graph"]) == digest(asdict(parsed.final_graph)))
            roles = {w.worker_id: w.role for w in parsed.final_graph.workers}
            for node in node_groups[item["execution_id"]]:
                name = node["payload"]["node"]
                outcomes = result["nodes"]
                if name == "finalizer":
                    expected = finalizer_messages(job["task"]["question"], {w: outcomes[w]["content"] for w in roles})
                else:
                    expected = worker_messages(job["task"]["question"], roles[name], {
                        a: outcomes[a]["content"] for a, b in parsed.final_graph.edges if b == name})
                checks["node_messages"] &= expected == node["payload"]["messages"]
                checks["node_messages"] &= view.get_record("execution_node_result", node["id"]) == outcomes[name]
                model = runtime.finalizer if name == "finalizer" else runtime.worker_model
                for request in view.requests_for_owner(node["id"]):
                    spec = json.loads(view.read_blob(request["spec_blob"]))
                    checks["node_messages"] &= spec["body"] == model.completion_body(expected)
            score = score_answer(result["answer"], normalize_reference(job["task"]["reference_answer"])) if result["status"] == "completed" else None
            checks["scores"] &= item["score"] == score and item["status"] == result["status"]
            if score is not None:
                checks["scores"] &= view.get_record("score", item["execution_id"]) == score
        label = aggregate_label(repeats, required=snapshot["required"], max_repeats=snapshot["maximum"])
        saved = view.get_record("label", cid)
        terminal = view.get_record("candidate_result", cid)
        checks["labels"] &= saved is None or saved == label
        checks["labels"] &= terminal is None or saved == label and terminal == {"status": label["status"]} and label["status"] != "pending"
        checks["budgets"] &= len(repeats) <= snapshot["maximum"]
        details.append({"key": job["key"], "task_id": job["task"]["task_id"], "arm": job["arm"],
                        "fingerprint": job["fingerprint"], "label": label, "terminal": terminal is not None, "repeats": repeats})
    checks["budgets"] &= all(len(view.attempts_for(r["id"])) <= runtime.requests.max_attempts for r in view.rows("requests"))
    comparisons = []
    for pair in snapshot["pairs"]:
        tid = pair["task_id"]
        arms = [{r["block"]: r["score"]["success"] for r in groups.get(f"{tid}:{arm}", []) if r["status"] == "completed"}
                for arm in ("left", "right")]
        comparisons.append({**pair, **paired_statistics(*arms, snapshot["seed"]),
            "independent_left_rate": sum(arms[0].values()) / len(arms[0]) if arms[0] else None,
            "independent_right_rate": sum(arms[1].values()) / len(arms[1]) if arms[1] else None})
    if snapshot["kind"] == "selected_training":
        holm_adjust(comparisons)
    statuses = Counter(r["status"] for rs in groups.values() for r in rs)
    return {"run_id": run["id"], "kind": snapshot["kind"], "executor_hash": snapshot["executor_hash"],
        "finished": len(details) == len(snapshot["jobs"]) and all(r["terminal"] for r in details) and run["halt_reason"] is None,
        "integrity_passed": all(checks.values()), "checks": checks, "halt_reason": run["halt_reason"],
        "planned": len(snapshot["jobs"]), "terminal": sum(r["terminal"] for r in details),
        "repeat_statuses": dict(statuses), "request_count": len(view.rows("requests")),
        "attempt_count": len(view.rows("attempts")), "usage": summarize_usage(view, view.rows("requests")),
        "candidates": details, "comparisons": comparisons}


async def run_replication(runtime, snapshot, run_id, credentials_path, *, shared_limits=None,
                          prepare_only=False, after_topup=False, transport=None, on_report=None):
    credentials = {} if prepare_only else load_credentials(credentials_path, {runtime.worker_model.api_key_env, runtime.finalizer.api_key_env})
    with RunStore(Path(runtime.artifacts_dir) / "runs", snapshot, run_id=run_id, provenance=provenance()) as store:
        write_json(store.path / "config.json", snapshot)
        jobs = install(store, snapshot)
        if after_topup:
            recover_inflight_budget(store, after_topup=True)
        try:
            if not replication_report(store.path)["integrity_passed"]:
                raise ValueError("Replication integrity checks failed")
            if not prepare_only:
                store.ensure_active()
                async with RequestRunner(store, runtime.requests, credentials, transport=transport, shared_limits=shared_limits) as runner:
                    for block in range(snapshot["maximum"]):
                        if store.get_record("block_done", str(block)):
                            continue
                        plan = store.get_record("block", str(block))
                        if plan is None:
                            plan = block_plan(snapshot, block, repeat_groups(store))
                            if not plan["jobs"]:
                                break
                            store.put_record("block", str(block), plan)
                        await execute_block(store, runner, runtime, snapshot, block, plan, jobs)
                        if not all(store.get_record("replication_repeat", f"{key}:{block}") for key in plan["jobs"]):
                            store.ensure_active()
                            raise ValueError("Block did not finish")
                        store.put_record("block_done", str(block), {"block": block})
                        state = replication_report(store.path)
                        write_json(store.path / "replication-report.json", state)
                        if on_report:
                            on_report(state)
                        print(f"replication {run_id}: block {block + 1}, repeats={sum(state['repeat_statuses'].values())}", flush=True)
                        store.ensure_active()
                groups = repeat_groups(store)
                for key, (_, _, parents) in jobs.items():
                    label = aggregate_label(groups.get(key, []), required=snapshot["required"], max_repeats=snapshot["maximum"])
                    if label["status"] == "pending":
                        raise ValueError("Unfinished replication budget")
                    cid = parents["candidate"]
                    store.put_record("label", cid, label, parents={"candidate": cid})
                    store.put_record("candidate_result", cid, {"status": label["status"]}, parents={"candidate": cid})
        except RunHalted:
            pass
        finally:
            report = replication_report(store.path)
            write_json(store.path / "replication-report.json", report)
            temporary = store.path / "records.jsonl.tmp"
            store.export_jsonl(temporary)
            temporary.replace(store.path / "records.jsonl")
            if on_report:
                on_report(report)
    return report, 0 if report["integrity_passed"] and not report["halt_reason"] and (prepare_only or report["finished"]) else 1
