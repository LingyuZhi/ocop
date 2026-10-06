import itertools
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from filelock import FileLock
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from ocop.collection.benchmark import BenchmarkConfig, validate_manifest
from ocop.graph import graph_fingerprint
from ocop.runtime.storage import write_json
from ocop.runtime.config import RuntimeConfig
from ocop.execution.executor import executor_hash
from ocop.graph import replay
from ocop.execution.labels import aggregate_label
from ocop.inference.transformers import parse_generation
from ocop.execution.scoring import normalize_reference, score_answer
from ocop.runtime.storage import export_run, read_database
from ocop.runtime.storage import digest, file_hash
from ocop.training.data import policy_messages
from ocop.runtime.usage import cost_budget_summary, summarize_usage


class EvaluationView:
    def __init__(self, path):
        self.path = path
        with read_database(path) as database:
            self.tables = {name: [dict(row) for row in database.execute(f"SELECT * FROM {name} ORDER BY rowid")]
                           for name in ("run", "records", "links", "requests", "attempts")}
        self.by_kind = defaultdict(dict)
        self.by_id = {}
        for row in self.tables["records"]:
            item = {"id": row["id"], "key": row["logical_key"], "blob": row["payload_blob"],
                    "payload": self.blob(row["payload_blob"])}
            self.by_kind[row["kind"]][row["logical_key"]] = item
            self.by_id[row["id"]] = item
        self.links = defaultdict(dict)
        for row in self.tables["links"]:
            self.links[row["child_id"]][row["relation"]] = row["parent_id"]
        self.archive = self.blob(self.tables["run"][0]["manifest_blob"])
        if self.archive["config"].get("purpose") != "evaluation":
            raise ValueError("Report requires an evaluation run")

    def blob(self, name):
        path = self.path / "blobs" / name
        if file_hash(path) != name:
            raise ValueError("Evaluation blob checksum mismatch")
        return json.loads(path.read_bytes())

    def record_items(self, kind):
        return list(self.by_kind[kind].values())

    def get_record(self, kind, key):
        row = self.by_kind[kind].get(key)
        return row["payload"] if row else None

    def attempts_for(self, request_id):
        return [row for row in self.tables["attempts"] if row["request_id"] == request_id]

    def rows(self, name):
        return self.tables[name]


def graph_comparison(left, right):
    result = {"left": left["candidate_id"], "right": right["candidate_id"], "task_id": left["task_id"],
              "split": left["split"], "comparable": bool(left["graph"] and right["graph"])}
    if result["comparable"]:
        a, b = left["graph"], right["graph"]
        roles_a = {w["worker_id"]: w["role"] for w in a["workers"]}
        roles_b = {w["worker_id"]: w["role"] for w in b["workers"]}
        edges_a, edges_b = set(map(tuple, a["edges"])), set(map(tuple, b["edges"]))
        result.update(same_graph=a["fingerprint"] == b["fingerprint"],
            role_changes=[{"worker_id": key, "left": roles_a[key], "right": roles_b[key]}
                          for key in sorted(roles_a) if roles_a[key] != roles_b[key]],
            added_edges=sorted(edges_b - edges_a), removed_edges=sorted(edges_a - edges_b))
    return result


def build_report(view, phase):
    from ocop.evaluation.pipeline import EvaluationConfig, candidate_plan

    snapshot = view.archive["config"]
    config = RuntimeConfig.model_validate(snapshot["runtime"])
    settings = EvaluationConfig.model_validate(snapshot["settings"])
    manifest = snapshot["task_manifest"]
    validate_manifest(manifest, BenchmarkConfig.model_validate(config.benchmark), config.seed)
    expected = candidate_plan(manifest, snapshot["training_smoke"], settings, config.seed)
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    candidates = view.record_items("candidate")
    run = view.tables["run"][0]
    checks = {"config_hash": digest({k: v for k, v in view.archive.items() if k != "provenance"}) == run["config_hash"],
        "candidate_plan": expected == snapshot["candidates"] == [row["payload"] for row in candidates],
        "executor": executor_hash(config) == snapshot["executor_hash"],
        "generation_records": True, "source_links": True, "legal_graphs": True,
        "scores_recomputed": True, "repeat_associations": True, "labels_recomputed": True,
        "terminal_results": True, "repeat_budgets": True,
        "request_budgets": all(len(view.attempts_for(r["id"])) <= config.requests.max_attempts for r in view.tables["requests"])}
    budget = cost_budget_summary(view, settings.cost_budget) if settings.cost_budget is not None else None
    if budget is not None:
        checks["cost_budget"] = budget["cap_respected"] and budget["all_attempts_reserved"]
    if settings.execution_task_limit is not None:
        checks["execution_selection"] = True
    candidate_ids = {row["id"] for row in candidates}
    for kind in ("generation", "generation_result", "trajectory", "graph", "label", "candidate_result"):
        if not set(view.by_kind[kind]).issubset(candidate_ids):
            raise ValueError("Record references an unknown evaluation candidate")
    repeats = defaultdict(list)
    for row in view.record_items("evaluation_repeat"):
        item = row["payload"]
        if item["candidate_id"] not in candidate_ids:
            raise ValueError("Repeat references an unknown candidate")
        repeats[item["candidate_id"]].append(item)
    details = []
    for row in candidates:
        cid, candidate = row["id"], row["payload"]
        task = tasks[candidate["task_id"]]
        parents = view.links[cid]
        checks["source_links"] &= (view.by_id[parents["task"]]["payload"] == task
            and view.by_id[parents["model"]]["payload"] == snapshot["models"][candidate["model"]]
            and task["split"] == ("train" if candidate["split"] == "train_smoke" else settings.task_split))
        raw = view.get_record("generation", cid)
        generated = view.get_record("generation_result", cid)
        graph = view.get_record("graph", cid)
        label = view.get_record("label", cid)
        terminal = view.get_record("candidate_result", cid)
        trajectory = view.get_record("trajectory", cid)
        if generated is not None:
            if raw is None or trajectory is None:
                raise ValueError("Derived generation lacks raw output or trajectory")
            parsed = parse_generation(raw)
            checks["generation_records"] &= (digest(parsed) == digest(trajectory)
                and all(generated[k] == parsed[k] for k in ("json_parsed", "valid_graph", "eligible_for_execution", "error"))
                and raw["candidate_id"] == cid and raw["output_tokens"] == len(raw["token_ids"])
                and raw["input_tokens"] == len(raw["input_ids"])
                and raw["reasoning_tokens"] + raw["content_tokens"] == raw["output_tokens"]
                and raw["reached_eos"] == (bool(raw["token_ids"]) and raw["token_ids"][-1] == raw["eos_token_id"])
                and len(raw["token_ids"]) <= snapshot["generation_config"]["max_new_tokens"])
            raw_row = view.by_kind["generation"][cid]
            attempt = view.by_id[raw["attempt_id"]]["payload"]
            checks["source_links"] &= (view.links[raw_row["id"]] == {"candidate": cid, "attempt": raw["attempt_id"]}
                and attempt["candidate_id"] == cid and attempt["seed"] == candidate["seed"])
        if graph:
            replayed = replay(trajectory["raw_content"], reasoning=trajectory["raw_reasoning"])
            checks["legal_graphs"] &= bool(generated and generated["eligible_for_execution"] and replayed.valid
                and graph["fingerprint"] == graph_fingerprint(replayed.final_graph)
                and graph["workers"] == [{"worker_id": w.worker_id, "role": w.role} for w in replayed.final_graph.workers]
                and graph["edges"] == [list(edge) for edge in replayed.final_graph.edges])
        elif generated and generated["eligible_for_execution"]:
            checks["legal_graphs"] = False
        items = sorted(repeats[cid], key=lambda item: int(item["repeat_id"]))
        selected = candidate.get("execution_selected", True)
        if settings.execution_task_limit is not None:
            checks["execution_selection"] &= selected or (not items and not label and not any(
                view.links[item["id"]].get("candidate") == cid for item in view.record_items("execution")))
        checks["repeat_budgets"] &= len(items) <= settings.max_repeats
        for item in items:
            execution = view.get_record("execution_result", item["execution_id"])
            expected_score = score_answer(execution["answer"], normalize_reference(task["reference_answer"])) if execution["status"] == "completed" else None
            checks["scores_recomputed"] &= item["score"] == expected_score and item["status"] == execution["status"]
            if expected_score is not None:
                checks["scores_recomputed"] &= view.get_record("score", item["execution_id"]) == expected_score
            checks["repeat_associations"] &= bool(graph and view.links[item["execution_id"]]["candidate"] == cid
                and item["executor_hash"] == execution["executor_hash"] == snapshot["executor_hash"]
                and item["repeat_id"] == execution["repeat_id"])
        aggregated = aggregate_label(items, required=settings.complete_repeats, max_repeats=settings.max_repeats)
        if label:
            checks["labels_recomputed"] &= (all(label[k] == value for k, value in aggregated.items())
                and label["task_id"] == task["task_id"] and label["split"] == candidate["split"]
                and bool(graph) and label["graph_fingerprint"] == graph["fingerprint"])
        if terminal:
            checks["terminal_results"] &= bool(generated) and (
                (not generated["eligible_for_execution"] and terminal["status"] == "generation_invalid")
                or (generated["eligible_for_execution"] and not selected and not items and label is None
                    and terminal["status"] == "execution_not_selected")
                or (generated["eligible_for_execution"] and selected and label is not None
                    and aggregated["status"] in {"complete", "incomplete"} and terminal["status"] == aggregated["status"]))
        details.append({**candidate, "candidate_id": cid, "status": terminal["status"] if terminal else "pending",
            "generated": generated is not None, "generation": generated,
            "raw_blob": view.by_kind["generation"][cid]["blob"] if raw else None,
            "reached_eos": raw["reached_eos"] if raw else False, "finish_reason": raw["finish_reason"] if raw else None,
            "lengths": {k: raw[k] for k in ("input_tokens", "output_tokens", "reasoning_tokens", "content_tokens")} if raw else None,
            "generation_seconds": raw["seconds"] if raw else None,
            "peak_allocated_bytes": raw["peak_allocated_bytes"] if raw else None,
            "graph": graph, "label": label, "repeats": items,
            "observed_successes": aggregated["success_count"],
            "content_excerpt": trajectory["raw_content"][:1200] if trajectory else None,
            "reasoning_excerpt": trajectory["raw_reasoning"][:400] if trajectory else None})
    groups = []
    grouped = defaultdict(list)
    for item in details:
        grouped[(item["split"], item["model"], item["z"])].append(item)
    for (split, model, z), rows in sorted(grouped.items()):
        planned = len(rows)
        generated = sum(row["generated"] for row in rows)
        eligible = sum(bool(row["generation"] and row["generation"]["eligible_for_execution"]) for row in rows)
        execution_rows = [row for row in rows if row.get("execution_selected", True)]
        execution_planned = len(execution_rows)
        execution_eligible = sum(bool(row["generation"] and row["generation"]["eligible_for_execution"]) for row in execution_rows)
        complete = [row for row in rows if row["label"] and row["label"]["status"] == "complete"]
        successful = sum(row["label"]["success_count"] for row in complete)
        incomplete = sum(row["status"] == "incomplete" for row in rows)
        observed = sum(row["observed_successes"] > 0 for row in rows)
        cids = {row["candidate_id"] for row in rows}
        executions = {row["id"] for row in view.record_items("execution") if view.links[row["id"]].get("candidate") in cids}
        owners = {row["id"] for row in view.record_items("execution_node") if view.links[row["id"]].get("execution") in executions}
        requests = [row for row in view.tables["requests"] if row["owner_id"] in owners]
        group = {"split": split, "model": model, "z": z, "planned": planned, "generated": generated,
            "pending": sum(row["status"] == "pending" for row in rows), "eligible_graphs": eligible,
            "complete_labels": len(complete), "incomplete_labels": incomplete,
            "execution_planned": execution_planned, "execution_eligible_graphs": execution_eligible,
            "execution_not_selected": planned - execution_planned,
            "execution_incomplete_rate": incomplete / execution_eligible if execution_eligible else None,
            "completed_graph_successes": successful, "completed_graph_execution_denominator": len(complete) * settings.complete_repeats,
            "success_rate": successful / (len(complete) * settings.complete_repeats) if complete else None,
            "label_distribution": dict(Counter(str(row["label"]["mean_outcome"]) for row in complete)),
            "correct_execution_coverage_count": observed, "correct_execution_coverage_denominator": execution_planned,
            "correct_execution_coverage": observed / execution_planned,
            "generation_rate_denominator": generated, "execution_status": "无可执行图" if generated == planned and not eligible else "有可执行图" if eligible else "生成待完成",
            "generation_seconds": sum(row["generation_seconds"] or 0 for row in rows),
            "token_totals": {key: sum((row["lengths"] or {}).get(key, 0) for row in rows)
                             for key in ("input_tokens", "output_tokens", "reasoning_tokens", "content_tokens")},
            "repeat_statuses": dict(Counter(item["status"] for row in rows for item in row["repeats"])),
            "error_counts": dict(Counter(row["generation"]["error"] for row in rows if row["generation"] and row["generation"]["error"])),
            "usage": summarize_usage(view, requests)}
        for name, count in (("normal_termination", sum(row["generated"] and row["reached_eos"] for row in rows)),
                            ("json_parsed", sum(bool(row["generation"] and row["generation"]["json_parsed"]) for row in rows)),
                            ("valid_graph", sum(bool(row["generation"] and row["generation"]["valid_graph"]) for row in rows)),
                            ("truncated", sum(row["generated"] and row["finish_reason"] == "length" for row in rows))):
            group[name + "_count"] = count
            group[name + "_rate"] = count / generated if generated else None
        groups.append(group)
    comparisons = []
    for left, right in itertools.combinations(details, 2):
        if left["task_id"] == right["task_id"] and left["split"] == right["split"] and left["slot"] == right["slot"]:
            if (left["model"] != right["model"] and left["z"] == right["z"]) or (left["model"] == right["model"] and left["z"] != right["z"]):
                comparisons.append(graph_comparison(left, right))
    terminal = sum(row["status"] != "pending" for row in details)
    finished = terminal == len(expected) and run["halt_reason"] is None
    policy_executor_evidence = any(row["model"] == "last_checkpoint" and any(
        item["status"] == "completed" for item in row["repeats"]) for row in details)
    return {"version": "ocop.evaluation_report.v1", "run_id": run["id"], "phase": phase,
        "snapshot_records": len(view.tables["records"]), "planned": len(expected), "generated": sum(r["generated"] for r in details),
        "terminal": terminal, "finished": finished, "halt_reason": run["halt_reason"],
        "integrity_passed": all(checks.values()), "checks": checks,
        "policy_executor_evidence": policy_executor_evidence,
        "engineering_passed": finished and all(checks.values()) and policy_executor_evidence,
        "groups": groups, "candidates": details, "comparisons": comparisons,
        "request_count": len(view.tables["requests"]), "attempt_count": len(view.tables["attempts"]),
        "usage": summarize_usage(view, view.tables["requests"]),
        **({"cost_budget": budget} if budget is not None else {}),
        "provenance": {**{key: snapshot[key] for key in ("source", "training", "models", "generation_config", "executor_hash", "environment")},
                       **({"inference_environment": snapshot["inference_environment"]} if "inference_environment" in snapshot else {})}}


def write_metrics(writer, report):
    logged = getattr(writer, "ocop_candidates", set())
    for row in report["candidates"]:
        if not row["generated"] or row["candidate_id"] in logged:
            continue
        prefix = f"generation/{row['split']}/{row['model']}"
        values = {**row["lengths"], "normal_termination": int(row["reached_eos"]),
                  "eligible": int(row["generation"]["eligible_for_execution"]), "seconds": row["generation_seconds"],
                  "peak_allocated_bytes": row["peak_allocated_bytes"]}
        for name, value in values.items():
            if value is not None:
                writer.add_scalar(f"{prefix}/{name}", value, row["ordinal"])
        logged.add(row["candidate_id"])
    writer.ocop_candidates = logged
    if getattr(writer, "ocop_snapshot", None) != report["snapshot_records"]:
        for group in report["groups"]:
            prefix = f"evaluation/{group['split']}/{group['model']}/z_{group['z']}"
            for key, value in group.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"{prefix}/{key}", value, report["snapshot_records"])
        writer.add_scalar("evaluation/policy_executor_evidence", int(report["policy_executor_evidence"]), report["snapshot_records"])
        writer.ocop_snapshot = report["snapshot_records"]
    writer.flush()


def audit_metrics(path, report):
    events = EventAccumulator(str(path / "tensorboard"), size_guidance={"scalars": 0}).Reload()
    for group in report["groups"]:
        prefix = f"evaluation/{group['split']}/{group['model']}/z_{group['z']}"
        for key in ("planned", "generated", "complete_labels", "correct_execution_coverage"):
            points = events.Scalars(f"{prefix}/{key}")
            if not points or not math.isclose(points[-1].value, group[key], rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError("TensorBoard group totals differ from persisted records")
    for split, model in {(row["split"], row["model"]) for row in report["candidates"] if row["generated"]}:
        expected = {row["ordinal"]: row["lengths"]["output_tokens"] for row in report["candidates"]
                    if row["generated"] and row["split"] == split and row["model"] == model}
        points = events.Scalars(f"generation/{split}/{model}/output_tokens")
        if len(points) != len(expected) or {p.step: p.value for p in points} != expected:
            raise ValueError("TensorBoard generation events are duplicated or incomplete")
    return {"passed": True, "generated": report["generated"], "groups": len(report["groups"])}


def audit_generation_tokens(view):
    snapshot = view.archive["config"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot["runtime"]["policy"]["model_path"], local_files_only=True)
    tasks = {task["task_id"]: task for task in snapshot["task_manifest"]["tasks"]}
    count = 0
    for row in view.record_items("generation"):
        raw = row["payload"]
        candidate = view.by_id[row["key"]]["payload"]
        expected = tokenizer.apply_chat_template(policy_messages(tasks[candidate["task_id"]]["question"], candidate["z"]),
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
        if (raw["input_ids"] != expected or tokenizer.decode(raw["token_ids"], skip_special_tokens=False) != raw["text"]
                or raw["eos_token_id"] != tokenizer.eos_token_id or raw["eos_token"] != tokenizer.eos_token):
            raise ValueError("Saved generation tokens, text or policy prompt mismatch")
        count += 1
    return {"passed": True, "generations": count}


def markdown_report(report):
    evaluation_tasks = {row["task_id"] for row in report["candidates"] if row["split"] != "train_smoke"}
    evaluation_splits = sorted({row["split"] for row in report["candidates"] if row["split"] != "train_smoke"})
    lines = ["# PolicyLLM 评估", "", f"Run：`{report['run_id']}`；已生成 {report['generated']}/{report['planned']}，"
             f"候选终态 {report['terminal']}/{report['planned']}。", "",
             f"记录一致性：{report['integrity_passed']}；预算完成：{report['finished']}；"
             f"重载 PolicyLLM 到公共 executor 的实证：{report['policy_executor_evidence']}。", "",
             "| split | 模型 | z | 生成/计划 | 可执行图 | 完整标签 | 成功执行/完整标签执行 | 正确执行覆盖 | 状态 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for g in report["groups"]:
        score = f"{g['completed_graph_successes']}/{g['completed_graph_execution_denominator']}" if g["complete_labels"] else "未测得"
        lines.append(f"| {g['split']} | {g['model']} | {g['z']} | {g['generated']}/{g['planned']} | {g['eligible_graphs']} | "
                     f"{g['complete_labels']} | {score} | {g['correct_execution_coverage_count']}/{g['correct_execution_coverage_denominator']} | {g['execution_status']} |")
    execution_tasks = {row["task_id"] for row in report["candidates"] if row["split"] != "train_smoke" and row.get("execution_selected", True)}
    lines += ["", "终止率、解析率、合法率及截断率以已完成生成为分母；正确执行覆盖以预先选定的执行候选为分母。",
              "完整标签成功率仅使用完成规定次数的图；待完成和incomplete单列。训练题smoke单列。",
              f"评估来源：{', '.join(evaluation_splits)}；共 {len(evaluation_tasks)} 道题，执行抽样 {len(execution_tasks)} 道题。",
              "执行子集由 seed 与 task ID 确定，在生成前固定；未抽中题的合法图仅报告生成表现，不计为执行失败。",
              "完整长度、usage、错误、角色与边变化、原始记录关联见 evaluation-report.json。", "", "## 样例", ""]
    if "cost_budget" in report:
        budget = report["cost_budget"]
        lines[7:7] = [f"执行费用上限 ${budget['max_usd']:.2f}；累计含未知请求预留 ${budget['accounted_cost_usd']:.6f}；"
                      f"其中未知预留 ${budget['reserved_unknown_usd']:.6f}。", ""]
    seen = set()
    for row in report["candidates"]:
        key = (row["split"], row["model"], row["status"], (row["generation"] or {}).get("error"))
        if not row["generated"] or key in seen:
            continue
        seen.add(key)
        lines += [f"### {row['split']} / {row['model']} / z={row['z']} / {row['status']}", "",
                  f"任务：`{row['task_id']}`；错误：`{row['generation']['error']}`；[完整原始记录](blobs/{row['raw_blob']})。", "",
                  "    " + (row["content_excerpt"] or row["reasoning_excerpt"] or "空输出").replace("\n", "\n    "), ""]
    return "\n".join(lines) + "\n"


def save_report(store, phase, writer):
    view = EvaluationView(store.path)
    report = build_report(view, phase)
    write_metrics(writer, report)
    if phase in {"finished", "report"}:
        report["token_audit"] = audit_generation_tokens(view)
        report["tensorboard"] = audit_metrics(store.path, report)
    write_json(store.path / "evaluation-report.json", report)
    temporary = store.path / "evaluation-report.md.tmp"
    temporary.write_text(markdown_report(report), encoding="utf-8")
    os.replace(temporary, store.path / "evaluation-report.md")
    temporary = store.path / "records.jsonl.tmp"
    export_run(store.path, temporary)
    os.replace(temporary, store.path / "records.jsonl")
    return report


def report_evaluation(path):
    with FileLock(path / "writer.lock", timeout=0):
        with SummaryWriter(str(path / "tensorboard"), purge_step=0) as writer:
            return save_report(EvaluationView(path), "report", writer)
