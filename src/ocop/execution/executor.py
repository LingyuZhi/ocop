import asyncio
import hashlib
from dataclasses import asdict

import networkx as nx

from ocop.runtime.config import RuntimeConfig, canonical_json
from ocop.graph import contract_hash, load_contract, replay
from ocop.graph.replay import ReplayResult
from ocop.runtime.llm import RequestRunner
from ocop.execution.scoring import SCORER_VERSION
from ocop.runtime.storage import RunHalted, RunStore
from ocop.runtime.usage import summarize_usage


EXECUTOR_VERSION = "ocop.executor.v1"
FINALIZER_PROMPT = (
    "Solve the original problem using the supplied worker outputs as evidence. "
    "Assess their calculations and resolve inconsistencies. "
    "End with a final line exactly formatted as #### followed by one space and a signed or unsigned "
    "decimal integer or decimal number, with digits before and after any decimal point. "
    "Do not include units, thousands separators, fractions, scientific notation, or text after that line."
)


def executor_config(config: RuntimeConfig) -> dict:
    for model in (config.worker_model, config.finalizer):
        if model.verified_model_id != model.model_id:
            raise ValueError("Executor requires verified worker and finalizer model IDs")
    if config.finalizer.answer_format != "final_line_hashes_decimal":
        raise ValueError("Unsupported finalizer answer format")
    return {
        "version": EXECUTOR_VERSION, "contract_hash": contract_hash(), "roles": load_contract()["roles"],
        "worker_model": config.worker_model.model_dump(mode="json"),
        "finalizer_model": config.finalizer.model_dump(mode="json"),
        "requests": config.requests.model_dump(mode="json"), "finalizer_prompt": FINALIZER_PROMPT,
        "message_format": "ocop.messages.v1", "scorer_version": SCORER_VERSION,
        "failure_policy": "stop_scheduling_drain_inflight_fatal_infra_incomplete.v1",
        "answer_pattern": r"#### [+-]?[0-9]+(?:\.[0-9]+)?", "trailing_newlines": "allowed",
    }


def executor_hash(config: RuntimeConfig) -> str:
    return hashlib.sha256(canonical_json(executor_config(config))).hexdigest()


def worker_messages(question: str, role: str, predecessors: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load_contract()["roles"][role]},
        {"role": "user", "content": canonical_json({"question": question, "predecessors": [
            {"worker_id": worker, "output": predecessors[worker]} for worker in sorted(predecessors)
        ]}).decode("utf-8")},
    ]


def finalizer_messages(question: str, outputs: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FINALIZER_PROMPT},
        {"role": "user", "content": canonical_json({"question": question, "workers": [
            {"worker_id": worker, "output": outputs[worker]} for worker in load_contract()["worker_ids"]
        ]}).decode("utf-8")},
    ]


def execution_usage(store: RunStore, owners: dict[str, str]) -> dict:
    groups = {"worker": [], "finalizer": []}
    for node, owner in owners.items():
        groups["finalizer" if node == "finalizer" else "worker"].extend(store.requests_for_owner(owner))
    return {name: summarize_usage(store, requests) for name, requests in groups.items()}


async def execute_repeat(*, question: str, graph: ReplayResult, parents: dict[str, str], repeat_id: str,
                         config: RuntimeConfig, runner: RequestRunner) -> dict:
    if not repeat_id or not question.strip():
        raise ValueError("Question and repeat ID must be nonempty")
    if not graph.valid or replay(graph.raw_content, reasoning=graph.raw_reasoning) != graph:
        raise ValueError("Executor requires a complete, current, validated trajectory")
    if runner.config != config.requests:
        raise ValueError("Request runner limits differ from executor configuration")
    store = runner.store
    spec = executor_config(config)
    digest = executor_hash(config)
    spec_id = store.put_record("executor_config", digest, spec)
    repeat_key = canonical_json({"parents": parents, "repeat_id": repeat_id}).decode("utf-8")
    execution_id = store.put_record("execution", repeat_key, {
        "repeat_id": repeat_id, "question": question, "graph": asdict(graph.final_graph),
        "trajectory_sha256": hashlib.sha256(graph.raw_content.encode()).hexdigest(), "executor_hash": digest,
    }, parents={**parents, "executor_config": spec_id})
    existing = store.get_record("execution_result", execution_id)
    if existing is not None:
        return existing
    dag = nx.DiGraph()
    roles = {worker.worker_id: worker.role for worker in graph.final_graph.workers}
    dag.add_nodes_from(roles)
    dag.add_edges_from(graph.final_graph.edges)
    outcomes = {}
    owners = {}
    for node in (*sorted(roles), "finalizer"):
        record = store.get_record("execution_node", f"{execution_id}:{node}")
        if record is not None:
            owner = store.put_record("execution_node", f"{execution_id}:{node}", record, parents={"execution": execution_id})
            owners[node] = owner
            saved = store.get_record("execution_node_result", owner)
            if saved is not None:
                outcomes[node] = saved

    async def call_node(node, messages, model):
        owner = store.put_record("execution_node", f"{execution_id}:{node}",
                                 {"node": node, "messages": messages}, parents={"execution": execution_id})
        owners[node] = owner
        try:
            outcome = await runner.call(model, key=f"execution:{execution_id}:{node}", messages=messages, owner_id=owner)
        except RunHalted:
            reason = store.rows("run")[0]["halt_reason"]
            if not reason:
                raise
            outcome = {"status": "fatal" if reason != "consecutive_request_failures" else "infra_failed",
                       "error": f"run_halted:{reason}", "usage": None}
        if outcome["status"] == "completed":
            if outcome.get("response_model") != model.verified_model_id:
                outcome = {**outcome, "status": "fatal", "error": "response_model_mismatch"}
            elif model.provider_order and outcome.get("provider") not in model.provider_order:
                outcome = {**outcome, "status": "fatal", "error": "response_provider_mismatch"}
            if outcome["status"] == "fatal":
                store.halt(outcome["error"])
        store.put_record("execution_node_result", owner, outcome, parents={"node": owner})
        return outcome

    running = {}
    failed = any(outcome["status"] != "completed" for outcome in outcomes.values())
    try:
        while True:
            if not failed:
                active = set(running.values())
                for node in sorted(dag.nodes):
                    if node in outcomes or node in active:
                        continue
                    predecessors = sorted(dag.predecessors(node))
                    if all(parent in outcomes and outcomes[parent]["status"] == "completed" for parent in predecessors):
                        messages = worker_messages(question, roles[node], {parent: outcomes[parent]["content"] for parent in predecessors})
                        running[asyncio.create_task(call_node(node, messages, config.worker_model))] = node
            if not running:
                break
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                node = running.pop(task)
                outcomes[node] = task.result()
                failed = failed or outcomes[node]["status"] != "completed"
        if not failed and "finalizer" not in outcomes:
            outcomes["finalizer"] = await call_node("finalizer", finalizer_messages(question, {
                node: outcomes[node]["content"] for node in roles
            }), config.finalizer)
    finally:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
    statuses = {outcome["status"] for outcome in outcomes.values()}
    status = next((name for name in ("fatal", "infra_failed", "incomplete") if name in statuses), "completed")
    result = {"execution_id": execution_id, "repeat_id": repeat_id, "executor_hash": digest, "status": status,
              "nodes": {node: outcomes[node] for node in sorted(outcomes)},
              "unexecuted_nodes": sorted((set(roles) | {"finalizer"}) - set(outcomes)),
              "answer": outcomes.get("finalizer", {}).get("content") if status == "completed" else None,
              "usage": execution_usage(store, owners)}
    store.put_record("execution_result", execution_id, result, parents={"execution": execution_id})
    return result
