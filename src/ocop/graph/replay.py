import json
from dataclasses import dataclass
from typing import Any

import networkx as nx
from jsonschema import Draft202012Validator

from ocop.graph.contract import VERSION, contract_hash, load_contract, trajectory_schema


@dataclass(frozen=True)
class Worker:
    worker_id: str
    role: str | None


@dataclass(frozen=True)
class GraphSnapshot:
    workers: tuple[Worker, ...]
    edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ReplayError:
    code: str
    message: str
    step_index: int | None = None
    path: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class ReplayResult:
    version: str
    contract_hash: str
    raw_content: str
    raw_reasoning: str | None
    initial_graph: GraphSnapshot
    snapshots: tuple[GraphSnapshot, ...]
    final_graph: GraphSnapshot | None
    error: ReplayError | None

    @property
    def valid(self) -> bool:
        return self.error is None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def replay(content: str, *, reasoning: str | None = None) -> ReplayResult:
    workers = tuple(load_contract()["worker_ids"])
    roles: dict[str, str | None] = dict.fromkeys(workers)
    edges: list[tuple[str, str]] = []
    graph = nx.DiGraph()
    graph.add_nodes_from(workers)
    snapshots: list[GraphSnapshot] = []

    def snapshot() -> GraphSnapshot:
        return GraphSnapshot(tuple(Worker(worker, roles[worker]) for worker in workers), tuple(edges))

    initial = snapshot()

    def result(error: ReplayError | None = None) -> ReplayResult:
        return ReplayResult(
            VERSION, contract_hash(), content, reasoning, initial,
            tuple(snapshots), snapshot() if error is None else None, error,
        )

    try:
        document = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        return result(ReplayError("invalid_json", str(exc)))

    schema = trajectory_schema()
    envelope_schema = {**schema, "properties": {**schema["properties"], "steps": {"type": "array"}}}
    error = next(Draft202012Validator(envelope_schema).iter_errors(document), None)
    if error is not None:
        return result(ReplayError("invalid_envelope", error.message, path=tuple(error.absolute_path)))

    step_validator = Draft202012Validator(schema["$defs"]["step"])
    assigned = 0
    stopped = False
    for index, step in enumerate(document["steps"]):
        if stopped:
            return result(ReplayError("after_stop", "No steps are allowed after STOP.", index))
        error = next(step_validator.iter_errors(step), None)
        if error is not None:
            return result(ReplayError("invalid_step", error.message, index, tuple(error.absolute_path)))

        action = step["action"]
        kind = action["type"]
        if kind == "ASSIGN_ROLE":
            worker = action["worker_id"]
            if roles[worker] is not None:
                return result(ReplayError("duplicate_assignment", f"{worker} already has a role.", index))
            if worker != workers[assigned]:
                return result(ReplayError("assignment_order", f"Expected assignment to {workers[assigned]}.", index))
            roles[worker] = action["role"]
            assigned += 1
        elif kind == "ADD_EDGE":
            if assigned != len(workers):
                return result(ReplayError("action_phase", "Assign all roles before adding edges.", index))
            source, target = action["source"], action["target"]
            if source == target:
                return result(ReplayError("self_loop", "Self edges are forbidden.", index))
            if graph.has_edge(source, target):
                return result(ReplayError("duplicate_edge", "This edge already exists.", index))
            if nx.has_path(graph, target, source):
                return result(ReplayError("cycle", "This edge would create a cycle.", index))
            graph.add_edge(source, target)
            edges.append((source, target))
        else:
            if assigned != len(workers):
                return result(ReplayError("incomplete_roles", "STOP requires all roles to be assigned.", index))
            stopped = True
        snapshots.append(snapshot())

    if not stopped:
        return result(ReplayError("missing_stop", "The trajectory must end with STOP.", len(document["steps"])))
    return result()
