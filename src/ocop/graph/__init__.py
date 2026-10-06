import hashlib

from ocop.runtime.config import canonical_json
from ocop.graph.contract import VERSION, contract_hash, load_contract, trajectory_schema
from ocop.graph.replay import GraphSnapshot, ReplayError, ReplayResult, Worker, replay

__all__ = [
    "VERSION", "GraphSnapshot", "ReplayError", "ReplayResult", "Worker",
    "contract_hash", "load_contract", "replay", "trajectory_schema", "graph_fingerprint",
]


def graph_fingerprint(graph) -> str:
    payload = {"workers": sorted((worker.worker_id, worker.role) for worker in graph.workers),
               "edges": sorted(graph.edges)}
    return hashlib.sha256(canonical_json(payload)).hexdigest()
