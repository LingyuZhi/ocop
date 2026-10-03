import copy
import importlib.util
from pathlib import Path

import pytest

from test_collection import VALID_GRAPH


SPEC = importlib.util.spec_from_file_location("verify_evaluation",
    Path(__file__).resolve().parents[1] / "scripts/verify_evaluation.py")
verification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verification)


def reports():
    candidate = {"candidate_id": "c", "task_id": "q", "split": "eval", "model": "last_checkpoint",
        "z": 0.0, "seed": 42, "slot": 0, "generated": True, "raw_blob": "raw",
        "generation": {"eligible": True}, "graph": {"fingerprint": "g"}, "repeats": [{"repeat_id": "0"}],
        "status": "pending", "label": None}
    before = {"run_id": "run", "candidates": [candidate]}
    after = copy.deepcopy(before)
    after["candidates"][0].update(status="complete", label={"mean_outcome": 1.0})
    after["candidates"][0]["repeats"].append({"repeat_id": "1"})
    return before, after


def test_history_accepts_new_work_and_preserves_incomplete_terminal():
    before, after = reports()
    assert all(verification.history_checks(before, after).values())
    before["candidates"][0].update(status="incomplete", label={"mean_outcome": None})
    assert not verification.history_checks(before, after)["terminal_history"]


@pytest.mark.parametrize(("field", "value", "check"), [
    ("seed", 43, "candidate_history"), ("raw_blob", "changed", "generation_history"),
    ("repeats", [{"repeat_id": "1"}], "repeat_history"),
])
def test_history_rejects_rewritten_results(field, value, check):
    before, after = reports()
    after["candidates"][0][field] = value
    assert not verification.history_checks(before, after)[check]


def test_label_variation_for_same_graph_is_separate_from_topology_coverage():
    samples = [{"task_id": task, "z": z, "raw_content": VALID_GRAPH, "raw_reasoning": ""}
               for task, z in [("q1", 0.0), ("q1", 1.0), ("q2", 1.0)]]
    coverage = verification.training_coverage(samples)
    assert coverage["distinct_labels_per_task"] == {2: 1, 1: 1}
    assert coverage["distinct_graphs_per_task"] == {1: 2}
    assert coverage["task_graph_groups_with_multiple_labels"] == 1
