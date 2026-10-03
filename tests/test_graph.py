import copy
import itertools
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ocop.graph import VERSION, contract_hash, load_contract, replay, trajectory_schema


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "graph"


def step(kind, **fields):
    return {"explanation": "Design decision.", "action": {"type": kind, **fields}}


def assignments():
    return [step("ASSIGN_ROLE", worker_id=f"worker_{i}", role="Solver") for i in range(4)]


def edge(source, target):
    return step("ADD_EDGE", source=f"worker_{source}", target=f"worker_{target}")


def encode(steps):
    return json.dumps({"version": VERSION, "steps": steps})


def test_example_has_backward_edge_isolated_worker_and_multiple_sinks():
    content = (EXAMPLES / "valid.json").read_text()
    result = replay(content, reasoning="Independent native reasoning.")
    assert result.valid
    assert result.raw_content == content
    assert result.raw_reasoning == "Independent native reasoning."
    assert result.final_graph.edges == (("worker_3", "worker_0"), ("worker_0", "worker_1"))
    assert len(result.final_graph.workers) == 4
    assert result.final_graph.workers[2].role == "Solver"
    assert len(result.snapshots) == 7
    assert result.snapshots[-1] == result.snapshots[-2] == result.final_graph
    assert all(worker.role is None for worker in result.initial_graph.workers)
    assert result.initial_graph.edges == ()
    assert result == replay(content, reasoning="Independent native reasoning.")


def test_empty_graph_and_repeated_roles_are_valid():
    result = replay(encode(assignments() + [step("STOP")]))
    assert result.valid
    assert result.final_graph.edges == ()
    assert all(worker.role == "Solver" for worker in result.final_graph.workers)
    assert result.raw_reasoning is None


def test_invalid_fixture_retains_only_valid_prefix():
    result = replay((EXAMPLES / "invalid.json").read_text())
    assert not result.valid
    assert result.error.code == "action_phase"
    assert result.error.step_index == 1
    assert len(result.snapshots) == 1
    assert result.snapshots[0].workers[0].role == "Solver"
    assert result.final_graph is None


@pytest.mark.parametrize(("steps", "code", "index"), [
    ([], "missing_stop", 0),
    (assignments(), "missing_stop", 4),
    ([step("STOP")], "incomplete_roles", 0),
    (assignments()[:3] + [step("STOP")], "incomplete_roles", 3),
    ([edge(0, 1)], "action_phase", 0),
    ([assignments()[1]], "assignment_order", 0),
    (assignments()[:1] * 2, "duplicate_assignment", 1),
    (assignments() + [edge(0, 1), assignments()[0]], "duplicate_assignment", 5),
    (assignments() + [edge(0, 0)], "self_loop", 4),
    (assignments() + [edge(0, 1), edge(0, 1)], "duplicate_edge", 5),
    (assignments() + [edge(0, 1), edge(1, 0)], "cycle", 5),
    (assignments() + [edge(3, 0), edge(0, 2), edge(2, 3)], "cycle", 6),
    (assignments() + [step("STOP"), step("STOP")], "after_stop", 5),
    (assignments() + [step("STOP"), edge(0, 1)], "after_stop", 5),
    (assignments() + [step("STOP"), {}], "after_stop", 5),
    ([step("ASSIGN_ROLE", worker_id="worker_4", role="Solver")], "invalid_step", 0),
    ([step("ASSIGN_ROLE", worker_id=0, role="Solver")], "invalid_step", 0),
    ([step("ASSIGN_ROLE", worker_id="worker_0", role="Unknown")], "invalid_step", 0),
    (assignments() + [edge(0, 4)], "invalid_step", 4),
    ([step("UNKNOWN")], "invalid_step", 0),
    ([step("STOP", extra=True)], "invalid_step", 0),
    ([{"explanation": " ", "action": {"type": "STOP"}}], "invalid_step", 0),
    ([{"action": {"type": "STOP"}}], "invalid_step", 0),
])
def test_rejects_first_invalid_action_without_final_graph(steps, code, index):
    result = replay(encode(steps))
    assert not result.valid
    assert result.error.code == code
    assert result.error.step_index == index
    assert len(result.snapshots) == index
    assert result.final_graph is None


@pytest.mark.parametrize("content", [
    "", "```json\n{}\n```", '{"version":', '{} trailing',
    '{"version":"ocop.graph.v1","version":"ocop.graph.v1","steps":[]}',
    '{"version":"ocop.graph.v1","steps":[{"action":{"type":"STOP","type":"STOP"}}]}',
    '{"version":"ocop.graph.v1","steps":NaN}',
    '{"version":"ocop.graph.v1","steps":Infinity}',
])
def test_rejects_malformed_or_ambiguous_json(content):
    result = replay(content)
    assert result.error.code == "invalid_json"
    assert result.raw_content == content
    assert result.snapshots == ()


@pytest.mark.parametrize("document", [
    None, [], {"steps": []}, {"version": "unknown", "steps": []},
    {"version": VERSION, "steps": {}}, {"version": VERSION, "steps": [], "extra": True},
])
def test_envelope_validation(document):
    result = replay(json.dumps(document))
    assert result.error.code == "invalid_envelope"
    assert result.error.step_index is None


def test_earlier_semantic_error_wins_over_later_schema_error():
    result = replay(encode([edge(0, 1), {"bad": "step"}]))
    assert result.error.code == "action_phase"
    assert result.error.step_index == 0


def test_prefix_snapshots_do_not_change_with_later_steps():
    result = replay(encode(assignments() + [edge(0, 1), edge(1, 0)]))
    assert result.snapshots[0].workers[1].role is None
    assert result.snapshots[3].edges == ()
    assert result.snapshots[4].edges == (("worker_0", "worker_1"),)
    with pytest.raises(FrozenInstanceError):
        result.snapshots[0].workers[0].role = "Checker"


def test_every_topological_worker_order_and_edge_order_is_accepted():
    for order in itertools.permutations(range(4)):
        links = [edge(order[i], order[i + 1]) for i in range(3)]
        for insertion in itertools.permutations(links):
            steps = assignments() + list(insertion) + [step("STOP")]
            result = replay(encode(steps))
            assert result.valid
            assert result.final_graph.edges == tuple(
                (item["action"]["source"], item["action"]["target"]) for item in insertion
            )


def test_schema_and_contract_are_consistent():
    schema = trajectory_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads((EXAMPLES / "valid.json").read_text()))
    contract = load_contract()
    assert contract["version"] == VERSION
    assert tuple(contract["roles"]) == ("Solver", "Decomposer", "Checker", "Reviser")
    for role in contract["roles"]:
        steps = assignments() + [step("STOP")]
        steps[0]["action"]["role"] = role
        assert replay(encode(steps)).valid
    assert all(isinstance(prompt, str) and prompt.strip() for prompt in contract["roles"].values())
    assert len(contract_hash()) == 64
    assert replay(encode(assignments())).contract_hash == contract_hash()
    modified = copy.deepcopy(contract)
    modified["roles"]["Solver"] = "Changed locally."
    assert load_contract() == contract
