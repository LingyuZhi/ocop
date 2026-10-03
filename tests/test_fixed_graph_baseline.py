import asyncio
import copy
import importlib.util
import importlib
import json
from pathlib import Path

import httpx
import pytest

import ocop.evaluation as evaluation
from ocop.diagnostics import provenance
from ocop.storage import RunStore, StoreConflict
from test_collection import reply
from test_evaluation import environment, path, run, runtime, snapshot, tokenizer


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fixed_graph_baseline", ROOT / "scripts/fixed_graph_baseline.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


@pytest.fixture
def prepared(tmp_path, environment, tokenizer, monkeypatch):
    def source_snapshot(config, *args):
        value = snapshot(config, tokenizer)
        value["environment"] = provenance()
        return value, tokenizer

    monkeypatch.setattr(evaluation, "prepare_evaluation", source_snapshot)
    config = runtime(tmp_path)
    report, status = run(config, environment)
    assert status == 0 and report["finished"]
    settings = baseline.Settings.model_validate_json((ROOT / "config/fixed-graph-baseline.json").read_text())
    config, frozen = baseline.prepare(path(config), settings)
    return config, frozen


def execute(prepared, handler=reply, **kwargs):
    config, frozen = prepared
    return asyncio.run(baseline.run(config, frozen, "fixed-test", Path("absent.env"),
                                   transport=httpx.MockTransport(handler), **kwargs))


def run_path(prepared):
    return Path(prepared[0].artifacts_dir) / "runs/fixed-test"


def test_fixed_chain_budget_pairing_and_resume(prepared):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return reply(request)

    initial, code = execute(prepared, handler, prepare_only=True)
    assert code == 0 and initial["planned"] == 1 and not requests
    result, code = execute(prepared, handler)
    assert code == 0 and result["finished"] and result["integrity_passed"] and result["tensorboard_passed"]
    assert len(requests) == result["request_count"] == 25
    assert result["success_rate"] == 1 and result["complete_labels"] == 1
    assert all(row["execution_denominator_each"] == 5 for row in result["comparisons"])
    assert all("ReferenceSecret" not in json.dumps(request) for request in requests)
    view = baseline.ReadStore(run_path(prepared))
    nodes = view.record_items("execution_node")
    for row in nodes:
        node = row["payload"]["node"]
        if node.startswith("worker_"):
            inputs = json.loads(row["payload"]["messages"][-1]["content"])
            expected = [] if node == "worker_0" else [f"worker_{int(node[-1]) - 1}"]
            assert [p["worker_id"] for p in inputs["predecessors"]] == expected
    resumed, code = execute(prepared, handler)
    assert code == 0 and resumed["request_count"] == 25 and len(requests) == 25
    assert len(resumed["candidates"][0]["repeats"]) == 5
    database = run_path(prepared) / "records.sqlite3"
    checksum = baseline.file_hash(database)
    assert baseline.report(run_path(prepared))["integrity_passed"]
    assert baseline.file_hash(database) == checksum


@pytest.mark.parametrize("kind", ["execution_result", "score", "baseline_repeat", "label", "candidate_result"])
def test_interrupted_persistence_resumes_without_duplicate_calls(prepared, monkeypatch, kind):
    original = RunStore.put_record
    interrupted = False
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return reply(request)

    def save_then_interrupt(store, record_kind, *args, **kwargs):
        nonlocal interrupted
        value = original(store, record_kind, *args, **kwargs)
        if record_kind == kind and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted after persistence")
        return value

    monkeypatch.setattr(RunStore, "put_record", save_then_interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        execute(prepared, handler)
    result, code = execute(prepared, handler)
    assert code == 0 and result["integrity_passed"] and calls == 25


def test_failures_remain_bounded_and_incomplete_excluded(prepared):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(503, text="unavailable")

    result, code = execute(prepared, handler)
    assert code == 0 and result["finished"] and result["incomplete_labels"] == 1
    assert result["success_rate"] is None and result["execution_denominator"] == 0
    assert count == 8 * 3 and len(result["candidates"][0]["repeats"]) == 8
    assert all(r["policy_success_rate"] is None for r in result["comparisons"])


def test_source_and_graph_changes_refuse_resume(prepared):
    execute(prepared, prepare_only=True)
    config, frozen = prepared
    changed = copy.deepcopy(frozen)
    changed["source"]["report_hash"] = "changed"
    with pytest.raises(StoreConflict):
        execute((config, changed))


def test_pairing_keeps_zero_legal_groups_and_uses_common_tasks():
    details = [{"task_id": "a", "status": "complete", "graph_fingerprint": "g",
                "label": {"status": "complete", "success_count": 4}},
               {"task_id": "b", "status": "incomplete", "graph_fingerprint": "g",
                "label": {"status": "incomplete", "success_count": 2}}]
    rows = [{"task_id": task, "split": "eval", "model": model, "z": 0.0,
        "observed_successes": int(model == "last_checkpoint"),
        "label": {"status": "complete", "complete_count": 5, "success_count": 3} if model == "last_checkpoint" else None,
        "graph": {"fingerprint": "g"} if model == "last_checkpoint" else None}
        for model in ("base", "last_checkpoint") for task in ("a", "b")]
    groups = baseline.comparisons(details, {"candidates": rows})
    assert groups[0]["paired_tasks"] == [] and groups[0]["policy_success_rate"] is None
    assert groups[1]["paired_tasks"] == ["a"] and groups[1]["execution_denominator_each"] == 5
    assert groups[1]["baseline_successes"] == 4 and groups[1]["policy_successes"] == 3


def test_final_verification_detects_rewritten_report(prepared, monkeypatch):
    execute(prepared)
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    verification = importlib.import_module("verify_evaluation")
    result = verification.verify(run_path(prepared))
    assert result["passed"] and result["checks"]["jsonl_matches_database"]
    report_path = run_path(prepared) / "baseline-report.json"
    saved = json.loads(report_path.read_text())
    saved["successes"] = 999
    report_path.write_text(json.dumps(saved))
    rejected = verification.verify(run_path(prepared))
    assert not rejected["passed"] and not rejected["checks"]["report_recomputed"]
