import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import ocop.evaluation.supervision as supervision


class Runtime:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = str(artifacts_dir)

    def model_dump(self, mode="python"):
        return {"artifacts_dir": self.artifacts_dir, "purpose": "evaluation"}


def create_run(path, *, planned=2, generated=2, halt_reason=None, snapshot=None):
    path.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(path / "records.sqlite3")
    database.execute("CREATE TABLE run (halt_reason TEXT)")
    database.execute("INSERT INTO run VALUES (?)", (halt_reason,))
    database.execute("CREATE TABLE records (kind TEXT, logical_key TEXT)")
    database.executemany("INSERT INTO records VALUES ('generation_result', ?)",
                         [(str(index),) for index in range(generated)])
    database.commit()
    database.execute("PRAGMA user_version=1")
    database.commit()
    database.close()
    frozen = snapshot or {"candidates": [{} for _ in range(planned)]}
    (path / "evaluation-manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
    (path / "evaluation-report.json").write_text(json.dumps({
        "phase": "executing" if halt_reason else "starting", "planned": planned,
        "generated": generated, "terminal": 0, "halt_reason": halt_reason,
    }), encoding="utf-8")


def set_halt(path, value):
    database = sqlite3.connect(path / "records.sqlite3")
    database.execute("UPDATE run SET halt_reason=?", (value,))
    database.commit()
    database.close()


def run_supervisor(runtime, status_dir, *, recovery=None, **kwargs):
    return asyncio.run(supervision.supervise_evaluation(runtime,
        "source", "training", "credentials.env", "pilot-eval", "cuda:0", status_dir,
        recovery=recovery, progress_interval_seconds=0.01, **kwargs))


def test_circuit_recovery_reuses_run_and_switches_to_execute(tmp_path, monkeypatch):
    runtime = Runtime(tmp_path / "outputs")
    run_path = Path(runtime.artifacts_dir) / "runs" / "pilot-eval"
    status_dir = Path(runtime.artifacts_dir) / "experiments" / "pilot-eval-supervision"
    snapshot = {"purpose": "evaluation", "runtime": runtime.model_dump(mode="json"),
                "candidates": [{"key": "a"}, {"key": "b"}]}
    calls = []
    recovery_calls = []

    def prepare(*args):
        return snapshot, None

    monkeypatch.setattr(supervision, "_verify_completed_run", lambda *_args: {
        "run_id": "pilot-eval", "planned": 2, "generated": 2, "terminal": 2,
        "finished": True, "integrity_passed": True, "halt_reason": None,
    })

    async def evaluate(_runtime, *_args, phase):
        calls.append(phase)
        if phase == "all":
            create_run(run_path, snapshot=snapshot, halt_reason="consecutive_request_failures")
            return {"run_id": "pilot-eval", "planned": 2, "generated": 2, "terminal": 0,
                    "finished": False, "integrity_passed": True,
                    "halt_reason": "consecutive_request_failures"}, 1
        set_halt(run_path, None)
        (run_path / "evaluation-report.json").write_text(json.dumps({
            "phase": "finished", "planned": 2, "generated": 2, "terminal": 2, "halt_reason": None,
        }), encoding="utf-8")
        return {"run_id": "pilot-eval", "planned": 2, "generated": 2, "terminal": 2,
                "finished": True, "integrity_passed": True, "halt_reason": None}, 0

    async def recover(path, _runtime, saved, _settings, _credentials, **_kwargs):
        recovery_calls.append((path, saved))
        set_halt(run_path, None)

    monkeypatch.setattr(supervision, "prepare_evaluation", prepare)
    report, status = run_supervisor(runtime, status_dir, recovery=recover, evaluation=evaluate)

    assert status == 0
    assert report["finished"]
    assert calls == ["all", "execute"]
    assert len(recovery_calls) == 1
    state = json.loads((status_dir / "supervisor-state.json").read_text())
    assert state["status"] == "finished"
    assert state["recovery_episodes"] == 1
    assert state["attempt"] == 2


@pytest.mark.parametrize("halt_reason", ["evaluation_cost_budget_exhausted", "fatal_request_error"])
def test_budget_or_fatal_halt_stops_without_automatic_recovery(tmp_path, halt_reason):
    runtime = Runtime(tmp_path / "outputs")
    run_path = Path(runtime.artifacts_dir) / "runs" / "pilot-eval"
    status_dir = Path(runtime.artifacts_dir) / "experiments" / "pilot-eval-supervision"

    async def evaluate(_runtime, *_args, phase):
        assert phase == "all"
        create_run(run_path, halt_reason=halt_reason)
        return {"run_id": "pilot-eval", "planned": 2, "generated": 2, "terminal": 0,
                "finished": False, "integrity_passed": True,
                "halt_reason": halt_reason,
                "cost_budget": {"accounted_cost_usd": 1.35}}, 1

    async def reject_recovery(*_args, **_kwargs):
        raise AssertionError("Cost cap halts must never trigger connection recovery")

    report, status = run_supervisor(runtime, status_dir, recovery=reject_recovery, evaluation=evaluate)

    assert status == 1
    assert report["halt_reason"] == halt_reason
    state = json.loads((status_dir / "supervisor-state.json").read_text())
    assert state["status"] == "stopped"
    assert state["stop_reason"] == halt_reason


def test_restart_resumes_execute_when_generations_are_persisted(tmp_path):
    runtime = Runtime(tmp_path / "outputs")
    run_path = Path(runtime.artifacts_dir) / "runs" / "pilot-eval"
    status_dir = Path(runtime.artifacts_dir) / "experiments" / "pilot-eval-supervision"
    snapshot = {"purpose": "evaluation", "runtime": runtime.model_dump(mode="json"),
                "candidates": [{"key": "a"}, {"key": "b"}]}
    phases = []
    verified = {"run_id": "pilot-eval", "planned": 2, "generated": 2, "terminal": 2,
                "finished": True, "integrity_passed": True, "halt_reason": None}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(supervision, "_verify_completed_run", lambda *_args: verified)

    async def interrupted(_runtime, *_args, phase):
        phases.append(phase)
        create_run(run_path, snapshot=snapshot)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        run_supervisor(runtime, status_dir, evaluation=interrupted)

    monkeypatch.setattr(supervision, "prepare_evaluation", lambda *_args: (snapshot, None))

    async def complete(_runtime, *_args, phase):
        phases.append(phase)
        return {"run_id": "pilot-eval", "planned": 2, "generated": 2, "terminal": 2,
                "finished": True, "integrity_passed": True, "halt_reason": None}, 0

    try:
        report, status = run_supervisor(runtime, status_dir, evaluation=complete)
    finally:
        monkeypatch.undo()

    assert status == 0
    assert report["finished"]
    assert phases == ["all", "execute"]


def test_status_directory_cannot_be_inside_run(tmp_path):
    runtime = Runtime(tmp_path / "outputs")
    run_path = Path(runtime.artifacts_dir) / "runs" / "pilot-eval"
    with pytest.raises(ValueError, match="outside the evaluation run"):
        run_supervisor(runtime, run_path / "supervision")


def test_persisted_finished_state_is_revalidated(tmp_path, monkeypatch):
    runtime = Runtime(tmp_path / "outputs")
    status_dir = Path(runtime.artifacts_dir) / "experiments" / "pilot-eval-supervision"
    verified = {"run_id": "pilot-eval", "planned": 2, "generated": 2, "terminal": 2,
                "finished": True, "integrity_passed": True, "halt_reason": None}
    monkeypatch.setattr(supervision, "_verify_completed_run", lambda *_args: verified)

    async def complete(_runtime, *_args, phase):
        run_path = Path(runtime.artifacts_dir) / "runs" / "pilot-eval"
        create_run(run_path, snapshot={"purpose": "evaluation", "runtime": runtime.model_dump(mode="json"),
            "candidates": [{"key": "a"}, {"key": "b"}]})
        return verified, 0

    first, status = run_supervisor(runtime, status_dir, evaluation=complete)
    assert status == 0 and first["finished"]

    def reject_stale(*_args):
        raise supervision.StoreConflict("stale completion")

    monkeypatch.setattr(supervision, "_verify_completed_run", reject_stale)
    with pytest.raises(supervision.StoreConflict, match="stale completion"):
        run_supervisor(runtime, status_dir)
    state = json.loads((status_dir / "supervisor-state.json").read_text())
    assert state["status"] == "stopped"


def test_progress_monitor_updates_during_blocking_generation(tmp_path, monkeypatch):
    runtime = Runtime(tmp_path / "outputs")
    run_path = Path(runtime.artifacts_dir) / "runs" / "pilot-eval"
    status_dir = Path(runtime.artifacts_dir) / "experiments" / "pilot-eval-supervision"
    verified = {"run_id": "pilot-eval", "planned": 2, "generated": 2, "terminal": 2,
                "finished": True, "integrity_passed": True, "halt_reason": None}
    monkeypatch.setattr(supervision, "_verify_completed_run", lambda *_args: verified)

    def publish_one_generated_candidate():
        database = sqlite3.connect(run_path / "records.sqlite3")
        database.execute("INSERT INTO records VALUES ('generation_result', 'one')")
        database.commit()
        database.close()
        (run_path / "evaluation-report.json").write_text(json.dumps({
            "phase": "generating", "planned": 2, "generated": 1, "terminal": 0,
            "halt_reason": None,
        }), encoding="utf-8")

    async def blocking_generation(_runtime, *_args, phase):
        assert phase == "all"
        create_run(run_path, generated=0)
        timer = threading.Timer(0.03, publish_one_generated_candidate)
        timer.start()
        time.sleep(0.09)
        timer.join()
        return verified, 0

    report, status = run_supervisor(runtime, status_dir, evaluation=blocking_generation)

    assert status == 0
    assert report["finished"]
    state = json.loads((status_dir / "supervisor-state.json").read_text())
    assert state["progress"] == {"current": 1, "total": 2}
    assert state["last_progress_estimate"]["basis"] == "linear estimate from candidate progress in this phase"
