import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

from filelock import FileLock, Timeout

from ocop.collection import write_json
from ocop.evaluation import prepare_evaluation, run_evaluation
from ocop.evaluation_report import EvaluationView, build_report
from ocop.recovery import AutoRecoveryConfig, automatic_recovery
from ocop.storage import StoreConflict, read_database
from ocop.trajectories import digest


STATE_VERSION = "ocop.evaluation_supervisor.v1"
DEFAULT_MAX_RECOVERY_EPISODES = 5
DEFAULT_PROGRESS_INTERVAL_SECONDS = 30.0


def _run_path(runtime, run_id):
    return Path(runtime.artifacts_dir).resolve() / "runs" / run_id


def _status_summary(report):
    if not isinstance(report, dict):
        return None
    keys = ("run_id", "planned", "generated", "terminal", "finished", "integrity_passed",
            "engineering_passed", "policy_executor_evidence", "halt_reason", "cost_budget")
    return {key: report[key] for key in keys if key in report}


def _read_progress(run_path):
    manifest_path = run_path / "evaluation-manifest.json"
    database_path = run_path / "records.sqlite3"
    report_path = run_path / "evaluation-report.json"
    if not manifest_path.is_file() or not database_path.is_file() or not report_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    planned = len(manifest["candidates"])
    with read_database(run_path) as database:
        generated = database.execute("SELECT COUNT(*) FROM records WHERE kind='generation_result'").fetchone()[0]
        terminal = database.execute("SELECT COUNT(*) FROM records WHERE kind='candidate_result'").fetchone()[0]
        run = database.execute("SELECT halt_reason FROM run").fetchone()
    if generated > planned or terminal > planned:
        raise StoreConflict("Saved evaluation progress exceeds its frozen candidate plan")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {"planned": planned, "generated": generated, "terminal": terminal,
            "halt_reason": run["halt_reason"], "report": report}


def _next_phase(run_path):
    progress = _read_progress(run_path)
    if progress is None or progress["generated"] < progress["planned"]:
        return "all"
    return "execute"


def _phase_progress(progress, requested_phase):
    if progress is None:
        return ("generation" if requested_phase == "all" else "execution", 0, 0)
    report = progress["report"]
    saved_phase = report.get("phase")
    if saved_phase == "generating":
        return "generation", progress["generated"], progress["planned"]
    if saved_phase == "executing":
        return "execution", progress["terminal"], progress["planned"]
    if progress["generated"] < progress["planned"]:
        return "generation", progress["generated"], progress["planned"]
    if saved_phase == "finished" or requested_phase == "execute":
        return "execution", progress["terminal"], progress["planned"]
    return "generation", progress["generated"], progress["planned"]


def _read_state(path, manifest_hash, now):
    if not path.is_file():
        return {"version": STATE_VERSION, "manifest_hash": manifest_hash, "phase": "starting",
                "status": "running", "attempt": 0, "recovery_episodes": 0,
                "recovery_episode_active": False, "started_at": now, "updated_at": now,
                "estimated_finish_at": None, "estimate_basis": "waiting for progress"}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != STATE_VERSION or state.get("manifest_hash") != manifest_hash:
        raise StoreConflict("Supervisor state differs from its frozen inputs or policy")
    return state


def _write_state_unlocked(path, state, *, emit=True):
    state["updated_at"] = time.time()
    write_json(path, state)
    if emit:
        print("supervisor " + json.dumps(state, ensure_ascii=False, allow_nan=False), flush=True)


def _update_state(path, state, lock, *, emit=True, **fields):
    with lock:
        state.update(fields)
        _write_state_unlocked(path, state, emit=emit)


def _estimate_finish(progress, stage, baseline, started_at, now):
    if progress is None:
        return None, "waiting for the evaluation report"
    current = progress["generated"] if stage == "generation" else progress["terminal"]
    total = progress["planned"]
    remaining = max(0, total - current)
    completed_since_start = max(0, current - baseline)
    elapsed = max(0.0, now - started_at)
    if remaining == 0:
        return now, "all planned candidates are terminal"
    if completed_since_start == 0 or elapsed <= 0:
        return None, "waiting for a completed candidate in this phase"
    rate = completed_since_start / elapsed
    return now + remaining / rate, "linear estimate from candidate progress in this phase"


def _save_supervisor_manifest(status_dir, payload):
    path = status_dir / "supervisor-manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise StoreConflict("Supervisor inputs or recovery policy changed; use the original manifest")
    else:
        write_json(path, payload)
    return digest(payload)


def _validated_snapshot(runtime, source, training_run, run_path):
    try:
        with FileLock(run_path / "writer.lock", timeout=0):
            snapshot, _ = prepare_evaluation(runtime, source, training_run)
            manifest_path = run_path / "evaluation-manifest.json"
            if not manifest_path.is_file():
                raise StoreConflict("Evaluation manifest is missing; automatic recovery is unsafe")
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if digest(snapshot) != digest(saved):
                raise StoreConflict("Evaluation snapshot differs from the saved run")
            return snapshot
    except Timeout as exc:
        raise StoreConflict("Evaluation writer is active; recovery must wait for it to exit") from exc


def _verify_completed_run(runtime, source, training_run, run_path):
    try:
        with FileLock(run_path / "writer.lock", timeout=0):
            snapshot, _ = prepare_evaluation(runtime, source, training_run)
            saved_snapshot = json.loads((run_path / "evaluation-manifest.json").read_text(encoding="utf-8"))
            if digest(snapshot) != digest(saved_snapshot):
                raise StoreConflict("Completed run snapshot differs from current evaluation inputs")
            saved_report = json.loads((run_path / "evaluation-report.json").read_text(encoding="utf-8"))
            rebuilt = build_report(EvaluationView(run_path), "supervisor_completion_check")
            keys = ("run_id", "planned", "generated", "terminal", "finished", "halt_reason", "integrity_passed")
            if (not rebuilt["finished"] or not rebuilt["integrity_passed"] or rebuilt["halt_reason"]
                    or any(saved_report.get(key) != rebuilt.get(key) for key in keys)):
                raise StoreConflict("Evaluation database and saved completion report do not agree")
            return rebuilt
    except Timeout as exc:
        raise StoreConflict("Evaluation writer is active; completion cannot be verified") from exc


async def _run_phase(runtime, source, training_run, credentials_path, run_id, device, phase,
                     run_path, state, state_path, state_lock, progress_interval_seconds,
                     clock, evaluation):
    with state_lock:
        state["attempt"] += 1
        state["status"] = "running"
        state["requested_phase"] = phase
        state["phase"] = "generation" if phase == "all" else "execution"
        state["estimated_finish_at"] = None
        state["estimate_basis"] = "waiting for progress"
        _write_state_unlocked(state_path, state)
    baseline_progress = _read_progress(run_path)
    stage, baseline, _ = _phase_progress(baseline_progress, phase)
    phase_started_at = clock()
    stop_monitor = threading.Event()

    def monitor_progress():
        nonlocal stage, baseline, phase_started_at
        while not stop_monitor.wait(progress_interval_seconds):
            try:
                progress = _read_progress(run_path)
            except (OSError, ValueError, sqlite3.Error) as exc:
                _update_state(state_path, state, state_lock, monitor_error=f"{type(exc).__name__}: {exc}")
                return
            if progress is None:
                continue
            current_stage, current, total = _phase_progress(progress, phase)
            now = clock()
            if current_stage != stage:
                stage, baseline, phase_started_at = current_stage, current, now
            eta, basis = _estimate_finish(progress, stage, baseline, phase_started_at, now)
            fields = {"phase": stage, "status": "running", "progress": {"current": current, "total": total},
                      "estimated_finish_at": eta, "estimate_basis": basis,
                      "last_progress_estimate": {"estimated_finish_at": eta, "basis": basis},
                      "last_report": _status_summary(progress["report"])}
            _update_state(state_path, state, state_lock, **fields)

    monitor = threading.Thread(target=monitor_progress, name="evaluation-progress-monitor", daemon=False)
    monitor.start()
    try:
        return await evaluation(runtime, source, training_run, credentials_path,
                                run_id, device, phase=phase)
    except asyncio.CancelledError:
        raise
    finally:
        stop_monitor.set()
        monitor.join()


async def supervise_evaluation(runtime, source, training_run, credentials_path, run_id, device,
                               status_dir, *, recovery_settings=None,
                               max_recovery_episodes=DEFAULT_MAX_RECOVERY_EPISODES,
                               progress_interval_seconds=DEFAULT_PROGRESS_INTERVAL_SECONDS,
                               probe=None, sleep=asyncio.sleep, clock=time.time,
                               evaluation=None, recovery=None):
    if max_recovery_episodes < 1 or progress_interval_seconds <= 0:
        raise ValueError("Supervisor limits must be positive")
    source, training_run, credentials_path = map(Path, (source, training_run, credentials_path))
    status_dir = Path(status_dir).resolve()
    run_path = _run_path(runtime, run_id)
    try:
        status_dir.relative_to(run_path)
    except ValueError:
        pass
    else:
        raise ValueError("Supervisor state directory must be outside the evaluation run")
    status_dir.mkdir(parents=True, exist_ok=True)
    state_path = status_dir / "supervisor-state.json"
    policy = recovery_settings or AutoRecoveryConfig()
    manifest = {"version": STATE_VERSION, "run_id": run_id,
        "runtime_hash": digest(runtime.model_dump(mode="json")),
        "source": str(source.resolve()), "training_run": str(training_run.resolve()),
        "credentials_path": str(credentials_path.resolve()), "device": device,
        "run_path": str(run_path), "recovery_policy": policy.model_dump(mode="json"),
        "max_recovery_episodes": max_recovery_episodes}
    lock = FileLock(status_dir / "supervisor.lock", timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise StoreConflict("Another evaluation supervisor already owns this status directory") from exc
    state = None
    state_lock = threading.RLock()
    try:
        manifest_hash = _save_supervisor_manifest(status_dir, manifest)
        state = _read_state(state_path, manifest_hash, clock())
        if state["status"] == "finished":
            report = _verify_completed_run(runtime, source, training_run, run_path)
            _update_state(state_path, state, state_lock, status="finished", phase="finished",
                          last_report=_status_summary(report), estimate_basis="verified from evaluation database")
            return report, 0
        if state["status"] == "stopped":
            return state.get("last_report"), 1
        evaluate = evaluation or run_evaluation
        recover = recovery or automatic_recovery
        _update_state(state_path, state, state_lock, status="running", phase="starting")
        last_report = None
        while True:
            progress = _read_progress(run_path)
            halt_reason = progress["halt_reason"] if progress is not None else None
            if halt_reason == "consecutive_request_failures":
                if progress["generated"] != progress["planned"]:
                    raise StoreConflict("Cannot recover before every planned generation is persisted")
                if not state.get("recovery_episode_active", False):
                    if state["recovery_episodes"] >= max_recovery_episodes:
                        _update_state(state_path, state, state_lock, status="stopped", phase="stopped",
                                      stop_reason="recovery_episode_limit")
                        return last_report, 1
                    _update_state(state_path, state, state_lock,
                        recovery_episodes=state["recovery_episodes"] + 1, recovery_episode_active=True)
                _update_state(state_path, state, state_lock, phase="recovery", status="recovering",
                              estimated_finish_at=None, estimate_basis="persisted recovery probe schedule")
                snapshot = _validated_snapshot(runtime, source, training_run, run_path)

                def on_recovery_state(payload):
                    _update_state(state_path, state, state_lock, phase="recovery", status=payload["phase"],
                        probe_attempt=payload.get("probe"), max_probe_attempts=payload.get("max_probes"),
                        next_probe_at=payload.get("not_before"), estimated_finish_at=payload.get("not_before"),
                        estimate_basis="next persisted connectivity probe")

                recovery_options = {"on_state": on_recovery_state}
                if probe is not None:
                    recovery_options["probe"] = probe
                if sleep is not asyncio.sleep:
                    recovery_options["sleep"] = sleep
                if clock is not time.time:
                    recovery_options["clock"] = clock
                await recover(run_path, runtime, snapshot, policy, credentials_path, **recovery_options)
                _update_state(state_path, state, state_lock, recovery_episode_active=False,
                    status="recovered", phase="recovery", next_probe_at=None, estimated_finish_at=None)
                continue
            if halt_reason:
                _update_state(state_path, state, state_lock, status="stopped", phase="stopped",
                              stop_reason=halt_reason, estimated_finish_at=None)
                return last_report, 1
            phase = _next_phase(run_path)
            report, status = await _run_phase(runtime, source, training_run, credentials_path,
                run_id, device, phase, run_path, state, state_path, state_lock,
                progress_interval_seconds, clock, evaluate)
            last_report = report
            _update_state(state_path, state, state_lock, last_report=_status_summary(report), last_exit_code=status)
            if status == 0 and report.get("finished") and not report.get("halt_reason"):
                verified = _verify_completed_run(runtime, source, training_run, run_path)
                _update_state(state_path, state, state_lock, status="finished", phase="finished",
                              stop_reason=None, estimated_finish_at=clock(),
                              estimate_basis="verified from evaluation database",
                              last_report=_status_summary(verified))
                return verified, 0
            if report.get("halt_reason") == "consecutive_request_failures":
                _update_state(state_path, state, state_lock, status="waiting_recovery", phase="recovery",
                              estimated_finish_at=None, estimate_basis="waiting for recovery validation")
                continue
            stop_reason = report.get("halt_reason") or "evaluation_incomplete_without_recoverable_halt"
            _update_state(state_path, state, state_lock, status="stopped", phase="stopped",
                          stop_reason=stop_reason, estimated_finish_at=None)
            return report, 1
    except asyncio.CancelledError:
        if state is not None:
            current = _read_progress(run_path)
            fields = {"status": "interrupted", "phase": state.get("phase", "interrupted"),
                      "estimated_finish_at": None,
                      "estimate_basis": "resume from persisted run and probe state"}
            if current is not None:
                fields["last_report"] = _status_summary(current["report"])
            _update_state(state_path, state, state_lock, **fields)
        raise
    except Exception as exc:
        if state is not None:
            _update_state(state_path, state, state_lock, status="stopped", phase="stopped",
                          stop_reason=type(exc).__name__, error=str(exc), estimated_finish_at=None)
        raise
    finally:
        lock.release()
