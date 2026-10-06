import argparse
import asyncio
import json
import signal
import sys
import traceback
from pathlib import Path

from ocop.config import load_config
from ocop.evaluation_supervision import supervise_evaluation
from ocop.recovery import AutoRecoveryConfig


def main(argv=None):
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    parser = argparse.ArgumentParser(description="Run evaluation with bounded, persistent connection recovery")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--credentials", type=Path, default=Path("my_docs/my_docs/secrets/credentials.env"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--status-dir", type=Path)
    parser.add_argument("--recovery-config", type=Path)
    parser.add_argument("--max-recovery-episodes", type=int, default=5)
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        runtime = load_config(args.config)
        status_dir = args.status_dir or (Path(runtime.artifacts_dir) / "experiments" / f"{args.run_id}-supervision")
        recovery = AutoRecoveryConfig.model_validate_json(args.recovery_config.read_text(encoding="utf-8")) \
            if args.recovery_config else AutoRecoveryConfig()
        report, status = asyncio.run(supervise_evaluation(runtime, args.source,
            args.training_run, args.credentials, args.run_id, args.device, status_dir,
            recovery_settings=recovery, max_recovery_episodes=args.max_recovery_episodes,
            progress_interval_seconds=args.progress_interval_seconds))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ocop: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 2
    report_summary = {key: report[key] for key in ("run_id", "planned", "generated", "terminal",
        "finished", "integrity_passed", "engineering_passed", "policy_executor_evidence",
        "halt_reason", "cost_budget") if isinstance(report, dict) and key in report}
    print(json.dumps({"run_id": args.run_id, "status": status,
        "report": report_summary or None,
        "state": str(status_dir / "supervisor-state.json")}, ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
