import argparse
import asyncio
import json
import signal
import sys
from dataclasses import asdict
from pathlib import Path

from ocop import __version__


CONFIG = Path("configs/prototype.json")
CREDENTIALS = Path("my_docs/secrets/credentials.env")


def build_parser():
    parser = argparse.ArgumentParser(prog="ocop", description="OCOP research prototype")
    parser.add_argument("--version", action="version", version=f"ocop {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    graph = commands.add_parser("graph", help="Inspect and replay graph trajectories")
    graph_commands = graph.add_subparsers(dest="graph_command", required=True)
    graph_commands.add_parser("contract")
    graph_commands.add_parser("schema")
    validate = graph_commands.add_parser("validate")
    validate.add_argument("content", type=Path)
    validate.add_argument("--reasoning-file", type=Path)
    smoke = commands.add_parser("smoke", help="Check real remote model requests")
    smoke.add_argument("--config", type=Path, default=CONFIG)
    smoke.add_argument("--credentials", type=Path, default=CREDENTIALS)
    smoke.add_argument("--run-id")
    execute = commands.add_parser("execute", help="Execute one graph repeat")
    execute.add_argument("--task", type=Path, required=True)
    execute.add_argument("--trajectory", type=Path, required=True)
    execute.add_argument("--config", type=Path, default=CONFIG)
    execute.add_argument("--credentials", type=Path, default=CREDENTIALS)
    execute.add_argument("--run-id")
    execute.add_argument("--repeat-id", required=True)
    collect = commands.add_parser("collect", help="Collect proposals and measured labels")
    collect.add_argument("--config", type=Path, default=CONFIG)
    collect.add_argument("--credentials", type=Path, default=CREDENTIALS)
    collect.add_argument("--run-id")
    collect.add_argument("--manifest-only", action="store_true")
    collect.add_argument("--resume-inflight-budget", action="store_true")
    collect.add_argument("--resume-after-topup", action="store_true")
    prepare = commands.add_parser("prepare-sft", help="Prepare and audit raw SFT data")
    prepare.add_argument("--config", type=Path, default=CONFIG)
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    update_check = commands.add_parser("verify-sft", help="Check actual updates, saves and reloads")
    update_check.add_argument("--config", type=Path, default=CONFIG)
    update_check.add_argument("--data", type=Path, required=True)
    update_check.add_argument("--output", type=Path, required=True)
    update_check.add_argument("--device", default="cuda:0")
    update_check.add_argument("--reload-checkpoint", type=Path, help=argparse.SUPPRESS)
    train = commands.add_parser("train", help="Train a full-parameter policy model")
    train.add_argument("--config", type=Path)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--resume-checkpoint", type=Path)
    train.add_argument("--verify-final", action="store_true", help=argparse.SUPPRESS)
    evaluate = commands.add_parser("evaluate", help="Compare base and trained policies")
    evaluate.add_argument("--source", type=Path, required=True)
    evaluate.add_argument("--training-run", type=Path, required=True)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--phase", choices=("all", "generate", "execute"), default="all")
    evaluate.add_argument("--credentials", type=Path, default=CREDENTIALS)
    evaluate.add_argument("--resume-inflight-budget", action="store_true")
    evaluate.add_argument("--resume-after-topup", action="store_true")
    evaluate.add_argument("--supervise", action="store_true", help="Persist progress and recover connections")
    evaluate.add_argument("--status-dir", type=Path)
    evaluate.add_argument("--recovery-config", type=Path)
    evaluate.add_argument("--max-recovery-episodes", type=int, default=5)
    evaluate.add_argument("--progress-interval-seconds", type=float, default=30.0)
    report = commands.add_parser("report", help="Rebuild evaluation or baseline reports")
    report.add_argument("--run", type=Path, required=True)
    runs = commands.add_parser("runs", help="Inspect, export or recover run records")
    run_commands = runs.add_subparsers(dest="run_command", required=True)
    show = run_commands.add_parser("show")
    show.add_argument("path", type=Path)
    export = run_commands.add_parser("export")
    export.add_argument("path", type=Path)
    export.add_argument("--output", type=Path, required=True)
    recover = run_commands.add_parser("recover")
    recover.add_argument("--run", type=Path, required=True)
    recover.add_argument("--credentials", type=Path, default=CREDENTIALS)
    recover.add_argument("--recover-only", action="store_true")
    verify = commands.add_parser("verify", help="Verify artifacts or run an executor smoke")
    verifications = verify.add_subparsers(dest="kind", required=True)
    for name in ("collection", "training", "evaluation", "experiment"):
        check = verifications.add_parser(name)
        check.add_argument("path", type=Path)
        check.add_argument("--output", type=Path)
        if name == "collection":
            check.add_argument("--allow-pending", action="store_true")
        if name == "evaluation":
            check.add_argument("--baseline", type=Path)
    executor = verifications.add_parser("executor", help="Run fresh and resumed executor repeats")
    executor.add_argument("--config", type=Path, default=CONFIG)
    executor.add_argument("--task", type=Path, default=Path("examples/execution/task.json"))
    executor.add_argument("--trajectory", type=Path, default=Path("examples/graph/valid.json"))
    executor.add_argument("--credentials", type=Path, default=CREDENTIALS)
    executor.add_argument("--run-id", required=True)
    diagnose = commands.add_parser("diagnose", help="Inspect models, graphs or policy results")
    diagnostics = diagnose.add_subparsers(dest="kind", required=True)
    model = diagnostics.add_parser("model")
    model.add_argument("--model-path", type=Path, required=True)
    model.add_argument("--output", type=Path, required=True)
    model.add_argument("--decode", action="store_true")
    graphs = diagnostics.add_parser("graphs")
    for name in ("collection", "evaluation", "samples", "output"):
        graphs.add_argument(f"--{name}", type=Path, required=True)
    policy = diagnostics.add_parser("policy")
    policy.add_argument("--runs", type=Path, nargs="+", required=True)
    policy.add_argument("--output", type=Path, required=True)
    policy.add_argument("--tasks", nargs="+", default=[])
    experiment = commands.add_parser("experiment", help="Run baseline, expansion or SFT workflows")
    experiments = experiment.add_subparsers(dest="kind", required=True)
    baseline = experiments.add_parser("baseline")
    baseline.add_argument("--source", type=Path, required=True)
    baseline.add_argument("--config", type=Path, default=Path("configs/fixed-graph-baseline.json"))
    baseline.add_argument("--run-id", required=True)
    baseline.add_argument("--credentials", type=Path, default=CREDENTIALS)
    baseline.add_argument("--prepare-only", action="store_true")
    baseline.add_argument("--resume-after-topup", action="store_true")
    expansion = experiments.add_parser("expansion")
    expansion.add_argument("--config", type=Path, required=True)
    expansion.add_argument("--run-id", required=True)
    expansion.add_argument("--credentials", type=Path, default=CREDENTIALS)
    expansion.add_argument("--prepare-only", action="store_true")
    expansion.add_argument("--resume-after-topup", action="store_true")
    expansion.add_argument("--recovery-config", type=Path, default=Path("configs/auto-recovery.json"))
    pilot = experiments.add_parser("pilot")
    for name in ("config", "source", "data", "output", "experiment-dir"):
        pilot.add_argument(f"--{name}", type=Path, required=True)
    pilot.add_argument("--run-id", required=True)
    pilot.add_argument("--gpu-wait-hours", type=float, default=24)
    pilot.add_argument("--preflight-only", action="store_true")
    return parser


def dispatch(args):
    if args.command == "graph":
        from ocop.graph import contract_hash, load_contract, replay, trajectory_schema

        if args.graph_command == "schema":
            return trajectory_schema(), 0
        if args.graph_command == "contract":
            return {**load_contract(), "schema": trajectory_schema(), "hash": contract_hash()}, 0
        parsed = replay(args.content.read_text(), reasoning=args.reasoning_file.read_text() if args.reasoning_file else None)
        return {"valid": parsed.valid, **asdict(parsed)}, 0 if parsed.valid else 1
    if args.command == "runs":
        from ocop.runtime.storage import export_run, inspect_run

        if args.run_command == "recover":
            from ocop.runtime.recovery import resume_run

            return resume_run(args.run, args.credentials, args.recover_only)
        if args.run_command == "export":
            export_run(args.path, args.output)
            return {"output": str(args.output)}, 0
        return inspect_run(args.path), 0
    if args.command == "verify":
        return verify_artifacts(args)
    if args.command == "diagnose":
        if args.kind == "model":
            from ocop.diagnostics.model import inspect_model

            return inspect_model(args.model_path, args.output, args.decode), 0
        if args.kind == "graphs":
            from ocop.diagnostics.graphs import analyze_graphs

            return analyze_graphs(args.collection, args.evaluation, args.samples, args.output), 0
        from ocop.diagnostics.policy import diagnose_policy

        return diagnose_policy(args.runs, args.output, args.tasks), 0
    if args.command == "experiment":
        return run_experiment(args)
    if args.command == "report":
        from ocop.runtime.storage import ReadStore

        if ReadStore(args.run).archive["config"].get("purpose") == "fixed_graph_baseline":
            from filelock import FileLock
            from torch.utils.tensorboard import SummaryWriter
            from ocop.evaluation.baseline import save_report

            with FileLock(args.run / "writer.lock", timeout=0), SummaryWriter(str(args.run / "tensorboard"), purge_step=0) as writer:
                result = save_report(args.run, writer)
            return result, 0 if result["finished"] and result["integrity_passed"] and result["tensorboard_passed"] else 1
        from ocop.evaluation.report import report_evaluation

        result = report_evaluation(args.run)
        return result, 0 if result["integrity_passed"] else 1
    if args.command == "train":
        from ocop.training.engine import train

        return train(args.config, args.data, args.output, args.device, args.resume_checkpoint, args.verify_final)
    from ocop.runtime.config import load_config

    if args.command == "evaluate":
        from ocop.evaluation.pipeline import run_evaluation

        path = args.config or CONFIG
        runtime = load_config(path)
        if args.supervise:
            from ocop.evaluation.supervision import supervise_evaluation
            from ocop.runtime.recovery import AutoRecoveryConfig

            if args.phase != "all" or args.resume_inflight_budget or args.resume_after_topup:
                raise ValueError("Supervision controls phases and recovery; manual phase or recovery flags cannot be combined")

            def interrupt(signum, frame):
                raise KeyboardInterrupt

            for signum in (signal.SIGTERM, signal.SIGHUP):
                signal.signal(signum, interrupt)
            recovery = AutoRecoveryConfig.model_validate_json(args.recovery_config.read_text()) if args.recovery_config else AutoRecoveryConfig()
            status_dir = args.status_dir or (Path(runtime.artifacts_dir) / "experiments" / f"{args.run_id}-supervision")
            return asyncio.run(supervise_evaluation(runtime, args.source, args.training_run, args.credentials,
                args.run_id, args.device, status_dir, recovery_settings=recovery,
                max_recovery_episodes=args.max_recovery_episodes, progress_interval_seconds=args.progress_interval_seconds))
        return asyncio.run(run_evaluation(runtime, args.source, args.training_run, args.credentials,
            args.run_id, args.device, phase=args.phase, resume_inflight_budget=args.resume_inflight_budget,
            resume_after_topup=args.resume_after_topup))
    runtime = load_config(args.config)
    if args.command == "prepare-sft":
        from ocop.training.data import prepare_sft

        return prepare_sft(runtime, args.source, args.output)
    if args.command == "verify-sft":
        from ocop.training.updates import verify_sft

        return verify_sft(runtime, args.data, args.output, args.device, reload_checkpoint=args.reload_checkpoint, config_path=args.config)
    if args.command == "collect":
        from ocop.collection.pipeline import run_collection

        return asyncio.run(run_collection(runtime, args.credentials, args.run_id, manifest_only=args.manifest_only,
            resume_inflight_budget=args.resume_inflight_budget, resume_after_topup=args.resume_after_topup))
    if args.command == "execute":
        from ocop.execution.single import run_execution

        return asyncio.run(run_execution(runtime, args.task, args.trajectory, args.credentials, args.run_id, args.repeat_id))
    from ocop.diagnostics.services import smoke_services

    return asyncio.run(smoke_services(runtime, args.credentials, args.run_id))


def verify_artifacts(args):
    from ocop.runtime.storage import write_json

    if args.kind in {"training", "evaluation"} and args.output and args.output.resolve().is_relative_to(args.path.resolve()):
        raise ValueError("Verification output must be outside the source run")
    if args.kind == "collection":
        from ocop.collection.verification import verify

        result = verify(args.path, allow_pending=args.allow_pending)
        passed = result["passed"]
    elif args.kind == "training":
        from ocop.training.verification import verify

        result = verify(args.path)
        passed = result["passed"]
    elif args.kind == "evaluation":
        from ocop.evaluation.verification import verify, verify_fixed_graph
        from ocop.runtime.storage import ReadStore

        purpose = ReadStore(args.path).archive["config"].get("purpose")
        if purpose == "fixed_graph_baseline":
            if args.baseline:
                raise ValueError("Fixed graph verification does not accept a policy history baseline")
            result = verify_fixed_graph(args.path)
        else:
            result = verify(args.path, args.baseline)
        passed = result["passed"]
    elif args.kind == "experiment":
        from filelock import FileLock
        from ocop.collection.expansion import verify_suite

        with FileLock(args.path / "writer.lock", timeout=0):
            result = verify_suite(args.path)
        passed = result["integrity_passed"]
    else:
        from ocop.execution.verification import verify

        return asyncio.run(verify(args))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    return result, 0 if passed else 1


def run_experiment(args):
    if args.kind == "pilot":
        from ocop.experiments.pilot import run_pilot

        return run_pilot(args), 0
    from ocop.runtime.config import load_config

    if args.kind == "baseline":
        from ocop.evaluation.baseline import Settings, prepare, run

        settings = Settings.model_validate_json(args.config.read_text())
        runtime, snapshot = prepare(args.source, settings)
        return asyncio.run(run(runtime, snapshot, args.run_id, args.credentials,
            prepare_only=args.prepare_only, after_topup=args.resume_after_topup))
    from ocop.collection.expansion import ExpansionConfig, supervise_expansion
    from ocop.runtime.recovery import AutoRecoveryConfig

    settings = ExpansionConfig.model_validate_json(args.config.read_text())
    recovery = AutoRecoveryConfig.model_validate_json(args.recovery_config.read_text())
    result, status = asyncio.run(supervise_expansion(settings, args.run_id, args.credentials,
        recovery_settings=recovery, prepare_only=args.prepare_only, after_topup=args.resume_after_topup))
    return result, status


def main(argv=None):
    args = build_parser().parse_args(argv)
    from ocop.runtime.storage import StoreConflict
    from filelock import Timeout

    try:
        output, status = dispatch(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, StoreConflict, Timeout) as exc:
        print(f"ocop: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return status
