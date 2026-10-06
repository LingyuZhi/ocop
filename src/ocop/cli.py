import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ocop import __version__
from ocop.graph import contract_hash, load_contract, replay, trajectory_schema
from ocop.runtime.config import load_config
from ocop.diagnostics.services import smoke_services
from ocop.execution.single import run_execution
from ocop.collection.pipeline import run_collection
from ocop.runtime.storage import StoreConflict, export_run, inspect_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ocop", description="OCOP research prototype")
    parser.add_argument("--version", action="version", version=f"ocop {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    graph = commands.add_parser("graph", help="Inspect the contract and replay trajectories")
    graph_commands = graph.add_subparsers(dest="graph_command", required=True)
    graph_commands.add_parser("contract", help="Print role prompts, rules, schema and content hash")
    graph_commands.add_parser("schema", help="Print the JSON Schema for trajectory content")
    validate = graph_commands.add_parser("validate", help="Validate content and return every accepted graph state")
    validate.add_argument("content", type=Path)
    validate.add_argument("--reasoning-file", type=Path)
    smoke = commands.add_parser("smoke", help="Verify real services with persisted requests")
    smoke.add_argument("--config", type=Path, default=Path("config/prototype.json"))
    smoke.add_argument("--credentials", type=Path, default=Path("my_docs/secrets/credentials.env"))
    smoke.add_argument("--run-id")
    execute = commands.add_parser("execute", help="Execute one graph repeat and score its final answer")
    execute.add_argument("--task", type=Path, required=True)
    execute.add_argument("--trajectory", type=Path, required=True)
    execute.add_argument("--config", type=Path, default=Path("config/prototype.json"))
    execute.add_argument("--credentials", type=Path, default=Path("my_docs/secrets/credentials.env"))
    execute.add_argument("--run-id")
    execute.add_argument("--repeat-id", required=True)
    collect = commands.add_parser("collect", help="Collect GSM8K organizations, executions and labels")
    collect.add_argument("--config", type=Path, default=Path("config/prototype.json"))
    collect.add_argument("--credentials", type=Path, default=Path("my_docs/secrets/credentials.env"))
    collect.add_argument("--run-id")
    collect.add_argument("--manifest-only", action="store_true")
    collect.add_argument("--resume-inflight-budget", action="store_true",
                         help="Resume a saved, verified in-flight budget halt with cumulative budgets")
    collect.add_argument("--resume-after-topup", action="store_true",
                         help="Resume verified OpenRouter credits failures after replenishing credits")
    runs = commands.add_parser("runs", help="Inspect or export existing run records")
    run_commands = runs.add_subparsers(dest="run_command", required=True)
    show = run_commands.add_parser("show")
    show.add_argument("path", type=Path)
    export = run_commands.add_parser("export")
    export.add_argument("path", type=Path)
    export.add_argument("--output", type=Path, required=True)
    prepare_sft = commands.add_parser("prepare-sft", help="Prepare and audit raw SFT training samples")
    prepare_sft.add_argument("--source", type=Path, required=True)
    prepare_sft.add_argument("--output", type=Path, required=True)
    prepare_sft.add_argument("--config", type=Path, default=Path("config/prototype.json"))
    verify_sft = commands.add_parser("verify-sft", help="Verify full-parameter updates and checkpoint reload")
    verify_sft.add_argument("--data", type=Path, required=True)
    verify_sft.add_argument("--output", type=Path, required=True)
    verify_sft.add_argument("--config", type=Path, default=Path("config/prototype.json"))
    verify_sft.add_argument("--device", default="cuda:0")
    verify_sft.add_argument("--reload-checkpoint", type=Path, help=argparse.SUPPRESS)
    train = commands.add_parser("train", help="Run full-parameter SFT with resumable checkpoints")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--config", type=Path)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--resume-checkpoint", type=Path)
    train.add_argument("--verify-final", action="store_true", help=argparse.SUPPRESS)
    evaluate = commands.add_parser("evaluate", help="Compare base and final policy checkpoints across target outcomes")
    evaluate.add_argument("--source", type=Path, required=True)
    evaluate.add_argument("--training-run", type=Path, required=True)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--phase", choices=("all", "generate", "execute"), default="all")
    evaluate.add_argument("--credentials", type=Path, default=Path("my_docs/secrets/credentials.env"))
    evaluate.add_argument("--resume-inflight-budget", action="store_true")
    evaluate.add_argument("--resume-after-topup", action="store_true")
    report = commands.add_parser("report", help="Rebuild evaluation reports and TensorBoard from stored records")
    report.add_argument("--run", type=Path, required=True)
    for name in ("prepare",):
        commands.add_parser(name, help="Reserved pipeline command (not implemented yet)")
    args = parser.parse_args(argv)
    if args.command in {"evaluate", "report"}:
        from filelock import Timeout

        from ocop.evaluation.pipeline import run_evaluation
        from ocop.evaluation.report import report_evaluation

        try:
            if args.command == "report":
                output = report_evaluation(args.run)
                status = 0 if output["integrity_passed"] else 1
            else:
                config_path = args.config
                if config_path is None:
                    archived = Path("artifacts/runs") / args.run_id / "config.json"
                    config_path = archived if archived.is_file() else Path("config/prototype.json")
                output, status = asyncio.run(run_evaluation(load_config(config_path), args.source, args.training_run,
                    args.credentials, args.run_id, args.device, phase=args.phase,
                    resume_inflight_budget=args.resume_inflight_budget, resume_after_topup=args.resume_after_topup))
        except (OSError, ValueError, StoreConflict, Timeout) as exc:
            print(f"ocop: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({key: output[key] for key in ("run_id", "planned", "generated", "terminal", "finished",
            "integrity_passed", "engineering_passed", "policy_executor_evidence", "halt_reason")}, ensure_ascii=False, indent=2))
        return status
    if args.command == "train":
        from filelock import Timeout

        from ocop.training.engine import train

        try:
            output, status = train(args.config, args.data, args.output, args.device,
                                   args.resume_checkpoint, args.verify_final)
        except (OSError, ValueError, Timeout) as exc:
            print(f"ocop: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return status
    if args.command in {"prepare-sft", "verify-sft"}:
        try:
            if args.command == "prepare-sft":
                from ocop.training.data import prepare_sft

                output, status = prepare_sft(load_config(args.config), args.source, args.output)
            else:
                from ocop.training.updates import verify_sft

                output, status = verify_sft(load_config(args.config), args.data, args.output,
                    args.device, reload_checkpoint=args.reload_checkpoint, config_path=args.config)
        except (OSError, ValueError, StoreConflict) as exc:
            print(f"ocop: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return status
    if args.command == "collect":
        try:
            output, status = asyncio.run(run_collection(load_config(args.config), args.credentials, args.run_id,
                                                       manifest_only=args.manifest_only,
                                                       resume_inflight_budget=args.resume_inflight_budget,
                                                       resume_after_topup=args.resume_after_topup))
        except (OSError, ValueError, StoreConflict) as exc:
            print(f"ocop: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return status
    if args.command == "execute":
        try:
            output, status = asyncio.run(run_execution(load_config(args.config), args.task, args.trajectory,
                                                      args.credentials, args.run_id, args.repeat_id))
        except (OSError, ValueError, StoreConflict) as exc:
            print(f"ocop: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return status
    if args.command == "runs":
        if args.run_command == "export":
            export_run(args.path, args.output)
        else:
            print(json.dumps(inspect_run(args.path), ensure_ascii=False, indent=2))
        return 0
    if args.command == "smoke":
        config = load_config(args.config)
        output, status = asyncio.run(smoke_services(config, args.credentials, args.run_id))
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return status
    if args.command != "graph":
        parser.error(f"{args.command} is not implemented yet.")
    if args.graph_command == "schema":
        output = trajectory_schema()
        status = 0
    elif args.graph_command == "contract":
        output = {**load_contract(), "schema": trajectory_schema(), "hash": contract_hash()}
        status = 0
    else:
        try:
            content = args.content.read_text(encoding="utf-8")
            reasoning = args.reasoning_file.read_text(encoding="utf-8") if args.reasoning_file else None
        except (OSError, UnicodeError) as exc:
            print(f"ocop: {exc}", file=sys.stderr)
            return 2
        result = replay(content, reasoning=reasoning)
        output = {"valid": result.valid, **asdict(result)}
        status = 0 if result.valid else 1
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return status
