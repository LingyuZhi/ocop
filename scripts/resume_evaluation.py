import argparse
import asyncio
import json
from pathlib import Path

import httpx

from ocop.config import RuntimeConfig, load_config
from ocop.evaluation import prepare_evaluation, run_evaluation
from ocop.evaluation_report import EvaluationView
from ocop.llm import load_credentials
from ocop.recovery import probe_models, recover_connections
from ocop.storage import ReadStore, StoreConflict
from ocop.trajectories import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, default=Path("my_docs/secrets/credentials.env"))
    parser.add_argument("--recover-only", action="store_true")
    args = parser.parse_args()
    path = args.run.resolve()
    view = ReadStore(path)
    if view.archive["config"]["purpose"] == "collection":
        if not args.recover_only:
            parser.error("Collection recovery requires --recover-only; resume its experiment separately")
        archived = view.archive["config"]
        config = RuntimeConfig.model_validate(archived["runtime"])
        if (Path(config.artifacts_dir) / "runs" / path.name).resolve() != path:
            raise StoreConflict("Run directory differs from its archived configuration")
        recovery = recover_connections(path, config, archived, credentials_path=args.credentials)
        print(json.dumps({"recovery": recovery["policy"], "failed_requests": len(recovery["evidence"]),
                          "historical_results": "preserved", "budgets": "cumulative"}), flush=True)
        return
    config = load_config(path / "config.json")
    archived = EvaluationView(path).archive["config"]
    source, training = Path(archived["source"]["path"]), Path(archived["training"]["path"])
    snapshot, _ = prepare_evaluation(config, source, training)
    if digest(snapshot) != digest(archived):
        raise StoreConflict("Evaluation inputs or implementation changed since the saved run")
    if (Path(config.artifacts_dir) / "runs" / path.name).resolve() != path:
        raise StoreConflict("Run directory differs from its archived configuration")
    load_credentials(args.credentials, {config.worker_model.api_key_env, config.finalizer.api_key_env})
    recovery = recover_connections(path, config, snapshot, credentials_path=args.credentials)
    print(json.dumps({"recovery": recovery["policy"], "failed_requests": len(recovery["evidence"]),
                      "historical_results": "preserved", "budgets": "cumulative"}), flush=True)
    if args.recover_only:
        return
    report, status = asyncio.run(run_evaluation(config, source, training, args.credentials,
                                              path.name, "cuda:0", phase="execute"))
    print(json.dumps({key: report[key] for key in ("run_id", "generated", "terminal", "planned",
        "finished", "integrity_passed", "engineering_passed", "halt_reason")}, indent=2), flush=True)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
