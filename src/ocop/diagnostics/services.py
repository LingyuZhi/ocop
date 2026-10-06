import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path

from ocop import __version__
from ocop.runtime.config import RuntimeConfig
from ocop.graph import load_contract, replay, trajectory_schema
from ocop.runtime.llm import RequestRunner, load_credentials
from ocop.runtime.storage import RunStore


def provenance() -> dict:
    source = Path(__file__).resolve().parents[1]
    hasher = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        hasher.update(str(path.relative_to(source)).encode())
        hasher.update(path.read_bytes())
    return {
        "ocop_version": __version__, "source_sha256": hasher.hexdigest(), "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("httpx", "filelock", "jsonschema", "networkx", "pydantic", "python-dotenv")},
    }


async def smoke_services(config: RuntimeConfig, credentials_path: Path, run_id: str | None) -> tuple[dict, int]:
    models = {"strong": config.strong_model, "worker": config.worker_model, "finalizer": config.finalizer}
    credentials = load_credentials(credentials_path, {model.api_key_env for model in models.values()})
    snapshot = {"purpose": "service_smoke", "runtime": config.model_dump(mode="json")}
    with RunStore(Path(config.artifacts_dir) / "runs", snapshot, run_id=run_id, provenance=provenance()) as store:
        (store.path / "config.json").write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
        report = {"run_id": store.run_id, "path": str(store.path), "models": {}, "passed": False}
        task = store.put_record("task", "service-smoke", {"split": "smoke", "question": "A box contains 2 red balls and 3 blue balls. How many balls are in the box?"})
        task_payload = json.loads(store.read_blob(next(row["payload_blob"] for row in store.rows("records") if row["id"] == task)))
        problem = task_payload["question"]
        contract = {**load_contract(), "schema": trajectory_schema()}
        prompts = {
            "strong": [{"role": "system", "content": json.dumps(contract, ensure_ascii=False)}, {"role": "user", "content": problem}],
            "worker": [{"role": "system", "content": load_contract()["roles"]["Solver"]}, {"role": "user", "content": problem}],
        }
        async with RequestRunner(store, config.requests, credentials) as runner:
            for label, model in models.items():
                catalog = await runner.call(model, key=f"catalog:{model.provider}")
                if catalog["status"] != "completed":
                    report["models"][label] = {"passed": False, "catalog_status": catalog["status"], "error": catalog["error"]}
                    break
                entries = json.loads(store.read_blob(catalog["body_blob"]))["data"]
                found = next((entry for entry in entries if entry.get("id") == model.model_id), None)
                if found is None:
                    report["models"][label] = {"passed": False, "error": "requested_model_not_in_catalog", "model_id": model.model_id}
                    break
                if label == "finalizer":
                    worker = report["models"].get("worker", {})
                    if not worker.get("passed"):
                        break
                    worker_body = json.loads(store.read_blob(worker["body_blob"]))
                    worker_text = worker_body["choices"][0]["message"]["content"]
                    prompts[label] = [
                        {"role": "system", "content": "Solve the problem using the supplied worker output. End with a final line formatted as #### followed by a signed decimal integer or decimal number. Do not include units or separators on that line."},
                        {"role": "user", "content": json.dumps({"question": problem, "worker_0": worker_text})},
                    ]
                proposal = store.put_record("proposal" if label == "strong" else "smoke_call", label, {"slot": 0, "model_role": label}, parents={"task": task})
                outcome = await runner.call(model, key=f"smoke:{label}", messages=prompts[label], owner_id=proposal)
                passed = outcome["status"] == "completed"
                item = {key: outcome.get(key) for key in ("status", "error", "request_id", "attempt_id", "response_id", "response_model", "provider", "finish_reason", "usage", "body_blob", "elapsed_seconds")}
                item.update(model_id=model.model_id, content_chars=len(outcome.get("content") or ""), reasoning_chars=len(outcome.get("reasoning") or ""))
                if passed and outcome.get("response_model") != model.model_id:
                    passed = False
                    item["verification_error"] = "response_model_mismatch"
                if passed and model.provider_order and outcome.get("provider") not in model.provider_order:
                    passed = False
                    item["verification_error"] = "response_provider_mismatch"
                if label == "strong" and passed:
                    parsed = replay(outcome["content"], reasoning=outcome.get("reasoning"))
                    store.put_record("trajectory", label, {"valid": parsed.valid, **asdict(parsed)}, parents={"proposal": proposal})
                    item["graph_valid"] = parsed.valid
                    item["graph_error"] = asdict(parsed.error) if parsed.error else None
                    passed = parsed.valid and bool(outcome.get("reasoning"))
                item["passed"] = passed
                report["models"][label] = item
                if outcome["status"] == "fatal":
                    break
        report["passed"] = len(report["models"]) == 3 and all(item["passed"] for item in report["models"].values())
        report["request_count"] = len(store.rows("requests"))
        report["attempt_count"] = len(store.rows("attempts"))
        store.put_record("smoke_report", "services", report, parents={"task": task})
        store.export_jsonl(store.path / "records.jsonl")
        (store.path / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report, 0 if report["passed"] else 1
