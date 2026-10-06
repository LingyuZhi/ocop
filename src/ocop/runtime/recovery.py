import asyncio
import json
import time
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field, model_validator

from ocop.collection.verification import verify as verify_collection
from ocop.runtime.config import StrictModel
from ocop.diagnostics.services import provenance
from ocop.evaluation.report import EvaluationView, audit_generation_tokens, build_report
from ocop.runtime.llm import load_credentials
from ocop.runtime.storage import RunStore, StoreConflict
from ocop.runtime.storage import digest, file_hash


CONNECTION_ERRORS = {"ConnectError", "RemoteProtocolError", "ReadError", "WriteError",
                     "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutError"}


class AutoRecoveryConfig(StrictModel):
    version: Literal["ocop.auto_recovery.v1"] = "ocop.auto_recovery.v1"
    initial_delay_seconds: float = Field(default=120.0, ge=0)
    max_delay_seconds: float = Field(default=900.0, gt=0)
    max_probe_attempts: int = Field(default=96, gt=0)

    @model_validator(mode="after")
    def ordered_delays(self):
        if self.initial_delay_seconds > self.max_delay_seconds:
            raise ValueError("Initial delay exceeds maximum")
        return self

    def delay(self, number):
        return min(self.max_delay_seconds, self.initial_delay_seconds * 2 ** min(number, 20))


def probe_models(config, *, providers, credentials_path):
    models = [model for model in (config.worker_model, config.finalizer, config.strong_model)
              if model.provider in providers]
    credentials = load_credentials(credentials_path, {m.api_key_env for m in models if m.provider == "deepseek"})
    evidence = []
    with httpx.Client(timeout=15, follow_redirects=False) as client:
        for url in sorted({model.base_url + "/models" for model in models}):
            model = next(m for m in models if m.base_url + "/models" == url)
            headers = {"Authorization": f"Bearer {credentials[model.api_key_env]}"} if model.provider == "deepseek" else {}
            response = client.get(url, headers=headers)
            response.raise_for_status()
            available = {entry["id"] for entry in response.json()["data"]}
            required = {model.model_id for model in models if model.base_url + "/models" == url}
            if not required <= available:
                raise StoreConflict("Configured execution model missing from service catalog")
            evidence.append({"url": url, "provider": model.provider, "http_status": response.status_code,
                "models": sorted(required), "checked_at": time.time()})
    return evidence


def validate_recovery_records(path, snapshot):
    if snapshot["purpose"] == "collection":
        report = verify_collection(path, allow_pending=True)
        passed = all(value for key, value in report["checks"].items() if key != "not_halted")
    elif snapshot["purpose"] == "graph_replication":
        from ocop.collection.replication import replication_report

        passed = replication_report(path)["integrity_passed"]
    else:
        view = EvaluationView(path)
        report = build_report(view, "connection_recovery_check")
        passed = report["integrity_passed"] and report["generated"] == report["planned"]
        if passed:
            audit_generation_tokens(view)
    if not passed:
        raise StoreConflict("Recovery requires verified execution and source records")


def retryable_attempt(attempt, allow_http):
    if attempt["state"] != "infra_failed":
        return False
    outcome = json.loads(attempt["result_json"] or "{}")
    code = attempt["http_status"]
    if code is None:
        return attempt["body_blob"] is None and outcome.get("error") in CONNECTION_ERRORS
    return allow_http and (code in {408, 429} or 500 <= code < 600)


def recovery_evidence(store, config, *, allow_http=False):
    run = store.rows("run")[0]
    if run["halt_reason"] != "consecutive_request_failures":
        raise StoreConflict("Recovery requires a consecutive request failure halt")
    requests = store.rows("requests")
    if any(r["state"] not in {"completed", "incomplete", "infra_failed", "fatal", "pending"} for r in requests):
        raise StoreConflict("Recovery requires settled requests")
    attempts = {}
    for row in store.rows("attempts"):
        attempts.setdefault(row["request_id"], []).append(row)
    if any(a["finished"] is None or a["state"] in {"in_flight", "response_saved"}
           for rows in attempts.values() for a in rows):
        raise StoreConflict("Recovery requires finalized request attempts")
    for request in requests:
        if request["state"] == "pending" and (request["result_json"] is not None or any(
                not retryable_attempt(a, allow_http) for a in attempts.get(request["id"], []))):
            raise StoreConflict("Pending request has unverified outcomes")
    settled = [r for r in requests if r["state"] != "pending"]
    if any(not attempts.get(r["id"]) for r in settled):
        raise StoreConflict("Settled request lacks attempt evidence")
    cutoff = max((r["created"] for r in store.rows("records") if r["kind"] == "recovery"), default=0)
    ordered = sorted((r for r in settled if attempts[r["id"]][-1]["finished"] > cutoff),
                     key=lambda r: attempts[r["id"]][-1]["finished"])
    streak, triggering = [], []
    for request in ordered:
        if request["state"] == "infra_failed":
            streak.append(request)
            if len(streak) >= config.requests.consecutive_exhausted_request_limit:
                triggering = list(streak)
        else:
            streak = []
    if not triggering:
        raise StoreConflict("Failure evidence does not meet the configured circuit threshold")
    evidence, providers = [], set()
    models = (config.worker_model, config.finalizer, config.strong_model)
    for request in triggering:
        rows = attempts[request["id"]]
        if len(rows) != config.requests.max_attempts or not all(retryable_attempt(a, allow_http) for a in rows):
            raise StoreConflict("Failure streak is not exhausted transient failure evidence")
        spec = json.loads(store.read_blob(request["spec_blob"]))
        if not any(spec["provider"] == m.provider and spec["url"] == m.base_url + "/chat/completions"
                   and spec["body"]["model"] == m.model_id for m in models):
            raise StoreConflict("Failed request is not a configured model")
        providers.add(spec["provider"])
        evidence.append({"request_id": request["id"], "attempt_ids": [a["id"] for a in rows]})
    return evidence, providers


def apply_recovery(store, evidence, network, *, policy, extra=None):
    run = store.rows("run")[0]
    key = "connection-recovery:" + digest(evidence)
    recovery = {"policy": policy, "previous_halt_reason": run["halt_reason"],
        "previous_failure_streak": run["failure_streak"], "evidence": evidence, "connectivity": network,
        "config_hash": run["config_hash"], "implementation_sha256": file_hash(Path(__file__)),
        "historical_results": "preserved", "budgets": "cumulative", **(extra or {})}
    store.recover_circuit(key, recovery)
    return recovery


def export_store(store):
    temporary = store.path / "records.jsonl.tmp"
    store.export_jsonl(temporary)
    temporary.replace(store.path / "records.jsonl")


def recover_connections(path, config, snapshot, *, probe=probe_models,
                        credentials_path=Path("my_docs/secrets/credentials.env")):
    if config.model_dump(mode="json") != snapshot["runtime"]:
        raise StoreConflict("Recovery configuration differs from archived runtime")
    with RunStore(path.parent, snapshot, run_id=path.name, provenance=snapshot.get("environment", provenance())) as store:
        validate_recovery_records(path, snapshot)
        evidence, providers = recovery_evidence(store, config)
        network = probe(config, providers=providers, credentials_path=credentials_path)
        if not network:
            raise StoreConflict("Recovery requires successful service probes")
        recovery = apply_recovery(store, evidence, network, policy="transport_error_recovery.v3")
        export_store(store)
        return recovery


async def automatic_recovery(path, config, snapshot, settings, credentials_path, *,
                             probe=probe_models, sleep=asyncio.sleep, clock=time.time, on_state=None):
    if config.model_dump(mode="json") != snapshot["runtime"]:
        raise StoreConflict("Recovery configuration differs from archived runtime")
    with RunStore(path.parent, snapshot, run_id=path.name, provenance=provenance()) as store:
        validate_recovery_records(path, snapshot)
        evidence, providers = recovery_evidence(store, config, allow_http=True)
        episode = digest(evidence)
        policy_id = store.put_record("recovery_policy", digest(settings.model_dump()), settings.model_dump())
        try:
            for number in range(settings.max_probe_attempts):
                key = f"{episode}:{number}"
                result = store.get_record("recovery_probe_result", key)
                if result is None:
                    plan = store.get_record("recovery_probe_plan", key)
                    if plan is None:
                        previous = store.get_record("recovery_probe_result", f"{episode}:{number - 1}")
                        base = previous["finished_at"] if previous else clock()
                        plan = {"episode": episode, "number": number, "not_before": base + settings.delay(number),
                                "providers": sorted(providers), "config_hash": store.rows("run")[0]["config_hash"]}
                        store.put_record("recovery_probe_plan", key, plan, parents={"policy": policy_id})
                    started = store.get_record("recovery_probe_started", key)
                    if started is not None:
                        result = {"status": "interrupted", "error": "probe_interrupted", "finished_at": clock()}
                    else:
                        if on_state:
                            on_state({"phase": "cooldown", "run_id": path.name, "probe": number + 1,
                                      "max_probes": settings.max_probe_attempts, "not_before": plan["not_before"]})
                        await sleep(max(0, plan["not_before"] - clock()))
                        store.put_record("recovery_probe_started", key, {"started_at": clock()})
                        try:
                            network = await asyncio.to_thread(probe, config, providers=providers,
                                                              credentials_path=credentials_path)
                            if not network:
                                raise StoreConflict("Empty probe evidence")
                        except (httpx.TransportError, TimeoutError) as exc:
                            result = {"status": "retryable", "error": type(exc).__name__, "finished_at": clock()}
                        except httpx.HTTPStatusError as exc:
                            code = exc.response.status_code
                            result = {"status": "retryable" if code in {408, 429} or 500 <= code < 600 else "fatal",
                                      "http_status": code, "error": "probe_http_error", "finished_at": clock()}
                        except (ValueError, KeyError, TypeError) as exc:
                            result = {"status": "fatal", "error": type(exc).__name__, "finished_at": clock()}
                        else:
                            result = {"status": "success", "connectivity": network, "finished_at": clock()}
                    store.put_record("recovery_probe_result", key, result)
                if result["status"] == "success":
                    recovery = apply_recovery(store, evidence, result["connectivity"], policy=settings.version,
                                              extra={"episode": episode, "probe_key": key, "recovery_policy_id": policy_id})
                    if on_state:
                        on_state({"phase": "recovered", "run_id": path.name, "probe": number + 1})
                    return recovery
                if result["status"] == "fatal":
                    raise StoreConflict("Recovery probe encountered a non-transient error")
            raise StoreConflict("Automatic recovery probe budget exhausted")
        finally:
            export_store(store)
