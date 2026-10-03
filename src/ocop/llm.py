import asyncio
import json
import os
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from uuid import UUID

import httpx
from dotenv import dotenv_values

from ocop.config import ModelConfig, RequestConfig
from ocop.storage import RunStore, StoreConflict


def load_credentials(path: Path, names: set[str]) -> dict[str, str]:
    values = dotenv_values(path, interpolate=False) if path.is_file() else {}
    credentials = {name: os.environ.get(name) or values.get(name) for name in names}
    missing = [name for name, value in credentials.items() if not value]
    if missing:
        raise ValueError(f"Missing credential variables: {', '.join(sorted(missing))}")
    return credentials


def normalize_response(body: bytes, status_code: int, *, catalog: bool) -> dict:
    outcome = {"status": "infra_failed", "error": None, "content": None, "reasoning": None, "usage": None}
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        outcome["error"] = "invalid_response_json"
        payload = {}
    if not isinstance(payload, dict):
        outcome["error"] = "invalid_response_shape"
        payload = {}
    error = payload.get("error")
    metadata = error.get("metadata") if isinstance(error, dict) else None
    if (isinstance(metadata, dict) and str(error.get("code")) == "402"
            and (status_code == 402 or 200 <= status_code < 300)
            and metadata.get("reason") == "in_flight_budget_exhausted"
            and metadata.get("limit_source") == "openrouter_in_flight_budget"):
        outcome["error"] = "in_flight_budget_exhausted"
        return outcome
    if not 200 <= status_code < 300:
        outcome.update(status="infra_failed" if status_code in {408, 429} or status_code >= 500 else "fatal", error=f"http_{status_code}")
        return outcome
    if payload.get("error"):
        error = payload["error"]
        code = error.get("code") if isinstance(error, dict) else None
        outcome.update(status="fatal" if str(code) in {"400", "401", "402", "403", "404", "422"} else "infra_failed", error="provider_error")
        return outcome
    if catalog:
        if isinstance(payload.get("data"), list):
            outcome.update(status="completed", error=None)
        else:
            outcome["error"] = outcome["error"] or "missing_model_catalog"
        return outcome
    outcome.update(response_id=payload.get("id"), response_model=payload.get("model"),
                   provider=payload.get("provider"), system_fingerprint=payload.get("system_fingerprint"), usage=payload.get("usage"))
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        outcome["error"] = outcome["error"] or "invalid_choices"
        return outcome
    choice = choices[0]
    if choice.get("error"):
        outcome["error"] = "provider_choice_error"
        return outcome
    message = choice.get("message")
    if not isinstance(message, dict):
        outcome["error"] = "missing_message"
        return outcome
    finish = choice.get("finish_reason")
    outcome.update(content=message.get("content"), reasoning=message.get("reasoning_content", message.get("reasoning")),
                   finish_reason=finish, native_finish_reason=choice.get("native_finish_reason"))
    if finish in {"error", "insufficient_system_resource", "aborted"}:
        outcome["error"] = f"provider_{finish}"
    elif finish == "stop" and isinstance(outcome["content"], str) and not message.get("tool_calls"):
        outcome.update(status="completed", error=None)
    else:
        outcome.update(status="incomplete", error=f"incomplete_{finish or 'missing_finish_reason'}")
    return outcome


def budget_retry_deadline(attempt: dict) -> float:
    received = attempt["finished"]
    value = json.loads(attempt["headers_json"]).get("retry-after", "120")
    try:
        delay = int(value)
    except (ValueError, TypeError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - received
        except (ValueError, TypeError, OverflowError):
            delay = 120
    return received + max(0, min(delay, 300))


def recover_inflight_budget(store: RunStore, *, after_topup: bool = False):
    run = store.rows("run")[0]
    if run["halt_reason"] is None:
        return
    if run["halt_reason"] != "fatal_request_error":
        raise StoreConflict("Recovery requires a fatal request halt")
    failures = [row for row in store.rows("requests") if row["state"] == "fatal"]
    evidence = []
    topup_provider = None
    for request in failures:
        attempts = store.attempts_for(request["id"])
        attempt = attempts[-1] if attempts else None
        if not attempt or not attempt["body_blob"]:
            raise StoreConflict("Recovery refused: missing saved fatal response")
        body = store.read_blob(attempt["body_blob"])
        budget_error = normalize_response(body, attempt["http_status"], catalog=False)["error"] == "in_flight_budget_exhausted"
        request_topup_provider = None
        if after_topup:
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeError):
                payload = None
            error = payload.get("error") if isinstance(payload, dict) else None
            metadata = error.get("metadata") if isinstance(error, dict) else None
            spec = json.loads(store.read_blob(request["spec_blob"]))
            openrouter_credits_error = (spec.get("provider") == "openrouter" and attempt["http_status"] == 402
                and isinstance(metadata, dict) and str(error.get("code")) == "402"
                and metadata.get("limit_source") == "openrouter_credits")
            message = error.get("message") if isinstance(error, dict) else None
            request_id_valid = False
            prefix = "Insufficient Balance (request_id: "
            if isinstance(message, str) and message.startswith(prefix) and message.endswith(")"):
                try:
                    UUID(message[len(prefix):-1])
                except ValueError:
                    pass
                else:
                    request_id_valid = True
            deepseek_balance_error = (spec.get("provider") == "deepseek" and attempt["http_status"] == 402
                and isinstance(error, dict) and error.get("code") == "invalid_request_error"
                and error.get("type") == "unknown_error" and request_id_valid)
            if openrouter_credits_error:
                request_topup_provider = "openrouter"
            elif deepseek_balance_error:
                request_topup_provider = "deepseek"
            else:
                request_topup_provider = None
            if request_topup_provider:
                if topup_provider and topup_provider != request_topup_provider:
                    raise StoreConflict("Recovery refused: fatal requests contain mixed provider top-up errors")
                topup_provider = request_topup_provider
        if not (budget_error or (after_topup and request_topup_provider)):
            raise StoreConflict("Recovery refused: fatal request is not a verified eligible budget or credits error")
        evidence.append({"request_id": request["id"], "attempt_id": attempt["id"], "body_blob": attempt["body_blob"]})
    if not evidence:
        raise StoreConflict("Recovery requires saved budget error evidence")
    if after_topup and topup_provider == "deepseek":
        policy, prefix = "deepseek_topup.v1", "deepseek-topup:"
    else:
        policy = "openrouter_topup.v1" if after_topup else "inflight_budget_retry.v1"
        prefix = "openrouter-topup:" if after_topup else "inflight-budget:"
    store.put_record("recovery", prefix + ":".join(item["attempt_id"] for item in evidence), {
        "policy": policy, "previous_halt_reason": run["halt_reason"],
        "previous_failure_streak": run["failure_streak"], "evidence": evidence,
        "config_hash": run["config_hash"], "historical_results": "preserved", "budgets": "cumulative"})
    with store.transaction():
        store.db.execute("UPDATE run SET halt_reason=NULL, failure_streak=0")


class RequestLimits:
    def __init__(self, concurrency: int):
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.cooldowns: dict[str, float] = {}


class RequestRunner:
    def __init__(self, store: RunStore, config: RequestConfig, credentials: dict[str, str], *, transport=None,
                 shared_limits: RequestLimits | None = None):
        self.store = store
        self.config = config
        self.credentials = credentials
        self.client = httpx.AsyncClient(timeout=config.timeout_seconds, transport=transport, follow_redirects=False)
        limits = shared_limits or RequestLimits(config.concurrency)
        if limits.concurrency != config.concurrency:
            raise ValueError("Shared request concurrency differs from configuration")
        self.semaphore = limits.semaphore
        self._locks: dict[str, asyncio.Lock] = {}
        self._cooldowns = limits.cooldowns
        requests = {row["id"]: row for row in store.rows("requests")}
        for attempt in store.rows("attempts"):
            if attempt["body_blob"] and attempt["http_status"] in {200, 402}:
                if normalize_response(store.read_blob(attempt["body_blob"]), attempt["http_status"], catalog=False)["error"] == "in_flight_budget_exhausted":
                    spec = json.loads(store.read_blob(requests[attempt["request_id"]]["spec_blob"]))
                    origin = str(httpx.URL(spec["url"]).copy_with(path="/", query=None))
                    self._cooldowns[origin] = max(self._cooldowns.get(origin, 0), budget_retry_deadline(attempt))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def call(self, model: ModelConfig, *, key: str, messages: list[dict] | None = None, owner_id: str | None = None) -> dict:
        catalog = messages is None
        method = "GET" if catalog else "POST"
        url = model.base_url + ("/models" if catalog else "/chat/completions")
        origin = str(httpx.URL(url).copy_with(path="/", query=None))
        body = None if catalog else model.completion_body(messages)
        spec = {"method": method, "url": url, "body": body, "provider": model.provider, "api_key_env": model.api_key_env}
        async with self._locks.setdefault(key, asyncio.Lock()):
            request = self.store.request(key, spec, owner_id)
            if request["result_json"] is not None:
                return json.loads(request["result_json"])
            self.store.ensure_active()
            while True:
                attempts = self.store.attempts_for(request["id"])
                saved = attempts[-1] if attempts and attempts[-1]["state"] == "response_saved" else None
                if saved is None and len(attempts) >= self.config.max_attempts:
                    return self.store.exhaust_uncertain(request["id"], self.config.consecutive_exhausted_request_limit)
                if saved is not None:
                    attempt = saved
                    outcome = self._from_saved(attempt, catalog)
                else:
                    if attempts:
                        await asyncio.sleep(min(self.config.retry_backoff_seconds * 2 ** (len(attempts) - 1), 30))
                    async with self.semaphore:
                        while self._cooldowns.get(origin, 0) > time.time():
                            await asyncio.sleep(self._cooldowns[origin] - time.time())
                        self.store.ensure_active()
                        attempt = self.store.start_attempt(request["id"])
                        started = time.monotonic()
                        try:
                            async with asyncio.timeout(self.config.timeout_seconds):
                                response = await self.client.request(method, url, json=body,
                                    headers={"Authorization": f"Bearer {self.credentials[model.api_key_env]}"})
                        except (httpx.TransportError, TimeoutError) as exc:
                            outcome = {"status": "infra_failed", "error": type(exc).__name__, "remote_outcome_unknown": True,
                                       "usage": None, "elapsed_seconds": time.monotonic() - started}
                        else:
                            raw = response.content
                            for secret in self.credentials.values():
                                raw = raw.replace(secret.encode(), b"[REDACTED]")
                            headers = {name: response.headers[name] for name in ("x-request-id", "request-id", "x-provider-name", "retry-after", "content-type") if name in response.headers}
                            self.store.save_response(attempt["id"], body=raw, status=response.status_code, headers=headers,
                                                     elapsed=time.monotonic() - started, redacted=raw != response.content)
                            attempt = self.store.attempts_for(request["id"])[-1]
                            outcome = self._from_saved(attempt, catalog)
                            if outcome["error"] == "in_flight_budget_exhausted":
                                self._cooldowns[origin] = max(self._cooldowns.get(origin, 0), outcome["retry_not_before"])
                outcome.update(request_id=request["id"], attempt_id=attempt["id"])
                terminal = outcome["status"] != "infra_failed" or attempt["number"] >= self.config.max_attempts
                self.store.finish_attempt(attempt, outcome, terminal=terminal, failure_limit=self.config.consecutive_exhausted_request_limit)
                if terminal:
                    return outcome

    def _from_saved(self, attempt: dict, catalog: bool) -> dict:
        outcome = normalize_response(self.store.read_blob(attempt["body_blob"]), attempt["http_status"], catalog=catalog)
        outcome.update(body_blob=attempt["body_blob"], http_status=attempt["http_status"],
                       response_headers=json.loads(attempt["headers_json"]), elapsed_seconds=attempt["elapsed"],
                       body_redacted=json.loads(attempt["result_json"] or "{}").get("body_redacted", False))
        if outcome["error"] == "in_flight_budget_exhausted":
            outcome["retry_not_before"] = budget_retry_deadline(attempt)
        return outcome
