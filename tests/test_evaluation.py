import asyncio
import importlib.util
import json
from collections import Counter
from pathlib import Path

import httpx
import pytest
from transformers import AutoTokenizer

import ocop.evaluation as evaluation
from ocop.benchmark import BenchmarkConfig, build_manifest
from ocop.config import load_config
from ocop.evaluation_report import EvaluationView, audit_metrics, report_evaluation
from ocop.executor import executor_config, executor_hash
from ocop.policy_generation import candidate_seed, generation_config, parse_generation
from ocop.storage import RunStore, StoreConflict
from ocop.trajectories import policy_messages
from test_collection import SOURCE, VALID_GRAPH, reply


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("/data1/zhilingyu/models/Qwen3.5-2B", local_files_only=True)


def runtime(tmp_path):
    config = load_config(ROOT / "config/prototype.json")
    config.artifacts_dir = str(tmp_path / "artifacts")
    config.benchmark.update(train_tasks=1, eval_tasks=1)
    config.requests.retry_backoff_seconds = 0.0
    return config


def snapshot(config, tokenizer):
    settings = evaluation.EvaluationConfig.model_validate(config.evaluation)
    manifest = build_manifest(SOURCE, BenchmarkConfig.model_validate(config.benchmark), config.seed)
    smoke = {"candidate_id": "source-candidate", "task_id": manifest["tasks"][0]["task_id"], "z": 1.0}
    return {"purpose": "evaluation", "version": "ocop.evaluation.v1", "runtime": config.model_dump(),
        "settings": settings.model_dump(), "task_manifest": manifest, "training_smoke": smoke,
        "candidates": evaluation.candidate_plan(manifest, smoke, settings, config.seed),
        "executor": executor_config(config), "executor_hash": executor_hash(config),
        "generation_config": generation_config(config.policy, tokenizer).to_dict(),
        "models": {"base": {"path": "base"}, "last_checkpoint": {"path": "checkpoint"}},
        "source": {"path": "source"}, "training": {"path": "training"}, "environment": {}}


def raw_output(tokenizer, question, z, content=VALID_GRAPH, *, eos=True, thinking=True):
    text = ("Design reasoning</think>\n\n" if thinking else "Still reasoning") + content
    if eos:
        text += tokenizer.eos_token
    ids = tokenizer.encode(text, add_special_tokens=False)
    inputs = tokenizer.apply_chat_template(policy_messages(question, z), tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=True)
    close = tokenizer.convert_tokens_to_ids("</think>")
    boundary = ids.index(close) + 1 if close in ids else len(ids)
    return {"text": text, "token_ids": ids, "input_ids": inputs, "input_tokens": len(inputs),
        "output_tokens": len(ids), "reasoning_tokens": boundary, "content_tokens": len(ids) - boundary,
        "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id, "reached_eos": eos,
        "finish_reason": "eos" if eos else "length", "seconds": 1.0, "tokens_per_second": len(ids),
        "peak_allocated_bytes": 1024}


@pytest.fixture
def environment(monkeypatch, tokenizer):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(evaluation, "prepare_evaluation", lambda config, *args: (snapshot(config, tokenizer), tokenizer))
    calls = []
    loaded = []

    class Generator:
        def __init__(self, path, device, config, reference_tokenizer):
            self.path = path
            loaded.append(path)

        def generate(self, question, z, seed):
            calls.append((self.path, question, z, seed))
            return raw_output(tokenizer, question, z)

        def close(self):
            pass

    return Generator, calls, loaded


def run(config, environment, handler=reply, **kwargs):
    return asyncio.run(evaluation.run_evaluation(config, Path("source"), Path("training"), Path("absent.env"),
        "eval-test", "cuda:0", generator_factory=environment[0], transport=httpx.MockTransport(handler), **kwargs))


def path(config):
    return Path(config.artifacts_dir) / "runs/eval-test"


def test_candidate_matrix_split_and_seed_pairing(tmp_path, tokenizer):
    config = load_config(ROOT / "config/prototype.json")
    plan = snapshot(config, tokenizer)["candidates"]
    assert len(plan) == 38
    assert sum(c["split"] == "eval" for c in plan) == 36
    assert sum(c["split"] == "train_smoke" for c in plan) == 2
    for task in {c["task_id"] for c in plan}:
        assert len({c["seed"] for c in plan if c["task_id"] == task}) == 1
    assert candidate_seed(42, "q", 0) != candidate_seed(42, "q", 1)
    assert candidate_seed(42, "q", 0) != candidate_seed(43, "q", 0)
    config = snapshot(runtime(tmp_path), tokenizer)
    manifest = config["task_manifest"]
    bad_smoke = {**config["training_smoke"], "task_id": manifest["tasks"][-1]["task_id"]}
    with pytest.raises(ValueError, match="training task"):
        evaluation.candidate_plan(manifest, bad_smoke, evaluation.EvaluationConfig.model_validate(config["settings"]), 42)


def test_pilot_plans_all_fifty_holdout_tasks():
    config = load_config(ROOT / "config/gsm8k-pilot-sft.json")
    source = [SOURCE[index % len(SOURCE)] for index in range(7500)]
    manifest = build_manifest(source, BenchmarkConfig.model_validate(config.benchmark), config.seed)
    smoke = {"task_id": manifest["tasks"][0]["task_id"], "z": 1.0}
    settings = evaluation.EvaluationConfig.model_validate(config.evaluation)
    plan = evaluation.candidate_plan(manifest, smoke, settings, config.seed)
    holdout = [candidate for candidate in plan if candidate["split"] == "holdout"]
    train_ids = {task["task_id"] for task in manifest["tasks"] if task["split"] == "train"}
    holdout_ids = {task["task_id"] for task in manifest["tasks"] if task["split"] == "holdout"}
    assert len(plan) == 302 and len(holdout) == 300
    assert {candidate["task_id"] for candidate in holdout} == holdout_ids
    assert not train_ids & holdout_ids
    assert Counter((candidate["model"], candidate["z"]) for candidate in holdout) == Counter(
        {(model, z): 50 for model in settings.models for z in settings.target_z})
    assert all(len({candidate["seed"] for candidate in holdout if candidate["task_id"] == task}) == 1
               for task in holdout_ids)
    assert {candidate["task_id"] for candidate in plan if candidate["split"] == "train_smoke"} <= train_ids


def test_evaluation_split_is_explicit_and_preserves_default_serialization(tmp_path, tokenizer):
    config = runtime(tmp_path)
    settings = evaluation.EvaluationConfig.model_validate(config.evaluation)
    assert settings.task_split == "eval" and settings.model_dump() == config.evaluation
    with pytest.raises(ValueError):
        evaluation.EvaluationConfig.model_validate({**config.evaluation, "task_split": "train"})
    config.evaluation["task_split"] = "holdout"
    with pytest.raises(ValueError, match="split is empty"):
        snapshot(config, tokenizer)


def test_execution_subsample_keeps_all_generations_and_paired_tasks(tmp_path, environment):
    config = runtime(tmp_path)
    config.benchmark["holdout_tasks"] = 2
    config.evaluation.update(task_split="holdout", execution_task_limit=1)
    report, status = run(config, environment)
    assert status == 0 and report["finished"] and report["integrity_passed"]
    assert report["generated"] == 14 and len(environment[1]) == 14
    holdout = [row for row in report["candidates"] if row["split"] == "holdout"]
    selected = [row for row in holdout if row["execution_selected"]]
    assert len(selected) == 6 and len({row["task_id"] for row in selected}) == 1
    assert all(row["status"] == "execution_not_selected" and row["label"] is None and not row["repeats"]
               for row in holdout if not row["execution_selected"])
    assert report["request_count"] == 200
    groups = [group for group in report["groups"] if group["split"] == "holdout"]
    assert all(group["planned"] == 2 and group["execution_planned"] == 1
               and group["correct_execution_coverage_denominator"] == 1
               and group["correct_execution_coverage"] == 1 for group in groups)
    again, status = run(config, environment, phase="execute")
    assert status == 0 and again["request_count"] == 200 and len(environment[1]) == 14


def test_evaluation_cost_cap_stops_without_faking_completion(tmp_path, environment):
    config = runtime(tmp_path)
    config.evaluation["cost_budget"] = {"max_usd": 0.0001, "input_per_million": 0.15, "output_per_million": 0.6}
    report, status = run(config, environment, lambda request: pytest.fail("Unexpected paid request"))
    assert status == 1 and not report["finished"] and report["integrity_passed"]
    assert report["halt_reason"] == "evaluation_cost_budget_exhausted"
    assert report["cost_budget"]["cap_respected"] and report["attempt_count"] == 0
    assert not report["engineering_passed"]


def test_subsample_interruption_recovers_without_executing_excluded_graphs(tmp_path, environment, monkeypatch):
    config = runtime(tmp_path)
    config.benchmark["holdout_tasks"] = 2
    config.evaluation.update(task_split="holdout", execution_task_limit=1)
    original = RunStore.put_record
    interrupted = []
    def save(store, kind, key, payload, **kwargs):
        if kind == "candidate_result" and payload["status"] == "execution_not_selected" and not interrupted:
            interrupted.append(key)
            raise RuntimeError("interrupt before exclusion is saved")
        return original(store, kind, key, payload, **kwargs)
    monkeypatch.setattr(RunStore, "put_record", save)
    with pytest.raises(RuntimeError, match="before exclusion"):
        run(config, environment)
    report, status = run(config, environment)
    assert status == 0 and report["finished"] and report["integrity_passed"]
    assert report["request_count"] == 200 and len(environment[1]) == 14
    assert all(not row["repeats"] for row in report["candidates"] if not row["execution_selected"])


def vllm_runtime(config, tmp_path):
    config.evaluation.update(backend="vllm", vllm={"python": "worker-python", "environment_lock": "worker-lock",
        "cache_dir": str(tmp_path / "cache"), "gpu_uuids": [f"GPU-{index}" for index in range(6)]})
    return config


def vllm_mock_pool(tokenizer, calls, *, interrupt=False):
    class Pool:
        def __init__(self, settings, snapshot, log_dir):
            self.settings = settings

        def generate(self, jobs):
            for job in reversed(jobs):
                if interrupt and calls:
                    raise RuntimeError("interrupt parallel generation")
                calls.append(job["key"])
                ids = tokenizer.encode("Design reasoning</think>\n\n" + VALID_GRAPH + tokenizer.eos_token, add_special_tokens=False)
                gpu = (0 if job["model"] == "base" else 3) + (job["ordinal"] - 1) % 3
                yield job, {"input_ids": job["input_ids"], "token_ids": ids, "finish_reason": "stop", "seconds": 1.0,
                            "gpu_uuid": self.settings.gpu_uuids[gpu], "stop_reason": tokenizer.eos_token_id}

        def close(self):
            pass
    return Pool


def test_vllm_parallel_generation_preserves_raw_tokens_and_public_execution(tmp_path, environment, tokenizer):
    config = vllm_runtime(runtime(tmp_path), tmp_path)
    calls = []
    pool = vllm_mock_pool(tokenizer, calls)
    report, status = run(config, environment, vllm_pool_factory=pool)
    assert status == 0 and report["finished"] and report["integrity_passed"]
    assert len(calls) == 8 and report["request_count"] == 200
    view = EvaluationView(path(config))
    raw = [row["payload"] for row in view.record_items("generation")]
    assert {row["backend_metrics"]["gpu_uuid"] for row in raw} == {f"GPU-{i}" for i in range(6)}
    assert all(row["token_ids"][-1] == tokenizer.eos_token_id and row["reached_eos"] for row in raw)
    assert all(row["backend"] == "vllm" and row["peak_allocated_bytes"] is None for row in raw)
    repeated, status = run(config, environment, phase="execute", vllm_pool_factory=pool)
    assert status == 0 and len(calls) == 8 and repeated["request_count"] == 200


def test_vllm_interruption_reuses_saved_candidates(tmp_path, environment, tokenizer):
    config = vllm_runtime(runtime(tmp_path), tmp_path)
    calls = []
    with pytest.raises(RuntimeError, match="parallel generation"):
        run(config, environment, vllm_pool_factory=vllm_mock_pool(tokenizer, calls, interrupt=True))
    assert len(calls) == 1
    first = calls[0]
    report, status = run(config, environment, vllm_pool_factory=vllm_mock_pool(tokenizer, calls))
    assert status == 0 and report["finished"] and report["integrity_passed"]
    assert len(calls) == 8 and calls.count(first) == 1
    results = [row["payload"]["status"] for row in EvaluationView(path(config)).record_items("generation_attempt_result")]
    assert results.count("interrupted") == 7


def test_holdout_execution_report_and_resume_preserve_source_split(tmp_path, environment):
    config = runtime(tmp_path)
    config.benchmark["holdout_tasks"] = 1
    config.evaluation["task_split"] = "holdout"
    report, status = run(config, environment)
    assert status == 0 and report["finished"] and report["integrity_passed"]
    view = EvaluationView(path(config))
    tasks = {task["task_id"]: task for task in view.archive["config"]["task_manifest"]["tasks"]}
    assert len(environment[1]) == 8 and report["request_count"] == 200
    assert all(tasks[candidate["task_id"]]["split"] == candidate["split"]
               for candidate in report["candidates"] if candidate["split"] == "holdout")
    assert not any(tasks[candidate["task_id"]]["split"] == "eval" for candidate in report["candidates"])
    assert "holdout；共 1 道题" in (path(config) / "evaluation-report.md").read_text()
    repeated, status = run(config, environment)
    assert status == 0 and repeated["request_count"] == 200 and len(environment[1]) == 8


@pytest.mark.parametrize(("content", "eos", "thinking", "error"), [
    (VALID_GRAPH, True, True, None), (VALID_GRAPH, False, True, "truncated"),
    ("", True, False, "missing_thinking_end"), ("{", True, True, "invalid_json"),
    ('{"version":"ocop.graph.v1","steps":[]}', True, True, "missing_stop")])
def test_generation_classification(tokenizer, content, eos, thinking, error):
    raw = raw_output(tokenizer, "q", 0.5, content, eos=eos, thinking=thinking)
    actual = parse_generation(raw)
    assert actual["error"] == error
    assert actual["eligible_for_execution"] is (error is None)
    if content == VALID_GRAPH:
        assert actual["valid_graph"]
    assert raw["text"].endswith(tokenizer.eos_token) is eos


def test_generation_config_is_shared_and_explicit(tmp_path, tokenizer):
    config = generation_config(runtime(tmp_path).policy, tokenizer)
    assert config.do_sample and config.temperature == 0.7 and config.top_p == 0.8
    assert config.top_k == 0 and config.num_beams == 1 and config.repetition_penalty == 1
    assert config.max_new_tokens == 8192 and config.eos_token_id == tokenizer.eos_token_id


def test_complete_run_duplicate_graphs_and_resume_do_not_repeat_calls(tmp_path, environment, tokenizer):
    config = runtime(tmp_path)
    remote = []
    def handler(request):
        remote.append(json.loads(request.content))
        return reply(request)
    report, status = run(config, environment, handler)
    assert status == 0 and report["engineering_passed"] and report["finished"]
    assert report["integrity_passed"] and all(report["checks"].values())
    assert len(environment[1]) == 8 and environment[2] == ["base", "checkpoint"]
    assert len(remote) == 8 * 5 * 5
    assert all("ReferenceSecret" not in json.dumps(request) and '"z"' not in request["messages"][1]["content"] for request in remote)
    assert all(c["observed_successes"] == 5 for c in report["candidates"])
    assert all(item["same_graph"] for item in report["comparisons"])
    assert all(g["success_rate"] == 1 and g["correct_execution_coverage"] == 1 for g in report["groups"])
    again, status = run(config, environment, handler)
    assert status == 0 and again["request_count"] == report["request_count"]
    assert len(environment[1]) == 8 and len(remote) == 200
    assert audit_metrics(path(config), again)["passed"]
    report_files = report_evaluation(path(config))
    assert report_files["groups"] == report["groups"]
    assert (path(config) / "evaluation-report.md").exists()
    assert all("ReferenceSecret" not in tokenizer.decode(EvaluationView(path(config)).get_record("generation", c["candidate_id"])["input_ids"])
               for c in report["candidates"])


def test_generate_only_then_execute_never_regenerates(tmp_path, environment):
    config = runtime(tmp_path)
    generated, status = run(config, environment, phase="generate")
    assert status == 0 and generated["generated"] == 8 and generated["request_count"] == 0
    assert generated["terminal"] == 0 and not generated["finished"]
    executed, status = run(config, environment, phase="execute")
    assert status == 0 and executed["finished"] and len(environment[1]) == 8
    changed = config.model_copy(deep=True)
    changed.policy["temperature"] = 0.5
    with pytest.raises(StoreConflict, match="configuration"):
        run(changed, environment)


def test_execute_requires_generations(tmp_path, environment):
    with pytest.raises(ValueError, match="requires all planned"):
        run(runtime(tmp_path), environment, phase="execute")


def test_zero_legal_graphs_keep_groups_and_no_remote_calls(tmp_path, environment, monkeypatch, tokenizer):
    monkeypatch.setattr(environment[0], "generate", lambda self, q, z, seed: raw_output(tokenizer, q, z, "{"))
    report, status = run(runtime(tmp_path), environment, lambda r: pytest.fail("Unexpected execution"))
    assert status == 0 and report["finished"] and report["integrity_passed"]
    assert not report["engineering_passed"] and not report["policy_executor_evidence"]
    assert len(report["groups"]) == 8
    assert all(g["execution_status"] == "无可执行图" and g["success_rate"] is None for g in report["groups"])
    assert all(g["correct_execution_coverage"] == 0 for g in report["groups"])


@pytest.mark.parametrize(("answer", "reason"), [("#### 6", "wrong_answer"), ("five", "format_error")])
def test_answer_failures_are_complete_and_do_not_expand_budget(tmp_path, environment, answer, reason):
    report, status = run(runtime(tmp_path), environment, lambda r: reply(r, answer=answer))
    assert status == 0 and report["engineering_passed"]
    assert report["request_count"] == 200
    assert all(g["success_rate"] == 0 for g in report["groups"])
    assert all(r["score"]["reason"] == reason for c in report["candidates"] for r in c["repeats"])


def test_infrastructure_failures_use_bounded_repeats(tmp_path, environment):
    config = runtime(tmp_path)
    config.requests.consecutive_exhausted_request_limit = 1000
    report, status = run(config, environment, lambda r: httpx.Response(503, text="unavailable"))
    assert status == 0 and report["finished"] and report["integrity_passed"]
    assert all(len(c["repeats"]) == 8 and c["label"]["mean_outcome"] is None for c in report["candidates"])
    assert all(g["success_rate"] is None and g["incomplete_labels"] == 1 for g in report["groups"])
    assert not report["policy_executor_evidence"]


def recovery_module():
    spec = importlib.util.spec_from_file_location("resume_evaluation", ROOT / "scripts/resume_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("initial_disconnects", [0, 2])
def test_connection_recovery_preserves_history_and_budgets(tmp_path, environment, initial_disconnects):
    config = runtime(tmp_path)
    config.requests.consecutive_exhausted_request_limit = 2

    calls = 0

    def offline(request):
        nonlocal calls
        calls += 1
        if calls <= initial_disconnects:
            raise httpx.RemoteProtocolError("connection interrupted", request=request)
        raise httpx.ConnectError("connection unavailable", request=request)

    report, status = run(config, environment, offline)
    assert status == 1 and report["halt_reason"] == "consecutive_request_failures"
    before = EvaluationView(path(config))
    generation_count = len(environment[1])
    module = recovery_module()

    def failed_probe(config, **kwargs):
        raise httpx.ConnectError("still unavailable")

    with pytest.raises(httpx.ConnectError):
        module.recover_connections(path(config), config, before.archive["config"], probe=failed_probe)
    assert EvaluationView(path(config)).tables["run"][0]["halt_reason"] == "consecutive_request_failures"
    assert not EvaluationView(path(config)).record_items("recovery")
    module.recover_connections(path(config), config, before.archive["config"],
        probe=lambda config, **kwargs: [{"http_status": 200, "models": [config.worker_model.model_id]}])
    recovered = EvaluationView(path(config))
    assert recovered.tables["attempts"] == before.tables["attempts"]
    assert recovered.tables["requests"] == before.tables["requests"]
    assert all(recovered.by_id[key] == value for key, value in before.by_id.items())
    assert len(recovered.record_items("recovery")) == 1
    report, status = run(config, environment, phase="execute")
    assert status == 0 and report["engineering_passed"]
    assert len(environment[1]) == generation_count
    assert all(len(c["repeats"]) <= 8 for c in report["candidates"])
    assert sum(r["status"] == "infra_failed" for c in report["candidates"] for r in c["repeats"]) == 2
    assert all(len(rows) <= config.requests.max_attempts for rows in
        [EvaluationView(path(config)).attempts_for(r["id"]) for r in recovered.tables["requests"]])


@pytest.mark.parametrize("http_status", [402, 503])
def test_connection_recovery_rejects_http_errors(tmp_path, environment, http_status):
    config = runtime(tmp_path)
    config.requests.consecutive_exhausted_request_limit = 2
    report, status = run(config, environment, lambda request: httpx.Response(http_status, text="unavailable"))
    assert status == 1 and report["halt_reason"]
    before = EvaluationView(path(config))
    with pytest.raises(StoreConflict):
        recovery_module().recover_connections(path(config), config, before.archive["config"],
            probe=lambda config, **kwargs: pytest.fail("Ineligible recovery must not probe the service"))
    after = EvaluationView(path(config))
    assert after.tables["run"] == before.tables["run"]
    assert after.tables["attempts"] == before.tables["attempts"]
    assert not after.record_items("recovery")


def test_runtime_generation_failure_resumes_same_seed(tmp_path, environment, monkeypatch):
    config = runtime(tmp_path)
    original = environment[0].generate
    attempted = []
    def fail_once(self, q, z, seed):
        attempted.append((q, z, seed))
        if len(attempted) == 2:
            raise RuntimeError("simulated CUDA error")
        return original(self, q, z, seed)
    monkeypatch.setattr(environment[0], "generate", fail_once)
    with pytest.raises(RuntimeError, match="CUDA error"):
        run(config, environment, phase="generate")
    assert EvaluationView(path(config)).get_record("generation_attempt_result",
        EvaluationView(path(config)).record_items("generation_attempt")[-1]["id"])["status"] == "runtime_error"
    report, status = run(config, environment, phase="generate")
    assert status == 0 and len(environment[1]) == 8
    assert attempted[1] == attempted[2]
    assert report["generated"] == 8


def test_saved_raw_output_is_reparsed_without_generating_again(tmp_path, environment, monkeypatch):
    config = runtime(tmp_path)
    original = evaluation.finalize_generation
    def interrupted(*args):
        raise RuntimeError("interrupted after raw save")
    monkeypatch.setattr(evaluation, "finalize_generation", interrupted)
    with pytest.raises(RuntimeError, match="raw save"):
        run(config, environment, phase="generate")
    assert len(environment[1]) == 1
    monkeypatch.setattr(evaluation, "finalize_generation", original)
    report, status = run(config, environment, phase="generate")
    assert status == 0 and len(environment[1]) == 8 and report["generated"] == 8


def test_tampered_blob_is_rejected(tmp_path, environment):
    config = runtime(tmp_path)
    run(config, environment, phase="generate")
    view = EvaluationView(path(config))
    name = view.record_items("generation")[0]["blob"]
    (path(config) / "blobs" / name).write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        report_evaluation(path(config))


@pytest.mark.parametrize("kind", ["evaluation_repeat", "score", "label", "candidate_result"])
def test_execution_persistence_interruption_does_not_repeat_calls(tmp_path, environment, monkeypatch, kind):
    config = runtime(tmp_path)
    config.evaluation["candidate_concurrency"] = 1
    original = RunStore.put_record
    failed = False
    calls = []
    def interrupt(store, record_kind, *args, **kwargs):
        nonlocal failed
        if record_kind == kind and not failed:
            failed = True
            raise asyncio.CancelledError()
        return original(store, record_kind, *args, **kwargs)
    def handler(request):
        calls.append(request)
        return reply(request)
    monkeypatch.setattr(RunStore, "put_record", interrupt)
    with pytest.raises(asyncio.CancelledError):
        run(config, environment, handler)
    monkeypatch.setattr(RunStore, "put_record", original)
    report, status = run(config, environment, handler)
    assert status == 0 and report["engineering_passed"] and len(calls) == 200
    assert len(environment[1]) == 8


def test_credit_recovery_preserves_failed_repeats_and_generation(tmp_path, environment):
    config = runtime(tmp_path)
    config.evaluation["candidate_concurrency"] = 1
    def credits(request):
        return httpx.Response(402, json={"error": {"code": 402, "message": "Insufficient credits",
            "metadata": {"limit_source": "openrouter_credits"}}})
    halted, status = run(config, environment, credits)
    assert status == 1 and halted["halt_reason"] == "fatal_request_error"
    with pytest.raises(StoreConflict, match="halted"):
        run(config, environment)
    completed, status = run(config, environment, resume_after_topup=True)
    assert status == 0 and completed["engineering_passed"] and len(environment[1]) == 8
    view = EvaluationView(path(config))
    assert len(view.record_items("recovery")) == 1
    assert any(c["label"]["repeat_count"] == 6 for c in completed["candidates"])
    assert all(c["label"]["complete_count"] == 5 for c in completed["candidates"])
    assert any(a["http_status"] == 402 for a in view.tables["attempts"])


def test_report_command_does_not_modify_database(tmp_path, environment):
    from ocop.cli import main
    from ocop.storage import export_run

    config = runtime(tmp_path)
    run(config, environment, phase="generate")
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    export_run(path(config), before)
    assert main(["report", "--run", str(path(config))]) == 0
    export_run(path(config), after)
    assert before.read_bytes() == after.read_bytes()
