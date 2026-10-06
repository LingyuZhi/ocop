import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from ocop.runtime.config import load_config
from ocop.execution.executor import execute_repeat, executor_hash, finalizer_messages, worker_messages
from ocop.graph import replay
from ocop.runtime.llm import RequestRunner
from ocop.runtime.storage import RunStore, StoreConflict


ROOT = Path(__file__).resolve().parents[1]
QUESTION = "Original problem"


def config():
    value = load_config(ROOT / "config/prototype.json")
    value.requests.retry_backoff_seconds = 0.0
    return value


def graph(edges=None):
    document = json.loads((ROOT / "examples/graph/valid.json").read_text())
    if edges is not None:
        document["steps"] = document["steps"][:4] + [
            {"explanation": "Communicate", "action": {"type": "ADD_EDGE", "source": source, "target": target}}
            for source, target in edges
        ] + [document["steps"][-1]]
    return replay(json.dumps(document))


def response(content="output", finish="stop", model="openai/gpt-4o-mini"):
    return httpx.Response(200, json={"model": model, "provider": "OpenAI", "choices": [
        {"finish_reason": finish, "message": {"content": content}}
    ], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})


async def execute(store, handler, *, repeat="0", parsed=None, runtime=None, question=QUESTION):
    runtime = runtime or config()
    async with RequestRunner(store, runtime.requests, {"OPENAI_API_KEY": "test"},
                             transport=httpx.MockTransport(handler)) as runner:
        return await execute_repeat(question=question, graph=parsed or graph(), parents={}, repeat_id=repeat,
                                    config=runtime, runner=runner)


def test_messages_are_sorted_and_have_only_allowed_fields():
    messages = worker_messages(QUESTION, "Checker", {"worker_3": "third", "worker_0": "first"})
    assert json.loads(messages[1]["content"]) == {"question": QUESTION, "predecessors": [
        {"worker_id": "worker_0", "output": "first"}, {"worker_id": "worker_3", "output": "third"}]}
    outputs = {f"worker_{i}": str(i) for i in (3, 1, 2, 0)}
    assert json.loads(finalizer_messages(QUESTION, outputs)[1]["content"]) == {
        "question": QUESTION, "workers": [{"worker_id": f"worker_{i}", "output": str(i)} for i in range(4)]}


@pytest.mark.parametrize("edges", [[], [("worker_3", "worker_0"), ("worker_0", "worker_1")],
    [("worker_3", "worker_0"), ("worker_2", "worker_0"), ("worker_0", "worker_1")]])
def test_dag_concurrency_boundaries_and_finalizer(tmp_path, edges):
    active = maximum = 0
    captured = []
    async def handler(request):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        payload = json.loads(request.content)
        inputs = json.loads(payload["messages"][1]["content"])
        captured.append(inputs)
        await asyncio.sleep(0.005)
        active -= 1
        return response("#### 5" if "workers" in inputs else f"output-{len(captured)}")
    with RunStore(tmp_path, {}) as store:
        result = asyncio.run(execute(store, handler, parsed=graph(edges)))
        assert result["status"] == "completed"
        assert len(captured) == 5
        assert maximum >= 2
        assert result["unexecuted_nodes"] == []
        for row in store.rows("records"):
            if row["kind"] != "execution_node":
                continue
            node = json.loads(store.read_blob(row["payload_blob"]))
            inputs = json.loads(node["messages"][1]["content"])
            if node["node"] == "finalizer":
                assert set(inputs) == {"question", "workers"}
                assert [item["output"] for item in inputs["workers"]] == [result["nodes"][f"worker_{i}"]["content"] for i in range(4)]
            else:
                assert set(inputs) == {"question", "predecessors"}
                predecessors = sorted(source for source, target in edges if target == node["node"])
                assert inputs["predecessors"] == [{"worker_id": parent, "output": result["nodes"][parent]["content"]} for parent in predecessors]
        assert result["usage"]["worker"]["known_usage_sum"]["total_tokens"] == 20
        assert result["usage"]["finalizer"]["known_usage_sum"]["total_tokens"] == 5


def test_resume_and_new_repeat(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return response()
    with RunStore(tmp_path, {}, run_id="run") as store:
        first = asyncio.run(execute(store, handler))
    with RunStore(tmp_path, {}, run_id="run") as store:
        assert asyncio.run(execute(store, handler)) == first
        assert len(calls) == 5
        second = asyncio.run(execute(store, handler, repeat="1"))
        assert second["execution_id"] != first["execution_id"]
        assert len(calls) == 10
        with pytest.raises(StoreConflict):
            asyncio.run(execute(store, handler, question="changed"))
        changed = config()
        changed.worker_model.temperature = 0.2
        with pytest.raises(StoreConflict):
            asyncio.run(execute(store, handler, runtime=changed))
        with pytest.raises(StoreConflict):
            asyncio.run(execute(store, handler, parsed=graph([])))


def test_partial_resume_uses_completed_predecessors(tmp_path):
    calls = []
    def interrupt(request):
        inputs = json.loads(json.loads(request.content)["messages"][1]["content"])
        calls.append(inputs)
        if inputs.get("predecessors"):
            raise asyncio.CancelledError()
        return response("saved output")
    with RunStore(tmp_path, {}, run_id="run") as store:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(execute(store, interrupt))
    def resume(request):
        calls.append(json.loads(json.loads(request.content)["messages"][1]["content"]))
        return response()
    with RunStore(tmp_path, {}, run_id="run") as store:
        result = asyncio.run(execute(store, resume))
        assert result["status"] == "completed"
        assert len(calls) == 6
        assert len(store.rows("requests")) == 5
        assert len(store.rows("attempts")) == 6
        assert result["nodes"]["worker_3"]["content"] == "saved output"
        assert result["usage"]["worker"]["unknown_usage_attempts"] == 1


@pytest.mark.parametrize(("failure", "status", "attempts"), [("length", "incomplete", 1), ("infra", "infra_failed", 3), ("auth", "fatal", 1), ("model", "fatal", 1)])
def test_failed_worker_stops_downstream_and_finalizer(tmp_path, failure, status, attempts):
    def handler(request):
        if failure == "infra":
            return httpx.Response(503)
        if failure == "auth":
            return httpx.Response(401)
        return response(finish="length" if failure == "length" else "stop", model="wrong" if failure == "model" else "openai/gpt-4o-mini")
    linear = graph([(f"worker_{i}", f"worker_{i+1}") for i in range(3)])
    with RunStore(tmp_path, {}) as store:
        result = asyncio.run(execute(store, handler, parsed=linear))
        assert result["status"] == status
        assert result["answer"] is None
        assert set(result["nodes"]) == {"worker_0"}
        assert len(store.rows("attempts")) == attempts
        assert asyncio.run(execute(store, handler, parsed=linear)) == result


def test_finalizer_truncation_is_incomplete(tmp_path):
    def handler(request):
        inputs = json.loads(json.loads(request.content)["messages"][1]["content"])
        return response("#### 5", finish="length" if "workers" in inputs else "stop")
    with RunStore(tmp_path, {}) as store:
        result = asyncio.run(execute(store, handler))
        assert result["status"] == "incomplete"
        assert result["answer"] is None
        assert len(store.rows("attempts")) == 5


def test_invalid_or_tampered_graph_never_calls_provider(tmp_path):
    with RunStore(tmp_path, {}) as store:
        for invalid in (replay("{}"), replace(graph(), final_graph=graph([]).final_graph)):
            with pytest.raises(ValueError, match="validated trajectory"):
                asyncio.run(execute(store, lambda request: pytest.fail("Unexpected call"), parsed=invalid))
        assert not store.rows("requests")


def test_executor_hash_is_sensitive_to_execution_settings_only():
    original = config()
    changed = config()
    changed.training["epochs"] = 100
    assert executor_hash(original) == executor_hash(changed)
    changed.requests.concurrency = 1
    assert executor_hash(original) != executor_hash(changed)


def test_failure_drains_inflight_and_preserves_priority(tmp_path):
    async def handler(request):
        system = json.loads(request.content)["messages"][0]["content"]
        if system.startswith("Break"):
            return response(finish="length")
        await asyncio.sleep(0.01)
        return httpx.Response(503)
    with RunStore(tmp_path, {}) as store:
        result = asyncio.run(execute(store, handler))
        assert result["status"] == "infra_failed"
        assert set(result["nodes"]) == {"worker_2", "worker_3"}
        assert result["nodes"]["worker_3"]["status"] == "incomplete"
        assert result["nodes"]["worker_2"]["status"] == "infra_failed"
        assert all(row["state"] != "in_flight" for row in store.rows("attempts"))


def test_fatal_with_queued_requests_produces_report(tmp_path):
    runtime = config()
    runtime.requests.concurrency = 1
    async def handler(request):
        await asyncio.sleep(0.001)
        return httpx.Response(401)
    with RunStore(tmp_path, {}) as store:
        result = asyncio.run(execute(store, handler, parsed=graph([]), runtime=runtime))
        assert result["status"] == "fatal"
        assert len(store.rows("attempts")) == 1
        assert result["unexecuted_nodes"] == ["finalizer"]


def test_saved_node_results_survive_interruption_before_execution_result(tmp_path, monkeypatch):
    with RunStore(tmp_path, {}, run_id="run") as store:
        put = store.put_record
        def interrupt(kind, *args, **kwargs):
            if kind == "execution_result":
                raise asyncio.CancelledError()
            return put(kind, *args, **kwargs)
        monkeypatch.setattr(store, "put_record", interrupt)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(execute(store, lambda request: response()))
    with RunStore(tmp_path, {}, run_id="run") as store:
        result = asyncio.run(execute(store, lambda request: pytest.fail("Unexpected API call")))
        assert result["status"] == "completed"
        assert len(store.rows("attempts")) == 5


def test_saved_failure_is_not_rescheduled(tmp_path, monkeypatch):
    with RunStore(tmp_path, {}, run_id="run") as store:
        put = store.put_record
        def interrupt(kind, *args, **kwargs):
            if kind == "execution_result":
                raise asyncio.CancelledError()
            return put(kind, *args, **kwargs)
        monkeypatch.setattr(store, "put_record", interrupt)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(execute(store, lambda request: response(finish="length")))
    with RunStore(tmp_path, {}, run_id="run") as store:
        result = asyncio.run(execute(store, lambda request: pytest.fail("Unexpected API call")))
        assert result["status"] == "incomplete"
        assert set(result["nodes"]) == {"worker_2", "worker_3"}


def test_ready_successor_does_not_wait_for_unrelated_root(tmp_path):
    async def check(store):
        successor_started = asyncio.Event()
        async def handler(request):
            messages = json.loads(request.content)["messages"]
            inputs = json.loads(messages[1]["content"])
            if messages[0]["content"].startswith("Solve the given"):
                if inputs["predecessors"]:
                    successor_started.set()
                else:
                    await asyncio.wait_for(successor_started.wait(), timeout=1)
            return response()
        result = await execute(store, handler)
        assert result["status"] == "completed"
        assert successor_started.is_set()
    with RunStore(tmp_path, {}) as store:
        asyncio.run(check(store))
