import json
from dataclasses import asdict
from pathlib import Path

from pydantic import Field

from ocop.config import RuntimeConfig, StrictModel
from ocop.diagnostics import provenance
from ocop.executor import execute_repeat, executor_config
from ocop.graph import replay
from ocop.llm import RequestRunner, load_credentials
from ocop.scoring import normalize_reference, score_answer
from ocop.storage import RunStore


class ExecutionTask(StrictModel):
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    split: str = "smoke"


async def run_execution(config: RuntimeConfig, task_path: Path, trajectory_path: Path,
                        credentials_path: Path, run_id: str | None, repeat_id: str) -> tuple[dict, int]:
    task = ExecutionTask.model_validate_json(task_path.read_text(encoding="utf-8"), strict=True)
    reference = normalize_reference(task.reference_answer)
    graph = replay(trajectory_path.read_text(encoding="utf-8"))
    if not graph.valid:
        return {"status": "invalid_graph", "error": asdict(graph.error)}, 2
    if not task.question.strip() or not repeat_id:
        raise ValueError("Question and repeat ID must be nonempty")
    spec = executor_config(config)
    credentials = load_credentials(credentials_path, {config.worker_model.api_key_env, config.finalizer.api_key_env})
    snapshot = {"purpose": "graph_execution", "runtime": config.model_dump(mode="json"), "executor": spec}
    with RunStore(Path(config.artifacts_dir) / "runs", snapshot, run_id=run_id, provenance=provenance()) as store:
        (store.path / "config.json").write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
        task_id = store.put_record("task", task.task_id, task.model_dump(mode="json"))
        trajectory_id = store.put_record("trajectory", task_id, asdict(graph), parents={"task": task_id})
        graph_id = store.put_record("graph", trajectory_id, asdict(graph.final_graph), parents={"trajectory": trajectory_id})
        async with RequestRunner(store, config.requests, credentials) as runner:
            result = await execute_repeat(question=task.question, graph=graph,
                parents={"task": task_id, "trajectory": trajectory_id, "graph": graph_id},
                repeat_id=repeat_id, config=config, runner=runner)
        score = score_answer(result["answer"], reference) if result["status"] == "completed" else None
        if score is not None:
            store.put_record("score", result["execution_id"], score,
                             parents={"execution": result["execution_id"], "task": task_id})
        report = {"run_id": store.run_id, "path": str(store.path), **result, "score": score,
                  "request_count": len(store.rows("requests")), "attempt_count": len(store.rows("attempts"))}
        store.export_jsonl(store.path / "records.jsonl")
        (store.path / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report, 0 if result["status"] == "completed" else 1
