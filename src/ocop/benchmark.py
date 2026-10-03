import hashlib
import os
import random
from pathlib import Path
from typing import Literal

from pydantic import Field, model_serializer, model_validator

from ocop.config import StrictModel, canonical_json
from ocop.scoring import normalize_reference


class BenchmarkConfig(StrictModel):
    name: Literal["gsm8k"]
    source_split: Literal["train"]
    train_tasks: int = Field(gt=0)
    eval_tasks: int = Field(ge=0)
    holdout_tasks: int = Field(default=0, ge=0)
    excluded_rows: list[int] = Field(default_factory=list)
    dataset_id: Literal["openai/gsm8k"] = "openai/gsm8k"
    subset: Literal["main"] = "main"
    revision: str = Field(default="cc7b047b6e5bb11b4f1af84efc572db110a51b3c", pattern=r"^[a-f0-9]{40}$")

    @model_validator(mode="after")
    def check_exclusions(self):
        if len(set(self.excluded_rows)) != len(self.excluded_rows) or any(i < 0 for i in self.excluded_rows):
            raise ValueError("Excluded rows must be unique nonnegative indices")
        return self

    @model_serializer(mode="wrap")
    def serialize_sampling(self, handler):
        result = handler(self)
        for key in ("holdout_tasks", "excluded_rows"):
            if not result[key]:
                result.pop(key)
        return result


def sampled_rows(size: int, config: BenchmarkConfig, seed: int) -> list[int]:
    excluded = set(config.excluded_rows)
    if any(i >= size for i in excluded):
        raise ValueError("Excluded row outside source dataset")
    population = [i for i in range(size) if i not in excluded]
    count = config.train_tasks + config.eval_tasks + config.holdout_tasks
    if len(population) < count:
        raise ValueError("Dataset has fewer eligible rows than the requested task count")
    return random.Random(seed).sample(population, count)


def task_split(position: int, config: BenchmarkConfig) -> str:
    return "train" if position < config.train_tasks else "eval" if position < config.train_tasks + config.eval_tasks else "holdout"


def load_source(config: BenchmarkConfig, artifacts: Path):
    cache = (artifacts / "datasets").resolve()
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache / "arrow")
    from datasets import load_dataset

    return load_dataset(config.dataset_id, config.subset, split=config.source_split,
                        revision=config.revision, cache_dir=str(cache / "arrow"))


def build_manifest(source, config: BenchmarkConfig, seed: int) -> dict:
    indices = sampled_rows(len(source), config, seed)
    tasks = []
    for position, index in enumerate(indices):
        row = source[index]
        question, answer = row["question"], row["answer"]
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str):
            raise ValueError(f"Invalid GSM8K row: {index}")
        normalized = str(normalize_reference(answer))
        tasks.append({"task_id": f"gsm8k-main-train-{index}", "source_row": index,
                      "split": task_split(position, config),
                      "question": question, "reference_answer": answer, "normalized_reference": normalized})
    manifest = {"version": "ocop.tasks.v1", "source": config.model_dump(mode="json"),
                "source_rows": len(source), "seed": seed, "sampling": "python_random_sample.v1", "tasks": tasks}
    return {**manifest, "hash": hashlib.sha256(canonical_json(manifest)).hexdigest()}


def validate_manifest(manifest: dict, config: BenchmarkConfig, seed: int):
    payload = {key: value for key, value in manifest.items() if key != "hash"}
    if hashlib.sha256(canonical_json(payload)).hexdigest() != manifest["hash"]:
        raise ValueError("Task manifest hash mismatch")
    if manifest["source"] != config.model_dump(mode="json") or manifest["seed"] != seed:
        raise ValueError("Task manifest configuration mismatch")
    tasks = manifest["tasks"]
    expected = sampled_rows(manifest["source_rows"], config, seed)
    if [task["source_row"] for task in tasks] != expected:
        raise ValueError("Task manifest sampling mismatch")
    for position, task in enumerate(tasks):
        split = task_split(position, config)
        if task["split"] != split or task["task_id"] != f"gsm8k-main-train-{task['source_row']}":
            raise ValueError("Task manifest identity or split mismatch")
        if str(normalize_reference(task["reference_answer"])) != task["normalized_reference"]:
            raise ValueError("Task manifest reference mismatch")
