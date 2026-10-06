import json
import os
import queue
import signal
import subprocess
import threading
from pathlib import Path

from pydantic import Field, model_validator

from ocop.runtime.config import StrictModel
from ocop.runtime.storage import file_hash


class VllmConfig(StrictModel):
    python: str
    environment_lock: str
    cache_dir: str
    gpu_uuids: list[str] = Field(min_length=2, max_length=6)
    batch_size: int = Field(default=16, gt=0)
    max_model_len: int = Field(default=10240, gt=0)
    max_num_batched_tokens: int = Field(default=4096, gt=0)
    gpu_memory_utilization: float = Field(default=0.75, gt=0, lt=1)
    enforce_eager: bool = True

    @model_validator(mode="after")
    def check_devices(self):
        if len(self.gpu_uuids) % 2 or len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError("vLLM requires distinct, equally partitioned base/checkpoint GPUs")
        return self


def worker_environment(gpu_uuid=None, cache=None):
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    for key in list(env):
        if key.endswith("_API_KEY"):
            env.pop(key)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", VLLM_WORKER_MULTIPROC_METHOD="spawn",
               CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
    if gpu_uuid:
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    if cache:
        for key, suffix in (("TRITON_CACHE_DIR", "triton"), ("VLLM_CACHE_ROOT", "vllm"), ("CUDA_CACHE_PATH", "cuda")):
            env[key] = str(cache / suffix)
    return env


def inference_environment(settings):
    result = subprocess.run([settings.python, "-m", "ocop.inference.vllm_worker", "--environment"],
        env=worker_environment(), capture_output=True, text=True, check=True, timeout=60)
    return {**json.loads(result.stdout), "environment_lock_sha256": file_hash(Path(settings.environment_lock))}


class VllmPool:
    def __init__(self, settings, snapshot, log_dir):
        self.settings, self.snapshot = settings, snapshot
        self.log_dir = Path(log_dir)
        self.processes = []
        self.threads = []
        self.logs = []
        self.events = queue.Queue()

    def _send(self, index, payload):
        process = self.processes[index]
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def _read(self, index, process):
        try:
            for line in process.stdout:
                self.events.put((index, json.loads(line)))
        except Exception as exc:
            self.events.put((index, {"event": "error", "error": f"{type(exc).__name__}: {exc}"}))
        finally:
            self.events.put((index, {"event": "exit"}))

    def generate(self, jobs):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        assignments = [[] for _ in self.settings.gpu_uuids]
        half = len(assignments) // 2
        model_offsets = {name: number * half for number, name in enumerate(self.snapshot["settings"]["models"])}
        for job in jobs:
            assignments[model_offsets[job["model"]] + (job["ordinal"] - 1) % half].append(job)
        batches = [[items[start:start + self.settings.batch_size] for start in range(0, len(items), self.settings.batch_size)]
                   for items in assignments]
        closed, yielded = set(), set()
        expected = {job["key"]: job for job in jobs}
        for index, gpu_uuid in enumerate(self.settings.gpu_uuids):
            log = (self.log_dir / f"worker-{index}.log").open("a")
            self.logs.append(log)
            process = subprocess.Popen([self.settings.python, "-m", "ocop.inference.vllm_worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                env=worker_environment(gpu_uuid, Path(self.settings.cache_dir) / f"worker-{index}"), start_new_session=True)
            self.processes.append(process)
            model = self.snapshot["settings"]["models"][index // half]
            self._send(index, {"model_path": self.snapshot["models"][model]["path"],
                "generation_config": self.snapshot["generation_config"], "engine": self.settings.model_dump()})
            thread = threading.Thread(target=self._read, args=(index, process), daemon=True)
            thread.start()
            self.threads.append(thread)
        while len(closed) != len(self.processes):
            try:
                index, item = self.events.get(timeout=30)
            except queue.Empty:
                failed = [index for index, process in enumerate(self.processes) if process.poll() is not None and index not in closed]
                if failed:
                    raise RuntimeError(f"vLLM workers exited unexpectedly: {failed}; inspect {self.log_dir}")
                continue
            event = item["event"]
            if event in {"ready", "batch_complete"}:
                if event == "ready" and item["environment"] != self.snapshot["inference_environment"]:
                    expected_env = {key: value for key, value in self.snapshot["inference_environment"].items()
                                    if key != "environment_lock_sha256"}
                    if item["environment"] != expected_env:
                        raise RuntimeError("vLLM worker environment differs from the frozen snapshot")
                self._send(index, {"event": "batch", "jobs": batches[index].pop(0)} if batches[index] else {"event": "close"})
            elif event == "generated":
                key = item["key"]
                if key not in expected or key in yielded or item["input_ids"] != expected[key]["input_ids"]:
                    raise RuntimeError("vLLM returned duplicate, unknown or changed candidates")
                yielded.add(key)
                yield expected[key], item
            elif event == "closed":
                closed.add(index)
            elif event == "error":
                raise RuntimeError(f"vLLM worker {index}: {item.get('error')}; inspect {self.log_dir}")
            elif event == "exit" and index not in closed:
                raise RuntimeError(f"vLLM worker {index} exited before completion; inspect {self.log_dir}")
        if yielded != set(expected):
            raise RuntimeError("vLLM did not finish the frozen candidate plan")

    def close(self):
        for process in self.processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in self.processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
        for thread in self.threads:
            thread.join(timeout=5)
        for log in self.logs:
            log.close()
