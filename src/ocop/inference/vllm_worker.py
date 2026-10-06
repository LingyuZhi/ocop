import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback


def environment():
    return {"python": platform.python_version(), "executable": os.path.realpath(sys.executable),
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ("vllm", "torch", "transformers", "tokenizers", "triton", "flashinfer-python", "numpy")}}


def serve():
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def emit(payload):
        protocol.write(json.dumps(payload, allow_nan=False) + "\n")

    from pynvml import nvmlDeviceGetHandleByUUID, nvmlDeviceGetIndex, nvmlInit, nvmlShutdown

    gpu_uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    nvmlInit()
    try:
        physical_index = nvmlDeviceGetIndex(nvmlDeviceGetHandleByUUID(gpu_uuid))
    finally:
        nvmlShutdown()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_index)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    import torch
    from vllm import LLM, SamplingParams

    try:
        init = json.loads(sys.stdin.readline())
        options = init["engine"]
        llm = LLM(model=init["model_path"], tokenizer=init["model_path"],
            tensor_parallel_size=1, dtype="bfloat16", trust_remote_code=False,
            max_model_len=options["max_model_len"], gpu_memory_utilization=options["gpu_memory_utilization"],
            max_num_seqs=options["batch_size"], max_num_batched_tokens=options["max_num_batched_tokens"],
            enforce_eager=options["enforce_eager"], enable_prefix_caching=False,
            skip_tokenizer_init=True, generation_config="vllm", seed=0,
            limit_mm_per_prompt={"image": 0, "video": 0}, language_model_only=True, disable_log_stats=True)
        emit({"event": "ready", "environment": environment(), "gpu_uuid": gpu_uuid, "physical_index": physical_index})
        sampling = init["generation_config"]
        for line in sys.stdin:
            command = json.loads(line)
            if command["event"] == "close":
                emit({"event": "closed"})
                return
            jobs = command["jobs"]
            prompts = [{"prompt_token_ids": job["input_ids"]} for job in jobs]
            params = [SamplingParams(n=1, temperature=sampling["temperature"], top_p=sampling["top_p"],
                top_k=-1, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                max_tokens=sampling["max_new_tokens"], seed=job["seed"],
                ignore_eos=True, stop_token_ids=[sampling["eos_token_id"]],
                detokenize=False, skip_special_tokens=False) for job in jobs]
            started = time.monotonic()
            outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
            seconds = time.monotonic() - started
            if len(outputs) != len(jobs):
                raise RuntimeError("vLLM returned an unexpected number of candidates")
            for job, output in zip(jobs, outputs, strict=True):
                if list(output.prompt_token_ids) != job["input_ids"] or len(output.outputs) != 1:
                    raise RuntimeError("vLLM changed the frozen prompt or candidate count")
                result = output.outputs[0]
                emit({"event": "generated", "key": job["key"], "input_ids": job["input_ids"],
                      "token_ids": list(result.token_ids), "seconds": seconds,
                      "finish_reason": result.finish_reason, "stop_reason": result.stop_reason,
                      "timing_source": "shared_batch_wall_seconds", "gpu_uuid": gpu_uuid,
                      "batch_size": len(jobs), "gpu_memory_used_bytes": torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]})
            emit({"event": "batch_complete", "seconds": seconds, "jobs": len(jobs)})
    except Exception as exc:
        emit({"event": "error", "error_type": type(exc).__name__, "error": str(exc)})
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    if sys.argv[1:] == ["--environment"]:
        print(json.dumps(environment()))
    else:
        serve()
