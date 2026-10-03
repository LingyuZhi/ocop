import gc
import random
import time
from dataclasses import asdict

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, GenerationConfig

from ocop.graph import replay
from ocop.trajectories import digest, policy_messages


def candidate_seed(seed, task_id, slot):
    return int(digest({"seed": seed, "task_id": task_id, "slot": slot})[:8], 16)


def generation_config(policy, tokenizer):
    if not policy["thinking"] or policy["constrained_decoding"] or policy["temperature"] <= 0:
        raise ValueError("Evaluation requires unconstrained thinking with positive sampling temperature")
    if policy["max_new_tokens"] < 1 or not 0 < policy["top_p"] <= 1:
        raise ValueError("Invalid generation budget or top_p")
    return GenerationConfig(do_sample=True, temperature=policy["temperature"], top_p=policy["top_p"],
        top_k=0, repetition_penalty=1.0, num_beams=1, max_new_tokens=policy["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)


def parse_generation(raw):
    text = raw["text"]
    eos = raw["reached_eos"]
    body = text.removesuffix(raw["eos_token"]) if eos else text
    reasoning, separator, content = body.partition("</think>")
    content = content.strip() if separator else ""
    parsed = replay(content, reasoning=reasoning)
    error = ("truncated" if raw["finish_reason"] == "length" else "not_terminated") if not eos else None
    if error is None and not separator:
        error = "missing_thinking_end"
    if error is None and not parsed.valid:
        error = parsed.error.code
    return {"raw_content": content, "raw_reasoning": reasoning, "has_thinking_end": bool(separator),
        "json_parsed": parsed.error is None or parsed.error.code != "invalid_json",
        "valid_graph": parsed.valid, "eligible_for_execution": error is None,
        "error": error, "replay": asdict(parsed)}


class TransformersGenerator:
    def __init__(self, path, device, config, reference_tokenizer):
        if not torch.cuda.is_available() or not device.startswith("cuda:"):
            raise ValueError("Evaluation requires an available CUDA GPU")
        torch.cuda.set_device(device)
        if torch.cuda.mem_get_info(device)[0] < 12 * 1024**3:
            raise ValueError("Evaluation requires at least 12 GiB free GPU memory")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if (self.tokenizer.get_vocab() != reference_tokenizer.get_vocab()
                or self.tokenizer.special_tokens_map != reference_tokenizer.special_tokens_map
                or self.tokenizer.chat_template != reference_tokenizer.chat_template):
            raise ValueError("Model tokenizer or chat template differs from the frozen policy tokenizer")
        self.config = GenerationConfig.from_dict(config)
        self.network = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.network.requires_grad_(False)
        torch.backends.cuda.matmul.allow_tf32 = False

    def generate(self, question, z, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        messages = policy_messages(question, z)
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
            return_tensors="pt", add_generation_prompt=True, enable_thinking=True).to(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        started = time.monotonic()
        with torch.inference_mode():
            output = self.network.generate(**inputs, generation_config=self.config)
        torch.cuda.synchronize(self.device)
        seconds = time.monotonic() - started
        ids = output[0, inputs["input_ids"].shape[1]:].tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        reached_eos = bool(ids) and ids[-1] == self.tokenizer.eos_token_id
        close = self.tokenizer.convert_tokens_to_ids("</think>")
        if close == self.tokenizer.unk_token_id or self.tokenizer.decode([close]) != "</think>":
            raise ValueError("Tokenizer thinking delimiter is not a standalone token")
        boundary = ids.index(close) + 1 if close in ids else len(ids)
        return {"input_ids": inputs["input_ids"][0].tolist(), "token_ids": ids, "text": text,
            "eos_token": self.tokenizer.eos_token, "eos_token_id": self.tokenizer.eos_token_id,
            "reached_eos": reached_eos,
            "finish_reason": "eos" if reached_eos else "length" if len(ids) == self.config.max_new_tokens else "other",
            "input_tokens": inputs["input_ids"].shape[1], "output_tokens": len(ids),
            "reasoning_tokens": boundary, "content_tokens": len(ids) - boundary,
            "token_count_convention": "reasoning includes closing think; content includes message end",
            "seconds": seconds, "tokens_per_second": len(ids) / seconds if seconds else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)}

    def close(self):
        del self.network
        gc.collect()
        torch.cuda.empty_cache()
