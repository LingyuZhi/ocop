import asyncio
import copy
import hashlib
import json
import runpy
from pathlib import Path
import httpx
import pytest
import ocop.collection.pipeline as collection
import ocop.runtime.llm as llm
from ocop.collection.benchmark import BenchmarkConfig, build_manifest, validate_manifest
from ocop.cli import main
from ocop.runtime.config import canonical_json, load_config
from ocop.graph import replay
from ocop.execution.labels import aggregate_label
from ocop.runtime.storage import RunStore, StoreConflict, read_database
import ocop.runtime.recovery as recovery
import importlib.util
from collections import Counter
from transformers import AutoTokenizer
import ocop.evaluation.pipeline as evaluation
from ocop.collection.benchmark import BenchmarkConfig, build_manifest
from ocop.runtime.config import load_config
from ocop.evaluation.report import EvaluationView, audit_metrics, report_evaluation
from ocop.execution.executor import executor_config, executor_hash
from ocop.inference.transformers import candidate_seed, generation_config, parse_generation
from ocop.runtime.storage import RunStore, StoreConflict
from ocop.training.data import policy_messages
from pydantic import ValidationError
from ocop.runtime.config import RequestConfig, load_config
from ocop.runtime.llm import RequestRunner, budget_retry_deadline, load_credentials, normalize_response, recover_inflight_budget
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen3_5Config
import ocop.training.data as trajectories
from ocop.training.updates import collate, select_samples, token_losses, validate_checkpoint, verification_settings
from ocop.runtime.storage import digest
from ocop.training.data import encode_sample, load_prepared, prepare_sft, source_samples

ROOT = Path(__file__).resolve().parents[1]


VALID_GRAPH = (ROOT / 'examples/graph/valid.json').read_text()


SOURCE = [{'question': f'Question {i}', 'answer': 'ReferenceSecret calculation\n#### 5'} for i in range(50)]


def collection_runtime(tmp_path, *, candidates=1, repeats=5, maximum=8, concurrency=2):
    config = load_config(ROOT / 'configs/prototype.json')
    config.artifacts_dir = str(tmp_path / 'artifacts')
    config.benchmark.update(train_tasks=1, eval_tasks=1)
    config.collection.update(candidates_per_task=candidates, complete_repeats=repeats, max_repeats=maximum, candidate_concurrency=concurrency)
    config.requests.retry_backoff_seconds = 0.0
    return config


@pytest.fixture(autouse=True)
def local_environment(monkeypatch):
    monkeypatch.setattr(collection, 'load_source', lambda config, artifacts: SOURCE)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')


def reply(request, *, graph=VALID_GRAPH, answer='#### 5', finish='stop', reasoning='Design reasoning', usage=True):
    body = json.loads(request.content)
    strong = body['model'] == 'deepseek-flash'
    return httpx.Response(200, json={'model': body['model'], 'provider': 'OpenAI', 'choices': [{'finish_reason': finish, 'message': {'content': graph if strong else answer, 'reasoning_content': reasoning if strong else None}}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5} if usage else None})


def collect(config, handler, *, manifest_only=False):
    return asyncio.run(collection.run_collection(config, Path('absent.env'), 'test', manifest_only=manifest_only, transport=httpx.MockTransport(handler)))


@pytest.fixture(scope='module', name='tokenizer')
def policy_tokenizer():
    return AutoTokenizer.from_pretrained('/data1/zhilingyu/models/Qwen3.5-2B', local_files_only=True)


def evaluation_runtime(tmp_path):
    config = load_config(ROOT / 'configs/prototype.json')
    config.artifacts_dir = str(tmp_path / 'artifacts')
    config.benchmark.update(train_tasks=1, eval_tasks=1)
    config.requests.retry_backoff_seconds = 0.0
    return config


def evaluation_snapshot(config, tokenizer):
    settings = evaluation.EvaluationConfig.model_validate(config.evaluation)
    manifest = build_manifest(SOURCE, BenchmarkConfig.model_validate(config.benchmark), config.seed)
    smoke = {'candidate_id': 'source-candidate', 'task_id': manifest['tasks'][0]['task_id'], 'z': 1.0}
    return {'purpose': 'evaluation', 'version': 'ocop.evaluation.v1', 'runtime': config.model_dump(), 'settings': settings.model_dump(), 'task_manifest': manifest, 'training_smoke': smoke, 'candidates': evaluation.candidate_plan(manifest, smoke, settings, config.seed), 'executor': executor_config(config), 'executor_hash': executor_hash(config), 'generation_config': generation_config(config.policy, tokenizer).to_dict(), 'models': {'base': {'path': 'base'}, 'last_checkpoint': {'path': 'checkpoint'}}, 'source': {'path': 'source'}, 'training': {'path': 'training'}, 'environment': {}}


def raw_output(tokenizer, question, z, content=VALID_GRAPH, *, eos=True, thinking=True):
    text = ('Design reasoning</think>\n\n' if thinking else 'Still reasoning') + content
    if eos:
        text += tokenizer.eos_token
    ids = tokenizer.encode(text, add_special_tokens=False)
    inputs = tokenizer.apply_chat_template(policy_messages(question, z), tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
    close = tokenizer.convert_tokens_to_ids('</think>')
    boundary = ids.index(close) + 1 if close in ids else len(ids)
    return {'text': text, 'token_ids': ids, 'input_ids': inputs, 'input_tokens': len(inputs), 'output_tokens': len(ids), 'reasoning_tokens': boundary, 'content_tokens': len(ids) - boundary, 'eos_token': tokenizer.eos_token, 'eos_token_id': tokenizer.eos_token_id, 'reached_eos': eos, 'finish_reason': 'eos' if eos else 'length', 'seconds': 1.0, 'tokens_per_second': len(ids), 'peak_allocated_bytes': 1024}


@pytest.fixture
def policy_environment(monkeypatch, tokenizer):
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    monkeypatch.setattr(evaluation, 'prepare_evaluation', lambda config, *args: (evaluation_snapshot(config, tokenizer), tokenizer))
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
    return (Generator, calls, loaded)


def evaluate(config, environment, handler=reply, **kwargs):
    return asyncio.run(evaluation.run_evaluation(config, Path('source'), Path('training'), Path('absent.env'), 'eval-test', 'cuda:0', generator_factory=environment[0], transport=httpx.MockTransport(handler), **kwargs))


def evaluation_path(config):
    return Path(config.artifacts_dir) / 'runs/eval-test'


CONFIG = Path(__file__).resolve().parents[1] / 'configs/prototype.json'


MODEL = load_config(CONFIG).worker_model


LIMITS = RequestConfig(concurrency=2, timeout_seconds=2.0, max_attempts=3, consecutive_exhausted_request_limit=2, retry_backoff_seconds=0.0)


MESSAGES = [{'role': 'user', 'content': 'Test input'}]


def response(finish='stop', content='5', **extra):
    return {'id': 'provider-response', 'model': MODEL.model_id, 'choices': [{'finish_reason': finish, 'message': {'content': content, 'reasoning_content': 'A thought'}}], **extra}


def tiny_qwen():
    config = Qwen3_5Config(text_config={'vocab_size': 19, 'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 8, 'layer_types': ['linear_attention', 'full_attention'], 'linear_conv_kernel_dim': 4, 'linear_key_head_dim': 8, 'linear_value_head_dim': 8, 'linear_num_key_heads': 2, 'linear_num_value_heads': 2, 'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000, 'partial_rotary_factor': 1.0, 'mrope_section': [1, 1, 2]}}, vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 32, 'num_heads': 2, 'out_hidden_size': 16})
    network = AutoModelForImageTextToText.from_config(config, attn_implementation='sdpa')
    network.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return network
