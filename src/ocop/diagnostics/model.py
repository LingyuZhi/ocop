import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer, GenerationConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


def inspect_model(model_path, output_path, decode=False):
    model = model_path
    config = json.loads((model / 'config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    generation = GenerationConfig.from_pretrained(model, local_files_only=True) if (model / 'generation_config.json').exists() else None
    messages = [{'role': 'user', 'content': 'How many balls are 2 red balls and 3 blue balls in total?'}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    prefix_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
    completed = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'reasoning_content': 'Add 2 and 3.', 'content': '5'}], tokenize=False, enable_thinking=True)
    completed_ids = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'reasoning_content': 'Add 2 and 3.', 'content': '5'}], tokenize=True, return_dict=False, enable_thinking=True)
    assert prefix.count('<think>') == 1 and '</think>' not in prefix
    assert completed.count('<think>') == completed.count('</think>') == 1
    if completed_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError(f'Token prefix mismatch: prefix={prefix!r}, completed={completed!r}, prefix_ids={prefix_ids}, completed_ids={completed_ids}')
    tensors = []
    weight_files = sorted(model.glob('*.safetensors'))
    for path in weight_files:
        with safe_open(path, framework='pt', device='cpu') as weights:
            dtypes = {}
            count = 0
            for key in weights.keys():
                tensor = weights.get_slice(key)
                dtype = tensor.get_dtype()
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
                count += 1
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            tensors.append({'file': path.name, 'sha256': digest, 'size_bytes': path.stat().st_size, 'tensor_count': count, 'dtype_counts': dtypes})
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({'index': index, 'name': properties.name, 'total_memory': properties.total_memory, 'capability': list(torch.cuda.get_device_capability(index))})
        value = torch.tensor([2.0, 3.0], device='cuda:0', dtype=torch.bfloat16).sum()
        assert value.item() == 5.0
    report = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'upstream_revision': None, 'model_path': str(model.resolve()), 'model_type': config['model_type'], 'architectures': config.get('architectures'), 'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'safetensors')}, 'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'devices': devices, 'file_hashes': {name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'chat_template.jinja', 'generation_config.json') if (model / name).exists()}, 'weights': tensors, 'tokenizer_class': type(tokenizer).__name__, 'tokenizer_eos': tokenizer.eos_token_id, 'model_eos': config.get('text_config', config).get('eos_token_id'), 'generation_eos': generation.eos_token_id if generation else None, 'transformers_supports_architecture': config['model_type'] in CONFIG_MAPPING, 'thinking_prefix': prefix, 'serialized_completion': completed, 'prefix_ids': prefix_ids, 'completion_ids': completed_ids, 'prefix_matches_completion': True, 'weight_loading_and_decoding_verified': False}
    if decode:
        if not devices:
            raise RuntimeError('CUDA is required for the decoding check')
        network = AutoModelForImageTextToText.from_pretrained(model, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
        generation_config = GenerationConfig(do_sample=False, max_new_tokens=128, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        inputs = torch.tensor([prefix_ids], device='cuda:0')
        with torch.inference_mode():
            generated = network.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), generation_config=generation_config)
        generated_ids = generated[0, len(prefix_ids):].tolist()
        report['decoding'] = {'purpose': 'environment_smoke', 'backend': 'transformers', 'generation_config': generation_config.to_dict(), 'token_ids': generated_ids, 'text': tokenizer.decode(generated_ids, skip_special_tokens=False), 'reached_eos': bool(generated_ids) and generated_ids[-1] == tokenizer.eos_token_id, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(0)}
        report['weight_loading_and_decoding_verified'] = len(generated_ids) > 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'output': str(output_path), 'cuda_available': report['cuda_available'], 'gpu_count': len(devices), 'template_check': 'passed', 'versions': report['versions']}
