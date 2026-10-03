#!/usr/bin/env python3
"""Export Qwen3.5-0.8B text weights for Orion (no MLX dependency at runtime).

Accepts local Hugging Face or affine-quantized MLX safetensors. Conversion
requires numpy and mlx; --reference also requires mlx-lm and tokenizers.
Weights are little-endian float32, mapped by the native runtime. Quantized
checkpoints are dequantized, so the output is larger than the source.
"""
import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np


def validate_config(c):
    expected = dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=8,
                    num_key_value_heads=2, head_dim=256, intermediate_size=3584,
                    vocab_size=248320, linear_num_key_heads=16,
                    linear_num_value_heads=16, linear_key_head_dim=128,
                    linear_value_head_dim=128, linear_conv_kernel_dim=4,
                    full_attention_interval=4)
    for key, value in expected.items():
        if c.get(key) != value:
            raise ValueError(f"Expected Qwen3.5-0.8B: {key} must be {value}")
    if not c.get('tie_word_embeddings') or c.get('num_experts', 0):
        raise ValueError('Only the dense, tied-embedding 0.8B text model is supported')
    if c.get('attention_bias', False) or not c.get('attn_output_gate', True) or c.get('hidden_act') != 'silu':
        raise ValueError('Unsupported attention or activation configuration')
    layers = ['full_attention' if (i + 1) % 4 == 0 else 'linear_attention' for i in range(24)]
    if c.get('layer_types', layers) != layers:
        raise ValueError('Unsupported layer order')


def convert(source, output):
    config = json.loads((source / 'config.json').read_text())
    text = config.get('text_config', config)
    validate_config(text)
    if (output / 'manifest.json').exists():
        raise ValueError('Output already contains a manifest; choose a new directory')
    weights = {}
    for shard in sorted(source.glob('*.safetensors')):
        weights.update(mx.load(str(shard)))
    if not weights:
        raise ValueError('No safetensors found')
    # MLX stores Conv1d [channel, kernel, 1] and ordinary RMSNorm multipliers;
    # HF stores [channel, 1, kernel] and zero-centered norm parameters.
    unsanitized = any('conv1d.weight' in k and v.shape[-1] != 1 for k, v in weights.items())
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(format='orion-qwen35-f32-v1', config=text, tensors={},
                    source=str(source), source_quantization=config.get('quantization'))
    norm_suffixes = ('input_layernorm.weight', 'post_attention_layernorm.weight',
                     'model.norm.weight', 'q_norm.weight', 'k_norm.weight')
    quant = config.get('quantization', {})
    for original, value in sorted(weights.items()):
        name = original.removeprefix('language_model.')
        if name.startswith('model.language_model.'):
            name = 'model.' + name.removeprefix('model.language_model.')
        if name.startswith(('model.visual.', 'vision_tower.')):
            continue
        if not name.startswith('model.') or 'mtp.' in name or name.endswith(('.scales', '.biases')):
            continue
        if original.removesuffix('.weight') + '.scales' in weights:
            prefix = original.removesuffix('.weight')
            q = quant.get(prefix, quant)
            value = mx.dequantize(value, weights[prefix + '.scales'], weights.get(prefix + '.biases'),
                                  group_size=q.get('group_size', 64), bits=q.get('bits', 4),
                                  mode=q.get('mode', 'affine'))
        value = value.astype(mx.float32)
        if 'conv1d.weight' in name and unsanitized:
            value = mx.moveaxis(value, 2, 1)
        if unsanitized and name.endswith(norm_suffixes):
            value = value + 1
        mx.eval(value)
        array = np.asarray(value).astype('<f4', copy=False)
        filename = name + '.f32'
        array.tofile(output / filename)
        manifest['tensors'][name] = dict(file=filename, shape=list(array.shape))
        del array, value
        mx.clear_cache()
    for filename in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        if (source / filename).exists():
            shutil.copyfile(source / filename, output / filename)
    if 'model.embed_tokens.weight' not in manifest['tensors'] or len(manifest['tensors']) != 320:
        raise ValueError('Checkpoint does not contain the expected 320 text tensors')
    # Publish only after all tensors are written.
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Exported {len(manifest['tensors'])} tensors to {output}", flush=True)


def reference(output):
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from tokenizers import Tokenizer
    manifest = json.loads((output / 'manifest.json').read_text())
    model = TextModel(TextModelArgs.from_dict(manifest['config']))
    # Cast parameters to fp32 and use exactly the exported tensors. This separates
    # native math/layout correctness from checkpoint quantization/rounding.
    for name, entry in manifest['tensors'].items():
        array = np.fromfile(output / entry['file'], dtype='<f4').reshape(entry['shape'])
        model.load_weights([(name, mx.array(array))], strict=False)
    tokenizer = Tokenizer.from_file(str(output / 'tokenizer.json'))
    texts = ['Hello, world!', '你好，世界！', "WE'RE testing 12345\n\n", 'e\u0301 café 😀',
             '<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n']
    cases = [{'text': t, 'tokens': tokenizer.encode(t).ids,
              'decoded': tokenizer.decode(tokenizer.encode(t).ids, skip_special_tokens=False)} for t in texts]
    prompt = '<|im_start|>user\nWhat is 2 + 2? Answer briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    ids = tokenizer.encode(prompt).ids
    cache = model.make_cache()
    step_ids = ids
    generated = []
    for step in range(5):
        logits = model(mx.array([step_ids]), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(logits)
        np.asarray(logits).tofile(output / f'reference_logits_{step}.f32')
        token = int(mx.argmax(logits).item())
        generated.append(token)
        step_ids = [token]
    (output / 'reference.json').write_text(json.dumps(dict(tokenizer_cases=cases, prompt=prompt,
            prompt_tokens=ids, generated_tokens=generated), ensure_ascii=False, indent=2) + '\n')
    print('Reference tokens:', generated, tokenizer.decode(generated), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', action='store_true')
    args = parser.parse_args()
    if args.source:
        convert(args.source.expanduser().resolve(), args.output)
    if args.reference:
        reference(args.output)
    elif not args.source:
        parser.error('--source or --reference is required')
