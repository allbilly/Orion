#!/usr/bin/env python3
"""Compare actual greedy Orion token IDs with the original MLX checkpoint.

Requires mlx-lm. Uses the checkpoint's chat template with thinking disabled;
checks prompt token IDs, every generated token and EOS, not just decoded text.
"""
import argparse
import gc
import json
import subprocess
from pathlib import Path

import mlx.core as mx
from mlx_lm import load


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path.home()/'.omlx/models/mlx-community/Qwen3.5-0.8B-4bit')
    p.add_argument('--weights', type=Path, default=root/'model/blobs/qwen35_0_8b')
    p.add_argument('--orion', type=Path, default=root/'orion')
    p.add_argument('--prompt', default='What is 2 + 2? Answer briefly.')
    p.add_argument('--max-tokens', type=int, default=16)
    p.add_argument('--output', type=Path, default=root/'build/qwen35-token-comparison.json')
    args = p.parse_args()
    if not 1 <= args.max_tokens <= 256:
        p.error('--max-tokens must be 1..256')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(str(args.source.expanduser()))
    prompt = tokenizer.apply_chat_template([{'role':'user', 'content':args.prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt)
    cache = model.make_cache()
    x = prompt_ids
    output = []
    stop = None
    for _ in range(args.max_tokens):
        logits = model(mx.array([x]), cache=cache)[0,-1]
        next_id = int(mx.argmax(logits).item())
        if next_id in (248044, 248046):
            stop = next_id
            break
        output.append(next_id)
        x = [next_id]
    reference = dict(prompt_tokens=prompt_ids, generated_tokens=output,
                     stop_token=stop, text=tokenizer.decode(output))
    del model, cache, logits
    gc.collect()
    mx.clear_cache()
    results = {'reference_backend':'MLX-LM original quantized checkpoint',
        'source':str(args.source), 'prompt':args.prompt, 'reference':reference, 'orion':{}}
    matched = True
    # Run sequentially so ANE compile/load work and model mappings don't overlap.
    for mode, flags in [('cpu', []), ('ane', ['--ane'])]:
        dump = args.output.parent/f'qwen35-{mode}-tokens.json'
        command = [str(args.orion), 'infer', '--model', 'qwen35_0_8b',
            '--weights', str(args.weights), '--prompt', args.prompt,
            '--max_tokens', str(args.max_tokens), '--temperature', '0',
            '--dump-tokens', str(dump), *flags]
        run = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if run.returncode:
            raise RuntimeError(f'{mode} inference failed: {run.stderr}')
        actual = json.loads(dump.read_text())
        checks = {key:actual[key] == reference[key] for key in
            ('prompt_tokens', 'generated_tokens', 'stop_token')}
        result = dict(actual, matches=checks, stdout=run.stdout, diagnostics=run.stderr)
        results['orion'][mode] = result
        matched &= all(checks.values())
        print(mode.upper(), checks, 'tokens=', actual['generated_tokens'], flush=True)
    results['all_match'] = matched
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('MLX tokens:',output,'EOS:',stop, 'text:',reference['text'],flush=True)
    print('PASS' if matched else 'MISMATCH',args.output,flush=True)
    return 0 if matched else 1


if __name__ == '__main__':
    raise SystemExit(main())
