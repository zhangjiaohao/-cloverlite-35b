#!/usr/bin/env python3
"""Experimental single-turn Clover causal demo for the Strata checkout.

The GGUF tokenizer is reconstructed locally; no model files are downloaded.
This does not implement Clover's block-diffusion generation.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

try:
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
except ImportError:
    sys.exit('Install the tokenizers package: .venv-clover/bin/pip install tokenizers')

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / 'models/Clover-35B-A3B-Q4_K_M.gguf'
ENGINE = ROOT / 'third_party/Strata/build-clover/strata-q36'


def tokenizer_from_gguf(path):
    with path.open('rb') as f:
        def number(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]

        def string():
            return f.read(number('Q')).decode('utf-8')

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                subtype, count = number('I'), number('Q')
                return [value(subtype) for _ in range(count)]
            formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                       6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
            return number(formats[kind])

        if f.read(4) != b'GGUF' or number('I') != 3:
            raise ValueError('Expected GGUF v3')
        number('Q')  # tensor count
        metadata_count = number('Q')
        metadata = {}
        for _ in range(metadata_count):
            name = string()
            data = value(number('I'))
            if name.startswith('tokenizer.ggml.'):
                metadata[name] = data
    if metadata.get('tokenizer.ggml.pre') != 'qwen35':
        raise ValueError('Expected the Clover qwen35 tokenizer')
    tokens = metadata['tokenizer.ggml.tokens']
    types = metadata['tokenizer.ggml.token_type']
    merges = [tuple(pair.split(' ', 1)) for pair in metadata['tokenizer.ggml.merges']]
    tokenizer = Tokenizer(models.BPE({token: i for i, token in enumerate(tokens)}, merges))
    tokenizer.normalizer = normalizers.NFC()
    pattern = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
               r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([
        AddedToken(token, special=True, normalized=False)
        for token, token_type in zip(tokens, types) if token_type in (3, 4)
    ])
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt', nargs='?', default='What is 2+2?')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--engine', type=Path, default=ENGINE)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--expert-vram', type=float, default=4.0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--adaptive-experts', action='store_true', help='enable experimental expert promotion')
    parser.add_argument('--async-cuda', action='store_true', help='enable experimental asynchronous CUDA execution')
    args = parser.parse_args()
    tokenizer = tokenizer_from_gguf(args.model)
    prompt = ('<|im_start|>user\n' + args.prompt + '<|im_end|>\n'
              '<|im_start|>assistant\n<think>\n\n</think>\n\n')
    ids = tokenizer.encode(prompt).ids
    if len(ids) + args.max_tokens > 1024:
        sys.exit('Prompt plus output exceeds this demo’s 1024-token context')
    cmd = [str(args.engine), '--model', str(args.model), '--experimental-clover-causal',
           '--max-context', '1024', '--chunk', '128', '--expert-vram', str(args.expert_vram),
           '--cpu-threads', str(args.threads), '--no-warmup', '--tokens',
           ','.join(map(str, ids)), '--n', str(args.max_tokens)]
    env = os.environ.copy()
    if args.adaptive_experts:
        env.pop('STRATA_Q36_FIXED_SLOTS', None)
    else:
        env['STRATA_Q36_FIXED_SLOTS'] = '1'
    env['CUDA_LAUNCH_BLOCKING'] = '0' if args.async_cuda else '1'
    print('Experimental causal demo; the full Clover diffusion decoder is not implemented.', file=sys.stderr)
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, env=env)
    if result.returncode:
        return result.returncode
    line = next((line for line in result.stdout.splitlines() if line.startswith('gen:')), None)
    if line is None:
        sys.exit('Engine did not produce token output')
    generated = [int(token) for token in line[4:].split()]
    print(tokenizer.decode(generated, skip_special_tokens=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
