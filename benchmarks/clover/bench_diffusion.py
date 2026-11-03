#!/usr/bin/env python3
"""End-to-end Clover speed/quality fixtures. Run on an idle local service.

Example: python benchmarks/clover/bench_diffusion.py --mode diffusion_spec --tokens 128
Reports decoded token count, finish, TTFT, decode and end-to-end rates, and the
text itself for inspection. These are single-request numbers, not batch throughput.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request

PROMPTS = [
    '写一篇2000字的武侠小说',
    '用通俗语言解释混合专家模型如何工作，给出一个具体例子。',
    '用 Python 实现二分查找，并说明时间复杂度。',
]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:7860')
    p.add_argument('--mode', choices=['ar', 'diffusion', 'diffusion_spec', 'dflash'], default='diffusion')
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--min-tokens', type=int, default=0)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--spec-accept', choices=['exact', 'entropy'], default='exact')
    p.add_argument('--spec-delta', type=float, default=0.6)
    p.add_argument('--spec-width', type=int, choices=[2, 4, 8], default=4)
    p.add_argument('--spec-refine', type=int, default=0)
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    out = []
    for repeat in range(args.repeat):
        for prompt in PROMPTS:
            payload = {'messages': [{'role': 'user', 'content': prompt}], 'mode': args.mode,
                       'max_tokens': args.tokens, 'min_tokens': args.min_tokens, 'diffusion_steps': args.steps,
                       'spec_accept': args.spec_accept, 'spec_delta': args.spec_delta, 'spec_width': args.spec_width,
                       'spec_refine': args.spec_refine}
            req = urllib.request.Request(args.url + '/api/chat', json.dumps(payload, ensure_ascii=False).encode(),
                                         headers={'Content-Type': 'application/json'})
            start = time.monotonic(); ttft = None; text = ''; done = None
            with opener.open(req, timeout=600) as response:
                for line in response:
                    event = json.loads(line)
                    if event['type'] == 'text':
                        if ttft is None: ttft = (time.monotonic() - start) * 1000
                        text = event['text']
                    elif event['type'] == 'done': done = event
                    elif event['type'] == 'error': raise RuntimeError(event['error'])
            if done is None: raise RuntimeError('stream ended without DONE')
            elapsed = (time.monotonic() - start) * 1000
            record = {'repeat': repeat, 'prompt': prompt, 'mode': args.mode, 'steps': args.steps, 'spec_accept': args.spec_accept, 'spec_delta': args.spec_delta, 'spec_width': args.spec_width, 'spec_refine': args.spec_refine,
                      'ttft_ms': ttft, 'elapsed_ms': elapsed, 'visible_chars': len(text), 'text': text,
                      'decode_tokens_per_second': done['tokens'] * 1000 / max(1, done['decode_ms']),
                      'end_to_end_tokens_per_second': done['tokens'] * 1000 / max(1, elapsed), **done}
            out.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if args.output: args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')

if __name__ == '__main__': main()
