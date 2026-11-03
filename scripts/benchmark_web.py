#!/usr/bin/env python3
"""Measure the already-running local engine; only completed decode steps count."""
import argparse
import json
import statistics
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:7860')
p.add_argument('--runs', type=int, default=3)
p.add_argument('--tokens', type=int, default=256)
p.add_argument('--prompt', default='Explain in detail how mixture of experts models work, including expert routing, training, inference and efficiency.')
a = p.parse_args()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
results = []
for i in range(a.runs):
    data = json.dumps({'messages': [{'role': 'user', 'content': a.prompt}], 'max_tokens': a.tokens}).encode()
    req = urllib.request.Request(a.url + '/api/chat', data=data, headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    response = opener.open(req, timeout=300)
    text = ''
    done = None
    for raw in response:
        event = json.loads(raw)
        if event['type'] == 'text': text = event['text']
        elif event['type'] == 'done': done = event
        elif event['type'] == 'error': raise RuntimeError(event['error'])
    if not done or done['finish'] == 'error': raise RuntimeError(f'Unsuccessful run: {done}')
    steps = max(0, done['tokens'] - 1)
    speed = 1000 * steps / done['decode_ms'] if done['decode_ms'] else 0
    result = {'run': i + 1, 'committed_tokens': done['tokens'], 'decode_steps': steps,
              'decode_ms': done['decode_ms'], 'decode_tokens_per_s': round(speed, 2),
              'prefill_ms': done['prefill_ms'], 'total_s': round(time.monotonic() - start, 3),
              'finish': done['finish'], 'answer': text}
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
print(json.dumps({'median_decode_tokens_per_s': statistics.median(r['decode_tokens_per_s'] for r in results)}, ensure_ascii=False))
