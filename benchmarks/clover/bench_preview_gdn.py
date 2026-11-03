#!/usr/bin/env python3
"""Compare preview GDN kernels on an idle SM86 GPU after building the test wrapper."""
import ctypes
import json
import torch

lib = ctypes.CDLL('/tmp/clover-preview-gdn.so')
bench = lib.clover_bench_preview_gdn
bench.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5
bench.restype = ctypes.c_float
torch.manual_seed(42)
for t in (2, 4, 8, 16, 32):
    hk, hv, d = 16, 32, 128
    tensors = [torch.randn(t, (2 * hk + hv) * d, device='cuda'),
               torch.randn(d, hv, d, device='cuda'), torch.randn(t, hv, device='cuda'),
               torch.randn(hv, device='cuda'), torch.randn(hv, device='cuda'),
               torch.randn(hv, device='cuda'), torch.empty(t, hv, d, device='cuda')]
    args = [x.data_ptr() for x in tensors] + [t, hk, hv]
    samples = {0: [], 1: []}
    for iteration in range(6):
        for variant in ((0, 1) if iteration % 2 == 0 else (1, 0)):
            ms = bench(*args, variant, 100)
            if ms < 0:
                raise RuntimeError('CUDA benchmark failed')
            samples[variant].append(ms)
    print(json.dumps({'T': t, 'baseline_ms': samples[0], 'warp_ms': samples[1]}), flush=True)
