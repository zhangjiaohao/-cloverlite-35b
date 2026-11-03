"""Numerical GPU regression against the published Hugging Face Clover function.

Build the test wrapper to /tmp/clover-preview-gdn.so, and download the official
modeling_clover_moe.py to /tmp/clover-hf/ before running this script.
"""
import ast
import ctypes
from pathlib import Path
import torch

source = Path('/tmp/clover-hf/modeling_clover_moe.py').read_text()
module = ast.parse(source)
node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'bd_gdn_local_memory')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<official Clover>', 'exec'), namespace)
reference = namespace['bd_gdn_local_memory']
lib = ctypes.CDLL('/tmp/clover-preview-gdn.so')
run = lib.clover_test_preview_gdn
run.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 3
run.restype = ctypes.c_int
warp = lib.clover_test_preview_gdn_warp
warp.argtypes = run.argtypes
warp.restype = ctypes.c_int

torch.manual_seed(42)
for T, HK, HV in [(1, 2, 4), (4, 2, 4), (16, 4, 8), (32, 16, 32)]:
    D = 128
    qkv = torch.randn(T, (2 * HK + HV) * D, device='cuda').to(torch.bfloat16).float()
    state = torch.randn(D, HV, D, device='cuda') * 0.1
    old_state = state.clone()
    gate = -torch.rand(T, HV, device='cuda')
    scale = torch.randn(HV, device='cuda')
    bias = torch.randn(HV, device='cuda')
    history_c = torch.randn(HV, device='cuda')
    out = torch.empty(T, HV, D, device='cuda')
    # HF has grouped V heads; Clover's GGUF converter tiles them.
    # Convert every V-headed tensor before calling the Strata-layout kernel.
    tile = torch.arange(HV, device='cuda').reshape(HK, HV // HK).T.reshape(-1)
    inverse = torch.argsort(tile)
    grouped_qkv, grouped_state, grouped_gate = qkv.clone(), state.clone(), gate.clone()
    grouped_scale, grouped_bias, grouped_c = scale.clone(), bias.clone(), history_c.clone()
    qkv[:, 2 * HK * D:] = qkv[:, 2 * HK * D:].reshape(T, HV, D)[:, tile].reshape(T, HV * D)
    state = state[:, tile, :].contiguous()
    old_state = state.clone()
    gate, scale, bias, history_c = gate[:, tile].contiguous(), scale[tile].contiguous(), bias[tile].contiguous(), history_c[tile].contiguous()
    torch.cuda.synchronize()
    status = run(*(x.data_ptr() for x in (qkv, state, gate, scale, bias, history_c, out)), T, HK, HV)
    if status:
        raise RuntimeError(f'CUDA error {status}')
    q = grouped_qkv[:, :HK * D].reshape(1, T, HK, D)
    k = grouped_qkv[:, HK * D:2 * HK * D].reshape(1, T, HK, D)
    v = grouped_qkv[:, 2 * HK * D:].reshape(1, T, HV, D)
    expected = reference(q, k, v, grouped_gate, clean_ssm_states=grouped_state.permute(1, 2, 0).unsqueeze(0),
                         cache_indices=torch.tensor([0], device='cuda'),
                         active_valid_mask=torch.ones(T, device='cuda', dtype=torch.bool),
                         scale=D ** -0.5, softmax_logit_scale=grouped_scale, softmax_history_bias=grouped_bias,
                         softmax_history_c=grouped_c).reshape(T, HV, D)
    torch.testing.assert_close(out[:, inverse], expected, atol=2e-6, rtol=2e-5)
    fast = torch.empty_like(out)
    status = warp(*(x.data_ptr() for x in (qkv, state, gate, scale, bias, history_c, fast)), T, HK, HV)
    if status:
        raise RuntimeError(f'warp CUDA error {status}')
    torch.testing.assert_close(fast[:, inverse], expected, atol=2e-6, rtol=2e-5)
    assert torch.equal(state, old_state)
    print(f'T={T}, HK={HK}, HV={HV}: max abs error {(out[:, inverse] - expected).abs().max().item():.3g}; state unchanged')
