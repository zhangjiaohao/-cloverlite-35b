"""Compare the CUDA convolution to the published Hugging Face Clover function."""
import ast
import ctypes
from pathlib import Path
import torch
import torch.nn.functional as F

module = ast.parse(Path('/tmp/clover-hf/modeling_clover_moe.py').read_text())
node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'bd_gdn_conv1d')
namespace = {'torch': torch, 'F': F}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<official Clover>', 'exec'), namespace)
reference = namespace['bd_gdn_conv1d']
run = ctypes.CDLL('/tmp/clover-preview-conv.so').clover_test_preview_conv
run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2
run.restype = ctypes.c_int
torch.manual_seed(10)
for T in (1, 4, 16, 32):
    C = 8192
    qkv = torch.randn(T, C, device='cuda').to(torch.bfloat16).float()
    history = torch.randn(C, 3, device='cuda').to(torch.bfloat16).float()
    before = history.clone()
    forward = torch.randn(C, 4, device='cuda').to(torch.bfloat16).float()
    right = torch.randn(C, 3, device='cuda').to(torch.bfloat16).float()
    out = torch.empty_like(qkv)
    torch.cuda.synchronize()
    status = run(*(x.data_ptr() for x in (qkv, history, forward, right, out)), C, T)
    if status:
        raise RuntimeError(f'CUDA error {status}')
    expected = reference(qkv, clean_conv_states=history.unsqueeze(0),
                         cache_indices=torch.tensor([0], device='cuda'),
                         conv_weights=forward.unsqueeze(1), block_size=T,
                         active_valid_mask=torch.ones(T, dtype=torch.bool, device='cuda'),
                         conv_right_weight=right.unsqueeze(1))
    torch.testing.assert_close(out, expected, atol=3e-6, rtol=3e-5)
    assert torch.equal(history, before)
    print(f'T={T}, channels={C}: max abs error {(out - expected).abs().max().item():.3g}; history unchanged')
