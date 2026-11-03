"""Compare the CUDA coreset to the published Hugging Face Clover function."""
import ast
import ctypes
from pathlib import Path
import torch

module = ast.parse(Path('/tmp/clover-hf/modeling_clover_moe.py').read_text())
node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'coreset_torch')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<official Clover>', 'exec'), namespace)
reference = namespace['coreset_torch']
run = ctypes.CDLL('/tmp/clover-preview-coreset.so').clover_test_preview_coreset
run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
run.restype = ctypes.c_int
parallel = ctypes.CDLL('/tmp/clover-preview-coreset.so').clover_test_preview_coreset_parallel
parallel.argtypes = run.argtypes
parallel.restype = ctypes.c_int
torch.manual_seed(61)
for T, stride, top_k, top_p, ties in [(1, 256, 8, .9, False), (4, 257, 8, .9, False),
                                       (32, 256, 8, .9, False), (32, 257, 8, .5, True),
                                       (16, 256, 1, .99, False)]:
    logits = torch.randn(T, stride, device='cuda')
    if ties:
        logits[:, :256] = 0
    out = logits.clone()
    torch.cuda.synchronize()
    status = run(logits.data_ptr(), out.data_ptr(), T, stride, top_k, top_p)
    if status:
        raise RuntimeError(f'CUDA error {status}')
    expected = reference(logits[:, :256].unsqueeze(0), torch.ones(1, T, dtype=torch.bool, device='cuda'),
                         top_k, top_p).squeeze(0)
    torch.testing.assert_close(out[:, :256], expected, rtol=0, atol=0)
    fast = logits.clone()
    status = parallel(logits.data_ptr(), fast.data_ptr(), T, stride, top_k, top_p)
    if status:
        raise RuntimeError(f'parallel CUDA error {status}')
    torch.testing.assert_close(fast, out, rtol=0, atol=0)
    if stride == 257:
        assert torch.equal(out[:, 256], logits[:, 256])
    print(f'T={T}, stride={stride}, top_k={top_k}, top_p={top_p}, ties={ties}: exact match')

# The fixed-resident experiment presents -inf logits for excluded experts.
logits = torch.randn(32, 256, device='cuda')
logits[:, 105:] = -float('inf')
slow, fast = logits.clone(), logits.clone()
assert not run(logits.data_ptr(), slow.data_ptr(), 32, 256, 8, .9)
assert not parallel(logits.data_ptr(), fast.data_ptr(), 32, 256, 8, .9)
torch.testing.assert_close(fast, slow, rtol=0, atol=0)
print('fixed-resident logits: exact match')
