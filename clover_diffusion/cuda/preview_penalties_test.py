"""Compare the GPU penalty logic with the authors' response-only implementation."""
import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace
import torch

source = Path('/tmp/clover-hf/sampling_clover_moe.py').read_text()
module = ast.parse(source)
node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'CloverPenaltyState')
ns = {'torch': torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<official Clover>', 'exec'), ns)
gen = SimpleNamespace(presence_penalty=.9, frequency_penalty=0., repetition_penalty=1.05,
                      dllm_length_penalty=10., dllm_length_penalty_start=8192,
                      dllm_length_penalty_window=8192, dllm_think_end_bias=10.,
                      dllm_think_end_start=4096, dllm_think_end_window=4096,
                      dllm_think_end_token_id=248069)
run = ctypes.CDLL('/tmp/clover-preview-penalties.so').clover_test_penalties
run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
run.restype = ctypes.c_int
vocab, T = 248070, 4
torch.manual_seed(42)
for committed, closed in [(0, False), (4096, False), (8192, False), (12288, True), (16384, False)]:
    state = ns['CloverPenaltyState'](gen, vocab, torch.device('cuda'), [248044, 248046])
    state.length = committed
    state.counts[[1, 2, 155, 248044]] = 2
    block = torch.full((1,T), 151669, dtype=torch.int64, device='cuda')
    if closed: block[0, 0] = 248069
    logits = torch.randn(T, vocab, device='cuda')
    expected = state.apply(logits.clone(), block, 151669)
    actual = logits.clone()
    counts = state.counts.to(torch.int32)
    torch.cuda.synchronize()
    status = run(actual.data_ptr(), counts.data_ptr(), T, vocab, committed, closed)
    if status: raise RuntimeError(f'CUDA error {status}')
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
    print(f'committed={committed}, closed={closed}: max abs error {(actual-expected).abs().max().item():.3g}')
