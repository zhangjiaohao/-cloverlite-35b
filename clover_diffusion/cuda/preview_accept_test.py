"""Compare entropy-threshold draft acceptance with a torch reference."""
import ctypes
import torch
run = ctypes.CDLL('/tmp/clover-preview-accept.so').clover_test_accept
run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
run.restype = ctypes.c_int
torch.manual_seed(234)
for T, V, scale in [(8, 257, 1.), (8, 248070, 1.), (8, 248070, 4.)]:
    logits = torch.randn(T, V, device='cuda') * scale
    draft = torch.randint(V, (T,), device='cuda', dtype=torch.int32)
    # Include the argmax and a low-probability candidate.
    draft[1] = logits[0].argmax()
    draft[2] = logits[1].argmin()
    out = torch.empty(T - 1, device='cuda')
    torch.cuda.synchronize()
    status = run(logits.data_ptr(), V, draft.data_ptr(), T, out.data_ptr())
    if status: raise RuntimeError(f'CUDA error {status}')
    log_probs = torch.log_softmax(logits[:-1], dim=1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=1)
    p = probs.gather(1, draft[1:].long().unsqueeze(1)).squeeze(1)
    expected = (p >= torch.minimum(torch.ones_like(p), .6 * (-entropy).exp())).float()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    print(f'T={T} V={V} scale={scale}: exact acceptance decisions')
