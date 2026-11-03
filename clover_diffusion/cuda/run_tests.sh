#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p /tmp/clover-hf
if [[ ! -s /tmp/clover-hf/modeling_clover_moe.py ]]; then
  curl --retry 5 --retry-all-errors -fsSL \
    'https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/resolve/main/modeling_clover_moe.py' \
    -o /tmp/clover-hf/modeling_clover_moe.py
fi
if [[ ! -s /tmp/clover-hf/sampling_clover_moe.py ]]; then
  curl --retry 5 --retry-all-errors -fsSL \
    'https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/resolve/main/sampling_clover_moe.py' \
    -o /tmp/clover-hf/sampling_clover_moe.py
fi
nvcc -std=c++17 -O2 -arch=sm_86 clover_diffusion/cuda/preview_conv_test.cu -o /tmp/clover-preview-conv-test
nvcc -std=c++17 -O2 -arch=sm_86 -shared -Xcompiler -fPIC clover_diffusion/cuda/preview_conv_test_wrapper.cu -o /tmp/clover-preview-conv.so
nvcc -std=c++17 -O2 -arch=sm_86 -shared -Xcompiler -fPIC clover_diffusion/cuda/preview_gdn_test_wrapper.cu -o /tmp/clover-preview-gdn.so
nvcc -std=c++17 -O2 -arch=sm_86 -shared -Xcompiler -fPIC clover_diffusion/cuda/preview_coreset_test_wrapper.cu -o /tmp/clover-preview-coreset.so
nvcc -std=c++17 -O2 -arch=sm_86 -shared -Xcompiler -fPIC clover_diffusion/cuda/preview_penalties_test_wrapper.cu -o /tmp/clover-preview-penalties.so
nvcc -std=c++17 -O2 -arch=sm_86 -shared -Xcompiler -fPIC clover_diffusion/cuda/preview_accept_test_wrapper.cu -o /tmp/clover-preview-accept.so
/tmp/clover-preview-conv-test
python clover_diffusion/cuda/preview_conv_test.py
python clover_diffusion/cuda/preview_gdn_test.py
python clover_diffusion/cuda/preview_coreset_test.py
python clover_diffusion/cuda/preview_penalties_test.py
python clover_diffusion/cuda/preview_accept_test.py
python -m unittest discover -s tests -p test_clover_diffusion.py -v
