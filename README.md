# Clover on Strata

基于 [Strata](https://github.com/Niko1221/Strata) 的 Clover-35B-A3B Q4_K_M 本地推理实验：使用有限的 GPU 显存与主机内存，探索 MoE 模型的因果推理、块扩散与专家缓存优化，并提供本地 Web 聊天界面。

> **研发状态：实验项目。** 纯扩散、AR 和扩散草稿验证具有不同语义。本项目尚未在 12 GB 显存机器上实现或验证稳定的 120+ tokens/s 块扩散；输出质量、长上下文及官方后端的整模型数值对齐仍在研究中。
> clover原生模型未做优化版的生成速度：
> <img width="922" height="668" alt="截屏2026-11-03 14 29 28" src="https://github.com/user-attachments/assets/e9d55c1e-7953-4157-a2cf-ef3f968c1ae8" />
> 本方案生成速度：
<img width="1425" height="804" alt="截屏2026-11-03 14 43 05" src="https://github.com/user-attachments/assets/82452087-42bb-4e62-b113-8f57bd21fe0b" />
<img width="1176" height="802" alt="截屏2026-11-03 14 41 27" src="https://github.com/user-attachments/assets/1dd0a56c-e205-4988-b142-993a3c25231e" />


## 功能

- **AR 因果生成**：逐 token 生成。
- **块扩散**：对 32-token 块进行多轮并行去噪；每块所需轮数取决于内容和置信度。
- **扩散草稿 + 验证**：Clover 生成草稿，再用因果路径验证；它不是纯块扩散。可选择严格或熵阈值接受，后者可能接受非贪心 token。
- **资源监控与聊天**：多轮对话、流式输出、取消生成、GPU／内存监控。
- **MoE 混合推理**：热点专家常驻显存，其他专家在主机内存中并由 CPU 参与计算；尚未完成通用的磁盘专家调度。

### 性能状态

在一台 RTX 3080 Ti 12 GB、Clover-35B-A3B Q4_K_M、单请求的开发测试中，纯扩散武侠写作样例约为 **14–18 tokens/s**；其他任务、缓存状态和去噪轮数可能明显不同。扩散自推测短样例约为 **70–81 tokens/s**，与纯扩散不是同一算法，不能直接作为它的速度。数据不是通用基准或性能保证；完整条件和文本见 [实验记录](benchmarks/clover/research-2026-11-02/README.md)。

## 环境

- Linux、NVIDIA GPU、CUDA Toolkit（含 `nvcc`）、CMake 3.24 或更新版本、C/C++ 编译器、Python 3。
- 已验证的主要开发环境：RTX 3080 Ti（CUDA 架构 `86`）、约 12 GB 显存，以及足够容纳约 18 GiB 专家权重和系统开销的主机内存。其他硬件和内存配置需自行测量。
- Clover 的 GGUF Q4_K_M 模型文件需自行获取；仓库不包含模型权重。模型信息及许可证见 [Clover 模型主页](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview)。
- 首次 CMake 配置默认从 GitHub 获取 Strata 固定版本所需的 llama.cpp / ggml 源码，需要网络和 Git；也可通过 `STRATA_GGML_DIR` 指定兼容的本地 llama.cpp 源码目录。

## 安装与运行

从仓库根目录执行：

```bash
python3 -m venv .venv-clover
.venv-clover/bin/pip install tokenizers cmake

.venv-clover/bin/cmake -S third_party/Strata -B third_party/Strata/build-clover \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_MMQ_KQUANTS=ON
.venv-clover/bin/cmake --build third_party/Strata/build-clover --target strata-q36 -j 4
```

`86` 适用于 RTX 3080 Ti；其他 NVIDIA GPU 应改为对应的 CUDA 架构。若已有兼容的 llama.cpp 源码，可在配置时追加 `-DSTRATA_GGML_DIR=/path/to/llama.cpp`；本机开发使用的 ggml 版本记录在 [VERSION.txt](third_party/Strata/third_party/ggml/VERSION.txt)。

将模型放在仓库**上一级**的 `models/Clover-35B-A3B-Q4_K_M.gguf`，或用 `--model` 指定路径：

```bash
./scripts/run_clover_web.sh --model /path/to/Clover-35B-A3B-Q4_K_M.gguf \
  --port 7860 --context 8192 --max-output-tokens 4096
```

打开 <http://127.0.0.1:7860>，等待模型加载；终端按 `Ctrl+C` 停止服务。默认不带参数时，服务使用 `--context 4096`、`--max-output-tokens 4096`、`--expert-vram 7.5` GiB 和 `--threads 10`；这些是启动配置，不保证适合所有机器。

```bash
./scripts/run_clover_web.sh --help
./scripts/run_clover_web.sh --model /path/to/model.gguf \
  --context 8192 --max-output-tokens 4096 --expert-vram 7 --threads 8
```

上下文需要容纳提示、历史和生成空间；块扩散还须为最后一个完整预览块预留空间。增加上下文会占用更多显存，可能需要降低专家显存预算。Web 服务绑定本地回环地址；远程使用可通过 SSH 本地端口转发。公开映射时，请自行配置访问控制。

## 验证与基准

```bash
.venv-clover/bin/python -m unittest discover -s tests
# 服务已启动、GPU 空闲时运行：
.venv-clover/bin/python benchmarks/clover/bench_diffusion.py \
  --mode diffusion --tokens 96 --output /tmp/clover-bench.json
```

基准会保存生成文本、轮数和耗时；比较优化时需固定模型、提示、参数和缓存状态，并同时检查质量。CUDA 算子与官方参考对照需要额外的 PyTorch CUDA、`nvcc` 和官方参考源码；说明见 [clover_diffusion](clover_diffusion/README.md) 与 [实验记录](benchmarks/clover/research-2026-11-04/README.md)。

## 项目结构

```text
scripts/             Web 服务、模型加载及启动脚本
web/                 聊天与监控页面
clover_diffusion/    块扩散逻辑、CUDA 原型及参考测试
benchmarks/clover/   性能与质量实验记录
tests/               Python 回归测试
third_party/Strata/  包含 Clover 本地适配的 Strata 源码快照
```

Strata 源码基于上游 commit `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`，本仓库包含本地修改，并非上游官方发布。保留其 [LICENSE](third_party/Strata/LICENSE) 和第三方许可；模型权重适用各自的许可证。项目使用的 Web 界面参考 Strata，并调用其部分界面资源。

## 已知限制

- 生成模式为实验实现；纯扩散当前为单会话贪心生成，可能出现提前结束、重复或缺字。
- 模型与官方实现的完整分层数值对齐尚未完成。CPU/GPU 专家计算、缓存布局或近似实验可能改变生成文本。
- 当前研究用固定候选专家、并行 GDN、外部 DFlash2 等开关默认关闭；不把它们的速度或输出视为已验证的默认结果。
- 本地测试结论不能推断其他 GPU、量化版本或长上下文的性能。
