# Clover / Strata 实验记录

2026-11-01；RTX 3080 Ti 12 GiB，容器 CPU 额度 12 核、RAM 上限 90 GiB。模型为本机 `Clover-35B-A3B-Q4_K_M.gguf`，Strata `clover-adaptation` 分支的未提交更改。

当前仅实现了**实验性因果生成**，没有实现 Clover 的双向块扩散、self-draft、正式服务和磁盘专家缓存。输出合理不代表已经验证完整 Clover 推理语义。

运行（从 `codes` 目录）：

```bash
.venv-clover/bin/python scripts/clover_demo.py '用中文简单解释什么是混合专家模型。' --max-tokens 128
```

可选的热点调度实验（异步 CUDA 和专家动态提升仍需更多稳定性测试）：

```bash
.venv-clover/bin/python scripts/clover_demo.py 'Explain what a mixture of experts model is in two sentences.' --max-tokens 96 --expert-vram 8 --adaptive-experts --async-cuda
```

脚本根据 GGUF 自带的 Qwen35 tokenizer 元数据重建 tokenizer，使用单轮模板，不具备多轮聊天历史。每次运行重新加载、重排并驻留模型；若已删除虚拟环境，可执行 `python -m venv .venv-clover && .venv-clover/bin/pip install tokenizers`。使用本地构建的 `third_party/Strata/build-clover/strata-q36`。

`causal-demo.log` 是上面英文问题的一次运行，输出在 `causal-demo-output.txt`：25 个 prompt token、58 个生成 token，生成阶段耗时 631.3 ms；首个 token 来自 prompt 前向，故速率按剩余 57 / 0.6313 = 90.3 token/s 报告，不包含冷启动与 prefill。初始化专家约 21.2 s，prefill 约 0.60 s。单次短输出不能代表稳态性能，更不能宣称达到 100+ token/s。专家预算 8 GiB 包含约 0.97 GiB staging；实际专家槽约 6.96 GiB，98/256 每层，GPU 路由命中 57.5%；完整专家约 18.17 GiB 放在主机内存。

`expert-parity.log`：从本地 GGUF 的第 0 层（Q4_K/Q6_K）、第 5 层（Q4_K/Q4_K）各取一个实际专家，CPU/GPU 对照测试通过。`model_family_test` 和 `clover_expert_parity` 测试通过。官方 Clover llama.cpp 因果模式在同一 2+2 模板下回答 `4.`；本实验路径回答 `4`。输出相近不代表 logits 或所有提示已经对齐。官方 self-speculative 测试本机发生崩溃，尚未用于对照。

## 2026-11-02：常驻服务和热点专家实验

启动聊天页面：

```bash
./scripts/run_clover_web.sh --port 7860
```

浏览器访问 `http://127.0.0.1:7860`。若浏览器在另一台机器，请通过 SSH 本地端口转发连接；服务仅监听本机。引擎在后台加载后保持常驻，页面支持中文、多轮、流式输出和停止生成。可通过 `--context`、`--expert-vram`、`--threads` 配置。界面沿用 Strata 的配色设计资源，当前模式仍是实验性因果生成。

新策略在较小的替换预算下轮流从不同层挑选专家，防止每次都从最前层开始。仅 Clover 启用，Strata 的 Qwen 默认行为不变；Web 端默认启用。对比原顺序启动可使用 `--legacy-rebalance`。当前 Web 默认上下文 4096、专家与 staging 总预算 7 GiB、8 个 CPU 工作线程；这台机器实际是 84/256 个专家每层常驻 GPU，专家槽约 5.96 GiB、staging 约 0.97 GiB。

三次同一英文长提示、每次 256 生成 token 的测试，按 `(输出 token 数 - 1) / decode 时间` 计算：原策略分别 78.11、126.24、126.69 token/s；新策略 91.33、120.75、131.38 token/s。热身后的重复任务超过 100，但第一次和换题任务没有稳定超过 100。不同进程及先前缓存状态不完全相同，不能把差异视为严格配对结论。

换题单次实验：原策略中文 / 代码 / 历史分别 91.07 / 67.98 / 70.16 token/s；新策略分别 111.61 / 75.49 / 83.61 token/s。中文输出长度不同；所有结果只说明当前样例的测量范围，不能当成所有输入都提速。原始数据见 `web-speed*.jsonl`。运行 `python scripts/benchmark_web.py` 可以对当前运行的 Web 端点重复测速。

代码提示固定 64 个后续 token，旧策略与新策略各自执行 65 次前向并比较 logits：65/65 的 top-1 相同，平均 KL 0.000264，最大 KL 0.00687；完整统计见 `fair-teacher-forced.json`。这只覆盖一个固定序列，不等于全面质量验证。可用 `--sync-cuda` 调试异步执行，`--fixed-experts` 禁用动态提升。

界面复用了 Strata 仓库的 `serve/web/tokens.css` 和 `components.css`，源代码在 `web/` 与 `scripts/clover_web.py`；没有修改或发布上游 Strata。启动和运行日志默认输出到终端。模型不支持图片、工具调用或并发请求；超过上下文会明确报错。生成统计不包含首次 token 的 prefill。双向块扩散、自推测和磁盘后备仍未实现。
