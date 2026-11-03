# Clover 3080 Ti 推理实验（2026-11-02）

这些是本机单请求的开发测试，不是跨机器可复现的标准基准。模型为 Clover-35B-A3B Q4_K_M，RTX 3080 Ti 12 GiB、上下文 8192、专家在 GPU/CPU 混合，CPU 配额 12 核；多次请求的专家缓存会随历史变化。JSON 保留固定测试提示、完整回复、首 token 延迟、解码和端到端速度以及草稿接受统计。对比时输出 token 数和内容可能不同，不应将单次数字解释成严格的加速比或质量证明。

## 测量与结论

- 纯块扩散、官方 `threshold=0.95`、32 token 块的写作样例：192 tokens 共 190 次预览，即每块 31.67 轮。这里一个位置在多数轮次才确定，因此吞吐量低。降低阈值到 0.8 或 0.6 仍分别需要约 30.75 和 28.25 轮/块，不能解决主要问题。
- 固定将块扩散压到 16、8、4 轮会改变算法。8 轮和 4 轮的写作样例出现严重重复和乱码，16 轮也有明显缺字和重复。该实验不在界面提供，不能作为质量合格的提速结果。
- 扩散自推测：Clover 自身的双向预览产生草稿，因果前向批量验证。严格接受只保留因果 argmax；熵阈值接受遵循 llama.cpp 的 `p(draft) >= min(1, 0.6 * exp(-entropy))`，可能接受非贪心 token。二者的输出语义均不同于纯块扩散。
- 固定专家布局、关闭响应惩罚时，中文写作与二分查找两个提示各生成 96 tokens，2/4/8 位置严格自推测与逐 token AR 的解码文本完全相同；这只是小样本验证，不是全面等价证明。
- 相同固定提示的对照中，4 位置自推测通常快于 8 位置；二次细化草稿虽增加接受数，但额外前向抵消了收益。跳过低排名未命中专家的实验未见稳定收益，且改变结果，默认关闭。
- 128 行以内均走 CPU/GPU 混合专家计算时，约 0.97 GiB 的流式专家 staging 在本机配置中不可达。回收后 7 GiB 专家预算下每层常驻专家从 84 提高到 98。`STRATA_CLOVER_KEEP_STAGING=1` 可用于对照。
- 10 个 CPU 工作线程参数比 8 个略快，11 个参数在本机没有进一步收益。`--threads` 的具体池线程数量看启动日志。
- 7 GiB 与 8 GiB 专家预算、10 线程、4 位置、熵阈值样例见 `clover-nostage-w4-threads10.json` 和 `clover-nostage-w4-vram8.json`。8 GiB 样例约 64–81 tokens/s；上下文 8192 时本机只剩约 225 MiB 显存，不能视为安全余量充足的通用配置。
- 长篇单次样例：上限 2048、最少 1400 tokens，扩散自推测的熵阈值模式生成 1976 tokens、约 2799 字符，正常结束；解码约 51.7 tokens/s。字数不等于 tokens，最少 tokens 会抑制提前结束符，也可能迫使模型继续输出。

`benchmarks/clover/bench_diffusion.py` 可在空闲服务上重复这些固定任务；要比较不同设置，请重新启动服务、交错重复并检查 `finish`、文本及端到端延迟。`STRATA_CLOVER_PROFILE=1` 用 CUDA event 测首轮层内耗时（包含流中的 host 等待），不是独占 GPU kernel 时间；本机一次 4 行冷启动样例，草稿的 GDN 2.36 ms、注意力 0.85 ms、MoE 19.50 ms。`ncu` 硬件计数器因 `ERR_NVGPUCTRPERM` 不可用。

实现参考：[Clover 模型卡](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview)、[Clover 官方后端说明](https://github.com/DiffuSpace/Clover-35B-A3B-Preview)、[Strata 设计和测量](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)。当前仍未证明达到 120 tokens/s 的目标，尤其没有建立覆盖长上下文、长输出和多种任务的稳定质量数据。

## Strata DFlash2 外部草稿器实验

参考 Strata 的专家驻留、CPU/GPU 混合与 DFlash2 实现，移植 Qwen3.6-35B-A3B 的 6 层、2048 维草稿器；使用 Q8 pack，占约 755 MiB 显存（另 8 MiB 特征缓冲）。PyTorch FP32 随机特征参考与 GPU 的 8 行余弦相似度 0.99467–0.99802、7 个草稿 token 一致；这只是草稿器内核验证，不代表它适用于 Clover。

在固定专家布局、关闭惩罚、8192 上下文的单次实验中，严格接受的 DFlash2 输出与 AR 三个样例解码文本相同，但 DFlash2 只有 23.6、28.0、42.9 tokens/s，AR 分别为 77.5、77.2、79.6 tokens/s；DFlash2 接受率分别为 22/518、38/378、22/98。熵阈值实验约 25.7、28.4、43.0 tokens/s，并可接受非贪心 token。前两次每轮草稿约 3 ms，验证约 52–57 ms，大部分验证时间来自 CPU 专家。实验数据见 `dflash2-exact.json`、`dflash2-entropy.json`、`dflash2-ar.json`。输出长度、内容和短序列缓存效应使这些数据仅适合作为探索性证据。

该草稿器是在 Qwen 上训练的，接受率低可能与 Clover 的隐藏特征/训练分布差异或适配细节有关，目前没有证明具体原因。该实验通过 `STRATA_CLOVER_DFLASH_PACK` 显式启用，仅供研究，前端不展示；不应视为纯扩散、质量已验证或速度优化。下一步应优先研究更匹配 Clover 的草稿训练、减少 MoE 验证 CPU 缺页和计算量，并使用长文本、多任务、冷/热缓存交错测试。

## 块级专家统计与固定候选（32 token，2026-11-02）

测试命令：`python benchmarks/clover/bench_diffusion.py --mode diffusion --tokens 96 --output FILE`。原模式为默认每层最多晋升 8 个热点专家；以 `STRATA_CLOVER_TRACE_ROUTES=1` 输出每块、40 层的统计。`block-cache.json` 和 `block-cache-routes.log` 保存结果。武侠提示首两块，相邻轮次已选专家集合的交集/并集分别为 27086/32696（82.8%）和 24125/29072（83.0%）；第一轮专家覆盖后续路由条目的 61.4% 和 67.4%，全块不同专家并集平均每层 55.6 和 47.3。其他任务和块变化很大，这不是普遍规律。GPU 命中率按路由条目计，不等于每层专家覆盖率。

提高块第一轮后晋升的最大数量到每层 32 个（`STRATA_CLOVER_PREVIEW_PROMOTE=32`），武侠样例 15.0 tok/s，原配置 14.7 tok/s；非固定路由的 GPU/CPU 混合计算存在数值差异，输出和去噪轮数可变化，这一次不能证明稳定加速。

实验性固定显存候选：`STRATA_CLOVER_BLOCK_FIXED=1 STRATA_CLOVER_PREVIEW_PROMOTE=32`。第一轮沿用原始 coreset，随后先提升热点；余下各轮禁止该层未常驻专家参与路由，每层常驻候选集在块内保持固定。所有候选权重计算和概率归一化均来自受限路由，是近似算法。三个样本 17.9、15.5、66.3 tok/s；原样本 14.7、11.2、57.0 tok/s，但输出长度、内容和去噪轮数不尽相同，不能作为严格加速比。武侠样例 95 轮、96 tokens，生成期间（含块间提交）GPU 专家条目 986046/993280=99.27%，CPU 专家时间约 212 ms，基线约 2317 ms。可见输出有缺字，256-token 样本可持续生成但不足以确认质量。保留默认关闭，不能用于自推测或 AR 的精确性声明。`block-fixed*.json` 和 `block-fixed-routes.log` 为完整数据。

固定显存专家时 32 行冷启动单次预览分解为 GDN 9.62 ms、attention 2.30 ms、MoE 31.70 ms（CUDA 流事件包含主机等待）。针对 coreset 增加了按行并行实现（`STRATA_CLOVER_PARALLEL_CORESET=1`），与旧内核、官方函数在多组输入逐项相同，固定路由三次端到端文本相同；同步微测旧版约 0.186 ms/次、新版约 0.138 ms/次（含 host 同步，后者还含 scratch 分配），整机短样本未见稳定提速，因此暂不默认开启。

官方 Clover 参考实现对 preview 有 `t <= config.block_size` 且整除的检查，配置为 32；我们的 GDN `weights[33]`、输出 logits、host staging 等也按 32 构造。因此 64/128 块不是简单调参，属于超出参考约束的探索，尚未实现或测试。
