# 块扩散置信度初步诊断

使用 3080 Ti、Q4_K_M、context 8192、专家预算 7.5 GiB、默认 threshold=0.95、每提示最多输出 32 tokens。运行 `STRATA_CLOVER_TRACE_CONFIDENCE=1 ./scripts/run_clover_web.sh --port 7860 --context 8192 --max-output-tokens 4096`，再运行 `python benchmarks/clover/bench_diffusion.py --mode diffusion --tokens 32 --output confidence-results.json`。服务加载一次，三个提示依序运行，专家缓存会变化。

| 任务 | 第一个块轮数 | 无位置超过 0.95 的轮数 | 首轮 pending 平均置信度 |
| --- | ---: | ---: | ---: |
| 武侠写作 | 31 | 29 | 0.1360 |
| MoE 解释 | 31 | 26 | 0.1527 |
| 二分查找 | 7 | 2 | 0.2212 |

采集的概率均有限且在 [0,1]；不证明概率数值正确，也不能区分 Q4、实现和模型本身的影响。仅三个短样本，不能推广到所有提示。接下来需要使用相同 GGUF 权重和完全相同的块状态对照整模型 logits。日志中每轮记录 pending、平均及最大概率、超过阈值数和非法值数；最后一块即使遇到提前 EOS 也可能已完成完整块预览。
