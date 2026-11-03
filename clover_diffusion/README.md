# Clover 块扩散：开发状态

当前网页可在 AR 和实验性块扩散之间切换。块扩散已接入完整模型，并在本机 Q4 模型上测试了生成、跨块提交和停止。完整模型与官方的逐层数值对齐仍未完成，不能把独立算子测试结果当作完整模型等价验证。

## 已实现

- `decoding.py`：固定宽度块的置信度解码、完整块完成后再截断或终止、提交已完成的历史块，以及专家 coreset 的参考实现。运行时仍需提供真实模型的预览、采样和因果提交。
- `cuda/preview_conv.cuh`：读取但不修改历史状态的双向卷积与 SiLU。
- `cuda/preview_gdn.cuh`：单请求、全有效块、128 维头的双向 GDN 和历史桶。采用本机 Clover GGUF 的 tiled value-head 排列；原始 Hugging Face 模型使用 grouped 排列，转换器对 value 相关权重做了重排。
- `cuda/preview_coreset.cuh`：单请求、全有效块、256 专家的 coreset。它以正确性为目标，尚未做性能优化。
- Strata 已加载并使用右卷积、历史桶和 logit scale 权重；块预览接入了双向注意力和专家 coreset，并在完成块后因果提交。

各算子的输入精度、缓存布局、激活舍入及层之间行为仍需要进行完整的逐层验证。

## 测试

在本机执行：

```bash
clover_diffusion/cuda/run_tests.sh
```

需要 CUDA 编译器、可用的 NVIDIA GPU、支持 CUDA 的 PyTorch 以及 `/tmp/clover-hf/modeling_clover_moe.py`；缺少官方文件时脚本会下载。脚本当前以本机 GPU 的 `sm_86` 构建。测试含独立 CPU 双向卷积数值比较，以及从官方 Clover 源码提取的双向卷积、GDN 和 coreset 参考函数比较。GPU 算子测试只覆盖单请求、全有效块；尚未覆盖批处理、不同有效长度或完整模型前向。

## 尚需完成

1. 与官方模型比较逐层状态、logits、完整生成质量和各种精度舍入。
2. 增加抽样参数、批处理及更广泛的长上下文和取消测试。
3. 测量并优化块解码速度、显存与内存占用，研究热点专家和更高效的 CPU/GPU 调度。

同一短提示、48 token 上限的两次测试中，开启热点专家预加载时实际生成分别为 38 和 35 tokens，约 5.9 和 15.1 tokens/s；关闭时为 36 和 41 tokens，约 4.7 和 11.5 tokens/s。两种策略生成内容和去噪轮数可能不同，样本很少，不能视为严格基准；尚未达到 100+ tokens/s。网页服务默认开启此优化，设置 `STRATA_CLOVER_PREVIEW_REBALANCE=0` 可关闭。
