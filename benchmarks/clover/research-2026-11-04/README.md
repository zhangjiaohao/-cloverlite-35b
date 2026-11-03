# 双向 GDN warp 原型

`clover_preview_gdn_warp` 将不同 key 的分数分给 4 个 warp 并行计算，减少块内同步；浮点归约顺序变化，保留 `STRATA_CLOVER_GDN_WARP=1` 显式开启，默认仍为原内核。

官方 PyTorch 参考逐元素测试通过，T=1/4/16/32，不修改历史状态；这不代表整模型逐 token 等价。执行：`nvcc -std=c++17 -O2 -arch=sm_86 -shared -Xcompiler -fPIC clover_diffusion/cuda/preview_gdn_test_wrapper.cu -o /tmp/clover-preview-gdn.so`，再运行 `preview_gdn_test.py` 和 `benchmarks/clover/bench_preview_gdn.py`，需要 torch CUDA 与 `/tmp/clover-hf/modeling_clover_moe.py`。

GPU event 微基准在 RTX 3080 Ti、HK=16/HV=32、随机输入、每次 100 次循环、6 次交错采样，T=32 中位数 0.0652 → 0.0387 ms（约 1.69 倍），详见 `gdn-micro.jsonl`；微基准时服务处于空闲但模型常驻，未证明整机加速。

端到端初筛：相同三提示和 96-token 上限；原版与新内核写作 14.12 → 14.43 t/s、解释 10.79 → 10.97、代码 55.28 → 58.38。输出文本、部分结束长度与轮数不同，且仅各一轮、缓存与机器状态不完全相同，不能视为稳定收益或无质量退化。后续需固定输入状态做多次交错 A/B、多任务和长文本评价，再决定是否开启。
