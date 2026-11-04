# 示例

## V2 真实音频推理

用 V2 在 CPU 上处理三条公开 FSDD 人声。[图文结果](real_audio/RESULTS.md) · [实测 CSV](real_audio/result.csv) · [音频署名与许可](real_audio/ATTRIBUTION.md)

按[安装指南](setup.md)准备环境和 V2 权重后，从仓库根目录运行：

```bash
python examples/real_audio/fetch_inputs.py
mkdir -p output/public-demo
python predict.py examples/real_audio/input output/public-demo/result.csv \
  --config configs/v2.json --weights /path/to/v2_weights --device cpu
python examples/real_audio/verify_results.py output/public-demo/result.csv
```

将 `/path/to/v2_weights` 换成你的权重目录。复跑时请选择新的输出路径，确保 CSV 和同名 `.state/` 目录均不存在。
