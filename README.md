# Spectra Audio Deepfake Detection

基于 Spectra-0 和 XLS-R 的音频伪造检测研究代码。命令行工具读取本地音频目录，为每个文件输出一个真假标签和分数。

## 快速开始

先查看命令参数。此步骤只需要 Python。

```bash
python predict.py --help
```

真实推理需要 Linux、模型依赖和已下载的权重。按[环境与模型资源](docs/setup.md)准备 V2 所需文件，然后从仓库根目录运行：

```bash
mkdir -p output
python predict.py /path/to/audio output/result.csv \
  --config configs/v2.json --weights /path/to/v2_weights
```

输出路径须指向尚未创建的 CSV。全部音频处理成功后才发布结果，中间进度保存在 `output/result.csv.state/`。

CSV 列为 `filename,label,confidence`。分数介于 0 和 1，越高越偏向真实。高于 0.5 输出 `real`，否则输出 `fake`。分数未经概率校准。

## 选择配置

| 配置 | 用途 | 资源清单 |
| --- | --- | --- |
| [V2](configs/v2.json) | 对首、中、尾窗口聚合评分 | [V2](configs/resources/v2.json) |
| [Top2](configs/top2.json) | 聚合得分最高的两个窗口 | [Top2](configs/resources/top2.json) |
| [DomainMoE](configs/domain_moe.json) | 按声学域混合评分，并对短音频使用编码视图 | [DomainMoE](configs/resources/domain_moe.json) |

仓库未附带模型实现或权重。V2 使用上游资源。Top2 和 DomainMoE 还需要未公开发布的研究权重。

## 源码与验证

- [predict.py](predict.py) 提供推理入口，[audio_spectra/](audio_spectra/) 实现解码、评分和输出。
- [本地检查](docs/setup.md#本地检查)覆盖资源校验、评分输出和失败处理，无需 GPU 或模型权重。
- [研究范围与许可](docs/research.md)记录实验局限和第三方资源的许可边界。
