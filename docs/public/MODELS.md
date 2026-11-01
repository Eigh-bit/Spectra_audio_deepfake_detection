# 版本与资源要求

本页描述源码依赖，不提供模型下载授权或不存在的下载链接。基础模型来自Spectra-0与XLS-R；本地研究产物不能用公开基础模型权重直接替代。

| 版本 | 聚合/模型资源 | 当前分发状态 |
|---|---|---|
| V2 | 原Spectra checkpoint与XLS-R；首/中/尾去重窗口logit差分均值 | 仅本项目推理模块/配置，权重与上游实现未附带 |
| Top2 | 本地完整F1-A checkpoint、XLS-R；固定Top2聚合 | 研究权重没有本仓库公开下载入口 |
| DomainMoE | F1-A、XLS-R、V2 head与学习得到的gate.json | head/gate/完整checkpoint均未分发 |

各版本`configs/resources.json`列明所需文件的相对路径、大小、revision与SHA256；路径相对`model/weights/`。资源配置用于识别，不证明这些资源能够从公开链接获取。源码目录内的资源缺口是有意排除，不能对缺少资源的树执行“完整模型已通过”的断言。

需要的上游版本：

- Spectra-0：`lab260/spectra_0`，revision `b3fb6d5c9d8ce069996f100cbe10117fd74165f3`。
- XLS-R：`facebook/wav2vec2-xls-r-300m`，revision `1a640f32ac3e39899438a2931f9924c02f080a54`。

请按上游要求独立取得模型实现、配置和权重，并逐项核对资源配置。快照的`weights/`没有随公开版收录，上游Spectra模型实现`weights/spectra0/model.py`也须独立取得。Top2/DomainMoE当前不具备从纯公开资源一键复现的条件；本项目没有承诺重新训练即可得到完全一致模型。

运行历史使用Python3.10、PyTorch/torchaudio 2.7.1+cu126。各源码入口的`requirements-runtime.lock.txt`记录固定版本，但不含系统驱动或全部平台依赖。CUDA、FFmpeg、系统库及实际硬件应自行核验；本轮没有重建运行环境或执行模型推理。

[快速示例](QUICKSTART.md)使用现有`--local-sync`进度模式，运行本地`model.py`；不需要比赛平台的`rts`。这与正式平台接口验收是两种用途，不以本地进度日志冒称平台验证。
