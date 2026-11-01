# 来源与第三方边界

项目原创的推理编排、聚合、输出适配及公开说明采用MIT；本页不能重新授权上游代码、模型、数据或二进制。上游组件未随本公开选择分发，取得与使用仍须遵守相应条件。

| 来源 | 用途与证据 | 当前说明 |
|---|---|---|
| [Spectra-0固定版本](https://huggingface.co/lab260/spectra_0/tree/b3fb6d5c9d8ce069996f100cbe10117fd74165f3) | 基础检测器与原模型实现 | 网页树元数据显示Apache-2.0；已保存的同revision模型卡正文写MIT。冲突未解决，不以项目MIT覆盖 |
| [XLS-R固定版本](https://huggingface.co/facebook/wav2vec2-xls-r-300m/tree/1a640f32ac3e39899438a2931f9924c02f080a54) | 语音表征 | 本地模型卡记录Apache-2.0；本轮网页模型卡读取失败，未声称重新核实完整条款 |
| FFmpeg / imageio-ffmpeg | 解码工具/二进制分发来源 | 已保存记录分别涉及GPL v3或更高与BSD包装器许可；本仓库不分发二进制或将二者视为同一许可 |
| PyTorch、torchaudio、Transformers等 | 运行依赖 | 通过固定依赖清单引用，依赖包按各自许可处理 |
| CFAD、InTheWild、LlamaPartialSpoof等公开语料 | 历史评估/研究来源 | 原音频、范围缓存与逐文件协议不随本公开选择分发；“公开可获取”不等于无使用条件 |

2026-11-01核验范围：Spectra固定版本文件树可读取；两份模型卡网页读取失败。模型卡内部冲突来自本地固定revision资料，不伪装成已获得作者澄清。公开选择主动不携带上游模型实现/模型卡、官方示例和比赛原始资料；这些本地文件仍保留，不删除其版权声明。

本项目MIT采用集体归属名“AIGC Audio project contributors”，不推断个人真实身份。未纳入公开选择的完整研究工作区、第三方资源和任何模型权重均不因根目录LICENSE而取得新的分发授权；适用范围见[NOTICE](../../NOTICE)。
