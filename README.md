# AIGC Audio · 基于Spectra-0的音频伪造检测实践

面向音频深度伪造检测的推理工程与研究复现资料，关注多窗口聚合、跨编码鲁棒性和离线推理。模型能力建立在Spectra-0与XLS-R之上，本项目的工作集中在推理适配、聚合策略、受控实验及部署复核；不宣称从零训练基础模型或取得官方最优成绩。

## 可以阅读和使用什么

- 三版推理模块、配置与固定依赖清单：V2、Top2、DomainMoE。
- [方法与研究结论](reports/public/RESEARCH_OVERVIEW.md)：公开语料上的历史观察、失败路线和局限。
- [快速示例](docs/public/QUICKSTART.md)：只用Python标准库生成无语音测试信号，并说明具备模型资源后的本地推理方法。
- [版本和资源要求](docs/public/MODELS.md)、[公开范围](docs/public/RELEASE_SCOPE.md)、[第三方来源](docs/public/THIRD_PARTY.md)。

## 源码版本

| 版本 | 入口 | 说明 |
|---|---|---|
| V2 | [推理源码](experiments/submissions/archive_sources_20261101_01/V2/model/) | 首/中/尾窗口去重后按logit差分均值聚合；建议从此理解流程 |
| Top2 | [推理源码](experiments/submissions/archive_sources_20261101_01/Top2/model/) | 固定Top2文件级聚合；没有重新训练基础模型 |
| DomainMoE | [推理源码](experiments/submissions/archive_sources_20261101_01/DomainMoE_P17/model/) | 声学域门控混合；学习得到的gate、研究权重不随源码分发 |

源码快照保持冻结版本字节。**克隆后不能直接完成真实模型推理**：本仓库不提供权重、上游模型实现、FFmpeg二进制或数据集；部分研究权重没有公开下载入口。请先查[资源要求](docs/public/MODELS.md)，不要用随机初始化或占位分数代替缺失模型。

输出CSV为`filename,label,confidence`；本项目适配器中confidence越大越真，不等同于所有评测工具的分数约定。合成测试信号仅用于文件/解码检查，没有Real/Fake真值，不能据此计算检测准确率或EER。

## 研究边界

公开语料抽样、跨域和编码实验不能替代独立完整基准。本项目没有可公开核验的官方EER或比赛成绩；不同候选存在误报和局部伪造检测退化，详细限制见研究说明。此公开选择不包含比赛原始样本、目标逐条预测、伪标签、平台操作档案或完整内部实验历史。

## 许可与来源

项目原创部分采用[MIT License](LICENSE)，适用边界见[NOTICE](NOTICE)。第三方模型、数据及工具按其各自条件处理；MIT不重新授权这些资源。上游许可标注仍有待确认事项，见[第三方说明](docs/public/THIRD_PARTY.md)。

[逐文件源码清单](docs/GITHUB_UPLOAD_MANIFEST.tsv)记录公开候选集的路径、字节数和SHA。公开范围以该清单及白名单为准；完整私有工作区和大资源仍另行保留，不随此源码选择公开。
