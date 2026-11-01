# Spectra Audio Deepfake Detection

基于Spectra-0与XLS-R的音频伪造检测实践，提供文件级推理、多窗口聚合和编码鲁棒性研究的推理配置。基础模型来自上游；本项目不宣称从零训练基础模型或取得官方最优成绩。

## 目录

```text
predict.py          # 本地推理入口
audio_spectra/      # 共用的解码、模型适配、聚合与输出模块
configs/            # V2 / Top2 / DomainMoE配置及资源散列
tests/              # 无需GPU的接口与失败处理回归测试
requirements.txt    # 历史运行环境的固定依赖
```

三种方案共用一套实现，通过配置选择，不需要复制三份工程。

## 环境与资源

历史运行环境为Python 3.10、Linux、PyTorch/torchaudio 2.7.1+cu126；其他环境需自行适配。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

如需MP3解码或DomainMoE的编码视图，还需可用的FFmpeg。安装在运行前完成；推理不联网下载模型。

**仓库不包含权重和上游模型实现，克隆后不能直接进行真实推理。** 对应资源须按`configs/resources/*.json`中的路径和SHA准备，文件位于传给`--weights`的目录下：

| 配置 | 方法 | 所需模型资源 |
|---|---|---|
| `configs/v2.json` | 首/中/尾窗口去重，logit差分均值 | 原Spectra checkpoint与XLS-R |
| `configs/top2.json` | 固定Top2窗口聚合 | 本地F1-A checkpoint与XLS-R |
| `configs/domain_moe.json` | 声学域门控混合与短音频编码视图 | F1-A、XLS-R、V2 head及学习得到的gate |

上游来源：[Spectra-0](https://huggingface.co/lab260/spectra_0)、[XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-300m)。固定revision及文件散列记录在资源清单中。Spectra实现应放在`<weights>/spectra0/model.py`，其余资源按清单放置。

F1-A、V2 head和gate没有本仓库的公开下载入口，不能用上游基础权重替代。当前源码可用于阅读和工程复用，尚不具备从公开资源一键复现所有研究模型的条件。

## 推理

输入为音频目录，输出CSV必须是新文件，且父目录已存在：

```bash
mkdir -p output
python predict.py /path/to/audio output/result.csv \
  --config configs/v2.json --weights /path/to/v2_weights
```

切换Top2或DomainMoE时，同时选择对应配置和资源目录。`--device cpu`可覆盖设备配置，但未作为性能承诺。入口使用普通终端进度，不依赖比赛平台的`rts`。

CSV列为`filename,label,confidence`；confidence为有限[0,1]数值，越大越真。大于0.5判real，其余判fake；这只是当前输出合同，不代表经过概率校准或对所有语料通用的最优阈值。

缺少资源、散列不符或处理失败应报错，不用占位分数补齐。中间进度记录在`result.csv.state/`；全部输入成功后才发布最终CSV，已有结果不覆盖。

## 无需模型的检查

```bash
python predict.py --help
python -m unittest discover -s tests -v
```

测试覆盖配置与权重身份、资源篡改、文件覆盖/分数方向，以及失败时不发布部分CSV。测试使用显式桩评分，不冒充模型推理或检测准确率。

可用以下命令生成2秒单声道测试音，作为资源齐备后的解码输入；它没有Real/Fake真值，不能用于EER：

```bash
python - <<'PY'
import math, struct, tempfile, wave
from pathlib import Path
folder = Path(tempfile.mkdtemp(prefix='spectra-demo-'))
path = folder / 'tone.wav'
with wave.open(str(path), 'wb') as wav:
    wav.setparams((1, 2, 16000, 32000, 'NONE', 'not compressed'))
    wav.writeframes(struct.pack('<32000h', *(int(3000*math.sin(2*math.pi*440*i/16000)) for i in range(32000))))
print(folder)
PY
```

## 研究范围与许可

本项目研究了首窗漏检、多窗口聚合、跨编码鲁棒性与模型错误互补。开发集改善不保证跨域保持，部分候选会增加真实语音误报或降低局部伪造检出。没有可公开核验的官方EER或比赛成绩；本次结构整理只验证接口、资源选择及评分逻辑保持，未重新执行真实GPU模型推理。

原创部分采用[MIT](LICENSE)，第三方来源与适用边界见[NOTICE](NOTICE)。上游Spectra模型卡的Apache-2.0/MIT标注存在未解决的不一致；本仓库不分发上游模型实现、权重、数据、FFmpeg二进制或比赛原始资料，也不以项目MIT重新授权这些资源。
