# 快速示例

示例分两步：第一步无模型、无第三方依赖即可运行；第二步需要[模型资源](MODELS.md)与兼容环境，当前没有在干净环境中重新验证真实推理。第一步通过不代表第二步已通过。

## 1. 生成无语音测试信号

从仓库根目录运行以下命令。仅生成2秒、16kHz、单声道PCM16测试音，不使用比赛样本、录音或第三方音频；临时目录不进入Git。

```bash
export DEMO_ROOT="$(mktemp -d /tmp/aigc-audio-demo.XXXXXX)"
python3 - <<'PYDEMO'
import math
import os
from pathlib import Path
import struct
import wave

root = Path(os.environ['DEMO_ROOT'])
folder = root / 'input'
folder.mkdir()
path = folder / 'synthetic_tone.wav'
rate = 16000
samples = [int(3000 * math.sin(2 * math.pi * 440 * i / rate))
           for i in range(2 * rate)]
with wave.open(str(path), 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(rate)
    wav.writeframes(struct.pack('<%dh' % len(samples), *samples))
with wave.open(str(path), 'rb') as wav:
    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
            wav.getnframes()) == (1, 2, 16000, 32000)
print(path)
PYDEMO
```

这是合成的非语音工程测试信号，**不赋予Real/Fake标签，不用于EER或模型有效性评价**。正常输出一个临时WAV路径；音频本身不打包上传。

## 2. 资源齐备后的本地推理

先按`configs/resources.json`补齐选定版本的上游源码、权重和配置，准备兼容Python3.10环境及FFmpeg。缺少任何一项都不能运行真实模型。V2示例命令如下；`/path/to/python3.10`替换为已配置解释器：

```bash
MODEL_ROOT="$PWD/experiments/submissions/archive_sources_20261101_01/V2/model"
/path/to/python3.10 "$MODEL_ROOT/model.py"   "$DEMO_ROOT/input" "$DEMO_ROOT/result.csv" --local-sync
```

该命令调用现有本地进度模式，不修改推理实现，也不创建假的`rts`模块。`--local-sync`仅替换进度记录，不替换模型或分数；缺资源/解码失败应报错。

输出列为`filename,label,confidence`；检查仅有输入对应文件、没有缺失/重复，confidence为有限[0,1]数值，越大越真。合成测试信号的预测没有准确率含义；要评估检测性能，需使用有许可和真实标签的独立语料及预先固定的数据划分。不要将该示例当作完整基准或部署时限验证。
