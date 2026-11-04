# 环境与模型资源

请在 Linux 上从仓库根目录执行安装命令。历史运行环境为 Python 3.10、Linux、PyTorch/torchaudio 2.7.1+cu126；其他环境需自行适配。

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

## 准备 V2 资源

[V2 清单](../configs/resources/v2.json)列出每个文件的来源仓库、固定版本、目标路径和 SHA-256。打开对应 Hugging Face 仓库的固定版本，按目标路径的文件名下载。将文件放到权重目录下的 `spectra0/` 和 `xlsr/` 子目录。两个权重文件合计约 2.54 GB。将权重目录传给 `--weights`，CLI 会在加载模型前校验文件。

## 本地检查

可在独立 Python 环境中运行。测试只需 NumPy，可用以下命令安装。

~~~bash
python -m pip install numpy==1.26.4
~~~

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
