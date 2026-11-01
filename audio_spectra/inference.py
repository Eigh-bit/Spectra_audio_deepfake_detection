"""Local-only Spectra-0 adapter; upstream architecture and weights are unchanged."""
import importlib.util
import os
from pathlib import Path
import time

# Set before importing any Hub consumer. No network fallback is allowed.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[1] / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_CACHE', str(Path(os.environ['HF_HOME']) / 'hub'))

import soundfile as sf
import torch
import torchaudio
from safetensors.torch import load_file
from transformers import Wav2Vec2Model

WINDOW = 64600


def official_module(weights):
    """Use the byte-identical official source with only its model locator redirected."""
    weights = Path(weights).resolve()
    source = weights / 'spectra0/model.py'
    spec = importlib.util.spec_from_file_location('_spectra0_official', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    diagnostics = []

    class LocalWav2Vec2:
        @staticmethod
        def from_pretrained(name, **kwargs):
            if name != 'facebook/wav2vec2-xls-r-300m':
                raise ValueError(f'Unexpected upstream encoder: {name}')
            model, info = Wav2Vec2Model.from_pretrained(
                str(weights / 'xlsr'), local_files_only=True,
                output_loading_info=True, **kwargs)
            diagnostics.append(info)
            # Pretraining-only heads can be absent from the bare encoder architecture.
            allowed_prefixes = ('quantizer.', 'project_q.', 'project_hid.')
            bad_extra = [k for k in info['unexpected_keys'] if not k.startswith(allowed_prefixes)]
            if info['missing_keys'] or info.get('mismatched_keys') or info.get('error_msgs') or bad_extra:
                raise RuntimeError(f'Incomplete XLS-R encoder loading: {info}')
            return model

    module.Wav2Vec2Model = LocalWav2Vec2
    return module, diagnostics


def load_model(weights, device='cuda'):
    start = time.perf_counter()
    module, base_info = official_module(weights)
    model = module.Spectra0Model()
    state = load_file(str(Path(weights) / 'spectra0/model.safetensors'), device='cpu')
    state_count = len(state)
    result = model.load_state_dict(state, strict=True)
    del state
    model = model.eval().to(device=device, dtype=torch.float32)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    return model, {'load_seconds': time.perf_counter() - start,
                   'checkpoint_tensor_count': state_count,
                   'spectra_missing_keys': result.missing_keys,
                   'spectra_unexpected_keys': result.unexpected_keys,
                   'xlsr_loading_info': base_info}


def preprocess(path):
    audio, sr = sf.read(str(path), dtype='float32')
    audio = torch.from_numpy(audio)
    if audio.ndim > 1:
        audio = audio.mean(dim=1)
    if not audio.numel() or not torch.isfinite(audio).all():
        raise ValueError(f'Empty or non-finite decoded audio: {path}')
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = torchaudio.functional.preemphasis(audio.unsqueeze(0), coeff=0.97).squeeze(0)
    length = audio.shape[0]
    if length >= WINDOW:
        audio = audio[:WINDOW]
    else:
        audio = audio.repeat(int(WINDOW / length) + 1)[:WINDOW]
    return audio.unsqueeze(0).contiguous()


def scores(logits):
    if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite [batch, 2] logits')
    return [{'logit_spoof': float(row[0]), 'logit_bonafide': float(row[1]),
             'fake_score_negative_bonafide': -float(row[1]),
             'alternative_spoof_minus_bonafide': float(row[0] - row[1])}
            for row in logits.detach().cpu()]
