"""Frozen V1/V2 directory inference; no training, downloads or default predictions."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
from .output_adapter import OutputAdapter

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.pending')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def top2_score(scores):
    """Frozen Top2: mean the largest min(2,n) existing window fake scores."""
    import numpy as np
    values = [float(value) for value in scores]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('Top2 requires nonempty finite window scores')
    return float(np.mean(sorted(values, reverse=True)[:2]))


def load_config(path=None):
    config = json.loads(Path(path or ROOT / 'configs/spectra.json').read_text(encoding='utf-8'))
    if config['mode'] not in ('V1-Base', 'V2-Candidate', 'V3-Top2'):
        raise ValueError('Unknown frozen model mode')
    if config['threads'] < 1:
        raise ValueError('threads must be positive')
    return config


def verify_weights(weights):
    manifest = json.loads((ROOT / 'configs/resources.json').read_text())
    for entry in manifest:
        path = Path(weights) / entry['path']
        if path.is_symlink() or not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError('Model resource missing or SHA256 mismatch: ' + str(path))
    return manifest


def decoder_binary():
    local = ROOT / 'tools/ffmpeg'
    return str(local) if local.is_file() else shutil.which('ffmpeg')


def decode_audio(path, timeout=120):
    """Probe bytes with libsndfile; optional offline ffmpeg fallback, retaining sr/channels."""
    import numpy as np
    import soundfile as sf
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Not a regular audio input: ' + str(path))
    decoder = 'soundfile'
    try:
        data, sr = sf.read(str(path), dtype='float32')
    except (sf.LibsndfileError, RuntimeError) as first:
        binary = decoder_binary()
        if not binary:
            raise ValueError(f'Audio decode failed; offline ffmpeg unavailable: {path}: {first}') from first
        command = [binary, '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-protocol_whitelist', 'file,pipe', '-i', str(path.resolve()),
                   '-map', '0:a:0', '-vn', '-f', 'wav', '-acodec', 'pcm_f32le', 'pipe:1']
        process = subprocess.run(command, capture_output=True, timeout=timeout)
        if process.returncode:
            raise ValueError(f'ffmpeg decode failed ({process.returncode}): {path}: '
                             + process.stderr.decode('utf-8', errors='replace')[-2000:])
        data, sr = sf.read(io.BytesIO(process.stdout), dtype='float32')
        decoder = 'ffmpeg'
    if sr <= 0 or data.ndim not in (1, 2) or not len(data) or not np.isfinite(data).all():
        raise ValueError('Empty, invalid or non-finite decoded waveform: ' + str(path))
    return data, sr, {'decoder': decoder, 'original_sample_rate': sr,
                      'original_channels': 1 if data.ndim == 1 else data.shape[1],
                      'duration_seconds': len(data) / sr}


def prepare_audio(path, timeout=120):
    import torch
    import torchaudio
    data, sr, metadata = decode_audio(path, timeout)
    audio = torch.from_numpy(data)
    if audio.ndim > 1:
        audio = audio.mean(dim=1)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    # D1 semantics: full-waveform filtering before first/middle/tail slicing.
    audio = torchaudio.functional.preemphasis(audio.unsqueeze(0), coeff=.97).squeeze(0)
    if not audio.numel() or not torch.isfinite(audio).all():
        raise ValueError('Invalid waveform after preprocessing')
    metadata['resampled_samples'] = len(audio)
    return audio, metadata


class SpectraEngine:
    def __init__(self, config, weights=None):
        import numpy as np
        import torch
        from .inference import load_model
        self.torch = torch
        self.config = config
        start = time.perf_counter()
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.set_num_threads(config['threads'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = config['device']
        self.cuda = self.device.startswith('cuda')
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        weights = Path(weights or ROOT / 'weights')
        verify_weights(weights)
        hash_seconds = time.perf_counter() - start
        self.model, self.loading = load_model(weights, device=self.device)
        self.loading.update(resource_verification_seconds=hash_seconds,
                            initialization_seconds=time.perf_counter() - start)
        self.domain_gate = None
        fusion = config.get('score_fusion', 'off')
        if fusion not in ('off', 'domain_moe'):
            raise ValueError('Unknown frozen score fusion')
        if fusion == 'domain_moe':
            import copy
            from safetensors.torch import load_file
            from .domain_moe import DomainGate
            gate_path = weights / 'domain_moe/gate.json'
            head_path = weights / 'domain_moe/v2_head.safetensors'
            self.domain_gate = DomainGate(json.loads(gate_path.read_text()))
            self.v2_bridge = copy.deepcopy(self.model.bridge).eval().requires_grad_(False)
            self.v2_ecapa = copy.deepcopy(self.model.ecapa_tdnn).eval().requires_grad_(False)
            head = load_file(str(head_path), device='cpu')
            expected = {'bridge.' + k for k in self.v2_bridge.state_dict()} | {'ecapa_tdnn.' + k for k in self.v2_ecapa.state_dict()}
            if set(head) != expected:
                raise ValueError('Incomplete auxiliary V2 head')
            self.v2_bridge.load_state_dict({k[7:]: v for k, v in head.items() if k.startswith('bridge.')}, strict=True)
            self.v2_ecapa.load_state_dict({k[11:]: v for k, v in head.items() if k.startswith('ecapa_tdnn.')}, strict=True)
            self.loading.update(auxiliary_head_state_tensors=len(head), auxiliary_head_sha256=sha256(head_path),
                                domain_gate_sha256=sha256(gate_path), encoder_loads=1,
                                initialization_seconds=time.perf_counter() - start)
            del head
        self.load_count = 1

    def score_file(self, path, mode=None):
        begin = time.perf_counter()
        policy = self.config.get('duration_policy', 'off')
        if policy == 'off':
            result = self._score_file_native(path, mode)
        elif policy == 'short_tta64':
            from .duration_policy import score_short_tta
            result = score_short_tta(self, path, mode, decoder_binary())
        else:
            raise ValueError('Unknown duration policy')
        if self.domain_gate is not None:
            if (mode or self.config['mode']) != 'V3-Top2':
                raise ValueError('DomainMoE requires the frozen Top2 base')
            embedding = result.pop('_domain_embedding')
            short_score = result['fake_score']
            mixture = self.domain_gate.score(embedding, short_score, result['v2_mean_fake_score'])
            result.update(short_tta_fake_score=short_score, base_file_aggregation=result['file_aggregation'],
                          fake_score=mixture['score'], file_aggregation='domain_moe',
                          domain_probabilities=mixture['domain_probabilities'],
                          calibrated_expert_logits=mixture['expert_logits'], p_v2=mixture['p_v2'],
                          file_seconds=time.perf_counter() - begin)
        return result

    def _score_file_native(self, path, mode=None):
        import numpy as np
        from .multiwindow import window_tensors
        mode = mode or self.config['mode']
        if mode not in ('V1-Base', 'V2-Candidate', 'V3-Top2'):
            raise ValueError('Unknown mode')
        start = time.perf_counter()
        audio, metadata = prepare_audio(path, self.config['decode_timeout_seconds'])
        sequence = window_tensors(audio)
        if mode == 'V1-Base':
            sequence = sequence[:1]
        decode_seconds = time.perf_counter() - start
        windows = []
        first_embedding = None
        for offset, tensor in sequence:
            first_hash = hashlib.sha256(tensor.numpy().tobytes()).hexdigest() if offset == 0 else None
            tensor = tensor.to(self.device)
            if self.cuda:
                self.torch.cuda.synchronize(self.device)
            begin = time.perf_counter()
            with self.torch.inference_mode():
                if self.domain_gate is None:
                    logits = self.model(tensor)
                else:
                    feature = self.model.ssl_encoder(tensor)
                    logits = self.model.ecapa_tdnn(self.model.bridge(feature))
                    auxiliary = self.v2_ecapa(self.v2_bridge(feature))[0].cpu().tolist()
                    if offset == 0:
                        first_embedding = feature[0].cpu().numpy().astype(np.float64).mean(0).tolist()
            if self.cuda:
                self.torch.cuda.synchronize(self.device)
            seconds = time.perf_counter() - begin
            if tuple(logits.shape) != (1, 2) or not self.torch.isfinite(logits).all():
                raise ValueError('Expected finite [1,2] model logits')
            spoof, bona = logits[0].cpu().tolist()
            windows.append({'offset': offset, 'spoof_logit': spoof, 'bonafide_logit': bona,
                            'fake_score': spoof - bona, 'forward_seconds': seconds,
                            'preprocessed_sha256': first_hash})
            if self.domain_gate is not None:
                windows[-1].update(v2_spoof_logit=auxiliary[0], v2_bonafide_logit=auxiliary[1], v2_fake_score=auxiliary[0] - auxiliary[1])
        # Match D1's float64 arithmetic mean, including its order of operations.
        mean_score = float(np.mean([window['fake_score'] for window in windows]))
        score = top2_score([window['fake_score'] for window in windows]) if mode == 'V3-Top2' else mean_score
        if not math.isfinite(score):
            raise ValueError('Non-finite aggregate score')
        result = {'filename': Path(path).name, 'mode': mode, 'fake_score': score,
                'mean_fake_score': mean_score, 'file_aggregation': 'top2' if mode == 'V3-Top2' else 'mean',
                'windows': windows, **metadata, 'decode_preprocess_seconds': decode_seconds,
                'file_seconds': time.perf_counter() - start}
        if self.domain_gate is not None:
            result.update(_domain_embedding=first_embedding, v2_mean_fake_score=float(np.mean([w['v2_fake_score'] for w in windows])))
        return result

    def resources(self):
        return {'model_load_count': self.load_count,
                'peak_cpu_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                'peak_gpu_allocated_bytes': self.torch.cuda.max_memory_allocated(self.device) if self.cuda else None,
                'peak_gpu_reserved_bytes': self.torch.cuda.max_memory_reserved(self.device) if self.cuda else None}


def input_files(input_dir, recursive=False):
    root = Path(input_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Input must be a directory')
    entries = sorted(root.rglob('*') if recursive else root.iterdir(), key=lambda p: p.relative_to(root).as_posix())
    files = []
    for path in entries:
        if path.is_symlink():
            raise ValueError('Symlink input forbidden: ' + str(path))
        if path.is_dir() and recursive:
            continue
        if not path.is_file():
            raise ValueError('Expected regular audio files; no filtering: ' + str(path))
        files.append((path.relative_to(root).as_posix(), path))
    if not files:
        raise ValueError('Empty audio input; no placeholder CSV')
    return root, files


def validate_csv(path, expected, adapter, predictions=None):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ['filename', 'label', 'confidence']:
            raise ValueError('CSV columns/order mismatch')
        rows = list(reader)
    names = [r[0] for r in rows if len(r) == 3]
    if len(rows) != len(expected) or len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError('CSV row count/uniqueness/coverage mismatch')
    for row in rows:
        if len(row) != 3 or row[1] not in {str(v) for v in adapter.labels.values()}:
            raise ValueError('Invalid CSV label/row')
        value = float(row[2])
        if not math.isfinite(value) or (adapter.transform == 'sigmoid' and not 0 <= value <= 1):
            raise ValueError('Invalid CSV confidence')
        if row[1] != str(adapter.label_from_confidence(value)):
            raise ValueError('CSV label/confidence decision mismatch')
        if predictions is not None:
            label, confidence = predictions[row[0]]
            if row[1] != str(label) or value != confidence:
                raise ValueError('CSV prediction differs from native-score decision adapter')
    return {'rows': len(rows), 'unique_filenames': len(names), 'sha256': sha256(path)}


def run_directory(input_dir, output_csv, config=None, sync_factory=None, engine=None,
                  require_official=False):
    """One row per decoded file; all-or-nothing final CSV plus independent audit state."""
    config = config or load_config()
    adapter = OutputAdapter(config['output'], require_official=require_official)
    root, entries = input_files(input_dir, config['recursive'])
    if Path(output_csv).is_symlink():
        raise ValueError('Output symlink forbidden')
    output = Path(output_csv).resolve()
    if output.is_relative_to(root):
        raise ValueError('Output must be outside the input tree')
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError('Output must be new and parent must exist')
    state_dir = output.with_name(output.name + '.state')
    state_dir.mkdir(exist_ok=False)
    status = {'status': 'Running', 'expected': len(entries), 'completed': 0, 'config': config,
              'output_scope': 'official' if require_official else 'local/unverified official mapping',
              'input_directory': str(root), 'output_csv': str(output)}
    start = time.perf_counter()
    save_json(state_dir / 'status.json', status)
    current = None
    try:
        sync = sync_factory(0, len(entries)) if sync_factory else None
        engine = engine or SpectraEngine(config)
        rows = []
        begin_files = time.perf_counter()
        with (state_dir / 'scores.jsonl').open('x', encoding='utf-8') as audit:
            for index, (name, path) in enumerate(entries):
                current = name
                result = engine.score_file(path, config['mode'])
                result['filename'] = name
                label, confidence = adapter.prediction(result['fake_score'])
                rows.append((name, label, confidence))
                result.update(label=label, confidence=confidence)
                audit.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                audit.flush()
                if sync:
                    sync.update(index + 1)
                status['completed'] = index + 1
                save_json(state_dir / 'status.json', status)
                print(json.dumps({'file': name, 'done': index + 1, 'total': len(entries)}, ensure_ascii=False),
                      file=sys.stderr, flush=True)
        directory_seconds = time.perf_counter() - begin_files
        temporary = state_dir / 'result.csv.pending'
        with temporary.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['filename', 'label', 'confidence'])
            for name, label, confidence in rows:
                writer.writerow([name, label, format(confidence, '.17g')])
            stream.flush()
            os.fsync(stream.fileno())
        checked = validate_csv(temporary, [name for name, _ in entries], adapter,
                               {name: (label, confidence) for name, label, confidence in rows})
        # Link fails on any concurrent existing output; never clobber prior evidence.
        os.link(temporary, output)
        temporary.unlink()
        status.update(status='Completed', csv=checked, loading=engine.loading,
                      directory_seconds=directory_seconds, files_per_second=len(rows) / directory_seconds,
                      total_seconds=time.perf_counter() - start, resources=engine.resources())
        save_json(state_dir / 'status.json', status)
        return rows
    except BaseException as exc:
        status.update(status='Failed', failed_filename=current, error=repr(exc),
                      elapsed_seconds=time.perf_counter() - start)
        save_json(state_dir / 'status.json', status)
        (state_dir / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise
