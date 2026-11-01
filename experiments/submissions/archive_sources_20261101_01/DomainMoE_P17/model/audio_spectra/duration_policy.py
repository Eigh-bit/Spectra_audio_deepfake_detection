"""Fixed single-window codec view averaging; no learned thresholds or class branches."""
import math
from pathlib import Path
import subprocess
import tempfile
import time


def score_short_tta(engine, path, mode, ffmpeg):
    start = time.perf_counter()
    native = engine._score_file_native(path, mode)
    native['duration_policy'] = 'short_tta64'
    native['duration_policy_applied'] = False
    if native['resampled_samples'] > 64600:
        return native
    with tempfile.TemporaryDirectory(prefix='spectra0_short_tta_') as folder:
        encoded = Path(folder) / 'view64.mp3'
        command = [str(ffmpeg), '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-protocol_whitelist', 'file,pipe', '-i', str(Path(path).resolve()),
                   '-codec:a', 'libmp3lame', '-b:a', '64k', str(encoded)]
        proc = subprocess.run(command, capture_output=True,
                              timeout=engine.config['decode_timeout_seconds'])
        if proc.returncode:
            raise RuntimeError('Short TTA codec failed: ' + proc.stderr.decode('utf-8', errors='replace')[-2000:])
        transformed = engine._score_file_native(encoded, mode)
    first, second = native['fake_score'], transformed['fake_score']
    score = (first + second) / 2
    if not all(math.isfinite(x) for x in [first, second, score]):
        raise ValueError('Non-finite short TTA score')
    native.update(native_fake_score=first, codec_fake_score=second, fake_score=score,
                  native_file_aggregation=native.get('file_aggregation'),
                  file_aggregation='short_tta64', duration_policy_applied=True,
                  tta_windows=transformed['windows'],
                  codec_view_metadata={k: transformed.get(k) for k in ['duration_seconds', 'resampled_samples', 'original_sample_rate', 'original_channels']},
                  file_seconds=time.perf_counter() - start)
    return native
