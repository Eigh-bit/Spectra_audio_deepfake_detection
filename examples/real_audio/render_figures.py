"""Render measured waveforms and model scores; requires matplotlib and numpy."""
import csv
import json
from pathlib import Path
import wave

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'inputs.json').read_text())
    with (ROOT / 'result.csv').open() as stream:
        scores = {row['filename']: row for row in csv.DictReader(stream)}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.hashsalt': 'public_audio_demo_20261104_01'})
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.5), gridspec_kw={'width_ratios': [2.2, 1]})
    for index, entry in enumerate(manifest['files']):
        with wave.open(str(ROOT / 'input' / entry['filename'])) as wav:
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2') / 32768.
            rate = wav.getframerate()
        wave_ax, score_ax = axes[index]
        wave_ax.plot(np.arange(len(samples)) / rate, samples, color='#2563eb', linewidth=.7)
        wave_ax.set(title=entry['filename'] + f" | {len(samples)/rate:.6f} s",
                    xlabel='Original recording time (s)', ylabel='Amplitude', ylim=(-1, 1))
        wave_ax.grid(alpha=.18)
        score = float(scores[entry['filename']]['confidence'])
        label = scores[entry['filename']]['label']
        score_ax.barh([0], [score], color='#059669' if label == 'real' else '#dc2626', height=.5)
        score_ax.plot(score, 0, 'o', color='#065f46' if label == 'real' else '#dc2626', markersize=5)
        score_ax.axvline(.5, color='#475569', linestyle='--', linewidth=1)
        score_ax.set(xlim=(0, 1), yticks=[], xlabel='Confidence (higher = real)',
                     title=f'{label} | confidence={score:.6f}')
        score_ax.set_xticks([0, .5, 1])
        score_ax.text(.52, .08, 'Threshold: 0.5', transform=score_ax.transAxes,
                      fontsize=9, color='#475569', bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none'})
    fig.suptitle('V2 on three public human voice recordings | CPU float32 | 2026-11-04', fontsize=15)
    fig.text(.5, .01, 'FSDD recordings and waveform derivatives: CC BY-SA 4.0; Jackson and contributors. '
             'Small inference demo; no EER or generalization claim.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    target = ROOT / 'assets/public_audio_demo_20261104_01'
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / 'overview.svg', metadata={'Date': '2026-11-04'})
    fig.savefig(target / 'overview.png', dpi=160)
    plt.close(fig)
    print(target / 'overview.svg')


if __name__ == '__main__':
    main()
