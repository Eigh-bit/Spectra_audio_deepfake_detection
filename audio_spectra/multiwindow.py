"""D1 fixed first/middle/last windows. Baseline inference.py is not changed."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
from .metrics import roc_metrics

SCORES = ['score_neg_bonafide', 'score_logit_difference']
AGGS = ['first', 'mean', 'max']


def load_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def offsets(length, width=64600):
    if length <= 0:
        raise ValueError('Empty waveform')
    if length <= width:
        return [0]
    return sorted({0, (length - width) // 2, length - width})


def window_tensors(audio, width=64600):
    """Input is already mono/resampled/preemphasized; never re-filter a slice."""
    length = audio.shape[0]
    positions = offsets(length, width)
    if length < width:
        audio = audio.repeat(int(width / length) + 1)[:width]
    return [(p, audio[p:p + width].unsqueeze(0).contiguous()) for p in positions]


def test():
    import torch
    import torchaudio
    assert offsets(1) == [0] and offsets(64600) == [0]
    assert offsets(64601) == [0, 1] and offsets(64602) == [0, 1, 2]
    assert offsets(100000) == [0, 17700, 35400]
    x = torch.tensor([1., 2.])
    assert torch.equal(window_tensors(x, 5)[0][1], torch.tensor([[1., 2., 1., 2., 1.]]))
    signal = torch.arange(12, dtype=torch.float32)
    full = torchaudio.functional.preemphasis(signal.unsqueeze(0), coeff=.97).squeeze(0)
    windows = window_tensors(full, 4)
    assert [p for p, _ in windows] == [0, 4, 8]
    assert torch.equal(windows[1][1], full[4:8].unsqueeze(0))
    assert windows[1][1][0, 0] != torchaudio.functional.preemphasis(signal[4:8].unsqueeze(0))[0, 0]
    try:
        offsets(0)
        raise AssertionError('Empty waveform accepted')
    except ValueError:
        pass
    return {'status': 'passed', 'checks': ['offset_floor', 'deduplicate', 'short_repeat', 'full_preemphasis_before_slice', 'reject_empty']}


def evaluate(run, baseline, weights, manifests=None, destination='development', check_c=True):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    from .network_guard import deny_internet_sockets
    guard = deny_internet_sockets()
    import soundfile as sf
    import torch
    import torchaudio
    from .inference import load_model
    out = run / 'raw' / destination
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, loading = load_model(weights)
    summary = {'loading': loading, 'network_guard': guard, 'datasets': {}, 'status': 'Running'}
    sets = manifests or {ds: baseline / 'manifests' / (ds + '_sample2000_local.jsonl') for ds in ['CFAD', 'InTheWild']}
    for ds, manifest in sets.items():
        rows = load_rows(manifest)
        old = {r['unique_id']: r for r in load_rows(baseline / 'raw/sample2000' / (ds + '_scores.jsonl'))} if check_c else {}
        begin_ds = time.perf_counter()
        count = Counter()
        with (out / (ds + '_windows.jsonl')).open('x') as f, (out / (ds + '_errors.jsonl')).open('x') as errors:
            for i, row in enumerate(rows):
                begin = time.perf_counter()
                result = {**row, 'inference_status': 'pending', 'windows': []}
                try:
                    signal, sr = sf.read(row['local_audio_path'], dtype='float32')
                    result['duration'] = len(signal) / sr
                    audio = torch.from_numpy(signal)
                    if audio.ndim > 1:
                        audio = audio.mean(dim=1)
                    if not torch.isfinite(audio).all():
                        raise ValueError('Non-finite waveform')
                    if sr != 16000:
                        audio = torchaudio.functional.resample(audio, sr, 16000)
                    audio = torchaudio.functional.preemphasis(audio.unsqueeze(0), coeff=.97).squeeze(0)
                    result['resampled_samples'] = len(audio)
                    sequence = window_tensors(audio)
                    result['decode_preprocess_seconds'] = time.perf_counter() - begin
                    for p, tensor in sequence:
                        first_hash = hashlib.sha256(tensor.numpy().tobytes()).hexdigest() if p == 0 else None
                        tensor = tensor.cuda()
                        torch.cuda.synchronize()
                        start_forward = time.perf_counter()
                        with torch.inference_mode():
                            logits = model(tensor)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - start_forward
                        if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
                            raise ValueError('Expected finite [1,2] logits')
                        spoof, bona = logits[0].cpu().tolist()
                        result['windows'].append({'window_offset': p, 'offset_seconds': p / 16000,
                            'spoof_logit': spoof, 'bonafide_logit': bona,
                            'score_neg_bonafide': -bona, 'score_logit_difference': spoof - bona,
                            'forward_seconds': seconds})
                        if p == 0:
                            result['first_window_e2e_seconds'] = time.perf_counter() - begin
                            result['first_preprocessed_sha256'] = first_hash
                            if check_c:
                                previous = old[row['unique_id']]
                                difference = max(abs(spoof - previous['spoof_logit']), abs(bona - previous['bonafide_logit']))
                                result['C_first_max_abs_logit_difference'] = difference
                                if first_hash != previous['preprocessed_sha256'] or difference != 0:
                                    raise RuntimeError('First-window mismatch against C')
                    result['aggregations'] = {s: {'first': result['windows'][0][s],
                        'mean': float(np.mean([w[s] for w in result['windows']])),
                        'max': max(w[s] for w in result['windows'])} for s in SCORES}
                    result['multiwindow_e2e_seconds'] = time.perf_counter() - begin
                    result['inference_status'] = 'success'
                except Exception as exc:
                    result.update(inference_status='failed', error=repr(exc))
                    errors.write(json.dumps({'unique_id': row['unique_id'], 'error': repr(exc), 'traceback': traceback.format_exc()}, ensure_ascii=False) + '\n')
                    errors.flush()
                    f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                    f.flush()
                    raise
                f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                count[len(result['windows'])] += 1
                if (i + 1) % 50 == 0:
                    f.flush()
                    print(json.dumps({'dataset': ds, 'done': i + 1, 'seconds': time.perf_counter() - begin_ds}), flush=True)
        summary['datasets'][ds] = {'n': len(rows), 'seconds': time.perf_counter() - begin_ds,
                                   'window_count_distribution': dict(count)}
    summary.update(status='Completed', total_seconds=time.perf_counter() - start,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
    dump(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def measure(rows, score, agg, threshold):
    y = np.array([r['true_label'] for r in rows])
    s = np.array([r['aggregations'][score][agg] for r in rows])
    counts = Counter(y.tolist())
    pred = s >= threshold
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    value = {'n': len(rows), 'bonafide': counts[0], 'spoof': counts[1],
             'fixed_threshold': threshold, 'fp': fp, 'fn': fn,
             'fpr': fp / counts[0] if counts[0] else None,
             'fnr': fn / counts[1] if counts[1] else None,
             'roc': roc_metrics(y, s) if min(counts[0], counts[1]) >= 20 else None,
             'eer_reason': None if min(counts[0], counts[1]) >= 20 else 'Need >=20 per class',
             'distribution': {}}
    for label in [0, 1]:
        x = s[y == label]
        if len(x):
            value['distribution'][str(label)] = {'mean': float(x.mean()), 'std': float(x.std()),
                'quantiles': {str(q): float(np.quantile(x, q)) for q in [0, .05, .25, .5, .75, .95, 1]}}
    return value


def analyze(run, destination='development'):
    config = json.loads((run / 'config.json').read_text())
    out = run / 'raw' / destination
    results, datasets = {}, {}
    for ds in ['CFAD', 'InTheWild']:
        path = out / (ds + '_windows.jsonl')
        if not path.exists():
            continue
        rows = load_rows(path)
        datasets[ds] = rows
        results[ds] = {}
        for score in SCORES:
            t = config['fixed_thresholds'][ds][score]
            results[ds][score] = {}
            for agg in AGGS:
                result = {'overall': measure(rows, score, agg, t), 'partitions': {}, 'first_e2e_seconds': sum(r['first_window_e2e_seconds'] for r in rows),
                          'multi_e2e_seconds': sum(r['multiwindow_e2e_seconds'] for r in rows),
                          'first_forward_seconds': sum(r['windows'][0]['forward_seconds'] for r in rows),
                          'all_forward_seconds': sum(w['forward_seconds'] for r in rows for w in r['windows'])}
                for partition in sorted({r['partition'] for r in rows}):
                    result['partitions'][partition] = measure([r for r in rows if r['partition'] == partition], score, agg, t)
                if ds == 'CFAD':
                    pf = [r for r in rows if r['source'] == 'partiallyfake']
                    bona = [r for r in rows if r['partition'] == 'test_unseen' and r['true_label'] == 0]
                    result['partiallyfake'] = measure(pf, score, agg, t)
                    result['PF_vs_unseen_bonafide'] = measure(pf + bona, score, agg, t)
                    old_misses = [r for r in pf if r['aggregations'][score]['first'] < t]
                    long_old = [r for r in old_misses if r['duration'] > 64600/16000]
                    result['original_missed_cohort'] = measure(old_misses, score, agg, t)
                    result['original_long_missed_cohort'] = measure(long_old, score, agg, t)
                    result['PF_new_misses'] = [r['sample_id'] for r in pf if r['aggregations'][score]['first'] >= t and r['aggregations'][score][agg] < t]
                results[ds][score][agg] = result
    dump(out / 'metrics.json', results)
    if 'CFAD' in datasets:
        with (out / 'partiallyfake_83.jsonl').open('x') as f:
            for r in datasets['CFAD']:
                if r['source'] == 'partiallyfake':
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
    if destination == 'development':
        candidates = []
        for score in SCORES:
            base_cf, base_it = results['CFAD'][score]['first'], results['InTheWild'][score]['first']
            base_fn = base_cf['partiallyfake']['fn']
            for agg in ['mean', 'max']:
                cf, it = results['CFAD'][score][agg], results['InTheWild'][score][agg]
                reduction = base_fn - cf['partiallyfake']['fn']
                tests = {'PF_FN_reduction': reduction >= max(2, math.ceil(.2 * base_fn)),
                    'CFAD_FPR': cf['overall']['fpr'] <= base_cf['overall']['fpr'] + .005 + 1e-12,
                    'ITW_FPR': it['overall']['fpr'] <= base_it['overall']['fpr'] + .005 + 1e-12,
                    'unseen_bonafide_FPR': cf['partitions']['test_unseen']['fpr'] <= base_cf['partitions']['test_unseen']['fpr'] + .01 + 1e-12,
                    'CFAD_EER': cf['overall']['roc']['eer'] <= base_cf['overall']['roc']['eer'] + .002 + 1e-12,
                    'ITW_EER': it['overall']['roc']['eer'] <= base_it['overall']['roc']['eer'] + .002 + 1e-12,
                    'ITW_FNR': it['overall']['fnr'] <= base_it['overall']['fnr'] + .005 + 1e-12}
                candidates.append({'score': score, 'aggregation': agg, 'gates': tests,
                    'passed': all(tests.values()), 'PF_FN_reduction': reduction,
                    'PF_relative_reduction': reduction / base_fn,
                    'CFAD_EER': cf['overall']['roc']['eer'], 'ITW_EER': it['overall']['roc']['eer']})
        eligible = [c for c in candidates if c['passed']]
        eligible.sort(key=lambda c: (-c['PF_relative_reduction'], c['CFAD_EER'], c['ITW_EER'],
                                     c['aggregation'] != 'mean', c['score'] != SCORES[0]))
        dump(out / 'selection.json', {'candidates': candidates, 'selected': eligible[0] if eligible else None,
                                      'confirmation_required': bool(eligible), 'stop_route': not bool(eligible)})
    print(json.dumps({ds:{s:{a:{'eer':v['overall']['roc']['eer'] if v['overall']['roc'] else None,'fp':v['overall']['fp'],'fn':v['overall']['fn'], 'PF_FN':v.get('partiallyfake',{}).get('fn')} for a,v in methods.items()} for s,methods in scores.items()} for ds,scores in results.items()},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['test', 'evaluate', 'analyze'])
    p.add_argument('--run', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--weights', type=Path)
    a = p.parse_args()
    if a.action == 'test':
        print(json.dumps(test(), indent=2))
    elif a.action == 'evaluate':
        evaluate(a.run, a.baseline, a.weights)
    else:
        analyze(a.run)
