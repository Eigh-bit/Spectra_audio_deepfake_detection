"""File-level fake-oriented ROC/AUC and linearly interpolated EER (no scipy)."""
import numpy as np


def roc_metrics(labels, scores):
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not len(y):
        raise ValueError('Expected nonempty aligned 1D labels/scores')
    if not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Non-finite score or label outside {0,1}')
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    if not positives or not negatives:
        raise ValueError('Both classes required')
    order = np.argsort(-s, kind='stable')
    ys, ss = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.cumsum(ys == 1)[ends]
    fp = ends + 1 - tp
    tpr, fpr = np.r_[0., tp / positives], np.r_[0., fp / negatives]
    thresholds = np.r_[np.inf, ss[ends]]
    auc = float(np.trapz(tpr, fpr))  # trapezoid gives ties half credit
    delta = fpr - (1 - tpr)
    hi = int(np.flatnonzero(delta >= 0)[0])
    lo = max(0, hi - 1)
    fraction = 0. if hi == lo else float(-delta[lo] / (delta[hi] - delta[lo]))
    eer = float(fpr[lo] + fraction * (fpr[hi] - fpr[lo]))
    if fraction == 1:
        threshold = float(thresholds[hi])
    elif fraction == 0:
        threshold = float(thresholds[lo]) if np.isfinite(thresholds[lo]) else None
    elif np.isfinite(thresholds[[lo, hi]]).all():
        threshold = float(thresholds[lo] + fraction * (thresholds[hi] - thresholds[lo]))
    else:
        threshold = None
    return {'n': len(y), 'bonafide': negatives, 'spoof': positives, 'eer': eer,
            'roc_auc': auc, 'diagnostic_interpolated_threshold': threshold,
            'threshold_interval': [float(x) if np.isfinite(x) else None for x in thresholds[[lo, hi]]],
            'roc_interpolation_fraction': fraction,
            'note': 'score>=threshold predicts spoof; interpolated EER may require randomization at ties; threshold is diagnostic only'}


def unit_tests():
    tests = []
    def check(name, y, s, eer, auc):
        result = roc_metrics(y, s)
        assert np.isclose(result['eer'], eer) and np.isclose(result['roc_auc'], auc), result
        tests.append({'name': name, 'status': 'passed', 'result': result})
    check('perfect', [0, 0, 1, 1], [-2, -1, 1, 2], 0, 1)
    check('reversed_no_auto_flip', [0, 0, 1, 1], [2, 1, -1, -2], 1, 0)
    check('all_ties_half_credit', [0, 1, 0, 1], [0, 0, 0, 0], .5, .5)
    check('random_rank_with_matched_class_pairs', [0, 1, 0, 1, 0, 1], [.2, .2, .9, .9, .5, .5], .5, .5)
    rng = np.random.default_rng(42)
    y = np.r_[np.zeros(20000), np.ones(20000)]
    result = roc_metrics(y, rng.normal(size=len(y)))
    assert abs(result['eer'] - .5) < .015 and abs(result['roc_auc'] - .5) < .015
    tests.append({'name': 'seed42_independent_random_40000', 'status': 'passed', 'result': result})
    # Hand-check AUC via pairwise wins and half ties, separately from ROC integration.
    y, s = np.array([0, 1, 0, 1, 0]), np.array([.4, .7, .7, .1, .3])
    pairwise = np.mean([(p > n) + .5 * (p == n) for p in s[y == 1] for n in s[y == 0]])
    assert np.isclose(roc_metrics(y, s)['roc_auc'], pairwise)
    for y, s in [([1, 1], [0, 1]), ([0, 1], [0, np.nan]), ([0, 2], [0, 1])]:
        try:
            roc_metrics(y, s)
            raise AssertionError('Invalid metric input accepted')
        except ValueError:
            pass
    tests.append({'name': 'pairwise_auc_and_invalid_inputs', 'status': 'passed'})
    return tests


if __name__ == '__main__':
    import json
    print(json.dumps(unit_tests(), indent=2))
