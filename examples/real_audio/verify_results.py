"""Validate demo CSV coverage and compare reruns to the published reference."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_csv(path, expected):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['filename', 'label', 'confidence']:
            raise ValueError('CSV header mismatch')
        rows = list(reader)
    names = [row['filename'] for row in rows]
    if len(names) != len(expected) or len(set(names)) != len(names) or set(names) != expected:
        raise ValueError('CSV coverage/duplicates mismatch')
    for row in rows:
        score = float(row['confidence'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid confidence')
        if row['label'] != ('real' if score > .5 else 'fake'):
            raise ValueError('Label/score mismatch')
    return {row['filename']: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--atol', type=float, default=1e-5,
                        help='Absolute score tolerance; not a guarantee across hardware')
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        raise ValueError('atol must be finite and nonnegative')
    expected = {row['filename'] for row in json.loads((ROOT / 'inputs.json').read_text())['files']}
    reference = read_csv(ROOT / 'result.csv', expected)
    actual = read_csv(args.csv, expected)
    differences = {name: abs(float(actual[name]['confidence']) - float(reference[name]['confidence']))
                   for name in expected}
    if any(actual[name]['label'] != reference[name]['label'] for name in expected):
        raise ValueError('Label differs from reference')
    maximum = max(differences.values())
    if maximum > args.atol:
        raise ValueError(f'Maximum score difference {maximum} exceeds {args.atol}')
    print(json.dumps({'rows': len(actual), 'coverage': '3/3', 'max_abs_score_difference': maximum,
                      'atol': args.atol, 'status': 'Passed'}, indent=2))


if __name__ == '__main__':
    main()
