#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def load_best(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return data['summary']['best']


def main() -> None:
    p = argparse.ArgumentParser(description='Run paired_significance_test.py using selected operating points from JSON files.')
    p.add_argument('--name-a', required=True)
    p.add_argument('--config-a', required=True)
    p.add_argument('--checkpoint-a', required=True)
    p.add_argument('--select-a', required=True)
    p.add_argument('--name-b', required=True)
    p.add_argument('--config-b', required=True)
    p.add_argument('--checkpoint-b', required=True)
    p.add_argument('--select-b', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bootstrap-iters', type=int, default=10000)
    args = p.parse_args()

    a = load_best(args.select_a)
    b = load_best(args.select_b)

    cmd = [
        'python3', 'scripts/paired_significance_test.py',
        '--name-a', args.name_a,
        '--config-a', args.config_a,
        '--checkpoint-a', args.checkpoint_a,
        '--threshold-a', str(a['threshold']),
        '--tta-a', str(a['tta']),
        '--min-area-a', str(a['min_area']),
        '--name-b', args.name_b,
        '--config-b', args.config_b,
        '--checkpoint-b', args.checkpoint_b,
        '--threshold-b', str(b['threshold']),
        '--tta-b', str(b['tta']),
        '--min-area-b', str(b['min_area']),
        '--bootstrap-iters', str(args.bootstrap_iters),
        '--output', args.output,
    ]
    if a.get('keep_largest'):
        cmd.append('--keep-largest-a')
    if a.get('fill_holes'):
        cmd.append('--fill-holes-a')
    if b.get('keep_largest'):
        cmd.append('--keep-largest-b')
    if b.get('fill_holes'):
        cmd.append('--fill-holes-b')

    print('Running:', ' '.join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
