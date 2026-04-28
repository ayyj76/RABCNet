from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate seed-specific YAML configs for repeated experiments.')
    p.add_argument('--config', required=True, help='Base YAML config path')
    p.add_argument('--seeds', nargs='+', type=int, required=True, help='Training seeds to materialize')
    p.add_argument('--out-dir', default='configs/generated', help='Directory to write generated configs')
    p.add_argument('--output-prefix', default=None, help='Override output_dir prefix; defaults to base output_dir')
    p.add_argument('--keep-ph2-split-seed', action='store_true', help='Do not change data.ph2_split_seed when varying seeds')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.config)
    data = yaml.safe_load(base_path.read_text(encoding='utf-8'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_output = str(args.output_prefix or data.get('output_dir', f'outputs/{base_path.stem}'))
    for seed in args.seeds:
        cfg = yaml.safe_load(base_path.read_text(encoding='utf-8'))
        cfg['seed'] = int(seed)
        cfg['output_dir'] = f"{base_output}_s{seed}"
        if not args.keep_ph2_split_seed:
            if isinstance(cfg.get('data'), dict) and 'ph2_split_seed' in cfg['data']:
                cfg['data']['ph2_split_seed'] = int(seed)
        out_path = out_dir / f"{base_path.stem}_s{seed}.yaml"
        out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding='utf-8')
        print(out_path)


if __name__ == '__main__':
    main()
