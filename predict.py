"""Run a selected inference profile on a directory of local audio files."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


class ConsoleProgress:
    def __init__(self, start, total):
        self.total = total
        self.update(start)

    def update(self, done):
        print(f'PROGRESS {done}/{self.total}', file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/v2.json')
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights')
    parser.add_argument('--device', help='Override the profile device, for example cuda or cpu')
    args = parser.parse_args(argv)
    from audio_spectra.production import load_config, run_directory, SpectraEngine
    config = load_config(args.config)
    if args.device:
        config['device'] = args.device
    engine = SpectraEngine(config, weights=args.weights)
    run_directory(args.input_dir, args.output_csv, config,
                  sync_factory=ConsoleProgress, engine=engine)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'Inference failed: {exc}', file=sys.stderr)
        sys.exit(2)
