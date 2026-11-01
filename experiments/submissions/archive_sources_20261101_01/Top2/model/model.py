"""Competition model adapter; official run.py is preserved separately."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def main(input_dir, output_csv, config_path=None, mode=None, sync_factory=None,
         require_official=False, offline_verify=False):
    if offline_verify:
        from audio_spectra.network_guard import deny_internet_sockets
        print(json.dumps({'offline_guard': deny_internet_sockets()}), file=sys.stderr)
    from audio_spectra.production import load_config, run_directory
    config = load_config(config_path)
    if mode:
        config['mode'] = mode
    if sync_factory is None:
        from rts import Sync
        sync_factory = Sync
    return run_directory(input_dir, output_csv, config, sync_factory,
                         require_official=require_official)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_csv')
    parser.add_argument('--config', default=os.environ.get('SPECTRA_CONFIG'))
    parser.add_argument('--mode', choices=['V1-Base', 'V2-Candidate', 'V3-Top2'], default=os.environ.get('SPECTRA_MODE'))
    parser.add_argument('--local-sync', action='store_true', help='Local progress logger; never claims platform rts')
    parser.add_argument('--require-official', action='store_true', default=os.environ.get('SPECTRA_REQUIRE_OFFICIAL') == '1')
    parser.add_argument('--offline-verify', action='store_true', default=os.environ.get('SPECTRA_OFFLINE_VERIFY') == '1')
    args = parser.parse_args()
    try:
        factory = None
        if args.local_sync:
            class LocalSync:
                def __init__(self, start, total):
                    self.total = total
                    print(f'LOCAL_SYNC {start}/{total}', file=sys.stderr, flush=True)
                def update(self, done):
                    print(f'LOCAL_SYNC {done}/{self.total}', file=sys.stderr, flush=True)
            factory = LocalSync
        main(args.input_dir, args.output_csv, args.config, args.mode, factory,
             args.require_official, args.offline_verify)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)
