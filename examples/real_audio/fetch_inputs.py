"""Fetch byte-identical, pinned FSDD inputs; never overwrite different bytes."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'inputs.json').read_text())
    folder = ROOT / 'input'
    folder.mkdir(exist_ok=True)
    for entry in manifest['files']:
        path = folder / entry['filename']
        data = path.read_bytes() if path.exists() else urllib.request.urlopen(entry['url'], timeout=30).read()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Input SHA256/size mismatch: ' + entry['filename'])
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(data)
        print(entry['filename'], entry['sha256'], 'OK')


if __name__ == '__main__':
    main()
