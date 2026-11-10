#!/usr/bin/env python3
"""Provision the explicitly selected, non-commercial InstantX FLUX.1 Canny ControlNet."""
import argparse
import json
from pathlib import Path
import urllib.request

from provision_flux1_text_encoders import provision_resource

ROOT = Path(__file__).resolve().parents[1] / 'resources'
REPO = 'InstantX/FLUX.1-dev-Controlnet-Canny'
REVISION = 'e7cee4b2afa335bf8f913f2b044764b1a1b01881'
FILES = ('config.json', 'diffusion_pytorch_model.safetensors')


def validate_catalog(root):
    item = json.loads((root / 'generation-defaults.json').read_text()).get('flux1_instantx_canny_resource')
    if (not isinstance(item, dict) or item.get('repo_id') != REPO or item.get('revision') != REVISION
            or item.get('license') != 'flux-1-dev-non-commercial-license' or not isinstance(item.get('files'), list)):
        raise ValueError('Invalid InstantX FLUX Canny source or identity.')
    entries = item['files']
    if len(entries) != len(FILES) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError('Invalid InstantX FLUX Canny file catalog.')
    by_name = {entry.get('source_file'): entry for entry in entries}
    if set(by_name) != set(FILES):
        raise ValueError('Invalid InstantX FLUX Canny file catalog.')
    for name, entry in by_name.items():
        if (entry.get('file') != f'models/flux1-instantx-canny/{name}'
                or type(entry.get('size')) is not int or entry['size'] <= 0
                or not isinstance(entry.get('sha256'), str) or len(entry['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in entry['sha256'])):
            raise ValueError(f'Invalid InstantX FLUX Canny file identity: {name}')
    return by_name


def provision(root, *, check=False, opener=urllib.request.urlopen):
    root = Path(root).resolve()
    files = validate_catalog(root)
    results = {}
    for name in FILES:
        item = files[name]
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true'
        results[name] = provision_resource(root, item, item['file'], f'flux1-instantx-canny-{name}', url,
                                           f'InstantX FLUX Canny {name}', check=check, opener=opener)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify both files without downloading')
    parser.add_argument('--resources', type=Path, default=ROOT, help='Directory containing generation-defaults.json')
    args = parser.parse_args()
    print(json.dumps(provision(args.resources, check=args.check), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
