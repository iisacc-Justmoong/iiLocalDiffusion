#!/usr/bin/env python3
"""Provision or verify the pinned FLUX.1 Schnell FP8 single-file checkpoint."""
import argparse
import json
from pathlib import Path
import urllib.request

from provision_flux1_text_encoders import provision_resource

ROOT = Path(__file__).resolve().parents[1] / 'resources'
REPO = 'Comfy-Org/Flux-1-Schnell'
REVISION = 'd31e04067066d92c55ece2b01748dcc337725d7a'
SOURCE = 'flux1-schnell-fp8.safetensors'
TARGET = 'models/flux1-schnell-fp8.safetensors'
FORMAT = 'flux1-schnell-original-fp8-full'


def validate_catalog(root):
    item = json.loads((root / 'generation-defaults.json').read_text()).get('flux1_schnell_checkpoint_resource')
    if (not isinstance(item, dict) or item.get('repo_id') != REPO or item.get('revision') != REVISION
            or item.get('source_file') != SOURCE or item.get('file') != TARGET or item.get('format') != FORMAT
            or item.get('license') != 'apache-2.0' or type(item.get('size')) is not int or item['size'] <= 0
            or not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in item['sha256'])):
        raise ValueError('Invalid FLUX.1 Schnell pinned checkpoint source or identity.')
    return item


def provision(root, *, check=False, opener=urllib.request.urlopen):
    root = Path(root).resolve()
    item = validate_catalog(root)
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{SOURCE}?download=true'
    return provision_resource(root, item, TARGET, 'flux1-schnell', url, 'FLUX.1 Schnell',
                              check=check, opener=opener)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify the resource without downloading')
    parser.add_argument('--resources', type=Path, default=ROOT, help='Directory containing generation-defaults.json')
    args = parser.parse_args()
    print(f'flux1-schnell: {provision(args.resources, check=args.check)}', flush=True)


if __name__ == '__main__':
    main()
