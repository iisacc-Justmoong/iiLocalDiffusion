#!/usr/bin/env python3
"""Provision or verify the pinned MOSS-Image color/depth control adapter."""
import argparse
import json
from pathlib import Path
import urllib.request

from provision_flux1_text_encoders import provision_resource

ROOT = Path(__file__).resolve().parents[1] / 'resources'
REPO = 'Comfy-Org/MOSS-Image'
REVISION = 'f7a9772e063542422b66dd492c4e17aa64f7797b'
SOURCE = 'loras/moss_image_color_depth_control.safetensors'
TARGET = 'models/moss_image_color_depth_control.safetensors'
FORMAT = 'moss-color-depth-control'


def validate_catalog(root):
    manifest = json.loads((root / 'generation-defaults.json').read_text())
    item = manifest.get('moss_color_depth_control_resource')
    if (not isinstance(item, dict) or item.get('repo_id') != REPO or item.get('revision') != REVISION
            or item.get('source_file') != SOURCE or item.get('file') != TARGET or item.get('format') != FORMAT
            or item.get('license') != 'apache-2.0' or type(item.get('size')) is not int or item['size'] <= 0
            or not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in item['sha256'])):
        raise ValueError('Invalid MOSS pinned color/depth control source or identity.')
    return item


def provision(root, *, check=False, opener=urllib.request.urlopen):
    root = Path(root).resolve()
    item = validate_catalog(root)
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{SOURCE}?download=true'
    return provision_resource(root, item, TARGET, 'moss-color-depth-control', url, 'MOSS control',
                              check=check, opener=opener)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify the resource without downloading')
    parser.add_argument('--resources', type=Path, default=ROOT, help='Directory containing generation-defaults.json')
    args = parser.parse_args()
    print(f'moss-color-depth-control: {provision(args.resources, check=args.check)}', flush=True)


if __name__ == '__main__':
    main()
