#!/usr/bin/env python3
"""Provision or verify the pinned Qwen3-0.6B-Base Anima text encoder."""
import argparse
import json
from pathlib import Path
import urllib.request

from provision_flux1_text_encoders import provision_resource

ROOT = Path(__file__).resolve().parents[1] / 'resources'
REPO = 'Qwen/Qwen3-0.6B-Base'
REVISION = 'da05741fc6420bf928f7e1eaacd5210b8a4c1b64'
SOURCE = 'model.safetensors'
TARGET = 'text_encoders/anima_qwen3_0_6b_base.safetensors'
SIZE = 1192135096
SHA256 = 'cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba'


def validate_catalog(root):
    manifest = json.loads((root / 'generation-defaults.json').read_text())
    entries = manifest.get('fallback_llms', [])
    if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
        raise ValueError('Invalid Anima pinned text encoder source or identity.')
    matches = [item for item in entries if isinstance(item.get('families'), list)
               and 'anima' in item['families']]
    if len(matches) != 1:
        raise ValueError('Invalid Anima pinned text encoder source or identity.')
    item = matches[0]
    if (item.get('repo_id') != REPO or item.get('revision') != REVISION
            or item.get('source_file') != SOURCE or item.get('file') != TARGET
            or item.get('format') != 'safetensors' or item.get('class_name') != 'Qwen3ForCausalLM'
            or item.get('families') != ['anima'] or item.get('optional') is not True
            or item.get('license') != 'apache-2.0' or type(item.get('size')) is not int
            or item['size'] != SIZE or item.get('sha256') != SHA256):
        raise ValueError('Invalid Anima pinned text encoder source or identity.')
    return item


def provision(root, *, check=False, opener=urllib.request.urlopen):
    root = Path(root).resolve()
    item = validate_catalog(root)
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{SOURCE}?download=true'
    return provision_resource(root, item, TARGET, 'anima-qwen3', url, 'Anima', check=check, opener=opener)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify the resource without downloading')
    parser.add_argument('--resources', type=Path, default=ROOT, help='Directory containing generation-defaults.json')
    args = parser.parse_args()
    print(f'anima-qwen3: {provision(args.resources, check=args.check)}', flush=True)


if __name__ == '__main__':
    main()
