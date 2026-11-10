#!/usr/bin/env python3
"""Provision pinned Transformers-format Depth Anything V2 resources."""
import argparse
import json
from pathlib import Path
import urllib.request

from provision_flux1_text_encoders import provision_resource

ROOT = Path(__file__).resolve().parents[1] / 'resources'
SOURCES = {
    'small': ('depth-anything/Depth-Anything-V2-Small-hf', '5426e4f0f36572d16453bbda7a8389317b1bef99', 'apache-2.0'),
    'large': ('depth-anything/Depth-Anything-V2-Large-hf', '7581137eff8d4e94f6e796d3baea0e9fa79b22d2', 'cc-by-nc-4.0'),
}
FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors')


def validate_catalog(root, variant):
    entries = json.loads((root / 'generation-defaults.json').read_text()).get('depth_anything_v2_resources')
    item = entries.get(variant) if isinstance(entries, dict) else None
    repo, revision, license_name = SOURCES[variant]
    if (not isinstance(item, dict) or item.get('repo_id') != repo or item.get('revision') != revision
            or item.get('license') != license_name or not isinstance(item.get('files'), list)):
        raise ValueError(f'Invalid Depth Anything V2 {variant} source or identity.')
    files = item['files']
    if len(files) != len(FILES) or any(not isinstance(entry, dict) for entry in files):
        raise ValueError(f'Invalid Depth Anything V2 {variant} file catalog.')
    by_name = {entry.get('source_file'): entry for entry in files}
    if set(by_name) != set(FILES):
        raise ValueError(f'Invalid Depth Anything V2 {variant} file catalog.')
    for name, entry in by_name.items():
        if (entry.get('file') != f'depth-estimators/depth-anything-v2-{variant}/{name}'
                or type(entry.get('size')) is not int or entry['size'] <= 0
                or not isinstance(entry.get('sha256'), str) or len(entry['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in entry['sha256'])):
            raise ValueError(f'Invalid Depth Anything V2 {variant} file identity: {name}')
    return by_name


def provision(root, variant, *, check=False, opener=urllib.request.urlopen):
    root = Path(root).resolve()
    files = validate_catalog(root, variant)
    repo, revision, _ = SOURCES[variant]
    results = {}
    for name in FILES:
        item = files[name]
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true'
        results[name] = provision_resource(root, item, item['file'], f'depth-v2-{variant}-{name}', url,
                                            f'Depth Anything V2 {variant} {name}', check=check, opener=opener)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=tuple(SOURCES), required=True)
    parser.add_argument('--check', action='store_true', help='Verify every file without downloading')
    parser.add_argument('--resources', type=Path, default=ROOT, help='Directory containing generation-defaults.json')
    args = parser.parse_args()
    print(json.dumps(provision(args.resources, args.variant, check=args.check), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
