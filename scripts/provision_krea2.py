#!/usr/bin/env python3
"""Provision pinned Krea 2 Turbo, depth and pose resources."""
import argparse
import json
from pathlib import Path
import urllib.request

from provision_flux1_text_encoders import provision_resource

ROOT = Path(__file__).resolve().parents[1] / 'resources'
SOURCES = {
    'turbo': ('Comfy-Org/Krea-2', 'eb1eddd3983a54678545a9b2c178c5853b30f7be',
              'diffusion_models/krea2_turbo_fp8_scaled.safetensors',
              'models/krea2_turbo_fp8_scaled.safetensors', 'krea2-turbo-scalar-fp8', 'krea-2-community-license'),
    'depth': ('Patil/Krea-2-depth-controlnet', '21889413aa7282a6e78bd510247cceccad034b24',
              'depth-control-lora.safetensors', 'models/krea2-depth-control-lora.safetensors',
              'krea2-depth-control-lora', 'krea-2-community-license'),
    'pose': ('thedeoxen/Krea-2-pose-controlnet', '1da9762ca46e6c9e4014b309bf0147889f10982a',
             'krea2_turbo_openpose_controlnet.safetensors', 'models/krea2-turbo-openpose-controlnet.safetensors',
             'krea2-ostris-pose-lora', 'apache-2.0'),
}


def validate_catalog(root, role):
    entries = json.loads((root / 'generation-defaults.json').read_text()).get('krea2_provisioned_resources')
    item = entries.get(role) if isinstance(entries, dict) else None
    repo, revision, source, target, format_name, license_name = SOURCES[role]
    if (not isinstance(item, dict) or item.get('repo_id') != repo or item.get('revision') != revision
            or item.get('source_file') != source or item.get('file') != target or item.get('format') != format_name
            or item.get('license') != license_name or type(item.get('size')) is not int
            or item['size'] <= 0 or not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in item['sha256'])):
        raise ValueError(f'Invalid Krea 2 {role} pinned resource source or identity.')
    return item


def provision(root, role, *, check=False, opener=urllib.request.urlopen):
    root = Path(root).resolve()
    item = validate_catalog(root, role)
    repo, revision, source, target, _, _ = SOURCES[role]
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{source}?download=true'
    return provision_resource(root, item, target, f'krea2-{role}', url, f'Krea 2 {role}',
                              check=check, opener=opener)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=tuple(SOURCES), required=True)
    parser.add_argument('--check', action='store_true', help='Verify the resource without downloading')
    parser.add_argument('--resources', type=Path, default=ROOT, help='Directory containing generation-defaults.json')
    args = parser.parse_args()
    print(f'krea2-{args.role}: {provision(args.resources, args.role, check=args.check)}', flush=True)


if __name__ == '__main__':
    main()
