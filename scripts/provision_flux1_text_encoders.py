#!/usr/bin/env python3
"""Provision or verify the pinned FLUX.1 CLIP and T5 safetensors resources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / 'resources'
ROLES = {'clip': 'text_encoder', 't5': 'text_encoder_2'}
REPO = 'comfyanonymous/flux_text_encoders'
REVISION = '3bcfe977a9a2617a0b9025076030e217795fe028'
SOURCES = {'clip': 'clip_l.safetensors', 't5': 't5xxl_fp8_e4m3fn_scaled.safetensors'}
TARGETS = {'clip': 'text_encoders/flux1_clip_l.safetensors', 't5': 'text_encoders/flux1_t5xxl_fp8_scaled.safetensors'}


def validate_catalog(root, role):
    manifest = json.loads((root / 'generation-defaults.json').read_text())
    entries = manifest.get('fallback_flux_text_weights', [])
    matches = [item for item in entries if isinstance(item, dict) and item.get('component') == ROLES[role]]
    if len(matches) != 1:
        raise ValueError(f'Invalid FLUX {role} resource catalog.')
    item = matches[0]
    if (item.get('repo_id') != REPO or item.get('revision') != REVISION or item.get('source_file') != SOURCES[role]
            or item.get('file') != TARGETS[role] or not isinstance(item.get('size'), int) or item['size'] <= 0
            or not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in item['sha256'])):
        raise ValueError(f'Invalid FLUX {role} pinned source or identity.')
    return item


def verify(path, item, label='FLUX'):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'{label} resource is missing or is a symlink: {path}')
    before = path.stat()
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    after = path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise RuntimeError(f'{label} resource changed while hashing.')
    if after.st_size != item['size'] or digest != item['sha256']:
        raise ValueError(f'{label} resource does not match its pinned size and SHA-256: {path}')


def provision(root, role, *, check=False, opener=urllib.request.urlopen):
    root = Path(root).resolve()
    item = validate_catalog(root, role)
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{SOURCES[role]}?download=true"
    return provision_resource(root, item, TARGETS[role], f'flux1-{role}', url, 'FLUX', check=check, opener=opener)


def provision_resource(root, item, relative_target, name, url, label, *, check=False, opener=urllib.request.urlopen):
    """Resume, verify and atomically publish one fixed, catalog-validated resource."""
    root = Path(root).resolve()
    target = root / relative_target
    if target.exists() or target.is_symlink():
        verify(target, item, label)
        return 'verified'
    if check:
        raise ValueError(f'{label} resource is missing: {target}')
    state = root / '.state'
    if state.is_symlink() or target.parent.is_symlink():
        raise ValueError(f'{label} resource state and target directories must not be symlinks.')
    state.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = state / f'{name}-download.part'
    lock = state / f'{name}-download.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        if partial.is_symlink() or (partial.exists() and not partial.is_file()):
            raise ValueError(f'{label} partial download must be a regular file.')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item['size']:
            raise ValueError(f'{label} partial download exceeds its expected size.')
        if offset < item['size']:
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'})
            with opener(request, timeout=60) as response:
                content_range = response.headers.get('Content-Range', '')
                if response.status != 206 or not content_range.startswith(f'bytes {offset}-') or not content_range.endswith(f'/{item["size"]}'):
                    raise RuntimeError(f'The server did not honor the requested {label} download range.')
                with partial.open('ab') as output:
                    while chunk := response.read(1024 * 1024):
                        offset += len(chunk)
                        if offset > item['size']:
                            raise RuntimeError(f'{label} download exceeded its expected size.')
                        output.write(chunk)
        verify(partial, item, label)
        # A hard-link creation fails if a destination appeared concurrently.
        os.link(partial, target)
        partial.unlink()
        return 'provisioned'
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify installed resources without downloading')
    parser.add_argument('--role', choices=('clip', 't5', 'all'), default='all')
    parser.add_argument('--resources', type=Path, default=ROOT, help='Directory containing generation-defaults.json')
    args = parser.parse_args()
    for role in ROLES if args.role == 'all' else (args.role,):
        print(f'{role}: {provision(args.resources, role, check=args.check)}', flush=True)


if __name__ == '__main__':
    main()
