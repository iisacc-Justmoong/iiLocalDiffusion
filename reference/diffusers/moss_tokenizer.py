"""Reuse the verified Qwen3.5 tokenizer for text-only MOSS prompts."""
from pathlib import Path

from generation_defaults import resource_catalogs
from krea2_tokenizer import NAMES, Selection, verify_selection
from weight_files import file_sha256, file_signature


def resolve(args, index, folder):
    if (args.pipeline_class or index['_class_name']) != 'MOSSImagePipeline':
        return None, 'not-configured-for-pipeline'
    if index.get('tokenizer') != ['transformers', 'Qwen2Tokenizer']:
        raise ValueError('MOSS requires a Qwen2Tokenizer declaration.')
    component = Path(folder) / 'tokenizer'
    if component.is_dir() and any(path.is_file() for path in component.iterdir()):
        return None, 'model'
    matches = []
    for root, manifest in resource_catalogs(getattr(args, 'generation_resources', None)):
        entries = manifest.get('fallback_tokenizers', [])
        if not isinstance(entries, list) or len(entries) > 64 or any(not isinstance(e, dict) for e in entries):
            raise ValueError('Invalid fallback tokenizer catalog.')
        matches = [entry for entry in entries if isinstance(entry.get('families'), list)
                   and 'moss-image' in entry['families']]
        if matches:
            break
    if len(matches) != 1 or matches[0].get('class_name') != 'Qwen2Tokenizer':
        raise ValueError('Install the verified MOSS Qwen3.5 tokenizer resources.')
    files = matches[0].get('files')
    if (not isinstance(files, list) or len(files) != len(NAMES)
            or any(not isinstance(item, dict) or not isinstance(item.get('file'), str) for item in files)
            or {Path(item['file']).name for item in files} != NAMES):
        raise ValueError('Invalid MOSS tokenizer resource manifest.')
    selected = []
    for item in files:
        relative = Path(item['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('MOSS tokenizer path escapes the resource package.')
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('MOSS tokenizer path escapes the resource package.')
        before = file_signature(path)
        digest = file_sha256(path)
        if file_signature(path) != before or path.stat().st_size != item.get('size') or digest != item.get('sha256'):
            raise ValueError('MOSS tokenizer resource differs from its manifest.')
        selected.append((str(path), path.stat().st_size, digest))
    folders = {str(Path(path).parent) for path, _, _ in selected}
    if len(folders) != 1:
        raise ValueError('MOSS tokenizer files must share one directory.')
    selection = Selection(folders.pop(), tuple(selected))
    verify_selection(selection)
    return selection, 'fallback'


def load(selection):
    from transformers import Qwen2Tokenizer
    verify_selection(selection)
    tokenizer = Qwen2Tokenizer.from_pretrained(selection.directory, local_files_only=True, trust_remote_code=False)
    if (tokenizer.pad_token_id != 248044 or tokenizer.eos_token_id != 248046
            or tokenizer.convert_tokens_to_ids('<|im_start|>') != 248045):
        raise ValueError('MOSS tokenizer special token IDs do not match Qwen3.5.')
    verify_selection(selection)
    return tokenizer
