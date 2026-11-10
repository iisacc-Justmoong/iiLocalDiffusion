"""Reuse the SDK's verified Qwen tokenizer for Krea 2 without network access."""
from dataclasses import dataclass
from pathlib import Path

from generation_defaults import read_defaults
from weight_files import file_sha256, file_signature

NAMES = frozenset(('vocab.json', 'merges.txt', 'tokenizer_config.json'))
# Transformers can prefer these optional files over or in addition to the
# catalogued resources. Do not consume untracked overrides from this folder.
OVERRIDES = ('tokenizer.json', 'added_tokens.json', 'special_tokens_map.json', 'chat_template.jinja')


@dataclass(frozen=True)
class Selection:
    directory: str
    files: tuple

    def metadata(self):
        return {'status': 'fallback', 'class': 'Qwen2Tokenizer', 'directory': self.directory,
                'files': [{'path': path, 'size_bytes': size, 'sha256': digest} for path, size, digest in self.files]}


def resolve(args, index, folder, encoder_status):
    if (args.pipeline_class or index['_class_name']) != 'Krea2Pipeline':
        return None, 'not-configured-for-pipeline'
    if encoder_status == 'precomputed-embeddings':
        return None, 'precomputed-embeddings'
    declaration = index.get('tokenizer')
    if declaration not in (None, [None, None], ['transformers', 'Qwen2Tokenizer'],
                           ['transformers', 'Qwen2TokenizerFast']):
        raise ValueError('Krea 2 requires a Qwen2Tokenizer declaration.')
    component = Path(folder) / 'tokenizer'
    if declaration not in (None, [None, None]) and component.is_dir() and any(p.is_file() for p in component.iterdir()):
        return None, 'model'
    root, manifest = read_defaults(getattr(args, 'generation_resources', None))
    entries = manifest.get('fallback_tokenizers', [])
    if not isinstance(entries, list) or len(entries) > 64 or any(not isinstance(e, dict) for e in entries):
        raise ValueError('Invalid fallback tokenizer catalog.')
    matches = [e for e in entries if isinstance(e.get('families'), list) and 'krea2' in e['families']]
    if len(matches) != 1 or matches[0].get('class_name') != 'Qwen2Tokenizer':
        raise ValueError('Install the verified Krea 2 tokenizer resources.')
    files = matches[0].get('files')
    if (not isinstance(files, list) or len(files) != len(NAMES)
            or any(not isinstance(item, dict) or not isinstance(item.get('file'), str) for item in files)
            or {Path(item['file']).name for item in files} != NAMES):
        raise ValueError('Invalid Krea 2 tokenizer resource manifest.')
    selected = []
    for item in files:
        relative = Path(item['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Krea 2 tokenizer path escapes the resource package.')
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Krea 2 tokenizer path escapes the resource package.')
        before = file_signature(path)
        digest = file_sha256(path)
        if file_signature(path) != before or path.stat().st_size != item.get('size') or digest != item.get('sha256'):
            raise ValueError('Krea 2 tokenizer resource differs from its manifest.')
        selected.append((str(path), path.stat().st_size, digest))
    folders = {str(Path(path).parent) for path, _, _ in selected}
    if len(folders) != 1:
        raise ValueError('Krea 2 tokenizer files must share one directory.')
    selection = Selection(folders.pop(), tuple(selected))
    verify_selection(selection)
    return selection, 'fallback'


def verify_selection(selection):
    if selection is None:
        return
    if any((Path(selection.directory) / name).exists() for name in OVERRIDES):
        raise RuntimeError('Krea 2 tokenizer resource changed: untracked tokenizer override.')
    for name, size, digest in selection.files:
        path = Path(name)
        if not path.is_file() or path.stat().st_size != size or file_sha256(path) != digest:
            raise RuntimeError('Krea 2 tokenizer resource changed during generation.')


def load(selection):
    from transformers import Qwen2Tokenizer
    verify_selection(selection)
    result = Qwen2Tokenizer.from_pretrained(selection.directory, local_files_only=True, trust_remote_code=False)
    if (result.pad_token_id != 151643 or result.eos_token_id != 151645
            or result.convert_tokens_to_ids('<|im_start|>') != 151644):
        raise ValueError('Krea 2 tokenizer special token IDs do not match the model.')
    verify_selection(selection)
    return result
