"""Keep FLUX.1 CLIP tokenization within its loaded encoder's position table."""
from dataclasses import dataclass
from pathlib import Path

from generation_defaults import read_defaults
from weight_files import file_sha256, file_signature

FLUX1_CLIP_PIPELINES = frozenset({
    'FluxPipeline', 'FluxImg2ImgPipeline', 'FluxInpaintPipeline', 'FluxFillPipeline',
    'FluxKontextPipeline', 'FluxKontextInpaintPipeline', 'FluxControlPipeline',
    'FluxControlImg2ImgPipeline', 'FluxControlInpaintPipeline', 'FluxControlNetPipeline',
    'FluxControlNetImg2ImgPipeline', 'FluxControlNetInpaintPipeline',
})

COMPONENTS = {
    'tokenizer': ('CLIPTokenizer', frozenset(('vocab.json', 'merges.txt', 'tokenizer_config.json', 'special_tokens_map.json'))),
    'tokenizer_2': ('T5TokenizerFast', frozenset(('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'))),
}


@dataclass(frozen=True)
class Selection:
    role: str
    class_name: str
    directory: str
    files: tuple

    def metadata(self):
        return {'status': 'fallback', 'class': self.class_name, 'directory': self.directory,
                'files': [{'path': path, 'size_bytes': size, 'sha256': digest} for path, size, digest in self.files]}


def verify_selections(selections):
    for selection in (selections or {}).values():
        directory = Path(selection.directory)
        expected = {Path(path).name for path, _, _ in selection.files}
        if not directory.is_dir() or {p.name for p in directory.iterdir()} != expected:
            raise RuntimeError('FLUX tokenizer resource has an untracked or missing file.')
        for name, size, digest in selection.files:
            path = Path(name)
            if not path.is_file() or path.stat().st_size != size or file_sha256(path) != digest:
                raise RuntimeError('FLUX tokenizer resource changed during generation.')


def uses_precomputed(inputs):
    has_negative_text = any(inputs.get(key) is not None for key in ('negative_prompt', 'negative_prompt_2'))
    has_negative_embeddings = all(inputs.get(key) is not None for key in ('negative_prompt_embeds', 'negative_pooled_prompt_embeds'))
    return (inputs.get('prompt_embeds') is not None and inputs.get('pooled_prompt_embeds') is not None
            and (not has_negative_text or has_negative_embeddings))


def resolve(args, index, folder):
    if (args.pipeline_class or index['_class_name']) not in FLUX1_CLIP_PIPELINES:
        return {}, {}
    selections, statuses = {}, {}
    precomputed = uses_precomputed(args.inputs)
    for role, (class_name, names) in COMPONENTS.items():
        declaration = index.get(role)
        if declaration in (None, [None, None]):
            statuses[role] = 'not-declared'
            continue
        component = Path(folder) / role
        if component.is_dir() and any(path.is_file() for path in component.iterdir()):
            statuses[role] = 'model'
            continue
        allowed = ('CLIPTokenizer', 'CLIPTokenizerFast') if role == 'tokenizer' else ('T5Tokenizer', 'T5TokenizerFast')
        if not isinstance(declaration, list) or len(declaration) != 2 or declaration[0] != 'transformers' or declaration[1] not in allowed:
            raise ValueError(f'FLUX {role} declares an unsupported tokenizer class.')
        if precomputed:
            statuses[role] = 'precomputed-embeddings'
            continue
        root, manifest = read_defaults(getattr(args, 'generation_resources', None))
        entries = manifest.get('fallback_tokenizers', [])
        if not isinstance(entries, list) or len(entries) > 64 or any(not isinstance(entry, dict) for entry in entries):
            raise ValueError('Invalid fallback tokenizer catalog.')
        matches = [entry for entry in entries if isinstance(entry.get('families'), list)
                   and 'flux1' in entry['families'] and entry.get('component') == role]
        if len(matches) != 1 or matches[0].get('class_name') != class_name:
            raise ValueError(f'Install the verified FLUX {role} resources.')
        files = matches[0].get('files')
        if (not isinstance(files, list) or len(files) != len(names)
                or any(not isinstance(item, dict) or not isinstance(item.get('file'), str) for item in files)
                or {Path(item['file']).name for item in files} != names):
            raise ValueError(f'Invalid FLUX {role} resource manifest.')
        selected = []
        for item in files:
            relative = Path(item['file'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('FLUX tokenizer path escapes the resource package.')
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('FLUX tokenizer path escapes the resource package.')
            before = file_signature(path)
            digest = file_sha256(path)
            size = path.stat().st_size
            if file_signature(path) != before or size != item.get('size') or digest != item.get('sha256'):
                raise ValueError('FLUX tokenizer resource differs from its manifest.')
            selected.append((str(path), size, digest))
        directories = {str(Path(path).parent) for path, _, _ in selected}
        if len(directories) != 1:
            raise ValueError('FLUX tokenizer files must share one directory.')
        selections[role] = Selection(role, class_name, directories.pop(), tuple(selected))
        statuses[role] = 'fallback'
    verify_selections(selections)
    return selections, statuses


def load(selection):
    from transformers import CLIPTokenizer, T5TokenizerFast
    verify_selections({selection.role: selection})
    cls = {'CLIPTokenizer': CLIPTokenizer, 'T5TokenizerFast': T5TokenizerFast}[selection.class_name]
    tokenizer = cls.from_pretrained(selection.directory, local_files_only=True, trust_remote_code=False)
    expected = (49406, 49407, 49407) if selection.role == 'tokenizer' else (None, 1, 0)
    if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != expected:
        raise ValueError(f'FLUX {selection.role} special token IDs do not match the existing tokenizer.')
    verify_selections({selection.role: selection})
    return tokenizer


def bound_clip_length(pipeline):
    name = type(pipeline).__name__
    if name not in FLUX1_CLIP_PIPELINES:
        return None
    tokenizer = getattr(pipeline, 'tokenizer', None)
    encoder = getattr(pipeline, 'text_encoder', None)
    if tokenizer is None or encoder is None:
        return None
    capacity = getattr(getattr(encoder, 'config', None), 'max_position_embeddings', None)
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
        raise ValueError('FLUX CLIP text encoder has an invalid position capacity.')
    limit = getattr(tokenizer, 'model_max_length', None)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError('FLUX CLIP tokenizer has an invalid maximum length.')
    if limit <= capacity:
        return None
    tokenizer.model_max_length = capacity
    pipeline.tokenizer_max_length = capacity
    return {'original': limit, 'effective': capacity}
