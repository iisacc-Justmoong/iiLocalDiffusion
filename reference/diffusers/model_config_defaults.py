"""Select immutable local configs when a single-file family is explicit."""
from pathlib import Path
from generation_defaults import read_defaults
from weight_files import file_sha256, file_signature

COMMON = {
    'scheduler/scheduler_config.json': (447, '52584b0765b0e256ed57111b8ace84dc4db6f1a1fc860c5f41ac7fd379ed150a'),
    'transformer/config.json': (549, '1486a7ca1bc41f32eaa02d5385f49edd32c12a61e9877a1cf9cfdd438d56b396'),
}
FILES = {
    'krea2-raw': {**COMMON, 'model_index.json': (549, 'd9af0734ddcb5f35ac6fed29c1739f4cb017a3b54530cf3dde53dde0d55efcc8')},
    'krea2-turbo': {**COMMON, 'model_index.json': (548, 'f6a91da66c70f4c974fb3009928c41ec15db4f6b725c117198e4ea2f0cb4da59')},
    'moss-image-sft': {
        'model_index.json': (412, 'ca80b1f562a506e0eb24afb1acb464fb583163e3c240c04e4bc69fea090db173'),
        'scheduler/scheduler_config.json': (494, '707322832cc507dfb8c51c0e0cd04d44ee2d30a9b036b6f3c47d6c17aa71adf0'),
        'transformer/config.json': (240, '3291c3a1c7adc22b8182a0de25ca8ec97474d72715ce84adbab09cc0be24f666'),
    },
    'flux1-schnell': {
        'model_index.json': (497, 'f9890c6f1367fcc9acfb51ac61c4e389ff4dd2a00db0ea76690dfaede0bb8c44'),
        'scheduler/scheduler_config.json': (448, '3ae6397b27645b6050ee4fb8a4af9aa0308de1b58838b70028aa190f64fefa46'),
        'transformer/config.json': (356, '995604317b84dc567da77f43553bc91ddaf1c0ab597e5989fba36e680088ca2d'),
        'text_encoder/config.json': (1784, '90aae9e87060f6c193eb4380bba91faeda6d2ab780e97db9a92465d06221a8ce'),
        'text_encoder_2/config.json': (593, 'a58c2192a7166501ad2382c3d7ca3d694a1259b71a23a1925887e5afe7adcbd8'),
        'vae/config.json': (774, 'bc1e208f414a315365fbecf426838f43b87c9d5c051219e0968a56e2644b2998'),
        'transformer-source-shapes.json': (36026, '1ace153d593b4aba486422efe467c34aea5945d23d2cf34d98462256d68837ff'),
    },
}


def resolve(pipeline, variant, resources=None, *, preset=None):
    if preset is not None:
        if preset not in FILES:
            raise ValueError('Unknown model configuration preset.')
        expected_pipeline = ('MOSSImagePipeline' if preset == 'moss-image-sft' else
                             'FluxPipeline' if preset == 'flux1-schnell' else 'Krea2Pipeline')
        if pipeline is not None and pipeline != expected_pipeline:
            raise ValueError('Model configuration preset conflicts with --pipeline-class.')
        if preset.startswith('krea2-') and variant not in ('auto', preset.removeprefix('krea2-')):
            raise ValueError('Model configuration preset conflicts with --krea2-variant.')
        name = preset
    elif pipeline == 'Krea2Pipeline' and variant in ('raw', 'turbo'):
        name = 'krea2-' + variant
    else:
        return None
    root, _ = read_defaults(resources)
    requested = root / 'model-configs' / name
    # A separate Society catalog may omit immutable SDK config files. A present
    # but incomplete or modified override remains an error.
    if not requested.exists() and not requested.is_symlink():
        root = (Path(__file__).resolve().parents[2] / 'resources').resolve()
        requested = root / 'model-configs' / name
    folder = requested.resolve()
    if not folder.is_relative_to(root):
        raise ValueError('Default model configuration escapes the resource package.')
    for relative, (size, digest) in FILES[name].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError(f'Default model configuration is missing or escapes its package: {relative}.')
        before = file_signature(path)
        if path.stat().st_size != size or file_sha256(path) != digest or file_signature(path) != before:
            raise ValueError(f'Default model configuration differs from its pinned identity: {relative}.')
    return folder
