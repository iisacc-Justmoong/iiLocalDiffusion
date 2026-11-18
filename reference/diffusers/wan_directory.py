"""Adapt declared local Wan text/image-to-video directories to native requests."""
import copy
import json
from pathlib import Path

from animation_video import output_targets
from image_package import inspect_wan_directory
from video_options import configuration
from weight_files import file_signature


def pipeline_class(directory):
    """Inspect only bounded JSON to select a backend; native validation follows."""
    path = Path(directory) / 'model_index.json'
    try:
        with path.open('rb') as stream:
            raw = stream.read(1024 * 1024 + 1)
    except OSError:
        return None
    if len(raw) > 1024 * 1024:
        raise ValueError('Video model_index.json exceeds the 1 MiB limit.')
    try:
        value = json.loads(raw.decode('utf-8'))
    except (UnicodeError, ValueError):
        return None
    return value.get('_class_name') if isinstance(value, dict) else None


def prepare(args):
    """Select declared files, preserving the original request and source identity."""
    # Desktop clients send the common LTX defaults even for models that do not
    # use them. Accept those neutral values, but never ignore a changed value.
    neutral = {'decode_timestep': 0.05, 'decode_noise_scale': 0.025,
               'image_cond_noise_scale': 0, 'max_sequence_length': 256}
    provided = set(getattr(args, '_provided', ()))
    unsupported = {name for name, default in neutral.items()
                   if name in provided and getattr(args, name) != default}
    for name in ('text_encoder', 'video_vae', 'high_noise_model', 'vision_encoder', 'moe_boundary'):
        if getattr(args, name, None) is not None:
            unsupported.add(name)
    if unsupported:
        raise ValueError('Native Wan directory routing does not support these explicit options: '
                         + ', '.join(sorted(unsupported)))
    root = Path(args.model).resolve()
    targets = [path.resolve() for path in output_targets(args.output)]
    if any(target == root or target.is_relative_to(root) for target in targets):
        raise ValueError('Video output must be outside the source model directory.')
    route = inspect_wan_directory(root)
    declared = pipeline_class(root)
    pipeline = route.get('pipeline')
    # Earlier versions of the v1 inspector accepted only WanPipeline and did
    # not emit these two fields. Preserve that text-only contract, but do not
    # infer an image encoder from an incomplete newer report.
    if pipeline is None and declared == 'WanPipeline':
        pipeline = 'WanPipeline'
    if pipeline != declared or (pipeline == 'WanImageToVideoPipeline' and 'vision_encoder' not in route):
        raise ValueError('The selected Wan pipeline requires an updated iiLocalDiffusion SDK runtime.')
    vision_encoder = route.get('vision_encoder')
    for shot in args.shots:
        if pipeline == 'WanPipeline':
            if shot['conditions'] or shot['continue_previous']:
                raise ValueError('WanPipeline is text-to-video and does not accept image conditions.')
        elif pipeline == 'WanImageToVideoPipeline':
            if any(condition['frame'] != 0 for condition in shot['conditions']):
                raise ValueError('WanImageToVideoPipeline supports a first-frame image only.')
            if not shot['continue_previous'] and not shot['conditions']:
                raise ValueError('WanImageToVideoPipeline requires a first-frame image for each shot.')
        else:
            raise ValueError('Unsupported Wan directory pipeline.')
    sources = {Path(path) for path in route['dependencies']}
    if any(source.resolve() in targets or source.resolve().is_relative_to(targets[2]) for source in sources):
        raise ValueError('Video output must not replace a model component or shard.')
    snapshots = {path: file_signature(path) for path in sources}
    if inspect_wan_directory(root) != route or any(file_signature(path) != stamp for path, stamp in snapshots.items()):
        raise RuntimeError('Wan model directory changed during component selection.')

    def verify():
        try:
            if any(file_signature(path) != stamp for path, stamp in snapshots.items()):
                raise RuntimeError('Wan model directory changed during generation.')
        except OSError as error:
            raise RuntimeError('Wan model directory changed during generation.') from error

    adapted = copy.copy(args)
    adapted._native_original_configuration = configuration(args)
    adapted._native_source_verifier = verify
    components = {name: route[name] for name in ('model', 'text_encoder', 'video_vae', 'high_noise_model')}
    components['vision_encoder'] = vision_encoder
    adapted._native_directory_source = {'directory': str(root), 'declared_pipeline': pipeline,
        'declared_scheduler': route['scheduler'], 'execution_scheduler': 'native-engine-default',
        'components': components,
        'moe_boundary': route['moe_boundary']}
    adapted.model = route['model']
    adapted.text_encoder = Path(route['text_encoder'])
    adapted.video_vae = Path(route['video_vae'])
    adapted.high_noise_model = Path(route['high_noise_model']) if route['high_noise_model'] else None
    adapted.vision_encoder = Path(vision_encoder) if vision_encoder else None
    adapted.moe_boundary = route['moe_boundary']
    return adapted
