"""Safely convert explicit legacy video weights without losing their sources."""

from pathlib import Path

from weight_files import file_signature


def _verify_snapshots(args):
    for records in getattr(args, '_native_legacy_sources', {}).values():
        for path, signature in records:
            try:
                unchanged = file_signature(Path(path)) == tuple(signature)
            except (OSError, ValueError):
                unchanged = False
            if not unchanged:
                raise RuntimeError(f'PyTorch video source changed after selection: {path}')


def prepare(args):
    from checkpoint_conversion import materialize_safetensors, verify_conversion_source
    from downloaded_model import model_container_format
    from video_options import configuration

    _verify_snapshots(args)
    original_configuration = configuration(args)
    conversions, replacements = {}, {}
    for name in ('model', 'text_encoder', 'video_vae', 'high_noise_model', 'vision_encoder'):
        value = getattr(args, name, None)
        if value is None or not Path(value).is_file():
            continue
        if model_container_format(value) not in ('pytorch', 'pytorch-index'):
            continue
        conversion = materialize_safetensors(value, args.cache_dir / 'checkpoint-conversions')
        if not conversion.get('converted'):
            raise RuntimeError('PyTorch video conversion did not produce Safetensors weights.')
        verify_conversion_source(conversion)
        conversions[name] = conversion
        replacements[name] = str(conversion['converted_path']) if name == 'model' else Path(conversion['converted_path'])
    _verify_snapshots(args)
    for conversion in conversions.values():
        verify_conversion_source(conversion)
    if conversions:
        args._native_original_configuration = original_configuration
        args._native_conversions = conversions
        for name, value in replacements.items():
            setattr(args, name, value)
    return args


def verify(args):
    from checkpoint_conversion import verify_conversion_source
    _verify_snapshots(args)
    for conversion in getattr(args, '_native_conversions', {}).values():
        verify_conversion_source(conversion)
