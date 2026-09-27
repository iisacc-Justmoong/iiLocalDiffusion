"""Krea 2 controls with explicit variant, latent and numerical contracts.

Raw defaults follow krea-ai/krea-2's published quality example (52 / 3.5).
Krea guidance is cond + g*(cond-uncond), not conventional CFG.
"""
from contextlib import contextmanager
from functools import wraps
import math


def add_options(parser):
    parser.add_argument('--krea2-variant', choices=('auto', 'raw', 'turbo'), default='auto',
                        help='Krea 2 training variant; auto requires package is_distilled metadata')
    parser.add_argument('--krea2-mu', type=float,
                        help='Explicit Krea 2 exponential timestep shift; otherwise resolution-aware Raw / 1.15 Turbo')


def default_mu(variant, width, height):
    if variant == 'turbo':
        return 1.15
    return .5 + ((width // 16) * (height // 16) - 256) * (.65 / (6400 - 256))


def resolve(args, index):
    variant = getattr(args, 'krea2_variant', 'auto')
    mu = getattr(args, 'krea2_mu', None)
    if index.get('_class_name') != 'Krea2Pipeline':
        if variant != 'auto' or mu is not None:
            raise ValueError('Krea 2 controls require Krea2Pipeline, not another architecture.')
        return None
    declared = index.get('is_distilled')
    if declared is not None and type(declared) is not bool:
        raise ValueError('Krea 2 is_distilled must be a boolean.')
    detected = ('turbo' if declared else 'raw') if declared is not None else None
    if variant == 'auto':
        if detected is None:
            raise ValueError('Krea 2 variant is unknown; set --krea2-variant raw or turbo explicitly.')
        variant = detected
    elif detected is not None and variant != detected:
        raise ValueError('Krea 2 variant conflicts with package is_distilled metadata.')
    if args.dtype == 'float16':
        raise ValueError('Krea 2 float16 is refused: use float32 or bfloat16 to retain the trained dynamic range.')
    if mu is not None and (not math.isfinite(mu) or not 0 <= mu <= 4):
        raise ValueError('Krea 2 mu must be finite and in [0,4].')
    inputs = args.inputs
    sigmas = inputs.get('sigmas')
    if sigmas is not None:
        if (not isinstance(sigmas, list) or not 1 <= len(sigmas) <= 1000
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1 for v in sigmas)
                or any(a <= b for a, b in zip(sigmas, sigmas[1:]))):
            raise ValueError('Krea 2 sigmas must be a strictly descending finite list in (0,1].')
        if 'num_inference_steps' in inputs and inputs['num_inference_steps'] != len(sigmas):
            raise ValueError('Krea 2 custom sigma count must match requested steps.')
        inputs['num_inference_steps'] = len(sigmas)
    inputs.setdefault('num_inference_steps', 8 if variant == 'turbo' else 52)
    inputs.setdefault('guidance_scale', 0.0 if variant == 'turbo' else 3.5)
    inputs.setdefault('width', 1024)
    inputs.setdefault('height', 1024)
    for name, high in (('num_inference_steps', 1000), ('max_sequence_length', 8192), ('num_images_per_prompt', 128)):
        if name in inputs and (type(inputs[name]) is not int or not 1 <= inputs[name] <= high):
            raise ValueError(f'Krea 2 {name} must be an integer in [1,{high}].')
    for name in ('width', 'height'):
        if type(inputs[name]) is not int or not 16 <= inputs[name] <= 8192 or inputs[name] % 16:
            raise ValueError('Krea 2 dimensions must be multiples of 16 in [16,8192]; implicit rounding is refused.')
    guidance = inputs['guidance_scale']
    if type(guidance) not in (int, float) or not math.isfinite(guidance) or not 0 <= guidance <= 100:
        raise ValueError('Krea 2 guidance must be finite and in [0,100].')
    return {'variant': variant, 'mu_override': mu,
            'resolved_mu': mu if mu is not None else default_mu(variant, inputs['width'], inputs['height']),
            'guidance_convention': 'cond + g * (cond - uncond)', 'precision': args.dtype,
            'latent_contract': 'qwen-image-rgb-16ch-f8-patch2', 'finite_latents_required': True}


def validate_pipeline(pipeline, contract):
    if contract is None:
        return
    if type(pipeline.vae).__name__ != 'AutoencoderKLQwenImage' or pipeline.vae.config.z_dim != 16:
        raise ValueError('Krea 2 requires the 16-channel Qwen Image RGB VAE, not SDXL or FLUX VAE.')
    if pipeline.vae_scale_factor != 8 or pipeline.patch_size != 2 or pipeline.transformer.config.in_channels != 64:
        raise ValueError('Krea 2 latent packing must be 16 channels, f8, patch 2 (64 packed channels).')
    layers = pipeline.text_encoder_select_layers
    if (len(layers) != pipeline.transformer.config.num_text_layers or len(set(layers)) != len(layers)
            or any(type(v) is not int or v < 0 for v in layers)):
        raise ValueError('Krea 2 text layer selection must match the trained text-fusion layer count without duplicates.')
    encoder = pipeline.text_encoder
    if encoder is not None:
        config = getattr(encoder.config, 'text_config', encoder.config)
        if max(layers) > config.num_hidden_layers:
            raise ValueError('Krea 2 selected text layers exceed the Qwen3-VL encoder depth.')
    scheduler = pipeline.scheduler
    if type(scheduler).__name__ != 'FlowMatchEulerDiscreteScheduler':
        raise ValueError('Krea 2 requires FlowMatchEulerDiscreteScheduler for its validated flow schedule.')
    config = scheduler.config
    if not config.use_dynamic_shifting or getattr(config, 'time_shift_type', 'exponential') != 'exponential':
        raise ValueError('Krea 2 scheduler requires dynamic exponential time shifting.')
    # This contract describes the published resolution-dependent schedule. A
    # per-request mu override is the supported way to explore another shift.
    for key, expected in (('base_shift', .5), ('max_shift', 1.15),
                          ('base_image_seq_len', 256), ('max_image_seq_len', 6400)):
        if not math.isclose(float(getattr(config, key)), expected):
            raise ValueError(f'Krea 2 scheduler {key} does not match the published schedule.')
    pipeline.register_to_config(is_distilled=contract['variant'] == 'turbo')


@contextmanager
def execution(pipeline, inputs, contract, torch):
    """Scope scheduler overrides and compose finite checks with SDK previews."""
    if contract is None:
        yield None
        return
    validate_pipeline(pipeline, contract)
    for key in ('latents', 'prompt_embeds', 'negative_prompt_embeds'):
        value = inputs.get(key)
        if value is not None and (not torch.is_tensor(value) or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Krea 2 {key} must be a finite tensor.')
    for prefix in ('', 'negative_'):
        embeds = inputs.get(prefix + 'prompt_embeds')
        if embeds is None:
            continue
        mask = inputs.get(prefix + 'prompt_embeds_mask')
        cfg = pipeline.transformer.config
        if (embeds.ndim != 4 or tuple(embeds.shape[2:]) != (cfg.num_text_layers, cfg.text_hidden_dim)
                or mask is None or not torch.is_tensor(mask) or mask.dtype != torch.bool
                or tuple(mask.shape) != tuple(embeds.shape[:2])):
            raise ValueError('Krea 2 embeddings require BSLC layout and a matching boolean BS attention mask.')
    latent = inputs.get('latents')
    if latent is not None:
        embeds = inputs.get('prompt_embeds')
        prompt = inputs.get('prompt')
        batch = embeds.shape[0] if embeds is not None else len(prompt) if isinstance(prompt, list) else 1
        expected = (batch * inputs.get('num_images_per_prompt', 1),
                    (inputs['height'] // 16) * (inputs['width'] // 16), 64)
        if tuple(latent.shape) != expected:
            raise ValueError(f'Krea 2 packed latents must have shape {expected}.')
    trace = {'executed_steps': 0, 'finite_latents': True, 'mu': contract['resolved_mu'],
             'negative_conditioning_enabled': inputs['guidance_scale'] > 0,
             'component_dtypes': {name: str(getattr(getattr(pipeline, name, None), 'dtype', None))
                                  for name in ('transformer', 'text_encoder', 'vae')}}
    expected_dtype = 'torch.' + contract['precision']
    if any(dtype not in ('None', expected_dtype) for dtype in trace['component_dtypes'].values()):
        raise ValueError('Krea 2 loaded component dtype does not match the requested precision contract.')
    previous_callback = inputs.get('callback_on_step_end')
    previous_tensors = inputs.get('callback_on_step_end_tensor_inputs')
    def checked(pipe, step, timestep, values):
        latent = values['latents']
        if not bool(torch.isfinite(latent).all()):
            raise RuntimeError(f'Krea 2 non-finite latents at denoising step {step + 1}; output was not published.')
        trace['executed_steps'] += 1
        return previous_callback(pipe, step, timestep, values) if previous_callback else values
    scheduler = pipeline.scheduler
    previous_method = scheduler.set_timesteps
    had_instance_method = 'set_timesteps' in vars(scheduler)
    instance_method = vars(scheduler).get('set_timesteps')
    @wraps(previous_method)
    def controlled(*args, **kwargs):
        kwargs['mu'] = contract['resolved_mu']
        result = previous_method(*args, **kwargs)
        trace['sigmas'] = scheduler.sigmas.detach().float().cpu().tolist()
        return result
    inputs['callback_on_step_end'] = checked
    inputs['callback_on_step_end_tensor_inputs'] = list(dict.fromkeys([*(previous_tensors or []), 'latents']))
    scheduler.set_timesteps = controlled
    try:
        yield trace
    finally:
        if had_instance_method:
            scheduler.set_timesteps = instance_method
        else:
            del scheduler.set_timesteps
        for key, value in (('callback_on_step_end', previous_callback),
                           ('callback_on_step_end_tensor_inputs', previous_tensors)):
            if value is None:
                inputs.pop(key, None)
            else:
                inputs[key] = value
