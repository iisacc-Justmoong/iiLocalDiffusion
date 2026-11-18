"""A bounded two-stage MOSS target-size refinement plan."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Plan:
    base_size: tuple[int, int]
    target_size: tuple[int, int]
    strength: float
    upscaler: str

    def metadata(self):
        return {'base_size': list(self.base_size), 'target_size': list(self.target_size),
                'denoising_strength': self.strength, 'upscaler': self.upscaler}


def resolve(args, pipeline_name):
    if not args.hires_fix:
        if args.hires_upscaler is not None or args.hires_strength is not None:
            raise ValueError('HiRes refinement options require --hires-fix.')
        return None
    if pipeline_name != 'MOSSImagePipeline':
        raise ValueError('HiRes refinement is unavailable for this generic pipeline; use --no-hires-fix.')
    if args.offload != 'none':
        raise ValueError('MOSS HiRes currently requires --offload none.')
    if args.generic_controlnet or args.moss_regions or args.moss_rapid:
        raise ValueError('MOSS HiRes does not yet support control, regions, or rapid adapters.')
    inputs = args.inputs
    if any(name in inputs for name in ('initial_image', 'denoising_strength', 'latents',
                                        'prompt_embeds', 'negative_prompt_embeds')):
        raise ValueError('MOSS HiRes requires a text prompt and owns its initial image and latents.')
    if inputs.get('output_type', 'pil') != 'pil' or args.tensor_outputs:
        raise ValueError('MOSS HiRes requires PIL image output.')
    prompt = inputs.get('prompt')
    if isinstance(prompt, list) and len(prompt) != 1:
        raise ValueError('MOSS HiRes currently supports a single prompt.')
    count = inputs.get('num_images_per_prompt', 1)
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError('MOSS HiRes image count must be in 1..1000.')
    target = tuple(inputs.get(axis, 1024) for axis in ('width', 'height'))
    if any(type(value) is not int or value < 32 or value > 16384 or value % 16 for value in target):
        raise ValueError('MOSS HiRes dimensions must be multiples of 16 in 32..16384.')
    base = tuple(((value + 16) // 32) * 16 for value in target)
    strength = .35 if args.hires_strength is None else args.hires_strength
    if not math.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError('MOSS HiRes strength must be finite and in (0, 1].')
    return Plan(base, target, strength, args.hires_upscaler or 'lanczos')


def refine(pipeline, images, inputs, plan, torch):
    """Re-encode pixels and start a fresh MOSS flow schedule at the chosen noise level."""
    from diffusers.utils.torch_utils import randn_tensor

    if pipeline.vae is None:
        raise ValueError('MOSS HiRes requires a matching FLUX.2 VAE.')
    device = pipeline._execution_device
    pixels = pipeline.image_processor.preprocess(images, height=plan.target_size[1], width=plan.target_size[0])
    pixels = pixels.to(device=device, dtype=pipeline.vae.dtype)
    z = pipeline.vae.encode(pixels).latent_dist.mode()
    batch, channels, height, width = z.shape
    if batch != len(images) or height % 2 or width % 2:
        raise ValueError('MOSS HiRes VAE produced an invalid batch or unpatchable latent.')
    z = z.reshape(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
    z = z.reshape(batch, channels * 4, height // 2, width // 2).float()
    expected = (len(images), pipeline.transformer.config.latent_channels,
                plan.target_size[1] // 16, plan.target_size[0] // 16)
    mean = pipeline.vae.bn.running_mean.to(z).view(1, -1, 1, 1)
    std = (pipeline.vae.bn.running_var + pipeline.vae.config.batch_norm_eps).sqrt().to(z).view(1, -1, 1, 1)
    if tuple(z.shape) != expected or mean.shape[1] != expected[1]:
        raise ValueError('MOSS HiRes requires a matching FLUX.2 VAE.')
    z = (z - mean) / std
    if not bool(torch.isfinite(z).all()):
        raise ValueError('MOSS HiRes VAE produced non-finite latents.')

    original_scheduler = pipeline.scheduler
    scheduler = type(original_scheduler).from_config(original_scheduler.config)
    steps = inputs.get('num_inference_steps', 50)
    scheduler.set_timesteps(steps, device=device)
    full_timesteps = scheduler.timesteps.clone()
    start = len(full_timesteps) - max(1, math.ceil(len(full_timesteps) * plan.strength))
    scheduler.set_begin_index(start)
    noise = randn_tensor(expected, generator=inputs.get('generator'), device=device, dtype=torch.float32)
    latents = scheduler.scale_noise(z, full_timesteps[start:start + 1].expand(batch), noise=noise)
    set_timesteps = scheduler.set_timesteps
    step = scheduler.step

    def selected_timesteps(*args, **kwargs):
        set_timesteps(*args, **kwargs)
        if not torch.equal(scheduler.timesteps, full_timesteps):
            raise RuntimeError('MOSS HiRes scheduler changed its planned refinement sequence.')
        scheduler.set_begin_index(start)
        scheduler.timesteps = scheduler.timesteps[start:]

    def finite_step(*args, **kwargs):
        output = step(*args, **kwargs)
        sample = output[0] if isinstance(output, tuple) else output.prev_sample
        if not bool(torch.isfinite(sample).all()):
            raise RuntimeError('MOSS HiRes produced non-finite denoising latents.')
        return output

    scheduler.set_timesteps = selected_timesteps
    scheduler.step = finite_step
    pipeline.scheduler = scheduler
    try:
        return pipeline(**dict(inputs, latents=latents))
    finally:
        pipeline.scheduler = original_scheduler


def run(pipeline, inputs, plan, *, expected_count, Image, torch):
    filters = {'nearest': Image.Resampling.NEAREST, 'bilinear': Image.Resampling.BILINEAR,
               'bicubic': Image.Resampling.BICUBIC, 'lanczos': Image.Resampling.LANCZOS}
    callback = inputs.get('progress_callback')
    progress = [[], []]

    def stage_callback(stage):
        def advance(step, total):
            if (type(step) is not int or type(total) is not int or total < 1
                    or step != len(progress[stage]) + 1 or step > total
                    or (progress[stage] and total != progress[stage][0][1])):
                raise RuntimeError('MOSS HiRes reported an invalid denoising sequence.')
            progress[stage].append((step, total))
            if callback:
                callback(stage * 500 + (step * 500 // total), 1000)
        return advance

    def validate(result, size, stage):
        images = getattr(result, 'images', None)
        if not isinstance(images, list) or len(images) != expected_count:
            raise RuntimeError(f'MOSS HiRes {stage} returned an unexpected image count.')
        if any(not isinstance(image, Image.Image) or image.size != size for image in images):
            raise RuntimeError(f'MOSS HiRes {stage} returned an invalid image size or type.')
        events = progress[stage]
        if not events or events[-1][0] != events[-1][1]:
            raise RuntimeError('MOSS HiRes did not complete its denoising sequence.')
        return images

    base_inputs = dict(inputs, width=plan.base_size[0], height=plan.base_size[1],
                       progress_callback=stage_callback(0))
    base = validate(pipeline(**base_inputs), plan.base_size, 0)
    upscaled = [image.resize(plan.target_size, resample=filters[plan.upscaler]) for image in base]
    refine_inputs = dict(inputs, width=plan.target_size[0], height=plan.target_size[1],
                         progress_callback=stage_callback(1))
    result = refine(pipeline, upscaled, refine_inputs, plan, torch)
    validate(result, plan.target_size, 1)
    return result, {**plan.metadata(), 'completed_passes': 1,
                    'base_steps': len(progress[0]), 'refinement_steps': len(progress[1])}
