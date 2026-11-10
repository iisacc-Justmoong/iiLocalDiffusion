"""Shared local color-map regional prompting contracts."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from weight_files import file_sha256, file_signature


@dataclass(frozen=True)
class Region:
    color: tuple
    prompt: str
    label: str


@dataclass(frozen=True)
class Selection:
    path: str
    signature: tuple
    sha256: str
    map_path: str
    regions: tuple
    isolate_until: float

    def metadata(self):
        return {'path': self.path, 'sha256': self.sha256, 'map': self.map_path,
                'regions': [{'color': '#%02x%02x%02x' % r.color, 'prompt': r.prompt,
                             'label': r.label} for r in self.regions],
                'isolate_until': self.isolate_until, 'layout': 'native'}


def resolve(args, pipeline_name, *, option, expected, family):
    if getattr(args, option) is None:
        return None
    if pipeline_name != expected or args.offload != 'none':
        raise ValueError(f'{family} regions require {expected} and --offload none.')
    if not isinstance(args.inputs.get('prompt'), str) or any(k in args.inputs for k in ('prompt_embeds', 'prompt_attention_mask', 'prompt_embeds_mask')):
        raise ValueError('Regional prompting requires a single string prompt without supplied prompt embeddings.')
    path = Path(getattr(args, option)).expanduser().absolute()
    signature = file_signature(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Regional manifest exceeds 1 MiB.')
    digest = file_sha256(path)
    try:
        data = json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError('Invalid regional manifest.') from error
    if file_signature(path) != signature:
        raise RuntimeError('Regional manifest changed while reading.')
    if not isinstance(data, dict) or set(data) - {'map', 'regions', 'isolate_until', 'layout', 'control'}:
        raise ValueError('Regional manifest contains unsupported fields.')
    if data.get('layout', 'native') != 'native':
        raise ValueError('Only native regional layout is supported.')
    isolation = data.get('isolate_until', 0.0)
    if isinstance(isolation, bool) or not isinstance(isolation, (int, float)) or not math.isfinite(isolation) or not 0 <= isolation <= 1:
        raise ValueError('isolate_until must be a finite fraction in [0,1].')
    def local(value):
        if not isinstance(value, str) or not value:
            raise ValueError('Region resources must have local file paths.')
        p = Path(value).expanduser()
        p = (p if p.is_absolute() else path.parent / p).absolute()
        if not p.is_file() or not p.stat().st_size:
            raise ValueError(f'Region resource is missing or empty: {p}')
        return str(p)
    map_path = local(data.get('map'))
    entries = data.get('regions')
    if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
        raise ValueError('Regional prompting requires 1 to 32 color/prompt entries.')
    regions = []
    for item in entries:
        if not isinstance(item, dict) or set(item) - {'color', 'prompt', 'label', 'loras'}:
            raise ValueError('Unsupported regional fields.')
        color, prompt, label = item.get('color'), item.get('prompt'), item.get('label', '')
        if (not isinstance(color, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', color)
                or not isinstance(prompt, str) or not prompt.strip() or not isinstance(label, str)
                or item.get('loras', []) != []):
            raise ValueError('Each region needs an RGB hex color, prompt and no per-region LoRA.')
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        if any(r.color == rgb for r in regions):
            raise ValueError('Regional colors must be unique.')
        regions.append(Region(rgb, prompt, label))
    control = data.get('control', {})
    if not isinstance(control, dict) or set(control) - {'adapter', 'color_map', 'depth', 'strength', 'color_end', 'depth_end'}:
        raise ValueError('Unsupported regional control fields.')
    for key in ('strength', 'color_end', 'depth_end'):
        if key in control:
            value = control[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Regional control {key} must be a finite number.')
    # Explicit CLI options override the corresponding manifest settings.
    for key, name in (('adapter', 'controlnet'), ('color_map', 'control_color_image'), ('depth', 'control_image')):
        if key in control and getattr(args, name) is None:
            setattr(args, name, local(control[key]))
    for key, name in (('strength', 'controlnet_scale'), ('color_end', 'control_color_end'), ('depth_end', 'control_depth_end')):
        if key in control and getattr(args, name) is None:
            setattr(args, name, control[key])
    return Selection(str(path), signature, digest, map_path, tuple(regions), float(isolation))


def verify(selection):
    if file_signature(Path(selection.path)) != selection.signature:
        raise RuntimeError('Regional manifest changed during generation.')


def image_groups(image, selection, width, height, device, torch, np, *, map_fit='stretch'):
    from PIL import Image, ImageOps
    if any(type(v) is not int or not 16 <= v <= 8192 or v % 16 for v in (width, height)):
        raise ValueError('Regional dimensions must be multiples of 16 in [16,8192].')
    # Color-control adapters first resize the map to the output dimensions.
    # Nearest-neighbour resizes are not generally associative: sampling the
    # original directly at latent size can select a different region ID.
    if map_fit == 'center':
        canvas = ImageOps.fit(image.convert('RGB'), (width, height), method=Image.Resampling.NEAREST)
    elif map_fit == 'stretch':
        canvas = image.convert('RGB').resize((width, height), Image.Resampling.NEAREST)
    else:
        raise ValueError('Unknown regional map geometry.')
    pixels = np.asarray(canvas.resize((width // 16, height // 16), Image.Resampling.NEAREST))
    groups = torch.zeros((height // 16, width // 16), dtype=torch.long, device=device)
    for index, region in enumerate(selection.regions, 1):
        matches = torch.from_numpy(np.all(pixels == region.color, axis=2).copy()).to(device)
        if not bool(matches.any()):
            raise ValueError(f'Regional color {region.color} is absent at latent resolution.')
        groups[matches] = index
    return groups.flatten()


class Attention:
    def __init__(self, image_groups, text_groups, isolate_until, torch, *, order='image-first'):
        self.image_groups, self.text_groups = image_groups, text_groups
        self.isolate_until, self.torch, self.order = isolate_until, torch, order
        if order not in ('image-first', 'text-first'):
            raise ValueError('Unknown regional token order.')
        self.cache = {}

    def masks(self, step, total):
        isolated = step / total < self.isolate_until
        if isolated not in self.cache:
            image, text, torch = self.image_groups, self.text_groups, self.torch
            count = image.numel() + text.numel()
            if count * count > 80_000_000:
                raise ValueError('Regionalal attention exceeds the 80 million element limit.')
            tt = text[:, None] == text[None, :]
            ii = (image[:, None] == image[None, :]) if isolated else torch.ones(
                (image.numel(), image.numel()), device=image.device, dtype=torch.bool)
            it = (text[None, :] == 0) | (image[:, None] == text[None, :])
            ti = (text[:, None] == 0) | (text[:, None] == image[None, :])
            joint = (torch.cat((torch.cat((ii, it), 1), torch.cat((ti, tt), 1)), 0) if self.order == 'image-first'
                     else torch.cat((torch.cat((tt, ti), 1), torch.cat((it, ii), 1)), 0))
            self.cache[isolated] = (tt, joint)
        return self.cache[isolated]


