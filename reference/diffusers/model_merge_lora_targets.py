"""Resolve adapter module names to actual checkpoint tensors without guessing suffixes."""

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class LoraTarget:
    address: tuple[str, str]
    rows: tuple[int, int] | None = None
    transpose: bool = False


def _module(key):
    return key.removesuffix(".weight")


def normalize_module(name):
    name = name.removeprefix("base_model.model.").replace(".processor.", ".")
    return name


def lora_target_aliases(layout, readers):
    aliases = {}

    def add(name, target):
        aliases.setdefault(normalize_module(name), set()).add(target)

    def component_aliases(component, key, target):
        module = _module(key)
        add(f"{component}.{module}", target)
        add(module, target)
        prefixes = {"unet": ("lora_unet_",), "transformer": ("lora_transformer_",),
                    "text_encoder": ("lora_te_", "lora_te1_"), "text_encoder_2": ("lora_te2_",)}
        names = [module]
        if component.startswith("text_encoder"):
            # Transformers 5.6 flattened CLIPTextModel; older adapters retain text_model.
            names.append(module.removeprefix("text_model.") if module.startswith("text_model.")
                         else "text_model." + module)
        for name in names:
            add(name, target)
            add(f"{component}.{name}", target)
            for prefix in prefixes.get(component, ()):
                add(prefix + name.replace(".", "_"), target)

    single_keys = {}
    for address in layout:
        component, key = address
        target = LoraTarget(address)
        add(_module(key), target)
        if component != ".":
            component_aliases(component, key, target)
        else:
            single_keys[key] = target
            for prefix in ("unet", "transformer", "text_encoder", "text_encoder_2"):
                if key.startswith(prefix + "."):
                    component_aliases(prefix, key[len(prefix) + 1:], target)
            for prefix in ("cond_stage_model.transformer.", "conditioner.embedders.0.transformer."):
                if key.startswith(prefix):
                    component_aliases("text_encoder", key[len(prefix):], target)
            if key.startswith("model.diffusion_model."):
                component_aliases("unet", key.removeprefix("model.diffusion_model."), target)

    if any(key.startswith("model.diffusion_model.") for key in single_keys):
        # The pinned Diffusers converter only moves UNet values. Passing original
        # key strings builds an inverse map without loading an entire UNet/pipeline.
        from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
        mapped = convert_ldm_unet_checkpoint({key: key for key in single_keys}, {"layers_per_block": 2})
        for key, original in mapped.items():
            if original in single_keys:
                component_aliases("unet", key, single_keys[original])

    openclip_prefixes = {"cond_stage_model.model.": "text_encoder",
                         "conditioner.embedders.1.model.": "text_encoder_2"}
    if any(key.startswith(tuple(openclip_prefixes)) for key in single_keys):
        from diffusers.loaders.single_file_utils import DIFFUSERS_TO_LDM_MAPPING
        mapping = DIFFUSERS_TO_LDM_MAPPING["openclip"]
        for prefix, component in openclip_prefixes.items():
            for key, target in single_keys.items():
                if not key.startswith(prefix):
                    continue
                local = key[len(prefix):]
                for new, old in mapping["layers"].items():
                    if local == old:
                        component_aliases(component, new, LoraTarget(target.address, transpose=old == "text_projection"))
                if not local.startswith("transformer."):
                    continue
                mapped = local.removeprefix("transformer.")
                for new, old in mapping["transformer"].items():
                    mapped = mapped.replace(old, new)
                if mapped.endswith(".in_proj_weight"):
                    shape = readers[layout[target.address]].get_slice(key).get_shape()
                    if len(shape) != 2 or shape[0] % 3:
                        raise ValueError(f"Invalid packed OpenCLIP QKV weight: {key}")
                    width = shape[0] // 3
                    for i, projection in enumerate(("q_proj", "k_proj", "v_proj")):
                        component_aliases(component, mapped.removesuffix(".in_proj_weight") + f".{projection}.weight",
                                          LoraTarget(target.address, (i * width, (i + 1) * width)))
                else:
                    component_aliases(component, mapped, target)
    return aliases


def resolve_lora_target(module, aliases, base_model):
    found = aliases.get(normalize_module(module), set())
    if not found and module.startswith("lora_unet_"):
        # Resolve SGM/Kohya block spellings using the same pinned converters as
        # Diffusers. Convert one name at a time so mixed spellings cannot cause
        # the upstream batch converter to drop another adapter's keys.
        from diffusers.loaders.lora_conversion_utils import (_convert_unet_lora_key,
                                                            _maybe_map_sgm_blocks_to_diffusers)
        layers = 2
        if "unet/config.json" in base_model.assets:
            config = json.loads(Path(base_model.assets["unet/config.json"].resolved_file).read_text(encoding="utf-8"))
            layers = config.get("layers_per_block", 2)
        if type(layers) is not int or layers < 1:
            raise ValueError("SGM LoRA mapping requires a uniform integer UNet layers_per_block.")
        try:
            mapped = _maybe_map_sgm_blocks_to_diffusers(
                {module + ".lora_down.weight": None}, SimpleNamespace(layers_per_block=layers))
            key = _convert_unet_lora_key(next(iter(mapped)))
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(f"Unsupported SGM/Kohya LoRA target: {module}") from error
        if key.endswith("_lora.down.weight"):
            key = key.removesuffix("_lora.down.weight")
            if key.endswith(".to_out"):
                key += ".0"
        else:
            key = key.removesuffix(".lora.down.weight")
        converted = "unet." + key
        found = aliases.get(normalize_module(converted), set())
    if len(found) != 1:
        raise ValueError(f"LoRA target is {'ambiguous' if found else 'unmatched'} in the base checkpoint: {module}")
    return next(iter(found))
