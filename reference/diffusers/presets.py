"""Pinned model contracts shared by the iiLocalDiffusion reference tools."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from weight_files import LocalWeightFile, SAFETENSORS_SUFFIXES, resolve_weight_file


@dataclass(frozen=True)
class RuntimePolicy:
    cpu_dtype: str
    accelerator_dtype: str
    weight_variant: str | None
    dimension_multiple: int
    max_sequence_length: int | None
    passes_negative_prompt: bool
    accelerator_execution: str
    supports_attention_slicing: bool
    accelerator_vae_slicing: bool
    accelerator_vae_tiling: bool


@dataclass(frozen=True)
class PipelinePreset:
    name: str
    model_id: str
    revision: str
    pipeline_class: str
    expected_components: tuple[tuple[str, str], ...]
    width: int
    height: int
    steps: int
    guidance_scale: float
    generation_filename: str
    inspection_filename: str
    runtime: RuntimePolicy
    strict_contract: bool = True
    requires_guidance_embeds: bool = False
    requires_model_override: bool = False
    default_scheduler: str | None = None
    scheduler_defaults: tuple[tuple[str, Any], ...] = ()

    @property
    def family(self) -> str:
        """Architecture routing is independent from a checkpoint's display name."""
        return {
            "StableDiffusionPipeline": "sd15",
            "StableDiffusionControlNetPipeline": "sd15",
            "StableDiffusionXLPipeline": "sdxl-base",
            "StableDiffusionXLControlNetPipeline": "sdxl-base",
            "FluxPipeline": "flux1-schnell",
            "FluxControlNetPipeline": "flux1-schnell",
        }[self.pipeline_class]


@dataclass(frozen=True)
class ModelSelection:
    source: str
    requested_revision: str | None
    is_local: bool
    single_file: LocalWeightFile | None = None


SD15_PRESET = PipelinePreset(
    name="sd15",
    model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    revision="451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    pipeline_class="StableDiffusionPipeline",
    expected_components=(
        ("tokenizer", "CLIPTokenizer"),
        ("text_encoder", "CLIPTextModel"),
        ("unet", "UNet2DConditionModel"),
        ("vae", "AutoencoderKL"),
        ("scheduler", "PNDMScheduler"),
    ),
    width=512,
    height=512,
    steps=20,
    guidance_scale=7.5,
    generation_filename="sd15-red-cube.png",
    inspection_filename="pipeline-inspection.json",
    runtime=RuntimePolicy(
        cpu_dtype="float32",
        accelerator_dtype="float16",
        weight_variant="fp16",
        dimension_multiple=8,
        max_sequence_length=None,
        passes_negative_prompt=True,
        accelerator_execution="resident",
        supports_attention_slicing=True,
        accelerator_vae_slicing=False,
        accelerator_vae_tiling=False,
    ),
)

SDXL_BASE_PRESET = PipelinePreset(
    name="sdxl-base",
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    revision="462165984030d82259a11f4367a4eed129e94a7b",
    pipeline_class="StableDiffusionXLPipeline",
    expected_components=(
        ("tokenizer", "CLIPTokenizer"),
        ("tokenizer_2", "CLIPTokenizer"),
        ("text_encoder", "CLIPTextModel"),
        ("text_encoder_2", "CLIPTextModelWithProjection"),
        ("unet", "UNet2DConditionModel"),
        ("vae", "AutoencoderKL"),
        ("scheduler", "EulerDiscreteScheduler"),
    ),
    width=1024,
    height=1024,
    steps=20,
    guidance_scale=5.0,
    generation_filename="sdxl-base-red-cube.png",
    inspection_filename="pipeline-inspection-sdxl-base.json",
    runtime=RuntimePolicy(
        cpu_dtype="float32",
        accelerator_dtype="float16",
        weight_variant="fp16",
        dimension_multiple=8,
        max_sequence_length=None,
        passes_negative_prompt=True,
        accelerator_execution="resident",
        supports_attention_slicing=True,
        accelerator_vae_slicing=False,
        accelerator_vae_tiling=False,
    ),
)

FLUX1_SCHNELL_PRESET = PipelinePreset(
    name="flux1-schnell",
    model_id="black-forest-labs/FLUX.1-schnell",
    revision="741f7c3ce8b383c54771c7003378a50191e9efe9",
    pipeline_class="FluxPipeline",
    expected_components=(
        ("tokenizer", "CLIPTokenizer"),
        ("tokenizer_2", "T5TokenizerFast"),
        ("text_encoder", "CLIPTextModel"),
        ("text_encoder_2", "T5EncoderModel"),
        ("transformer", "FluxTransformer2DModel"),
        ("vae", "AutoencoderKL"),
        ("scheduler", "FlowMatchEulerDiscreteScheduler"),
    ),
    width=1024,
    height=1024,
    steps=4,
    guidance_scale=0.0,
    generation_filename="flux1-schnell-red-cube.png",
    inspection_filename="pipeline-inspection-flux1-schnell.json",
    runtime=RuntimePolicy(
        cpu_dtype="bfloat16",
        accelerator_dtype="bfloat16",
        weight_variant=None,
        dimension_multiple=16,
        max_sequence_length=256,
        passes_negative_prompt=False,
        accelerator_execution="sequential-cpu-offload",
        supports_attention_slicing=False,
        accelerator_vae_slicing=True,
        accelerator_vae_tiling=True,
    ),
)

# Family presets retain model training configuration instead of requiring every
# scheduler/precision policy to equal the original reference snapshot.
SD15_COMPATIBLE_PRESET = replace(
    SD15_PRESET, name="sd15-compatible", strict_contract=False,
    generation_filename="sd15-compatible-red-cube.png",
    inspection_filename="pipeline-inspection-sd15-compatible.json",
    runtime=replace(SD15_PRESET.runtime, weight_variant=None),
)
FLUX1_SCHNELL_COMPATIBLE_PRESET = replace(
    FLUX1_SCHNELL_PRESET, name="flux1-schnell-compatible", strict_contract=False,
    generation_filename="flux1-schnell-compatible-red-cube.png",
    inspection_filename="pipeline-inspection-flux1-schnell-compatible.json",
)
SDXL_PRESET = replace(
    SDXL_BASE_PRESET, name="sdxl", strict_contract=False,
    generation_filename="sdxl-red-cube.png",
    inspection_filename="pipeline-inspection-sdxl.json",
    runtime=replace(SDXL_BASE_PRESET.runtime, weight_variant=None),
)
ILLUSTRIOUS_PRESET = replace(
    SDXL_PRESET, name="illustrious",
    model_id="OnomaAIResearch/Illustrious-xl-early-release-v0",
    revision="dca0dac303e6dc4b0c31d8001bc685b89b5d0204",
    generation_filename="illustrious-red-cube.png",
    inspection_filename="pipeline-inspection-illustrious.json",
)
NOOBAI_PRESET = replace(
    SDXL_PRESET, name="noobai", model_id="Laxhar/NoobAI-XL-1.0",
    revision="70dee4d903b83cc6d1e8d12e65051cfdbcf54ab3",
    generation_filename="noobai-red-cube.png",
    inspection_filename="pipeline-inspection-noobai.json",
)
NOOBAI_V_PRED_PRESET = replace(
    NOOBAI_PRESET, name="noobai-v-pred", model_id="Laxhar/noobai-XL-Vpred-1.0",
    revision="66aa55e3469c27c29a89813cd35dd95fb7485fa1",
    steps=28, default_scheduler="EulerDiscreteScheduler",
    scheduler_defaults=(("prediction_type", "v_prediction"),
                        ("rescale_betas_zero_snr", True)),
    generation_filename="noobai-v-pred-red-cube.png",
    inspection_filename="pipeline-inspection-noobai-v-pred.json",
)
# Pony's official release is a checkpoint, not a Diffusers package. Its preset
# supplies SDXL conversion configuration only; generation requires --model.
PONY_PRESET = replace(
    SDXL_PRESET, name="pony", requires_model_override=True,
    generation_filename="pony-red-cube.png",
    inspection_filename="pipeline-inspection-pony.json",
)
FLUX1_DEV_PRESET = replace(
    FLUX1_SCHNELL_PRESET, name="flux1-dev",
    model_id="black-forest-labs/FLUX.1-dev",
    revision="3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
    steps=28, guidance_scale=3.5, strict_contract=False,
    requires_guidance_embeds=True,
    generation_filename="flux1-dev-red-cube.png",
    inspection_filename="pipeline-inspection-flux1-dev.json",
    runtime=replace(FLUX1_SCHNELL_PRESET.runtime, max_sequence_length=512),
)
FLUX1_KREA_DEV_PRESET = replace(
    FLUX1_DEV_PRESET, name="flux1-krea-dev",
    model_id="black-forest-labs/FLUX.1-Krea-dev",
    revision="8162a9c7b05a641be098422bf2fcf335615c2f28",
    guidance_scale=4.5,
    generation_filename="flux1-krea-dev-red-cube.png",
    inspection_filename="pipeline-inspection-flux1-krea-dev.json",
)
DEFAULT_PRESET_NAME = SD15_PRESET.name
PRESETS: Mapping[str, PipelinePreset] = MappingProxyType({
    preset.name: preset for preset in (
        SD15_PRESET, SDXL_BASE_PRESET, FLUX1_SCHNELL_PRESET,
        SD15_COMPATIBLE_PRESET, FLUX1_SCHNELL_COMPATIBLE_PRESET,
        SDXL_PRESET, ILLUSTRIOUS_PRESET, NOOBAI_PRESET,
        NOOBAI_V_PRED_PRESET, PONY_PRESET, FLUX1_DEV_PRESET, FLUX1_KREA_DEV_PRESET,
    )
})


# Built-in Diffusers scheduler declarations accepted before single-file loading.
# Custom module paths/classes never enter this allowlist.
DIFFUSION_SCHEDULERS = frozenset({
    "DDIMScheduler", "DDPMScheduler", "PNDMScheduler", "LMSDiscreteScheduler",
    "EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler", "HeunDiscreteScheduler",
    "KDPM2DiscreteScheduler", "KDPM2AncestralDiscreteScheduler",
    "DPMSolverSinglestepScheduler", "DPMSolverMultistepScheduler",
    "DEISMultistepScheduler", "UniPCMultistepScheduler", "DPMSolverSDEScheduler",
    "DDIMInverseScheduler", "LCMScheduler", "TCDScheduler", "EDMEulerScheduler",
    "EDMDPMSolverMultistepScheduler", "IPNDMScheduler", "SASolverScheduler",
    "CosineDPMSolverMultistepScheduler",
})
FLOW_SCHEDULERS = frozenset({
    "FlowMatchEulerDiscreteScheduler", "FlowMatchHeunDiscreteScheduler",
    "FlowMatchLCMScheduler", "FlowMatchDPMSolverMultistepScheduler",
})


def compatible_scheduler_class(preset: PipelinePreset, class_name: str | None) -> bool:
    if not isinstance(class_name, str):
        return False
    if preset.strict_contract:
        return class_name == dict(preset.expected_components)["scheduler"]
    names = FLOW_SCHEDULERS if preset.family == "flux1-schnell" else DIFFUSION_SCHEDULERS
    return class_name in names


def resolve_model_selection(
    preset: PipelinePreset,
    model_override: str | None,
    revision_override: str | None,
    *,
    allow_single_file: bool = False,
    model_argument: str = "--model",
    revision_argument: str = "--revision",
) -> ModelSelection:
    model = preset.model_id if model_override is None else model_override
    if not model:
        raise ValueError(f"{model_argument} must not be empty.")
    local_model = Path(model).expanduser()
    if local_model.exists():
        if revision_override is not None:
            raise ValueError(f"A local model does not accept a commit {revision_argument}.")
        if local_model.is_file() and allow_single_file:
            weight = resolve_weight_file(model, model_argument)
            return ModelSelection(weight.path, None, True, weight)
        if not local_model.is_dir():
            raise ValueError(f"Local model path is not a directory: {local_model}")
        return ModelSelection(str(local_model.resolve()), None, True)

    if (
        local_model.is_absolute()
        or model.startswith((".", "~"))
        or local_model.suffix.lower() in (*SAFETENSORS_SUFFIXES, ".ckpt", ".bin", ".pt")
    ):
        raise ValueError(f"Local model path does not exist: {local_model}")

    revision = revision_override
    if revision is None and model == preset.model_id:
        revision = preset.revision
    if revision is None:
        raise ValueError(
            f"A 40-character commit {revision_argument} is required for a remote "
            f"{model_argument} override."
        )
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Remote model revision must be a 40-character lowercase commit SHA.")
    return ModelSelection(model, revision, False)


def _require_configuration_value(
    component: Any,
    member: str,
    expected: Any,
    description: str,
) -> None:
    configuration = component.config if hasattr(component, "config") else component
    actual = getattr(configuration, member, None)
    if isinstance(expected, tuple) and isinstance(actual, (list, tuple)):
        matches = tuple(actual) == expected
    else:
        matches = actual == expected
    if not matches:
        raise RuntimeError(
            f"Unexpected {description} {member}: expected {expected}, found {actual}"
        )


def validate_pipeline_contract(pipeline: Any, preset: PipelinePreset) -> None:
    actual_pipeline_class = type(pipeline).__name__
    if actual_pipeline_class != preset.pipeline_class:
        raise RuntimeError(
            f"Unexpected pipeline class: expected {preset.pipeline_class}, "
            f"found {actual_pipeline_class}"
        )
    for name, expected_class in preset.expected_components:
        component = pipeline.components.get(name)
        actual_class = None if component is None else type(component).__name__
        class_matches = actual_class == expected_class
        if expected_class == "T5TokenizerFast" and actual_class == "T5Tokenizer":
            class_matches = True
        if name == "scheduler":
            class_matches = compatible_scheduler_class(preset, actual_class)
        if not class_matches:
            raise RuntimeError(
                f"Unexpected {name} class: expected {expected_class}, found {actual_class}"
            )

    validate_vae_contract(pipeline.vae, preset)

    if preset.family == "flux1-schnell":
        _validate_flux1_schnell_contract(pipeline, preset)
        return
    if preset.family == "sd15":
        if not preset.strict_contract:
            _require_configuration_value(pipeline.tokenizer, "model_max_length", 77, "tokenizer")
            _require_configuration_value(pipeline.text_encoder, "hidden_size", 768, "text_encoder")
            _require_configuration_value(pipeline.text_encoder, "max_position_embeddings", 77, "text_encoder")
            for member, expected in (("in_channels", 4), ("out_channels", 4),
                                     ("cross_attention_dim", 768)):
                _require_configuration_value(pipeline.unet, member, expected, "UNet")
            _validate_diffusion_scheduler_contract(pipeline.scheduler)
        return

    if preset.strict_contract:
        _require_configuration_value(
            pipeline.config, "force_zeros_for_empty_prompt", True, "pipeline"
        )
    _require_configuration_value(pipeline.tokenizer, "model_max_length", 77, "tokenizer")
    _require_configuration_value(pipeline.tokenizer_2, "model_max_length", 77, "tokenizer_2")
    _require_configuration_value(pipeline.text_encoder, "hidden_size", 768, "text_encoder")
    _require_configuration_value(
        pipeline.text_encoder, "max_position_embeddings", 77, "text_encoder"
    )
    _require_configuration_value(pipeline.text_encoder, "projection_dim", 768, "text_encoder")
    _require_configuration_value(pipeline.text_encoder_2, "hidden_size", 1280, "text_encoder_2")
    _require_configuration_value(
        pipeline.text_encoder_2, "max_position_embeddings", 77, "text_encoder_2"
    )
    _require_configuration_value(
        pipeline.text_encoder_2, "projection_dim", 1280, "text_encoder_2"
    )
    for member, expected in (
        ("in_channels", 4),
        ("out_channels", 4),
        ("sample_size", 128),
        ("cross_attention_dim", 2048),
        ("addition_embed_type", "text_time"),
        ("addition_time_embed_dim", 256),
        ("projection_class_embeddings_input_dim", 2816),
        ("use_linear_projection", True),
    ):
        if preset.strict_contract or member != "sample_size":
            _require_configuration_value(pipeline.unet, member, expected, "UNet")
    for member, expected in (
        ("in_channels", 3),
        ("out_channels", 3),
        ("latent_channels", 4),
        ("sample_size", 1024),
        ("scaling_factor", 0.13025),
        ("force_upcast", True),
    ):
        if preset.strict_contract or member not in ("sample_size", "force_upcast"):
            _require_configuration_value(pipeline.vae, member, expected, "VAE")
    if not preset.strict_contract:
        _validate_diffusion_scheduler_contract(pipeline.scheduler)
        return
    for member, expected in (
        ("num_train_timesteps", 1000),
        ("beta_start", 0.00085),
        ("beta_end", 0.012),
        ("beta_schedule", "scaled_linear"),
        ("prediction_type", "epsilon"),
        ("timestep_spacing", "leading"),
    ):
        _require_configuration_value(pipeline.scheduler, member, expected, "scheduler")


def validate_vae_contract(vae: Any, preset: PipelinePreset) -> None:
    latent_channels, scaling_factor = {
        "sd15": (4, 0.18215),
        "sdxl-base": (4, 0.13025),
        "flux1-schnell": (16, 0.3611),
    }[preset.family]
    for member, expected in (
        ("in_channels", 3),
        ("out_channels", 3),
        ("latent_channels", latent_channels),
        ("scaling_factor", scaling_factor),
    ):
        _require_configuration_value(vae, member, expected, "VAE")
    channels = getattr(vae.config, "block_out_channels", ())
    if len(channels) != 4:
        raise RuntimeError("VAE must use four blocks for 8x spatial downsampling.")
    shift = getattr(vae.config, "shift_factor", None)
    if preset.family == "flux1-schnell":
        _require_configuration_value(vae, "shift_factor", 0.1159, "VAE")
    elif shift not in (None, 0.0):
        raise RuntimeError(f"Unexpected VAE shift_factor for {preset.name}: {shift}")


def _validate_flux1_schnell_contract(pipeline: Any, preset: PipelinePreset) -> None:
    _require_configuration_value(pipeline.tokenizer, "model_max_length", 77, "tokenizer")
    _require_configuration_value(pipeline.tokenizer_2, "model_max_length", 512, "tokenizer_2")
    for member, expected in (
        ("hidden_size", 768),
        ("max_position_embeddings", 77),
        ("projection_dim", 768),
        ("intermediate_size", 3072),
        ("num_attention_heads", 12),
        ("num_hidden_layers", 12),
        ("vocab_size", 49408),
    ):
        _require_configuration_value(pipeline.text_encoder, member, expected, "text_encoder")
    for member, expected in (
        ("d_model", 4096),
        ("d_ff", 10240),
        ("d_kv", 64),
        ("num_heads", 64),
        ("num_layers", 24),
        ("vocab_size", 32128),
        ("feed_forward_proj", "gated-gelu"),
    ):
        _require_configuration_value(pipeline.text_encoder_2, member, expected, "text_encoder_2")
    for member, expected in (
        ("patch_size", 1),
        ("in_channels", 64),
        ("num_layers", 19),
        ("num_single_layers", 38),
        ("attention_head_dim", 128),
        ("num_attention_heads", 24),
        ("joint_attention_dim", 4096),
        ("pooled_projection_dim", 768),
        ("guidance_embeds", preset.requires_guidance_embeds),
        ("axes_dims_rope", (16, 56, 56)),
    ):
        _require_configuration_value(pipeline.transformer, member, expected, "transformer")
    configured_out_channels = getattr(pipeline.transformer.config, "out_channels", None)
    if configured_out_channels not in (None, 64):
        raise RuntimeError(
            "Unexpected transformer out_channels: expected null or 64, "
            f"found {configured_out_channels}"
        )
    effective_out_channels = getattr(
        pipeline.transformer,
        "out_channels",
        pipeline.transformer.config.in_channels
        if configured_out_channels is None
        else configured_out_channels,
    )
    if effective_out_channels != 64:
        raise RuntimeError(
            "Unexpected transformer effective out_channels: expected 64, "
            f"found {effective_out_channels}"
        )
    for member, expected in (
        ("in_channels", 3),
        ("out_channels", 3),
        ("latent_channels", 16),
        ("sample_size", 1024),
        ("layers_per_block", 2),
        ("norm_num_groups", 32),
        ("scaling_factor", 0.3611),
        ("shift_factor", 0.1159),
        ("force_upcast", True),
        ("use_quant_conv", False),
        ("use_post_quant_conv", False),
        ("block_out_channels", (128, 256, 512, 512)),
        (
            "down_block_types",
            (
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
        ),
        (
            "up_block_types",
            (
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
        ),
        ("mid_block_add_attention", True),
    ):
        _require_configuration_value(pipeline.vae, member, expected, "VAE")
    if not preset.strict_contract:
        return
    for member, expected in (
        ("num_train_timesteps", 1000),
        ("base_image_seq_len", 256),
        ("max_image_seq_len", 4096),
        ("base_shift", 0.5),
        ("max_shift", 1.15),
        ("shift", 1.0),
        ("use_dynamic_shifting", False),
    ):
        _require_configuration_value(pipeline.scheduler, member, expected, "scheduler")


def _validate_diffusion_scheduler_contract(scheduler: Any) -> None:
    configuration = scheduler.config
    prediction = getattr(configuration, "prediction_type", None)
    if prediction is None and isinstance(configuration, Mapping):
        prediction = configuration.get("prediction_type")
    if prediction not in ("epsilon", "v_prediction", "sample", "original_sample"):
        raise RuntimeError(f"Unsupported diffusion scheduler prediction_type: {prediction}")


def validate_preset_definition(preset: PipelinePreset) -> None:
    supported_dtypes = {"bfloat16", "float16", "float32"}
    for name, dtype in (
        ("cpu_dtype", preset.runtime.cpu_dtype),
        ("accelerator_dtype", preset.runtime.accelerator_dtype),
    ):
        if dtype not in supported_dtypes:
            raise RuntimeError(f"Preset {preset.name} has unsupported {name}: {dtype}")
    if preset.runtime.accelerator_execution not in (
        "resident",
        "sequential-cpu-offload",
    ):
        raise RuntimeError(
            f"Preset {preset.name} has unsupported accelerator execution policy: "
            f"{preset.runtime.accelerator_execution}"
        )
    if preset.runtime.dimension_multiple <= 0:
        raise RuntimeError(f"Preset {preset.name} has an invalid dimension multiple")
    if (
        preset.runtime.max_sequence_length is not None
        and preset.runtime.max_sequence_length <= 0
    ):
        raise RuntimeError(f"Preset {preset.name} has an invalid maximum sequence length")


for _preset in PRESETS.values():
    validate_preset_definition(_preset)
    if re.fullmatch(r"[0-9a-f]{40}", _preset.revision) is None:
        raise RuntimeError(f"Preset {_preset.name} does not use an immutable model revision")
