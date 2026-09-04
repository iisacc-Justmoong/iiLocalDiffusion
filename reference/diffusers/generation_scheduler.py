"""Configure supported Diffusers schedulers without implementing sampling math."""

from __future__ import annotations

import inspect
import json
from typing import Any

from presets import PRESETS, PipelinePreset


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Scheduler configuration is not JSON data: {type(value).__name__}")


def configure_scheduler(pipeline: Any, args: Any, *, inherit_current: bool = False) -> dict[str, Any]:
    """Apply base settings, or refine an already configured scheduler.

    Refinement inherits the actual class/configuration, including previously
    applied prediction settings. Only its scheduler selection and constructor
    overrides are new inputs; replaying base preset or prediction hints would
    overwrite those inherited settings and conflict with refinement overrides.
    The caller remains responsible for creating fresh scheduler runtime state.
    """
    current = pipeline.scheduler
    preset = None if inherit_current else PRESETS.get(getattr(args, "preset", ""))
    requested = getattr(args, "scheduler", "auto")
    effective_request = requested
    overrides = dict(preset.scheduler_defaults) if preset is not None else {}
    if requested == "auto" and preset is not None and preset.default_scheduler is not None:
        effective_request = preset.default_scheduler
    overrides.update(getattr(args, "scheduler_config", {}))
    prediction = None if inherit_current else getattr(args, "prediction_type", None)
    if prediction not in (None, "auto"):
        if prediction not in ("epsilon", "v_prediction", "sample"):
            raise ValueError("--prediction-type must be auto, epsilon, v_prediction, or sample.")
        explicit_config = getattr(args, "scheduler_config", {})
        if "prediction_type" in explicit_config and explicit_config["prediction_type"] != prediction:
            raise ValueError("--prediction-type conflicts with --scheduler-config prediction_type.")
        overrides["prediction_type"] = prediction
    compatible = {type(current).__name__: type(current)}
    compatible.update({cls.__name__: cls for cls in current.compatibles})
    selected = type(current) if effective_request == "auto" else compatible.get(effective_request)
    if selected is None:
        raise ValueError(f"Scheduler {requested!r} is incompatible; choose auto or {', '.join(sorted(compatible))}.")
    constructor = inspect.signature(selected.__init__).parameters
    for name, value in overrides.items():
        if name.startswith("_") or name not in constructor or name in ("self", "args", "kwargs"):
            raise ValueError(f"Unknown or protected scheduler configuration value: {name}")
        default = constructor[name].default
        if isinstance(default, bool) and not isinstance(value, bool):
            raise ValueError(f"Scheduler {name} requires a boolean.")
        if isinstance(default, int) and not isinstance(default, bool) and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ValueError(f"Scheduler {name} requires an integer.")
        if isinstance(default, float) and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            raise ValueError(f"Scheduler {name} requires a number.")
        if isinstance(default, str) and not isinstance(value, str):
            raise ValueError(f"Scheduler {name} requires a string.")
    if selected is not type(current) or overrides:
        replacement = selected.from_config(current.config, **overrides)
        if type(replacement) is not selected:
            raise RuntimeError("Scheduler construction returned an unexpected class.")
        pipeline.scheduler = replacement
    return {
        "requested_class": requested,
        "inherit_current": inherit_current,
        "class": type(pipeline.scheduler).__name__,
        "preset_defaults": dict(preset.scheduler_defaults) if preset is not None else {},
        "overrides": overrides,
        "config": json.loads(json.dumps(dict(pipeline.scheduler.config),
                                      default=_json_default, allow_nan=False)),
    }


def validate_scheduler_values(pipeline: Any, args: Any) -> None:
    scheduler = pipeline.scheduler
    parameters = inspect.signature(scheduler.set_timesteps).parameters
    for name in ("timesteps", "sigmas"):
        values = getattr(args, name, None)
        if values is not None and name not in parameters:
            raise ValueError(f"{type(scheduler).__name__} does not support --{name}.")
    if getattr(args, "eta", 0) != 0 and "eta" not in inspect.signature(scheduler.step).parameters:
        raise ValueError(f"{type(scheduler).__name__} does not use --eta; select a compatible DDIM scheduler.")
    timesteps = getattr(args, "timesteps", None)
    training_steps = scheduler.config.get("num_train_timesteps")
    if timesteps and training_steps is not None and any(value >= training_steps for value in timesteps):
        raise ValueError("Custom timesteps must be less than the scheduler's num_train_timesteps.")


def validate_clip_skip(pipeline: Any, preset: PipelinePreset, args: Any) -> None:
    skip = getattr(args, "clip_skip", None)
    if skip is None:
        return
    names = ("text_encoder", "text_encoder_2") if preset.family == "sdxl-base" else ("text_encoder",)
    for name in names:
        encoder = getattr(pipeline, name, None)
        layers = getattr(getattr(encoder, "config", None), "num_hidden_layers", None)
        limit = layers - (1 if preset.family == "sdxl-base" else 0) if isinstance(layers, int) else None
        if limit is None or skip > limit:
            raise ValueError(f"--clip-skip exceeds {name}'s supported hidden-state range.")
