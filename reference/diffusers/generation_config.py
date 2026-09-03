"""Strict data-only JSON configuration using the CLI's existing argument schema."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _check_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite.")
    if isinstance(value, dict):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item)


def json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        _check_finite(value)
        return value
    except (ValueError, RecursionError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


class ConfigurationArgumentParser(argparse.ArgumentParser):
    """Prepend typed file values, so explicit CLI values always take precedence."""

    def parse_values(self, values: dict[str, Any] | None = None, *, base_directory: Path | None = None):
        """The same typed schema for Python callers, with no CLI or config file required."""
        values = {} if values is None else values
        if not isinstance(values, dict):
            self.error("Generation values must be a mapping.")
        actions = {
            action.dest: action for action in self._actions
            if action.dest not in ("help", "config", "print_config")
        }
        tokens = []
        try:
            _check_finite(values)
            for key, value in values.items():
                if key not in actions:
                    raise ValueError(f"Unknown generation value: {key}")
                if value is not None:
                    tokens.extend(self._configuration_tokens(actions[key], value, base_directory or Path.cwd()))
        except ValueError as error:
            self.error(str(error))
        return self.parse_args(tokens)

    def parse_args(self, args=None, namespace=None):
        tokens = list(sys.argv[1:] if args is None else args)
        if any(token in ("-h", "--help") for token in tokens):
            return super().parse_args(tokens, namespace)
        preliminary = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        preliminary.add_argument("--config", type=Path)
        selected, _ = preliminary.parse_known_args(tokens)
        actions = {
            action.dest: action for action in self._actions
            if action.dest not in ("help", "config", "print_config")
        }
        configured: dict[str, Any] = {}
        defaults: list[str] = []
        if selected.config is not None:
            path = selected.config.expanduser().resolve()
            try:
                if path.stat().st_size > 1024 * 1024:
                    raise ValueError("Configuration exceeds the 1 MiB limit.")
                configured = json_object(path.read_text(encoding="utf-8"))
                for key, value in configured.items():
                    if key not in actions:
                        raise ValueError(f"Unknown configuration key: {key}")
                    if value is not None:
                        defaults.extend(self._configuration_tokens(actions[key], value, path.parent))
            except (OSError, UnicodeError, ValueError, argparse.ArgumentTypeError) as error:
                self.error(f"--config {path}: {error}")
        result = super().parse_args(defaults + tokens, namespace)
        result._argument_names = tuple(actions)
        cli_fields = {
            self._option_string_actions[token.split("=", 1)[0]].dest
            for token in tokens if token.split("=", 1)[0] in self._option_string_actions
        }
        result._provided = cli_fields | {key for key, value in configured.items() if value is not None}
        return result

    @staticmethod
    def _configuration_tokens(action: argparse.Action, value: Any, root: Path) -> list[str]:
        flag = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise ValueError(f"{action.dest} requires a JSON boolean.")
            return [flag if value else "--no-" + flag.removeprefix("--")]
        if action.type is json_object:
            if not isinstance(value, dict):
                raise ValueError(f"{action.dest} requires a JSON object.")
            return [flag + "=" + json.dumps(value, allow_nan=False)]
        if action.nargs is not None:
            if not isinstance(value, list) or not value:
                raise ValueError(f"{action.dest} requires a non-empty JSON array.")
            if isinstance(action.nargs, int) and len(value) != action.nargs:
                raise ValueError(f"{action.dest} requires exactly {action.nargs} values.")
            return [flag, *(ConfigurationArgumentParser._scalar(action, member, root) for member in value)]
        return [flag + "=" + ConfigurationArgumentParser._scalar(action, value, root)]

    @staticmethod
    def _scalar(action: argparse.Action, value: Any, root: Path) -> str:
        kind = action.type
        if kind is int:
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif kind is float:
            valid = isinstance(value, (float, int)) and not isinstance(value, bool)
        elif action.dest == "attention_slice_size":
            valid = isinstance(value, (str, int)) and not isinstance(value, bool)
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f"Wrong JSON value type for {action.dest}.")
        text = str(value)
        if kind is Path:
            if not text:
                raise ValueError(f"{action.dest} path must not be empty.")
            path = Path(text).expanduser()
            return str(path if path.is_absolute() else (root / path).resolve())
        if action.dest in ("model", "model_config", "vae", "lora", "latents", "embeddings",
                           "controlnet", "controlnet_config"):
            path = Path(text).expanduser()
            local = (
                path.is_absolute() or text.startswith((".", "~"))
                or (root / path).exists()
                or path.suffix.lower() in (".safetensors", ".safetensor", ".ckpt", ".bin", ".pt")
            )
            if text and local:
                return str(path if path.is_absolute() else (root / path).resolve())
        return text


def configuration_values(args: argparse.Namespace) -> dict[str, Any]:
    """A replayable flat object, excluding private runtime objects and control flags."""
    values = {name: getattr(args, name) for name in args._argument_names}
    values["model"] = args.model_selection.source
    values["revision"] = args.model_selection.requested_revision
    if args.config_selection is not None:
        values["model_config"] = args.config_selection.source
        values["model_config_revision"] = args.config_selection.requested_revision
    if args.vae_file is not None:
        values["vae"] = args.vae_file.path
    for name in ("latents", "embeddings"):
        selection = getattr(args, name + "_file", None)
        if selection is not None:
            values[name] = selection.path
    if args.lora_selection is not None and args.lora_selection.is_local:
        values["lora"] = (
            args.lora_selection.source if args.lora_weight_name is not None
            else str(Path(args.lora_selection.source) / args.lora_selection.weight_name)
        )
    if args.controlnet_selection is not None:
        values["controlnet"] = args.controlnet_selection.source
        values["controlnet_revision"] = args.controlnet_selection.requested_revision
    if args.controlnet_config_selection is not None:
        values["controlnet_config"] = args.controlnet_config_selection.source
        values["controlnet_config_revision"] = args.controlnet_config_selection.requested_revision
    # Paths and coordinate tuples are CLI values, never executable Python objects.
    return json.loads(json.dumps(values, default=lambda value: str(value.expanduser().resolve()) if isinstance(value, Path)
                                else _unsupported_value(value), allow_nan=False))


def _unsupported_value(value: Any) -> None:
    raise TypeError(f"Cannot serialize configuration value: {type(value).__name__}")
