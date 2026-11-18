"""Generate local checkpoints with the SDK's Diffusers/PyTorch or native image engine."""

from _file_provider import provider

import argparse
import json
import os
from pathlib import Path
import tempfile
import uuid

import generate
from checkpoint_config import inspect_checkpoint
from downloaded_model import inspect_downloaded_model
from generation_config import configuration_values, json_object
from inference_session import is_preparing
from weight_files import file_sha256, file_signature, verify_weight_file
from presets import PRESETS
from resource_paths import OUTPUT_DIRECTORY
from vae_defaults import pipeline_vae_family, verify_vae_selection

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIRECTORY = OUTPUT_DIRECTORY / "standalone-image"


def build_parser():
    parser = generate.build_parser()
    parser.description = __doc__
    parser.add_argument("--output-dir", type=Path, help="Empty output directory for images and generation.json")
    parser.add_argument("--work-dir", type=Path, help="Empty caller-owned directory for the resolved request")
    parser.add_argument("--model-info", type=Path, help="Optional Civitai metadata for the local checkpoint")
    parser.add_argument("--validate-only", action="store_true", help="Validate the offline model/configuration request without inference")
    parser.add_argument("--engine", choices=("auto", "native", "diffusers"), default="auto", help="Local execution engine; auto selects by tensor architecture")
    parser.add_argument("--components", type=json_object, default={}, help="Native split weights: vae, clip_l, clip_g, t5xxl, llm as local paths")
    parser.add_argument("--embedded-guidance", type=float, help="Distilled guidance for native flow models (separate from CFG)")
    from krea2_contract import add_options
    add_options(parser)
    parser.add_argument("--native-sampler", choices=("auto", "euler", "heun"), default="auto")
    # Accept previous desktop callers while removing the server cold-start requirement.
    parser.add_argument("--startup-timeout", type=float, help=argparse.SUPPRESS)
    return parser


def resolve_arguments(args):
    from image_package import current as current_package, verify_source as verify_package_source
    args.image_package = current_package()
    verify_package_source(args.image_package)
    if args.image_package:
        args.original_model_path = args.image_package['source']
    if not args.model or not Path(args.model).expanduser().is_file():
        raise ValueError("The standalone checkpoint backend requires a local --model-path file.")
    if args.output is not None and args.output_dir is not None:
        raise ValueError("Choose --output or --output-dir, not both.")
    original_model = args.model
    original_signature = file_signature(Path(original_model).expanduser())
    inspection = inspect_downloaded_model(original_model, args.model_info)
    if file_signature(Path(original_model).expanduser()) != original_signature:
        raise RuntimeError("Source checkpoint changed during model inspection.")
    execution_inspection = inspection
    args.model_conversion = None
    from backend_registry import PROFILES
    native = (args.engine == "native" or bool(args.components)
              or (args.engine == "auto" and inspection.get("architecture") in PROFILES
                  and inspection.get("architecture") not in ("sd1", "sdxl") and not args.model_config))
    if inspection.get('format') == 'safetensors-index':
        if args.engine == 'diffusers':
            raise ValueError('A standalone Safetensors shard index requires the native engine, not a Diffusers single-file route.')
        native = True
    if native and args.engine == "diffusers":
        raise ValueError("--components are native weight files; select --engine native.")
    selected = None
    if inspection.get("format") in ("pytorch", "pytorch-index"):
        if inspection.get("role") != "checkpoint" or inspection.get("task") != "text-to-image":
            raise ValueError("Legacy conversion requires a text-to-image checkpoint; inspect the model role and task first.")
        if native and inspection.get("architecture") not in PROFILES:
            raise ValueError("No native image backend supports this checkpoint architecture.")
        if not native:
            selected, _ = inspect_checkpoint(original_model, args.base_model, args.model_info)
        if args.print_config or args.validate_only:
            raise ValueError("Legacy checkpoint conversion is deferred until generation. "
                             "Use --inspect-model for a metadata-only check.")
        from checkpoint_conversion import materialize_safetensors, verify_conversion_source
        conversion = materialize_safetensors(original_model, args.cache_dir / "checkpoint-conversions")
        if not conversion.get("converted"):
            raise RuntimeError("Legacy checkpoint conversion did not produce safetensors weights.")
        if file_signature(Path(original_model).expanduser()) != original_signature:
            raise RuntimeError("Source checkpoint changed between inspection and conversion.")
        if inspection.get('format') == 'pytorch-index':
            indexed = inspection.get('sharded_index', {})
            paths = [Path(original_model).expanduser(), *map(Path, indexed.get('shards', ()))]
            signatures = indexed.get('source_signatures', ())
            if len(paths) != len(signatures) or any(file_signature(path) != tuple(signature)
                                                    for path, signature in zip(paths, signatures)):
                raise RuntimeError('PyTorch index or shard changed between inspection and conversion.')
        verify_conversion_source(conversion)
        args.model_conversion = conversion
        args.original_model_path = str(Path(original_model).expanduser().absolute())
        args.model = conversion["converted_path"]
        execution_inspection = dict(inspection, format="safetensors")
    if native:
        from native_image import resolve_arguments as resolve_native
        return resolve_native(args, execution_inspection)
    if args.krea2_variant != 'auto' or args.krea2_mu is not None or args.native_sampler != 'auto':
        raise ValueError('Krea 2 / native sampling controls require their matching backend.')
    if selected is None:
        selected, inspection = inspect_checkpoint(original_model, args.base_model, args.model_info)
    if "preset" not in getattr(args, "_provided", ()):
        args.preset = selected
        args._provided = set(args._provided) | {"preset"}
    if not args.model_config:
        missing = set(inspection.get("missing_components", []))
        if args.vae or pipeline_vae_family(PRESETS[args.preset].pipeline_class):
            missing.discard("vae")
        if missing:
            raise ValueError("Checkpoint is missing components: " + ", ".join(sorted(missing))
                             + ". Provide a complete local --model-config with companion weights.")
    if args.prediction_type == "auto" and inspection.get("prediction_type") in ("epsilon", "v_prediction"):
        args.prediction_type = inspection["prediction_type"]
    if inspection.get("zero_terminal_snr"):
        args.scheduler_config = dict(args.scheduler_config)
        args.scheduler_config.setdefault("rescale_betas_zero_snr", True)
        args.scheduler_config.setdefault("timestep_spacing", "trailing")
    if args.output is not None and args.output_dir is not None:
        raise ValueError("Choose --output or --output-dir, not both.")
    if args.animation_mode != "none":
        raise ValueError("Use --backend deforum or interpolator for checkpoint animation.")
    preset, resolved = generate.resolve_arguments(args)
    resolved.engine = "diffusers"
    generate.validate_generation_arguments(preset, resolved)
    return preset, resolved


def empty_directory(path, *, defer_creation=False):
    path = path.expanduser().absolute()
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("The output/work directory must not be redirected.")
    if path.exists() and (not path.is_dir() or any(provider.iterdir(path))):
        raise ValueError("The output/work directory must be empty.")
    # A new batch is published by renaming its complete private stage. Creating
    # its destination first prevents that rename on Windows.
    provider.mkdir(path.parent if defer_creation and not path.exists() else path, parents=True, exist_ok=True)
    return path


def relocate(value, stage, output):
    if isinstance(value, dict):
        return {key: relocate(member, stage, output) for key, member in value.items()}
    if isinstance(value, list):
        return [relocate(member, stage, output) for member in value]
    if isinstance(value, str) and value.startswith(str(stage) + os.sep):
        return str(output / Path(value).relative_to(stage))
    return value


def main(argv=None):
    try:
        preset, args = resolve_arguments(build_parser().parse_args(argv))
        runner = generate.run
        configuration = configuration_values
        if getattr(args, "engine", None) == "native":
            from native_image import run as runner
            from native_image import configuration_values as configuration
        if args.print_config or args.validate_only:
            print(json.dumps(configuration(args), indent=2))
            return 0
        if is_preparing() or (args.output_dir is None and not args.output_was_default):
            result = runner(preset, args)
            from checkpoint_conversion import verify_conversion_source
            verify_conversion_source(getattr(args, "model_conversion", None))
            from image_package import verify_source as verify_package_source
            verify_package_source(getattr(args, 'image_package', None))
            return result
        output = empty_directory(args.output_dir or DEFAULT_OUTPUT_DIRECTORY / uuid.uuid4().hex, defer_creation=True)
        if args.work_dir:
            work = empty_directory(args.work_dir)
            provider.write_text(work / "request.json", json.dumps(configuration(args), indent=2))
        # Inference stays in this process. Publish the complete directory only
        # after images, hashes and per-image provenance have all been checked.
        with provider.TemporaryDirectory(prefix=".iild-generation-", dir=output.parent) as temporary:
            stage = Path(temporary)
            args.output, args.output_was_default = stage / "image.png", False
            args.xet_cache_dir = args.cache_dir / "xet"
            result = runner(preset, args)
            if result:
                return result
            from checkpoint_conversion import verify_conversion_source
            verify_conversion_source(getattr(args, "model_conversion", None))
            verify_weight_file(args.model_selection.single_file, "model")
            if args.vae_file:
                verify_weight_file(args.vae_file, "VAE")
            verify_vae_selection(getattr(args, "vae_selection", None))
            images = sorted(p for p in stage.glob("*.png") if not p.stem.endswith("-base"))
            if len(images) != args.num_images:
                raise RuntimeError("Inference did not produce the requested image count.")
            reports = []
            from PIL import Image
            for path in images:
                with Image.open(path) as image:
                    image.load()
                    if image.format != "PNG":
                        raise RuntimeError("Inference output is not a PNG image.")
                report = json.loads(path.with_suffix(".json").read_text())
                if report["output"]["sha256"] != file_sha256(path):
                    raise RuntimeError("Generated image does not match its provenance.")
                reports.append(relocate(report, stage, output))
            for path in stage.glob("*.json"):
                provider.write_text(path, json.dumps(relocate(json.loads(path.read_text()), stage, output), indent=2))
            manifest = {"schema": "iild-standalone-image-v1", "status": "complete", "backend": getattr(args, "engine", "diffusers"),
                        "output_directory": str(output), "outputs": [report["output"] for report in reports],
                        "images": reports}
            if getattr(args, "native_plan", None):
                manifest["backend_plan"] = args.native_plan
                manifest["architecture"] = args.architecture
            provider.write_text(stage / "generation.json", json.dumps(manifest, indent=2))
            if output.resolve() != output or output.is_symlink():
                raise RuntimeError("The output directory was redirected during generation.")
            verify_conversion_source(getattr(args, "model_conversion", None))
            from image_package import verify_source as verify_package_source
            verify_package_source(getattr(args, 'image_package', None))
            # rename cannot replace a nonempty destination; concurrent output is preserved.
            provider.replace(stage, output)
        print(f"Generation: {output / 'generation.json'}")
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        raise SystemExit(f"Standalone image generation failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
