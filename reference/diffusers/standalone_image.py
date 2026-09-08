"""Generate local checkpoint images directly with the SDK's Diffusers/PyTorch runtime."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import uuid

import generate
from checkpoint_config import inspect_checkpoint
from generation_config import configuration_values
from inference_session import is_preparing
from weight_files import file_sha256, verify_weight_file

ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    parser = generate.build_parser()
    parser.description = __doc__
    parser.add_argument("--output-dir", type=Path, help="Empty output directory for images and generation.json")
    parser.add_argument("--work-dir", type=Path, help="Empty caller-owned directory for the resolved request")
    parser.add_argument("--model-info", type=Path, help="Optional Civitai metadata for the local checkpoint")
    parser.add_argument("--validate-only", action="store_true", help="Validate the offline model/configuration request without inference")
    # Accept previous desktop callers while removing the server cold-start requirement.
    parser.add_argument("--startup-timeout", type=float, help=argparse.SUPPRESS)
    return parser


def resolve_arguments(args):
    if not args.model or not Path(args.model).expanduser().is_file():
        raise ValueError("The standalone checkpoint backend requires a local --model-path file.")
    selected, inspection = inspect_checkpoint(args.model, args.base_model, args.model_info)
    if "preset" not in getattr(args, "_provided", ()):
        args.preset = selected
        args._provided = set(args._provided) | {"preset"}
    if not args.model_config:
        missing = set(inspection.get("missing_components", []))
        if args.vae:
            missing.discard("vae")
        if missing:
            raise ValueError("Checkpoint is missing components: " + ", ".join(sorted(missing))
                             + ". Provide a complete local --model-config with companion weights.")
    if args.prediction_type == "auto" and inspection.get("prediction_type") in ("epsilon", "v_prediction"):
        args.prediction_type = inspection["prediction_type"]
    if args.output is not None and args.output_dir is not None:
        raise ValueError("Choose --output or --output-dir, not both.")
    if args.animation_mode != "none":
        raise ValueError("Use --backend deforum or interpolator for checkpoint animation.")
    preset, resolved = generate.resolve_arguments(args)
    generate.validate_generation_arguments(preset, resolved)
    return preset, resolved


def empty_directory(path):
    path = path.expanduser().absolute()
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("The output/work directory must not be redirected.")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("The output/work directory must be empty.")
    path.mkdir(parents=True, exist_ok=True)
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
        if args.print_config or args.validate_only:
            print(json.dumps(configuration_values(args), indent=2))
            return 0
        if is_preparing() or (args.output_dir is None and not args.output_was_default):
            return generate.run(preset, args)
        output = empty_directory(args.output_dir or ROOT / "build/reference/standalone-image" / uuid.uuid4().hex)
        if args.work_dir:
            work = empty_directory(args.work_dir)
            (work / "request.json").write_text(json.dumps(configuration_values(args), indent=2))
        # Inference stays in this process. Publish the complete directory only
        # after images, hashes and per-image provenance have all been checked.
        with tempfile.TemporaryDirectory(prefix=".iild-generation-", dir=output.parent) as temporary:
            stage = Path(temporary)
            args.output, args.output_was_default = stage / "image.png", False
            args.xet_cache_dir = args.cache_dir / "xet"
            result = generate.run(preset, args)
            if result:
                return result
            verify_weight_file(args.model_selection.single_file, "model")
            if args.vae_file:
                verify_weight_file(args.vae_file, "VAE")
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
                path.write_text(json.dumps(relocate(json.loads(path.read_text()), stage, output), indent=2))
            manifest = {"schema": "iild-standalone-image-v1", "status": "complete", "backend": "diffusers",
                        "output_directory": str(output), "outputs": [report["output"] for report in reports],
                        "images": reports}
            (stage / "generation.json").write_text(json.dumps(manifest, indent=2))
            if output.resolve() != output or output.is_symlink():
                raise RuntimeError("The output directory was redirected during generation.")
            # rename cannot replace a nonempty destination; concurrent output is preserved.
            os.replace(stage, output)
        print(f"Generation: {output / 'generation.json'}")
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        raise SystemExit(f"Standalone image generation failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
