"""Execute a portable unified model through the native ordered RGB cascade."""
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

from generation_seed import resolve_seed
from generation_output import write_png
from inference_session import cached_pipeline, is_preparing, record_execution, verify_pipeline_sources
from model_merge_unified import SCHEMA
from native_image import NativeEngine, NativePreviewWriter
from standalone_image import build_parser, empty_directory
from weight_files import model_content_sha256, file_sha256, metadata_model_validation
from iild_package import inspect_archive, materialize_archive


def inspect_package(path, *, hashes=False):
    root = Path(path).expanduser().absolute()
    archive_entries = None
    if root.is_file() and root.suffix.lower() == ".iildmodel":
        root, data, archive_entries = inspect_archive(root, hashes=hashes)
    else:
        manifest = root / "model_index.json"
        if not root.is_dir() or root.resolve() != root or not manifest.is_file() or manifest.resolve() != manifest:
            raise ValueError("Choose an available canonical unified model package.")
        if manifest.stat().st_size > 1024 * 1024:
            raise ValueError("Unified model manifest exceeds 1 MiB.")
        data = json.loads(manifest.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("_class_name") != "IILDUnifiedCascade"
            or data.get("composition") != "ordered-image-refinement"):
        raise ValueError("Unsupported unified model manifest.")
    stages = data.get("stages")
    if not isinstance(stages, list) or not 1 <= len(stages) <= 64:
        raise ValueError("Unified models require 1 to 64 stages.")
    seen = set()
    variants = data.get("vae_variants", [])
    variant_ids = set()
    if variants:
        if not isinstance(variants, list) or len(variants) > 64:
            raise ValueError("Unified VAE variants must be a list of at most 64 entries.")
        for variant in variants:
            if not isinstance(variant, dict) or not isinstance(variant.get("id"), str) or not variant["id"]:
                raise ValueError("Invalid unified VAE variant.")
            if variant["id"] in variant_ids:
                raise ValueError("Duplicate unified VAE variant.")
            variant_ids.add(variant["id"])
            for field in ("weights", "config"):
                relative = variant.get(field)
                if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
                        or "\0" in relative or relative.startswith("/")
                        or any(part in ("", ".", "..") for part in relative.split("/"))):
                    raise ValueError("Invalid unified VAE variant path.")
                file = root / relative
                if ((archive_entries is not None and relative not in archive_entries)
                        or (archive_entries is None and (not file.is_file() or file.resolve() != file or file.stat().st_size <= 0))):
                    raise ValueError("Unified VAE variant is missing, empty or redirected.")
        catalog = data.get("vae_catalog")
        if catalog is not None:
            if (not isinstance(catalog, str) or not catalog or "\\" in catalog or ":" in catalog
                    or "\0" in catalog or catalog.startswith("/")
                    or any(part in ("", ".", "..") for part in catalog.split("/"))):
                raise ValueError("Invalid unified VAE catalog path.")
            catalog_path = root / catalog
            if ((archive_entries is not None and catalog not in archive_entries)
                    or (archive_entries is None and (not catalog_path.is_file() or catalog_path.resolve() != catalog_path))):
                raise ValueError("Unified VAE catalog is missing or redirected.")
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError("Invalid unified model stage.")
        relative, strength = stage.get("model"), stage.get("strength")
        if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "\0" in relative
                or relative.startswith("/") or any(part in ("", ".", "..") for part in relative.split("/"))):
            raise ValueError("Invalid unified model member path.")
        member = root / relative
        member_size = archive_entries[relative].file_size if archive_entries is not None and relative in archive_entries else (
            member.stat().st_size if archive_entries is None and member.is_file() else -1)
        member_identity = relative if archive_entries is not None else member
        if (member_identity in seen or type(stage.get("size_bytes")) is not int or stage["size_bytes"] <= 0
                or member_size <= 0 or (not metadata_model_validation() and member_size != stage["size_bytes"])
                or (archive_entries is None and member.resolve() != member)):
            raise ValueError("Unified model member is missing, changed, duplicated or redirected.")
        seen.add(member_identity)
        if "vae_variant" in stage and stage["vae_variant"] not in variant_ids:
            raise ValueError("Unified stage references an unknown VAE variant.")
        if "vae" in stage:
            name = stage["vae"]
            if (not isinstance(name, str) or not name or any(c in name for c in "\\:\0")
                    or name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/"))):
                raise ValueError("Invalid unified VAE path.")
            vae = root / name
            if ((archive_entries is not None and (name not in archive_entries or archive_entries[name].file_size <= 0))
                    or (archive_entries is None and (not vae.is_file() or vae.resolve() != vae or vae.stat().st_size <= 0))):
                raise ValueError("Unified VAE is missing, empty or redirected.")
        if (type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1
                or (index == 0 and strength != 1)):
            raise ValueError("Invalid unified refinement strength.")
        if hashes and archive_entries is None and (digest := model_content_sha256(member)) is not None and digest != stage.get("sha256"):
            raise ValueError("Unified model member hash differs from the published object.")
    return root, data


def main(argv=None):
    args = build_parser().parse_args(argv)
    supported = {"model", "output", "output_dir", "work_dir", "print_config", "validate_only", "prompt", "negative_prompt",
                 "width", "height", "steps", "seed", "num_images", "seed_stride", "generation_resources", "cache_dir",
                 "preview_dir", "default_modifiers", "progress", "device", "local_files_only", "png_compress_level", "png_optimize"}
    unknown = set(args._provided) - supported
    if unknown:
        raise ValueError("Unified cascades do not support these overrides: " + ", ".join(sorted(unknown)))
    source, manifest = inspect_package(args.model)
    root = materialize_archive(source, args.cache_dir / "iildmodel") if source.is_file() else source
    if args.device != "auto" or not args.local_files_only:
        raise ValueError("Unified cascades use local native inference with --device auto.")
    if args.output is not None and args.output_dir is not None:
        raise ValueError("Choose --output or --output-dir, not both.")
    args.model = str(root)
    args.width = 512 if args.width is None else args.width
    args.height = 512 if args.height is None else args.height
    args.steps = 20 if args.steps is None else args.steps
    args.seed = resolve_seed(args.seed)
    args.native_loras = []
    if any(not 64 <= n <= 2048 or n % 8 for n in (args.width, args.height)):
        raise ValueError("Unified image dimensions must be multiples of 8 in [64,2048].")
    if not 1 <= args.steps <= 1000 or not 1 <= args.num_images <= 1000 or not 0 <= args.png_compress_level <= 9:
        raise ValueError("Invalid unified image steps, image count or PNG compression.")
    if not all(0 <= seed < 2**63 for seed in (args.seed, args.seed + (args.num_images - 1) * args.seed_stride)):
        raise ValueError("Unified seeds must be in [0,2^63).")
    if not args.prompt.strip() or any("\0" in text or len(text.encode()) > 128000 for text in (args.prompt, args.negative_prompt)):
        raise ValueError("Use a nonempty prompt and bounded text without NUL.")
    configuration = {"backend": "unified", "model": str(source), "composition": manifest["composition"],
                     "model_validation": "metadata" if metadata_model_validation() else "sha256",
                     "stages": manifest["stages"], "width": args.width, "height": args.height, "steps": args.steps, "seed": args.seed}
    if args.print_config or args.validate_only:
        print(json.dumps(configuration, indent=2)); return 0
    inspect_package(source, hashes=True)
    manifest_digest = file_sha256(source if source.is_file() else root / "model_index.json")
    sources = [source, *[root / stage["model"] for stage in manifest["stages"]]]
    sources.extend(root / stage["vae"] for stage in manifest["stages"] if "vae" in stage)
    key = ("native-unified", source, str(args.generation_resources), args.default_modifiers)
    engine, _ = cached_pipeline(key, sources, NativeEngine)
    if is_preparing():
        engine.image(args, args.seed, prepare=True)
        verify_pipeline_sources()
        record_execution("native-auto", "managed", source)
        return 0
    if args.preview_dir:
        args._native_preview = NativePreviewWriter(args.preview_dir)
    if args.output and args.num_images != 1:
        raise ValueError("Use --output-dir for multiple unified images.")
    if args.output and args.output.suffix.lower() != ".png":
        raise ValueError("Unified image output must end in .png.")
    output = args.output.expanduser().absolute() if args.output else empty_directory(args.output_dir or
        Path(__file__).resolve().parents[2] / "build/reference/unified-image" / uuid.uuid4().hex)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.work_dir:
        work = empty_directory(args.work_dir)
        (work / "request.json").write_text(json.dumps(configuration, indent=2))
    with tempfile.TemporaryDirectory(prefix=".iild-unified-", dir=output.parent) as temporary:
        staging = Path(temporary) / "images"
        staging.mkdir()
        reports = []
        for index in range(args.num_images):
            seed = args.seed + index * args.seed_stride
            image, performance = engine.image(args, seed)
            verify_pipeline_sources()
            inspect_package(source, hashes=True)
            if file_sha256(source if source.is_file() else root / "model_index.json") != manifest_digest:
                raise RuntimeError("Unified manifest changed during generation.")
            name = output.name if args.output else f"image-{index + 1:04d}.png"
            target = staging / name
            write_png(image, target, compress_level=args.png_compress_level, optimize=args.png_optimize, overwrite=False)
            report = {**configuration, "seed": seed, "prompt": args.prompt, "negative_prompt": args.negative_prompt,
                      "performance": performance, "output": {"path": str(output if args.output else output / name),
                      "width": args.width, "height": args.height, "sha256": file_sha256(target)}}
            target.with_suffix(".json").write_text(json.dumps(report, indent=2))
            reports.append(report)
        record_execution("native-auto", "managed", source)
        if args.output:
            published = []
            try:
                for source, destination in ((staging / output.with_suffix(".json").name, output.with_suffix(".json")),
                                             (staging / output.name, output)):
                    os.link(source, destination); published.append(destination)
            except BaseException:
                for path in published:
                    path.unlink()
                raise
        else:
            (staging / "generation.json").write_text(json.dumps({"schema": "iild-standalone-image-v1", "status": "complete",
                "backend": "unified", "output_directory": str(output), "outputs": [r["output"] for r in reports], "images": reports}, indent=2))
            if output.resolve() != output or output.is_symlink():
                raise RuntimeError("Unified output directory was redirected.")
            os.replace(staging, output)
    print(f"Generation: {output if args.output else output / 'generation.json'}")
    return 0
