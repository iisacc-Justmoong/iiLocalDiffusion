"""Compose downloaded checkpoint files in an isolated, managed local runtime."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

from comfyui_runtime import Client, run as run_workflow, write_json
from generation_config import json_object
from weight_files import file_sha256

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "build/reference/ComfyUI"
DEFAULT_PYTHON = ROOT / "build/reference/comfyui-venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--model-info", type=Path)
    parser.add_argument("--components", type=json_object, default={}, help="Local vae/text_encoder/text_encoder_2/3/4 paths")
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--decoder", type=Path)
    parser.add_argument("--model-negative", type=Path)
    for index in range(1, 5):
        parser.add_argument("--text-encoder" + (f"-{index}" if index > 1 else ""), type=Path)
    parser.add_argument("--model-type", choices=("auto", "checkpoint", "diffusion_model"), default="auto")
    parser.add_argument("--prompt", default="a red cube on a white table")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--embedded-guidance", type=float)
    parser.add_argument("--sampling-shift", type=float)
    parser.add_argument("--clip-skip", type=int)
    parser.add_argument("--zsnr", action="store_true")
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--sampler")
    parser.add_argument("--scheduler")
    parser.add_argument("--hires-fix", action=argparse.BooleanOptionalAction, default=False,
                        help="Upscale the latest generated image and refine it repeatedly")
    parser.add_argument("--hires-passes", type=int,
                        help="Additional refinement passes after base generation (default: 1)")
    parser.add_argument("--hires-scale", type=float, help="Per-pass size multiplier (default: 2)")
    parser.add_argument("--hires-denoising-strength", "--hires-strength", dest="hires_strength", type=float,
                        help="Per-pass denoising strength in (0,1] (default: 0.35)")
    parser.add_argument("--hires-steps", type=int,
                        help="Per-pass full schedule length; defaults to --steps, then reduced by strength")
    parser.add_argument("--hires-upscaler", choices=("nearest", "bilinear", "bicubic", "lanczos"),
                        help="Interpolation before each refinement (default: lanczos)")
    parser.add_argument("--prediction-type", choices=("epsilon", "v_prediction"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "metal", "cuda", "rocm"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--runtime-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--runtime-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def source_file(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"{name} must be a nonempty local model file: {path}")
    if path.suffix.lower() not in (".safetensors", ".safetensor", ".gguf", ".ckpt", ".pt", ".pth", ".bin"):
        raise ValueError(f"Unsupported downloaded model format: {path.suffix}")
    return path


def resolved_request(args):
    from downloaded_model import inspect_downloaded_model
    from civitai_catalog import lookup_base_model
    from comfyui_image_workflow import workflow_requirements, resolve_workflow_hires
    for name in ("timeout", "startup_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    model = source_file(args.model, "--model")
    inspection = inspect_downloaded_model(model, args.model_info)
    if inspection["role"] in ("lora", "vae", "embedding", "controlnet"):
        raise ValueError(f"This download is a {inspection['role']}, not a base checkpoint; supply a base --model and use its component/adapter input.")
    base = args.base_model or inspection.get("base_model")
    if base is None:
        base = {"sd1": "SD 1.5", "sd2": "SD 2.1", "sdxl": "SDXL 1.0",
                "sd3": "SD 3", "flux1-dev": "Flux.1 D", "flux1-schnell": "Flux.1 S"}.get(inspection.get("architecture"))
    if base is None:
        raise ValueError("Cannot determine the downloaded model architecture; supply its Civitai --base-model or --model-info.")
    if args.base_model and inspection.get("base_model"):
        if lookup_base_model(args.base_model)["name"] != inspection["base_model"]:
            raise ValueError("--base-model conflicts with the downloaded model's metadata.")
    record = lookup_base_model(base)
    architecture = inspection.get("architecture")
    if architecture:
        family = "flux1" if architecture.startswith("flux1") else "sdxl" if architecture.startswith("sdxl") else architecture
        if family != record["family"]:
            raise ValueError("--base-model conflicts with the downloaded tensor architecture.")
        if architecture == "flux1-schnell" and record["name"] in ("Flux.1 D", "Flux.1 Krea"):
            raise ValueError("This Flux download has no required guidance embeddings.")
        if architecture == "flux1-dev" and record["name"] == "Flux.1 S":
            raise ValueError("This Flux download contains guidance embeddings; it is not Schnell.")
    if inspection.get("task") in ("inpainting", "image-to-image"):
        raise ValueError("This model requires image conditioning; use its Diffusers pipeline or explicit workflow with an input image.")
    recipe = workflow_requirements(base)
    hires = resolve_workflow_hires(base, width=args.width, height=args.height, steps=args.steps,
                                   hires_fix=args.hires_fix, hires_passes=args.hires_passes,
                                   hires_scale=args.hires_scale, hires_strength=args.hires_strength,
                                   hires_steps=args.hires_steps, hires_upscaler=args.hires_upscaler)
    components = dict(args.components)
    component_names = ("vae", "text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "model_negative", "decoder")
    for name in component_names:
        if getattr(args, name) is not None:
            components[name] = str(getattr(args, name))
    unknown = set(components) - set(component_names)
    if unknown:
        raise ValueError("Unknown component keys: " + ", ".join(sorted(unknown)))
    files = {name: str(source_file(value, name)) for name, value in components.items()}
    model_type = args.model_type
    if model_type == "auto":
        bundled = set(inspection.get("available_components", [])) & {"vae", "text_encoder"}
        model_type = "diffusion_model" if (not bundled and (inspection.get("weights_role") == "denoiser"
                                            or {"vae", "text_encoder"} <= set(files))
                                            or model.suffix.lower() == ".gguf") else "checkpoint"
    if model_type == "diffusion_model" and not {"vae", "text_encoder"} <= set(files):
        raise ValueError(f"Split {base} denoiser requires --vae and --text-encoder components.")
    prediction = args.prediction_type or inspection.get("prediction_type")
    if prediction not in (None, "epsilon", "v_prediction"):
        prediction = None  # Flow models use their architecture's native sampling implementation.
    return {"base_model": record["name"], "model": str(model), "model_type": model_type,
            "components": files, "inspection": inspection, "recipe": recipe, "prediction_type": prediction,
            "hires": hires}


def stage_models(request, job):
    from checkpoint_conversion import materialize_safetensors
    paths = {"model": request["model"], **request["components"]}
    staged, identities = {}, {}
    for name, value in paths.items():
        original = source_file(value, name)
        if original.suffix.lower() in (".ckpt", ".pt", ".pth", ".bin"):
            conversion = materialize_safetensors(original, ROOT / "build/reference/converted-checkpoints")
            path = Path(conversion["converted_path"])
            identities[name] = conversion
        else:
            path = original
            identities[name] = {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
        folder = (("checkpoints" if request["model_type"] == "checkpoint" else "diffusion_models") if name == "model"
                  else "diffusion_models" if name in ("model_negative", "decoder")
                  else "vae" if name == "vae" else "text_encoders")
        target = job / "models" / folder / (name + (".safetensors" if path.suffix == ".safetensor" else path.suffix))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(path)
        identities[name]["staged_path"] = str(target)
        identities[name]["staged_target"] = str(path.resolve())
        staged[name] = target.name
    return staged, identities


def verify_identities(identities):
    for identity in identities.values():
        target = Path(identity["staged_path"])
        if not target.is_symlink() or str(target.resolve()) != identity["staged_target"]:
            raise RuntimeError(f"Staged model path changed during generation: {target}")
        entries = [identity] if "converted_path" not in identity else [identity["original"],
            {"path": identity["converted_path"], **identity["output"]}]
        for entry in entries:
            path = Path(entry["path"])
            if path.stat().st_size != entry["size_bytes"] or file_sha256(path) != entry["sha256"]:
                raise RuntimeError(f"Model file changed during generation: {path}")


def verify_images(result, expected_count, expected_size=None):
    from PIL import Image
    artifacts = result.get("artifacts", [])
    if len(artifacts) != expected_count:
        raise RuntimeError(f"Expected {expected_count} images, received {len(artifacts)} artifacts.")
    images = []
    for artifact in artifacts:
        with Image.open(artifact["path"]) as image:
            image.load()
            if expected_size is not None and [image.width, image.height] != list(expected_size):
                raise RuntimeError(f"Expected final image size {list(expected_size)}, received {[image.width, image.height]}.")
            images.append({"path": artifact["path"], "width": image.width, "height": image.height,
                           "format": image.format, "extrema": image.convert("RGB").getextrema()})
    return images


@contextmanager
def managed_server(args, job):
    if not (args.runtime_source / "main.py").is_file() or not args.runtime_python.is_file():
        raise ValueError("Local image runtime is missing. Run reference/setup_comfyui.py once.")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    for folder in ("input", "server-output", "user", "temp", "custom_nodes"):
        (job / folder).mkdir(exist_ok=True)
    # The selected runtime's one pinned GGUF plugin remains visible when the
    # model and writable directories are isolated per invocation.
    config = job / "paths.json"
    config.write_text(json.dumps({"iild": {"custom_nodes": str(args.runtime_source.resolve() / "custom_nodes")}}))
    # Keep the venv executable path: resolving its symlink selects system Python
    # and discards the isolated runtime's installed dependencies.
    command = [str(args.runtime_python.expanduser().absolute()), str(args.runtime_source.resolve() / "main.py"),
               "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch",
               "--disable-api-nodes", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "ComfyUI-GGUF",
               "--base-directory", str(job), "--extra-model-paths-config", str(config),
               "--output-directory", str(job / "server-output"), "--input-directory", str(job / "input"),
               "--user-directory", str(job / "user"), "--temp-directory", str(job / "temp")]
    if args.device == "cpu":
        command.append("--cpu")
    if args.dtype != "auto":
        command.append("--force-fp32" if args.dtype == "float32" else "--force-fp16")
    environment = dict(os.environ)
    environment.update(HF_HOME=str(ROOT / "build/reference/huggingface"),
                       XDG_CACHE_HOME=str(ROOT / "build/reference/comfyui-cache"),
                       TORCH_HOME=str(ROOT / "build/reference/torch-cache"),
                       PYTHONPYCACHEPREFIX=str(ROOT / "build/pycache"))
    log = job / "runtime.log"
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=args.runtime_source, env=environment)
        try:
            client = Client(f"http://127.0.0.1:{port}", request_timeout=2)
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Local runtime exited during startup; see {log}")
                try:
                    objects = client.json("/object_info")
                    stats = client.json("/system_stats")
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Local runtime startup timed out; see {log}")
                    time.sleep(0.25)
            devices = [item.get("type") for item in stats.get("devices", [])]
            expected = "mps" if args.device in ("mps", "metal") else args.device
            if (args.device == "auto" and (not devices or all(device == "cpu" for device in devices))) or (
                    expected not in ("auto", "rocm") and expected not in devices):
                raise RuntimeError(f"Requested {args.device}, but local runtime selected {devices}.")
            if expected == "rocm" and "rocm" not in str(stats.get("system", {}).get("pytorch_version", "")).lower():
                raise RuntimeError("Requested ROCm, but the runtime did not report a ROCm PyTorch build.")
            (job / "object_info.json").write_text(json.dumps(objects))
            yield client.url, objects, stats
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main(argv=None):
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(tokens)
    try:
        # The public wrapper can run on plain system Python. Conversion, image
        # decoding and inference dependencies belong to the managed environment.
        interpreter = args.runtime_python.expanduser().absolute()
        needs_runtime = not args.print_config or args.model.suffix.lower() in (".ckpt", ".pt", ".pth", ".bin")
        if needs_runtime and interpreter.is_file() and str(Path(sys.executable).absolute()) != str(interpreter):
            return subprocess.run([str(interpreter), str(Path(__file__).absolute()), *tokens], check=False).returncode
        request = resolved_request(args)
        if args.print_config:
            print(json.dumps(request, indent=2))
            return 0
        from comfyui_image_workflow import build_workflow
        job = ROOT / "build/reference/local-image-jobs" / uuid.uuid4().hex
        job.mkdir(parents=True)
        staged, identities = stage_models(request, job)
        with managed_server(args, job) as (url, objects, stats):
            graph = build_workflow(request["base_model"], staged["model"],
                {name: value for name, value in staged.items() if name != "model"}, args.prompt, objects,
                model_type=request["model_type"], negative_prompt=args.negative_prompt,
                width=args.width, height=args.height, seed=args.seed, steps=args.steps,
                cfg=args.guidance_scale, sampler_name=args.sampler, scheduler=args.scheduler,
                batch_size=args.num_images, prediction_type=request["prediction_type"],
                guidance=args.embedded_guidance, sampling_shift=args.sampling_shift,
                clip_skip=args.clip_skip, zsnr=args.zsnr, hires_fix=args.hires_fix,
                hires_passes=args.hires_passes, hires_scale=args.hires_scale, hires_strength=args.hires_strength,
                hires_steps=args.hires_steps, hires_upscaler=args.hires_upscaler)
            workflow = job / "workflow.json"
            workflow.write_text(json.dumps(graph, indent=2))
            result = run_workflow(argparse.Namespace(
                workflow=workflow, workflow_inputs={}, base_model=request["base_model"],
                comfyui_url=url, output_dir=args.output_dir or job / "result", timeout=args.timeout,
                poll_interval=1, max_artifact_bytes=2 * 1024**3, print_config=False, validate_only=args.validate_only))
        verify_identities(identities)
        report = {"request": request, "files": identities, "runtime": stats,
                  "generation": result, "job_directory": str(job)}
        if not args.validate_only:
            final_stage = request["hires"]["stages"][-1] if request["hires"]["enabled"] else None
            expected_size = [final_stage["width"], final_stage["height"]] if final_stage else None
            report["images"] = verify_images(result, args.num_images, expected_size)
            write_json(Path(result["output_directory"]) / "local-image.json", report)
        write_json(job / "local-image.json", report)
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"Local image generation failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
