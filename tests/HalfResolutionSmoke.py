#!/usr/bin/env python3
"""Offline two-stage image smoke; artifacts stay under build/.

The default uses random tiny SD1/SDXL weights and the public preset runner.
--native-model additionally exercises a caller-provided real checkpoint via
the built C bridge. Neither path establishes perceptual quality.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "reference/diffusers"), str(ROOT / "tests")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-model", type=Path)
    args = parser.parse_args()
    output = Path(tempfile.mkdtemp(prefix="half-resolution-", dir=ROOT / "build"))
    reports = []
    if args.native_model:
        from native_image import NativeEngine
        engine = NativeEngine()
        request = SimpleNamespace(model=str(args.native_model.resolve()), prompt="a white cockatoo on a branch",
                                  negative_prompt="", width=512, height=512, steps=2,
                                  default_modifiers=False, generation_resources=None, native_loras=[], progress=True)
        image, performance = engine.image(request, 42)
        assert image.size == (512, 512)
        assert any(low != high for low, high in image.getextrema())
        image.save(output / "native.png")
        reports.append({"backend": "native", "requested_size": [512, 512],
                        "base_request_size": [256, 256], "output_size": list(image.size),
                        "performance": performance})
    else:
        import torch
        import generate
        from inference_session import InferenceSession
        from UniversalLoraDiffusersSmoke import fixture
        torch.set_num_threads(2)
        for family, preset in (("sd15", "sd15-compatible"), ("sdxl-base", "sdxl")):
            fixture(output / family, family)
            with InferenceSession():
                for index, size in enumerate(((64, 64), (88, 72), (64, 64))):
                    path = output / family / f"result-{index}.png"
                    preview = output / f"{family}-preview-{index}"
                    selected, request = generate.resolve_request({
                        "model": str(output / family / "model"), "preset": preset,
                        "default_modifiers": False, "device": "cpu", "dtype": "float32",
                        "offload": "none", "cpu_threads": 2, "prompt": "a", "seed": 42,
                        "width": size[0], "height": size[1], "steps": 2,
                        "guidance_scale": 3, "progress": False, "hires_save_base": True,
                        "preview_dir": str(preview), "output": str(path),
                        "cache_dir": str(output / "cache"), "xet_cache_dir": str(output / "xet")})
                    assert generate.run(selected, request) == 0
                    report = json.loads(path.with_suffix(".json").read_text())
                    assert report["output"]["size"] == list(size)
                    expected_base = [((axis + 8) // 16) * 8 for axis in size]
                    assert report["hires_fix"]["base"]["size"] == expected_base
                    assert report["hires_fix"]["base"]["executed_steps"] > 0
                    assert report["hires_fix"]["completed_passes"] == 1
                    assert report["hires_fix"]["refinement"]["executed_steps"] > 0
                    assert list(request.preview_dir.glob("step-*.png"))
                    assert (report["parameters"]["width"], report["parameters"]["height"]) == size
                    if index:
                        assert report["runtime"]["model_configuration_cache_hit"]
                        assert not report["runtime"]["device_placement_cache_hit"]
                    reports.append({"backend": "diffusers", "family": family,
                                    "report": str(path.with_suffix(".json")),
                                    "base_size": expected_base, "output_size": list(size)})
    result = output / "verification.json"
    result.write_text(json.dumps({"runs": reports}, indent=2) + "\n")
    print(f"Verification: {result}", flush=True)


if __name__ == "__main__":
    main()
