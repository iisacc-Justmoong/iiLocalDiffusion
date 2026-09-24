"""Bounded, reproducible validation against real checkpoint tensors.

Usage: build/merge-python/bin/python tests/verify_model_merge_samples.py BASE MATERIAL
Artifacts stay in a new build/real-merge-* directory; originals are read-only.
This checks sampled arithmetic and the complete header plan, not inference.
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from model_merge import inspect_merge_request, merge_models
from model_merge_options import resolve_merge_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("material", type=Path)
    args = parser.parse_args()
    output = Path(tempfile.mkdtemp(prefix="real-merge-", dir=ROOT / "build"))
    sources = [args.base, args.material]
    identities = [(p.stat().st_size, p.stat().st_mtime_ns) for p in sources]
    inspection = inspect_merge_request(resolve_merge_request(*sources, output=output / "full-plan.safetensors"))
    (output / "full-header-plan.json").write_text(json.dumps(inspection, indent=2) + "\n")
    with safe_open(args.base, framework="pt") as base, safe_open(args.material, framework="pt") as material:
        shared = [key for key in base.keys() if key in material.keys()
                  and base.get_slice(key).get_shape() == material.get_slice(key).get_shape()
                  and torch.tensor(base.get_slice(key).get_shape()).prod().item() <= 262144
                  and key.endswith((".weight", ".bias"))]
        if not shared:
            raise ValueError("No bounded exact-layout learned tensors to check.")
        keys = list(dict.fromkeys(shared[i * (len(shared) - 1) // min(63, len(shared) - 1)]
                                 for i in range(min(64, len(shared))))) if len(shared) > 1 else shared
        base_tensors = {key: base.get_tensor(key) for key in keys}
        material_tensors = {key: material.get_tensor(key) for key in keys}
        # Include the reported missing scheduler and integer state when present.
        for key in base.keys():
            value = base.get_slice(key)
            if key == "denoiser.sigmas" or (value.get_dtype() == "I64" and len(value.get_shape()) <= 2):
                base_tensors[key] = base.get_tensor(key)
                if key in material.keys():
                    material_tensors[key] = material.get_tensor(key)
        # A bounded sample may omit the structural family signature. Preserve
        # the family evidence obtained from the complete original inventories.
        compatibility = inspection["resource_compatibility"][0]
        base_metadata = {**(base.metadata() or {}), "iild.model_family": ",".join(compatibility["base_ecosystems"])}
        material_metadata = {**(material.metadata() or {}), "iild.model_family": ",".join(compatibility["material_ecosystems"])}
        save_file(base_tensors, output / "base-sample.safetensors", metadata=base_metadata)
        save_file(material_tensors, output / "material-sample.safetensors", metadata=material_metadata)
    report = merge_models(output / "base-sample.safetensors", output / "material-sample.safetensors",
                          weights=.5, output=output / "merged-sample.safetensors")
    merged = load_file(output / "merged-sample.safetensors")
    checked = []
    for key in keys:
        expected = (base_tensors[key].float() * .5 + material_tensors[key].float() * .5).to(base_tensors[key].dtype)
        torch.testing.assert_close(merged[key], expected, rtol=0, atol=0)
        checked.append({"tensor": key, "shape": list(expected.shape), "dtype": str(expected.dtype),
                        "changed": not torch.equal(merged[key], base_tensors[key])})
    if "denoiser.sigmas" in merged:
        torch.testing.assert_close(merged["denoiser.sigmas"], base_tensors["denoiser.sigmas"], rtol=0, atol=0)
    if identities != [(p.stat().st_size, p.stat().st_mtime_ns) for p in sources]:
        raise RuntimeError("Original source identity changed during verification.")
    evidence = {"sources": [str(p) for p in sources], "originals_unchanged": True,
                "full_model_inference_tested": False, "sample_arithmetic": checked, "merge_report": report}
    (output / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"artifact_directory": str(output), "checked_tensors": len(keys),
                      "changed_tensors": report["changed_tensor_count"],
                      "full_header_transform_counts": inspection["common_layers"]["1"]["transform_counts"]}, indent=2))


if __name__ == "__main__":
    main()
