"""Replay nonfinite checkpoint failures using real tensors and the installed CLI.

Original models are read-only. Evidence and sample outputs remain in build/.
This validates arithmetic and publication, not a full model or image generation.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from model_merge_ecosystem import identify_ecosystem

TARGET = "conditioner.embedders.0.transformer.text_model.encoder.layers.11.mlp.fc1.weight"
PROBE = "conditioner.embedders.0.transformer.text_model.encoder.layers.10.mlp.fc1.bias"


def sample(path):
    with safe_open(path, framework="pt") as reader:
        layout = {(".", key): "model" for key in reader.keys()}
        family = identify_ecosystem({"model": reader}, layout)
        metadata = {**(reader.metadata() or {}), "iild.model_family": ",".join(family.families)}
        return {key: reader.get_tensor(key).clone() for key in (TARGET, PROBE)}, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--material", required=True, type=Path, action="append")
    parser.add_argument("--repair-base", action="store_true", help="Replay a real nonfinite base with usable material.")
    parser.add_argument("--executable", type=Path, default=ROOT / "build/unified-runtime/bin/iild-merge")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="nonfinite-real-", dir=ROOT / "build"))
    sources = [args.base, *args.material]
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in sources]
    base, metadata = sample(args.base)
    base_invalid = sum(int((~torch.isfinite(value.float())).sum()) for value in base.values())
    assert base_invalid > 0 if args.repair_base else base_invalid == 0
    save_file(base, root / "base.safetensors", metadata=metadata)
    results = []
    for index, path in enumerate(args.material, 1):
        material, metadata = sample(path)
        invalid = int((~torch.isfinite(material[TARGET].float())).sum())
        assert invalid > 0 or args.repair_base, "This verifier requires a real nonfinite target or base."
        assert torch.isfinite(material[PROBE].float()).all(), "Probe must provide a normal contribution."
        material_path = root / f"material-{index}.safetensors"
        save_file(material, material_path, metadata=metadata)
        for mode in ("weighted-sum", "weighted-difference"):
            output = root / f"merged-{index}-{mode}.safetensors"
            process = subprocess.run([str(args.executable), "--base-model", str(root / "base.safetensors"),
                "--additional-model", str(material_path), "--weights", "0.5", "--mode", mode,
                "--output", str(output)], text=True, capture_output=True, check=True)
            report = json.loads(process.stdout)
            (root / f"report-{index}-{mode}.json").write_text(json.dumps(report, indent=2) + "\n")
            merged = load_file(output)
            for key in (TARGET, PROBE):
                a = torch.nan_to_num(base[key].float(), nan=0., posinf=0., neginf=0.)
                b = material[key].float()
                valid = torch.isfinite(b)
                expected = torch.where(valid, .5 * a + .5 * b, a) if mode == "weighted-sum" else torch.where(valid, a - .5 * b, a)
                limits = torch.finfo(base[key].dtype)
                expected = expected.clamp(limits.min, limits.max)
                torch.testing.assert_close(merged[key], expected.to(base[key].dtype), rtol=0, atol=0)
                assert torch.isfinite(merged[key].float()).all()
            assert report["numeric_normalization"]["nonfinite_material_values"] == invalid
            assert report["numeric_normalization"].get("base_nonfinite_values", 0) == base_invalid
            assert report["changed_tensor_count"] > 0, "Normal probe must still merge."
            verification = report["output_verification"]
            assert verification["status"] == "passed" and verification["tensor_count"] == len(merged)
            assert verification["changed_tensor_count"] == report["changed_tensor_count"]
            results.append({"material": str(path), "mode": mode, "nonfinite_values": invalid,
                            "output_verification": verification,
                            "changed_tensors": report["changed_tensor_count"], "output": str(output),
                            "output_sha256": report["output_files"][0]["sha256"]})
    assert before == [(p.stat().st_size, p.stat().st_mtime_ns) for p in sources]
    evidence = {"base": str(args.base), "tensor": TARGET, "probe": PROBE,
                "installed_executable": str(args.executable), "originals_unchanged": True,
                "base_nonfinite_values": base_invalid,
                "full_model_inference_tested": False, "results": results}
    (root / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"evidence": str(root / "verification.json"), "verified_merges": len(results)}, indent=2))


if __name__ == "__main__":
    main()
