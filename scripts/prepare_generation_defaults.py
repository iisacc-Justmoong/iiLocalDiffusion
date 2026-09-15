#!/usr/bin/env python3
"""Regenerate verified, tensor-only copies of the bundled legacy embeddings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference/diffusers"))
from generation_defaults import fallback_loras


def prepare(root, check=False):
    import torch
    from safetensors.torch import load_file, save_file

    manifest = json.loads((root / "generation-defaults.json").read_text())
    for item in manifest["negative_embeddings"]:
        if "source" not in item:
            continue
        source = root / item["source"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["source_sha256"]:
            raise ValueError(f"Source embedding changed: {source}")
        state = torch.load(source, map_location="cpu", weights_only=True)
        vectors = state["string_to_param"]["*"].detach().contiguous()
        if (list(vectors.shape) != item["shape"] or vectors.dtype != torch.float32
                or not torch.isfinite(vectors).all()):
            raise ValueError(f"Unexpected learned vectors: {source}")
        target = root / item["file"]
        if check:
            actual = load_file(target)["clip_l"]
            if not torch.equal(actual, vectors):
                raise ValueError(f"Converted embedding differs from its source: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            save_file({"clip_l": vectors}, target, metadata={"token": item["token"],
                      "source_sha256": item["source_sha256"]})
    vaes = ([manifest["fallback_vae"]] if "fallback_vae" in manifest else []) + manifest.get("fallback_vaes", [])
    for item in [*fallback_loras(manifest), *manifest["negative_embeddings"],
                 *(resource for vae in vaes for resource in (vae, vae["config"]))]:
        path = root / item["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if check:
            if digest != item["sha256"] or path.stat().st_size != item["size"]:
                raise ValueError(f"Bundled resource identity changed: {path}")
        else:
            item.update(sha256=digest, size=path.stat().st_size)
    if not check:
        (root / "generation-defaults.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    prepare(Path(__file__).resolve().parents[1] / "resources", options.check)
