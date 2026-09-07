#!/usr/bin/env python3
"""Real LTX inference with small random weights; no quality or trained-model claim."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def create_fixture(directory):
    import torch
    from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXConditionPipeline, LTXVideoTransformer3DModel
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, T5Config, T5EncoderModel
    with torch.random.fork_rng():
        torch.manual_seed(17)
        words = ["[PAD]", "[EOS]", "[UNK]", "A", "red", "cube", "blue", "table", "camera",
                 "forward", "left", "rotates", "slowly", "."]
        tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]",
                                            unk_token="[UNK]", model_max_length=256)
        text_encoder = T5EncoderModel(T5Config(vocab_size=len(words), d_model=32, d_ff=64, num_layers=1,
                                               num_heads=4, d_kv=8, pad_token_id=0, eos_token_id=1))
        vae = AutoencoderKLLTXVideo(latent_channels=4, block_out_channels=(8, 8, 8, 8),
                                    decoder_block_out_channels=(8, 8, 8, 8),
                                    layers_per_block=(1, 1, 1, 1, 1), decoder_layers_per_block=(1, 1, 1, 1, 1))
        transformer = LTXVideoTransformer3DModel(in_channels=4, out_channels=4, num_attention_heads=2,
                                                attention_head_dim=16, cross_attention_dim=32,
                                                caption_channels=32, num_layers=1)
        pipeline = LTXConditionPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
                                        transformer=transformer, scheduler=FlowMatchEulerDiscreteScheduler())
        pipeline.save_pretrained(directory, safe_serialization=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--fps", type=float, default=24, help="Final rate; >12 exercises LTX plus frame interpolation")
    args = parser.parse_args()
    directory = ROOT / "build/video-smoke"
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "model"
    if not (model / "model_index.json").exists():
        create_fixture(model)
    from PIL import Image
    reference = directory / "keyframe.png"
    image = Image.new("RGB", (32, 32), (24, 40, 56))
    for x in range(8, 24):
        for y in range(8, 24):
            image.putpixel((x, y), (240, 50, 30))
    image.save(reference)
    storyboard = directory / "storyboard.json"
    storyboard.write_text(json.dumps({"shots": [{"prompt": "A red cube.", "frames": 9},
                                               {"prompt": "A blue cube.", "frames": 9,
                                                "continue_previous": True, "camera": ["pan-left"]}]}))
    environment = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    results = []
    for mode, extra in [("text", []), ("keyframes", ["--first-frame", str(reference), "--last-frame", str(reference)]),
                        ("storyboard", ["--storyboard", str(storyboard), "--cpu-text-encoding"])]:
        output = directory / f"{mode}-{args.device}-{args.fps:g}fps.mp4"
        command = [sys.executable, str(ROOT / "reference/generate.py"), "--backend", "video",
                   "--model", str(model), "--device", args.device, "--dtype", "float32", "--frames", "9",
                   "--steps", "2", "--width", "32", "--height", "32", "--fps", str(args.fps),
                   "--max-sequence-length", "64", "--guidance-scale", "3", "--camera", "dolly-in",
                   "--no-vae-tiling", "--output", str(output), "--overwrite", "--local-files-only", *extra]
        with output.with_suffix(".log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f"Video smoke failed; inspect {output.with_suffix('.log')}")
        report = json.loads(output.with_suffix(".json").read_text())
        assert report["video"]["verified_decode"]
        assert report["video"]["frame_count"] == (18 if mode == "storyboard" else 9)
        if args.fps > 12:
            assert [stage["name"] for stage in report["stages"]] == ["LTX", "Interpolator"]
            assert report["interpolation"]["verified"] and report["interpolation"]["inserted_frames"] > 0
            source_hashes = {frame["index"]: frame["sha256"] for frame in report["source_frames"]}
            assert all(frame["sha256"] == source_hashes[frame["index"]]
                       for frame in report["frames"] if frame["index"] in source_hashes)
        else:
            assert report["method"] == "joint-spatiotemporal-diffusion" and not report["interpolation"]["enabled"]
        assert all(len(shot["denoising"]) == 2 for shot in report["shots"])
        assert len({frame["sha256"] for frame in report["frames"]}) > 1
        results.append({"mode": mode, "device": args.device, "fps": args.fps, "video": str(output),
                        "stages": [stage["name"] for stage in report["stages"]], "verified": True})
        print(json.dumps(results[-1]), flush=True)
    (directory / f"verification-{args.device}-{args.fps:g}fps.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
