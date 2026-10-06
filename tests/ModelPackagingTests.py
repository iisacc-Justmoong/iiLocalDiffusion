"""Exercise native packaging with real file formats and independent byte checks."""
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import unittest

PROGRAM = Path(sys.argv.pop(1)).resolve()


def safetensors(path, tensors, metadata=None):
    header = {}
    data = bytearray()
    for name, dtype, shape, payload in tensors:
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(data), len(data) + len(payload)]}
        data.extend(payload)
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def gguf(path):
    def string(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, 2)
    for key, value in [("general.architecture", "gemma3"), ("general.name", "Gemma 3")]:
        data += string(key) + struct.pack("<I", 8) + string(value)
    data += string("token_embd.weight") + struct.pack("<IQIQ", 1, 1, 0, 0)
    data += b"\0" * (-len(data) % 32) + struct.pack("<f", 0.25)
    path.write_bytes(data)


def read_header(path):
    with path.open("rb") as file:
        size, = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(size))
    return header, size + 8


def fixture(folder, encoder=True):
    model = [("model.diffusion_model.block.weight", "F8_E4M3", [4], b"\x01\x02\x03\x04"),
             ("model.diffusion_model.block.weight_scale", "F32", [1], struct.pack("<f", 0.5)),
             ("vae.decoder.weight", "BF16", [2], b"\x01\x02\x03\x04"),
             ("audio_vae.decoder.weight", "BF16", [1], b"\x05\x06"),
             ("vocoder.vocoder.weight", "BF16", [1], b"\x07\x08"),
             ("text_embedding_projection.audio_aggregate_embed.weight", "BF16", [1], b"\x09\x0a")]
    safetensors(folder / "ltxv23_fp8mixed.safetensors", model,
                {"model_version": "2.3.0", "license": "fixture-license", "config": '{"_class_name":"LTXModel"}'})
    safetensors(folder / "video_vae.safetensors", [("decoder.weight", "BF16", [2], b"\x01\x02\x03\x04")])
    safetensors(folder / "audio_vae.safetensors", model[3:5])
    safetensors(folder / "projections.safetensors", model[5:6])
    safetensors(folder / "spatial_upscaler_x2.safetensors", [("initial_conv.weight", "BF16", [2], b"\x10\x11\x12\x13")],
                {"config": '{"_class_name":"LatentUpsampler"}'})
    (folder / "ltx-failed.safetensors").write_bytes(b"Auth failed: credentials expired")
    if encoder:
        gguf(folder / "gemma-3.gguf")
    return model


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="model-packaging-test-", dir=PROGRAM.parent)
        self.root = Path(self.temporary.name)
        self.input = self.root / "input"
        self.input.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def run_tool(self, operation, *args, success=True):
        result = subprocess.run([str(PROGRAM), operation, *map(str, args)], text=True,
                                capture_output=True, timeout=30)
        reports = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertTrue(reports, result.stderr)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return reports[-1]

    def test_ltx_deduplication_formats_and_byte_exact_extraction(self):
        fixture(self.input)
        original = {p.name: p.read_bytes() for p in self.input.iterdir()}
        scan = self.run_tool("scan", "--input", self.input)
        self.assertTrue(scan["ready"])
        self.assertEqual(scan["included_file_count"], 3)
        self.assertEqual(scan["duplicate_file_count"], 3)
        self.assertEqual(scan["invalid_file_count"], 1)
        self.assertEqual(scan["component_count"], 6)
        output = self.root / "LTX package.safetensors"
        report = self.run_tool("create", "--input", self.input, "--output", output)
        self.assertTrue(report["verified"])
        self.assertEqual(report["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        header, base = read_header(output)
        self.assertEqual(header["model.diffusion_model.block.weight"]["dtype"], "F8_E4M3")
        self.assertEqual(header["__metadata__"]["license"], "fixture-license")
        manifest = json.loads(header["__metadata__"]["iild_package_manifest"])
        self.assertEqual(manifest["schema"], "iild-safetensors-package-v1")
        text_encoder = next(f for f in manifest["files"] if f["format"] == "gguf")
        tensor = header[text_encoder["tensors"][0]["output_key"]]
        self.assertEqual(tensor["dtype"], "U8")
        start, end = tensor["data_offsets"]
        self.assertEqual(output.read_bytes()[base + start:base + end], original["gemma-3.gguf"])
        restored = self.root / "restored"
        self.run_tool("extract", "--input", output, "--output", restored)
        for name, contents in original.items():
            if name != "ltx-failed.safetensors":
                self.assertEqual((restored / name).read_bytes(), contents, name)
        self.assertFalse((restored / "ltx-failed.safetensors").exists())
        self.assertEqual({p.name: p.read_bytes() for p in self.input.iterdir()}, original)

    def test_optional_component_exclusion_is_reflected_everywhere(self):
        fixture(self.input)
        excluded = "spatial_upscaler_x2.safetensors"
        report = self.run_tool("scan", "--input", self.input, "--exclude", excluded)
        self.assertEqual(report["included_file_count"], 2)
        self.assertEqual(report["component_count"], 5)
        output = self.root / "core.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output, "--exclude", excluded)
        self.run_tool("extract", "--input", output, "--output", self.root / "core")
        self.assertFalse((self.root / "core" / excluded).exists())

    def test_missing_required_encoder_blocks_creation(self):
        fixture(self.input, encoder=False)
        report = self.run_tool("scan", "--input", self.input)
        self.assertFalse(report["ready"])
        self.assertIn("text encoder", " ".join(report["errors"]).lower())
        output = self.root / "missing.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output, success=False)
        self.assertFalse(output.exists())

    def test_nonidentical_embedded_component_blocks_and_never_averages(self):
        fixture(self.input)
        safetensors(self.input / "video_vae.safetensors", [("decoder.weight", "BF16", [2], b"\xff\xff\xff\xff")])
        report = self.run_tool("scan", "--input", self.input)
        self.assertFalse(report["ready"])
        self.assertIn("conflict", " ".join(report["errors"]).lower())

    def test_generic_shards_lora_and_tokenizer_roundtrip(self):
        model = self.input / "transformer"
        model.mkdir()
        safetensors(model / "model-00001-of-00002.safetensors", [("layer.0", "F32", [1], struct.pack("<f", 1))])
        safetensors(model / "model-00002-of-00002.safetensors", [("layer.1", "I8", [2], b"\x01\x02")])
        (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer.0": "model-00001-of-00002.safetensors", "layer.1": "model-00002-of-00002.safetensors"}}))
        safetensors(self.input / "detail_lora.safetensors", [("lora_x.lora_A.weight", "F16", [1], b"\x01\x02")])
        (self.input / "tokenizer.json").write_text('{"vocab":{"hello":1}}')
        (self.input / "config.json").write_text('{"family":"example"}')
        output = self.root / "shards.safetensors"
        report = self.run_tool("create", "--input", self.input, "--output", output)
        self.assertTrue(report["verified"])
        header, _ = read_header(output)
        self.assertIn("layer.0", header)
        self.assertIn("layer.1", header)
        self.run_tool("extract", "--input", output, "--output", self.root / "restored")
        for source in self.input.rglob("*"):
            if source.is_file():
                self.assertEqual(source.read_bytes(), (self.root / "restored" / source.relative_to(self.input)).read_bytes())

    def test_missing_shard_and_corrupt_header_are_reported(self):
        (self.input / "model.safetensors.index.json").write_text('{"weight_map":{"a":"missing.safetensors"}}')
        report = self.run_tool("scan", "--input", self.input)
        self.assertFalse(report["ready"])
        self.assertIn("shard", " ".join(report["errors"]).lower())

    def test_gguf_tensor_extent_is_checked_without_dequantization(self):
        path = self.input / "gemma.gguf"
        gguf(path)
        path.write_bytes(path.read_bytes()[:-1])
        report = self.run_tool("scan", "--input", self.input)
        self.assertFalse(report["ready"])
        self.assertEqual(report["invalid_file_count"], 1)

    def test_invalid_indexes_and_config_only_required_components_block(self):
        safetensors(self.input / "unet/model.safetensors", [("weight", "F32", [1], struct.pack("<f", 1))])
        (self.input / "model.safetensors.index.json").write_bytes(b"{broken")
        report = self.run_tool("scan", "--input", self.input)
        self.assertFalse(report["ready"])
        self.assertIn("index", " ".join(report["errors"]).lower())
        (self.input / "model.safetensors.index.json").unlink()
        (self.input / "model_index.json").write_text(json.dumps({"_class_name": "FixturePipeline", "unet": ["diffusers", "UNet"], "vae": ["diffusers", "VAE"]}))
        (self.input / "vae").mkdir()
        (self.input / "vae/config.json").write_text("{}")
        report = self.run_tool("scan", "--input", self.input)
        self.assertFalse(report["ready"])
        self.assertIn("vae", " ".join(report["errors"]).lower())
        self.assertFalse(next(file for file in report["files"] if file["relative_path"] == "vae/config.json")["optional"])

    def test_output_is_never_overwritten_and_corruption_is_detected(self):
        fixture(self.input)
        output = self.root / "package.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output)
        original = output.read_bytes()
        self.run_tool("create", "--input", self.input, "--output", output, success=False)
        self.assertEqual(output.read_bytes(), original)
        with output.open("r+b") as file:
            file.seek(-1, 2)
            file.write(bytes([original[-1] ^ 1]))
        self.run_tool("verify", "--input", output, success=False)
        self.assertFalse(any(".partial-" in p.name for p in self.root.iterdir()))

    def test_source_symlink_retains_logical_name(self):
        safetensors(self.root / "weights.safetensors", [("network.weight", "F32", [1], struct.pack("<f", 1))])
        (self.input / "model.safetensors").symlink_to(self.root / "weights.safetensors")
        output = self.root / "symlink.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output)
        self.run_tool("extract", "--input", output, "--output", self.root / "restored")
        self.assertEqual((self.root / "restored" / "model.safetensors").read_bytes(), (self.root / "weights.safetensors").read_bytes())
        self.assertFalse((self.root / "restored" / "weights.safetensors").exists())

    def test_invalid_shape_offsets_duplicate_keys_and_path_traversal_fail(self):
        for header in [
                '{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}',
                '{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}']:
            raw = header.encode()
            (self.input / "bad.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
            self.assertFalse(self.run_tool("scan", "--input", self.input)["ready"])
        (self.input / "model.safetensors.index.json").write_text('{"weight_map":{"x":"../outside.safetensors"}}')
        self.assertFalse(self.run_tool("scan", "--input", self.input)["ready"])

    @unittest.skipIf(os.name == "nt", "POSIX cancellation signal")
    def test_cancellation_cleans_temporary_output_and_keeps_sources(self):
        size = 32 * 1024 * 1024
        safetensors(self.input / "model.safetensors", [("weight", "U8", [size], b"\x11" * size)])
        output = self.root / "cancelled.safetensors"
        process = subprocess.Popen([str(PROGRAM), "create", "--input", str(self.input), "--output", str(output)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            event = json.loads(line)
            if event.get("phase") == "write":
                process.send_signal(signal.SIGTERM)
                break
        stdout, stderr = process.communicate(timeout=15)
        self.assertNotEqual(process.returncode, 0, stdout + stderr)
        self.assertFalse(output.exists())
        self.assertFalse(any(".partial-" in p.name for p in self.root.iterdir()))
        self.assertEqual((self.input / "model.safetensors").stat().st_size, size + read_header(self.input / "model.safetensors")[1])


if __name__ == "__main__":
    unittest.main()
