"""Exercise native packaging with real file formats and independent byte checks."""
import resource_fixture
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
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
        safetensors(folder / "text_encoder.safetensors",
                    [("text_model.encoder.weight", "F32", [1], struct.pack("<f", 0.25))])
    return model


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = resource_fixture.TemporaryDirectory(prefix="model-packaging-test-")
        self.root = Path(self.temporary.name)
        self.input = self.root / "input"
        self.input.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def run_tool(self, operation, *args, success=True):
        result = subprocess.run([str(PROGRAM), operation, *map(str, args)], text=True,
                                capture_output=True, timeout=30)
        reports = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertTrue(reports, f"exit={result.returncode:#x}; stdout={result.stdout!r}; stderr={result.stderr!r}")
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return reports[-1]

    def test_python_api_packaging_selection(self):
        spec = importlib.util.spec_from_file_location("iild_model_packaging",
            Path(__file__).resolve().parents[1] / "src/ModelPackaging/python/iild_model_packaging.py")
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        main = self.input / "Checkpoint" / "main.safetensors"
        safetensors(main, [("model.diffusion_model.weight", "BF16", [1], b"\x01\x02")])
        safetensors(self.input / "Checkpoint" / "other.safetensors", [("other", "BF16", [1], b"\x03\x04")])
        selected = ["Checkpoint/main.safetensors"]
        report = api.scan_selection(self.input, selected, executable=PROGRAM)
        self.assertEqual([item["relative_path"] for item in report["files"]], selected)
        output = self.root / "python.safetensors"
        self.assertTrue(api.create_selection(self.input, selected, output, executable=PROGRAM)["verified"])
        with self.assertRaises(ValueError):
            api.scan_selection(self.input, [], executable=PROGRAM)
        with self.assertRaises(api.ModelPackagingError):
            api.scan_selection(self.input, ["../outside.safetensors"], executable=PROGRAM)

    def test_manual_roles_and_main_position_are_packaged_and_survive_calibration(self):
        large = self.input / "Checkpoint" / "large.safetensors"
        small = self.input / "Checkpoint" / "small.safetensors"
        encoder = self.input / "Other" / "ambiguous.safetensors"
        safetensors(large, [("large.weight", "BF16", [100], b"\x01\x02" * 100)])
        safetensors(small, [("small.weight", "BF16", [1], b"\x03\x04")])
        safetensors(encoder, [("encoder.weight", "BF16", [1], b"\x05\x06")])
        options = ("--assign-position", "Checkpoint/small.safetensors=main",
                   "--assign-role", "Other/ambiguous.safetensors=text_encoder")
        report = self.run_tool("scan", "--input", self.input, *options)
        self.assertTrue(report["ready"], report["errors"])
        rows = {entry["relative_path"]: entry for entry in report["files"]}
        self.assertEqual(rows["Checkpoint/small.safetensors"]["position"], "main")
        self.assertEqual(rows["Checkpoint/large.safetensors"]["position"], "component")
        self.assertEqual(rows["Other/ambiguous.safetensors"]["role"], "text_encoder")
        self.assertTrue(any(c["id"] == "Other/ambiguous.safetensors" and c["label"] == "Text encoder" for c in report["components"]))
        output = self.root / "manual.safetensors"
        self.assertTrue(self.run_tool("create", "--input", self.input, "--output", output, *options)["verified"])
        header, _ = read_header(output)
        self.assertIn("small.weight", header)
        self.assertIn("components/model/Checkpoint/large.safetensors/large.weight", header)
        self.assertIn("components/text_encoder/Other/ambiguous.safetensors/encoder.weight", header)
        manifest = json.loads(header["__metadata__"]["iild_package_manifest"])
        rows = {entry["path"]: entry for entry in manifest["files"]}
        self.assertEqual(rows["Checkpoint/small.safetensors"]["position_source"], "user")
        self.assertEqual(rows["Other/ambiguous.safetensors"]["role_source"], "user")
        restored = self.root / "restored-manual"
        self.run_tool("extract", "--input", output, "--output", restored)
        for source in (large, small, encoder):
            self.assertEqual((restored / source.relative_to(self.input)).read_bytes(), source.read_bytes())
        calibrated = self.root / "calibrated-manual.safetensors"
        self.run_tool("calibrate", "--input", output, "--output", calibrated)
        calibrated_header, _ = read_header(calibrated)
        self.assertIn("small.weight", calibrated_header)
        self.assertIn("components/text_encoder/Other/ambiguous.safetensors/encoder.weight", calibrated_header)

    def test_manual_assignments_validate_and_python_api_forwards_them(self):
        help_result = subprocess.run([str(PROGRAM)], text=True, capture_output=True, timeout=30)
        self.assertIn("--assign-position", json.loads(help_result.stdout)["error"])
        safetensors(self.input / "a.safetensors", [("a", "BF16", [1], b"\x01\x02")])
        safetensors(self.input / "b.safetensors", [("b", "BF16", [1], b"\x03\x04")])
        bad = [
            ("--assign-role", "../a.safetensors=model"),
            ("--assign-role", "missing.safetensors=model"),
            ("--assign-role", "a.safetensors=imaginary"),
            ("--assign-position", "a.safetensors=front"),
            ("--assign-role", "a.safetensors=text_encoder", "--assign-position", "a.safetensors=main"),
            ("--assign-position", "a.safetensors=main", "--assign-position", "b.safetensors=main"),
        ]
        for options in bad:
            with self.subTest(options=options):
                output = self.root / "invalid.safetensors"
                self.run_tool("create", "--input", self.input, "--output", output, *options, success=False)
                self.assertFalse(output.exists())
        no_main = self.run_tool("scan", "--input", self.input,
                                "--assign-position", "a.safetensors=component",
                                "--assign-position", "b.safetensors=component")
        self.assertFalse(no_main["ready"])
        self.assertIn("main model", " ".join(no_main["errors"]).lower())
        spec = importlib.util.spec_from_file_location("iild_model_packaging", Path(__file__).resolve().parents[1] / "src/ModelPackaging/python/iild_model_packaging.py")
        api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
        choices = {"b.safetensors": {"position": "main"}}
        report = api.scan_selection(self.input, ["a.safetensors", "b.safetensors"], assignments=choices, executable=PROGRAM)
        self.assertEqual(next(f for f in report["files"] if f["id"] == "b.safetensors")["position"], "main")

    def test_manual_main_position_applies_to_whole_sharded_model(self):
        safetensors(self.input / "a.safetensors", [("a", "BF16", [1], b"\x01\x02")])
        safetensors(self.input / "b.safetensors", [("b", "BF16", [1], b"\x03\x04")])
        safetensors(self.input / "larger.safetensors", [("big", "BF16", [100], b"\x05\x06" * 100)])
        (self.input / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
        report = self.run_tool("scan", "--input", self.input, "--assign-position", "a.safetensors=main")
        positions = {f["id"]: f["position"] for f in report["files"]}
        self.assertEqual(positions["a.safetensors"], "main")
        self.assertEqual(positions["b.safetensors"], "main")
        self.assertEqual(positions["larger.safetensors"], "component")
        self.run_tool("create", "--input", self.input, "--output", self.root / "bad-shards.safetensors",
                      "--assign-role", "a.safetensors=model", "--assign-role", "b.safetensors=text_encoder", success=False)

    def test_excluding_a_manually_optional_shard_excludes_its_whole_group(self):
        safetensors(self.input / "a.safetensors", [("a", "BF16", [1], b"\x01\x02")])
        safetensors(self.input / "b.safetensors", [("b", "BF16", [1], b"\x03\x04")])
        safetensors(self.input / "main.safetensors", [("main", "BF16", [1], b"\x05\x06")])
        (self.input / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
        options = ("--assign-role", "a.safetensors=lora", "--exclude", "a.safetensors")
        report = self.run_tool("scan", "--input", self.input, *options)
        self.assertTrue(report["ready"], report["errors"])
        rows = {f["id"]: f for f in report["files"]}
        for name in ("a.safetensors", "b.safetensors", "model.safetensors.index.json"):
            self.assertEqual(rows[name]["status"], "excluded")
        output = self.root / "without-shards.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output, *options)
        restored = self.root / "without-shards"
        self.run_tool("extract", "--input", output, "--output", restored)
        self.assertTrue((restored / "main.safetensors").exists())
        self.assertFalse((restored / "a.safetensors").exists())
        self.assertFalse((restored / "b.safetensors").exists())
        self.assertFalse((restored / "model.safetensors.index.json").exists())

    def test_inconsistent_detected_shard_roles_require_one_explicit_role(self):
        safetensors(self.input / "a.safetensors", [("layer.lora_down.weight", "BF16", [1], b"\x01\x02")])
        safetensors(self.input / "b.safetensors", [("layer.weight", "BF16", [1], b"\x03\x04")])
        (self.input / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "layer.lora_down.weight": "a.safetensors", "layer.weight": "b.safetensors"}}))
        report = self.run_tool("scan", "--input", self.input)
        self.assertFalse(report["ready"])
        self.assertIn("sharded", " ".join(report["errors"]).lower())
        report = self.run_tool("scan", "--input", self.input, "--assign-role", "a.safetensors=model")
        self.assertTrue(report["ready"], report["errors"])
        for file in report["files"]:
            if file["id"] in ("a.safetensors", "b.safetensors"):
                self.assertEqual(file["position"], "main")

    def test_secondary_ltx_does_not_impose_its_requirements_on_a_different_main(self):
        safetensors(self.input / "ltx.safetensors", [("ltx.weight", "BF16", [10], b"\x01\x02" * 10)],
                    {"model_version": "2.3.0", "config": '{"_class_name":"LTXModel"}'})
        safetensors(self.input / "other.safetensors", [("other.weight", "BF16", [1], b"\x03\x04")])
        report = self.run_tool("scan", "--input", self.input, "--assign-position", "other.safetensors=main")
        self.assertTrue(report["ready"], report["errors"])
        self.assertEqual(report["family"], "Model")

    def test_automatic_gguf_main_survives_an_unrelated_role_override(self):
        gguf(self.input / "encoder.gguf"); gguf(self.input / "main.gguf")
        report = self.run_tool("scan", "--input", self.input, "--assign-role", "encoder.gguf=text_encoder")
        self.assertTrue(report["ready"], report["errors"])
        rows = {f["id"]: f for f in report["files"]}
        self.assertEqual(rows["main.gguf"]["position"], "main")
        self.assertEqual(rows["encoder.gguf"]["position"], "component")

    def test_selected_models_from_multiple_categories_roundtrip_without_other_files(self):
        checkpoint = self.input / "Checkpoint" / "main.safetensors"
        vae = self.input / "VAE" / "decoder.safetensors"
        encoder = self.input / "LLM" / "encoder.gguf"
        safetensors(checkpoint, [("model.diffusion_model.weight", "BF16", [1], b"\x01\x02")])
        safetensors(vae, [("decoder.weight", "BF16", [1], b"\x03\x04")])
        encoder.parent.mkdir()
        gguf(encoder)
        safetensors(self.input / "Checkpoint" / "unselected.safetensors", [("other", "BF16", [1], b"\x05\x06")])
        (checkpoint.parent / (checkpoint.name + ".iisacc.json")).write_text('{"private":true}')
        selected = ["Checkpoint/main.safetensors", "VAE/decoder.safetensors", "LLM/encoder.gguf"]
        options = [part for path in selected for part in ("--select", path)]
        report = self.run_tool("scan", "--input", self.input, *options)
        self.assertTrue(report["ready"], report["errors"])
        self.assertEqual({item["relative_path"] for item in report["files"]}, set(selected))
        output = self.root / "selected.safetensors"
        self.assertTrue(self.run_tool("create", "--input", self.input, *options, "--output", output)["verified"])
        restored = self.root / "restored"
        self.run_tool("extract", "--input", output, "--output", restored)
        for path in selected:
            self.assertEqual((restored / path).read_bytes(), (self.input / path).read_bytes())
        self.assertFalse((restored / "Checkpoint" / "unselected.safetensors").exists())
        self.assertFalse((restored / "Checkpoint" / (checkpoint.name + ".iisacc.json")).exists())

    def test_selected_models_reject_paths_outside_the_library_or_redirected(self):
        model = self.input / "model.safetensors"
        safetensors(model, [("model.weight", "BF16", [1], b"\x01\x02")])
        outside = self.root / "outside.safetensors"
        safetensors(outside, [("model.weight", "BF16", [1], b"\x01\x02")])
        (self.input / "model.safetensors.iisacc.json").write_text('{}')
        (self.input / "redirect.safetensors").symlink_to(outside)
        for selected in ("../outside.safetensors", "redirect.safetensors", "model.safetensors.iisacc.json"):
            with self.subTest(selected=selected):
                self.run_tool("scan", "--input", self.input, "--select", selected, success=False)

    def test_selected_directory_keeps_components_but_omits_provider_sidecars(self):
        package = self.input / "Checkpoint" / "Pipeline"
        model = package / "unet" / "model.safetensors"
        safetensors(model, [("model.diffusion_model.weight", "BF16", [1], b"\x01\x02")])
        (package / "model_index.json").write_text('{"_class_name":"Pipeline"}')
        (model.parent / (model.name + ".iisacc.json")).write_text('{"private":true}')
        report = self.run_tool("scan", "--input", self.input, "--select", "Checkpoint/Pipeline", "--select", "Checkpoint/Pipeline/unet/model.safetensors")
        names = [item["relative_path"] for item in report["files"]]
        self.assertTrue(report["ready"], report["errors"])
        self.assertEqual(names.count("Checkpoint/Pipeline/unet/model.safetensors"), 1)
        self.assertIn("Checkpoint/Pipeline/model_index.json", names)
        self.assertFalse(any(name.endswith(".iisacc.json") for name in names))

    def test_selected_safetensors_in_llm_category_are_text_encoder_components(self):
        safetensors(self.input / "Checkpoint" / "main.safetensors", [("model.diffusion_model.weight", "BF16", [1], b"\x01\x02")])
        safetensors(self.input / "LLM" / "encoder.safetensors", [("encoder.weight", "BF16", [1], b"\x03\x04")])
        report = self.run_tool("scan", "--input", self.input, "--select", "Checkpoint/main.safetensors", "--select", "LLM/encoder.safetensors")
        roles = {item["relative_path"]: item["role"] for item in report["files"]}
        self.assertEqual(roles["LLM/encoder.safetensors"], "text_encoder")

    def test_qwen_language_encoder_does_not_become_larger_moss_primary(self):
        # Synthetic tensors reproduce the role ambiguity without pretrained weights.
        main = self.input / "Other" / "moss.safetensors"
        encoder = self.input / "VLM" / "encoder.safetensors"
        safetensors(main, [("video_embedder.weight", "F32", [1], struct.pack("<f", 1)),
                           ("final_linear_video.weight", "F32", [1], struct.pack("<f", 2)),
                           ("layers.0.proj.qkv.weight", "F32", [1], struct.pack("<f", 3))])
        safetensors(encoder, [("model.language_model.embed_tokens.weight", "F32", [100],
                               struct.pack("<100f", *([0.5] * 100)))])
        report = self.run_tool("scan", "--input", self.input)
        roles = {item["relative_path"]: item["role"] for item in report["files"]}
        self.assertEqual(roles["Other/moss.safetensors"], "model")
        self.assertEqual(roles["VLM/encoder.safetensors"], "text_encoder")
        output = self.root / "moss.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output)
        header, _ = read_header(output)
        self.assertIn("video_embedder.weight", header)
        self.assertNotIn("model.language_model.embed_tokens.weight", header)

    def test_moss_primary_with_language_tensors_is_not_reclassified_as_encoder(self):
        safetensors(self.input / "model.safetensors", [
            ("video_embedder.weight", "F32", [1], struct.pack("<f", 1)),
            ("final_linear_video.weight", "F32", [1], struct.pack("<f", 2)),
            ("layers.0.proj.qkv.weight", "F32", [1], struct.pack("<f", 3)),
            ("model.language_model.embed_tokens.weight", "F32", [1], struct.pack("<f", 4))])
        report = self.run_tool("scan", "--input", self.input)
        self.assertEqual(report["files"][0]["role"], "model")

    def test_calibration_repackages_legacy_roles_without_changing_source_weights(self):
        main = self.input / "Other" / "moss.safetensors"
        encoder = self.input / "VLM" / "encoder.safetensors"
        safetensors(main, [("video_embedder.weight", "F32", [1], struct.pack("<f", 1)),
                           ("final_linear_video.weight", "F32", [1], struct.pack("<f", 2)),
                           ("layers.0.proj.qkv.weight", "F32", [1], struct.pack("<f", 3))])
        safetensors(encoder, [("model.language_model.embed_tokens.weight", "F32", [100],
                               struct.pack("<100f", *([0.5] * 100)))])
        valid = self.root / "valid.safetensors"
        self.run_tool("create", "--input", self.input, "--output", valid)
        header, base = read_header(valid)
        manifest = json.loads(header["__metadata__"]["iild_package_manifest"])
        for source in manifest["files"]:
            source["role"] = "model"
            for tensor in source["tensors"]:
                old = tensor["output_key"]
                new = ("components/model/" + source["path"] + "/" + tensor["key"]
                       if source["path"] == "Other/moss.safetensors" else tensor["key"])
                header[new] = header.pop(old)
                tensor["output_key"] = new
        header["__metadata__"]["iild_package_manifest"] = json.dumps(manifest, separators=(",", ":"))
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        legacy = self.root / "legacy.safetensors"
        legacy.write_bytes(struct.pack("<Q", len(encoded)) + encoded + valid.read_bytes()[base:])
        self.assertTrue(self.run_tool("verify", "--input", legacy)["verified"])
        original = legacy.read_bytes()
        cache = self.root / "selected-components"
        for relative, expected in (("Other/moss.safetensors", main), ("VLM/encoder.safetensors", encoder)):
            report = self.run_tool("materialize", "--input", legacy, "--output", cache,
                                   "--asset-path", relative)
            self.assertEqual(Path(report["output"]).read_bytes(), expected.read_bytes())
        self.run_tool("materialize", "--input", legacy, "--output", cache,
                      "--asset-path", "../outside.safetensors", success=False)
        spec = importlib.util.spec_from_file_location("iild_model_packaging",
            Path(__file__).resolve().parents[1] / "src/ModelPackaging/python/iild_model_packaging.py")
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        result = api.materialize_package_asset(legacy, cache, relative_source="Other/moss.safetensors", executable=PROGRAM)
        self.assertEqual(Path(result["output"]).read_bytes(), main.read_bytes())
        with self.assertRaises(ValueError):
            api.materialize_package_asset(legacy, cache, role="model", relative_source="Other/moss.safetensors", executable=PROGRAM)
        corrected = self.root / "corrected.safetensors"
        report = self.run_tool("calibrate", "--input", legacy, "--output", corrected)
        self.assertTrue(report["verified"])
        self.assertEqual(report["operation"], "calibrate")
        result, _ = read_header(corrected)
        self.assertIn("video_embedder.weight", result)
        self.assertNotIn("model.language_model.embed_tokens.weight", result)
        self.assertEqual(legacy.read_bytes(), original)
        restored = self.root / "restored-calibrated"
        self.run_tool("extract", "--input", corrected, "--output", restored)
        self.assertEqual((restored / "Other/moss.safetensors").read_bytes(), main.read_bytes())
        self.assertEqual((restored / "VLM/encoder.safetensors").read_bytes(), encoder.read_bytes())
        self.run_tool("calibrate", "--input", legacy, "--output", corrected, success=False)
        self.assertFalse(any(".partial-" in p.name for p in self.root.iterdir()))

    def test_weight_health_counts_nonfinite_values_and_skips_quantized_data(self):
        model = self.input / "model.safetensors"
        safetensors(model, [
            ("fp32", "F32", [3], struct.pack("<fff", 1, float("nan"), float("inf"))),
            ("fp16", "F16", [2], struct.pack("<HH", 0x3c00, 0xfc00)),
            ("bf16", "BF16", [2], struct.pack("<HH", 0x3f80, 0x7fc0)),
            ("fp64", "F64", [2], struct.pack("<dd", 1, -float("inf"))),
            ("int8", "I8", [2], b"\x7f\x80")])
        before = model.read_bytes()
        report = self.run_tool("weight-health", "--input", model)
        self.assertEqual(report["checked_tensors"], 4)
        self.assertEqual(report["checked_values"], 9)
        self.assertEqual(report["nonfinite_values"], 5)
        self.assertEqual(report["unchecked_tensors"], 1)
        self.assertEqual(model.read_bytes(), before)

    def test_calibration_repairs_nonfinite_weights_only_when_requested(self):
        main = self.input / "main.safetensors"
        safetensors(main, [("model.diffusion_model.weight", "F32", [3],
                            struct.pack("<fff", 1, float("nan"), float("inf")))])
        package = self.root / "package.safetensors"
        self.run_tool("create", "--input", self.input, "--output", package)
        original = package.read_bytes()
        output = self.root / "repaired.safetensors"
        report = self.run_tool("calibrate", "--input", package, "--output", output,
                               "--repair-nonfinite", "true")
        self.assertTrue(report["weights_modified"])
        self.assertEqual(report["nonfinite_values_replaced"], 2)
        restored = self.root / "repaired-sources"
        self.run_tool("extract", "--input", output, "--output", restored)
        header, base = read_header(restored / "main.safetensors")
        data = (restored / "main.safetensors").read_bytes()
        self.assertEqual(struct.unpack("<fff", data[base:base + 12]), (1, 0, 0))
        self.assertEqual(package.read_bytes(), original)

    def test_standalone_numeric_calibration_keeps_source_and_refuses_package(self):
        main = self.input / "main.safetensors"
        safetensors(main, [("model.diffusion_model.weight", "F32", [2],
                            struct.pack("<ff", 1, float("nan")))])
        before = main.read_bytes()
        output = self.root / "standalone-repaired.safetensors"
        report = self.run_tool("calibrate-file", "--input", main, "--output", output,
                               "--repair-nonfinite", "true")
        self.assertEqual(report["nonfinite_values_replaced"], 1)
        self.assertEqual(main.read_bytes(), before)
        self.assertEqual(self.run_tool("weight-health", "--input", output)["nonfinite_values"], 0)
        spec = importlib.util.spec_from_file_location("iild_model_packaging",
            Path(__file__).resolve().parents[1] / "src/ModelPackaging/python/iild_model_packaging.py")
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        self.assertEqual(api.inspect_weights(main, executable=PROGRAM)["nonfinite_values"], 1)
        python_output = self.root / "python-repaired.safetensors"
        self.assertTrue(api.repair_standalone_weights(main, python_output, executable=PROGRAM)["verified"])
        package = self.root / "packaged.safetensors"
        self.run_tool("create", "--input", self.input, "--output", package)
        rejected = self.root / "invalid.safetensors"
        self.run_tool("calibrate-file", "--input", package, "--output", rejected,
                      "--repair-nonfinite", "true", success=False)
        self.assertFalse(rejected.exists())

    def test_calibration_cancel_keeps_source_and_removes_workspace(self):
        size = 32 * 1024 * 1024
        safetensors(self.input / "main.safetensors", [("model.diffusion_model.weight", "U8", [size], b"\x11" * size)])
        package = self.root / "package.safetensors"
        self.run_tool("create", "--input", self.input, "--output", package)
        original = hashlib.sha256(package.read_bytes()).hexdigest()
        output = self.root / "cancelled-calibration.safetensors"
        process = subprocess.Popen([str(PROGRAM), "calibrate", "--input", str(package), "--output", str(output)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            if json.loads(line).get("phase") == "verify":
                process.send_signal(signal.SIGTERM)
                break
        stdout, stderr = process.communicate(timeout=30)
        self.assertNotEqual(process.returncode, 0, stdout + stderr)
        self.assertFalse(output.exists())
        self.assertFalse(any(".partial-" in p.name for p in self.root.iterdir()))
        self.assertEqual(hashlib.sha256(package.read_bytes()).hexdigest(), original)

    def test_selected_diffusers_directory_validates_its_declared_components(self):
        package = self.input / "Checkpoint" / "Pipeline"
        safetensors(package / "unet" / "model.safetensors", [("model.diffusion_model.weight", "BF16", [1], b"\x01\x02")])
        (package / "model_index.json").write_text(json.dumps({
            "_class_name": "Pipeline", "unet": ["diffusers", "UNet"], "vae": ["diffusers", "VAE"]}))
        options = ("--input", self.input, "--select", "Checkpoint/Pipeline")
        report = self.run_tool("scan", *options)
        self.assertFalse(report["ready"])
        self.assertIn("vae", " ".join(report["errors"]).lower())
        output = self.root / "incomplete.safetensors"
        self.run_tool("create", *options, "--output", output, success=False)
        self.assertFalse(output.exists())
        safetensors(package / "vae" / "decoder.safetensors", [("decoder.weight", "BF16", [1], b"\x03\x04")])
        report = self.run_tool("scan", *options)
        self.assertTrue(report["ready"], report["errors"])
        self.assertTrue(self.run_tool("create", *options, "--output", output)["verified"])

    def test_selected_unified_directory_is_not_treated_as_a_diffusers_index(self):
        package = self.input / "Checkpoint" / "legacy.iildmodel"
        safetensors(package / "model.safetensors", [("model.diffusion_model.weight", "BF16", [1], b"\x01\x02")])
        (package / "model_index.json").write_text(json.dumps({"schema": "iild-unified-model-v1",
            "stages": [{"model": "model.safetensors"}, {"model": "model.safetensors"}]}))
        report = self.run_tool("scan", "--input", self.input, "--select", "Checkpoint/legacy.iildmodel")
        self.assertTrue(report["ready"], report["errors"])

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
        text_encoder = next(f for f in manifest["files"] if f["role"] == "text_encoder")
        self.assertEqual(text_encoder["format"], "safetensors")
        tensor = header[text_encoder["tensors"][0]["output_key"]]
        self.assertEqual(tensor["dtype"], "F32")
        start, end = tensor["data_offsets"]
        self.assertEqual(output.read_bytes()[base + start:base + end], struct.pack("<f", 0.25))
        restored = self.root / "restored"
        self.run_tool("extract", "--input", output, "--output", restored)
        for name, contents in original.items():
            if name != "ltx-failed.safetensors":
                self.assertEqual((restored / name).read_bytes(), contents, name)
        self.assertFalse((restored / "ltx-failed.safetensors").exists())
        self.assertEqual({p.name: p.read_bytes() for p in self.input.iterdir()}, original)

    def test_materialize_only_verified_encoder_and_detect_cache_corruption(self):
        fixture(self.input)
        safetensors(self.input / "text_encoder.safetensors", [
            ("text_model.encoder.weight", "F32", [1], struct.pack("<f", 0.25)),
            ("text_model.encoder.bias", "F16", [2], b"\x01\x02\x03\x04")])
        output = self.root / "package.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output)
        before = hashlib.sha256(output.read_bytes()).hexdigest()
        cache = self.root / "assets"
        report = self.run_tool("materialize", "--input", output, "--output", cache)
        encoder = Path(report["output"])
        self.assertEqual(encoder.suffix, ".safetensors")
        self.assertEqual(encoder.read_bytes(), (self.input / "text_encoder.safetensors").read_bytes())
        self.assertEqual(list(cache.iterdir()), [encoder])
        self.assertEqual(self.run_tool("materialize", "--input", output, "--output", cache)["output"], str(encoder))
        encoder.write_bytes(b"x" * encoder.stat().st_size)
        self.run_tool("materialize", "--input", output, "--output", cache, success=False)
        self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), before)

    def test_materialize_verifies_selected_blob_without_hashing_other_weights(self):
        fixture(self.input)
        output = self.root / "package.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output)
        header, base = read_header(output)
        key = next(key for key in header if key.startswith("components/text_encoder/"))
        # Corrupt the chosen blob; no cache result may be published.
        with output.open("r+b") as file:
            file.seek(base + header[key]["data_offsets"][0])
            file.write(b"BAD!")
        cache = self.root / "assets"
        self.run_tool("materialize", "--input", output, "--output", cache, success=False)
        self.assertEqual(list(cache.iterdir()), [])

    def test_materialize_does_not_choose_one_of_multiple_encoder_shards(self):
        fixture(self.input)
        safetensors(self.input / "text_encoder_second.safetensors", [
            ("text_model.second.weight", "F32", [1], struct.pack("<f", 0.5))])
        output = self.root / "package.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output)
        before = output.read_bytes()
        cache = self.root / "assets"
        report = self.run_tool("materialize", "--input", output, "--output", cache, success=False)
        self.assertIn("multiple", report["error"].lower())
        self.assertFalse(cache.exists())
        self.assertEqual(output.read_bytes(), before)

    def test_materialize_rejects_hash_matching_but_invalid_cached_safetensors(self):
        fixture(self.input)
        output = self.root / "package.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output)
        header, base = read_header(output)
        manifest = json.loads(header["__metadata__"]["iild_package_manifest"])
        encoder = next(f for f in manifest["files"] if f["role"] == "text_encoder")
        mapped = encoder["tensors"][0]
        tensor = header[mapped["output_key"]]
        start, end = tensor["data_offsets"]
        payload = output.read_bytes()[base + start:base + end]
        bad_header = '{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}'
        invalid = struct.pack("<Q", len(bad_header)) + bad_header.encode() + payload
        encoder["header"] = bad_header
        encoder["size"] = len(invalid)
        encoder["sha256"] = hashlib.sha256(invalid).hexdigest()
        header["__metadata__"]["iild_package_manifest"] = json.dumps(manifest, separators=(",", ":"))
        tensors = [(name, info["dtype"], info["shape"],
                    output.read_bytes()[base + info["data_offsets"][0]:base + info["data_offsets"][1]])
                   for name, info in header.items() if name != "__metadata__"]
        altered = self.root / "altered.safetensors"
        safetensors(altered, tensors, header["__metadata__"])
        cache = self.root / "assets"
        cache.mkdir()
        cached = cache / (encoder["sha256"] + ".safetensors")
        cached.write_bytes(invalid)
        self.run_tool("materialize", "--input", altered, "--output", cache, success=False)
        self.assertEqual(cached.read_bytes(), invalid)

    def test_materialize_recovers_aborted_copy_without_removing_active_owner(self):
        fixture(self.input)
        output = self.root / "package.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output)
        cache = self.root / "assets"
        encoder = Path(self.run_tool("materialize", "--input", output, "--output", cache)["output"])
        finished = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                                  capture_output=True, text=True, check=True)
        stale = cache / f".{encoder.name}.partial-{int(finished.stdout.strip())}-{'a' * 32}"
        active = cache / f".{encoder.name}.partial-{os.getpid()}-{'b' * 32}"
        unrelated = cache / "user-file.partial-unknown"
        for path in (stale, active, unrelated):
            path.write_bytes(b"preserve active or unrelated data")
        self.run_tool("materialize", "--input", output, "--output", cache)
        self.assertFalse(stale.exists())
        self.assertEqual(active.read_bytes(), b"preserve active or unrelated data")
        self.assertEqual(unrelated.read_bytes(), b"preserve active or unrelated data")
        self.assertEqual(encoder.read_bytes(), (self.input / "text_encoder.safetensors").read_bytes())

    def test_successful_readonly_commands_always_emit_final_report(self):
        fixture(self.input)
        output = self.root / "package.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output)
        for iteration in range(16):
            with self.subTest(iteration=iteration):
                materialized = self.run_tool("materialize", "--input", output, "--output", self.root / "assets")
                self.assertTrue(materialized["verified"])
                inspected = self.run_tool("inspect", "--input", output)
                self.assertEqual(inspected["schema"], "iild-model-inspection-v1")

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

    def test_onnx_is_rejected_without_modifying_sources(self):
        safetensors(self.input / "model.safetensors", [("weight", "F32", [1], struct.pack("<f", 1))])
        for name in ("weights.onnx", "weights.ONNX"):
            with self.subTest(name=name):
                path = self.input / name
                path.write_bytes(b"do not wrap or delete this model")
                report = self.run_tool("scan", "--input", self.input)
                self.assertFalse(report["ready"])
                self.assertIn("onnx", " ".join(report["errors"]).lower())
                output = self.root / "forbidden.safetensors"
                self.run_tool("create", "--input", self.input, "--output", output, success=False)
                self.assertFalse(output.exists())
                self.assertEqual(path.read_bytes(), b"do not wrap or delete this model")
                path.unlink()

    def test_legacy_package_cannot_embed_onnx(self):
        manifest = {"schema": "iild-safetensors-package-v1", "files": [{
            "path": "text_encoder.onnx", "format": "asset", "status": "included",
            "role": "text_encoder"}]}
        legacy = self.root / "legacy.safetensors"
        safetensors(legacy, [("__iild_assets__/text_encoder/text_encoder.onnx", "U8", [4], b"onnx")],
                    {"iild_package_schema": "iild-safetensors-package-v1",
                     "iild_package_manifest": json.dumps(manifest)})
        original = legacy.read_bytes()
        self.run_tool("verify", "--input", legacy, success=False)
        self.run_tool("materialize", "--input", legacy, "--output", self.root / "cache", success=False)
        self.run_tool("extract", "--input", legacy, "--output", self.root / "extract", success=False)
        self.assertFalse((self.root / "cache").exists())
        self.assertFalse((self.root / "extract").exists())
        self.assertEqual(legacy.read_bytes(), original)
        (self.input / "legacy.safetensors").write_bytes(original)
        safetensors(self.input / "model.safetensors", [("weight", "F32", [1], struct.pack("<f", 1))])
        self.assertFalse(self.run_tool("scan", "--input", self.input)["ready"])

    def test_gguf_encoder_roundtrip_materialization_and_cache_integrity(self):
        fixture(self.input, encoder=False)
        source = self.input / "gemma-3.gguf"
        gguf(source)
        (self.input / "tokenizer.bin").write_bytes(b"existing supporting asset")
        self.assertTrue(self.run_tool("scan", "--input", self.input)["ready"])
        package = self.root / "gguf-package.safetensors"
        self.run_tool("create", "--input", self.input, "--output", package)
        self.run_tool("verify", "--input", package)
        restored = self.root / "restored-gguf"
        self.run_tool("extract", "--input", package, "--output", restored)
        self.assertEqual((restored / source.name).read_bytes(), source.read_bytes())
        self.assertEqual((restored / "tokenizer.bin").read_bytes(), b"existing supporting asset")
        cache = self.root / "gguf-cache"
        encoder = Path(self.run_tool("materialize", "--input", package, "--output", cache)["output"])
        self.assertEqual(encoder.suffix, ".gguf")
        self.assertEqual(encoder.read_bytes(), source.read_bytes())
        self.assertEqual(self.run_tool("materialize", "--input", package, "--output", cache)["output"], str(encoder))
        encoder.write_bytes(b"x" * encoder.stat().st_size)
        self.run_tool("materialize", "--input", package, "--output", cache, success=False)

    def test_materialize_rejects_hash_matching_but_invalid_gguf_cache(self):
        fixture(self.input, encoder=False)
        gguf(self.input / "text_encoder.gguf")
        package = self.root / "gguf-original.safetensors"
        self.run_tool("create", "--input", self.input, "--output", package)
        header, base = read_header(package)
        manifest = json.loads(header["__metadata__"]["iild_package_manifest"])
        encoder = next(f for f in manifest["files"] if f["role"] == "text_encoder")
        mapped = encoder["tensors"][0]
        invalid = b"GGUF" + b"x" * (encoder["size"] - 4)
        digest = hashlib.sha256(invalid).hexdigest()
        encoder["sha256"] = digest
        mapped["sha256"] = digest
        header["__metadata__"]["iild_package_manifest"] = json.dumps(manifest, separators=(",", ":"))
        payload = package.read_bytes()
        tensors = [(name, info["dtype"], info["shape"],
                    invalid if name == mapped["output_key"] else
                    payload[base + info["data_offsets"][0]:base + info["data_offsets"][1]])
                   for name, info in header.items() if name != "__metadata__"]
        altered = self.root / "invalid-gguf.safetensors"
        safetensors(altered, tensors, header["__metadata__"])
        cache = self.root / "invalid-gguf-cache"
        cache.mkdir()
        cached = cache / (digest + ".gguf")
        cached.write_bytes(invalid)
        self.run_tool("materialize", "--input", altered, "--output", cache, success=False)
        self.assertEqual(cached.read_bytes(), invalid)

    def test_gguf_only_main_model_and_truncation(self):
        path = self.input / "model.gguf"
        gguf(path)
        self.assertTrue(self.run_tool("scan", "--input", self.input)["ready"])
        package = self.root / "gguf-main.safetensors"
        self.run_tool("create", "--input", self.input, "--output", package)
        self.run_tool("extract", "--input", package, "--output", self.root / "gguf-main")
        self.assertEqual((self.root / "gguf-main/model.gguf").read_bytes(), path.read_bytes())
        path.write_bytes(path.read_bytes()[:-2])
        self.assertFalse(self.run_tool("scan", "--input", self.input)["ready"])

    def test_package_output_requires_standard_extension(self):
        safetensors(self.input / "model.safetensors", [("weight", "F32", [1], struct.pack("<f", 1))])
        for name in ("model.onnx", "model.gguf", "model.safetensor", "model"):
            with self.subTest(name=name):
                target = self.root / name
                self.run_tool("create", "--input", self.input, "--output", target, success=False)
                self.assertFalse(target.exists())

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
