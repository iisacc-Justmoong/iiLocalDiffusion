"""Exercise native packaging with real file formats and independent byte checks."""
import resource_fixture
import hashlib
import importlib.util
import json
import os
import pickle
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


def gguf(path, architecture="gemma3"):
    def string(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, 2)
    for key, value in [("general.architecture", architecture), ("general.name", "Fixture")]:
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

    def test_standalone_shard_index_inspection_validates_exact_coverage(self):
        first = self.input / 'first.safetensors'
        second = self.input / 'second.safetensors'
        safetensors(first, [('a.weight', 'F32', [1], bytes(4))], {'format': 'pt'})
        safetensors(second, [('b.weight', 'F16', [2], bytes(4))], {'format': 'pt'})
        originals = (first.read_bytes(), second.read_bytes())
        index = self.input / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'a.weight': first.name, 'b.weight': second.name}}))
        report = self.run_tool('inspect-index', '--input', index)
        self.assertEqual(report['schema'], 'iild-safetensors-index-v1')
        self.assertEqual({item['relative_path'] for item in report['shards']}, {first.name, second.name})
        self.assertEqual({item['name'] for item in report['tensors']}, {'a.weight', 'b.weight'})
        self.assertEqual(report['metadata']['format'], 'pt')
        self.assertEqual((first.read_bytes(), second.read_bytes()), originals)
        index.write_text(json.dumps({'weight_map': {'a.weight': first.name, 'b.weight': first.name}}))
        self.run_tool('inspect-index', '--input', index, success=False)
        index.write_text(json.dumps({'weight_map': {'a.weight': '../outside.safetensors'}}))
        self.run_tool('inspect-index', '--input', index, success=False)
        index.write_text('{"weight_map":{"a.weight":"first.safetensors","a.weight":"second.safetensors"}}')
        self.run_tool('inspect-index', '--input', index, success=False)
        index.write_text(json.dumps({'weight_map': {'a.weight': first.name}}))
        safetensors(first, [('a.weight', 'F32', [1], bytes(4)), ('unindexed.weight', 'F32', [1], bytes(4))])
        self.run_tool('inspect-index', '--input', index, success=False)

    def test_exact_tensor_name_probe_supports_files_shards_and_python(self):
        first = self.input / 'first.safetensors'
        second = self.input / 'second.safetensors'
        safetensors(first, [('alpha.weight', 'F32', [1], bytes(4))])
        safetensors(second, [('beta.weight', 'F32', [1], bytes(4))])
        index = self.input / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'alpha.weight': first.name, 'beta.weight': second.name}}))
        found = self.run_tool('contains-tensors', '--input', index, '--tensor', 'alpha.weight', '--tensor', 'beta.weight')
        self.assertEqual(found['schema'], 'iild-model-tensor-presence-v1')
        self.assertTrue(found['present'])
        partial = self.run_tool('contains-tensors', '--input', index, '--tensor', 'missing.weight', '--tensor', 'alpha.weight')
        self.assertFalse(partial['present'])
        self.assertEqual(partial['matches'], ['alpha.weight'])
        self.assertTrue(self.run_tool('contains-tensors', '--input', first, '--tensor', 'alpha.weight')['present'])
        self.assertFalse(self.run_tool('contains-tensors', '--input', first, '--tensor', 'beta.weight')['present'])
        gguf_file = self.input / 'encoder.gguf'
        gguf(gguf_file)
        self.assertTrue(self.run_tool('contains-tensors', '--input', gguf_file, '--tensor', 'token_embd.weight')['present'])
        self.run_tool('contains-tensors', '--input', index, success=False)
        spec = importlib.util.spec_from_file_location('iild_model_packaging',
            Path(__file__).resolve().parents[1] / 'src/ModelPackaging/python/iild_model_packaging.py')
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        self.assertTrue(api.contains_tensors(index, ['alpha.weight', 'beta.weight'], executable=PROGRAM)['present'])
        self.assertEqual(api.find_tensor_names(index, ['alpha.weight', 'missing.weight'], executable=PROGRAM), ['alpha.weight'])
        index.write_text(json.dumps({'weight_map': {'alpha.weight': '../outside.safetensors'}}))
        self.run_tool('contains-tensors', '--input', index, '--tensor', 'alpha.weight', success=False)

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
        self.assertTrue(api.contains_tensors(output, ['model.diffusion_model.weight'], executable=PROGRAM)['embedded_package'])
        self.assertFalse(api.contains_tensors(main, ['model.diffusion_model.weight'], executable=PROGRAM)['embedded_package'])
        with self.assertRaises(ValueError):
            api.scan_selection(self.input, [], executable=PROGRAM)
        with self.assertRaises(api.ModelPackagingError):
            api.scan_selection(self.input, ["../outside.safetensors"], executable=PROGRAM)

    def test_name_probe_can_inspect_unknown_dtype_without_claiming_packaging_support(self):
        source = self.input / 'future.safetensors'
        safetensors(source, [('model.weight', 'FUTURE8', [1], b'x')])
        original = source.read_bytes()
        result = self.run_tool('contains-tensors', '--input', source, '--tensor', 'model.weight')
        self.assertTrue(result['present'])
        self.assertFalse(result['embedded_package'])
        report = self.run_tool('scan', '--input', self.input)
        self.assertFalse(report['ready'])
        self.assertEqual(source.read_bytes(), original)
        header = {'a': {'dtype': 'FUTURE8', 'shape': [1], 'data_offsets': [0, 1]},
                  'b': {'dtype': 'FUTURE8', 'shape': [1], 'data_offsets': [0, 1]}}
        encoded = json.dumps(header).encode()
        source.write_bytes(struct.pack('<Q', len(encoded)) + encoded + b'x')
        self.run_tool('contains-tensors', '--input', source, '--tensor', 'a', success=False)
        safetensors(source, [('model.weight', 'FUTURE8', [1], b'x')],
                    {'iild_package_schema': 'iild-safetensors-package-v1'})
        self.run_tool('contains-tensors', '--input', source, '--tensor', 'model.weight', success=False)

    def test_package_preserves_fp8_fnuz_and_complex_tensor_bytes(self):
        model = self.input / 'model.safetensors'
        safetensors(model, [('model.diffusion_model.a', 'F8_E4M3FNUZ', [2], b'\x01\x02'),
                            ('model.diffusion_model.b', 'F8_E5M2FNUZ', [2], b'\x03\x04'),
                            ('model.diffusion_model.c', 'C64', [1], b'\x05\x06\x07\x08\x09\x0a\x0b\x0c')])
        package = self.root / 'fp8-complex.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package)
        restored = self.root / 'restored'
        self.run_tool('extract', '--input', package, '--output', restored)
        self.assertEqual((restored / 'model.safetensors').read_bytes(), model.read_bytes())

    def test_container_content_takes_precedence_over_file_extension(self):
        cases = (("weight.bin", "safetensors"), ("weight.json", "safetensors"),
                 ("weight.gguf", "safetensors"), ("weight.safetensors", "gguf"),
                 ("weight.data", "gguf"))
        for number, (name, expected) in enumerate(cases):
            with self.subTest(name=name, format=expected):
                directory = self.input / f"case-{number}"
                directory.mkdir()
                source = directory / name
                if expected == "safetensors":
                    safetensors(source, [("model.diffusion_model.weight", "F32", [1], struct.pack("<f", 0.25))])
                else:
                    gguf(source)
                (directory / "tokenizer.bin").write_bytes(b"opaque tokenizer support file")
                report = self.run_tool("scan", "--input", directory)
                rows = {row["relative_path"]: row for row in report["files"]}
                self.assertEqual(rows[name]["format"], expected)
                self.assertEqual(rows["tokenizer.bin"]["format"], "asset")
                self.assertTrue(report["ready"], report["errors"])
                package = self.root / f"content-{number}.safetensors"
                self.run_tool("create", "--input", directory, "--output", package)
                restored = self.root / f"restored-content-{number}"
                self.run_tool("extract", "--input", package, "--output", restored)
                self.assertEqual((restored / name).read_bytes(), source.read_bytes())
                self.assertEqual((restored / "tokenizer.bin").read_bytes(), b"opaque tokenizer support file")

    def test_pytorch_candidate_is_reported_by_cli_and_python_api(self):
        source = self.input / 'checkpoint.safetensors.index.json'
        source.write_bytes(pickle.dumps(0x1950a86a20f9469cfc6c, protocol=2) + b'fixture')
        self.assertEqual(self.run_tool('probe', '--input', source)['format'], 'pytorch')
        spec = importlib.util.spec_from_file_location('iild_model_packaging',
            Path(__file__).resolve().parents[1] / 'src/ModelPackaging/python/iild_model_packaging.py')
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        self.assertEqual(api.detect_container(source, executable=PROGRAM)['format'], 'pytorch')

    def test_pytorch_shard_index_is_exposed_to_cli_and_python(self):
        first, second = self.input / 'part-a.bin', self.input / 'part-b.data'
        payload = pickle.dumps(0x1950a86a20f9469cfc6c, protocol=2) + b'fixture'
        first.write_bytes(payload); second.write_bytes(payload)
        index = self.input / 'model.json'
        index.write_text(json.dumps({'weight_map': {'alpha': first.name, 'beta': second.name}}))
        self.assertEqual(self.run_tool('probe', '--input', index)['format'], 'pytorch-index')
        report = self.run_tool('inspect-pytorch-index', '--input', index)
        self.assertEqual(report['schema'], 'iild-pytorch-index-v1')
        self.assertEqual({item['relative_path'] for item in report['shards']}, {first.name, second.name})
        self.assertEqual({item['name'] for item in report['tensors']}, {'alpha', 'beta'})
        spec = importlib.util.spec_from_file_location('iild_model_packaging',
            Path(__file__).resolve().parents[1] / 'src/ModelPackaging/python/iild_model_packaging.py')
        api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
        self.assertEqual(api.inspect_pytorch_index(index, executable=PROGRAM), report)
        index.write_text(json.dumps({'weight_map': {'alpha': '../outside.bin'}}))
        self.run_tool('inspect-pytorch-index', '--input', index, success=False)

    def test_package_calibration_repairs_safetensors_with_nonstandard_extension(self):
        source = self.input / "main.bin"
        safetensors(source, [("model.diffusion_model.weight", "F32", [2], struct.pack("<ff", 1, float("nan")))])
        original = source.read_bytes()
        package = self.root / "renamed-package.safetensors"
        self.run_tool("create", "--input", self.input, "--output", package)
        corrected = self.root / "renamed-corrected.safetensors"
        report = self.run_tool("calibrate", "--input", package, "--output", corrected,
                               "--repair-nonfinite", "true")
        self.assertEqual(report["nonfinite_values_replaced"], 1)
        restored = self.root / "renamed-restored"
        self.run_tool("extract", "--input", corrected, "--output", restored)
        self.assertEqual(self.run_tool("weight-health", "--input", restored / "main.bin")["nonfinite_values"], 0)
        self.assertEqual(source.read_bytes(), original)

    def test_truncated_disguised_containers_are_not_treated_as_opaque_assets(self):
        safetensors(self.input / "main.safetensors", [("model.diffusion_model.weight", "F32", [1], struct.pack("<f", 1))])
        (self.input / "bad-safetensors.bin").write_bytes(struct.pack("<Q", 512) + b"{}")
        (self.input / "bad-gguf.data").write_bytes(b"GGUF" + b"\x03\0\0\0")
        report = self.run_tool("scan", "--input", self.input)
        rows = {row["relative_path"]: row for row in report["files"]}
        for name in ("bad-safetensors.bin", "bad-gguf.data"):
            with self.subTest(name=name):
                self.assertEqual(rows[name]["status"], "invalid")

    def test_renamed_package_remains_verifiable_extractable_and_usable(self):
        main = self.input / "main.safetensors"
        encoder = self.input / "text_encoder.gguf"
        safetensors(main, [("model.diffusion_model.weight", "F32", [1], struct.pack("<f", 1))])
        gguf(encoder)
        package = self.root / "package.safetensors"
        self.run_tool("create", "--input", self.input, "--output", package)
        for name in ("package.bin", "package.gguf", "package.json"):
            with self.subTest(name=name):
                renamed = self.root / name
                renamed.write_bytes(package.read_bytes())
                self.assertTrue(self.run_tool("verify", "--input", renamed)["verified"])
                restored = self.root / (name + "-extracted")
                self.run_tool("extract", "--input", renamed, "--output", restored)
                self.assertEqual((restored / main.name).read_bytes(), main.read_bytes())
                self.assertEqual((restored / encoder.name).read_bytes(), encoder.read_bytes())
                cache = self.root / (name + "-cache")
                selected = Path(self.run_tool("materialize", "--input", renamed, "--output", cache)["output"])
                self.assertEqual(selected.read_bytes(), encoder.read_bytes())
                corrected = self.root / (name + "-corrected.safetensors")
                self.assertTrue(self.run_tool("calibrate", "--input", renamed, "--output", corrected)["verified"])
                self.assertEqual(renamed.read_bytes(), package.read_bytes())

    def test_standalone_repair_accepts_safetensors_input_by_content(self):
        source = self.input / "renamed.bin"
        safetensors(source, [("model.diffusion_model.weight", "F32", [1], struct.pack("<f", float("nan")))])
        original = source.read_bytes()
        output = self.root / "standalone-corrected.safetensors"
        self.assertEqual(self.run_tool("calibrate-file", "--input", source, "--output", output,
                                       "--repair-nonfinite", "true")["nonfinite_values_replaced"], 1)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(self.run_tool("weight-health", "--input", output)["nonfinite_values"], 0)

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

    def test_wan_diffusers_experts_follow_model_index_not_file_size(self):
        high = self.input / "transformer" / "weights.safetensors"
        low = self.input / "transformer_2" / "weights.safetensors"
        safetensors(high, [("patch_embedding.weight", "F32", [100], bytes(400))])
        safetensors(low, [("patch_embedding.weight", "F32", [1], bytes(4))])
        index = {"_class_name": "WanPipeline", "boundary_ratio": 0.73, "transformer": ["diffusers", "WanTransformer3DModel"],
                 "transformer_2": ["diffusers", "WanTransformer3DModel"]}
        (self.input / "model_index.json").write_text(json.dumps(index))
        report = self.run_tool("scan", "--input", self.input)
        self.assertTrue(report["ready"], report["errors"])
        rows = {item["relative_path"]: item for item in report["files"]}
        self.assertEqual((rows[str(high.relative_to(self.input))]["role"], rows[str(high.relative_to(self.input))]["position"]),
                         ("high_noise_model", "component"))
        self.assertEqual((rows[str(low.relative_to(self.input))]["role"], rows[str(low.relative_to(self.input))]["position"]),
                         ("model", "main"))
        self.assertEqual(rows[str(high.relative_to(self.input))]["role_source"], "model_index")
        output = self.root / "wan.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output)
        header, _ = read_header(output)
        manifest = json.loads(header["__metadata__"]["iild_package_manifest"])
        self.assertEqual(manifest['wan_moe_boundary'], 0.73)
        rows = {entry["path"]: entry for entry in manifest["files"]}
        self.assertEqual(rows[str(low.relative_to(self.input))]["position_source"], "model_index")
        restored = Path(self.run_tool("materialize", "--input", output, "--output", self.root / "cache",
                                     "--role", "high_noise_model")["output"])
        self.assertEqual(restored.read_bytes(), high.read_bytes())
        calibrated = self.root / "calibrated.safetensors"
        self.run_tool("calibrate", "--input", output, "--output", calibrated)
        calibrated_header, _ = read_header(calibrated)
        calibrated_rows = {entry["path"]: entry for entry in
                           json.loads(calibrated_header["__metadata__"]["iild_package_manifest"])["files"]}
        self.assertEqual(calibrated_rows[str(low.relative_to(self.input))]["position"], "main")
        self.assertEqual(calibrated_rows[str(high.relative_to(self.input))]["role"], "high_noise_model")
        conflict = self.run_tool("scan", "--input", self.input, "--assign-position",
                                 f"{high.relative_to(self.input)}=main", success=False)
        self.assertIn("conflict", conflict["error"].lower())
        safetensors(high.parent / "extra.safetensors", [("other.weight", "F32", [1], bytes(4))])
        ambiguous = self.run_tool("scan", "--input", self.input)
        self.assertFalse(ambiguous["ready"])
        self.assertIn("transformer", " ".join(ambiguous["errors"]))

    def test_wan_image_to_video_package_keeps_vision_out_of_the_main_model(self):
        main = self.input / 'transformer' / 'model.safetensors'
        vision = self.input / 'image_encoder' / 'model.safetensors'
        safetensors(main, [('patch_embedding.weight', 'F32', [1], bytes(4))])
        safetensors(vision, [('vision_model.embeddings.patch_embedding.weight', 'F32', [100], bytes(400))])
        (self.input / 'image_processor').mkdir()
        (self.input / 'image_processor/preprocessor_config.json').write_text('{}')
        index = {'_class_name': 'WanImageToVideoPipeline',
                 'transformer': ['diffusers', 'WanTransformer3DModel'],
                 'image_encoder': ['transformers', 'CLIPVisionModelWithProjection'],
                 'image_processor': ['transformers', 'CLIPProcessor']}
        (self.input / 'model_index.json').write_text(json.dumps(index))
        report = self.run_tool('scan', '--input', self.input)
        self.assertTrue(report['ready'], report['errors'])
        rows = {row['relative_path']: row for row in report['files']}
        self.assertEqual((rows[str(main.relative_to(self.input))]['role'], rows[str(main.relative_to(self.input))]['position']),
                         ('model', 'main'))
        self.assertEqual((rows[str(vision.relative_to(self.input))]['role'], rows[str(vision.relative_to(self.input))]['position']),
                         ('vision_encoder', 'component'))
        package = self.root / 'wan-i2v.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package)
        restored = Path(self.run_tool('materialize', '--input', package, '--output', self.root / 'cache',
                                      '--role', 'vision_encoder')['output'])
        self.assertEqual(restored.read_bytes(), vision.read_bytes())
        conflict = self.run_tool('scan', '--input', self.input, '--assign-role',
                                 f'{vision.relative_to(self.input)}=text_encoder', success=False)
        self.assertIn('conflict', conflict['error'].lower())
        extra = vision.with_name('extra.safetensors')
        safetensors(extra, [('vision_model.extra.weight', 'F32', [1], bytes(4))])
        ambiguous = self.run_tool('scan', '--input', self.input)
        self.assertFalse(ambiguous['ready'])
        self.assertIn('image_encoder', ' '.join(ambiguous['errors']))
        extra.unlink()
        index['image_processor'] = [None, None]
        (self.input / 'model_index.json').write_text(json.dumps(index))
        inconsistent = self.run_tool('scan', '--input', self.input)
        self.assertFalse(inconsistent['ready'])
        self.assertIn('image_', ' '.join(inconsistent['errors']))
        index['image_processor'] = ['transformers', 'CLIPProcessor']
        index['transformer'] = ['custom', 'WanTransformer3DModel']
        (self.input / 'model_index.json').write_text(json.dumps(index))
        unsupported = self.run_tool('scan', '--input', self.input)
        self.assertFalse(unsupported['ready'])
        self.assertIn('transformer', ' '.join(unsupported['errors']))

    def test_wan_image_to_video_dual_experts_follow_boundary_and_roles(self):
        high = self.input / 'transformer' / 'model.safetensors'
        low = self.input / 'transformer_2' / 'model.safetensors'
        safetensors(high, [('patch_embedding.weight', 'F32', [100], bytes(400))])
        safetensors(low, [('patch_embedding.weight', 'F32', [1], bytes(4))])
        index = {'_class_name': 'WanImageToVideoPipeline', 'boundary_ratio': 0.9,
                 'transformer': ['diffusers', 'WanTransformer3DModel'],
                 'transformer_2': ['diffusers', 'WanTransformer3DModel'],
                 'image_encoder': [None, None], 'image_processor': [None, None]}
        (self.input / 'model_index.json').write_text(json.dumps(index))
        report = self.run_tool('scan', '--input', self.input)
        self.assertTrue(report['ready'], report['errors'])
        rows = {row['relative_path']: row for row in report['files']}
        self.assertEqual((rows[str(high.relative_to(self.input))]['role'], rows[str(high.relative_to(self.input))]['position']),
                         ('high_noise_model', 'component'))
        self.assertEqual((rows[str(low.relative_to(self.input))]['role'], rows[str(low.relative_to(self.input))]['position']),
                         ('model', 'main'))
        package = self.root / 'wan-dual-i2v.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package)
        header, _ = read_header(package)
        self.assertEqual(json.loads(header['__metadata__']['iild_package_manifest'])['wan_moe_boundary'], 0.9)

    def test_wan_dual_expert_index_requires_a_valid_boundary(self):
        for name in ('transformer', 'transformer_2'):
            safetensors(self.input / name / 'weights.safetensors', [('weight', 'F32', [1], bytes(4))])
        index = {'_class_name': 'WanPipeline', 'transformer': ['diffusers', 'WanTransformer3DModel'],
                 'transformer_2': ['diffusers', 'WanTransformer3DModel']}
        path = self.input / 'model_index.json'
        for boundary in (None, -0.1, 1.1, 'automatic'):
            with self.subTest(boundary=boundary):
                index['boundary_ratio'] = boundary
                path.write_text(json.dumps(index))
                report = self.run_tool('scan', '--input', self.input)
                self.assertFalse(report['ready'])
                self.assertIn('boundary', ' '.join(report['errors']).lower())

    def test_wan_components_can_be_assigned_without_a_diffusers_index(self):
        for name in ("main", "expert", "vision"):
            safetensors(self.input / f"{name}.safetensors", [(f"{name}.weight", "F32", [1], bytes(4))])
        options = ("--assign-position", "main.safetensors=main", "--assign-role", "expert.safetensors=high_noise_model",
                   "--assign-role", "vision.safetensors=vision_encoder")
        output = self.root / "roles.safetensors"
        self.run_tool("create", "--input", self.input, "--output", output, *options)
        for role, name in (("high_noise_model", "expert"), ("vision_encoder", "vision")):
            restored = Path(self.run_tool("materialize", "--input", output, "--output", self.root / "roles-cache",
                                         "--role", role)["output"])
            self.assertEqual(restored.read_bytes(), (self.input / f"{name}.safetensors").read_bytes())

    def test_wan_diffusers_primary_names_map_without_changing_source_files(self):
        names = ['patch_embedding.weight', 'blocks.0.attn1.to_q.weight', 'blocks.0.attn2.norm_k.weight',
                 'condition_embedder.time_embedder.linear_1.weight', 'blocks.0.ffn.net.0.proj.weight',
                 'blocks.0.norm2.weight', 'blocks.0.scale_shift_table', 'scale_shift_table', 'proj_out.weight']
        tensors = [(name, 'F32', [1], struct.pack('<f', index)) for index, name in enumerate(names)]
        high, low = self.input / 'transformer' / 'weights.safetensors', self.input / 'transformer_2' / 'weights.safetensors'
        safetensors(high, tensors); safetensors(low, tensors)
        originals = (high.read_bytes(), low.read_bytes())
        (self.input / 'model_index.json').write_text(json.dumps({
            '_class_name': 'WanPipeline', 'boundary_ratio': 0.875, 'transformer': ['diffusers', 'WanTransformer3DModel'],
            'transformer_2': ['diffusers', 'WanTransformer3DModel']}))
        package = self.root / 'wan-diffusers-names.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package)
        header, _ = read_header(package)
        expected = ['patch_embedding.weight', 'blocks.0.self_attn.q.weight', 'blocks.0.cross_attn.norm_k.weight',
                    'time_embedding.0.weight', 'blocks.0.ffn.0.weight', 'blocks.0.norm3.weight',
                    'blocks.0.modulation', 'head.modulation', 'head.head.weight']
        for name in expected:
            self.assertIn('model.diffusion_model.' + name, header)
        inspected = self.run_tool('inspect', '--input', package)
        self.assertEqual((inspected['family'], inspected['recognized']), ('Wan', True))
        restored = self.root / 'diffusers-restored'
        self.run_tool('extract', '--input', package, '--output', restored)
        self.assertEqual((restored / high.relative_to(self.input)).read_bytes(), originals[0])
        self.assertEqual((restored / low.relative_to(self.input)).read_bytes(), originals[1])
        self.assertEqual((high.read_bytes(), low.read_bytes()), originals)
        calibrated = self.root / 'diffusers-calibrated.safetensors'
        self.run_tool('calibrate', '--input', package, '--output', calibrated)
        calibrated_header, _ = read_header(calibrated)
        self.assertIn('model.diffusion_model.blocks.0.cross_attn.norm_k.weight', calibrated_header)
        calibrated_restored = self.root / 'diffusers-calibrated-restored'
        self.run_tool('extract', '--input', calibrated, '--output', calibrated_restored)
        self.assertEqual((calibrated_restored / low.relative_to(self.input)).read_bytes(), originals[1])

    def test_wan_diffusers_single_primary_maps_without_a_pipeline_index(self):
        names = ['patch_embedding.weight', 'blocks.0.attn1.to_q.weight', 'blocks.0.attn2.norm_k.weight',
                 'condition_embedder.time_embedder.linear_1.weight']
        source = self.input / 'model.safetensors'
        safetensors(source, [(name, 'F32', [1], struct.pack('<f', i)) for i, name in enumerate(names)])
        original = source.read_bytes()
        package = self.root / 'single-wan.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package)
        header, _ = read_header(package)
        self.assertIn('model.diffusion_model.blocks.0.cross_attn.norm_k.weight', header)
        self.assertIn('model.diffusion_model.blocks.0.self_attn.q.weight', header)
        restored = self.root / 'single-restored'
        self.run_tool('extract', '--input', package, '--output', restored)
        self.assertEqual((restored / source.name).read_bytes(), original)
        self.assertEqual(source.read_bytes(), original)

    def test_wan_indexed_expert_shards_share_the_role_and_materialize_together(self):
        high = self.input / 'transformer'
        safetensors(high / 'part-1.safetensors', [('patch_embedding.weight', 'F32', [1], b'abcd')])
        safetensors(high / 'part-2.safetensors', [('blocks.0.weight', 'F32', [1], b'efgh')])
        (high / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
            'patch_embedding.weight': 'part-1.safetensors', 'blocks.0.weight': 'part-2.safetensors'}}))
        safetensors(self.input / 'transformer_2' / 'model.safetensors', [('low.weight', 'F32', [1], b'ijkl')])
        (self.input / 'model_index.json').write_text(json.dumps({
            '_class_name': 'WanPipeline', 'boundary_ratio': 0.875, 'transformer': ['diffusers', 'WanTransformer3DModel'],
            'transformer_2': ['diffusers', 'WanTransformer3DModel']}))
        report = self.run_tool('scan', '--input', self.input)
        self.assertTrue(report['ready'], report['errors'])
        shards = [row for row in report['files'] if row['relative_path'].startswith('transformer/part-')]
        self.assertEqual([row['role'] for row in shards], ['high_noise_model', 'high_noise_model'])
        package = self.root / 'wan-shards.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package)
        restored = Path(self.run_tool('materialize', '--input', package, '--output', self.root / 'shard-cache',
                                      '--role', 'high_noise_model')['output'])
        header, offset = read_header(restored)
        self.assertEqual(set(header) - {'__metadata__'}, {'patch_embedding.weight', 'blocks.0.weight'})
        contents = restored.read_bytes()[offset:]
        for name, expected in (('patch_embedding.weight', b'abcd'), ('blocks.0.weight', b'efgh')):
            begin, end = header[name]['data_offsets']
            self.assertEqual(contents[begin:end], expected)

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

    def test_mixed_case_index_is_grouped_and_round_trips(self):
        safetensors(self.input / 'a.safetensors', [('a', 'BF16', [1], b'\x01\x02')])
        safetensors(self.input / 'b.safetensors', [('b', 'BF16', [1], b'\x03\x04')])
        index = self.input / 'model.SAFETENSORS.Index.JSON'
        index.write_text(json.dumps({'weight_map': {'a': 'a.safetensors', 'b': 'b.safetensors'}}))
        report = self.run_tool('scan', '--input', self.input, '--assign-position', 'a.safetensors=main')
        rows = {f['id']: f for f in report['files']}
        self.assertEqual(rows['b.safetensors']['position'], 'main')
        output = self.root / 'mixed-case-package.safetensors'
        self.run_tool('create', '--input', self.input, '--output', output, '--assign-position', 'a.safetensors=main')
        restored = self.root / 'restored'
        self.run_tool('extract', '--input', output, '--output', restored)
        self.assertEqual((restored / index.name).read_bytes(), index.read_bytes())

    def test_renamed_index_and_shards_are_grouped_and_materialized(self):
        first, second = self.input / 'a.bin', self.input / 'b.data'
        safetensors(first, [('a', 'BF16', [1], b'\x01\x02')])
        safetensors(second, [('b', 'BF16', [1], b'\x03\x04')])
        index = self.input / 'model-index.data'
        index.write_text(json.dumps({'weight_map': {'a': first.name, 'b': second.name}}))
        originals = {path.name: path.read_bytes() for path in (first, second, index)}
        options = ('--assign-role', 'a.bin=model', '--assign-position', 'a.bin=main')
        report = self.run_tool('scan', '--input', self.input, *options)
        rows = {entry['id']: entry for entry in report['files']}
        self.assertTrue(report['ready'], report['errors'])
        self.assertEqual(rows[index.name]['status'], 'included')
        self.assertEqual(rows[first.name]['group'], index.name)
        self.assertEqual(rows[second.name]['group'], index.name)
        self.assertEqual(rows[second.name]['position'], 'main')
        package = self.root / 'renamed-index-package.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package, *options)
        output = Path(self.run_tool('materialize', '--input', package, '--output', self.root / 'cache',
                                    '--asset-path', index.name)['output'])
        header, _ = read_header(output)
        self.assertEqual(set(header) - {'__metadata__'}, {'a', 'b'})
        restored = self.root / 'renamed-restored'
        self.run_tool('extract', '--input', package, '--output', restored)
        self.assertEqual({name: (restored / name).read_bytes() for name in originals}, originals)
        self.assertEqual({path.name: path.read_bytes() for path in (first, second, index)}, originals)
        for name in ('index.safetensors', 'index.gguf'):
            with self.subTest(index_name=name):
                renamed = index.with_name(name)
                index.rename(renamed)
                try:
                    rows = {entry['id']: entry for entry in self.run_tool('scan', '--input', self.input, *options)['files']}
                    self.assertEqual(rows[name]['status'], 'included')
                    self.assertEqual(rows[first.name]['group'], name)
                    other = self.root / (name + '.package.safetensors')
                    self.run_tool('create', '--input', self.input, '--output', other, *options)
                    merged = Path(self.run_tool('materialize', '--input', other, '--output', self.root / 'cache',
                                                '--asset-path', name)['output'])
                    self.assertEqual(set(read_header(merged)[0]) - {'__metadata__'}, {'a', 'b'})
                finally:
                    renamed.rename(index)
        second_index = self.input / 'second-index.data'
        second_index.write_bytes(index.read_bytes())
        conflict = self.run_tool('scan', '--input', self.input, *options)
        self.assertFalse(conflict['ready'])
        self.assertIn('multiple', ' '.join(conflict['errors']).lower())

    def test_index_link_and_same_directory_target_are_one_group(self):
        first, second = self.input / 'a.safetensors', self.input / 'b.safetensors'
        safetensors(first, [('a', 'BF16', [1], b'\x01\x02')])
        safetensors(second, [('b', 'BF16', [1], b'\x03\x04')])
        target = self.input / 'index-content.data'
        target.write_text(json.dumps({'weight_map': {'a': first.name, 'b': second.name}}))
        index = self.input / 'model.safetensors.index.json'
        try:
            index.symlink_to(target.name)
        except (OSError, NotImplementedError):
            self.skipTest('Symbolic links are unavailable')
        options = ('--assign-position', 'a.safetensors=main')
        report = self.run_tool('scan', '--input', self.input, *options)
        self.assertTrue(report['ready'], report['errors'])
        rows = {entry['id']: entry for entry in report['files']}
        self.assertEqual(rows[first.name]['group'], index.name)
        self.assertEqual(rows[second.name]['group'], index.name)
        package = self.root / 'linked-index-package.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package, *options)
        merged = Path(self.run_tool('materialize', '--input', package, '--output', self.root / 'cache',
                                    '--asset-path', index.name)['output'])
        self.assertEqual(set(read_header(merged)[0]) - {'__metadata__'}, {'a', 'b'})

    def test_single_safetensors_with_index_suffix_is_not_materialized_as_an_index(self):
        model = self.input / 'model.safetensors.index.json'
        safetensors(model, [('model.diffusion_model.weight', 'BF16', [1], b'\x01\x02')])
        package = self.root / 'single-package.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package)
        restored = Path(self.run_tool('materialize', '--input', package, '--output', self.root / 'cache',
                                      '--asset-path', model.name)['output'])
        self.assertEqual(restored.read_bytes(), model.read_bytes())

    def test_packaging_rejects_unindexed_tensors_in_a_renamed_group(self):
        safetensors(self.input / 'a.bin', [('a', 'BF16', [1], b'\x01\x02'),
                                          ('unindexed', 'BF16', [1], b'\x03\x04')])
        index = self.input / 'model-index.data'
        index.write_text(json.dumps({'weight_map': {'a': 'a.bin'}}))
        report = self.run_tool('scan', '--input', self.input)
        self.assertFalse(report['ready'])
        self.assertIn('unindexed', ' '.join(report['errors']).lower())
        self.run_tool('create', '--input', self.input, '--output', self.root / 'incomplete.safetensors', success=False)

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

    def test_wan_gguf_is_a_model_role_not_a_text_encoder(self):
        gguf(self.input / 'encoder.gguf', architecture='gemma3')
        for architecture in ('wan', 'wan2.1', 'wan2.2'):
            with self.subTest(architecture=architecture):
                gguf(self.input / 'unnamed.gguf', architecture=architecture)
                report = self.run_tool('scan', '--input', self.input)
                rows = {item['id']: item for item in report['files']}
                self.assertEqual((rows['unnamed.gguf']['role'], rows['unnamed.gguf']['position']), ('model', 'main'))
                self.assertEqual(rows['encoder.gguf']['role'], 'text_encoder')
        self.assertEqual(rows["encoder.gguf"]["position"], "component")

    def test_gguf_architecture_selects_diffusion_model_before_text_encoder(self):
        for architecture in ("ltxv", "flux", "anima"):
            with self.subTest(architecture=architecture):
                directory = self.input / architecture
                directory.mkdir()
                encoder = directory / "a-encoder.gguf"
                model = directory / "z-diffusion.data"
                gguf(encoder, "gemma3")
                gguf(model, architecture)
                report = self.run_tool("scan", "--input", directory)
                rows = {row["relative_path"]: row for row in report["files"]}
                self.assertTrue(report["ready"], report["errors"])
                self.assertEqual(rows[model.name]["role"], "model")
                self.assertEqual(rows[model.name]["position"], "main")
                self.assertEqual(rows[encoder.name]["role"], "text_encoder")
                self.assertEqual(rows[encoder.name]["position"], "component")
                package = self.root / (architecture + ".safetensors")
                self.assertTrue(self.run_tool("create", "--input", directory, "--output", package)["verified"])
                manifest = json.loads(read_header(package)[0]["__metadata__"]["iild_package_manifest"])
                packaged = {row["path"]: row for row in manifest["files"]}
                self.assertEqual(packaged[model.name]["role"], "model")
                self.assertEqual(packaged[encoder.name]["role"], "text_encoder")
                extracted = self.root / (architecture + "-extracted")
                self.run_tool("extract", "--input", package, "--output", extracted)
                self.assertEqual((extracted / model.name).read_bytes(), model.read_bytes())
                self.assertEqual((extracted / encoder.name).read_bytes(), encoder.read_bytes())

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

    def make_sharded_encoder(self, folder, *, extra_unindexed=False):
        first = folder / 'model-00001-of-00002.safetensors'
        second = folder / 'model-00002-of-00002.safetensors'
        left = [('model.embed_tokens.weight', 'F32', [1], struct.pack('<f', 0.25))]
        if extra_unindexed:
            left.append(('model.extra.weight', 'I8', [1], b'\x05'))
        safetensors(first, left, {'format': 'pt'})
        first_header, first_base = read_header(first)
        first_data = first.read_bytes()[first_base:]
        first_header['model.embed_tokens.weight']['encoding_hint'] = {'scheme': 'fixture'}
        encoded = json.dumps(first_header, separators=(',', ':')).encode()
        encoded += b' ' * (-len(encoded) % 8)
        first.write_bytes(struct.pack('<Q', len(encoded)) + encoded + first_data)
        safetensors(second, [('model.norm.weight', 'BF16', [2], b'\x01\x02\x03\x04'),
                             ('model.layers.0.weight', 'I8', [2], b'\x80\x7f')], {'format': 'pt'})
        index = folder / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'model.embed_tokens.weight': first.name,
                                                   'model.norm.weight': second.name,
                                                   'model.layers.0.weight': second.name}}))
        return index

    def test_materialize_indexed_encoder_shards_as_one_verified_file(self):
        fixture(self.input, encoder=False)
        index = self.make_sharded_encoder(self.input / 'text_encoder')
        package = self.root / 'package.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package)
        before = hashlib.sha256(package.read_bytes()).hexdigest()
        cache = self.root / 'assets'
        result = Path(self.run_tool('materialize', '--input', package, '--output', cache)['output'])
        header, base = read_header(result)
        self.assertEqual(set(header) - {'__metadata__'}, {'model.embed_tokens.weight', 'model.norm.weight',
                                                         'model.layers.0.weight'})
        self.assertEqual(header['__metadata__']['format'], 'pt')
        self.assertEqual(header['model.embed_tokens.weight']['encoding_hint'], {'scheme': 'fixture'})
        self.assertEqual(header['model.layers.0.weight']['dtype'], 'I8')
        data = result.read_bytes()
        for name, expected in [('model.embed_tokens.weight', struct.pack('<f', 0.25)),
                               ('model.norm.weight', b'\x01\x02\x03\x04'),
                               ('model.layers.0.weight', b'\x80\x7f')]:
            start, end = header[name]['data_offsets']
            self.assertEqual(data[base + start:base + end], expected)
        self.assertEqual(self.run_tool('materialize', '--input', package, '--output', cache,
                                       '--asset-path', index.relative_to(self.input))['output'], str(result))
        spec = importlib.util.spec_from_file_location('iild_model_packaging',
            Path(__file__).resolve().parents[1] / 'src/ModelPackaging/python/iild_model_packaging.py')
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        self.assertEqual(api.materialize_package_asset(package, cache, relative_source=index.relative_to(self.input),
                                                       executable=PROGRAM)['output'], str(result))
        self.assertEqual(hashlib.sha256(package.read_bytes()).hexdigest(), before)
        result.write_bytes(b'X' * result.stat().st_size)
        self.run_tool('materialize', '--input', package, '--output', cache, success=False)
        self.assertEqual(result.read_bytes(), b'X' * result.stat().st_size)

    def test_materialize_many_indexed_tensor_mappings(self):
        safetensors(self.input / 'main.safetensors', [('model.diffusion_model.weight', 'U8', [1], b'\x01')])
        folder = self.input / 'encoder'
        weight_map = {}
        for shard in range(2):
            filename = f'part-{shard}.safetensors'
            tensors = [(f'model.layers.{shard * 1024 + i}.weight', 'U8', [1], bytes([i % 256]))
                       for i in range(1024)]
            safetensors(folder / filename, tensors)
            weight_map.update({name: filename for name, _, _, _ in tensors})
        (folder / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
        package = self.root / 'many-tensors.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package,
                      '--assign-position', 'main.safetensors=main', '--assign-role', 'encoder/part-0.safetensors=text_encoder')
        result = self.run_tool('materialize', '--input', package, '--output', self.root / 'assets')
        merged, base = read_header(Path(result['output']))
        self.assertEqual(set(merged) - {'__metadata__'}, set(weight_map))
        payload = Path(result['output']).read_bytes()
        for index in (0, 1023, 1024, 2047):
            start, end = merged[f'model.layers.{index}.weight']['data_offsets']
            self.assertEqual(payload[base + start:base + end], bytes([index % 256]))

    def test_packaging_requires_exact_index_coverage(self):
        fixture(self.input, encoder=False)
        self.make_sharded_encoder(self.input / 'text_encoder', extra_unindexed=True)
        package = self.root / 'package.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package, success=False)
        self.assertFalse(package.exists())

    def test_materialize_rejects_legacy_index_without_exact_coverage(self):
        fixture(self.input, encoder=False)
        index = self.make_sharded_encoder(self.input / 'text_encoder')
        package = self.root / 'package.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package)
        header, base = read_header(package)
        manifest = json.loads(header['__metadata__']['iild_package_manifest'])
        embedded = next(item for item in manifest['files'] if item['path'] == str(index.relative_to(self.input)))
        record = embedded['tensors'][0]
        payload = package.read_bytes()
        key = record['output_key']
        begin, end = header[key]['data_offsets']
        original = payload[base + begin:base + end]
        mapping = json.loads(original)
        del mapping['weight_map']['model.layers.0.weight']
        modified = json.dumps(mapping, separators=(',', ':')).encode()
        self.assertLessEqual(len(modified), len(original))
        modified += b' ' * (len(original) - len(modified))
        digest = hashlib.sha256(modified).hexdigest()
        embedded['sha256'] = digest
        record['sha256'] = digest
        header['__metadata__']['iild_package_manifest'] = json.dumps(manifest, separators=(',', ':'))
        tensors = [(name, info['dtype'], info['shape'],
                    modified if name == key else payload[base + info['data_offsets'][0]:base + info['data_offsets'][1]])
                   for name, info in header.items() if name != '__metadata__']
        legacy = self.root / 'legacy-incomplete.safetensors'
        safetensors(legacy, tensors, header['__metadata__'])
        cache = self.root / 'legacy-assets'
        report = self.run_tool('materialize', '--input', legacy, '--output', cache,
                               '--asset-path', index.relative_to(self.input), success=False)
        self.assertIn('exactly cover', str(report))
        self.assertFalse(cache.exists() and list(cache.iterdir()))

    def test_materialize_shards_rejects_corrupted_index_or_tensor(self):
        fixture(self.input, encoder=False)
        index = self.make_sharded_encoder(self.input / 'text_encoder')
        original = self.root / 'package.safetensors'
        self.run_tool('create', '--input', self.input, '--output', original)
        header, base = read_header(original)
        keys = [next(key for key in header if key.startswith('components/text_encoder/')
                     and key.endswith('/model.norm.weight')),
                next(key for key in header if key.startswith('__iild_assets__/')
                     and key.endswith('model.safetensors.index.json'))]
        for position, key in enumerate(keys):
            package = self.root / f'corrupt-{position}.safetensors'
            package.write_bytes(original.read_bytes())
            with package.open('r+b') as stream:
                stream.seek(base + header[key]['data_offsets'][0])
                byte = stream.read(1)
                stream.seek(-1, os.SEEK_CUR)
                stream.write(bytes([byte[0] ^ 0x01]))
            cache = self.root / f'assets-{position}'
            self.run_tool('materialize', '--input', package, '--output', cache,
                          '--asset-path', index.relative_to(self.input), success=False)
            self.assertFalse(cache.exists() and list(cache.iterdir()))

    def test_explicit_shard_index_disambiguates_multiple_encoders(self):
        fixture(self.input, encoder=False)
        index_a = self.make_sharded_encoder(self.input / 'text_encoder_a')
        self.make_sharded_encoder(self.input / 'text_encoder_b')
        package = self.root / 'package.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package)
        cache = self.root / 'assets'
        self.run_tool('materialize', '--input', package, '--output', cache, success=False)
        report = self.run_tool('materialize', '--input', package, '--output', cache,
                               '--asset-path', index_a.relative_to(self.input))
        self.assertTrue(Path(report['output']).is_file())

    def test_misclassified_shard_requires_explicit_index_instead_of_partial_encoder(self):
        fixture(self.input, encoder=False)
        index = self.make_sharded_encoder(self.input / 'text_encoder')
        package = self.root / 'package.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package)
        header, base = read_header(package)
        data = package.read_bytes()
        manifest = json.loads(header['__metadata__']['iild_package_manifest'])
        shard = next(item for item in manifest['files'] if item['path'].endswith('model-00002-of-00002.safetensors'))
        shard['role'] = 'model'
        header['__metadata__']['iild_package_manifest'] = json.dumps(manifest, separators=(',', ':'))
        tensors = [(name, info['dtype'], info['shape'], data[base + info['data_offsets'][0]:base + info['data_offsets'][1]])
                   for name, info in header.items() if name != '__metadata__']
        old_package = self.root / 'misclassified.safetensors'
        safetensors(old_package, tensors, header['__metadata__'])
        cache = self.root / 'assets'
        self.run_tool('materialize', '--input', old_package, '--output', cache, success=False)
        self.assertFalse(cache.exists() and list(cache.iterdir()))
        report = self.run_tool('materialize', '--input', old_package, '--output', cache,
                               '--asset-path', index.relative_to(self.input))
        merged, _ = read_header(Path(report['output']))
        self.assertIn('model.embed_tokens.weight', merged)
        self.assertIn('model.norm.weight', merged)

    def test_materialize_indexed_component_when_all_shards_are_embedded_duplicates(self):
        safetensors(self.input / 'main.safetensors', [
            ('model.diffusion_model.block.weight', 'F32', [1], b'\x00\x00\x00\x00'),
            ('vae.encoder.weight', 'F32', [1], b'\x01\x02\x03\x04'),
            ('vae.decoder.weight', 'F32', [1], b'\x05\x06\x07\x08')])
        folder = self.input / 'vae'
        safetensors(folder / 'part-1.safetensors', [('encoder.weight', 'F32', [1], b'\x01\x02\x03\x04')])
        safetensors(folder / 'part-2.safetensors', [('decoder.weight', 'F32', [1], b'\x05\x06\x07\x08')])
        index = folder / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'encoder.weight': 'part-1.safetensors',
                                                   'decoder.weight': 'part-2.safetensors'}}))
        package = self.root / 'duplicate-vae.safetensors'
        self.run_tool('create', '--input', self.input, '--output', package,
                      '--assign-position', 'main.safetensors=main', '--assign-role', 'vae/part-1.safetensors=video_vae')
        header, _ = read_header(package)
        manifest = json.loads(header['__metadata__']['iild_package_manifest'])
        shards = [entry for entry in manifest['files'] if entry['path'].startswith('vae/part-')]
        self.assertEqual([entry['status'] for entry in shards], ['duplicate', 'duplicate'])
        cache = self.root / 'assets'
        report = self.run_tool('materialize', '--input', package, '--output', cache, '--role', 'video_vae')
        restored, _ = read_header(Path(report['output']))
        self.assertEqual(set(restored) - {'__metadata__'}, {'encoder.weight', 'decoder.weight'})
        self.assertEqual(report['output'], self.run_tool('materialize', '--input', package, '--output', cache,
                                                        '--asset-path', index.relative_to(self.input))['output'])
        first = self.run_tool('materialize', '--input', package, '--output', cache,
                              '--asset-path', 'vae/part-1.safetensors')
        self.assertEqual(Path(first['output']).read_bytes(), (folder / 'part-1.safetensors').read_bytes())

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
