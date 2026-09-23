"""Unified inference routing, integrity and publication contracts."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import unified_image
from iild_package import materialize_archive, write_archive
from inference_session import InferenceSession
from weight_files import file_sha256


class UnifiedImageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="unified-image-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.package = self.root / "model.iildmodel"
        self.package.mkdir()
        self.member = self.package / "member.safetensors"
        self.member.write_bytes(b"native engine boundary fixture")
        self.manifest = {"schema": "iild-unified-model-v1", "_class_name": "IILDUnifiedCascade",
                         "composition": "ordered-image-refinement", "stages": [{"model": self.member.name,
                         "strength": 1, "size_bytes": self.member.stat().st_size, "sha256": file_sha256(self.member)}]}
        self.write_manifest()

    def write_manifest(self):
        (self.package / "model_index.json").write_text(json.dumps(self.manifest))

    def test_inspect_rejects_escape_duplicate_boolean_and_changed_member(self):
        for key, value in (("model", "../member.safetensors"), ("strength", True), ("size_bytes", 99)):
            original = self.manifest["stages"][0][key]
            self.manifest["stages"][0][key] = value; self.write_manifest()
            with self.subTest(key=key), self.assertRaises(ValueError):
                unified_image.inspect_package(self.package)
            self.manifest["stages"][0][key] = original
        self.manifest["stages"].append(self.manifest["stages"][0]); self.write_manifest()
        with self.assertRaises(ValueError):
            unified_image.inspect_package(self.package)

    def test_hashes_detect_same_size_replacement(self):
        self.member.write_bytes(b"x" * self.member.stat().st_size)
        with self.assertRaisesRegex(ValueError, "hash differs"):
            unified_image.inspect_package(self.package, hashes=True)

    def test_explicit_vae_is_local_nonempty_and_shared_between_stages(self):
        vae = self.package / "decoder.safetensors"
        vae.write_bytes(b"decoder fixture")
        stage = self.manifest["stages"][0]
        stage["vae"] = vae.name
        self.write_manifest()
        with patch.object(unified_image, "model_content_sha256", side_effect=AssertionError("Unexpected hash")):
            unified_image.inspect_package(self.package)
        for value in ("../outside", "/absolute", "missing", "", None):
            stage["vae"] = value
            self.write_manifest()
            with self.subTest(value=value), self.assertRaises(ValueError):
                unified_image.inspect_package(self.package)
        stage["vae"] = vae.name
        vae.write_bytes(b"")
        self.write_manifest()
        with self.assertRaises(ValueError):
            unified_image.inspect_package(self.package)

    def test_architecture_scoped_vae_catalog_is_validated(self):
        vae = self.package / "decoder.safetensors"
        config = self.package / "decoder.json"
        vae.write_bytes(b"decoder fixture")
        config.write_text("{}")
        self.manifest["vae_catalog"] = "vae_catalog.json"
        self.manifest["vae_variants"] = [{"id": "sdxl-standard", "weights": vae.name, "config": config.name}]
        self.manifest["stages"][0]["vae_variant"] = "sdxl-standard"
        (self.package / "vae_catalog.json").write_text('{"schema":"iild-vae-catalog-v1"}')
        self.write_manifest()
        unified_image.inspect_package(self.package)
        self.manifest["stages"][0]["vae_variant"] = "unknown"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "unknown VAE variant"):
            unified_image.inspect_package(self.package)

    def test_print_configuration_does_not_load_engine_or_publish(self):
        output = self.root / "images"
        with patch.object(unified_image, "NativeEngine", side_effect=AssertionError("Unexpected inference")), redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(unified_image.main(["--model-path", str(self.package), "--output-dir", str(output), "--print-config"]), 0)
        self.assertEqual(json.loads(stream.getvalue())["backend"], "unified")
        self.assertFalse(output.exists())

    def test_single_file_package_is_inspected_materialized_and_routed(self):
        self.manifest["container"] = "zip-stored-v1"
        self.write_manifest()
        packaged = self.root / "packaged.iildmodel"
        write_archive(self.package, packaged)
        source, manifest = unified_image.inspect_package(packaged, hashes=True)
        self.assertEqual(source, packaged)
        self.assertEqual(manifest["container"], "zip-stored-v1")
        materialized = materialize_archive(packaged, self.root / "cache")
        self.assertEqual((materialized / self.member.name).read_bytes(), self.member.read_bytes())
        output = self.root / "packaged-images"
        with patch.object(unified_image, "NativeEngine", side_effect=AssertionError("Unexpected inference")), redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(unified_image.main(["--model-path", str(packaged), "--output-dir", str(output), "--print-config"]), 0)
        self.assertEqual(json.loads(stream.getvalue())["model"], str(packaged))
        spec = importlib.util.spec_from_file_location("packaged_router_fixture", ROOT / "reference/generate.py")
        router = importlib.util.module_from_spec(spec); spec.loader.exec_module(router)
        from types import SimpleNamespace
        self.assertEqual(router.select_backend(SimpleNamespace(backend="auto", base_model=None),
                         ["--model-path", str(packaged)]), "unified")

    def test_single_file_package_rejects_compression_and_traversal(self):
        for name, compression in (("compressed.iildmodel", zipfile.ZIP_DEFLATED),
                                  ("redirected.iildmodel", zipfile.ZIP_STORED)):
            packaged = self.root / name
            with zipfile.ZipFile(packaged, "w", compression=compression) as archive:
                archive.writestr("model_index.json", json.dumps(self.manifest))
                archive.writestr("../member.safetensors" if "redirected" in name else self.member.name,
                                 self.member.read_bytes())
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsupported|path"):
                unified_image.inspect_package(packaged)

    def test_foreground_retains_package_engine_and_reuses_it(self):
        output = self.root / "prepared-images"
        args = ["--model-path", str(self.package), "--output-dir", str(output)]
        with InferenceSession() as session, patch.object(unified_image, "NativeEngine") as engine:
            self.assertEqual(session.set_foreground(True, lambda: unified_image.main(args)), 0)
            self.assertTrue(session.residency()["ready"])
            self.assertEqual(session.residency()["model"], str(self.package))
            self.assertEqual(session.set_foreground(True, lambda: unified_image.main(args)), 0)
            engine.assert_called_once()
            self.assertEqual(engine.return_value.image.call_count, 2)
            self.assertTrue(engine.return_value.image.call_args.kwargs["prepare"])
            self.assertFalse(output.exists())
            self.member.write_bytes(b"x" * self.member.stat().st_size)
            with self.assertRaisesRegex(ValueError, "hash differs"):
                session.set_foreground(True, lambda: unified_image.main(args))
            self.assertFalse(session.residency()["ready"])

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required for image publication")
    def test_generation_publishes_complete_manifest_and_failure_publishes_no_images(self):
        from PIL import Image
        class Engine:
            def image(self, args, seed):
                return Image.new("RGB", (args.width, args.height), (12, 34, 56)), {"fixture": True}
        output = self.root / "images"
        args = ["--model-path", str(self.package), "--output-dir", str(output), "--width", "64", "--height", "64", "--seed", "10"]
        with patch.object(unified_image, "NativeEngine", Engine), redirect_stdout(io.StringIO()):
            self.assertEqual(unified_image.main(args), 0)
        report = json.loads((output / "generation.json").read_text())
        self.assertEqual(report["backend"], "unified")
        self.assertEqual(report["outputs"][0]["sha256"], file_sha256(output / "image-0001.png"))
        failed = self.root / "failed"
        args[3] = str(failed)
        with patch.object(unified_image.NativeEngine, "image", side_effect=RuntimeError("stage failed")), patch.object(unified_image.NativeEngine, "__init__", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "stage failed"):
                unified_image.main(args)
        self.assertFalse(any(failed.iterdir()))

    def test_auto_router_selects_unified_backend(self):
        spec = importlib.util.spec_from_file_location("unified_router_fixture", ROOT / "reference/generate.py")
        router = importlib.util.module_from_spec(spec); spec.loader.exec_module(router)
        from types import SimpleNamespace
        self.assertEqual(router.select_backend(SimpleNamespace(backend="auto", base_model=None),
                         ["--model-path", str(self.package)]), "unified")


if __name__ == "__main__":
    unittest.main()
