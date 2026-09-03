#!/usr/bin/env python3
"""Non-clobbering output defaults, complete batches, and explicit generator values."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import generate
import generation_options
import generation_output
import presets


class GenerationOutputTests(unittest.TestCase):
    def args(self, *tokens):
        return generate.resolve_arguments(generate.build_parser().parse_args(list(tokens)))[1]

    def test_default_collision_allocates_an_unused_name_without_deleting_old_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            path = Path(temporary) / "fixture.png"
            path.write_bytes(b"original")
            path.with_suffix(".json").write_text("{}")
            args = self.args()
            args.output = path
            outputs = generation_output.resolve_output_paths(args)
            self.assertEqual(outputs, [path.with_stem("fixture-run-0002")])
            self.assertEqual(path.read_bytes(), b"original")

    def test_batch_names_and_json_collisions_are_checked_before_generation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            path = Path(temporary) / "batch.png"
            args = self.args("--output", str(path), "--num-images", "2")
            outputs = generation_output.resolve_output_paths(args)
            self.assertEqual([p.name for p in outputs], ["batch-0001.png", "batch-0002.png"])
            outputs[1].with_suffix(".json").write_text("{}")
            with self.assertRaisesRegex(SystemExit, "overwrite"):
                generation_output.resolve_output_paths(args)

    def test_explicit_overwrite_preserves_the_requested_paths(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            path = Path(temporary) / "chosen.png"
            path.write_bytes(b"original")
            args = self.args("--output", str(path), "--overwrite")
            self.assertEqual(generation_output.resolve_output_paths(args), [path])

    def test_quoted_home_paths_are_expanded_before_execution_and_serialization(self):
        args = self.args("--output", "~/iild-result/image.png", "--cache-dir", "~/iild-cache",
                         "--xet-cache-dir", "~/iild-xet")
        expected = {
            "output": Path.home() / "iild-result" / "image.png",
            "cache_dir": Path.home() / "iild-cache",
            "xet_cache_dir": Path.home() / "iild-xet",
        }
        exported = generate.configuration_values(args)
        for name, path in expected.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(args, name), path)
                self.assertEqual(exported[name], str(path))

    def test_atomic_publication_refuses_a_racing_output_without_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            root = Path(temporary)
            candidate, target = root / "new.tmp", root / "image.png"
            candidate.write_bytes(b"new")
            target.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                generation_output.publish_file(candidate, target, overwrite=False)
            self.assertEqual(target.read_bytes(), b"existing")
            self.assertEqual(candidate.read_bytes(), b"new")
            generation_output.publish_file(candidate, target, overwrite=True)
            self.assertEqual(target.read_bytes(), b"new")
            self.assertFalse(candidate.exists())

    def test_failed_png_encoding_cleans_up_its_temporary_file(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            root = Path(temporary)
            image = Mock(save=Mock(side_effect=RuntimeError("encoding failed")))
            with self.assertRaisesRegex(RuntimeError, "encoding failed"):
                generation_output.write_png(image, root / "image.png", compress_level=6,
                                            optimize=False, overwrite=False)
            self.assertEqual(list(root.iterdir()), [])

    def test_seed_stride_and_generator_device_are_forwarded(self):
        args = self.args("--num-images", "3", "--seed", "0", "--seed-stride", "2",
                         "--generator-device", "execution", "--device-index", "1")
        torch = SimpleNamespace(Generator=Mock(side_effect=lambda **kw: Mock()))
        generators, seeds, device = generation_options.build_generators(torch, args, "cuda")
        self.assertEqual((len(generators), seeds, device), (3, [0, 2, 4], "cuda:1"))
        self.assertEqual(torch.Generator.call_count, 3)
        self.assertTrue(all(call.kwargs["device"] == "cuda:1" for call in torch.Generator.call_args_list))

    def test_execution_generator_cannot_secretly_fall_back_on_metal(self):
        with self.assertRaisesRegex(ValueError, "CUDA/ROCm"):
            generation_options.build_generators(Mock(), self.args("--generator-device", "execution"), "mps")

    def test_device_index_is_validated_and_selected_on_shared_cuda_namespace(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2, set_device=Mock()))
        generation_options.select_device_index(torch, "cuda", 1)
        torch.cuda.set_device.assert_called_once_with(1)
        with self.assertRaises(ValueError):
            generation_options.select_device_index(torch, "cuda", 2)
        with self.assertRaises(ValueError):
            generation_options.select_device_index(torch, "mps", 1)

    def test_dtype_override_is_separate_from_family_auto_defaults(self):
        args = self.args("--dtype", "float32")
        self.assertEqual(generation_options.resolved_dtype(presets.FLUX1_SCHNELL_PRESET, args, "cuda"), "float32")
        args.dtype = "auto"
        self.assertEqual(generation_options.resolved_dtype(presets.FLUX1_SCHNELL_PRESET, args, "cuda"), "bfloat16")
        self.assertEqual(generation_options.resolved_dtype(presets.SD15_PRESET, args, "cpu"), "float32")


if __name__ == "__main__":
    unittest.main()
