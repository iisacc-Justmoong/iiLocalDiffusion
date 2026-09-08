"""Live preview protocol, real tensor decoding and sampler isolation."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from generation_preview import DenoisingPreview, attach_preview, validate_preview_location
try:
    import torch
    from diffusers.image_processor import VaeImageProcessor
    from PIL import Image
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "The installed Torch/Diffusers/Pillow runtime is required")
class GenerationPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "build")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

        class Vae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.config = SimpleNamespace(scaling_factor=0.5, force_upcast=False)
                self.inputs = []

            @property
            def dtype(self):
                return self.weight.dtype

            def decode(self, latents, return_dict=False):
                self.inputs.append(latents.clone())
                return (torch.nn.functional.interpolate(latents[:, :3].tanh(), scale_factor=8),)

        class Pipeline:
            _callback_tensor_inputs = ["latents"]

            def __call__(self, callback_on_step_end=None, callback_on_step_end_tensor_inputs=None):
                pass

        self.pipeline = Pipeline()
        self.pipeline.vae = Vae()
        self.pipeline.image_processor = VaeImageProcessor()
        self.pipeline.scheduler = SimpleNamespace(timesteps=torch.tensor([3, 2, 1]))

    def test_every_step_is_atomic_unique_and_does_not_change_sampler_tensors_or_rng(self):
        arguments = {}
        attach_preview(self.pipeline, arguments, self.directory / "preview", torch, 64, 64)
        callback = arguments["callback_on_step_end"]
        self.assertEqual(arguments["callback_on_step_end_tensor_inputs"], ["latents"])
        images = []
        for step in range(3):
            latents = torch.full((1, 4, 8, 8), step / 4)
            before, rng = latents.clone(), torch.get_rng_state()
            values, stdout = {"latents": latents}, io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertIs(callback(self.pipeline, step, 3 - step, values), values)
            self.assertTrue(torch.equal(latents, before))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            event = json.loads(stdout.getvalue().removeprefix("IILD_PREVIEW "))
            self.assertEqual((event["schema"], event["step"], event["total_steps"]), ("iild-preview-v1", step + 1, 3))
            with Image.open(callback.directory / event["image"]) as image:
                image.load()
                self.assertEqual(image.size, (64, 64))
                images.append(image.tobytes())
        self.assertEqual(len(set(images)), 3)
        self.assertEqual(len(list(callback.directory.iterdir())), 3)
        self.assertFalse((callback.directory / "generation.json").exists())

    def test_vae_normalization_upcast_and_dtype_restore(self):
        vae = self.pipeline.vae
        vae.to(dtype=torch.float16)
        vae.config.force_upcast = True
        vae.config.latents_mean, vae.config.latents_std = [0.1] * 4, [0.2] * 4
        callback = DenoisingPreview(self.directory / "preview", torch, 64, 64)
        with contextlib.redirect_stdout(io.StringIO()):
            callback(self.pipeline, 0, 3, {"latents": torch.ones((1, 4, 8, 8), dtype=torch.float16)})
        self.assertEqual(vae.inputs[0].dtype, torch.float32)
        self.assertTrue(torch.allclose(vae.inputs[0], torch.full((1, 4, 8, 8), 0.5)))
        self.assertEqual(vae.dtype, torch.float16)

    def test_decode_retains_versioned_tensors_for_later_device_placement(self):
        callback = DenoisingPreview(self.directory / "preview", torch, 64, 64)
        with contextlib.redirect_stdout(io.StringIO()):
            callback(self.pipeline, 0, 3, {"latents": torch.ones((1, 4, 8, 8), requires_grad=True)})
        retained = self.pipeline.vae.inputs[0]
        self.assertFalse(retained.requires_grad)
        self.assertFalse(torch.is_inference(retained))
        # A parameter restored from these tensors must work outside the old call.
        parameter = torch.nn.Parameter(retained)
        self.assertIsInstance(parameter._version, int)

    def test_flux_packed_latents_use_pipeline_unpacking_and_shift(self):
        class FluxPreviewPipeline(type(self.pipeline)):
            vae_scale_factor = 8

            def _unpack_latents(self, latents, height, width, scale):
                self.unpack_shape = (height, width, scale)
                return latents.reshape(1, 4, 8, 8)
        pipeline = FluxPreviewPipeline()
        pipeline.__dict__.update(self.pipeline.__dict__)
        pipeline.vae.config.shift_factor = 0.1
        callback = DenoisingPreview(self.directory / "preview", torch, 64, 64)
        with contextlib.redirect_stdout(io.StringIO()):
            callback(pipeline, 0, 3, {"latents": torch.ones((1, 16, 16))})
        self.assertEqual(pipeline.unpack_shape, (64, 64, 8))
        self.assertTrue(torch.allclose(pipeline.vae.inputs[0], torch.full((1, 4, 8, 8), 2.1)))

    def test_invalid_latents_and_redirected_paths_never_publish_a_frame(self):
        callback = DenoisingPreview(self.directory / "preview", torch, 64, 64)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            callback(self.pipeline, 0, 3, {"latents": torch.full((1, 4, 8, 8), float("nan"))})
        self.assertEqual(list(callback.directory.iterdir()), [])
        callback.directory.rmdir()
        outside = self.directory / "outside"
        outside.mkdir()
        callback.directory.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "redirected"):
            callback(self.pipeline, 0, 3, {"latents": torch.zeros((1, 4, 8, 8))})
        self.assertEqual(list(outside.iterdir()), [])
        with self.assertRaisesRegex(ValueError, "redirected"):
            DenoisingPreview(callback.directory, torch, 64, 64)

    def test_existing_artifacts_and_callbacks_are_preserved(self):
        (self.directory / "keep").write_text("existing")
        with self.assertRaisesRegex(ValueError, "empty"):
            DenoisingPreview(self.directory, torch, 64, 64)
        with self.assertRaisesRegex(ValueError, "existing"):
            attach_preview(self.pipeline, {"callback_on_step_end": object()}, self.directory / "preview", torch, 64, 64)
        with self.assertRaisesRegex(ValueError, "does not expose"):
            attach_preview(SimpleNamespace(__call__=lambda: None), {}, self.directory / "preview", torch, 64, 64)
        self.assertEqual((self.directory / "keep").read_text(), "existing")
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_preview_location(self.directory / "preview", self.directory)
        validate_preview_location(self.directory / "preview", self.directory / "final")


if __name__ == "__main__":
    unittest.main()
