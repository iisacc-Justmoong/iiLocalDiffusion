#!/usr/bin/env python3
"""Repeated refinement must consume the last result and retain each stage's evidence."""

from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from HiresRuntimeTests import Image, Pipeline, RuntimeFixture
import hires
import presets


class HiresRepeatRuntimeTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.model = temporary.name

    def fixture(self, preset=presets.SD15_PRESET, *, controlnet=False, failure=None, passes=3):
        values = {"model": self.model} if preset.requires_model_override else {}
        fixture = RuntimeFixture(preset, controlnet=controlnet, num_images=2, seed_stride=5, **values)
        fixture.args.hires_passes = passes
        fixture.args.hires_stage_sizes = [[32 * 2**(i + 1)] * 2 for i in range(passes)]
        fixture.args.hires_target_width, fixture.args.hires_target_height = fixture.args.hires_stage_sizes[-1]
        fixture.created = []
        fixture.activation = SimpleNamespace(registered_components=("unet",), active_adapters=("custom",))

        def from_pipe(previous, **options):
            index = len(fixture.created)
            self.assertTrue(previous.hooks_removed)
            self.assertTrue(previous.on_cpu)
            self.assertIs(previous, fixture.base if index == 0 else fixture.created[-1])
            size = fixture.args.hires_stage_sizes[index]
            pipeline = Pipeline(fixture.events, width=size[0], height=size[1], count=2)
            pipeline.scheduler = options["scheduler"]
            pipeline.output = SimpleNamespace(images=[Image(tuple(size), label=f"stage-{index + 1}-{i}") for i in range(2)])
            if failure == index:
                pipeline.finite = False
            fixture.created.append(pipeline)
            return pipeline

        fixture.classes = {hires.PIPELINES[(preset.family, controlnet)]: SimpleNamespace(from_pipe=from_pipe)}
        return fixture

    def test_every_family_and_controlnet_variant_repeats_the_identical_procedure(self):
        for preset in presets.PRESETS.values():
            for controlnet in (False, True):
                with self.subTest(preset=preset.name, controlnet=controlnet):
                    fixture = self.fixture(preset, controlnet=controlnet)
                    helper_images = []

                    def refine(pipeline, request, images, call, *rest):
                        helper_images.append(images)
                        self.assertEqual(call["control_image"], "control-guide")
                        return pipeline(**call), {"fixture": True}

                    context = patch("hires_flux_controlnet.refine_flux_controlnet", side_effect=refine) if preset.family == "flux1-schnell" and controlnet else nullcontext()
                    with context:
                        result = fixture.run()
                    self.assertEqual(len(fixture.created), 3)
                    self.assertEqual(result.request.width, 256)
                    self.assertEqual(result.metadata["requested_passes"], 3)
                    self.assertEqual(result.metadata["completed_passes"], 3)
                    stages = result.metadata["stages"]
                    for index, pipeline in enumerate(fixture.created):
                        call = pipeline.calls[0]
                        source = fixture.images if index == 0 else fixture.created[index - 1].output.images
                        resized = helper_images[index] if helper_images else call["image"]
                        self.assertEqual([image.label for image in resized], [image.label + ":upscaled" for image in source])
                        self.assertEqual(stages[index]["input"], [hires.image_metadata(image) for image in source])
                        self.assertEqual(stages[index]["output"], [hires.image_metadata(image) for image in pipeline.output.images])
                        self.assertEqual(stages[index]["upscale"]["source_size"], list(source[0].size))
                        self.assertEqual(stages[index]["refinement"]["seeds"], [91, 96])
                        self.assertEqual(call["num_inference_steps"], fixture.args.hires_steps)
                        self.assertEqual(call["guidance_scale"], fixture.args.hires_guidance_scale)
                        self.assertNotIn("latents", call)
                        if controlnet:
                            self.assertEqual(call["control_image"], "control-guide")
                        if index:
                            self.assertIsNot(call["generator"][0], fixture.created[index - 1].calls[0]["generator"][0])
                            self.assertIsNot(pipeline.scheduler, fixture.created[index - 1].scheduler)
                    self.assertEqual(result.metadata["refinement"], stages[-1]["refinement"])
                    self.assertEqual(result.metadata["upscale"], stages[-1]["upscale"])
                    self.assertEqual((fixture.args.width, fixture.args.height), (32, 32))

    def test_a_failed_later_stage_stops_the_chain_without_returning_partial_success(self):
        fixture = self.fixture(failure=1, passes=4)
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            fixture.run()
        self.assertEqual(len(fixture.created), 2)
        self.assertEqual(fixture.created[0].output.images[0].size, (64, 64))

    def test_one_pass_keeps_existing_metadata_fields_and_adds_stage_history(self):
        fixture = self.fixture(passes=1)
        result = fixture.run()
        self.assertEqual(result.metadata["completed_passes"], 1)
        self.assertEqual(result.metadata["upscale"]["target_size"], [64, 64])
        self.assertEqual(result.metadata["refinement"]["seeds"], [91, 96])
        self.assertEqual(len(result.metadata["stages"]), 1)


if __name__ == "__main__":
    unittest.main()
