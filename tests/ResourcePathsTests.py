#!/usr/bin/env python3
"""Check default generation paths without running a model or service."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import comfyui_runtime
import generate
import generate_any
import inspect_pipeline
import local_image
import model_merge_options
import native_image
import remote_generation
import standalone_image
import unified_image
import video_options


class ResourcePathsTests(unittest.TestCase):
    def test_default_outputs_are_user_resources(self):
        output = ROOT / "resources/outputs/reference"
        for path in (generate.DEFAULT_OUTPUT, inspect_pipeline.DEFAULT_OUTPUT,
                     comfyui_runtime.DEFAULT_OUTPUT_DIRECTORY, generate_any.DEFAULT_OUTPUT_DIRECTORY,
                     local_image.DEFAULT_JOBS_DIRECTORY, model_merge_options.DEFAULT_DIRECTORY,
                     native_image.DEFAULT_OUTPUT, remote_generation.DEFAULT_OUTPUT,
                     standalone_image.DEFAULT_OUTPUT_DIRECTORY, unified_image.DEFAULT_OUTPUT_DIRECTORY,
                     video_options.DEFAULT_OUTPUT):
            with self.subTest(path=path):
                self.assertTrue(path.is_relative_to(output))
                self.assertFalse(path.is_relative_to(ROOT / "build"))

    def test_caches_and_runtime_sources_have_separate_resource_locations(self):
        state = ROOT / "resources/.state"
        for path in (generate.DEFAULT_CACHE_DIRECTORY, generate.DEFAULT_XET_CACHE_DIRECTORY,
                     generate_any.DEFAULT_CACHE, local_image.DEFAULT_CONVERSION_DIRECTORY,
                     local_image.DEFAULT_PYTHON, model_merge_options.DEFAULT_CONVERSION_CACHE,
                     video_options.DEFAULT_CACHE):
            with self.subTest(path=path):
                self.assertTrue(path.is_relative_to(state))
        self.assertEqual(local_image.DEFAULT_SOURCE, ROOT / "resources/dependencies/ComfyUI")


if __name__ == "__main__":
    unittest.main()
