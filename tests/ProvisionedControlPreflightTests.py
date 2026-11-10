"""Resolve real local ControlNet resources without loading or generating images."""
import json
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

import resource_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import generate_any
import generic_controlnet
import krea2_controlnet
import krea2_ostris_control
import native_control_preprocessing


class ProvisionedControlPreflightTests(unittest.TestCase):
    def setUp(self):
        self.mode = os.environ.get('IILD_VERIFY_CONTROL_PREFLIGHT')
        if self.mode not in ('header', 'real'):
            self.skipTest('Set IILD_VERIFY_CONTROL_PREFLIGHT=header or real to audit local resource selection')
        self.temporary = resource_fixture.TemporaryDirectory(prefix='provisioned-control-preflight-')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.image = self.folder / 'reference.png'
        # Resolution checks file identity only; no decoding or inference occurs.
        self.image.write_bytes(b'preflight-placeholder')

    def model_path(self, filename, fixture_name):
        if self.mode == 'real':
            return ROOT / 'resources/models' / filename
        header = json.loads((ROOT / 'resources/test-fixtures' / fixture_name).read_text())['header']
        content = json.dumps(header).encode('utf-8')
        path = self.folder / filename
        path.write_bytes(struct.pack('<Q', len(content)) + content)
        return path

    def resolve(self, arguments):
        with patch.dict(os.environ, {'IILD_MODEL_VALIDATION': 'metadata'}):
            parsed = generate_any.build_parser().parse_args(arguments + [
                '--seed', '42', '--dtype', 'bfloat16', '--offload', 'none', '--no-default-modifiers',
                '--output-dir', str(self.folder / 'unused-output')])
            selected = generate_any.resolve_arguments(parsed)
            report = generate_any.configuration(selected)
        self.assertFalse((self.folder / 'unused-output').exists())
        return selected, report

    def test_krea_depth_and_prepared_pose_resources_resolve(self):
        model = self.model_path('krea2_turbo_fp8_scaled.safetensors', 'krea2-turbo-fp8-header.json')
        common = ['--model', str(model), '--model-config-preset', 'krea2-turbo', '--krea2-variant', 'turbo',
                  '--krea2-fp8-storage', 'cpu-stream', '--control-image', str(self.image),
                  '--prompt', 'a sunlit studio', '--width', '1024', '--height', '1024',
                  '--steps', '8', '--guidance-scale', '0']
        depth = ROOT / 'resources/models/krea2-depth-control-lora.safetensors'
        depth_model = ROOT / 'resources/depth-estimators/depth-anything-v2-large'
        selected, report = self.resolve(common + ['--controlnet', str(depth), '--control-preprocess', 'depth',
                                                  '--control-depth-model', str(depth_model)])
        self.assertIsInstance(selected.generic_controlnet, krea2_controlnet.Selection)
        self.assertEqual(Path(selected.generic_controlnet.weight.resolved_file), depth.resolve())
        self.assertEqual(Path(selected.depth_preprocessor.directory), depth_model.resolve())
        self.assertEqual(selected.text_encoder_status, 'fallback')
        self.assertEqual(selected.vae_status, 'fallback')
        self.assertEqual(Path(selected.vae_selection.directory), ROOT / 'resources/vae/qwen-image/vae')
        self.assertEqual(report['krea2']['variant'], 'turbo')
        pose = ROOT / 'resources/models/krea2-turbo-openpose-controlnet.safetensors'
        selected, report = self.resolve(common + ['--controlnet', str(pose),
                                                  '--controlnet-format', 'krea2-ostris-edit'])
        self.assertIsInstance(selected.generic_controlnet, krea2_ostris_control.Selection)
        self.assertEqual(Path(selected.generic_controlnet.weight.resolved_file), pose.resolve())
        self.assertEqual(report['text_encoder']['vision_enabled'], True)
        self.assertIsNone(selected.depth_preprocessor)

    def test_flux_canny_resources_and_embedded_components_resolve(self):
        model = self.model_path('flux1-schnell-fp8.safetensors', 'flux1-schnell-fp8-full-header.json')
        adapter = ROOT / 'resources/models/flux1-instantx-canny'
        selected, report = self.resolve([
            '--model', str(model), '--model-config-preset', 'flux1-schnell',
            '--flux-transformer-storage', 'cpu-stream', '--flux-t5-fp8-storage', 'cpu-stream',
            '--controlnet', str(adapter), '--control-image', str(self.image), '--control-preprocess', 'canny',
            '--prompt', 'a city at night', '--width', '1024', '--height', '1024',
            '--steps', '4', '--guidance-scale', '3.5'])
        self.assertIsInstance(selected.generic_controlnet, generic_controlnet.Selection)
        self.assertEqual(Path(selected.generic_controlnet.directory), adapter.resolve())
        self.assertEqual(selected.flux_text_encoder_status['text_encoder_2'], 'embedded')
        self.assertEqual(selected.vae_status, 'model')
        self.assertEqual(report['flux_transformer_storage'], 'cpu-stream')
        self.assertEqual(report['flux_t5_fp8_storage'], 'cpu-stream')
        self.assertTrue(Path(native_control_preprocessing.library_path()).is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
