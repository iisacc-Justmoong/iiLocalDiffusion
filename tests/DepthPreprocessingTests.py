"""Local-only Depth Anything request and package contracts."""
import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import resource_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import depth_preprocessing as depth
import generic_controlnet
import control_hint_identity


class DepthPreprocessingTests(unittest.TestCase):
    def test_preprocessed_rgb_fingerprint_includes_raw_pixels_and_size(self):
        class Image:
            mode, size = 'RGB', (2, 1)
            def tobytes(self): return bytes([0, 1, 2, 3, 4, 5])
        info = control_hint_identity.fingerprint(Image(), None)
        self.assertEqual(info, {'encoding': 'rgb8-row-major', 'size': [2, 1],
                                'sha256': hashlib.sha256(bytes([0, 1, 2, 3, 4, 5])).hexdigest()})

    def setUp(self):
        self.tmp = resource_fixture.TemporaryDirectory(prefix='depth-preprocess-')
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        (self.folder / 'config.json').write_text(json.dumps({'model_type': 'depth_anything',
            'architectures': ['DepthAnythingForDepthEstimation'], 'depth_estimation_type': 'relative'}))
        (self.folder / 'preprocessor_config.json').write_text(json.dumps({'image_processor_type': 'DPTImageProcessor'}))
        (self.folder / 'model.safetensors').write_bytes(b'fixture')

    def test_depth_requires_local_explicit_package(self):
        selection = depth.resolve('depth', str(self.folder))
        self.assertEqual(selection.directory, str(self.folder.resolve()))
        self.assertEqual(selection.precision, 'float32')
        self.assertEqual(depth.resolve('depth', str(self.folder), 'float16').precision, 'float16')
        self.assertIsNone(depth.resolve(None, None))
        for mode, path in (('depth', None), ('depth', 'https://example.com/model'), ('canny', str(self.folder))):
            with self.subTest(mode=mode, path=path), self.assertRaises(ValueError):
                depth.resolve(mode, path)
        for mode, precision in (('depth', 'bfloat16'), ('canny', 'float16')):
            with self.subTest(mode=mode, precision=precision), self.assertRaises(ValueError):
                depth.resolve(mode, str(self.folder) if mode == 'depth' else None, precision)

    def test_metric_custom_and_incomplete_packages_are_rejected(self):
        config = self.folder / 'config.json'
        for data in ({'model_type': 'other'}, {'model_type': 'depth_anything', 'depth_estimation_type': 'metric'},
                     {'model_type': 'depth_anything', 'auto_map': {'AutoModel': 'unsafe.Custom'}}):
            config.write_text(json.dumps(data))
            with self.subTest(data=data), self.assertRaises(ValueError):
                depth.resolve('depth', str(self.folder))
        config.write_text('{"model_type":"depth_anything"}')
        (self.folder / 'model.safetensors').unlink()
        with self.assertRaisesRegex(ValueError, 'safetensors'):
            depth.resolve('depth', str(self.folder))

    def test_processor_class_is_constrained(self):
        (self.folder / 'preprocessor_config.json').write_text('{"image_processor_type":"OtherProcessor"}')
        with self.assertRaisesRegex(ValueError, 'DPTImageProcessor'):
            depth.resolve('depth', str(self.folder))

    def test_sharded_weights_are_local_and_complete(self):
        (self.folder / 'model.safetensors').unlink()
        index = self.folder / 'model.safetensors.index.json'
        shard = self.folder / 'model-00001-of-00001.safetensors'
        shard.write_bytes(b'fixture')
        index.write_text(json.dumps({'weight_map': {'tensor': shard.name}}))
        self.assertEqual(depth.resolve('depth', str(self.folder)).directory, str(self.folder.resolve()))
        index.write_text(json.dumps({'weight_map': {'tensor': '../outside.safetensors'}}))
        with self.assertRaisesRegex(ValueError, 'shard'):
            depth.resolve('depth', str(self.folder))

    def test_special_adapters_dispatch_depth_and_reject_absent_image(self):
        args = SimpleNamespace(controlnet='adapter', control_image='source.png', controlnet_config=None,
            controlnet_variant=None, controlnet_scale=None, control_guidance_start=None, control_guidance_end=None,
            guess_mode=None, control_mode=None, control_preprocess='depth', control_preprocessor_library=None,
            control_depth_model=str(self.folder), control_color_image=None, control_color_end=None,
            control_depth_end=None, pipeline_class=None)
        for family, module in (('Krea2Pipeline', 'krea2_controlnet'), ('Flux2Pipeline', 'flux2_controlnet'),
                               ('MOSSImagePipeline', 'moss_controlnet')):
            with self.subTest(family=family), patch.object(getattr(generic_controlnet, module), 'resolve', return_value=family):
                self.assertEqual(generic_controlnet.resolve(args, {'_class_name': family}), family)
                args.control_image = None
                with self.assertRaisesRegex(ValueError, 'requires --control-image'):
                    generic_controlnet.resolve(args, {'_class_name': family})
                args.control_image = 'source.png'


if __name__ == '__main__':
    unittest.main()
