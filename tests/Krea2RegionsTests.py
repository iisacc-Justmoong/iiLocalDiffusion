"""Krea 2 color-map regional prompt request contracts."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import resource_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference/diffusers'))
import krea2_regions


class Krea2RegionsTests(unittest.TestCase):
    def test_reference_style_control_defaults_to_color_depth_adapter(self):
        with resource_fixture.TemporaryDirectory(prefix='krea-regions-') as temporary:
            root = Path(temporary)
            for name in ('map.png', 'depth.png', 'control.safetensors'):
                (root / name).write_bytes(b'fixture')
            path = root / 'regions.json'
            path.write_text(json.dumps({'map': 'map.png', 'regions': [{'color': '#f01a00', 'prompt': 'person'}],
                'control': {'adapter': 'control.safetensors', 'color_map': 'map.png', 'depth': 'depth.png'}}))
            args = SimpleNamespace(krea2_regions=str(path), controlnet=None, control_color_image=None,
                control_image=None, controlnet_scale=None, control_color_end=None, control_depth_end=None,
                controlnet_format=None, inputs={'prompt': 'scene'}, offload='none')
            selection = krea2_regions.resolve(args, 'Krea2Pipeline')
            self.assertEqual(selection.map_path, str(root / 'map.png'))
            self.assertEqual(args.controlnet_format, 'krea2-color-depth')
            self.assertEqual(args.control_image, str(root / 'depth.png'))
            self.assertEqual(args.control_color_image, str(root / 'map.png'))
            with self.assertRaises(ValueError): krea2_regions.resolve(args, 'MOSSImagePipeline')


if __name__ == '__main__':
    unittest.main()
