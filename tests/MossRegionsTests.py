"""MOSS region manifest and existing color-map resource contracts."""
from pathlib import Path
import json
import sys
import unittest
from types import SimpleNamespace
import resource_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference/diffusers'))
import moss_regions


class MossRegionsTests(unittest.TestCase):
    def test_reference_style_manifest_resolves_existing_control_files(self):
        with resource_fixture.TemporaryDirectory(prefix='moss-regions-') as temporary:
            root = Path(temporary)
            for name in ('colors.png', 'depth.png', 'control.safetensors'):
                (root / name).write_bytes(b'fixture')
            manifest = {'map': 'colors.png', 'regions': [{'color': '#ff1a1a', 'prompt': 'person',
                        'loras': [], 'label': 'character'}], 'isolate_until': .2, 'layout': 'native',
                        'control': {'adapter': 'control.safetensors', 'color_map': 'colors.png',
                                    'depth': 'depth.png', 'strength': 1., 'color_end': .5, 'depth_end': .5}}
            path = root / 'regions.json'
            path.write_text(json.dumps(manifest))
            args = SimpleNamespace(moss_regions=str(path), controlnet=None, control_color_image=None,
                control_image=None, controlnet_scale=None, control_color_end=None, control_depth_end=None,
                inputs={'prompt': 'a room'}, offload='none')
            selection = moss_regions.resolve(args, 'MOSSImagePipeline')
            self.assertEqual(selection.map_path, str(root / 'colors.png'))
            self.assertEqual(args.controlnet, str(root / 'control.safetensors'))
            self.assertEqual(args.control_image, str(root / 'depth.png'))
            self.assertEqual(args.control_color_image, str(root / 'colors.png'))
            self.assertEqual(args.control_color_end, .5)
            self.assertEqual(selection.regions[0].color, (255, 26, 26))
            moss_regions.verify(selection)
            path.write_text('{}')
            with self.assertRaises(RuntimeError): moss_regions.verify(selection)

    def test_region_manifest_rejects_unsupported_features_and_other_families(self):
        with resource_fixture.TemporaryDirectory(prefix='moss-regions-') as temporary:
            root = Path(temporary)
            (root / 'map.png').write_bytes(b'fixture')
            path = root / 'regions.json'
            args = SimpleNamespace(moss_regions=str(path), controlnet=None, control_color_image=None,
                control_image=None, controlnet_scale=None, control_color_end=None, control_depth_end=None,
                inputs={'prompt': 'scene'}, offload='none')
            for regions in ([{'color': '#zz0000', 'prompt': 'x'}],
                            [{'color': '#ff0000', 'prompt': 'x', 'loras': ['other']}],
                            [{'color': '#ff0000', 'prompt': 'x'}, {'color': '#ff0000', 'prompt': 'y'}]):
                path.write_text(json.dumps({'map': 'map.png', 'regions': regions}))
                with self.assertRaises(ValueError): moss_regions.resolve(args, 'MOSSImagePipeline')
            with self.assertRaises(ValueError): moss_regions.resolve(args, 'Krea2Pipeline')


if __name__ == '__main__':
    unittest.main()
