import os
import importlib.util
import json
import math
import struct
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference' / 'diffusers'))
from native_upscaler import Image, Options, NativeUpscaler, upscale_image

def write_weights(path, tensors):
    header, data = {}, bytearray()
    for name, (shape, values) in tensors.items():
        start = len(data)
        data.extend(struct.pack('<' + 'f' * len(values), *values))
        header[name] = {'dtype': 'F32', 'shape': shape, 'data_offsets': [start, len(data)]}
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + data)


def swin_weights():
    def conv_identity(out, inp):
        values = [0.0] * (out * inp * 9)
        for i in range(min(out, inp)):
            values[(i * inp + i) * 9 + 4] = 1.0
        return values
    block = 'layers.0.residual_group.blocks.0.'
    tensors = {
        'conv_first.weight': ([2, 3, 3, 3], conv_identity(2, 3)), 'conv_first.bias': ([2], [0.] * 2),
        block + 'attn.relative_position_bias_table': ([9, 1], [0.] * 9),
        block + 'attn.qkv.weight': ([6, 2], [0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., -1.]),
        block + 'attn.qkv.bias': ([6], [0.] * 6),
        block + 'attn.proj.weight': ([2, 2], [1., 0., 0., 1.]), block + 'attn.proj.bias': ([2], [0., 0.]),
        'layers.0.conv.weight': ([2, 2, 3, 3], conv_identity(2, 2)), 'layers.0.conv.bias': ([2], [0., 0.]),
        'norm.weight': ([2], [1., 1.]), 'norm.bias': ([2], [0., 0.]),
        'conv_after_body.weight': ([2, 2, 3, 3], conv_identity(2, 2)), 'conv_after_body.bias': ([2], [0., 0.]),
        'upsample.0.bias': ([12], [0.] * 12),
    }
    for name in ('norm1.', 'norm2.'):
        tensors[block + name + 'weight'] = ([2], [1., 1.])
        tensors[block + name + 'bias'] = ([2], [0., 0.])
    for name in ('mlp.fc1.', 'mlp.fc2.'):
        tensors[block + name + 'weight'] = ([2, 2], [0.] * 4)
        tensors[block + name + 'bias'] = ([2], [0., 0.])
    up = [0.] * (12 * 2 * 9)
    for sub in range(4):
        up[(sub * 2) * 9 + 4] = .1
        up[((sub + 4) * 2 + 1) * 9 + 4] = .1
    tensors['upsample.0.weight'] = ([12, 2, 3, 3], up)
    return tensors


class Tests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow is required for the Hires image adapter')
    def test_native_model_reaches_existing_hires_image_adapter(self):
        from PIL import Image as PillowImage
        import hires
        path = Path(os.environ['IILD_TEST_WORKSPACE']) / 'hires-srvgg.safetensors'
        write_weights(path, {
            'body.0.weight': ([1, 3, 3, 3], [0.0] * 27),
            'body.0.bias': ([1], [0.2]),
            'body.1.weight': ([1], [0.5]),
            'body.2.weight': ([12, 1, 3, 3], [0.0] * 108),
            'body.2.bias': ([12], [0.1] * 12),
        })
        provider = NativeUpscaler('srvgg', model=path, device='cpu', threads=2,
                                  library=os.environ['IILD_NATIVE_LIBRARY'])
        result, reports = hires.upscale_images_with_provider(
            [PillowImage.new('RGB', (2, 2), (51, 102, 153))], 3, 3, provider, Options())
        self.assertEqual(result[0].size, (3, 3))
        self.assertEqual(reports[0]['details']['architecture'], 'srvgg')
        self.assertEqual((reports[0]['applied_scale'], reports[0]['hires_resampled']), (2, True))
        self.assertTrue(all(abs(a - b) <= 1 for a, b in zip(result[0].getpixel((0, 0)), (77, 128, 179))))

    def test_native_and_injected_upscaler(self):
        image = Image(2, 2, 1, bytes([0, 255, 255, 0]))
        events = []
        provider = NativeUpscaler('nearest', library=os.environ['IILD_NATIVE_LIBRARY'])
        out, report = upscale_image(image, provider, Options(scale=2, passes=2),
                                    progress=lambda done, total: events.append((done, total)) or True)
        self.assertEqual((out.width, out.height, len(out.pixels)), (8, 8, 64))
        self.assertEqual((report['upscaler'], events), ('nearest', [(1, 2), (2, 2)]))
        self.assertEqual(report['passes'], [{'width': 4, 'height': 4}, {'width': 8, 'height': 8}])
        auto, auto_report = upscale_image(image, provider, Options())
        self.assertEqual((auto.width, auto.height, auto_report['applied_scale']), (4, 4, 2))
        class Supplied:
            def upscale(self, image, options, *, progress=None):
                return Image(6, 6, image.channels, bytes(36 * image.channels)), {'received': options.scale, 'settings': options.settings}
        self.assertEqual(upscale_image(image, Supplied(), Options(scale=3, settings={'x': 'y'}))[1],
                         {'received': 3, 'settings': {'x': 'y'}})
        class CustomDefault:
            def upscale(self, image, options, *, progress=None):
                return Image(6, 6, image.channels, bytes(36 * image.channels)), {'applied_scale': 3}
        self.assertEqual(upscale_image(image, CustomDefault(), Options())[0].width, 6)
        class WrongShape:
            def upscale(self, image, options, *, progress=None):
                return image, {}
        with self.assertRaisesRegex(RuntimeError, 'invalid image'):
            upscale_image(image, WrongShape(), Options(scale=3))

    def test_native_srvgg_model(self):
        path = Path(os.environ['IILD_TEST_WORKSPACE']) / 'python-srvgg.safetensors'
        tensors = {
            'body.0.weight': ([1, 3, 3, 3], [0.0] * 27),
            'body.0.bias': ([1], [-0.2]),
            'body.1.weight': ([1], [0.5]),
            'body.2.weight': ([12, 1, 3, 3], [0.0] * 108),
            'body.2.bias': ([12], [0.1] * 12),
        }
        write_weights(path, tensors)
        provider = NativeUpscaler('srvgg', model=path, device='cpu', tile_size=0, threads=2,
                                  max_weight_bytes=1024 * 1024, max_working_bytes=1024 * 1024,
                                  library=os.environ['IILD_NATIVE_LIBRARY'])
        image, report = upscale_image(Image(1, 1, 3, bytes([51, 102, 153])), provider, Options())
        self.assertEqual((image.width, image.height, report['applied_scale']), (2, 2, 2))
        self.assertEqual(report['details']['architecture'], 'srvgg')
        self.assertEqual(report['details']['actual_device'], 'cpu')
        self.assertTrue(all(abs(actual - expected) <= 1 for actual, expected in zip(image.pixels[:3], (77, 128, 179))))
        with self.assertRaises(ValueError):
            NativeUpscaler('srvgg', model=path, swinir_image_range=2, library=os.environ['IILD_NATIVE_LIBRARY'])
        with self.assertRaisesRegex(RuntimeError, 'SwinIR|tensor'):
            wrong = NativeUpscaler('swinir', model=path, tile_size=0, library=os.environ['IILD_NATIVE_LIBRARY'])
            upscale_image(Image(1, 1, 3, bytes([51, 102, 153])), wrong, Options())

    def test_native_swinir_controls(self):
        path = Path(os.environ['IILD_TEST_WORKSPACE']) / 'python-swinir.safetensors'
        write_weights(path, swin_weights())
        provider = NativeUpscaler('swinir', model=path, device='cpu', threads=2,
                                  swinir_image_range=2, swinir_qk_scale=.5,
                                  library=os.environ['IILD_NATIVE_LIBRARY'])
        pixels = bytes([51, 204, 102] * 9)
        output, report = upscale_image(Image(3, 3, 3, pixels), provider, Options())
        self.assertEqual((output.width, output.height), (6, 6))
        self.assertEqual(report['details']['architecture'], 'swinir')
        self.assertEqual(float(report['details']['image_range']), 2.0)
        self.assertEqual(float(report['details']['qk_scales']), 0.5)
        a, b = (51 / 255 - .4488) * 2, (204 / 255 - .4371) * 2
        n = (a - b) / math.sqrt((a - b) ** 2 + 4e-5)
        ga, gb = 2 * a - n, 2 * b + n
        n2 = (ga - gb) / math.sqrt((ga - gb) ** 2 + 4e-5)
        expected = round((.4488 + .1 * (a + n2) / 2) * 255)
        self.assertLessEqual(abs(output.pixels[0] - expected), 1)
        tiled = NativeUpscaler('swinir', model=path, device='cpu', tile_size=2,
                               swinir_image_range=2, swinir_qk_scale=.5,
                               library=os.environ['IILD_NATIVE_LIBRARY'])
        tiled_output, tiled_report = upscale_image(Image(3, 3, 3, pixels), tiled, Options())
        self.assertEqual(tiled_output.pixels, output.pixels)
        self.assertEqual(tiled_report['details']['tile_size'], '2')
        for device, number in (('metal', 2), ('directx', 3), ('vulkan', 4)):
            selection = NativeUpscaler('swinir', model=path, device=device, tile_size=64,
                                       library=os.environ['IILD_NATIVE_LIBRARY'])
            wire = selection._bridge_request(Options())
            self.assertEqual((wire.upscaler.image.cpu, wire.upscaler.image.tile_size), (number, 64))

    def test_errors_and_cancellation(self):
        image = Image(1, 1, 3, b'abc')
        provider = NativeUpscaler('bilinear', library=os.environ['IILD_NATIVE_LIBRARY'])
        with self.assertRaisesRegex(RuntimeError, 'cancel'):
            upscale_image(image, provider, Options(), progress=lambda done, total: False)
        with self.assertRaises(ValueError):
            upscale_image(Image(2, 2, 3, b'x'), provider, Options())
        with self.assertRaises((ValueError, RuntimeError)):
            upscale_image(image, provider, Options(max_output_bytes=1))
        with self.assertRaises(ValueError):
            upscale_image(image, provider, Options(settings={'ignored': 'no'}))
        with self.assertRaisesRegex(RuntimeError, 'safetensors|gguf'):
            upscale_image(image, NativeUpscaler('esrgan', model='x.onnx', library=os.environ['IILD_NATIVE_LIBRARY']), Options(scale=4))
        def fail_callback(done, total):
            raise LookupError('callback failure')
        with self.assertRaisesRegex(LookupError, 'callback failure'):
            upscale_image(image, provider, Options(), progress=fail_callback)

if __name__ == '__main__':
    unittest.main()
