"""Synthetic Python-to-native GPU model execution; no pretrained weights."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from NativeUpscalerPythonTests import write_weights, swin_weights
from native_upscaler import Image, NativeUpscaler, Options, upscale_image


class NativeModelGpuPythonTests(unittest.TestCase):
    def test_explicit_device_and_tiling(self):
        model = Path(os.environ['IILD_TEST_WORKSPACE']) / 'python-gpu-swinir.safetensors'
        write_weights(model, swin_weights())
        image = Image(4, 4, 3, bytes([51, 204, 102] * 16))
        options = Options()
        cpu = NativeUpscaler('swinir', model=model, device='cpu', tile_size=2,
                             library=os.environ['IILD_NATIVE_LIBRARY'])
        gpu = NativeUpscaler('swinir', model=model, device=os.environ['IILD_TEST_GPU'], tile_size=2,
                             library=os.environ['IILD_NATIVE_LIBRARY'])
        expected, _ = upscale_image(image, cpu, options)
        actual, report = upscale_image(image, gpu, options)
        self.assertEqual(report['details']['actual_device'], os.environ['IILD_TEST_GPU'])
        self.assertEqual(report['details']['tile_size'], '2')
        self.assertGreater(int(report['details']['gpu_convolutions']), 0)
        self.assertGreater(int(report['details']['gpu_projections']), 0)
        self.assertEqual((actual.width, actual.height), (8, 8))
        self.assertTrue(all(abs(a-b) <= 1 for a,b in zip(actual.pixels, expected.pixels)))


if __name__ == '__main__':
    unittest.main()
