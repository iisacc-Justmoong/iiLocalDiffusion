"""Explicit in-process image upscaling through iiLocalDiffusion's C++ adapters.

Callers can pass any object implementing ``upscale(image, options, progress=...)``
to ``upscale_image``. No model download, polling or worker is started here.
"""
from contextlib import ExitStack
from dataclasses import dataclass, field
import ctypes as C
import json
import math
import os
from pathlib import Path
import sys


@dataclass(frozen=True)
class Image:
    width: int
    height: int
    channels: int
    pixels: bytes


@dataclass(frozen=True)
class Options:
    scale: float = 0.0  # Zero selects the provider/model default.
    target_width: int = 0
    target_height: int = 0
    passes: int = 1
    max_output_bytes: int = 512 * 1024 * 1024
    settings: dict[str, str] = field(default_factory=dict)


class _Request(C.Structure):
    _fields_ = [('size', C.c_size_t), ('pixels', C.POINTER(C.c_uint8)), ('pixel_bytes', C.c_size_t),
                ('width', C.c_uint32), ('height', C.c_uint32), ('channels', C.c_uint32),
                ('kind', C.c_int32), ('model', C.c_char_p), ('cpu', C.c_int32),
                ('tile_size', C.c_int32), ('threads', C.c_int32), ('scale', C.c_double),
                ('target_width', C.c_uint32), ('target_height', C.c_uint32), ('passes', C.c_int32),
                ('max_output_bytes', C.c_uint64)]


class _RequestV2(C.Structure):
    _fields_ = [('size', C.c_size_t), ('image', _Request), ('max_weight_bytes', C.c_uint64),
                ('max_working_bytes', C.c_uint64), ('srvgg_activation', C.c_int32)]


class _RequestV3(C.Structure):
    _fields_ = [('size', C.c_size_t), ('upscaler', _RequestV2),
                ('swinir_image_range', C.c_float), ('swinir_qk_scale', C.c_float)]


_Progress = C.CFUNCTYPE(C.c_int, C.c_int, C.c_int, C.c_void_p)


def _library_path():
    explicit = os.environ.get('IILD_NATIVE_LIBRARY')
    if explicit:
        return Path(explicit).expanduser().resolve(strict=True)
    here = Path(__file__).resolve()
    name = 'iiLocalDiffusion.dll' if os.name == 'nt' else 'libiiLocalDiffusion' + ('.dylib' if sys.platform == 'darwin' else '.so')
    prefix = here.parents[4] if here.parents[2].name == 'iiLocalDiffusion' else None
    candidates = ([prefix / ('bin' if os.name == 'nt' else 'lib') / name] if prefix else [])
    candidates.append(here.parents[2] / 'build' / name)
    if os.name == 'nt':
        candidates.extend(p.with_name('libiiLocalDiffusion.dll') for p in list(candidates))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError('The native iiLocalDiffusion library was not found.')


def _load_library(path):
    path = Path(path).resolve(strict=True)
    with ExitStack() as directories:
        if os.name == 'nt':
            seen = set()
            for directory in (path.parent, *map(Path, os.environ.get('PATH', '').split(os.pathsep))):
                key = os.path.normcase(str(directory.resolve()))
                if key not in seen and directory.is_dir():
                    seen.add(key)
                    directories.enter_context(os.add_dll_directory(str(directory.resolve())))
        return C.CDLL(str(path))


def _validate_request(image, options, applied_scale=None):
    if (not isinstance(image.width, int) or not isinstance(image.height, int)
        or not 1 <= image.width <= 0xffffffff or not 1 <= image.height <= 0xffffffff
        or image.channels not in (1, 3, 4) or len(image.pixels) != image.width * image.height * image.channels):
        raise ValueError('Invalid image dimensions, channels or byte count.')
    if (not isinstance(options.passes, int) or not 1 <= options.passes <= 8
        or not isinstance(options.max_output_bytes, int) or not 1 <= options.max_output_bytes <= 0xffffffffffffffff
        or not isinstance(options.target_width, int) or not isinstance(options.target_height, int)
        or not 0 <= options.target_width <= 0xffffffff or not 0 <= options.target_height <= 0xffffffff
        or not math.isfinite(options.scale)):
        raise ValueError('Invalid upscale options.')
    target = bool(options.target_width or options.target_height)
    if ((target and (not options.target_width or not options.target_height or options.passes != 1))
        or (not target and options.scale != 0 and options.scale < 1)):
        raise ValueError('Specify a scale or two target dimensions for one pass.')
    scale = options.scale
    if not target and scale == 0:
        if applied_scale is None:
            return None  # Provider resolves its model default at session open.
        if not isinstance(applied_scale, (int, float)) or not math.isfinite(applied_scale) or applied_scale < 1:
            raise RuntimeError('Upscaler did not report a valid default scale.')
        scale = applied_scale
    width, height = image.width, image.height
    for _ in range(options.passes):
        if target:
            width, height = options.target_width, options.target_height
        else:
            scaled_w, scaled_h = width * scale, height * scale
            if not math.isfinite(scaled_w) or not math.isfinite(scaled_h):
                raise ValueError('Upscale dimensions overflow.')
            width, height = math.floor(scaled_w + .5), math.floor(scaled_h + .5)
            if width > 0xffffffff or height > 0xffffffff:
                raise ValueError('Upscale dimensions overflow.')
        if width < image.width or height < image.height or width * height * image.channels > options.max_output_bytes:
            raise ValueError('Upscale output exceeds its limit or would shrink the input.')
    return width, height


class NativeUpscaler:
    """Select a native interpolation filter or a compatible local ESRGAN model.

    ``device`` is a requested backend; 'automatic' does not claim a particular GPU.
    Model resources are loaded only inside each upscale call and then released.
    """
    def __init__(self, kind, *, model=None, device='automatic', tile_size=None, threads=1,
                 max_weight_bytes=1024 * 1024 * 1024, max_working_bytes=512 * 1024 * 1024,
                 srvgg_activation='prelu', swinir_image_range=1.0, swinir_qk_scale=0.0, library=None):
        kinds = {'nearest': 0, 'bilinear': 1, 'bicubic': 2, 'lanczos3': 3, 'esrgan': 4,
                 'srvgg': 5, 'swinir': 6, 'model': 7}
        activations = {'prelu': 0, 'relu': 1, 'leakyrelu': 2}
        if kind not in kinds or device not in ('automatic', 'cpu', 'metal', 'directx', 'vulkan') or srvgg_activation not in activations:
            raise ValueError('Unsupported upscaler, device or SRVGG activation.')
        if kinds[kind] == 4 and device not in ('automatic', 'cpu'):
            raise ValueError('RRDB does not accept this explicit GPU device.')
        if (not math.isfinite(swinir_image_range) or swinir_image_range <= 0
            or not math.isfinite(swinir_qk_scale) or swinir_qk_scale < 0):
            raise ValueError('Invalid SwinIR image range or QK scale.')
        if kinds[kind] < 6 and (swinir_image_range != 1 or swinir_qk_scale != 0):
            raise ValueError('SwinIR settings do not apply to the selected upscaler.')
        if tile_size is None:
            tile_size = 0 if kinds[kind] >= 5 else 128
        if (not isinstance(max_weight_bytes, int) or not 1 <= max_weight_bytes <= 0xffffffffffffffff
            or not isinstance(max_working_bytes, int) or not 1 <= max_working_bytes <= 0xffffffffffffffff):
            raise ValueError('Invalid model resource limit.')
        if not isinstance(tile_size, int) or not 0 <= tile_size <= 8192 or not isinstance(threads, int) or not 1 <= threads <= 256:
            raise ValueError('Invalid tile size or thread count.')
        if kinds[kind] >= 4 and model is None:
            raise ValueError('A model path is required.')
        if kind == 'swinir' and srvgg_activation != 'prelu':
            raise ValueError('SRVGG activation cannot be applied to SwinIR.')
        if kinds[kind] < 4 and (model is not None or device != 'automatic' or tile_size != 128 or threads != 1):
            raise ValueError('Model settings apply only to model upscalers.')
        if kinds[kind] < 5 and (max_weight_bytes != 1024 * 1024 * 1024 or max_working_bytes != 512 * 1024 * 1024
                               or srvgg_activation != 'prelu'):
            raise ValueError('CPU model resource options apply only to native models.')
        self.kind, self.model, self.device = kinds[kind], model, device
        self.tile_size, self.threads = tile_size, threads
        self.max_weight_bytes, self.max_working_bytes = max_weight_bytes, max_working_bytes
        self.srvgg_activation = activations[srvgg_activation]
        self.swinir_image_range, self.swinir_qk_scale = swinir_image_range, swinir_qk_scale
        self._lib = _load_library(library or _library_path())
        self._lib.iild_upscale_v1.argtypes = [C.POINTER(_Request), _Progress, C.c_void_p]
        self._lib.iild_upscale_v1.restype = C.c_void_p
        self._lib.iild_upscale_v2.argtypes = [C.POINTER(_RequestV2), _Progress, C.c_void_p]
        self._lib.iild_upscale_v2.restype = C.c_void_p
        self._lib.iild_upscale_v3.argtypes = [C.POINTER(_RequestV3), _Progress, C.c_void_p]
        self._lib.iild_upscale_v3.restype = C.c_void_p
        self._lib.iild_upscale_metadata_v1.argtypes = [C.c_void_p]
        self._lib.iild_upscale_metadata_v1.restype = C.c_char_p
        self._lib.iild_upscale_pixels_v1.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
        self._lib.iild_upscale_pixels_v1.restype = C.c_void_p
        self._lib.iild_upscale_free_v1.argtypes = [C.c_void_p]
        self._lib.iild_upscale_free_v1.restype = None

    def _bridge_request(self, options, image=None):
        # A Hires request has no input pixels until the generation engine decodes
        # them. Validate scalar options now; native code checks actual dimensions.
        _validate_request(image if image is not None else Image(1, 1, 3, bytes(3)), options)
        if options.settings:
            raise ValueError('Native upscalers do not accept provider-specific settings.')
        pixels = (C.c_uint8 * len(image.pixels)).from_buffer_copy(image.pixels) if image is not None else None
        model = os.fsencode(self.model) if self.model is not None else None
        request = _Request(C.sizeof(_Request), pixels, len(pixels) if pixels is not None else 0,
                           image.width if image else 0, image.height if image else 0, image.channels if image else 0,
                           self.kind, model, {'automatic': 0, 'cpu': 1, 'metal': 2, 'directx': 3, 'vulkan': 4}[self.device],
                           self.tile_size, self.threads,
                           options.scale, options.target_width, options.target_height, options.passes,
                           options.max_output_bytes)
        extended = _RequestV2(C.sizeof(_RequestV2), request, self.max_weight_bytes,
                              self.max_working_bytes, self.srvgg_activation)
        return _RequestV3(C.sizeof(_RequestV3), extended, self.swinir_image_range, self.swinir_qk_scale)

    def upscale(self, image, options, *, progress=None):
        controlled = self._bridge_request(options, image)
        callback_errors = []
        def on_progress(done, total, _):
            try:
                return int(progress is not None and progress(done, total) is False)
            except BaseException as error:
                callback_errors.append(error)
                return 1
        callback = _Progress(on_progress)
        handle = self._lib.iild_upscale_v3(C.byref(controlled), callback, None)
        if not handle:
            raise RuntimeError('Native upscaling could not allocate a result.')
        try:
            if callback_errors:
                raise callback_errors[0]
            metadata = self._lib.iild_upscale_metadata_v1(handle)
            if not metadata:
                raise RuntimeError('Native upscaling did not return metadata.')
            report = json.loads(metadata)
            if report['error'] or report['cancelled']:
                raise RuntimeError(report['error'] or 'Upscaling cancelled.')
            size = C.c_size_t()
            address = self._lib.iild_upscale_pixels_v1(handle, C.byref(size))
            expected = report['width'] * report['height'] * report['channels']
            if not address or size.value != expected:
                raise RuntimeError('Native upscaling returned an invalid image.')
            return Image(report['width'], report['height'], report['channels'], C.string_at(address, size.value)), report
        finally:
            self._lib.iild_upscale_free_v1(handle)


def upscale_image(image, upscaler, options, *, progress=None):
    """Use an explicitly supplied native or application-provided upscaler."""
    expected = _validate_request(image, options)
    result, report = upscaler.upscale(image, options, progress=progress)
    if expected is None:
        expected = _validate_request(image, options, report.get('applied_scale'))
        if expected is None:
            raise RuntimeError('Upscaler did not report its default scale.')
    width, height = expected
    if ((result.width, result.height, result.channels) != (width, height, image.channels)
        or len(result.pixels) != width * height * image.channels):
        raise RuntimeError('Upscaler returned an invalid image.')
    return result, report
