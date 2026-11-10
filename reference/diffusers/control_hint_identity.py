"""Canonical identities for in-memory ControlNet preprocessor outputs."""
import hashlib

from depth_preprocessing import FloatDepth


def fingerprint(hint, np):
    """Describe the exact preprocessed pixels before model-specific resizing."""
    if isinstance(hint, FloatDepth):
        values = np.asarray(hint.values.detach().cpu().numpy(), dtype='<f4', order='C')
        if values.ndim != 2 or not all(values.shape):
            raise ValueError('Relative-depth preprocessor output must be a nonempty 2D tensor.')
        height, width = values.shape
        pixels = memoryview(values).cast('B')
        encoding = 'float32-le-row-major'
    else:
        width, height = hint.size
        if hint.mode != 'RGB' or width < 1 or height < 1:
            raise ValueError('Control preprocessor output must be a nonempty RGB image.')
        pixels = hint.tobytes()
        if len(pixels) != width * height * 3:
            raise ValueError('Control preprocessor RGB buffer has an invalid size.')
        encoding = 'rgb8-row-major'
    return {'encoding': encoding, 'size': [width, height], 'sha256': hashlib.sha256(pixels).hexdigest()}
