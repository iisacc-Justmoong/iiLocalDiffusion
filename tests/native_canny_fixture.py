"""Real native Canny hints for tiny, offline control-pipeline tests."""
import unittest


def hints():
    from PIL import Image, ImageDraw
    import native_control_preprocessing as preprocessing
    try:
        path = preprocessing.library_path()
    except ValueError as error:
        raise unittest.SkipTest('Build the iiLocalDiffusion shared library for native Canny tests') from error
    library = preprocessing.load_library(path)
    plain = Image.new('RGB', (64, 64), 'black')
    subject = plain.copy()
    ImageDraw.Draw(subject).rectangle((16, 16, 47, 47), fill='white')
    blank = preprocessing.apply_canny(plain, Image, library)
    edges = preprocessing.apply_canny(subject, Image, library)
    if blank.tobytes() == edges.tobytes():
        raise AssertionError('Native Canny fixture did not produce distinct hints')
    return blank, edges
