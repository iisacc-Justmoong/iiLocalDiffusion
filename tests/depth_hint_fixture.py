"""Offline Depth Anything hints for small ControlNet pipeline tests."""
import numpy as np
import torch
from PIL import Image
from transformers import (DepthAnythingConfig, DepthAnythingForDepthEstimation,
                          Dinov2Config, DPTImageProcessor)

import depth_preprocessing as depth
import resource_fixture


def save_model(folder):
    # Isolate model initialization from the pipeline's own random state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(834)
        backbone = Dinov2Config(image_size=56, patch_size=14, hidden_size=32,
            num_hidden_layers=4, num_attention_heads=4, intermediate_size=64,
            out_indices=[1, 2, 3, 4], reshape_hidden_states=False)
        config = DepthAnythingConfig(backbone_config=backbone, reassemble_hidden_size=32,
            neck_hidden_sizes=[8, 16, 32, 32], fusion_hidden_size=16, head_hidden_size=8)
        model = DepthAnythingForDepthEstimation(config)
    processor = DPTImageProcessor(size={'height': 56, 'width': 56},
                                  keep_aspect_ratio=False, ensure_multiple_of=14)
    model.save_pretrained(folder, safe_serialization=True)
    processor.save_pretrained(folder)


def hints(*, preserve_float=False, precision='float32'):
    pixels = np.random.default_rng(47).integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
    with resource_fixture.TemporaryDirectory(prefix='depth-control-runtime-') as folder:
        save_model(folder)
        selection = depth.resolve('depth', folder, precision)
        outputs = [depth.apply(Image.fromarray(image), selection, torch, np, Image, 'cpu',
                               (64, 64) if preserve_float else None,
                               preserve_float=preserve_float)[0] for image in pixels]
    if preserve_float:
        if torch.equal(outputs[0].values, outputs[1].values):
            raise AssertionError('Depth Anything fixture produced identical float hints')
    elif outputs[0].tobytes() == outputs[1].tobytes():
        raise AssertionError('Depth Anything fixture produced identical RGB hints')
    return outputs
