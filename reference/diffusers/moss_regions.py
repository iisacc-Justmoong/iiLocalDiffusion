"""Color-map regional prompting for the vetted MOSS pipeline."""
from contextlib import contextmanager

from regional_prompt import Attention, Region, Selection, verify
import regional_prompt


def resolve(args, pipeline_name):
    return regional_prompt.resolve(args, pipeline_name, option="moss_regions",
                                   expected="MOSSImagePipeline", family="MOSS")


@contextmanager
def execution(pipeline, inputs, selection, image, torch, np, *, map_fit='stretch'):
    verify(selection)
    width, height = inputs.get('width', 1024), inputs.get('height', 1024)
    device = pipeline._execution_device
    groups = regional_prompt.image_groups(image, selection, width, height, device, torch, np, map_fit=map_fit)
    maximum = inputs.get('max_sequence_length', 4096)
    if type(maximum) is not int or not 1 <= maximum <= 4096:
        raise ValueError('max_sequence_length must be between 1 and 4096.')
    prompts = [inputs['prompt'], *(region.prompt for region in selection.regions)]
    encoded, mask = pipeline.encode_prompt(prompts, maximum)
    rows = [encoded[i, mask[i].bool()] for i in range(len(prompts))]
    if any(not row.shape[0] for row in rows):
        raise ValueError('Each MOSS region prompt must produce tokens.')
    text_groups = torch.cat([torch.full((len(row),), i, device=device, dtype=torch.long) for i, row in enumerate(rows)])
    attention = Attention(groups.flatten(), text_groups, selection.isolate_until, torch)
    attention.masks(0, 1)  # Check the attention allocation before modifying pipeline state.
    previous = vars(pipeline).get('_iild_regions')
    had_previous = '_iild_regions' in vars(pipeline)
    original_prompt = inputs.pop('prompt')
    inputs['prompt_embeds'] = torch.cat(rows, 0).unsqueeze(0)
    inputs['prompt_attention_mask'] = torch.ones((1, len(text_groups)), device=device, dtype=torch.bool)
    pipeline._iild_regions = attention
    try:
        yield {'regions': len(selection.regions), 'isolate_until': selection.isolate_until,
               'token_counts': [len(row) for row in rows], 'mask_size': [width // 16, height // 16],
               'map_fit': map_fit}
    finally:
        inputs.pop('prompt_embeds', None)
        inputs.pop('prompt_attention_mask', None)
        inputs['prompt'] = original_prompt
        if had_previous:
            pipeline._iild_regions = previous
        else:
            del pipeline._iild_regions
