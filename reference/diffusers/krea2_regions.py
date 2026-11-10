"""Color-map regional prompting for the built-in Krea 2 Diffusers pipeline."""
from contextlib import contextmanager
import regional_prompt


def resolve(args, pipeline_name):
    previous_control = args.controlnet
    selection = regional_prompt.resolve(args, pipeline_name, option='krea2_regions',
                                        expected='Krea2Pipeline', family='Krea 2')
    if selection is None:
        return None
    if any(key in args.inputs for key in ('negative_prompt_embeds', 'negative_prompt_embeds_mask')):
        raise ValueError('Krea 2 regions require a string negative prompt rather than supplied embeddings.')
    if args.inputs.get('negative_prompt') is not None and not isinstance(args.inputs['negative_prompt'], str):
        raise ValueError('Krea 2 regions require a single string negative prompt.')
    if args.controlnet and (args.control_color_image is not None or previous_control is None):
        if args.controlnet_format is None:
            args.controlnet_format = 'krea2-color-depth'
    return selection


@contextmanager
def execution(pipeline, inputs, selection, image, torch, np):
    regional_prompt.verify(selection)
    width, height = inputs['width'], inputs['height']
    device = pipeline._execution_device
    groups = regional_prompt.image_groups(image, selection, width, height, device, torch, np)
    maximum = inputs.get('max_sequence_length', 512)
    if type(maximum) is not int or not 1 <= maximum <= 8192:
        raise ValueError('Krea 2 max_sequence_length must be in [1,8192].')
    positive_prompts = [inputs['prompt'], *(region.prompt for region in selection.regions)]
    guidance = inputs['guidance_scale'] > 0
    negative = inputs.get('negative_prompt') or ''
    prompts = positive_prompts + ([negative] if guidance else [])
    encoded, valid = pipeline.encode_prompt(prompts, device=device, max_sequence_length=maximum)
    cfg = pipeline.transformer.config
    if (encoded.ndim != 4 or tuple(encoded.shape[2:]) != (cfg.num_text_layers, cfg.text_hidden_dim)
            or valid.dtype != torch.bool or tuple(valid.shape) != tuple(encoded.shape[:2])
            or encoded.shape[0] != len(prompts) or not bool(torch.isfinite(encoded).all())):
        raise ValueError('Krea 2 regional text embeddings do not match the transformer contract.')
    rows = [encoded[i, valid[i]] for i in range(len(prompts))]
    if any(not row.shape[0] for row in rows):
        raise ValueError('Each Krea 2 regional prompt must produce tokens.')
    positive_rows = rows[:len(positive_prompts)]
    positive = torch.cat(positive_rows, 0)
    neg = rows[-1] if guidance else None
    length = max(len(positive), len(neg) if neg is not None else 0)
    text_groups = torch.cat([torch.full((len(row),), i, device=device, dtype=torch.long)
                             for i, row in enumerate(positive_rows)])
    if length > len(positive):
        text_groups = torch.cat((text_groups, torch.full((length - len(positive),), -1, device=device, dtype=torch.long)))
    def padded(row):
        output = row.new_zeros((1, length, *row.shape[1:]))
        output[0, :len(row)] = row
        mask = torch.zeros((1, length), dtype=torch.bool, device=device)
        mask[0, :len(row)] = True
        return output, mask
    pos_embeds, pos_mask = padded(positive)
    neg_embeds, neg_mask = padded(neg) if guidance else (None, None)
    attention = regional_prompt.Attention(groups, text_groups, selection.isolate_until, torch, order='text-first')
    attention.masks(0, 1)
    transformer, scheduler = pipeline.transformer, pipeline.scheduler
    state = {'step': 0, 'calls': 0, 'positive': False}
    expected_calls = 2 if guidance else 1
    activity = {'positive_forwards': 0, 'negative_forwards': 0}
    old_step = scheduler.step
    had_step, instance_step = 'step' in vars(scheduler), vars(scheduler).get('step')
    handles = []
    old_inputs = {key: inputs[key] for key in ('prompt', 'negative_prompt') if key in inputs}
    try:
        inputs.pop('prompt', None)
        inputs.pop('negative_prompt', None)
        inputs['prompt_embeds'], inputs['prompt_embeds_mask'] = pos_embeds, pos_mask
        if guidance:
            inputs['negative_prompt_embeds'], inputs['negative_prompt_embeds_mask'] = neg_embeds, neg_mask
        def select_branch(module, args, kwargs):
            if state['calls'] >= expected_calls:
                raise RuntimeError('Unexpected Krea 2 regional transformer call sequence.')
            state['positive'] = state['calls'] == 0
            key = 'positive_forwards' if state['positive'] else 'negative_forwards'
            activity[key] += 1
            state['calls'] += 1
        handles.append(transformer.register_forward_pre_hook(select_branch, with_kwargs=True))
        def hook(kind):
            def replace_mask(module, args, kwargs):
                if state['positive']:
                    total = len(scheduler.timesteps)
                    masks = attention.masks(state['step'], total)
                    chosen = masks[0 if kind == 'text' else 1]
                    if args[0].shape[1] != chosen.shape[0]:
                        raise ValueError('Krea 2 regional mask does not match the attention sequence.')
                    kwargs = {**kwargs, 'attention_mask': chosen[None, None, :, :]}
                return args, kwargs
            return replace_mask
        for block in transformer.text_fusion.refiner_blocks:
            handles.append(block.attn.register_forward_pre_hook(hook('text'), with_kwargs=True))
        for block in transformer.transformer_blocks:
            handles.append(block.attn.register_forward_pre_hook(hook('joint'), with_kwargs=True))
        def step(*args, **kwargs):
            if state['calls'] != expected_calls:
                raise RuntimeError('Incomplete Krea 2 regional transformer calls in a sampling step.')
            result = old_step(*args, **kwargs)
            state['step'] += 1
            state['calls'] = 0
            state['positive'] = False
            return result
        scheduler.step = step
        yield {'regions': len(selection.regions), 'isolate_until': selection.isolate_until,
               'token_counts': [len(row) for row in positive_rows], 'negative_tokens': len(neg) if guidance else 0,
               'mask_size': [width // 16, height // 16], 'forward_activity': activity}
    finally:
        if had_step:
            scheduler.step = instance_step
        else:
            vars(scheduler).pop('step', None)
        for handle in reversed(handles):
            handle.remove()
        for key in ('prompt_embeds', 'prompt_embeds_mask', 'negative_prompt_embeds', 'negative_prompt_embeds_mask'):
            inputs.pop(key, None)
        inputs.update(old_inputs)
