"""Scoped reference-token conditioning for a Diffusers Krea 2 transformer."""
from contextlib import contextmanager


@contextmanager
def execution(pipeline, reference, positions, guidance, torch, *, attention='isolated'):
    """Append one clean reference at timestep zero to each CFG forward."""
    transformer, scheduler = pipeline.transformer, pipeline.scheduler
    if attention not in ('isolated', 'joint'):
        raise ValueError('Krea 2 reference attention must be isolated or joint.')
    if (reference.ndim != 3 or reference.shape[0] != 1
            or reference.shape[2] != transformer.config.in_channels
            or positions.ndim != 2 or positions.shape != (reference.shape[1], 3)
            or not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(positions).all())):
        raise ValueError('Krea 2 reference tokens and rotary positions have incompatible shapes or values.')
    state = {'calls': 0, 'active': False, 'target': 0, 'normal_length': 0, 'ref_modulation': None}
    counts = {'conditional': 0, 'unconditional': 0}
    expected_calls = 2 if guidance else 1
    attention_masks = {}
    old_step = scheduler.step
    had_step, instance_step = 'step' in vars(scheduler), vars(scheduler).get('step')
    handles = []
    try:
        def before(module, args, kwargs):
            if args or state['calls'] >= expected_calls:
                raise RuntimeError('Unexpected Krea 2 reference transformer call sequence.')
            conditional = state['calls'] == 0
            state['calls'] += 1
            state['active'] = False
            counts['conditional' if conditional else 'unconditional'] += 1
            latent = kwargs['hidden_states']
            text = kwargs['encoder_hidden_states']
            ids = kwargs['position_ids']
            if latent.ndim != 3 or ids.shape != (latent.shape[1] + text.shape[1], 3):
                raise ValueError('Krea 2 reference requires text-first packed positions.')
            batch, target, _ = latent.shape
            value = reference.to(device=latent.device, dtype=latent.dtype).expand(batch, -1, -1)
            ref_ids = positions.to(device=ids.device, dtype=ids.dtype)
            zero = torch.zeros_like(kwargs['timestep'])
            zero_time = transformer.time_embed(zero, dtype=latent.dtype)
            zero_modulation = transformer.time_mod_proj(torch.nn.functional.gelu(zero_time, approximate='tanh'))
            state.update({'active': True, 'target': target, 'normal_length': text.shape[1] + target,
                          'ref_modulation': zero_modulation})
            return args, {**kwargs, 'hidden_states': torch.cat((latent, value), dim=1),
                          'position_ids': torch.cat((ids, ref_ids), dim=0)}
        def modulation(module, args, output):
            if not state['active']:
                return output
            count = reference.shape[1]
            return torch.cat((output.expand(-1, state['normal_length'], -1),
                              state['ref_modulation'].expand(-1, count, -1)), dim=1)
        def after(module, args, kwargs, output):
            if not state['active']:
                return output
            length = state['target']
            state['active'] = False
            if isinstance(output, tuple):
                return (output[0][:, :length], *output[1:])
            return type(output)(sample=output.sample[:, :length])
        handles.append(transformer.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(transformer.time_mod_proj.register_forward_hook(modulation))
        handles.append(transformer.register_forward_hook(after, with_kwargs=True))
        if attention == 'isolated':
            def restrict_reference(module, args, kwargs):
                if not state['active']:
                    return args, kwargs
                batch, length = args[0].shape[:2]
                if batch * length * length > 80_000_000:
                    raise ValueError('Krea 2 isolated reference attention exceeds the 80 million element limit.')
                branch = 'conditional' if state['calls'] == 1 else 'unconditional'
                key = (branch, batch, length, args[0].device)
                if key not in attention_masks:
                    base = kwargs.get('attention_mask')
                    if base is None:
                        mask = torch.ones((batch, 1, length, length), dtype=torch.bool, device=args[0].device)
                    else:
                        if base.dtype != torch.bool or tuple(base.shape) != (batch, 1, 1, length):
                            raise ValueError('Unsupported Krea 2 reference attention mask.')
                        mask = base.expand(batch, 1, length, length).clone()
                    mask[:, :, state['normal_length']:, :state['normal_length']] = False
                    attention_masks[key] = mask
                return args, {**kwargs, 'attention_mask': attention_masks[key]}
            for block in transformer.transformer_blocks:
                handles.append(block.attn.register_forward_pre_hook(restrict_reference, with_kwargs=True))
        def step(*args, **kwargs):
            if state['calls'] != expected_calls:
                raise RuntimeError('Incomplete Krea 2 reference transformer calls in a sampling step.')
            result = old_step(*args, **kwargs)
            state['calls'] = 0
            return result
        scheduler.step = step
        yield {'reference_tokens': reference.shape[1], 'reference_index': 1, 'reference_attention': attention,
               'reference_timestep': 0, 'forward_activity': counts}
    finally:
        if had_step:
            scheduler.step = instance_step
        else:
            vars(scheduler).pop('step', None)
        for handle in reversed(handles):
            handle.remove()
