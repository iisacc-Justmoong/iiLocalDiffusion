"""Encode all learned-token vectors in CLIP-sized chunks without truncation.

The native backend already uses 75-token chunks with BOS/EOS and the first
SDXL pooled chunk. Keep the Diffusers adapter on the same policy.
"""
from contextlib import contextmanager
from functools import wraps
import inspect
import math


def token_chunks(ids, tokenizer, count):
    size = tokenizer.model_max_length
    payload = size - 2
    if payload < 1 or size > 4096:
        raise ValueError("Invalid CLIP context length.")
    result = []
    for index in range(count):
        chunk = [tokenizer.bos_token_id, *ids[index * payload:(index + 1) * payload], tokenizer.eos_token_id]
        chunk.extend([tokenizer.pad_token_id] * (size - len(chunk)))
        result.append(chunk)
    return result


def _batch(value, count=None):
    if isinstance(value, str):
        return [value] * (count or 1)
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        if count is not None and len(value) != count:
            raise ValueError("Positive and negative prompt batches must have equal sizes.")
        return value
    raise ValueError("Long CLIP conditioning requires string prompts.")


def token_attention_masks(ids, tokenizer, count):
    size = tokenizer.model_max_length
    payload = size - 2
    masks = []
    for index in range(count):
        used = min(payload, max(0, len(ids) - index * payload)) + 2
        masks.append([1] * used + [0] * (size - used))
    return masks


def encode_long_prompt(pipeline, family, original, arguments):
    if arguments.get("prompt_embeds") is not None or arguments.get("prompt") is None:
        return original(**arguments)
    positive = _batch(arguments["prompt"])
    batch = len(positive)
    cfg = arguments.get("do_classifier_free_guidance", True)
    negative = _batch(arguments.get("negative_prompt") or "", batch)
    pairs = [(pipeline.tokenizer, pipeline.text_encoder, positive, negative)]
    if family == "sdxl-base":
        pairs.append((pipeline.tokenizer_2, pipeline.text_encoder_2,
                      _batch(arguments.get("prompt_2") or positive, batch),
                      _batch(arguments.get("negative_prompt_2") or negative, batch)))
    planned = []
    count = 1
    for tokenizer, encoder, prompts, negatives in pairs:
        rows = tokenizer(prompts + (negatives if cfg else []), add_special_tokens=False,
                         truncation=False, padding=False).input_ids
        count = max(count, *(math.ceil(len(row) / (tokenizer.model_max_length - 2)) for row in rows))
        planned.append((tokenizer, encoder, rows))
    if count == 1:
        return original(**arguments)
    if count > 64:
        raise ValueError("CLIP prompt exceeds 64 context chunks.")
    import torch
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    device = arguments.get("device") or pipeline._execution_device
    scale = arguments.get("lora_scale")
    encoders = [entry[1] for entry in planned]
    if scale is not None and not USE_PEFT_BACKEND:
        raise ValueError("Long CLIP LoRA scaling requires the installed PEFT backend.")
    if scale is not None:
        for encoder in encoders:
            scale_lora_layers(encoder, scale)
    try:
        positives, negatives, pooled, negative_pooled = [], [], None, None
        # Offload hooks may restore parameters here. Keep tensor version counters
        # so a warm pipeline can later move between CPU and accelerator layouts.
        with torch.no_grad():
            for tokenizer, encoder, rows in planned:
                ids = torch.tensor([chunk for row in rows for chunk in token_chunks(row, tokenizer, count)],
                                   dtype=torch.long, device=device)
                kwargs = {"output_hidden_states": True}
                if getattr(encoder.config, "use_attention_mask", False):
                    # Some CLIP tokenizers share EOS and PAD IDs. Mask by valid
                    # sequence length so the real EOS token remains visible.
                    kwargs["attention_mask"] = torch.tensor(
                        [mask for row in rows for mask in token_attention_masks(row, tokenizer, count)],
                        dtype=torch.long, device=device)
                encoded = encoder(ids, **kwargs)
                skip = arguments.get("clip_skip")
                if family == "sdxl-base":
                    hidden = encoded.hidden_states[-(2 + (skip or 0))]
                    negative_hidden = encoded.hidden_states[-2]
                elif skip is None:
                    hidden = encoded.last_hidden_state
                    negative_hidden = encoded.last_hidden_state
                else:
                    hidden = encoder.text_model.final_layer_norm(encoded.hidden_states[-(skip + 1)])
                    negative_hidden = encoded.last_hidden_state
                hidden = hidden.reshape(len(rows), count * tokenizer.model_max_length, -1)
                positives.append(hidden[:batch])
                if cfg:
                    negatives.append(negative_hidden.reshape(len(rows), count * tokenizer.model_max_length, -1)[batch:])
                projected = getattr(encoded, "text_embeds", None)
                if family == "sdxl-base" and projected is not None:
                    projected = projected.reshape(len(rows), count, -1)[:, 0]
                    pooled = projected[:batch]
                    negative_pooled = projected[batch:] if cfg else None
            dtype = encoders[-1].dtype
            repeats = arguments.get("num_images_per_prompt", 1)
            def expand(tensor):
                return None if tensor is None else tensor.to(device=device, dtype=dtype).repeat_interleave(repeats, dim=0)
            positive_result = expand(torch.cat(positives, dim=-1))
            negative_result = expand(torch.cat(negatives, dim=-1)) if cfg else None
            if family == "sd15":
                return positive_result, negative_result
            if pooled is None:
                raise ValueError("SDXL long conditioning requires projected pooled text embeddings.")
            return positive_result, negative_result, expand(pooled), expand(negative_pooled)
    finally:
        if scale is not None:
            for encoder in encoders:
                unscale_lora_layers(encoder, scale)


@contextmanager
def long_clip_prompt_context(pipeline, preset):
    if preset.family not in ("sd15", "sdxl-base") or not hasattr(pipeline, "encode_prompt"):
        yield
        return
    absent = object()
    previous = vars(pipeline).get("encode_prompt", absent)
    original = pipeline.encode_prompt
    signature = inspect.signature(original)
    @wraps(original)
    def encode(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        return encode_long_prompt(pipeline, preset.family, original, dict(bound.arguments))
    pipeline.encode_prompt = encode
    try:
        yield
    finally:
        if previous is absent:
            del pipeline.encode_prompt
        else:
            pipeline.encode_prompt = previous
