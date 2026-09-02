# LoRA generation contract

## Scope

The Python Diffusers generation oracle can apply one explicitly selected LoRA
adapter to any of the `sd15`, `sdxl-base`, or `flux1-schnell` presets. There is
no built-in or automatically selected LoRA. Omitting `--lora` preserves the
base-model generation path without calling an adapter loader.

LoRA loading is not part of the current C++ manifest inspector or a C++
inference backend. The base model must first pass its pinned pipeline contract;
the adapter is then loaded and activated before device placement or sequential
CPU offload hooks are installed.

## Local adapter

Pass an exact safetensors file:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset flux1-schnell \
  --lora /absolute/path/to/style.safetensors \
  --lora-scale 0.75 \
  --output build/reference/flux-style.png
```

A directory is also accepted only when its exact file is named explicitly:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset sdxl-base \
  --lora /absolute/path/to/adapter-directory \
  --lora-weight-name style.safetensors \
  --lora-scale 0.8
```

The selected local file must exist, be non-empty, and end in `.safetensors`.
Its SHA-256 and byte size are recorded in the result sidecar. When `--output`
is omitted, a LoRA run uses the preset's separate `*-lora.png` filename so it
cannot silently replace the canonical base fixture.

## Remote adapter

A Hugging Face adapter requires both an immutable 40-character lowercase
commit SHA and the exact safetensors filename:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset flux1-schnell \
  --lora owner/adapter-repository \
  --lora-revision 0123456789abcdef0123456789abcdef01234567 \
  --lora-weight-name adapter.safetensors \
  --lora-scale 1.0
```

Branch names, tags, implicit repository file selection, pickle `.bin`, `.pt`,
and `.ckpt` files are rejected. `--local-files-only` applies to both the base
model and the adapter.

## Runtime semantics

The fixed internal adapter name is `iild_lora`. Diffusers loads it with
`use_safetensors=True` and `low_cpu_mem_usage=True`, after which
`set_adapters("iild_lora", adapter_weights=scale)` activates the requested
finite scale. The adapter is not fused into the base weights. Negative, zero,
and greater-than-one finite scales remain available because Diffusers supports
them and some adapters depend on values outside zero to one.

The loader verifies that the adapter is both registered on at least one model
component and listed as active before inference begins. SD adapters can target
the UNet and CLIP encoder, SDXL adapters can additionally target its second
CLIP encoder, and FLUX adapters can target the transformer and first CLIP
encoder. The FLUX T5 encoder is not a LoRA-loadable component in Diffusers
0.40.0.

An adapter must have been trained for the selected base architecture. A
successful download does not establish shape compatibility, intended trigger
words, output quality, safety, ownership, or commercial rights. Trigger words
are not inferred or inserted into the prompt. A Diffusers-native safetensors
LoRA is runtime-verified. Other safetensors layouts are delegated to Diffusers
0.40's format conversion; Kohya, DoRA, LyCORIS variants, Control LoRA, fusion,
and multiple simultaneous adapters are neither independently verified nor
explicitly rejected by this interface.

## Provenance

The JSON sidecar contains an `adapters` array. A base-only run records an empty
array. A LoRA run records its source, exact file, requested revision, local file
hash and size when applicable, scale, safetensors format, fixed adapter name,
actual registered components, active adapter list, and `fused: false`.

Adapter licensing is independent of the base model and Diffusers/PEFT library
licenses. The operator must review the selected adapter repository or local
artifact terms before product use.
