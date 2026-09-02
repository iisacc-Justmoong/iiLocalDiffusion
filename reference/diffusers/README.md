# Diffusers reference oracle

These scripts preserve a known-good implementation against which future C++
components can be compared. They are not called by the iiLocalDiffusion
library.

## Reproducible environment

The verified host uses macOS arm64 and CPython 3.14. Create the environment on
the external workspace volume:

```bash
uv venv reference/diffusers/.venv --python 3.14
uv pip install \
  --python reference/diffusers/.venv/bin/python \
  -r reference/diffusers/requirements.txt
```

Both scripts use the shared immutable presets in `presets.py`:

| Preset | Model revision | Resolution | Guidance |
|---|---|---:|---:|
| `sd15` (default) | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` | 512 x 512 | 7.5 |
| `sdxl-base` | `462165984030d82259a11f4367a4eed129e94a7b` | 1024 x 1024 | 5.0 |
| `flux1-schnell` | `741f7c3ce8b383c54771c7003378a50191e9efe9` | 1024 x 1024 | 0.0 |

They use only safetensors and disable remote code. The default cache is
`build/reference/huggingface`, and the Xet chunk cache defaults to
`build/reference/huggingface-xet`, not the system-volume Hugging Face cache.

Inspect and persist component configs without running denoising:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/inspect_pipeline.py
```

Generate the fixed 512 x 512 reference image:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py
```

Inspect SDXL Base and generate its fixed 1,024 x 1,024 reference image:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/inspect_pipeline.py --preset sdxl-base

reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --preset sdxl-base
```

Inspect FLUX.1-schnell and generate its fixed 1,024 x 1,024 reference image:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/inspect_pipeline.py --preset flux1-schnell

reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --preset flux1-schnell
```

The official FLUX.1-schnell repository is gated even though its model license
is Apache-2.0. Accept its terms while signed in on Hugging Face, then make the
same account available to `huggingface_hub` before running these commands.

Apply one local LoRA safetensors file after the selected base pipeline has
passed its contract:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset flux1-schnell \
  --lora /absolute/path/to/style.safetensors \
  --lora-scale 0.75 \
  --output build/reference/flux-style.png
```

For a remote LoRA, pin both its commit and exact file:

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset flux1-schnell \
  --lora owner/adapter-repository \
  --lora-revision 0123456789abcdef0123456789abcdef01234567 \
  --lora-weight-name adapter.safetensors \
  --lora-scale 1.0
```

Only an explicitly provided adapter is loaded. The loader forces safetensors,
uses the fixed internal name `iild_lora`, activates the finite scale with
`set_adapters()`, verifies registration, and applies device/offload policy
afterward. Local adapters are identified by SHA-256 and byte size; remote
adapters require an immutable revision. A default LoRA output uses a separate
`*-lora.png` name. See the
[`LoRA generation contract`](../../docs/lora.md) for compatibility and
licensing limits.

Use `--local-files-only` after the pinned snapshot is cached. Existing outputs
are not overwritten unless `--overwrite` is explicit. A remote `--model`
override must also provide a 40-character commit `--revision`; a local model
directory does not use a revision.

## Interpretation

`inspect_pipeline.py` loads actual component weights through Diffusers and
writes normalized metadata to `build/reference/pipeline-inspection.json` for
SD 1.5, `build/reference/pipeline-inspection-sdxl-base.json` for SDXL Base, or
`build/reference/pipeline-inspection-flux1-schnell.json` for FLUX.1-schnell.
It verifies the selected pipeline's required component classes; for SDXL it
also checks the critical 2,048/2,816 conditioning widths and VAE contract. For
FLUX.1-schnell it checks both tokenizers and text encoders, the 64-channel
packed-latent transformer contract, 16-channel shifted VAE, and flow-matching
scheduler.

`generate.py` writes an RGB PNG and a JSON sidecar containing the model preset
and revision, all fixture parameters, direct package versions, device, dtype,
attention-slicing state, safety-checker and watermarker presence, and output
SHA-256. Its `adapters` array is empty for a base run or records the exact LoRA
source, identity, scale, registered components, and active state. For presets
that support it, attention slicing is enabled
automatically on MPS to reduce peak memory use and can be changed explicitly.
For SDXL on fp16 accelerators, the pinned Diffusers pipeline's verified
`force_upcast` path performs VAE decode in float32. Loading only the VAE as
float32 up front is intentionally avoided: Diffusers 0.40's current MPS branch
casts it back to the latent dtype and can produce NaNs. The oracle also rejects
uniform RGB output so that this failure cannot pass merely because a PNG was
written.

FLUX.1-schnell uses BF16 without an `fp16` weight variant, passes no negative
prompt, and caps `max_sequence_length` at 256. The fixed fixture uses four
steps, while positive step overrides remain available for diagnostic runs.
Guidance is fixed at 0.0, and width and height must be multiples of 16. On MPS
and CUDA the oracle keeps the full pipeline off the accelerator, enables VAE
slicing and tiling, and uses sequential CPU offload. Attention slicing is
deliberately rejected for this preset. The optimization sidecar records that
these switches were enabled, not that a particular VAE branch executed; at
batch one and exactly 1,024 square, slicing or tiling may not activate.

The full 12B official snapshot has not been loaded on the verified 32 GiB M1
Max host. Its BF16 weights leave too little headroom for Python, activations,
and the operating system, so swap pressure or process termination remains
possible even with sequential offload. A tiny FLUX pipeline verifies the API
and MPS path only; it is not a production-model memory or quality result.

The CPU generator reduces one source of platform drift but does not make PNGs
pixel-identical across PyTorch versions or accelerators. Future numerical
parity fixtures should export initial latents and intermediate tensors as
safetensors or NPY plus explicit shape, dtype, layout, hashes, and tolerances.

The SD 1.5 repository's CreativeML OpenRAIL-M terms, the SDXL Base repository's
CreativeML Open RAIL++-M terms, and FLUX.1-schnell's Apache-2.0 terms apply
independently of the Python package licenses. SDXL Base and FLUX.1-schnell
contain no safety checker. The fixed SDXL oracle explicitly disables its
optional invisible watermarker; FLUX.1-schnell has no watermarker component.
Those facts are not a product safety policy.
