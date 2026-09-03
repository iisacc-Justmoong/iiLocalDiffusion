# Learned text embeddings: Textual Inversion

The Python generator accepts local Textual Inversion embeddings through
`--text-embedding`. These files add learned tokens to a selected tokenizer
and text encoder; the prompt still passes through text encoding. This is
different from the existing `--embeddings` option, which supplies completed
prompt/pooled tensors and bypasses text encoding.

| Input | Content | How it is used |
|---|---|---|
| `--text-embedding` / `--textual-inversion` | Learned token vectors | Register the token, then use it in ordinary prompt text |
| `--embeddings` | Completed `prompt_embeds` and related tensors | Replace prompt text encoding with supplied conditioning tensors |

The two input modes cannot be combined in one request. Omitting
`--text-embedding` leaves learned-token loading disabled. This feature
supports the existing `sd15`, `sdxl-base`, and `flux1-schnell` presets,
including ControlNet and Hires Fix generation. It does not add textual
inversion training or native C++ image generation.

## CLI and token selection

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sd15 \
  --text-embedding /absolute/path/to/paint-style.safetensors \
  --text-embedding-token '<paintstyle>' \
  --prompt 'a ceramic cup in <paintstyle> style'
```

Quote tokens containing `<` or `>` so the shell treats them as text. A
selected token must be non-empty and contain no whitespace or control
characters. Use the token
in the prompt whose encoder should receive it. Merely loading an embedding
does not insert its token into the prompt; unused embeddings are allowed.

| Argument | Default / behavior |
|---|---|
| `--text-embedding FILE [FILE ...]` | Disabled; one or more local `.safetensors` / `.safetensor` files |
| `--textual-inversion FILE [FILE ...]` | Alias for `--text-embedding` |
| `--text-embedding-token TOKEN [TOKEN ...]` | Infer the source's token identity; explicit values choose the names used in prompts |
| `--text-embedding-encoder auto\|text_encoder\|text_encoder_2 [...]` | `auto` for each file; route using the supported tensor keys and loaded encoder dimensions |

When token or encoder lists are supplied, each list must contain exactly
one entry per embedding file, in the same order. The dependent options
require at least one embedding file. An explicit encoder choice selects
the intended component; incompatible vectors fail validation rather than
being resized or assigned to an unrelated encoder.

If the token override is omitted, token names are resolved in this order:

1. The file's safetensors metadata `token`, then `name`.
2. The tensor key for a single generic learned-token tensor.
3. The filename stem for `emb_params` or named encoder tensors.

The inferred name must pass the same token validation. Use an explicit
override when a filename or stored name is unsuitable for prompt text.

For example, load distinct learned style and object tokens:

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sd15 \
  --text-embedding /absolute/path/to/style.safetensors /absolute/path/to/object.safetensors \
  --text-embedding-token '<style>' '<object>' \
  --text-embedding-encoder text_encoder text_encoder \
  --prompt 'a photograph of <object> with <style> lighting'
```

## File layouts and encoder compatibility

Only local safetensors files are accepted. A remote repository, directory,
pickle `.pt` / `.bin` / `.ckpt` file, arbitrary Python object, or automatic
model download is not selected by this interface. The singular local
`.safetensor` spelling uses the existing checked safe-file path handling.

Supported files carry a single learned-token tensor, the `emb_params`
tensor, or supported named encoder tensors. Each tensor is a floating-point
vector `[embedding_dimension]` or a multi-vector matrix
`[number_of_vectors, embedding_dimension]`. Its width must match the loaded
target encoder's input embedding table.

| Preset | Available destinations | Named multi-encoder layout |
|---|---|---|
| SD 1.5 | `text_encoder`: CLIP | `text_encoder` |
| SDXL Base | `text_encoder`: CLIP; `text_encoder_2`: second CLIP | `clip_l` + `clip_g`, or canonical component keys |
| FLUX Schnell | `text_encoder`: CLIP; `text_encoder_2`: T5 | `clip_l` + `t5xxl`, or canonical component keys |

`auto` resolves a compatible destination from the format and tensor width.
An ambiguous destination requires an explicit encoder selection. Matching
the tensor dimension establishes load compatibility, not that weights were
trained for this particular model or that they will achieve the intended
visual effect.

Named multi-encoder files use `auto` to retain each entry's destination.
An explicit encoder is not a filter for such a file: a conflicting named
entry is rejected. For example, `clip_l` + `clip_g` cannot be restricted to
`text_encoder` merely by changing the encoder option.

Each vector in a multi-vector embedding receives a tokenizer entry. Prompt
handling expands the learned token into its vector tokens for the matching
encoder. Use the logical token in the prompt; do not manually repeat
internal vector suffixes. Existing encoder context-length limits still
apply after expansion.

## Composition and input forms

Learned tokens are registered before prompt encoding and accelerator/RAM
offload placement. Model/checkpoint replacement, VAE replacement, LoRA,
ControlNet, batching, CPU prompt encoding and Hires Fix retain their
existing roles. Learned-token loading changes tokenizer vocabulary and
encoder input embeddings; it does not replace the selected denoiser or VAE.

Secondary and negative prompt fields use the learned tokens in their
respective encoder paths. SDXL/FLUX omitted or empty secondary prompts
inherit the corresponding original primary prompt before expansion for the
second encoder, matching Diffusers' fallback behavior. For FLUX, negative text is used only by stages
whose true CFG is above 1. Hires Fix retains the registered tokenizer and
encoder components for both stages, including multi-vector expansion.
An embedding can therefore participate in the base pass, the refinement
pass, or both, according to the stage's prompt/guidance settings.

The CLI, JSON `--config`, and Python `resolve_request()` share the same
schema. JSON paths are relative to their configuration file; CLI paths are
relative to the working directory. A JSON request uses arrays for files
and corresponding optional token/encoder lists:

```json
{
  "preset": "sdxl-base",
  "text_embedding": ["./embeddings/portrait.safetensors"],
  "text_embedding_token": ["<portrait>"],
  "text_embedding_encoder": ["auto"],
  "prompt": "a portrait of <portrait>",
  "hires_fix": true,
  "hires_scale": 1.5
}
```

Absent lists and top-level JSON `null` leave the feature disabled or select
the documented default. `--print-config` exposes the replayable request;
encoder compatibility and tokenizer collisions are checked against the
loaded components before inference.

## Validation and output identity

Invalid tensor rank, dimension, dtype, empty vectors, non-finite values,
and overflow during conversion to the encoder dtype are rejected.
Tokenizer collisions include the requested logical token and any generated
multi-vector token names. A conflict is an error rather than a request to
overwrite existing vocabulary entries. Local file identity is checked
while loading and retained in provenance.

The default output stem adds `-embedding` after any `-lora` modifier and
before `-controlnet` / `-hires`. A learned-token SD 1.5 run therefore
defaults to `build/reference/sd15-red-cube-embedding.png`. The existing
collision, batch numbering and optional Hires Fix base-output rules apply.

The sidecar's `text_embeddings` array identifies each source file and
SHA-256 hash, requested token/encoder, and its `registrations`. Each
registration records the resolved logical `token`, expanded `tokens`,
encoder `component`, `token_ids`, `source_key`, `source_dtype`, and vector
`shape`, together with the loaded `dtype` and `vector_count`. The
`embedding_rows_before` / `embedding_rows_after` values record vocabulary
storage size; unused padded rows are preserved. Successful CLI loading
prints each resolved token, encoder and vector count so inferred token
names are visible before generation.
The original request and generation inputs remain available for replay.
Small execution tests establish loading, token expansion and pipeline
composition; they do not establish the visual quality or training
compatibility of arbitrary third-party embedding weights.

## Dependencies

The implementation reuses the pinned Diffusers, Transformers, PyTorch and
safetensors packages. Existing tokenizer and embedding-table operations own
vocabulary/weight storage; project code owns explicit selection,
compatibility, composition and provenance. No dependency, learned weights,
training routine or automatic downloader is added. Embedding weights retain
their own terms independently of the runtime packages.

Upstream describes token loading and multi-vector handling in the
[Textual Inversion loader](https://huggingface.co/docs/diffusers/api/loaders/textual_inversion)
and CLIP/T5 embedding files in the
[FLUX advanced training examples](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/README_flux.md).
