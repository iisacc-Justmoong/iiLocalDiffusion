# DiT-standard model conversion

iiLocalDiffusion standardizes merge inputs on an executable DiT checkpoint.
The first implementation targets the Tsubaki family. SD 1.5, SDXL derivatives
(including Haruka, Hoshino and Illustrious), existing DiT checkpoints and the
separate Reference Pro editing family can therefore produce checkpoints with one
identical tensor inventory, shape contract and dtype contract before ordinary
weight merging.

## Conversion contract

`--target-dit` is a real Tsubaki/DiT checkpoint and is the executable architecture
of the result. It is not inferred from a filename. Its tensor keys, shapes, dtypes,
non-floating buffers and architecture metadata define `iild-dit-standard-v1`.
Every converted output has that exact contract and can be passed to `iild-merge`
in `weighted-sum` or `weighted-difference` mode with another output made from the
same template.

For source tensors without an exact target name, policy
`role-depth-stat-match-template-fill-v1` selects a target by component
(denoiser/text/VAE), projection role, block depth and shape distance. It flattens
the leading/output dimension, crops the overlap, matches the source RMS to the
target RMS, and retains template values outside the overlap. `--transfer-strength`
blends this transplanted tensor into the DiT template and defaults to `1`.
Non-floating buffers always remain those of the target template.

This policy is deterministic and useful for structural merge bootstrapping, but
it is not learned distillation. The converter records
`semantic_equivalence: false` and the full mapping report. Image quality and the
preservation of a source model's concepts must be evaluated with real prompts;
production-quality architectural transfer should later replace the synthetic
mapping with trained Tsubaki deltas while retaining the same output contract.

## Commands

Inspect without hashing model payloads or writing output:

```sh
iild-convert \
  --source /Models/illustrious.safetensors \
  --source-family illustrious \
  --target-dit /Models/tsubaki.safetensors \
  --output /Models/illustrious-tsubaki.safetensors \
  --inspect
```

Build two normalized checkpoints and merge them:

```sh
iild-convert --source /Models/sd15.safetensors --target-dit /Models/tsubaki.safetensors \
  --output /Models/sd15-tsubaki.safetensors
iild-convert --source /Models/reference-pro.safetensors --source-family reference-pro \
  --target-dit /Models/tsubaki.safetensors --output /Models/reference-pro-tsubaki.safetensors
iild-merge --base-model /Models/sd15-tsubaki.safetensors \
  --additional-model /Models/reference-pro-tsubaki.safetensors --weight 0.25 \
  --output /Models/merged-tsubaki.safetensors
```

Auto detection uses checkpoint tensor signatures for SD 1.5, SDXL and supported
DiT architectures. Haruka, Hoshino, Illustrious and Reference Pro may be selected
explicitly because ecosystem names alone are not reliable architecture evidence.
When a private Tsubaki export lacks identifying metadata, pass
`--target-family tsubaki` after independently verifying the template.

## Safety and provenance

Inputs are safetensors only. The converter never loads pickle, never modifies an
input, rejects nonfinite transferred or template tensors, hashes and revalidates
both inputs before publication, and atomically creates a new output without
overwriting an existing path. The output embeds source/template SHA-256 values,
the conversion policy, family classification and transfer strength in
`iild_dit_conversion`; the target's loader metadata is retained.
