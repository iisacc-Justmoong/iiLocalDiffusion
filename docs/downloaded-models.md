# Downloaded model inspection

`reference/diffusers/downloaded_model.py` identifies a local model before a
generation backend is chosen. The inspector reads tensor names, dimensions and
model-version metadata. It never selects a model family from a filename.

```python
from downloaded_model import inspect_downloaded_model

identity = inspect_downloaded_model("/models/download.safetensors")
# A separately saved Civitai model-version response is also accepted:
identity = inspect_downloaded_model(
    "/models/download.safetensors", info_path="/models/model-version.json"
)
```

The result is a JSON-compatible dictionary:

| Field | Meaning |
| --- | --- |
| `format` | `safetensors`, `gguf`, or `pytorch` container |
| `base_model` | Exact, catalog-validated Civitai category from metadata, or null |
| `architecture` | Tensor-derived architecture, or the explicitly declared category family |
| `preset`, `pipeline_class` | Compatible initial generation route, or null when no route is established |
| `role` | `checkpoint`, `lora`, `vae`, `embedding`, `controlnet`, another declared component type, or `unknown` |
| `weights_role` | `denoiser` when a generative model lacks evidence of embedded VAE or text encoder weights; otherwise the component role |
| `available_components`, `missing_components` | Evidence that denoiser, VAE and text encoder weights occur in this container; this is not a complete component integrity check |
| `prediction_type` | Explicit epsilon, v-prediction, sample or flow parameterization when declared |
| `task` | Required task, including `inpainting` and `image-to-image` for detected input channels/refiners |
| `confidence` | `exact` means a supplied SHA256 matched; `metadata` means a category was declared without a hash; `architecture` means the structure was recognized; `unknown` means no route was established |
| `evidence`, `role_guidance` | Inspection basis and the appropriate attachment argument for a component |

`exact` identifies the local file described by the supplied metadata. It does not
authenticate the metadata publisher or prove successful generation. Presence of
VAE/text encoder tensor names is also insufficient to establish that every
required weight, tokenizer or scheduler configuration is included.

The public command also exposes inspection without starting a generation runtime:

```sh
python3 reference/generate.py --inspect-model --model /models/download.safetensors
```

For generation, `reference/generate.py --model /models/download.safetensors`
automatically selects the local image runtime for an existing weight file.
A complete local directory containing `model_index.json` selects Diffusers.
Supplying `--model-config` also selects the generic Diffusers single-file path;
the selected pipeline must actually implement `from_single_file`. A configuration
directory cannot provide a missing loader, as the Kolors example in
[generic-diffusers.md](generic-diffusers.md#kolors-and-task-specific-checkpoints)
explains.
Explicit `--backend`, `--preset`, `--pipeline-class` and `--workflow` selections
retain precedence. `--backend local` selects the local runtime directly.
Local component and runtime options (`--model-info`, `--components`,
`--text-encoder*`, `--model-type`, `--runtime-*`) also select it automatically.

The inspector recognizes SD1, SD2, SDXL base/refiner, SD3 and FLUX.1 signatures.
SD1/2/XL routing checks UNet cross-attention width. FLUX dev/schnell routing checks
guidance embedding weights. FLUX editing variants, Chroma and Flux2 are not
silently routed through the plain FLUX.1 text-to-image preset. The caller still
needs an explicit workflow for any identified task that requires source images.

Illustrious, Pony, NoobAI and Krea are category-specific routing decisions after
the architecture is checked. Their names cannot be reliably distinguished from
base models by tensor shapes. NoobAI v-prediction requires explicit prediction
metadata (`predictionType: "v_prediction"`, safetensors
`modelspec.prediction_type`, or `ss_v_parameterization: "true"`) or an explicit
generation preset. Merely naming the category NoobAI does not identify its
training parameterization.

## Civitai model-version metadata

The inspector automatically checks the adjacent `download.civitai.info` and
`download.safetensors.civitai.info` names. If both exist, select one explicitly
with `info_path`. The metadata may be a saved Civitai model-version API response:

```json
{
  "baseModel": "Illustrious",
  "model": { "type": "Checkpoint" },
  "files": [
    {
      "name": "original-download.safetensors",
      "hashes": {
        "SHA256": "64 hexadecimal characters from the actual download"
      }
    }
  ]
}
```

When SHA256 values exist, the selected file must match one. This permits local
renaming while rejecting an unrelated model-version sidecar. Invalid hashes,
unknown category names, hosted categories and conflicts between model type,
architecture or embedded metadata cause an error before model loading. A saved
JSON without hashes remains usable, but its file identity is explicitly
unverified. No network request, model download or remote code execution is part
of inspection.

LoRA, LyCORIS/LoCon, VAE, learned embeddings and ControlNet files are components.
They return an attachment instruction and no independent checkpoint pipeline.
For example, a LoRA belongs in `--lora` with a matching base model. A Civitai
`Checkpoint` sidecar cannot override tensor evidence that the selected file is a
LoRA. Unknown component types remain unknown instead of becoming SD1 by default.

## Containers and dependencies

Safetensors inspection uses the standard library to read at most 100 MB of JSON
header. It validates duplicate keys, supported dtypes, shapes, byte extents,
overlap, truncation and unindexed trailing bytes without decoding tensor data.
Safetensors payload interpretation remains the responsibility of the maintained
`safetensors` and Diffusers packages used by the generation backend.

GGUF v2/v3 inspection reads bounded metadata and tensor descriptors using the
published format. It reads `general.architecture`, reverses GGUF dimension order
for shape comparisons and checks alignment/data offsets. It does not decode or
prove the validity of a quantized payload. The maintained `gguf` Python package
was evaluated for this boundary: its full reader maps tensor data and brings
NumPy into the otherwise dependency-free inspection path. Quantization loading
is therefore delegated to the selected backend rather than reimplemented here.
Big-endian GGUF and unrecognized format/type revisions require a compatible
backend or an updated inspector.

`.ckpt`, `.pt`, `.pth` and `.bin` inspection requires a stable PyTorch 2.10.0 or
newer runtime and uses its
`torch.load(weights_only=True, map_location="meta")`. Unsupported objects cause
an error; there is no unsafe pickle fallback. If PyTorch is absent, the file is
left unopened and only the explicitly supplied metadata can establish a route.
The generation environment must provide its maintained PyTorch loader and all
required components before generation.

The public `reference/generate.py --inspect-model` command automatically reuses
the installed managed runtime for these legacy containers, so the host Python
does not need Torch. `--runtime-python /absolute/path/to/python` selects another
installed inspection environment. Re-execution stops when that interpreter is
already running. Safetensors and GGUF inspection remains in the host's standard
library path and does not start the managed runtime.

The module has no new mandatory runtime dependency. Its regression tests use
real binary safetensors/GGUF fixture containers and reject conflicting metadata,
invalid hashes, malformed headers and accidental component-as-checkpoint routes:

```sh
python3 tests/DownloadedModelTests.py
```

The test containers establish parsing and dispatch correctness. They contain
synthetic tensors and do not establish visual quality or successful inference
for any full Civitai checkpoint.

## Upstream references

- [Safetensors format and integrity rules](https://github.com/huggingface/safetensors/blob/main/README.md#format)
- [Diffusers checkpoint signature and single-file loading implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/single_file_utils.py)
- [ComfyUI model detection](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_detection.py)
- [GGUF file format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [Maintained GGUF Python reader](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py)
