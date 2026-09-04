# Automatic local image workflows

`reference/diffusers/comfyui_image_workflow.py` assembles a ComfyUI API graph from
a Civitai base-model name, registered local weight names, prompt, explicit
generation inputs, and the running server's `/object_info`. Users do not need to
author or export a workflow for these recipes. The builder has no additional
Python dependency, performs no filesystem/network I/O, and does not import Torch
or ComfyUI. Runtime setup, file registration, submission and artifact publication
belong to the caller.

```python
from comfyui_image_workflow import build_workflow

graph = build_workflow(
    base_model="Illustrious",
    model_name="my-illustrious.safetensors",  # exact server inventory name
    components={},
    prompt="a lighthouse above a quiet bay, painted illustration",
    object_info=client.json("/object_info"),
    negative_prompt="blur",
    width=1024, height=1024, seed=42, steps=25, cfg=7.0,
)
```

`workflow_requirements(base_model)` exposes the recipe, baseline resolution,
sampling defaults, text-loader type, and required split-component roles without
starting a server. It describes the recipe, not an inference result. The Civitai
label declares intended architecture; ComfyUI's tensor loader verifies the files
when the submitted graph executes.

## Weight roles

A `checkpoint` uses `CheckpointLoaderSimple` outputs MODEL, CLIP and VAE. Explicit
components replace its encoder/VAE outputs without replacing the checkpoint's
denoiser. A `diffusion_model` uses `UNETLoader` and requires separate encoder and
VAE files. `model_type="auto"` checks both live inventories; if a name occurs in
both inventories, the caller must specify `checkpoint` or `diffusion_model`.

`components` accepts these exact keys:

| Key | Role |
| --- | --- |
| `vae` | Decoder VAE; for Stable Cascade this is stage A |
| `text_encoder` | First encoder or the sole encoder |
| `text_encoder_2` | Second encoder |
| `text_encoder_3` | Third encoder |
| `text_encoder_4` | Fourth encoder |
| `model_negative` | Ideogram4's separate unconditional denoiser |
| `decoder` | Stable Cascade's stage-B denoiser |

Encoder keys must be consecutive. Flux.1 uses CLIP-L + T5XXL; SDXL uses CLIP-L +
CLIP-G. SD3 accepts one, two or three matching encoders through its upstream
single/dual/triple loaders. HiDream-I1 accepts the upstream single/dual/quadruple
forms; the full form is CLIP-L, CLIP-G, T5XXL and Llama 3.1. Reduced forms still
need the correct encoder tensors for their role.

GGUF denoisers select `UnetLoaderGGUF`. GGUF text encoders select the matching
`CLIPLoaderGGUF`, `DualCLIPLoaderGGUF`, `TripleCLIPLoaderGGUF` or
`QuadrupleCLIPLoaderGGUF`. A missing loader or unsupported architecture/enum fails
before submission. File extension alone is not proof of the optional GGUF
loader's tensor compatibility. The builder does not install custom nodes.

## Recipes and baseline defaults

| Base family | Image latent / special behavior | Baseline |
| --- | --- | --- |
| SD1 | Four-channel SD latent | 512, 20 steps, CFG 7 |
| SD2 | SD latent; `768` categories use 768 | 512 or 768, 20, CFG 7 |
| SDXL, Pony, Illustrious, NoobAI | SDXL latent and bundled or dual encoders | 1024, 25, CFG 7 |
| SD3 / 3.5 | 16-channel latent; single/dual/triple SD3 encoder | 1024, 28, CFG 5 |
| Flux.1 dev / Krea | 16-channel latent; separate embedded guidance 3.5 | 1024, 20, CFG 1 |
| Flux.1 Schnell | No embedded guidance | 1024, 4, CFG 1 |
| Flux.2 dev | 128-channel Flux2 latent and `Flux2Scheduler`; embedded guidance 4 | 1024, 20, CFG 1 |
| Flux.2 Klein | Flux2 latent/scheduler; distilled and base settings differ | Distilled 4 / CFG 1; base 20 / CFG 5 |
| AuraFlow / Pony V7 | Four-channel latent, T5-XL encoder detected by upstream | 1024, 30, CFG 3.5 |
| Chroma | T5 padding disabled, shift 1, beta schedule | 1024, 26, CFG 3.5 |
| PixArt Alpha / Sigma | Alpha additionally encodes resolution; Sigma does not | 1024, 30, CFG 4.5 |
| HunyuanDiT | Bundled checkpoint and BERT/MT5 prompt encoding | 1024, 30, CFG 6 |
| HiDream-I1 | 16-channel latent and native encoder forms | 1024, 50, CFG 5 |
| HiDream-O1 | Bundled checkpoint, native pixel-space latent/conversion VAE | 2048, 40, CFG 5 |
| Qwen Image | 16-channel latent, Qwen encoder and shift 3.1 | 1328, 20, CFG 4 |
| Lumina | Lumina2 encoder and its `superior` system prompt | 1024, 30, CFG 4 |
| ZImageBase / Turbo | Lumina2 loader detects Qwen3 encoder; 16-channel latent | Base 25 / CFG 4; Turbo 8 / CFG 1 |
| Anima | Qwen3-0.6B encoder detected by upstream, matching image VAE | 1024, 30, CFG 4 |
| Ernie | Ministral encoder and Flux2 latent/VAE | 1024, 20, CFG 4 |
| Boogu | Boogu encoder and native denoiser; baseline is Turbo | 1024, 4, CFG 1 |
| Krea 2 | Krea2 encoder and native denoiser; baseline is Turbo | 1024, 8, CFG 1 |
| Lens | Flux2 latent/VAE, size-aware Flux shift, pre-CFG norm | 1440, 20, CFG 5 |
| MageFlow | Native encoder supplies both conditioning and matching latent | 1024, 30, CFG 5 |
| Ideogram4 | Separate conditional/unconditional models and native scheduler | 1024, 20, CFG 7 |
| Stable Cascade | Stage C sampling → stage B conditioning/sampling → stage A VAE | 1024; C 20 / CFG 4; B 10 / CFG 1 |

These settings are editable baselines, not guarantees for every fine-tune,
distillation variant or model version sharing a Civitai category. In particular,
LCM, Turbo, Lightning, Hyper, HiDream fast/dev and Krea2 raw variants have
checkpoint-specific sampling contracts. Exact model-card settings take precedence
through explicit arguments. `Lumina` here means Lumina2; a Lumina1 bundle still
requires its own compatible pipeline. HunyuanDiT's split BERT/MT5 route is not
advertised because current built-in `DualCLIPLoader` does not expose a
`hunyuan_dit` enum. HiDream-O1 uses its checkpoint-injected tokenizer and
pixel-space conversion VAE.

## Validation and explicit controls

The builder checks every generated node/input against live `/object_info`,
including hosted-node exclusion, required inputs, model and sampler enums,
scalar types/ranges, and connection output types. Unknown component keys and
unconsumed options fail. It checks dimension divisibility before constructing
latent nodes; it never silently rounds a requested image size.

Optional arguments include `negative_prompt`, `width`, `height`, `seed`, `steps`,
`cfg`, `sampler_name`, `scheduler`, `batch_size`, `prediction_type`, `model_type`,
`guidance`, `sampling_shift`, `zsnr`, and `clip_skip`. Omitted values retain the
recipe or checkpoint defaults. `guidance` is Flux's embedded guidance and is
separate from CFG. Flux2 and Ideogram4 require their native named schedules.
`prediction_type` supports `epsilon`, `v_prediction`, `sample`, and `lcm` only
for SD1/2/SDXL; `zsnr=True` requires an explicit prediction type. NoobAI is not
assumed to be V-prediction solely because of its category. `clip_skip=0` means
the final CLIP layer, and `clip_skip=1` selects the penultimate layer.

The builder requires text-to-image task capability. Image editing, upscaling,
video, audio and 3D resources require their own input/output workflows. Hosted
and unknown categories fail explicitly. Kolors has no verified native ComfyUI
recipe in this module and remains available through its matching local
Diffusers pipeline. A LoRA, embedding or VAE file is a component, not a complete
denoiser/checkpoint.

`tests/ComfyUIImageWorkflowTests.py` verifies encoder arity, component replacement,
latent choice, sample/conditioning flow, NoobAI prediction overrides, Flux2
schedule, two-stage Cascade, Ideogram4 unconditional model, GGUF node boundaries,
and schema errors without downloading or loading weights. These tests establish
graph contracts, not actual generation or visual-quality evidence.

Sources checked against ComfyUI commit
[`e80c1570b6b44a2557d5d8e341e05782d18c9bbb`](https://github.com/Comfy-Org/ComfyUI/tree/e80c1570b6b44a2557d5d8e341e05782d18c9bbb):
[loaders](https://github.com/Comfy-Org/ComfyUI/blob/e80c1570b6b44a2557d5d8e341e05782d18c9bbb/nodes.py),
[text encoder dispatch](https://github.com/Comfy-Org/ComfyUI/blob/e80c1570b6b44a2557d5d8e341e05782d18c9bbb/comfy/sd.py),
[model contracts](https://github.com/Comfy-Org/ComfyUI/blob/e80c1570b6b44a2557d5d8e341e05782d18c9bbb/comfy/supported_models.py),
[native image/sampler nodes](https://github.com/Comfy-Org/ComfyUI/tree/e80c1570b6b44a2557d5d8e341e05782d18c9bbb/comfy_extras),
and [official workflow templates](https://github.com/Comfy-Org/workflow_templates/tree/main/templates).
