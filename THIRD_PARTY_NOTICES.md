# Third-party notices

iiLocalDiffusion currently links privately against
[`json-c`](https://github.com/json-c/json-c), distributed under the MIT
License. The dependency is discovered from the consumer's system with
pkg-config and is not vendored in this repository.

The tools under `reference/diffusers/` optionally use PyTorch, Hugging Face
Diffusers, Transformers, Accelerate, huggingface_hub, Hugging Face Xet, PEFT,
safetensors, NumPy, and Pillow. Those tools are development oracles and are not
part of the C++ runtime dependency graph. See each upstream distribution for
its full notices. Hugging Face Xet 1.6.0 and PEFT 0.20.0 declare Apache-2.0.

No model weights are included. The configured Stable Diffusion 1.5 reference
repository declares the CreativeML OpenRAIL-M model license. The configured
Stable Diffusion XL Base 1.0 reference declares the CreativeML Open RAIL++-M
model license. The configured Black Forest Labs FLUX.1-schnell reference is
licensed under Apache-2.0. FLUX.1-dev is not a configured reference and its
separate non-commercial terms are not treated as interchangeable with Schnell.
