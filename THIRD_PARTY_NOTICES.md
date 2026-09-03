# Third-party notices

iiLocalDiffusion currently links privately against
[`json-c`](https://github.com/json-c/json-c), distributed under the MIT
License. The dependency is discovered from the system using its CMake package
or pkg-config and is not vendored in this repository.

The default native computation build additionally links MLX 0.32.2 (MIT),
including its bundled JACCL code. Its fetched dependencies include
nlohmann/json 3.11.3 (MIT), fmt 12.1.0 (MIT), and, for Metal builds, Apple's
metal-cpp 26 (Apache-2.0). Source archives and SHA-256 hashes are fixed in
`cmake/IildMlx.cmake`. No MLX Python package is required by the C++ library.

CUDA builds additionally use NVIDIA CCCL 3.1.3 (Apache-2.0 with upstream
exceptions), NVTX 3.1.1 (Apache-2.0 with LLVM exceptions), cuDNN frontend
1.16.0 (MIT), and CUTLASS 4.4.2 headers (BSD-3-Clause). CUTLASS's separately
licensed Python DSL is not used. NVIDIA's CUDA/cuDNN binary libraries and
driver are separately installed platform dependencies and retain their own
terms; they are not redistributed by iiLocalDiffusion's install rules.

The optional ROCm backend links LibTorch (PyTorch, BSD-style license with
upstream third-party notices) and an externally installed AMD HIP/ROCm SDK.
Vendor libraries, kernel packages and drivers retain their own terms and are
not bundled by this project. The selected LibTorch SDK's LICENSE is included
under `share/licenses/iiLocalDiffusion/libtorch/` for the C++ header code.
No Python interpreter is embedded in C++.
The optional safetensors-cpp reader is pinned to
`af90b6c3006cdcecf8b7d7254f5f32d301728acc` under MIT. Its pinned header includes
the bundled code's MIT notices and the unused FP16 conversion helper's CC0
notice; the complete header and LICENSE are installed under
`share/licenses/iiLocalDiffusion/safetensors-cpp/`. LibTorch, rather than those
conversion helpers, owns all matrix arithmetic and dtype conversion.

The fetched upstream license documents are included by the native install
under `share/licenses/iiLocalDiffusion/`. Backend resources such as
`mlx.metallib` or CUDA JIT headers must accompany their runtime. The MLX API
and third-party tensor types remain private to the implementation.

The tools under `reference/diffusers/` optionally use PyTorch, Hugging Face
Diffusers, Transformers, Accelerate, huggingface_hub, Hugging Face Xet, PEFT,
safetensors, NumPy, and Pillow. Those tools are development oracles and are not
part of the C++ runtime dependency graph. See each upstream distribution for
its full notices. Hugging Face Xet 1.6.0 and PEFT 0.20.0 declare Apache-2.0.
Optional ControlNet uses the existing Diffusers 0.40.0 (Apache-2.0)
model/pipeline implementations and Pillow image handling. It introduces no
new third-party package or condition detector. User-selected ControlNet
weights retain their own license terms and are not bundled.

Apple builds link privately against the system Foundation and Core ML
frameworks, subject to Apple's platform/SDK terms. These frameworks are not
redistributed by the install. The optional offline `reference/coreml/` tool
uses coremltools 9.0 (BSD-3-Clause), NumPy 2.3.5 (BSD-3-Clause), safetensors
0.6.2 (Apache-2.0), and ml_dtypes 0.5.3 (Apache-2.0), plus their packaged
dependencies. Their Python environment is not part of the C++ runtime or
installed library. Each upstream distribution retains its full notices.

No model weights are included. The configured Stable Diffusion 1.5 reference
repository declares the CreativeML OpenRAIL-M model license. The configured
Stable Diffusion XL Base 1.0 reference declares the CreativeML Open RAIL++-M
model license. The configured Black Forest Labs FLUX.1-schnell reference is
licensed under Apache-2.0. FLUX.1-dev is not a configured reference and its
separate non-commercial terms are not treated as interchangeable with Schnell.
