# Neural Engine and Tensor Cores

These are different accelerators, not aliases for a GPU device. On Apple
Silicon, Apple Neural Engine (ANE) is reached through Core ML. NVIDIA Tensor
Cores are units inside CUDA GPUs; the MLX/cuBLAS or PyTorch backend selects
suitable matrix kernels. `--device neural`, `--device tensor`, and a universal
NPU backend are not provided.

## Discovery and evidence

`iild-run devices` reports Core ML availability, actual ANE core count when
the OS exposes it, compute-plan availability, and CUDA Tensor Core eligibility.
The public entry points are `coreMLCapabilities()` and
`tensorCoreCapabilities(deviceIndex)`. MLX's existing public capability structs
and factory overloads retain their layouts/signatures.

ANE discovery uses Apple's public
[`MLNeuralEngineComputeDevice.totalCoreCount`](https://developer.apple.com/documentation/coreml/mlneuralenginecomputedevice/totalcorecount),
not a guess from the processor name. Device discovery needs macOS 14;
compute-plan inspection needs macOS 14.4. An older OS reports discovery as
unknown, not a fictitious zero-core accelerator. The implementation requires
an SDK exposing these APIs. Basic explicitly permitted CPU prediction needs
macOS 13 or newer; that older deployment target is not verified by this build.

Tensor Core support is inferred from the CUDA compute capability and device
family, not from CUDA availability alone:

| CUDA device | Reported support | Eligible arithmetic |
|---|---|---|
| Compute capability below 7 | Unsupported | No Tensor Core claim |
| Volta 7.0 / Xavier 7.2 | Supported | FP16 |
| Turing 7.5 RTX / Tesla T4 | Supported | FP16 |
| GTX 16-series, 7.5 | Unsupported | CUDA still works without Tensor Cores |
| Unidentified 7.x family | Unknown | No affirmative Tensor Core claim |
| Ampere or newer, capability 8+ | Supported | FP16, BF16, TF32 |

Python ROCm devices are not labelled as NVIDIA Tensor Cores. Native discovery
uses the MLX CUDA runtime, which targets NVIDIA. This policy follows NVIDIA's
[capability reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
and [cuBLAS math modes](https://docs.nvidia.com/cuda/cublas/index.html).
Unknown device families are deliberately not assigned an invented core count.
AMD Radeon has its own [LibTorch/ROCm path](radeon-rocm.md). HIP execution
does not access NVIDIA TF32 setters; explicit NVIDIA TF32 switches are rejected.

Three claims stay separate: hardware is present, a runtime can schedule an
operation on it, and a profiler observed it doing so. Core ML's
[`MLComputePlan`](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n)
is an **anticipated** dispatch plan. `CoreMLModelInfo.hardwareUsageVerified`
and Tensor Core `usageVerified` / `usage_verified` remain false; successful
arithmetic or a preferred-device label is not a hardware counter measurement.

## C++ Core ML component execution

Apple builds enable `IILD_ENABLE_COREML=ON` independently of MLX. Foundation
and Core ML are private system-framework dependencies; no Python interpreter
or Core ML types appear in the installed C++ API. Other platforms, or
`IILD_ENABLE_COREML=OFF`, retain the API with unavailable capabilities and
throwing execution stubs. A metadata-only build disables Core ML, MLX,
and the optional LibTorch backend.

```cpp
#include <Compute/CoreMLModel.hpp>

auto component = iild::CoreMLModel::load("/absolute/path/component.mlmodelc");
// Default: CPU + Neural Engine permitted, GPU excluded.
// Requires at least one operation preferred on Neural Engine in the plan.
iild::CoreMLModel::Features inputs;
for (const auto &feature : component.info().inputs) {
    inputs.emplace(feature.name, std::vector<float>(feature.elementCount, 1.0F));
}
auto result = component.predict(inputs); // replace ones with application inputs
```

The model is always caller-provided. `load()` accepts only an existing
compiled `.mlmodelc` directory, not a raw checkpoint or an automatically
downloaded model. Native loading is not authentication or a security sandbox;
the caller must trust and version the compiled artifact. `predict()` accepts
exact named features in row-major host
float32 vectors using the declared shapes. Float16 interfaces are converted
by Core ML. It rejects missing/extra features, wrong lengths, non-finite or
out-of-range inputs, and invalid/non-finite outputs. Buffer copies honor Core
ML's actual strides, including padded accelerator buffers. Calls on one
instance are serialized; moved-from instances fail explicitly.

Supported interfaces have positive, declared, single-bound shapes and
non-optional float32/float16 multi-arrays. Image, integer, optional, flexible,
and undeclared-output interfaces are rejected. Plan inspection supports one
`main` ML Program function or a NeuralNetwork model, not pipeline or
multi-function plans. These are component constraints, not a promise of
arbitrary Core ML package compatibility.

`CoreMLOptions.computeUnits=all` permits CPU/GPU/ANE. `cpuOnly` is an explicit
diagnostic and requires `requireNeuralEnginePlan=false`. The default rejects
CPU-only/GPU-only plans even when an ANE exists. Disabling this requirement
permits a non-ANE plan within the selected compute-unit set; it does not
force an unavailable engine or bypass Core ML's scheduler. Apple exposes
[compute-unit sets](https://developer.apple.com/documentation/coreml/mlcomputeunits),
not a public ANE-only guarantee.

```bash
./build/iild-run neural-compute --model /absolute/path/component.mlmodelc
./build/iild-run neural-compute --model /absolute/path/component.mlmodelc --compute-units all
./build/iild-run neural-compute --model /absolute/path/component.mlmodelc \
  --compute-units cpu --allow-cpu-plan
```

This CLI fills inputs with ones and checks successful, finite prediction.
It is a hardware diagnostic, not image generation or a quality benchmark.
`--iterations 1..10000` repeats prediction for external profiling. Its duration
includes validation, input copies, and output readback, not only engine time.

## CUDA matrix precision

The additive `LinearMathOptions` overload selects GPU weight/input arithmetic:

```cpp
auto layer = iild::LinearLayer::fromSafetensors(
    "/absolute/path/component.safetensors", "weight", "bias",
    iild::ComputeOptions{iild::ComputeDevice::cuda},
    iild::LinearResourceOptions{},
    iild::LinearMathOptions{iild::LinearPrecision::float16});
```

`float32` remains the native default. `float16` and `bfloat16` enable eligible
reduced-precision CUDA/Metal math; CUDA BF16 requires a detected Ampere-or-newer
device. CPU-only reduced precision is rejected; a cooperative CPU shard still
uses float32. Host API outputs remain float32. RAM and staged-weight byte
counts use the actual GPU precision (two bytes for FP16/BF16, four for FP32),
while the CPU shard uses four-byte values. Reduced precision changes numeric
accuracy and range. Values that overflow during conversion or arithmetic
fail explicitly instead of returning an apparently successful non-finite result.
Tensor Core dispatch is runtime-selected, not forced for
every shape or operation. Small, unaligned, or non-matrix work may use other
units. On older Tensor Core GPUs, choose FP16 explicitly for native math;
the unchanged FP32 default does not imply Tensor Core use there.

```bash
./build/iild-run compute --device cuda --precision fp16
./build/iild-run compute --precision bf16 --cpu-share 0.25 --weight-storage ram
```

Native FP32 CUDA math retains MLX's TF32 policy; `MLX_ENABLE_TF32=0` before
startup requests full FP32 multiplication precision. Python generation now
automatically permits TF32 on eligible NVIDIA 8+ GPUs. `--no-cuda-tf32` selects
IEEE FP32 matrix/convolution math, and `--cuda-tf32` explicitly requires TF32
eligibility. Explicit TF32 flags on non-CUDA devices are rejected. Neither
flag disables FP16/BF16 Tensor Core kernels. The pinned PyTorch runtime's new
`fp32_precision` APIs are used without mixing deprecated `allow_tf32` controls;
see [PyTorch's CUDA notes](https://docs.pytorch.org/docs/main/notes/cuda.html).

The image sidecar records this policy at `runtime.hardware.tensor_cores`:
device family/capability, per-dtype eligibility, requested TF32 mode, applied
matrix/convolution precision, automatic kernel selection, and unverified
instruction utilization. Existing model/VAE/LoRA validation and activation
order are unchanged.

## Conversion, validation, and limits

[`reference/coreml/convert_linear.py`](../reference/coreml/convert_linear.py)
converts exact rank-two weights and optional bias from a local
`.safetensors`/`.safetensor` file. A channels-first `[1,in,1,batch]` 1x1
convolution implements the linear component in FP16; the external layout is
therefore different from `LinearLayer`'s `[batch,in]` layout. This follows
[Apple's ANE transformer layout guidance](https://machinelearning.apple.com/research/neural-engine-transformers).
Conversion does not merge LoRA, convert an entire checkpoint, or assemble a
diffusion pipeline. See the [conversion setup and commands](../reference/coreml/README.md).

The converter records source/artifact SHA-256 hashes, named keys, dimensions,
dtype, plan, versions, NumPy float32 oracle, and optional C++ validation.
Without `--native-test`, it reports conversion only, not verified prediction.
The output must be new/empty and under `build/`; prior evidence is preserved.
Inference is deliberately tested through C++, not through coremltools' Python
prediction buffer bridge. A crash was observed in that bridge's asynchronous
buffer-release path on macOS 27 / Python 3.13 / coremltools 9.0; conversion and
plan inspection do not use the problematic Python prediction path.

On the 2026-09-03 M1 Max host, public discovery returned 16 ANE cores. A real
SD 1.5 CLIP `fc1` component (`768 -> 3072`, batch 128) preferred ANE for its
convolution and passed native prediction against NumPy with maximum absolute
error `0.002185344696044922` (per-element tolerance `0.005 + 0.01*abs(expected)`).
Evidence: `build/reference/neural/sd15-clip-native-final/provenance.json`.
Separate float16-I/O and small CPU-preferred fixtures exercise conversion,
concurrency, invalid values, moved objects, and default non-ANE rejection.
CPU-only Core ML has different FP16 accumulation behavior; an independent,
exact zero-input/bias case checks that diagnostic path without relaxing the
ANE comparison tolerance.

No NVIDIA GPU is present on this host. CUDA device-family/TF32 policy tests,
real PyTorch precision-setting API checks with simulated device properties,
and actual Metal FP16/BF16 arithmetic pass. This is not physical CUDA/Tensor
Core or CUDA-build verification. No speedup or utilization percentage is
claimed. The C++ diffusion pipeline, automatic ANE use by PyTorch/Diffusers,
and Intel/AMD/Qualcomm NPU backends remain unimplemented. RAM is still storage,
not arithmetic hardware; Core ML allocations are independent of MLX's weight
staging budget.
