# Hardware compute

GPU computation is the default policy. MLX is an execution runtime, whereas
Metal and CUDA are hardware backends; they are not three interchangeable
device names.
AMD Radeon adds a separate optional LibTorch/ROCm backend, not an MLX device alias.

## Implemented paths

| Entry point | Runtime | Accelerator | Scope |
|---|---|---|---|
| C++ `LinearLayer` | MLX 0.32.2 | Metal on Apple Silicon, CUDA on Linux/NVIDIA | Batched linear component, including named safetensors weights |
| C++ `LinearLayer` | Optional LibTorch HIP | Supported AMD Radeon/ROCm GPUs | Same component API, precision, CPU partition and RAM staging |
| C++ `CoreMLModel` | System Core ML | Apple Neural Engine, with explicitly permitted CPU/GPU | Caller-provided compiled fixed-shape component; ANE-preferred plan required by default |
| Python `generate.py` | PyTorch / Diffusers | Metal through MPS, NVIDIA CUDA, AMD ROCm | Existing SD 1.5, SDXL Base, and FLUX.1-schnell generation oracle |
| `iild-run inspect` | C++ / json-c | None needed | Package metadata and artifact paths only |

MLX tensors, GPU kernels, allocation, and graph execution remain private
implementation details. The C++ library does not call Python. A complete MLX
text encoder, denoiser, VAE, or diffusion pipeline is not implemented by this
change. `--device mlx` is therefore not a Python generation option.
See [Neural Engine and Tensor Cores](neural-accelerators.md) for the separate
Core ML execution path, hardware discovery, CUDA matrix precision, and
plan-versus-measured-usage boundaries. Tensor Cores are CUDA execution units,
not a fourth device type; PyTorch/MLX do not automatically become ANE backends.

## Default selection and failure policy

`ComputeOptions{}` selects available MLX CUDA, then LibTorch ROCm, then MLX
Metal. Python `generate.py --device auto` selects the installed PyTorch
CUDA/HIP GPU first, then Metal. Neither silently selects CPU. An unavailable GPU,
unsupported operation, or out-of-memory failure is reported rather than
retrying on CPU. CPU-only arithmetic requires explicit `ComputeDevice::cpu`
or `--device cpu`. Cooperative CPU work is opt-in through `cpuShare` for the
native component or `--cpu-text-encoding` for the image oracle.

The Python option `--device metal` is an alias for `--device mps`. Before
loading model weights, generation executes and verifies a matrix product on
the selected device and dtype. After model placement/offload setup it checks
the pipeline's actual execution device. The JSON sidecar records both that
device and `runtime.hardware`, including the backend, requested device,
GPU-acceleration flag, preflight result, and no-fallback policy.
`PYTORCH_ENABLE_MPS_FALLBACK=1` is rejected for Metal generation.
`--device rocm` requires an actual HIP build and usable GPU. The Python
execution namespace remains `cuda` while provenance records `backend: rocm`.
The [Radeon guide](radeon-rocm.md) covers SDK selection, CPU/RAM cooperation,
NVIDIA-specific option rejection, model-free preflight and unverified platforms.

File I/O, tokenization, the deterministic CPU random generator, explicit
output readback, and FLUX's CPU weight storage/offload are host operations,
not a promise that every operation runs on a GPU. In particular, FLUX's
sequential offload places each neural component on the selected accelerator
when it executes; it does not select CPU as the denoising device.

## Native build

The default `IILD_ENABLE_MLX=ON` builds the pinned MLX C++ source and selected
backend. No Python runtime is required for native computation.

On Apple Silicon, install Xcode's Metal compiler in addition to the C++ build
requirements. Check `xcrun metal --version`; locating the `metal` shim alone
does not establish that the compiler is installed. If Xcode reports a missing
Metal Toolchain, its supported download command is:

```bash
xcodebuild -downloadComponent MetalToolchain \
  -exportPath "$PWD/build/dependencies/metal-toolchain"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/iild-run devices
./build/iild-run compute
```

Xcode manages the active compiler component; the export path retains a copy
under `build/`. The deployment target must also be compatible with the
installed json-c and compiler SDK. The verified build targets its host OS,
not an independently tested minimum macOS deployment version.

For a Linux/NVIDIA source build, install the CUDA toolkit, compatible NVIDIA
driver, cuDNN development libraries, and BLAS/LAPACK development libraries.
Then explicitly require the CUDA backend:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DIILD_MLX_GPU_BACKEND=cuda
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/iild-run compute --device cuda
```

Use the pinned runtime's [source-build requirements](https://github.com/ml-explore/mlx/blob/v0.32.2/docs/src/install.rst).
Its CMake configuration explicitly rejects CUDA Toolkit 13.1. A headless
cross-build without a visible GPU also needs `MLX_CUDA_ARCHITECTURES` set to
the target architecture. The toolkit and cuDNN are system dependencies, not
automatically purchased, installed, or redistributed by this project.

| CMake option | Default | Meaning |
|---|---|---|
| `IILD_ENABLE_MLX` | `ON` | Build the MLX component backend; independent of LibTorch and Core ML |
| `IILD_ENABLE_LIBTORCH` | `OFF` | Build the optional AMD ROCm / explicit CPU backend with an externally supplied Torch SDK |
| `IILD_ENABLE_COREML` | `ON` on Apple, otherwise `OFF` | Independent native Core ML component support; requires a macOS SDK exposing 14.4 APIs |
| `IILD_MLX_GPU_BACKEND` | `auto` | Apple Silicon selects Metal; Linux with a CUDA compiler selects CUDA |
| `IILD_MLX_GPU_BACKEND=metal` | Explicit | Require the Apple Silicon Metal build |
| `IILD_MLX_GPU_BACKEND=cuda` | Explicit | Require the Linux/NVIDIA CUDA build |
| `IILD_MLX_GPU_BACKEND=none` | Explicit | Build MLX's CPU backend only; automatic execution still fails |

On Linux without a CUDA compiler, automatic configuration reports a CPU-only
build. This does not enable automatic CPU execution. Apple Silicon instead
requires a working Metal compiler unless `none` or MLX `OFF` is explicit.
A lightweight metadata-only build disables all execution runtimes with
`-DIILD_ENABLE_MLX=OFF -DIILD_ENABLE_COREML=OFF -DIILD_ENABLE_LIBTORCH=OFF`.

MLX is a private shared-library dependency. Its installed Metal shader
library (`lib/mlx.metallib`), CUDA JIT headers when applicable, and third-party
notices are part of the install. The consumer test relocates the installation,
links only `iiLocalDiffusion::iiLocalDiffusion`, and executes native compute.
Source archives and hashes are pinned in `cmake/IildMlx.cmake`; see
[dependencies.md](dependencies.md) for dependency and license details.

## C++ component API

```cpp
#include <Compute/LinearLayer.hpp>
#include <vector>

auto layer = iild::LinearLayer::fromWeights(
    std::vector<float>{1, 2, 3, -1, 0, 1},
    3, 2, std::vector<float>{0.5F, -0.5F});
// Default is MLX + an available GPU, never implicit CPU.
auto output = layer.forward(std::vector<float>{1, 2, 3}, 1);
// output = {14.5F, 1.5F}
const auto &actualDevice = layer.computeInfo();

auto fromFile = iild::LinearLayer::fromSafetensors(
    "/absolute/path/to/text-encoder.safetensors",
    "text_model.encoder.layers.0.mlp.fc1.weight",
    "text_model.encoder.layers.0.mlp.fc1.bias");
```

Weights are `[outputFeatures, inputFeatures]`, bias is optional
`[outputFeatures]`, input is row-major `[batch, inputFeatures]`, and output is
row-major `[batch, outputFeatures]`. Native file loading accepts float16,
bfloat16, and float32 weights. Computation defaults to float32 arrays;
the additive `LinearMathOptions` overload selects GPU FP16 or BF16 while
keeping any cooperative CPU shard in FP32 and host outputs in FP32.
CUDA uses MLX's default TF32 acceleration policy for float32
matrix multiplication; set `MLX_ENABLE_TF32=0` before process startup when
full float32 multiplication precision is required. Quantized, integer,
complex, and float64 weights are not supported.
Both `.safetensors` and `.safetensor` are accepted through MLX's explicit safe
loader; there is no pickle fallback.

Only the caller's exact tensor keys are selected. File reading is a host
operation; selected weights are materialized on the requested stream before
construction returns. The file may then be released. This does not classify
or execute an entire SD/SDXL/FLUX checkpoint and does not apply LoRA adapters.
Caller-managed files are not authenticated by the native component API;
the comparison harness below separately records and checks SHA-256 identity.

Each instance retains its weights and explicit compute streams. Batched matrix
multiplication, bias addition, and conversion are delegated to MLX on that
stream. A call synchronizes and explicitly copies its result to host memory
before returning `std::vector<float>`. The component serializes calls on its
stream; it is not a multi-GPU model-sharding or asynchronous scheduling API.
`ComputeOptions.deviceIndex` selects one native device and rejects unavailable
indices.

`iild-run compute` runs a verified `[64,256] x [256,128]` linear operation;
it is a hardware diagnostic, not an image generator. Both `devices` and
`compute` report the actual native runtime/backend. Use `compute --device cpu`
only when CPU-only computation is intended.

## CPU/GPU cooperation and RAM storage

RAM is weight and workspace storage, not a processing device. On Apple
Silicon, CPU and GPU share physical unified memory; assigning weights to RAM
does not add memory capacity or avoid the total system-memory limit. On CUDA,
host RAM and GPU device allocations are separate. The implementation uses
the existing MLX CPU/GPU streams and Diffusers/Accelerate offload facilities,
not custom kernels or a new memory allocator. See the upstream
[MLX unified-memory guide](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
and [Diffusers memory guide](https://huggingface.co/docs/diffusers/optimization/memory).

### Native linear component

The additive `LinearResourceOptions` overloads preserve existing callers:

```cpp
iild::LinearResourceOptions resources;
resources.cpuShare = 0.25;
resources.weightStorage = iild::WeightStorage::ram;
resources.gpuWeightBudgetBytes = 64 * 1024 * 1024;
auto layer = iild::LinearLayer::fromSafetensors(
    "/absolute/path/to/text-encoder.safetensors",
    "text_model.encoder.layers.0.mlp.fc1.weight",
    "text_model.encoder.layers.0.mlp.fc1.bias",
    iild::ComputeOptions{}, resources);
const auto &allocation = layer.resourceInfo();
```

`fromWeights(weights, inputs, outputs, bias, options, resources)` offers the
same policy. The defaults remain `cpuShare=0`, device-resident weights, and
a 64 MiB staging budget used only when RAM storage is selected.

| Option | Contract |
|---|---|
| `cpuShare` / `--cpu-share` | Finite fraction in `[0,1)`. A positive share computes `floor(outputs * share)`, clamped to `[1, outputs-1]`, on CPU. Requires a GPU and at least two output features. This is not a CPU-utilization percentage. |
| `WeightStorage::device` / `--weight-storage device` | CPU shard stays on CPU, remaining weights stay on the GPU. |
| `WeightStorage::ram` / `--weight-storage ram` | Keep all weights in host RAM; send the GPU shard in output-feature blocks. CPU-only execution already uses RAM. |
| `gpuWeightBudgetBytes` / `--gpu-weight-mib` | Upper bound for one logical weight/bias block at the selected GPU precision. At least one output row and its bias must fit. CLI accepts positive whole MiB and requires RAM plus GPU. |

The CPU shard runs in an independent worker using its own MLX CPU stream,
while the calling thread runs the GPU shard. Results merge in the original
row-major output order. Calls on the same layer remain serialized; failure
also joins the CPU worker before returning. MLX owns scheduling and kernels;
concurrent submission does not guarantee a speedup, overlap for every kernel,
or a specific utilization level.

`resourceInfo()` reports CPU/GPU output counts, logical host/resident-GPU
weight bytes, and staged block bytes/row count. These are not measurements of
RSS, committed memory, or VRAM peak. Activations, output buffers, loading
temporaries, MLX allocation caches, and kernel scratch memory are outside the
weight budget. RAM is not disk swapping; no automatic disk offload is added.

```bash
./build/iild-run compute --cpu-share 0.25 --weight-storage ram --gpu-weight-mib 64
./build/iild-run compute --device cpu --weight-storage ram
```

### Python image generation

`--cpu-text-encoding` computes CLIP/T5 embeddings on CPU after LoRA activation
and before installing any device/offload hooks. The denoiser and VAE still
execute on the selected GPU. CPU encoding precedes GPU denoising because the
embeddings are its input; this path is not simultaneous partitioning of a
single denoising operation. SD 1.5/SDXL CPU encoding uses float32, FLUX uses
bfloat16 by default. `--dtype` and `--cpu-text-dtype` expose separate
model and CPU-encoder overrides. Encoders return to their original storage dtype before placement.
Python CUDA/ROCm device indices, generator placement and memory/precision
policies are included in the [generation value schema](generation-parameters.md).
Mixed-precision adapter tensors/buffers retain their original values and dtype
instead of being downcast to the base encoder's precision during restoration.
All resulting conditioning tensors, including SDXL pooled negatives, are
explicitly transferred to the selected device/dtype before inference.

`--cpu-threads N` sets positive PyTorch intra-op CPU threads. Omitting it
retains the runtime default; it does not configure the separate MLX runtime.

| `--offload` | GPU weight policy |
|---|---|
| `auto` (default) | Existing preset: SD/SDXL resident, FLUX sequential. With CPU text encoding, SD/SDXL use model offload so unused text encoders stay in RAM. |
| `none` | Move the pipeline to the GPU; no RAM offload hooks. This explicit override can move already-used text encoders to the GPU too. |
| `model` | Keep inactive models in RAM; move whole models to the GPU when called. The largest executing model and its activations must fit. |
| `sequential` | Keep weights in RAM; move submodules to GPU as called, generally at higher transfer cost. |

Explicit model/sequential offload requires GPU execution. Offload alone does
not compute text/denoising on CPU. With `--device cpu`, `auto` uses CPU/RAM
directly. Failures do not switch devices or offload policies automatically.

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sdxl-base --device auto --cpu-text-encoding --cpu-threads 4 \
  --offload model --local-files-only \
  --output build/reference/sdxl-cpu-ram.png
```

The JSON sidecar records `runtime.cpu_conditioning` (actual CPU tensors,
components, dtype, shapes), `runtime.cpu_threads`,
`runtime.hardware.participating_devices`, and `runtime.optimization`
(`requested_offload`, effective `offload_policy`, and `weight_storage`).
Model, VAE, and LoRA provenance is unchanged. These fields do not claim that
every operation was instrumented or that memory/latency improved.

## Numerical verification

The regular tests cover device policy, CPU/GPU partitioning, concurrent calls,
partial/no-bias RAM blocks, budget rejection, native safetensors lifetime,
CPU text encoding/LoRA/offload order, embedding transfers, CLI failure
contracts, and the installed cooperative API. Device-policy tests do not
count as physical CUDA verification.

Compare a real checkpoint component against an independently executed PyTorch
CPU linear operation:

```bash
reference/diffusers/.venv/bin/python reference/validation/check_mlx_linear.py \
  --model /absolute/path/to/text-encoder.safetensors \
  --weight-key text_model.encoder.layers.0.mlp.fc1.weight \
  --bias-key text_model.encoder.layers.0.mlp.fc1.bias \
  --output-dir build/reference/hardware/my-linear-parity
```

The output directory must be new or empty. The harness exports only the
selected tensors, runs the native GPU and CPU/GPU/RAM tests, and records
source/fixture hashes,
tensor keys, seed, dtype, runtime versions, shapes, and maximum absolute error.
The acceptance rule per element is `absolute_error <= 0.0004 +
0.0001 * abs(reference)`. The native comparison subprocess explicitly disables
TF32 with `MLX_ENABLE_TF32=0` and records that setting. A GPU is mandatory for
this comparison; runtime defaults in ordinary callers are not modified.
The hybrid comparison uses a 1 MiB weight block budget, raised to one row if
a single row is larger. A one-output layer can only run the GPU comparison
and records `hybrid: null`; no CPU/GPU cooperation is claimed for that shape.

On 2026-09-03, the verified M1 Max host executed a real SD 1.5 CLIP `fc1`
component with batch 4, input width 768, and output width 3072 through MLX /
Metal. Maximum absolute error against PyTorch was `9.05991e-06`. The existing
SD 1.5 image oracle also completed with automatic Metal selection and
explicit model, VAE, and LoRA inputs. SDXL Base completed at 128 x 128 with
two steps. A 64 x 64, one-step tiny FLUX smoke confirmed Metal execution with
sequential CPU weight offload; it does not establish the full 12B model's
memory requirements or image quality. Evidence is under
`build/reference/hardware/`; these are execution/numerical checks, not image
quality benchmarks or proof of a complete native diffusion pipeline.

The additional CPU/RAM validation uses
`build/reference/resources/mlx-cpu-gpu-ram-linear/provenance.json`: the same
CLIP layer runs with 768 CPU and 2,304 GPU output features. All 9,449,472
weight/bias bytes reside logically in RAM; GPU staging is limited to a
1,048,576-byte budget. Maximum absolute error is again `9.05991e-06` for
both RAM-staged and device-resident hybrid execution. Separate SD 1.5
(128 square, four steps, custom model/VAE/synthetic LoRA) and SDXL Base
(128 square, two steps) images also completed with CPU text encoding and
model offload to Metal. These are low-resolution execution tests, not a
performance or image-quality benchmark.
A cached tiny FLUX pipeline (64 square, one step, synthetic transformer LoRA)
also completed CPU encoding and sequential offload. Forward guards confirmed
that the offloaded text encoders were not called again on the GPU. This does
not bypass the official FLUX preset's full-model compatibility checks;
the deliberately tiny API smoke uses an isolated validation script.

No NVIDIA GPU is available on that host. CUDA build selection and runtime
failure policy are implemented, but CUDA compilation and physical execution
remain unverified until the same tests run on compatible NVIDIA hardware.
