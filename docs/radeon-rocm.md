# Radeon / AMD ROCm

AMD GPU execution is implemented through PyTorch/LibTorch's HIP runtime.
This is a separate backend from NVIDIA CUDA and Apple Metal, not a claim
that every Radeon card or operating system supports ROCm.

| Entry point | AMD path | Scope |
|---|---|---|
| Python `generate.py --device rocm` | Vendor HIP PyTorch + Diffusers | Existing SD 1.5, SDXL Base, FLUX.1-schnell, selected model/VAE/LoRA, CPU prompt encoding and RAM offload |
| C++ `LinearLayer`, `iild-run compute --device rocm` | Optional LibTorch HIP backend | Named linear component, FP32/FP16/BF16, opt-in CPU partition and bounded RAM weight staging |
| `iild-run devices`, Python `hardware.py` | Runtime discovery | HIP build and visible device count; Python additionally reports the GPU name, GFX architecture, HIP version and memory |

Whole diffusion generation still belongs to the independent Python oracle.
The C++ library does not embed Python, merge LoRAs, implement HIP kernels,
or assemble a complete SDXL/FLUX pipeline. This is Radeon GPU support, not
Ryzen AI NPU support or NVIDIA Tensor Core emulation. Intel-Mac Radeon/eGPU,
DirectML, Vulkan, and unlisted legacy Radeon support are not added.

## Select the correct vendor runtime first

Use AMD's GPU/OS/driver/framework compatibility matrix and installation guide
for the exact machine. Linux, WSL, and native Windows packages have different
requirements; a wheel for one platform is not interchangeable with another.
These requirements change independently of this project:

- [AMD ROCm compatibility matrix](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/docs-10.0.0/compatibility/compatibility-matrix.html)
- [AMD Windows PyTorch installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
- [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html)
- [LibTorch C++ SDK installation and CMake setup](https://docs.pytorch.org/cppdocs/installing.html)

The linked older Windows guide is release-specific; consult the current
unified AMD matrix before choosing a package. Do not copy the macOS Python
3.14 environment or install a generic CPU/NVIDIA Torch wheel on Radeon.
Do not spoof `HSA_OVERRIDE_GFX_VERSION` as evidence of supported hardware.
No driver, framework distribution, paid service, or model is automatically
downloaded by device discovery.

## Python image generation

Create a separate environment under the checkout's `build/` directory, using
the Python version supported by AMD's selected wheel. Install that vendor
HIP PyTorch distribution using AMD's instructions. Before any model download:

```bash
python reference/diffusers/hardware.py --device rocm --dtype float16
```

The standalone probe needs only Torch. It requires a non-empty
`torch.version.hip` and a usable GPU, executes a matrix product on that GPU,
waits for completion, and checks its result. A CPU-only or NVIDIA build,
unsupported GPU/driver/dtype, or incorrect result fails rather than falling
back to another device. Use `--dtype bfloat16` to test that dtype separately.

Then install the common pinned oracle dependencies into the **same**
environment, without imposing the macOS environment's Torch pin:

```bash
python -m pip install -r reference/diffusers/requirements-rocm.txt
python reference/diffusers/hardware.py --device rocm --dtype float16
python reference/diffusers/generate.py --preset sdxl-base --device rocm \
  --model /absolute/path/base.safetensors \
  --vae /absolute/path/vae.safetensors \
  --lora /absolute/path/style.safetensors --lora-scale 0.75 \
  --cpu-text-encoding --offload model \
  --output build/reference/radeon/sdxl-lora.png
```

Omit optional VAE/LoRA arguments when not supplied. The existing preset,
checkpoint-layout, auxiliary-config, revision and license contracts still
apply; the filename extension alone does not establish model compatibility.
Use `--local-files-only` when all required configurations and auxiliary
components are already cached. This feature supplies no new model defaults.

`--device auto` also chooses an available HIP GPU ahead of Metal. PyTorch
deliberately names HIP tensor devices `cuda`, so Diffusers/Accelerate receives
`cuda`, not the invalid Torch device `rocm`. The existing Python
`--device cuda` remains a namespace-compatible alias on HIP builds, but the
sidecar correctly records `runtime.hardware.backend = "rocm"` and a `rocm`
inventory object. `device` / `execution_device` may therefore say `cuda` on
AMD hardware. This must not be interpreted as NVIDIA execution.

`--cuda-tf32` and `--no-cuda-tf32` are rejected on HIP builds. Automatic
execution does not touch NVIDIA matmul/cuDNN precision setters on AMD.
FP16/BF16 kernels and matrix-unit dispatch remain the vendor runtime's job;
no matrix-core count or profiler-confirmed utilization is invented.
CPU encoding follows LoRA activation and precedes GPU placement. Model and
sequential offload continue to use Accelerate's existing RAM hooks.

`requirements.txt` still pins Torch 2.13.0 for the original environment;
`requirements-common.txt` holds the other unchanged direct pins.
`requirements-rocm.txt` does not pin or fetch an AMD Torch build itself.
Install the compatible vendor wheel first and rerun the probe after package
installation. Every generation records its actual installed versions.

## Native C++ / LibTorch

The optional backend uses an externally supplied LibTorch 2.x SDK, version
2.9 or newer, built with HIP for actual Radeon execution. Supply the CMake
package from that SDK; when using an appropriate PyTorch wheel, inspect
`torch.utils.cmake_prefix_path`. A CPU/NVIDIA SDK can compile the bridge, but
will never be reported as ROCm-capable. Linux C++11-ABI builds are required;
legacy `_GLIBCXX_USE_CXX11_ABI=0` distributions are rejected.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DIILD_ENABLE_MLX=OFF -DIILD_ENABLE_COREML=OFF \
  -DIILD_ENABLE_LIBTORCH=ON \
  -DTorch_DIR=/absolute/path/vendor/torch/share/cmake/Torch
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
./build/iild-run devices
./build/RocmTests --require-rocm
./build/iild-run compute --device rocm --precision fp16 \
  --cpu-share 0.25 --weight-storage ram --gpu-weight-mib 64
```

`IILD_ENABLE_LIBTORCH` defaults to `OFF` so Metal/CUDA-only installations do
not acquire a second large runtime. Once enabled with a working HIP SDK,
automatic native selection is NVIDIA/MLX CUDA, then AMD/LibTorch ROCm, then
MLX Metal. CPU is always explicit. `--device-index N` selects a visible AMD
device; invalid indices fail. Native `--device cuda` still means NVIDIA/MLX,
unlike PyTorch's shared device namespace.

LibTorch is a private dynamic dependency. Installed consumers need only
iiLocalDiffusion's C++ headers and CMake target, but the configured LibTorch
and ROCm libraries must remain installed. They are **not** copied into the
package. Unix installs include the explicit SDK library directory in their
runtime search path; on Windows the SDK DLL directories must be on `PATH`.
Relocation of iiLocalDiffusion does not relocate the external vendor SDK.
Native Windows ROCm compilation/execution has not been verified here.
The matching SDK `LICENSE` is installed with iiLocalDiffusion's notices.
If a custom SDK omits its license location, supply `IILD_LIBTORCH_LICENSE`.

```cpp
#include <Compute/LinearLayer.hpp>

const auto amd = iild::rocmCapabilities();
// amd.hip is the actual LibTorch ROCm build, not merely CUDA namespace support.
auto layer = iild::LinearLayer::fromSafetensors(
    "/absolute/path/text_encoder.safetensors",
    "text_model.encoder.layers.0.mlp.fc1.weight",
    "text_model.encoder.layers.0.mlp.fc1.bias",
    {iild::ComputeDevice::rocm, 0},
    {0.25, iild::WeightStorage::ram, 64 * 1024 * 1024},
    {iild::LinearPrecision::float16});
```

The existing `ComputeCapabilities` layout and existing enum values are
unchanged; `ComputeDevice::rocm`, `RocmCapabilities`, and the three-argument
`selectComputeDevice()` overload are additive. Native device labels identify
the logical ROCm index, not a queried marketing model name; use the Python
probe for detailed GPU inventory. `LinearLayer` uses a private component-only
backend boundary. No backend tensor type escapes the library.

CPU shares always use FP32; the GPU uses the requested precision. A real
arithmetic preflight rejects an unsupported dtype before weights are retained.
Scoped dispatch guards prevent a host application's autocast context from
silently changing the component's requested dtype; the caller's context is restored.
RAM mode stores reduced-precision GPU weights on the host and stages bounded
output-column blocks. Logical weight-byte counts exclude inputs, activations,
allocator caches and temporary workspace; they are not peak VRAM guarantees.
Concurrent calls on one component are serialized; independent CPU columns
execute alongside GPU work, and failed GPU calls still join the CPU worker.

Both `.safetensors` and `.safetensor` are accepted with exact keys. The private
pinned safetensors-cpp reader maps the file; the wrapper bounds shapes and
offsets, validates selected ranks/dtypes, and rejects non-finite values.
LibTorch owns dtype conversion and copies into its own storage. As with the
existing native reader, callers must supply trusted, immutable local files;
this is not a security sandbox or whole-checkpoint compatibility verifier.

## Verification boundary

On the current Apple Silicon host, the optional LibTorch 2.13.0 bridge was
compiled and executed on CPU, including fixed numerical oracles,
FP32/FP16/BF16 source files, invalid files, concurrent calls and installed
consumer use. MLX/Metal and Core ML regressions also pass. Python AMD policy
tests use explicit simulated HIP devices; they are not physical GPU evidence.

No Radeon is connected to this host. ROCm driver initialization, GPU kernels,
CPU/GPU overlap, large-model memory use, Linux/Windows deployment and AMD
SDXL/FLUX image quality remain unverified on actual AMD hardware. The native
`RocmTests --require-rocm` gate and Python `hardware.py --device rocm` probe
fail on this host rather than reporting that hardware validation succeeded.
Passing detection or arithmetic does not claim profiler-verified matrix-unit
utilization or a speedup.
