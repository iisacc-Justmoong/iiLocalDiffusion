<a id="hardware-compute"></a>

# 하드웨어 컴퓨팅

GPU 계산은 기본 정책입니다. MLX 은 실행 런타임 이고, Metal 와 CUDA 은 하드웨어 백엔드이며, 3 상호 교환 가능한 장치 이름이 아닙니다. AMD Radeon 은 별도의 선택적 LibTorch /ROCm 백엔드를 추가하며, MLX 장치 별칭이 아닙니다.

<a id="implemented-paths"></a>

## 구현된 경로

|진입점| 런타임 |가속기|범위|
|---|---|---|---|
| C++ `LinearLayer` | MLX 0.32.2 |Apple Silicon에 대한 Metal, Linux / NVIDIA에 대한 CUDA|이름이 지정된 safetensors 가중치를 포함하는 묶음 선형 구성 요소|
| C++ `LinearLayer` |옵션 LibTorch HIP|지원되는 AMD Radeon/ROCm GPU|동일한 구성 요소 API, 정밀도, CPU 파티션 및 RAM 스테이징|
| C++ `CoreMLModel` |시스템 Core ML|Apple 신경 엔진(명시적으로 허용된 CPU/GPU 포함)|호출자가 제공하는 컴파일된 고정 모양 구성 요소. ANE-기본적으로 필요한 계획|
| Python `generate.py` | PyTorch / Diffusers |Metal ~ MPS, NVIDIA CUDA, AMD ROCm|기존 SD 1.5, SDXL Base, 및 FLUX.1 -schnell 생성 오라클|
| `iild-run inspect` |C++ / json-c|필요 없음|패키지 메타데이터 및 아티팩트 경로만|

MLX 텐서·GPU 커널·할당·그래프 실행은 비공개 구현 세부사항으로 유지한다. C++ 라이브러리는 Python을 호출하지 않는다. 이 변경은 완전한 MLX 텍스트 인코더·디노이저·VAE·확산 파이프라인을 구현하지 않는다. 따라서 `--device mlx`는 Python 생성 옵션이 아니다. 별도의 Core ML 실행 경로·하드웨어 탐색·CUDA 행렬 정밀도·계획과 측정된 사용량의 구분은 [Neural Engine과 Tensor Cores](neural-accelerators.md)를 참고한다. Tensor Cores는 네 번째 장치 유형이 아니라 CUDA 실행 유닛이다. PyTorch/MLX가 자동으로 ANE 백엔드가 되는 것은 아니다.

<a id="default-selection-and-failure-policy"></a>

## 기본 선택 및 실패 정책

`ComputeOptions{}`  선택 가능한  MLX   CUDA 를 선택한 후  LibTorch ROCm, 다음  MLX   Metal 를 선택합니다. Python   `generate.py --device auto`  설치된  PyTorch   CUDA / HIP   GPU 를 먼저 선택한 다음  Metal 을 선택합니다. 아무런 알림 없이 선택하지도 CPU 를 선택하지도 않습니다. 사용할 수 없는 GPU, 지원되지 않는 작업, 또는 메모리 부족 실패가 CPU 에서 재시도 대신 보고됩니다. CPU 만의 산술 연산은 명시적인 `ComputeDevice::cpu` 또는 `--device cpu` 가 필요합니다. 협업 CPU 작업은 네이티브 컴포넌트를 위한 `cpuShare` 또는 이미지 오라클을 위한 `--cpu-text-encoding` 를 통해 옵트인됩니다.

Python 옵션은 `--device mps` 의 `--device metal` 의 별칭입니다. 모델 가중치를 로드하기 전에 생성이 선택된 장치와 데이터 형식에서 행렬 곱을 실행하고 확인합니다. 모델 배치/오프로드 설정 후 파이프라인의 실제 실행 장치를 확인합니다. 사이드카는 JSON, `runtime.hardware`, 백엔드, 요청된 장치, GPU-가속기 플래그, 사전 검사 결과, 그리고 대체 경로 정책이 모두 기록됩니다. `PYTORCH_ENABLE_MPS_FALLBACK=1` 은 Metal 생성이 거부됩니다. `--device rocm` 는 실제 HIP 빌드와 사용 가능한 GPU 를 필요로 합니다. Python 실행 네임스페이스는 provenance 기록이 `backend: rocm` 하는 동안 `cuda` 로 유지됩니다. The [Radeon 가이드](radeon-rocm.md) 는 SDK 선택, CPU / RAM 협력, NVIDIA -특정 옵션 거부, 모델 없는 사전 검사 사전 검사 및 검증되지 않은 플랫폼을 다룹니다.

파일 입출력, 토큰화, 결정론적 CPU 랜덤 생성기, 명시적 출력 리드백, 및 FLUX 의 CPU 가중치 저장/오프로드는 호스트 작업이며, 모든 작업이 GPU 에서 실행된다는 약속이 아닙니다. 특히, FLUX 의 순차적 오프로드는 신경 구성 요소를 선택된 가속기에 배치하며, CPU 를 디노이징 장치로 선택하지 않습니다.

<a id="native-build"></a>

## 네이티브 빌드

기본 `IILD_ENABLE_MLX=ON`는 고정된 MLX C++ 소스와 선택된 백엔드를 빌드한다. 네이티브 연산에는 Python 런타임가 필요하지 않다.

Apple Silicon 에서 C++ 빌드 요구 사항 외에도 Xcode 의 Metal 컴파일러를 설치하세요. `xcrun metal --version` 를 확인하세요; `metal` shim 만을 찾는 것은 컴파일러가 설치되었다는 것을 확립하지 않습니다. Xcode 가 누락된 Metal 툴체인 를 보고하면, 지원되는 다운로드 명령은 다음과 같습니다:

```bash
xcodebuild -downloadComponent MetalToolchain \
  -exportPath "$PWD/build/dependencies/metal-toolchain"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/iild-run devices
./build/iild-run compute
```

Xcode 는 활성 컴파일러 구성 요소를 관리하며, 내보내기 경로는 `build/` 하에 복사본을 유지합니다. 배포 대상은 설치된 json-c 및 컴파일러 SDK 와도 호환되어야 합니다. 검증된 빌드는 호스트 OS 를 대상으로 하며, 독립적으로 테스트된 최소 macOS 배포 버전이 아닙니다.

Linux/NVIDIA 소스 빌드의 경우 CUDA 도구 키트, 호환 가능한 NVIDIA 드라이버, cuDNN 개발 라이브러리 및 BLAS/LAPACK 개발 라이브러리를 설치합니다. 그런 다음 CUDA 백엔드를 명시적으로 요구합니다.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DIILD_MLX_GPU_BACKEND=cuda
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/iild-run compute --device cuda
```

고정된 런타임의 [소스 빌드 요구사항](https://github.com/ml-explore/mlx/blob/v0.32.2/docs/src/install.rst)을 따른다. CMake 구성은 CUDA Toolkit 13.1을 명시적으로 거부한다. 보이는 GPU가 없는 헤드리스 교차 빌드에서는 `MLX_CUDA_ARCHITECTURES`도 대상 아키텍처로 설정해야 한다. 툴킷과 cuDNN는 시스템 의존성이며 이 프로젝트가 자동으로 구매하거나 설치하거나 재배포하지 않는다.

|CMake 옵션|기본값|의미|
|---|---|---|
| `IILD_ENABLE_MLX` | `ON` |MLX 컴포넌트 백엔드를 빌드하세요; LibTorch 와 Core ML 에 독립적입니다.|
| `IILD_ENABLE_LIBTORCH` | `OFF` |외부에서 제공되는 Torch를 사용하여 선택적 AMD ROCm/명시적 CPU 백엔드 구축 SDK|
| `IILD_ENABLE_COREML` |`ON` (Apple) 또는 `OFF` (그 외)|독립형 네이티브 Core ML 컴포넌트 지원; macOS SDK 를 노출하는 14.4 API 를 필요로 함|
| `IILD_MLX_GPU_BACKEND` | `auto` |Apple Silicon 은 Metal 를 선택하며, Linux 에 CUDA 컴파일러가 있는 경우 CUDA 를 선택함|
| `IILD_MLX_GPU_BACKEND=metal` |명시적|Apple Silicon Metal 빌드를 필요로 함|
| `IILD_MLX_GPU_BACKEND=cuda` |명시적|Linux / NVIDIA CUDA 빌드를 필요로 함|
| `IILD_MLX_GPU_BACKEND=none` |명시적|MLX의 CPU 백엔드만 빌드합니다. 자동 실행이 여전히 실패함|

Linux 에 CUDA 컴파일러가 없는 경우, 자동 구성은 CPU 전용 빌드를 보고함. 이는 자동 CPU 실행을 활성화하지 않음. 대신 Apple Silicon 은 `none` 또는 MLX `OFF` 가 명시적이지 않은 한 작동하는 Metal 컴파일러를 필요로 함. 경량 메타데이터 전용 빌드는 `-DIILD_ENABLE_MLX=OFF -DIILD_ENABLE_COREML=OFF -DIILD_ENABLE_LIBTORCH=OFF` 로 모든 실행 런타임을 비활성화함

MLX 는 사내 공유 라이브러리 의존성임. 설치된 Metal 셰이더 라이브러리 ( `lib/mlx.metallib` ), CUDA JIT 헤더 (적용 가능한 경우), 및 제 3 자 공지사항은 설치의 일부임. 소비자 테스트는 설치를 재위치시키고 `iiLocalDiffusion::iiLocalDiffusion` 만 연결하며 네이티브 컴퓨팅을 실행함. 소스 아카이브와 해시는 `cmake/IildMlx.cmake` 에 고정됨; 참조
종속성 및 라이선스 세부정보는 [dependencies.md](dependencies.md)를 참조하세요.

<a id="c-component-api"></a>

## C++ 구성 요소 API

```cpp
#include <Compute/LinearLayer.hpp>
#include <vector>

auto layer = iild::LinearLayer::fromWeights(
    std::vector<float>{1, 2, 3, -1, 0, 1},
    3, 2, std::vector<float>{0.5F, -0.5F});
// 기본값은 MLX와 사용 가능한 GPU이며 CPU를 암묵적으로 사용하지 않는다.
auto output = layer.forward(std::vector<float>{1, 2, 3}, 1);
// output = {14.5F, 1.5F}
const auto &actualDevice = layer.computeInfo();

auto fromFile = iild::LinearLayer::fromSafetensors(
    "/absolute/path/to/text-encoder.safetensors",
    "text_model.encoder.layers.0.mlp.fc1.weight",
    "text_model.encoder.layers.0.mlp.fc1.bias");
```

가중치는 `[outputFeatures, inputFeatures]`, 편향은 선택 사항인 `[outputFeatures]`, 입력은 행 주 `[batch, inputFeatures]`, 출력은 행 주 `[batch, outputFeatures]` 임 네이티브 파일 로딩은 float16, bfloat16, 및 float32 가중치를 허용함 계산은 기본적으로 float32 배열이며, 가산 `LinearMathOptions` 오버로드는 GPU FP16 또는 BF16 를 선택하여 FP32 에 있는 모든 협력 CPU 샤드와 FP32 에 있는 호스트 출력을 유지함 CUDA 은 float32 행렬 곱셈에 대해 MLX 의 기본 TF32 가속화 정책을 사용하며, 전체 float32 곱셈 정밀도가 필요할 때는 프로세스 시작 전에 `MLX_ENABLE_TF32=0` 을 설정함 양자화, 정수, 복소수 및 부동소수점64 가중치가 지원되지 않습니다. 명시적인 안전한 로더인 MLX 를 통해 `.safetensors` 와 `.safetensor` 가 모두 허용되며, 피클 대체 경로 는 없습니다.

호스트의 정확한 텐스키만 선택됩니다. 파일 읽기는 호스트 작업이며, 선택된 가중치는 구성이 반환되기 전에 요청된 스트림에 구체화됩니다. 그런 다음 파일은 해제될 수 있습니다. 이는 전체 SD / SDXL / FLUX 체크포인트를 분류하거나 실행하지 않으며, LoRA 어댑터를 적용하지 않습니다. 호출자 관리 파일은 네이티브 구성 요소 API 에 의해 인증되지 않으며, 아래 비교 하네스는 별도로 SHA-256 신원을 기록하고 확인합니다.

각 인스턴스는 가중치와 명시적인 컴퓨스트림을 유지합니다. 배치된 행렬 곱셈, 편향 추가, 및 변환은 해당 스트림에서 MLX 로 위임됩니다. 호출은 결과를 호스트 메모리에 동기화하여 명시적으로 복사한 후 `std::vector<float>` 를 반환합니다. 구성 요소는 스트림에서 호출을 직렬화하며, 그것은 multi- GPU 모델 샤딩 또는 비동기 스케줄링 API 이 아닙니다. `ComputeOptions.deviceIndex` 는 하나의 네이티브 장치를 선택하고 사용 불가능한 인덱스를 거부합니다.

`iild-run compute`는 검증된 `[64,256] x [256,128]` 선형 작업을 실행합니다. 이는 이미지 생성기가 아닌 하드웨어 진단입니다. `devices` 및 `compute`는 모두 실제 네이티브 런타임/백엔드를 보고합니다. CPU 전용 계산을 의도하는 경우에만 `compute --device cpu`를 사용하세요.

<a id="cpugpu-cooperation-and-ram-storage"></a>

## CPU/GPU 협력 및 RAM 스토리지

RAM 는 가중치 및 워크스페이스 저장소이며, 처리 장치가 아닙니다. 애플 실리콘에서 CPU 와 GPU 는 물리적으로 통합된 메모리를 공유하며, RAM 에 가중치를 할당하면 메모리 용량을 늘리거나 시스템 메모리 한계를 피하지 않습니다. CUDA 에서 호스트 RAM 와 GPU 장치 할당은 별개입니다. 구현은 기존 MLX CPU / GPU 스트림과 Diffusers /Accelerate 오프로드 시설을 사용하며, 커스텀 커널이나 새로운 메모리 할당자를 사용하지 않습니다. 상위 공급 측 를 참조하세요.
[MLX 통합 메모리 가이드](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) 및 [Diffusers 메모리 가이드](https://huggingface.co/docs/diffusers/optimization/memory).

<a id="native-linear-component"></a>

### 네이티브 선형 구성요소

추가 `LinearResourceOptions` 오버로드는 기존 호출자를 보존합니다.

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

`fromWeights(weights, inputs, outputs, bias, options, resources)`는 동일한 정책을 제공합니다. 기본값은 `cpuShare=0`, 장치 상주 가중치 및 RAM 스토리지가 선택된 경우에만 사용되는 64 MiB 스테이징 예산으로 유지됩니다.

|옵션|계약|
|---|---|
| `cpuShare` / `--cpu-share` |`[0,1)`의 유한 분수. 양수 공유는 CPU에서 `[1, outputs-1]`에 고정된 `floor(outputs * share)`를 계산합니다. GPU와 최소 2개의 출력 기능이 필요합니다. 이는 CPU 활용률이 아닙니다.|
| `WeightStorage::device` / `--weight-storage device` |CPU 샤드는 CPU에 유지되고 나머지 가중치는 GPU에 유지됩니다.|
| `WeightStorage::ram` / `--weight-storage ram` |호스트 RAM의 모든 가중치를 유지합니다. 출력 기능 블록에 GPU 샤드를 보냅니다. CPU 전용 실행에서는 이미 RAM를 사용하고 있습니다.|
| `gpuWeightBudgetBytes` / `--gpu-weight-mib` |선택한 GPU 정밀도에서 하나의 논리적 가중치/편향 블록에 대한 상한입니다. 하나 이상의 출력 행과 해당 편향이 맞아야 합니다. CLI는 양수 전체 MiB를 허용하며 RAM와 GPU가 필요합니다.|

CPU 샤드는 자체 MLX CPU 스트림을 사용하는 독립적인 워커에서 실행되지만, 호출 스레드는 GPU 샤드를 실행합니다. 결과는 원래 행-주 출력 순서로 병합됩니다. 같은 레이어에 대한 호출은 직렬로 유지되며, 실패 또한 반환하기 전에 CPU 워커에 연결됩니다. MLX 는 스케줄링과 커널을 소유하며, 동시 제출은 속도 향상, 모든 커널에 대한 오버랩, 또는 특정 활용 수준을 보장하지 않습니다.

`resourceInfo()` 는 CPU / GPU 출력 개수, 논리적 호스트/거주 GPU 가중치 바이트, 및 스테이지된 블록 바이트/행 개수를 보고합니다. 이는 RSS, 커밋된 메모리, 또는 VRAM 피크의 측정값이 아닙니다. 활성화, 출력 버퍼, 로딩 임시 파일, MLX 할당 캐시, 및 커널 스크래치 메모리는 가중치 예산 밖에 있습니다. RAM 는 디스크 스왑이 아니며, 자동 디스크 오프로딩은 추가되지 않습니다.

```bash
./build/iild-run compute --cpu-share 0.25 --weight-storage ram --gpu-weight-mib 64
./build/iild-run compute --device cpu --weight-storage ram
```

<a id="python-image-generation"></a>

### Python 이미지 생성

`--cpu-text-encoding` 는 LoRA 활성화 후 CPU 에서 CLIP/T5 임베딩을 계산합니다. 디노이저와 VAE 는 여전히 선택된 GPU 에서 실행됩니다. CPU 인코딩은 GPU 디노이징보다 앞서 수행됩니다. 왜냐하면 임베딩이 그 입력이기 때문입니다. 이 경로는 단일 디노이징 작업의 동시 분할이 아닙니다. SD 1.5/SDXL CPU 인코딩은 부동소수점32를 사용하며, FLUX 는 기본적으로 bfloat16 를 사용합니다. `--dtype` 와 `--cpu-text-dtype` 는 별도의 모델 및 CPU -인코더 오버라이드를 노출합니다. 인코더는 배치 전에 원래 저장 데이터 타입으로 돌아갑니다. Python CUDA /ROCm 장치 인덱스, 생성자 배치 및 메모리/정밀도 정책은 [생성 값 스키마](generation-parameters.md)에 포함됩니다. 복합 정밀도 어댑터 텐서/버퍼는 복원 과정에서 기본 인코더의 정밀도로 강하되지 않고 원래 값과 dtype 을 유지합니다. 추론 전에 모든 결과 조건부 텐서, SDXL 풀링된 음수들을 포함하여 선택된 장치/dtype 으로 명시적으로 전송됩니다.

`--cpu-threads N`는 양의 PyTorch 작업 내 CPU 스레드를 설정합니다. 이를 생략하면 런타임 기본값이 유지됩니다. 별도의 MLX 런타임를 구성하지 않습니다.

| `--offload` |GPU 중량 정책|
|---|---|
|`auto`(기본값)|기존 사전 설정: SD/SDXL 상주, FLUX 순차. CPU 텍스트 인코딩을 사용하면 SD/SDXL가 모델 오프로드를 사용하므로 사용되지 않는 텍스트 인코더는 RAM에 유지됩니다.|
| `none` |파이프라인을 GPU로 이동합니다. RAM 오프로드 후크가 없습니다. 이 명시적 재정의는 이미 사용된 텍스트 인코더를 GPU로 이동할 수도 있습니다.|
| `model` |RAM에서 비활성 모델을 유지합니다. 호출 시 전체 모델을 GPU로 이동합니다. 가장 큰 실행 모델과 해당 활성화가 적합해야 합니다.|
| `sequential` |RAM에 가중치를 유지합니다. 일반적으로 전송 비용이 더 높은 호출된 대로 하위 모듈을 GPU로 이동합니다.|

명시적 모델/순차 오프로드에는 GPU 실행이 필요합니다. 오프로드만으로는 CPU에서 텍스트/노이즈 제거를 계산하지 않습니다. `--device cpu`를 사용하면 `auto`는 CPU/RAM를 직접 사용합니다. 장애가 발생해도 장치가 자동으로 전환되거나 정책이 오프로드되지는 않습니다.

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py --model /absolute/path/image-diffusers \
  --preset sdxl-base --device auto --cpu-text-encoding --cpu-threads 4 \
  --offload model --local-files-only \
  --output build/reference/sdxl-cpu-ram.png
```

사이드카 JSON 기록은 `runtime.cpu_conditioning` (실제 CPU 텐서, 구성 요소, dtype, 형상), `runtime.cpu_threads`, `runtime.hardware.participating_devices`, 및 `runtime.optimization` (`requested_offload`, 유효 `offload_policy`, 및 `weight_storage`)) 를 포함합니다. 모델, VAE, 및 LoRA provenance 는 변경되지 않습니다. 이 필드들은 모든 작업이 계측되었다는 주장이나 메모리/잠재 표현이 개선되었다는 주장을 하지 않습니다.

<a id="numerical-verification"></a>

## 수치 검증

일반 테스트는 장치 정책, CPU / GPU 분할, 동시 호출, 편향 없음 RAM 블록, 예산 거부, 네이티브 safetensors 수명, CPU 텍스트 인코딩/ LoRA /오프로드 순서, 임베딩 전송, CLI 실패 계약, 및 설치된 협력 API 를 다룹니다. 장치 정책 테스트는 물리적 CUDA 검증으로 간주되지 않습니다.

실제 체크포인트 구성 요소를 독립적으로 실행된 PyTorch CPU 선형 작업과 비교합니다.

```bash
reference/diffusers/.venv/bin/python reference/validation/check_mlx_linear.py \
  --model /absolute/path/to/text-encoder.safetensors \
  --weight-key text_model.encoder.layers.0.mlp.fc1.weight \
  --bias-key text_model.encoder.layers.0.mlp.fc1.bias \
  --output-dir build/reference/hardware/my-linear-parity
```

출력 디렉토리는 새이거나 비어 있어야 합니다. 해arness 는 선택된 텐서만 내보내고, 네이티브 GPU 및 CPU / GPU / RAM 테스트를 실행하며, 소스/ 픽스처 해시, 텐서 키, 시드, dtype, 런타임 버전, 형상, 및 최대 절대 오차를 기록합니다. 각 요소에 대한 수락 규칙은 `absolute_error <= 0.0004 + 0.0001 * abs(reference)` 입니다. 네이비 비교 하위 프로세스는 TF32 를 `MLX_ENABLE_TF32=0` 로 명시적으로 비활성화하고 해당 설정을 기록합니다. 이 비교에는 GPU 가 필수이며, 런타임 는 일반 호출자에서 기본값으로 설정되어 수정되지 않습니다. 하이브리드 비교는 1 MiB 가중치 블록 예산을 사용하며, 단일 행이 더 크다면 한 행으로 증가됩니다. 한 출력 레이어는 GPU 비교만 실행할 수 있으며 `hybrid: null` 를 기록하며 해당 형식에 대해 CPU / GPU 협력은 주장되지 않습니다.

2026-09-03에서 검증된 M1 Max 호스트는 배치 4, 입력 너비 768, 출력 너비 3072 를 통해 MLX / Metal 의 실제 SD 1.5 CLIP `fc1` 구성 요소를 실행했습니다. PyTorch 에 대한 최대 절대 오차는 `9.05991e-06` 입니다. 기존 SD 1.5 이미지 오라클도 자동 Metal 선택과 명시적 모델 VAE, 그리고 LoRA 입력으로 완료되었습니다. SDXL Base 는 128 x 128 에서 2 단계로 완료되었습니다. 64 x 64의 한 단계 작은 FLUX 스모크는 순차적 CPU 가중치 오프로드를 통해 Metal 실행을 확인하며 전체 12B 모델의 메모리 요구 사항이나 이미지 품질을 확립하지 않습니다. 증거는 `build/reference/hardware/` 아래에 있으며 이는 실행/수치 확인이며 이미지 품질 벤치마크나 완전한 네이티브 확산 파이프라인의 증거가 아닙니다.

추가 CPU / RAM 검증은 `build/reference/resources/mlx-cpu-gpu-ram-linear/provenance.json` 를 사용하며: 동일한 CLIP 레이어는 768 CPU 와 2,304 GPU 출력 기능을 사용하여 실행됩니다. 모든 9,449,472 가중치/편향 바이트는 논리적으로 RAM 에 존재하며 GPU 스테이지는 1,048,576바이트 예산으로 제한됩니다. RAM 스테이지와 장치 상주 하이브리드 실행 모두에 대해 최대 절대 오차는 다시 `9.05991e-06` 입니다. 별도의 SD 1.5 (128 정사각형, 4 단계, 사용자 정의 모델/ VAE /합성 LoRA ) 및 SDXL Base (128 정사각형, 2 단계) 이미지 또한 CPU 텍스트 인코딩과 Metal 모델 오프로드를 통해 완료되었습니다. 이것은 저해상도 실행 테스트이며 성능 또는 이미지 품질 벤치마크가 아닙니다. 캐싱된 작은 FLUX 파이프라인 ( 64 제곱, 한 단계, 합성 트랜스포머 LoRA ) 또한 CPU 인코딩과 순차적 오프로드를 완료했습니다. 포워드 가드가 오프로드된 텍스트 인코더가 GPU 에서 다시 호출되지 않았음을 확인했습니다. 이것은 공식 FLUX 프리셋의 전체 모델 호환성 검사를 우회하지 않으며, 고의적으로 작은 API 스모크는 격리된 검증 스크립트를 사용합니다.

해당 호스트에서는 NVIDIA GPU를 사용할 수 없습니다. CUDA 빌드 선택 및 런타임 실패 정책이 구현되지만 CUDA 컴파일 및 물리적 실행은 호환 가능한 NVIDIA 하드웨어에서 동일한 테스트가 실행될 때까지 확인되지 않은 상태로 유지됩니다.
