<a id="radeon--amd-rocm"></a>

# 라데온/AMD ROCm

AMD GPU 실행은 PyTorch/LibTorch의 HIP 런타임를 통해 구현됩니다. 이는 NVIDIA CUDA 및 Apple Metal와 별도의 백엔드이며 모든 Radeon 카드 또는 운영 체제가 ROCm을 지원한다는 주장은 아닙니다.

|진입점|AMD 경로|범위|
|---|---|---|
| Python `generate.py --device rocm` |공급업체 HIP PyTorch + Diffusers|기존 SD 1.5, SDXL 베이스, FLUX.1-schnell, 선택한 모델/VAE/LoRA, CPU 프롬프트 인코딩 및 RAM 오프로드|
| C++ `LinearLayer`, `iild-run compute --device rocm` |선택적 LibTorch HIP 백엔드|명명된 선형 구성 요소, FP32/FP16/BF16, 옵트인 CPU 파티션 및 한계가 설정된 RAM 웨이트 스테이징|
| `iild-run devices`, Python `hardware.py` |런타임 발견|HIP 빌드 및 가시 장치 수; Python는 추가로 GPU 이름, GFX 아키텍처, HIP 버전 및 메모리를 보고합니다.|

전체 확산 생성은 여전히 독립적인 Python 오라클에 속합니다. C++ 라이브러리는 Python 를 내장하거나 LoRAs 를 병합하거나 HIP 커널을 구현하거나 SDXL / FLUX 파이프라인을 조립하지 않습니다. 이는 Radeon GPU 지원이며, Ryzen AI NPU 지원 또는 NVIDIA 텐서 코어 에뮬레이션이 아닙니다. Intel-Mac Radeon/ eGPU, DirectML, Vulkan 및 목록에 없는 레거시 Radeon 지원은 추가되지 않습니다.

<a id="select-the-correct-vendor-runtime-first"></a>

## 먼저 올바른 공급업체 런타임를 선택하세요.

AMD 의 GPU / OS /드라이버/프레임워크 호환성 행렬 및 설치 가이드를 정확한 머신에 사용하세요. Linux, WSL 및 네이티브 Windows 패키지는 서로 다른 요구사항을 가지며, 하나의 플랫폼용 휠은 다른 플랫폼과 상호 교환할 수 없습니다. 이 요구사항은 이 프로젝트와 독립적으로 변경됩니다:

- [AMD ROCm 호환성 매트릭스](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/docs-10.0.0/compatibility/compatibility-matrix.html)
- [AMD Windows PyTorch 설치](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
- [PyTorch HIP 의미론](https://docs.pytorch.org/docs/stable/notes/hip.html)
- [LibTorch C++ SDK 설치 및 CMake 설정](https://docs.pytorch.org/cppdocs/installing.html)

연결된 이전 Windows 가이드는 릴리스별이며, 패키지를 선택하기 전에 현재 통합 AMD 행렬을 참조하십시오. Radeon 에서 macOS Python 3.14 환경을 복사하거나 일반적인 CPU / NVIDIA Torch 휠을 설치하지 마십시오. `HSA_OVERRIDE_GFX_VERSION` 를 지원되는 하드웨어의 증거로 위조하지 마십시오. 장치 발견을 통해 자동으로 다운로드되는 드라이버, 프레임워크 배포판, 유료 서비스 또는 모델은 없습니다.

<a id="python-image-generation"></a>

## Python 이미지 생성

AMD가 선택한 휠에서 지원하는 Python 버전을 사용하여 결제의 `build/` 디렉터리 아래에 별도의 환경을 만듭니다. AMD의 지침을 사용하여 공급업체 HIP PyTorch 배포판을 설치합니다. 모델을 다운로드하기 전:

```bash
python reference/diffusers/hardware.py --device rocm --dtype float16
```

독립 탐침에는 Torch만 필요하다. 비어 있지 않은 `torch.version.hip`와 사용할 수 있는 GPU가 필요하며 해당 GPU에서 행렬 곱을 실행하고 완료를 기다린 뒤 결과를 확인한다. CPU 전용 또는 NVIDIA 빌드, 지원하지 않는 GPU/드라이버/dtype 또는 잘못된 결과는 다른 장치로 대체하지 않고 실패한다. 해당 dtype을 따로 검사하려면 `--dtype bfloat16`를 사용한다.

그런 다음 macOS 환경의 Torch 핀을 적용하지 않고 공통 고정 Oracle 종속성을 **same** 환경에 설치합니다.

```bash
python -m pip install -r reference/diffusers/requirements-rocm.txt
python reference/diffusers/hardware.py --device rocm --dtype float16
python reference/diffusers/generate.py --preset sdxl-base --device rocm --model-config /absolute/path/model-config \
  --model /absolute/path/base.safetensors \
  --vae /absolute/path/vae.safetensors \
  --lora /absolute/path/style.safetensors --lora-scale 0.75 \
  --cpu-text-encoding --offload model \
  --output build/reference/radeon/sdxl-lora.png
```

공급되지 않은 경우 선택적 VAE / LoRA 인수를 생략합니다. 기존 프리셋, 체크포인트 레이아웃, 보조 구성, 버전 및 라이선스 계약은 여전히 적용되며, 파일 이름 확장자만으로는 모델 호환성을 결정하지 않습니다. 모든 필수 구성과 보조 구성 요소가 이미 캐싱되어 있는 경우 `--local-files-only` 를 사용하세요. 이 기능은 새로운 모델 기본값을 제공하지 않습니다.

`--device auto` 도 사용 가능한 HIP GPU 를 Metal 보다 선택합니다. PyTorch 의도적으로  HIP 텐서 장치를  `cuda` 명명하므로,  Diffusers /Accelerate 는  `cuda` 를, 무효인  Torch 장치  `rocm` 가 아닌 것을 받습니다. 기존 Python `--device cuda` 는 HIP 빌드에서 네임스페이스 호환성 있는 별칭으로 남지만, 사이드카는 `runtime.hardware.backend = "rocm"` 를 올바르게 기록하고 `rocm` 인벤토리 객체를 생성합니다. `device`  /  `execution_device`  따라서  `cuda`  를  AMD  하드웨어에서 말할 수 있습니다. 이것은 NVIDIA 실행으로 해석되어서는 안 됩니다.

`--cuda-tf32` 와 `--no-cuda-tf32` 는 HIP 빌드에서 거부됩니다. 자동 실행은 AMD 에서 NVIDIA matmul/ cuDNN 정밀도 설정자를 만지지 않습니다. FP16/BF16 커널 및 행렬 유닛 디스패치는 벤더 런타임 의 업무로 남아 있으며, 행렬 코어 수나 프로파일러 확인된 활용도가 발명되지 않습니다. CPU 인코딩은 LoRA 활성화에 따라 GPU 배치에 선행합니다. 모델 및 순차 오프로드는 Accelerate 의 기존 RAM 후크를 계속 사용합니다.

`requirements.txt` 는 여전히 원래 환경에 Torch 2.13.0 를 고정하며, `requirements-common.txt` 는 다른 변경되지 않은 직접 고정점을 유지합니다. `requirements-rocm.txt` 는 AMD Torch 빌드 자체를 고정하거나 가져오지 않습니다. 호환 가능한 벤더 휠을 먼저 설치한 후 패키지 설치 후 프로브를 다시 실행하세요. 모든 세대는 실제 설치된 버전을 기록합니다.

<a id="native-c--libtorch"></a>

## 네이티브 C++ / LibTorch

선택형 백엔드는 외부에서 제공된 LibTorch 2.x SDK, 버전 2.9 또는 그 이상으로, 실제 Radeon 실행을 위해 HIP 로 빌드됩니다. 해당 SDK 에서 CMake 패키지를 공급하고, 적절한 PyTorch 휠을 사용할 때 `torch.utils.cmake_prefix_path` 를 검사합니다. CPU / NVIDIA SDK 는 브릿지를 컴파일할 수 있지만, ROCm 지원 가능으로 보고되지는 않습니다. Linux C++11-ABI 빌드가 필요합니다; 레거시 `_GLIBCXX_USE_CXX11_ABI=0` 배포판은 거부됩니다.

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

`IILD_ENABLE_LIBTORCH` 는 `OFF` 로 기본값이므로 Metal / CUDA 전용 설치만으로는 두 번째 큰 런타임 를 획득하지 않습니다. 작동하는 HIP SDK 로 활성화되면 자동 네이티브 선택은 NVIDIA / MLX CUDA, 그 후 AMD / LibTorch ROCm, 그 후 MLX Metal 입니다. CPU 는 항상 명시적입니다. `--device-index N` 는 가시적인 AMD 장치를 선택하며, 유효하지 않은 인덱스는 실패합니다. 네이티브 `--device cuda` 는 여전히 NVIDIA / MLX 를 의미하며, PyTorch 의 공유 장치 네임스페이스와 다릅니다.

LibTorch 는 사적인 동적 의존성입니다. 설치된 소비자는 iiLocalDiffusion 의 C++ 헤더와 CMake 타겟만 필요하지만, 구성된 LibTorch 와 ROCm 라이브러리는 계속 설치되어 있어야 합니다. 그들은 **로** 복사되어 패키지에 포함되지 않습니다. 유닉스 설치에는 명시적인 SDK 라이브러리 디렉터리가 런타임 검색 경로에 포함되며, Windows 에서는 SDK DLL 디렉터리들이 `PATH` 에 있어야 합니다. iiLocalDiffusion 의 이동은 외부 벤더 SDK 를 이동시키지 않습니다. 네이티브 Windows ROCm 컴파일/실행은 여기에서 검증되지 않았습니다. 일치하는 SDK `LICENSE` 는 iiLocalDiffusion 의 공지사항과 함께 설치됩니다. 사용자 지정 SDK 가 라이선스 위치를 생략하면 `IILD_LIBTORCH_LICENSE` 를 제공해야 합니다.

```cpp
#include <Compute/LinearLayer.hpp>

const auto amd = iild::rocmCapabilities();
// amd.hip은 단순한 CUDA 네임스페이스 지원이 아니라 실제 LibTorch ROCm 빌드이다.
auto layer = iild::LinearLayer::fromSafetensors(
    "/absolute/path/text_encoder.safetensors",
    "text_model.encoder.layers.0.mlp.fc1.weight",
    "text_model.encoder.layers.0.mlp.fc1.bias",
    {iild::ComputeDevice::rocm, 0},
    {0.25, iild::WeightStorage::ram, 64 * 1024 * 1024},
    {iild::LinearPrecision::float16});
```

기존 `ComputeCapabilities` 레이아웃과 기존 열거형 값은 변경되지 않으며, `ComputeDevice::rocm`, `RocmCapabilities`, 및 3인수를 받는 `selectComputeDevice()` 오버로드는 추가적입니다. 네이티브 장치 레이블은 논리적 ROCm 인덱스를 식별하며, 조회된 마케팅 모델 이름을 식별하지 않습니다. 상세 GPU 재고 정보를 확인하려면 Python 프로브를 사용하세요. `LinearLayer` 는 비공개 컴포넌트 전용 백엔드 경계를 사용합니다. 백엔드 텐서 타입이 라이브러리를 벗어날 수 없습니다.

CPU 는 항상 FP32 를 공유하며, GPU 는 요청된 정밀도를 사용합니다. 실제 산술 사전 검사 은 가중치가 유지되기 전에 지원되지 않는 데이터 타입을 거부합니다. 범위 디스패치는 호스트 애플리케이션의 자동 변환 컨텍스트가 아무런 알림 없이 컴포넌트의 요청된 데이터 타입을 변경하는 것을 방지하며, 호출자의 컨텍스트가 복원됩니다. RAM 모드는 저정밀도 GPU 가중치를 호스트에 저장하고 한계가 설정된 출력 열 블록을 준비합니다. 논리적 가중치 바이트 수는 입력, 활성화, 할당자 캐시 및 임시 워크스페이스를 제외하며, 최대 VRAM 보장 사항이 아닙니다. 한 컴포넌트에서의 동시 호출은 직렬화되며, 독립적인 CPU 열은 GPU 작업과 함께 실행되고, 실패한 GPU 호출은 여전히 CPU 워커에 참여합니다.

`.safetensors` 과 `.safetensor` 는 정확한 키로 모두 허용됩니다. 비공개 고정 safetensors -cpp 리더는 파일을 매핑하며, 래퍼는 형상과 오프셋을 제한하고 선택된 랭크/데이터 타입을 검증하며, 비유한 값을 거부합니다. LibTorch 는 데이터 타입 변환을 소유하고 자체 저장소에 복사합니다. 기존 네이티브 리더와 마찬가지로 호출자는 신뢰할 수 있고 불변의 로컬 파일을 제공해야 하며, 이는 보안 샌드박스나 전체 체크포인트 호환성 검증자가 아닙니다.

<a id="verification-boundary"></a>

## 검증 경계

현재 Apple Silicon 호스트에서 선택 사항인 LibTorch 2.13.0 브릿지가 CPU 에서 컴파일 및 실행되었으며, 고정된 수치 오라클, FP32/FP16/BF16 소스 파일, 유효하지 않은 파일, 동시 호출 및 설치된 소비자 사용이 포함됩니다. MLX / Metal 및 Core ML 회귀도 통과합니다. Python AMD 정책 테스트는 명시적인 시뮬레이션된 HIP 장치를 사용하며, 물리적 GPU 증거가 아닙니다.

이 호스트에 Radeon 이 연결되어 있지 않습니다. ROCm 드라이버 초기화, GPU 커널, CPU / GPU 오버랩, 대규모 모델 메모리 사용, Linux / Windows 배포 및 AMD SDXL / FLUX 이미지 품질은 실제 AMD 하드웨어에서 확인되지 않았습니다. 네이티브 `RocmTests --require-rocm` 게이트 및 Python `hardware.py --device rocm` 프로브는 이 호스트에서 실패하며, 하드웨어 검증이 성공했다는 보고를 대신합니다. 통과 탐지 또는 산술은 프로파일러 검증된 매트릭스 유닛 활용도나 속도 향상을 주장하지 않습니다.
