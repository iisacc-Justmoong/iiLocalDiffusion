<a id="neural-engine-and-tensor-cores"></a>

# 신경 엔진 및 Tensor 코어

이는 GPU 장치의 별칭이 아닌 다른 가속기입니다. 애플 실리콘에서는 애플 뉴럴 엔진 (ANE) 이 Core ML 를 통해 도달됩니다. NVIDIA 텐서 코어는 CUDA GPU 내부의 단위이며, MLX / cuBLAS 또는 PyTorch 백엔드가 적합한 행렬 커널을 선택합니다. `--device neural`, `--device tensor`, 그리고 범용 NPU 백엔드는 제공되지 않습니다.

<a id="discovery-and-evidence"></a>

## 발견과 증거

`iild-run devices` 보고서는 Core ML 가용성, OS 가 이를 노출할 때 실제 ANE 코어 수, 컴퓨팅 플랜 가용성, 및 CUDA 텐서 코어 자격을 보고합니다. 공개 진입점은 `coreMLCapabilities()` 와 `tensorCoreCapabilities(deviceIndex)` 입니다. MLX 의 기존 공개 능력 구조체와 팩토리 오버로드는 레이아웃/서명을 유지합니다.

ANE 검색은 Apple의 공개를 사용합니다.
[`MLNeuralEngineComputeDevice.totalCoreCount`](https://developer.apple.com/documentation/coreml/mlneuralenginecomputedevice/totalcorecount)는 프로세서 이름에서 추측한 것이 아닙니다. 장치 발견에는 macOS 14 가 필요합니다. 컴퓨팅 플랜 검사에는 macOS 14.4가 필요합니다. 오래된 OS 는 발견을 알 수 없는 것으로 보고하며, 허구의 0코어 가속기가 아닙니다. 구현은 이러한 API 를 노출하는 SDK 를 필요로 합니다. 기본 명시 허용 CPU 예측에는 macOS 13 또는 그 이상이 필요합니다; 그 오래된 배포 타겟은 이 빌드에서 검증되지 않았습니다.

Tensor Core 지원은 CUDA 가용성에서만이 아니라 CUDA 컴퓨팅 기능 및 장치 제품군에서 추론됩니다.

|CUDA 장치|보고된 지원|적격 연산|
|---|---|---|
|미만의 컴퓨팅 기능 7|지원되지 않음|Tensor Core 청구 없음|
|볼타 7.0 / 자비에르 7.2|지원됨| FP16 |
|튜링 7.5 RTX / 테슬라 T4|지원됨| FP16 |
|GTX 16시리즈, 7.5|지원되지 않음|CUDA는 Tensor 코어 없이도 작동함|
|미확인 7.x 제품군|알 수 없음|긍정적인 Tensor Core 주장 없음|
|암페어 이상, 기능 8+|지원됨| FP16, BF16, TF32 |

Python ROCm 장치는 NVIDIA Tensor 코어로 표시되지 않습니다. 네이티브 검색에서는 NVIDIA를 대상으로 하는 MLX CUDA 런타임를 사용합니다. 이 정책은 NVIDIA의 정책을 따릅니다.
[능력 참조](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) 와 [cuBLAS 수학 모드](https://docs.nvidia.com/cuda/cublas/index.html)입니다. 알 수 없는 장치 가족은 고안된 코어 수를 의도적으로 할당하지 않습니다. AMD 라데온은 자체 [LibTorch /ROCm 경로](radeon-rocm.md)를 가집니다. HIP 실행은 NVIDIA TF32 세터에 접근하지 않으며, 명시적인 NVIDIA TF32 스위치는 거부됩니다.

3는 별개로 유지됩니다: 하드웨어가 존재하며, 런타임가 해당 장치에 대해 작업을 예약할 수 있고, 프로파일러가 이를 관찰했습니다. Core ML의
[`MLComputePlan`](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n)는 **예상** 파견 계획입니다. `CoreMLModelInfo.hardwareUsageVerified` 및 Tensor Core `usageVerified` / `usage_verified`는 false로 유지됩니다. 성공적인 산술 또는 선호하는 장치 레이블은 하드웨어 카운터 측정이 아닙니다.

<a id="c-core-ml-component-execution"></a>

## C++ Core ML 구성 요소 실행

애플 빌드는 MLX 에 독립적으로 `IILD_ENABLE_COREML=ON` 를 활성화합니다. Foundation 과 Core ML 는 사내 시스템 프레임워크 의존성이며, 설치된 C++ API 에 Python 인터프리터 또는 Core ML 타입이 나타나지 않습니다. 다른 플랫폼 또는 `IILD_ENABLE_COREML=OFF` 는 가용하지 않는 능력과 예외를 던지는 실행 스텁을 가진 API 를 유지합니다. 메타데이터 전용 빌드는 Core ML, MLX, 및 선택적 LibTorch 백엔드를 비활성화합니다.

```cpp
#include <Compute/CoreMLModel.hpp>

auto component = iild::CoreMLModel::load("/absolute/path/component.mlmodelc");
// 기본값: CPU와 Neural Engine을 허용하고 GPU는 제외한다.
// 계획에 Neural Engine을 우선 사용하는 연산이 최소 하나 있어야 한다.
iild::CoreMLModel::Features inputs;
for (const auto &feature : component.info().inputs) {
    inputs.emplace(feature.name, std::vector<float>(feature.elementCount, 1.0F));
}
auto result = component.predict(inputs); // 1로 채운 값을 애플리케이션 입력으로 바꾼다
```

모델은 항상 호출자에서 제공됩니다. `load()` 는 기존 컴파일된 `.mlmodelc` 디렉토리만 허용하며, 원시 체크포인트나 자동으로 다운로드된 모델을 허용하지 않습니다. 네이티브 로딩은 인증이나 보안 샌드박스가 아니며, 호출자는 컴파일된 아티팩트를 신뢰하고 버전을 지정해야 합니다. `predict()` 는 선언된 형식을 사용하여 행 주열 호스트 부동소수점32 벡터에 정확한 이름의 기능을 허용합니다. 부동소수점16 인터페이스는 Core ML 에 의해 변환됩니다. 그것은 누락된/추가된 기능, 잘못된 길이, 비유한 또는 범위 밖 입력, 그리고 유효하지 않거나 비유한 출력을 거부합니다. 버퍼 복사는 패딩된 가속기 버퍼를 포함한 Core ML 의 실제 스트라이드를 존중합니다. 한 인스턴스에서의 호출은 직렬화되며, 이동된 인스턴스는 명시적으로 실패합니다.

지원되는 인터페이스는 양의, 선언된, 단일 바운드 형식과 선택 부동소수점32/부동소수점16 다차원 배열을 가집니다. 이미지, 정수, 선택, 유연성, 및 undeclared-output 인터페이스는 거부됩니다. 계획 검사에는 `main` ML 프로그램 함수 하나 또는 NeuralNetwork 모델이 지원되며, 파이프라인이나 다중 함수 계획은 지원되지 않습니다. 이들은 구성 제약 사항이며, 임의의 Core ML 패키지 호환성을 약속하는 것이 아닙니다.

`CoreMLOptions.computeUnits=all` 는 CPU / GPU / ANE 를 허용합니다. `cpuOnly` 는 명시적인 진단이며 `requireNeuralEnginePlan=false` 를 요구합니다. 기본값은 CPU 만/ GPU 만 계획이라도 ANE 가 존재하더라도 거부합니다. 이 요구 사항을 비활성화하면 선택된 컴퓨팅 유닛 세트 내에서 ANE 가 아닌 계획을 허용하며, 사용할 수 없는 엔진을 강제하거나 Core ML 의 스케줄러를 우회하지 않습니다. 애플은 노출합니다
[컴퓨팅 유닛 세트](https://developer.apple.com/documentation/coreml/mlcomputeunits)는 공개 ANE 전용 보증이 아닙니다.

```bash
./build/iild-run neural-compute --model /absolute/path/component.mlmodelc
./build/iild-run neural-compute --model /absolute/path/component.mlmodelc --compute-units all
./build/iild-run neural-compute --model /absolute/path/component.mlmodelc \
  --compute-units cpu --allow-cpu-plan
```

이 CLI 는 입력을 1 로 채우고 성공적인 유한한 예측을 확인하며, 이는 하드웨어 진단이며 이미지 생성이나 품질 벤치마크가 아닙니다. `--iterations 1..10000` 는 외부 프로파일링을 위해 예측을 반복합니다. 지속 시간은 검증, 입력 복사, 출력 읽기 반환을 포함하며 엔진 시간만은 아닙니다.

<a id="cuda-matrix-precision"></a>

## CUDA 매트릭스 정밀도

추가 `LinearMathOptions` 과부하는 GPU 중량/입력 산술을 선택합니다.

```cpp
auto layer = iild::LinearLayer::fromSafetensors(
    "/absolute/path/component.safetensors", "weight", "bias",
    iild::ComputeOptions{iild::ComputeDevice::cuda},
    iild::LinearResourceOptions{},
    iild::LinearMathOptions{iild::LinearPrecision::float16});
```

`float32` 는 기본값으로 남아 있습니다. `float16` 와 `bfloat16` 는 자격 있는 저정밀도 CUDA / Metal 수학 연산을 가능하게 합니다; CUDA 는 BF16 가 아메퍼 또는 그 이후의 장치가 감지되었음을 요구합니다. CPU 만 감소 정밀도가 거부되며, 협력 CPU 샤드는 여전히 부동소수점32를 사용합니다. 호스트 API 출력은 부동소수점32로 남아 있습니다. RAM 및 staged-weight 바이트 수치는 실제 GPU 정밀도 ( 2 바이트의 FP16/BF16, 4 의 FP32 ) 를 사용하며, CPU 섹션은 4-바이트 값을 사용합니다. 감소 정밀도는 수치 정확도와 범위를 변경합니다. 변환 또는 산술 중 오버플로우하는 값은 명시적으로 실패하며, 겉보기에 성공한 비유한 결과를 반환하지 않습니다. 텐서 코어 디스패치는 런타임 -선택되며, 모든 형식이나 연산을 위해 강요되지 않습니다. 작고 정렬되지 않거나 행렬이 아닌 작업은 다른 단위를 사용할 수 있습니다. 오래된 텐서 코어 GPU 에서는 네이티브 수학을 위해 FP16 를 명시적으로 선택해야 하며, 변경되지 않은 FP32 기본값은 그곳에서 텐서 코어 사용을 함의하지 않습니다.

```bash
./build/iild-run compute --device cuda --precision fp16
./build/iild-run compute --precision bf16 --cpu-share 0.25 --weight-storage ram
```

네이티브 FP32 CUDA 수학 은 MLX 의 TF32 정책을 유지하며, `MLX_ENABLE_TF32=0` 는 시작 전 요청을 위해 전체 FP32 곱셈 정밀도를 사용합니다. Python 생성은 이제 자격이 있는 NVIDIA 8+ GPU 에서 TF32 를 자동으로 허용합니다. `--no-cuda-tf32` 는 IEEE FP32 행렬/합계곱 수학식을 선택하며, `--cuda-tf32` 는 명시적으로 TF32 자격을 요구합니다. 비 CUDA 장치에 대한 명시적 TF32 플래그는 거부됩니다. FP16/BF16 텐서 코어 커널을 비활성화하지 않습니다. 고정된 PyTorch 런타임 의 새로운 `fp32_precision` API 는 구식 `allow_tf32` 제어 항목과 혼합되지 않고 사용되며; [PyTorch 의 CUDA 주석](https://docs.pytorch.org/docs/main/notes/cuda.html)을 참조하십시오.

이미지 사이드카는 이 정책을 `runtime.hardware.tensor_cores` : 장치 패밀리/기능, 데이터 타입별 자격, 요청된 TF32 모드, 적용된 행렬/합계곱 정밀도, 자동 커널 선택, 그리고 검증되지 않은 명령어 활용에 기록합니다. 기존 모델/VAE / LoRA 검증 및 활성화 순서는 변경되지 않습니다.

<a id="conversion-validation-and-limits"></a>

## 변환, 검증 및 제한

[`reference/coreml/convert_linear.py`](../reference/coreml/convert_linear.py) 는 로컬 `.safetensors` / `.safetensor` 파일에서 정확한 랭크-2 가중치와 선택적 편향을 변환합니다. 채널 우선 `[1,in,1,batch]` 1x1 합계곱은 FP16 에서 선형 구성 요소를 구현하며; 따라서 외부 레이아웃은 `LinearLayer` 의 `[batch,in]` 레이아웃과 다릅니다. 이는 다음을 따릅니다
[Apple의 ANE 변압기 레이아웃 지침](https://machinelearning.apple.com/research/neural-engine-transformers). 변환은 LoRA를 병합하거나 전체 체크포인트를 변환하거나 확산 파이프라인을 조립하지 않습니다. [변환 설정 및 명령](../reference/coreml/README.md)를 참조하세요.

변환기는 원본/아티팩트 SHA-256 해시, 이름이 있는 키, 차원, dtype, 계획, 버전, NumPy float32 검증 기준과 선택적 C++ 검증을 기록한다. `--native-test`가 없으면 검증된 예측이 아니라 변환만 보고한다. 출력은 새 디렉터리 또는 빈 디렉터리여야 하며 `build/` 아래에 있어야 한다. 이전 근거는 보존한다. 추론은 coremltools의 Python 예측 버퍼 브리지가 아니라 C++를 통해 의도적으로 테스트한다. macOS 27 / Python 3.13 / coremltools 9.0에서 해당 브리지의 비동기 버퍼 해제 경로가 충돌하는 것을 관찰하였다. 변환과 계획 검사는 문제가 있는 Python 예측 경로를 사용하지 않는다.

2026-09-03 M1 Max 호스트에서 공개 발견은 16 ANE 코어를 반환했습니다. 실제 SD 1.5 CLIP `fc1` 구성 요소 ( `768 -> 3072` , 배치 128 ) 는 합계를 위해 ANE 를 선호하며, NumPy 에 대해 네이티브 예측을 통과시켰고 최대 절대 오차 `0.002185344696044922` (각 요소 허용 오차 `0.005 + 0.01*abs(expected)` ) 입니다. 증거: `build/reference/neural/sd15-clip-native-final/provenance.json` 입니다. 별도 부동소수점16-I/O 와 작은 CPU -선호 픽스처 연습 변환, 동시성, 유효하지 않은 값, 이동된 객체, 및 기본 ANE 거부. CPU 전용 Core ML 는 다른 FP16 누적 행동을 가지며, 독립적이고 정확한 0-입력/편향 사례는 ANE 비교 허용 오차를 완화하지 않고 해당 진단 경로를 확인합니다.

이 호스트에는 NVIDIA GPU 가 없습니다. CUDA 장치 계열/ TF32 정책 테스트, 실제 PyTorch 정밀도 설정 API 확인 및 실제 Metal FP16/BF16 산술 통과. 이는 물리적 CUDA /Tensor Core 또는 CUDA -빌드 확인이 아닙니다. 속도 향상 또는 활용 비율은 주장되지 않습니다. C++ 확산 파이프라인, PyTorch / Diffusers 의 자동 ANE 사용, 및 Intel/ AMD /Qualcomm NPU 백엔드는 구현되지 않았습니다. RAM 는 여전히 저장소이며 산술 하드웨어가 아닙니다; Core ML 할당은 MLX 의 중량 스테이지 예산에 독립적입니다.
