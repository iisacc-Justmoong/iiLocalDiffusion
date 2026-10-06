<a id="architecture"></a>

# 건축

<a id="purpose"></a>

## 목적

iiLocalDiffusion 는 로컬 생성 모델의 구성 요소를 로드, 표현, 연결 및 실행하는 런타임 입니다. 이는 텐서 연산 구현이 아닙니다. 이 경계는 라이브러리가 모델 및 파이프라인 의미에 집중되도록 유지하면서 저수준 추론 런타임 를 교체할 수 있게 합니다.

<a id="responsibility-boundary"></a>

## 책임 경계

iiLocalDiffusion 소유:

- 모델 패키지 구조 및 메타데이터
- 파이프라인 구성 및 구성요소 수명주기
- 생성 요청 및 검증
- 종자 및 재현성 정책
- 스케줄러 선택 및 잡음 제거 오케스트레이션
- 컨디셔닝 흐름 및 모델 어댑터
- 결과 메타데이터, 출처 및 진단
- 추론 백엔드가 선택되는 경계

추론 백엔드는 다음을 소유합니다.

- 텐서 할당, dtype, 스트라이드, 레이아웃 및 장치 메모리
- 행렬 곱셈, 컨볼루션 및 Attention 프리미티브
- 그래프 실행 및 동기화
- CPU, Metal, CUDA 및 기타 가속기 명령 및 커널

백엔드 텐서 유형은 구현 경계 뒤에 있어야 합니다. iiLocalDiffusion의 공개 요청 또는 결과 유형에 표시되어서는 안 됩니다.

<a id="current-dependency-direction"></a>

## 현재 종속성 방향

[생성 I/O 구성](generation-composition.md) 는 독립적인 설치된 `Generation/GenerationIO.hpp` 호스트 데이터 계약을 추가합니다. 그것은 타입화된 페이로드와 이름 지정된 단계 바인딩을 유효성 검사하고 호출자 제공 실행자를 호출합니다. 그것은 백엔드 텐서 또는 모델 구현을 도입하지 않습니다. Python 구성은 이미 로드된 Diffusers / Transformers 및 토크나이저 호출에 대한 어댑터를 가진 동일한 명시적 표현 공간 경계를 가집니다. 일반 파일 교환은 safetensors 를 재사용하고 저장된 메타데이터를 검증하며, 이 경로들 중 아무런 알림 없이 는 확산 v-예측을 유속과 동일시하거나 토큰 ID 를 로짓과 동일시하지 않습니다.

`reference/generate.py` 는 카탈로그 식별자와 실제 실행을 분리하고 내장 Diffusers 파이프라인 또는 명시적인 로컬 ComfyUI API 워크플로우를 선택합니다. Civitai 스냅샷은 상위 공급 측 모델 계열을 설명하며, 검증된 체크포인트 텐서를 설명하지 않습니다. 프셋 이름은 `PipelinePreset.family` 를 통해 아키텍처 라우팅을 공유하되 인코더, VAE, 예측 및 가이드 요구사항을 유지합니다. 각 백엔드는 입력과 출력을 검증하며 기원을 기록합니다. 이는 C++ 매니페스트 계약을 확장하거나 Python 를 C++ 에 포함시키지 않습니다.

메타데이터 검사와 네이티브 구성 요소 계산은 독립적인 경로입니다.

```text
iild-run -> loadModelManifest
loadModelManifest -> StableDiffusionModelManifest
loadModelManifest -> FluxModelManifest
StableDiffusionModelManifest -> private ModelManifestParser
FluxModelManifest -> private ModelManifestParser
ModelManifestParser -> filesystem and json-c
iild-run compute -> LinearLayer -> private MLX runtime -> Metal/CUDA
iild-run compute -> LinearLayer -> private LibTorch runtime -> AMD HIP/ROCm
LibTorch weight loading -> private safetensors-cpp reader -> owned LibTorch storage
LinearLayer -> ComputeRuntime (explicit device policy)
iild-run neural-compute -> CoreMLModel -> private system Core ML -> CPU/ANE/(optional GPU)
```

Python Diffusers 는 독립적인 참조 오라클입니다. C++ 라이브러리는 Python 를 포함하지도 참조 스크립트를 호출하지도 않습니다. Python 생성 오라클은 명시적으로 선택된 모델/체크포인트 및 VAE 파일을 조립하고 호환되는 ControlNet 하나를 첨부한 후 파이프라인을 검증한 후 선택된 LoRA 하나를 적용할 수 있습니다. `presets.py` 는 계열 및 구성 계약을 소유하고, `model_loading.py` 는 Diffusers 조립을 소유하며, `weight_files.py` 는 로컬 파일 식별자와 정통한 안전 로더 경로를 소유합니다. `controlnet.py` 는 선택 가능한 ControlNet 선택, 준비된 이미지 로딩, 구성 요소 호환성, Diffusers 파이프라인 첨부 및 기원을 소유합니다. 그것은 신경 실행, 단일 파일 변환 및 이미지 리사이징을 기존 런타임 의존성에 위임하며 조건 탐지기를 구현하지 않습니다. [ControlNet 계약](controlnet.md) 은 생성에만 적용됩니다. 선택적 [고용 수정](hires-fix.md) 는 모든 3 패밀리와 그들의 ControlNet 변형에 걸쳐 기본 패스, Pillow RGB 리사이징 및 img2img 정제 패스를 조정합니다. 그것은 선택된 신경 구성 요소를 재사용하며 기존 추론 백엔드와 오프로드 정책을 유지합니다. 프로젝트 코드는 단계 매개 변수, 스케줄러 수명 주기, 출력 예약 및 기원 소유하며 기존 의존성은 이미지 리샘플링, VAE 인코딩, 노이즈 및 신경 실행을 소유합니다. 단계별 일정과 출력 검증은 보편적인 시각적 품질을 주장하지 않고 두 번째 패스를 관찰 가능하게 만듭니다. `hires_options.py` 는 두 번째 단계 인수 해결을 소유하며, `hires.py` 는 이미지 리사이징, 정제 어셈블리 및 단계 검증을 소유합니다. `generation_output.py` 는 최종 및 선택적 기본 출력 경로를 예약합니다. `hires_flux_controlnet.py` 는 Diffusers ' 공개 FLUX img2img 잠재 표현/일정 준비와 기존 FLUX ControlNet 디노이징 루프를 결합합니다. 이는 고정된 상위 공급 측 ControlNet img2img 호출에서 부재된 부정 조건화와 참 CFG 을 보존하며, 트랜스포머 또는 ControlNet 계산을 복사하지 않습니다. 그의 스케줄러 조정은 하나의 순차 호출에 범위가 정해져 있으며 이후 복원됩니다. [모델 입력 계약](model-inputs.md) 는 임베딩된 체크포인트 가중치를 구성원 소스의 보조 구성 요소와 분리합니다. C++ 매니페스트는 가중치 텐서를 로드하거나 어댑터를 검사/실행하지 않습니다. 별도의 네이티브 `LinearLayer` 는 safetensors 파일에서 이름이 지정된 텐서를 선택할 수 있지만 완전한 체크포인트를 조립하거나 어댑터를 적용하지 않습니다. `hardware.py` 는 오라클의 GPU -필수 선택, 하드웨어 사전 검사 , 그리고 실제 실행 장치 검사 ( NVIDIA 텐서 코어 자격 및 TF32 정밀도 정책 포함) 를 소유합니다. 그것은 HIP 의 공유 CUDA 네임스페이스와 NVIDIA 실행을 구별하고, AMD 속성을 기록하며, ROCm 에서 NVIDIA TF32 컨트롤을 적용하지 않습니다. 그것은 생성을 MLX 또는 Core ML 에 위임하지 않습니다. `cpu_conditioning.py` 는 명시적 CPU 프롬프트 인코딩과 조건부 전송을 소유합니다. CPU 인코딩은 LoRA 활성화에 따라오고 배치 전에 수행되며, Diffusers /Accelerate 는 모델/순차적 RAM 오프로드 후크를 소유합니다. 이 CPU 단계는 GPU 소음 제거를 공급하며, 네이티브 동시 선형 분할과 별개입니다.

선택적 [텍스트 인버전](text-embeddings.md) 는 프롬프트 인코딩과 오프로드 배치 전에 학습된 토크나이저 항목과 일치하는 인코더 입력 벡터를 추가합니다. 기존 모델, LoRA , ControlNet 와 Hires Fix 경로와 함께 구성됩니다. 기존 토크나이저/임베딩 테이블 API 는 저장과 실행을 소유하며, 프로젝트 코드는 안전한 입력 선택, 차원, 토크널 충돌 검사, 다중 벡터 프롬프트 처리 및 출처를 소유합니다. `--embeddings` 에서 완료된 프롬프트 텐서는 인코딩을 우회하는 별도의 상호 배타적 입력 경로로 남아있습니다. `text_embedding_options.py` 는 순차 파일/토크널/인코더 선택을 소유하며, `text_embeddings.py` 는 벡터를 검증하고 기존 Diffusers 텍스트 인버전 로더를 호출하며 등록된 ID 와 벡터 값을 확인합니다. 그의 범용 프롬프트 처리는 각 인코더의 다중 벡터 토큰을 한 번씩 확장한 후 호출 이후에 정상 파이프라인 프롬프트 변환을 복원합니다.

헤더와 구현체는 이 워크스페이스의 라이브러리 관례에 따라 `src/` 하에 함께 존재하며, 별도의 소스 `include/` 트리는 사용되지 않습니다. 설치된 소비자는 자신의 접두사 하에 동일한 헤더 경로를 받습니다. 프로젝트가1.0이전 상태라면 이진 및 CMake 패키지 호환성은 동일한 `0.x` 마이너 릴리스 라인 내에서만 보장됩니다.

<a id="current-milestone"></a>

## 현재 이정표

`loadModelManifest()`는 파이프라인 클래스별로 디스패치됩니다. 기존 `StableDiffusionModelManifest::load()` 및 새로운 `FluxModelManifest::load()`는 모델 제품군별 진입점으로 유지됩니다. 그들은 함께 다음을 검증합니다.

- 액세스 가능한 모델 루트 및 유효한 객체 값 `model_index.json`
- 표준 SD v1, SDXL Base 또는 FLUX.1-schnell 구성 요소 세트
- 필수 구성 요소 디렉터리 및 구성 파일
- `stable-diffusion-v1.md`, `stable-diffusion-xl.md` 및 `flux1-schnell.md`에 문서화된 메타데이터 계약
- 각 선언된 텍스트 인코더, 디노이저 및 VAE에 대해 비어 있지 않은 `.safetensors` 경로

safetensors 헤더 또는 텐서 본문을 구문 분석하고, 모델을 인증하고, Stable Diffusion 1.4를 1.5 가중치와 구별하고, 신경 모듈을 인스턴스화하거나 추론을 실행하지 않습니다. CLI 문구는 의도적으로 `valid-metadata`를 보고하고 구성 요소가 로드된 것으로 보고하지 않습니다.

인스펙터는 패키지 호환성 진단 도구이며 보안 샌드박스가 아닙니다. Hugging Face 스냅샷이 콘텐츠 주소 지정 가능한 blob 캐시로 링크되므로 인스펙터는 정규 파일 심볼릭 링크를 따르며, 로컬 패키지가 동시적으로 변형되지 않는다고 가정합니다. 런타임 코드는 텐서를 로드하기 전에 별도로 리비전을 인증하고 더 강력한 파일 열기 정책을 적용해야 합니다.

`safety_checker`와 `feature_extractor`는 고정된 SD v1 참조에 존재하지만 코어5 계약의 범위 밖에 있다. SDXL Base와 FLUX.1-schnell은 안전 검사기를 제공하지 않는다. Python 검증 기준은 검사기와 워터마커의 존재 여부를 기록한다. 이미지 생성이 제품 기능이 되기 전에도 제품 안전 정책은 패키지 호환성과 분리하여 유지해야 한다.

<a id="backend-decision"></a>

## 백엔드 결정

MLX C++ 0.32.2는 이제 기본 활성화된 비공개 프로덕션 의존성이다. 텐서 저장소, 장치 전송, 행렬 곱, 편향 덧셈과 동기화를 담당한다. `LinearLayer`는 최초로 구현된 신경망 구성 요소이며 실제 SD 1.5 CLIP 가중치 및 PyTorch 검증 기준과 수치적으로 비교하였다. Apple Silicon에서는 Metal을 선택하며 CUDA는 Linux/NVIDIA 백엔드이다. 자동 런타임 선택에는 GPU가 필요하다. 명시적 CPU 연산과 선택적으로 활성화하는 CPU/GPU 협력이 가능하다. 사용할 수 없거나 실패한 GPU 작업을 CPU에서 다시 시도하지 않는다. 정확한 범위는 [hardware-compute.md](hardware-compute.md)를 참고한다.

구현된 AMD 요구사항은 동일한 `LinearLayer` API 뒤에 선택적인 LibTorch HIP 백엔드를 추가합니다. 작은 사설 `LinearBackend` 컴포넌트 경계는 구체적인 2 런타임을 분리하며, 이는 텐서/그래프 레지스트리에 설치되거나 일반화되지 않습니다. LibTorch 는 AMD 텐서, 데이터형 변환, 행렬 산술, CPU / GPU 전송 및 스케줄링을 소유합니다. 고정된 safetensors -cpp 리더는 파일 메타데이터/바이트를 공급하며, 프로젝트 코드는 복사될 선택된 가중치를 소유한 런타임 저장소에 대한 형식/오프셋 계약을 제한합니다. `rocmCapabilities()` 와 가산 선택 오버로드는 기존 기능 레이아웃과 열거형 값을 변경하지 않습니다. MLX 는 사용 가능한 경우 명시적인 CPU 호출에 여전히 선호되며, 그렇지 않으면 활성화된 LibTorch 가 CPU 를 제공합니다. Radeon 가용성은 결코 NVIDIA 텐서 코어 또는 AMD NPU 지원을 함의하지 않습니다. 배포 및 검증 경계에 대해서는 [radeon-rocm.md](radeon-rocm.md) 를 참조하십시오.

공개 구성 요소는 형태와 일반 호스트 값 또는 파일 경로와 정확한 가중치 키를 받는다. MLX 배열과 스트림은 PIMPL 내부에 유지한다. 호출 사이에 가중치를 보존하며 MLX의 전역 기본 장치를 바꾸지 않고 인스턴스 소유 스트림을 사용한다. `LinearResourceOptions`는 출력 특성을 CPU 및 GPU 작업자에 나누고 가중치를 RAM에 유지하며 순전파 반복마다 스테이징하는 GPU 가중치 블록의 크기를 제한할 수 있다. 텐서 저장소, 스케줄링, 복사와 수학 연산은 여전히 MLX가 담당한다. 리소스 수치는 RSS/VRAM 측정값이 아니라 논리적 수치이다. 파일 I/O와 결과 읽기는 명시적 호스트 경계이다. 사용자 정의 텐서 타입이나 커널을 도입하지 않는다. `LinearMathOptions`는 추가로 GPU FP32/FP16/BF16를 선택하면서 FP32 CPU 샤드 연산과 일반 float32 호스트 출력을 유지한다. CUDA 커널 선택은 사용자 정의 Tensor Core 커널이 아니라 MLX/cuBLAS의 책임으로 유지한다.

`CoreMLModel` 은 기본값으로 Apple 플랫폼에서 활성화되는 독립적인 네이티브 컴포넌트 런타임 입니다. 그 C++ 의 PIMPL 는 Objective-C Core ML 모델을 소유하며, 호출자는 컴파일된 `.mlmodelc` 경로와 이름이 지정된 호스트 기능 벡터를 제공합니다. Core ML 는 스토리지, 캐스팅, 합집합 및 CPU / GPU / ANE 스케줄링을 소유합니다. 플랜 인스펙션은 진단용이며, 새로운 그래프 실행기나 하드웨어 카운터가 아닙니다. 기본적으로 최소 한 가지 작업이 ANE 을 선호해야 하며, CPU / ANE 만 허용됩니다. 오프라인 coremltools 변환기는 이름이 지정된 선형 구성 요소와 NumPy 오라클을 생성하며, 라이브러리에 의해 결코 호출되지 않습니다. 이는 런타임 LoRA 병합, 자동 체크포인트 변환, 또는 전체 C++ 확산 파이프라인을 추가하지 않습니다. 보라
[neural-accelerators.md](neural-accelerators.md).

전체 C++ 확산 생성은 아직 구현되지 않았습니다.

`stable-diffusion.cpp`는 비교 구현으로 유용하지만 이미 대부분의 토크나이저, 스케줄러, 파이프라인 및 모델 의미 체계를 보유하고 있습니다. 이를 기본 종속성으로 만들면 iiLocalDiffusion가 래퍼로 바뀌고 별도의 아키텍처 결정이 필요합니다.

구현된 구성 요소에는 추측성 일반 텐서, 플러그인 레지스트리 또는 그래프 추상화가 아닌 작은 컴퓨팅 옵션/기능 유형 및 전용 선형 구성 요소 백엔드 경계만 필요합니다. 추가 런타임 및 모델 구현은 별도의 아키텍처 결정으로 유지됩니다.

<a id="explicit-non-goals-for-the-manifest-milestone"></a>

## 매니페스트 마일스톤에 대한 명시적인 비목표

- GUI, Qt, QML, LIMBO 또는 Dreamscapes 통합
- C++의 이미지 생성
- 모델 다운로드 UI 또는 원격 추론
- 사용자 정의 텐서, 커널 또는 그래프 실행기
- LoRA 또는 ControlNet 패키지 검사 또는 C++에서 실행
- 훈련, 외부 이미지2이미지 입력, 인페인팅, 멀티- ControlNet, SDXL 정제기, FLUX.1 -dev, 또는 임의의 FLUX 파생어
- `.ckpt` , 피클 기반 `.bin` , GGUF 또는 임의의 체크포인트 호환성; 전체 체크포인트 파이프라인 조립은 독립적인 Python 생성 오라클로 제한되며, 네이티브 safetensors 로딩은 하나의 선형 구성 요소를 선택합니다.
- `.iicharacter` 지원

이는 영구적인 제품 결정이 아닌 범위 제외입니다.
