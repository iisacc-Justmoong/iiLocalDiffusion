<a id="dependency-decisions"></a>

# 종속성 결정

<a id="model-merging"></a>

## 모델 병합

모델 병합은 설치된 PyTorch(BSD-3-Clause)와 고정한 `safetensors==0.8.0`(Apache-2.0)를 재사용하며, 추가 패키지·네이티브 의존성·모델 다운로드는 없다. 설치 환경에서 Torch 2.13.0와 safetensors 0.8.0를 확인했다. 유지보수되는 상위 공급 측 텐서 연산, 지연 읽기 모듈과 공식 직렬화기가 수치 연산과 파일 바이트를 처리한다. SDK 코드는 계수/기본값 정책·호환성·출처·출력 게시를 담당한다. 레거시 체크포인트는 기존 조건부 weights-only 변환기를 재사용한다. 이 방식은 별도의 WebUI/모델 병합 스택이나 자체 텐서/파일 구현을 추가하지 않도록 한다. 상위 공급 측 [safetensors Torch API](https://huggingface.co/docs/safetensors/api/torch)를 참고한다,
[지연 로딩 API](https://huggingface.co/docs/safetensors/index) 및
[라이센스](https://github.com/safetensors/safetensors/blob/main/LICENSE).

오프라인 LoRA 융합은 동일한 Torch 행렬/합동 텐서 연산을 재사용합니다. 기존 고정 Apache-2.0 Diffusers 0.40.0 는 SD LDM / OpenCLIP 이름 매핑을 공급하며, PEFT 0.20.0 ( Apache-2.0 )은 검증에서 독립적인 융합/순전파 오라클을 제공합니다. 둘 다 이미 유지되는 생성 의존성이며, WebUI , 어댑터 프레임워크, 모델 다운로드 또는 추가 런타임 패키지는 추가되지 않습니다. 직접 체크포인트 산술은 Diffusers 나 PEFT 를 가져오지 않습니다. SDK 코드는 재료 선택, 대상 해결, 계수 정책 및 검증만 담당합니다. 소스:
[Diffusers LoRA 병합](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras),
[PEFT LoRA 구성](https://huggingface.co/docs/peft/main/package_reference/lora).

기존 번들 SDXL 라이선스는 패키지 검증 중 고정 상위 공급 측 리비전에 대해 또한 확인되었습니다. 배포된 파일은 하나의 마지막 공백과 마지막 빈 줄의 제거로만 다릅니다. 구성 매니페스트는 이제 배포된 14,107-바이트 파일의 해시를 기록하며 이 공백 정규화를 표시합니다. 라이선스 문구는 변경되지 않았습니다. 소스:
[고정된 SDXL 라이센스](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/462165984030d82259a11f4367a4eed129e94a7b/LICENSE.md).

<a id="interpolator-animation"></a>

## 보간기 애니메이션

인터폴레이터는 고정 Diffusers / PyTorch 를 재사용하며, 기존 텍스트 인코딩, 파이프라인, 어댑터 및 잠재 표현 패킹 API 를 포함합니다. 텐서 인터폴레이션은 PyTorch 연산을 사용합니다. PNG 와 비디오 출력은 Pillow 와 아래에 설명된 공유 외부 FFmpeg /FFprobe 레이어를 재사용합니다. 새로운 패키지, 모델 다운로드 또는 네이티브 의존성이 없습니다. 기존 Diffusers 는 Apache-2.0 와 PyTorch   BSD-3 -절이며, 고정된 버전과 배포된 공지사항은 계속 유효합니다.

공식 [DiffusionBee 인터폴레이터 소스](https://github.com/divamgupta/diffusionbee-stable-diffusion-ui/blob/master/backends/stable_diffusion/applets/frame_interpolator.py) 는 프롬프트/시드 인터폴레이션을 정의합니다. 그 적용/플러그인/모델 컨테이너 결합은 이 SDK 의 로더 및 실행 정책을 중복하게 하므로, 소스는 복사되거나 가져오지지 않습니다. 작은 2-엔드포인트 조정기는 도메인 코드입니다. 유지되는 수치적, 신경 추론 및 미디어 라이브러리가 중량 작업을 수행합니다. 이 독립형 프롬프트/시드 모드는 프레임 삽입 모델을 필요로 하지 않습니다. 포스트 LTX 비디오는 아래 별도의 프레임 인터폴레이션 결정을 사용합니다.

<a id="post-ltx-frame-interpolator"></a>

## Post-LTX 프레임 보간기

두 번째 비디오 단계는 설치된 FFmpeg   `minterpolate` 필터에 운동 보상을 재사용하며, 타임스탬프/패딩 필터와 Pillow   PNG 확인을 포함합니다. 추론 패키지나 두 번째 모델이 도입되지 않습니다. 유지되는 상위 공급 측 필터는 운동 추정을 공급하며, SDK 코드만 소스 위치를 계획하고 앵커/컷을 보존하며 아티팩트를 확인/발행합니다. 이는 기본 인터폴레이션을 위해 두 번째 신경 런타임 또는 가중치 분포 요구사항을 추가하는 것을 피합니다. 기존 libx264- 활성화된 외부 FFmpeg 빌드는 GPL 로 유지되며 번들되지 않습니다. 필터 가용성과 실제 실행 가능 버전은 로컬에서 확인됩니다. 소스: [FFmpeg 필터 문서](https://ffmpeg.org/ffmpeg-filters.html#minterpolate),
[FFmpeg 라이센스](https://ffmpeg.org/legal.html).

<a id="deforum-2d-animation"></a>

## 디포럼 2D 애니메이션

애니메이션 런너는 고정된 Diffusers 0.40.0 ( Apache-2.0 ), PyTorch, NumPy 및 Pillow 를 추론, 수치 연산 및 PNG 에 재사용합니다. 선택된 `requirements-deforum.txt` 는 유지되는 아핀 카메라 변환과 명시적 엣지 모드를 위해 `opencv-python-headless==4.13.0.92` 를 고정합니다. 검토된 macOS arm64 휠은 약 46 MB 이며 이미 존재하는 NumPy 의존성만 추가합니다. 헤드리스 빌드는 Qt/X11 GUI 의존성을 피합니다. OpenCV 는 Apache-2.0 입니다; 그 휠 패키징과 포함된 제 3 자 구성 요소는 자체 공지를 포함합니다. 소스:
[패키지 릴리스](https://pypi.org/project/opencv-python-headless/4.13.0.92/),
[상위 공급 측 아핀 API](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html).

유지되는 FFmpeg /FFprobe 명령 줄 도구는 H.264 인코딩 및 디코딩된 스트림 확인을 처리합니다. 이들은 외부 실행 파일이며 번들되지 않습니다; libx264- 활성화된 FFmpeg 빌드는 GPL 입니다. 선택된 실행 파일/버전은 모든 애니메이션 보고서에 기록됩니다. [FFmpeg 라이선스](https://ffmpeg.org/legal.html) 및 [이미지 시퀀스 입력](https://ffmpeg.org/ffmpeg-formats.html#image2-1)을 참조하십시오.

상위 공급 측 Deforum 노트북 ( MIT 및 구성 요소 공지) 과 AUTOMATIC1111 확장 ( AGPL-3.0 ) 이 검토되었습니다. 그들의 노트북/ WebUI 실행 결합과 추가 모델/ 런타임 스택은 이 저장소의 기존 로딩, 하드웨어 및 어댑터 정책을 중복시킵니다. 상위 공급 측 소스는 벤더링되거나 가져오지지 않습니다. 작은 일정/피드백 조정기는 애플리케이션 도메인 코드이며, 이미지 변환, 신경망 추론 및 인코딩은 유지 관리된 라이브러리를 사용합니다. 참조: [노트북](https://github.com/deforum/deforum-stable-diffusion),
[확장](https://github.com/deforum/sd-webui-deforum),
[애니메이션 의미론](https://github.com/deforum/sd-webui-deforum/wiki/Animation-Settings).

<a id="broad-local-model-generation"></a>

## 광범위한 로컬 모델 생성

The [3 모델 소스 입력](model-sources.md) 는 추가 패키지 의존성을 포함하지 않습니다. 직접 원격 엔드포인트는 Python 의 표준 HTTP 클라이언트를 사용합니다. 클라우드 모델 ID 는 기존 고정된 Apache-2.0 `huggingface-hub==1.29.0` InferenceClient 및 그 유지 관리된 제공자 어댑터를 사용합니다. Pillow / FFmpeg 는 반환된 미디어를 검증하고 사후 처리합니다. 제공자는 별도의 서비스로 남아 있으며, 모델-ID 파싱 및 모의 분배 테스트는 라이브 가용성 또는 완료된 유료 추론을 설정하지 않습니다.

Civitai 호환성 작업은 추가 샘플링 알고리즘을 구현하는 대신 고정된 Diffusers 0.40.0 (Apache-2.0) 를 재사용합니다. 일반적인 런너는 내장 파이프라인만 로드하고 커스텀 원격 코드를 비활성화합니다. 선택적 아키텍처 의존성은 명시적인 오류로 남아 있습니다. 모델 라이선스는 별도입니다.

SentencePiece 0.2.2 (Apache-2.0) 는 Kolors/ ChatGLM 토크나이징 및 기타 SentencePiece 기반 인코더에 대해 고정되어 있습니다. 공식 PyPI 릴리스는 CPython 3.14 애플 실리콘 휠 (약 1.35 MB) 을 제공하고 필수 Python 패키지 의존성을 추가하지 않습니다. 이는 모델별 토크나이징을 구현하는 대신 구글의 유지 관리된 토크나이저 구현을 재사용합니다. 로컬 감사에서 Diffusers 가 다른 경우 가짜 Kolors 클래스를 노출한다는 것이 감지되었습니다. 소스:
[공식 패키지](https://pypi.org/project/sentencepiece/0.2.2/),
[upstream](https://github.com/google/sentencepiece)및 설치된 Diffusers `is_sentencepiece_available()` 종속성 게이트.

ComfyUI는 문서화된 HTTP API를 통해 접근하는 선택적이며 별도로 설치되는 로컬 프로세스이다. 유지보수되는 네이티브/사용자 정의 노드 생태계는 고정된 Diffusers에 없는 모델 아키텍처와 가중치 배치를 제공한다. ComfyUI는 GPL-3.0이며 사용자 정의 노드와 가중치는 별도 라이선스를 가진다. 이 변경으로 ComfyUI 코드, 노드 또는 가중치를 저장소에 포함하지 않는다. 표준 라이브러리 기반 HTTP 클라이언트는 Python 패키지 의존성을 추가하지 않는다. 이 프로세스 경계는 상당한 규모의 Torch/노드 환경을 C++와 고정된 Diffusers 환경 밖에 유지한다. 다음을 참고한다:
[워크플로 설정 및 확인](comfyui-generation.md).

선택적인 `reference/setup_comfyui.py` 설치기는 이제 ComfyUI 소스와 ComfyUI - GGUF 확장을 별도의 `build/reference/comfyui-venv` 에 고정하여 설치된 패키지 잠금을 유지합니다. 관리되는 로컬 이미지 백엔드는 호출마다 이 엔진을 시작하고 중지합니다. ComfyUI - GGUF 는 Apache-2.0 이며 유지되는 GGUF 리더/양자화 로더를 재사용하여 iiLocalDiffusion 에 텐서 디코더를 추가하지 않습니다. [설치 및 정확한 버전](local-image-generation.md)를 참조하십시오.

Kolors 원시 체크포인트 대안도 검토되었습니다. GPL-3.0
[ComfyUI -Kolors- MZ 소스](https://github.com/MinusZoneAI/ComfyUI-Kolors-MZ/tree/43ec2701a1390259a17ef3bea6244a3134aa5153) 는 체크포인트/ UNet 로더와 ChatGLM 조건부 경로를 포함하지만, 모델 후크는 2024에 기원하며, 최신 검토된 변경사항은 3 월 2025 레지스트리 업데이트이며, 의존성 매니페스트가 없습니다. 4비트 인코더 코드 또한 암시적인 패키지 설치 경로를 포함합니다. 이 2026 엔진과 호환되도록 설치되거나 광고된 바 없습니다. Apache-2.0
[KwaiKolorsWrapper](https://github.com/kijai/ComfyUI-KwaiKolorsWrapper/tree/6fc1cd9d20bb7537facf180e5494b486b9710e24)는 Diffusers 디렉터리를 사용하며 원시 체크포인트 간격을 좁히지 않습니다. 전체 Kolors Diffusers 패키지는 기존 일반 로컬 실행 경로를 유지합니다.

<a id="current-production-dependencies"></a>

## 현재 생산 종속성

### JSON-C

- 필수 버전: 0.18 이상
- 이 작업 공간에서 확인된 버전: 0.19
- 라이센스: MIT
- 연결: 비공개 구현 종속성; pkg-config 대체 경로와 함께 선호되는 CMake 패키지
- 목적: Diffusers JSON 메타데이터의 엄격한 구문 분석

C++ 표준 라이브러리에는 JSON 파서가 없습니다. 유지된 것을 사용하여
[`json-c`](https://github.com/json-c/json-c) 구현은 부분 파서를 생성하지 않고 JSON 타입을 공개 API 에서 제외시킵니다. 0.19 버전은 2026 년에 출시되었으며 현재 문서화된 릴리스입니다. 의존성은 추론 런타임 에 비해 작으며, 매니페스트 인터페이스를 변경하지 않고 대체할 수 있습니다.

<a id="mlx-native-computation"></a>

### MLX 네이티브 계산

MLX 0.32.2 는 기본으로 활성화되는 사내 공유 라이브러리 의존성입니다. 소스는 커밋 `1f8e74e3f12f31365464a6867c6579f0e9b29d85` 와 아카이브 SHA-256 `cb988a5bdc38c798918d042b9b1c6edda3ccc5f23a2155138d3aa5c1b2acc301` 에 고정되어 있습니다. 검토된 C++ API 는 safetensors 읽기, 장치 발견, 스트림, 행렬 연산 및 동기화를 포함하여 Python 를 임베딩하지 않고 텐서를 공급합니다. 이는 json-c 보다 상당히 크지만 사용자 정의 GPU 커널이나 텐서 시스템을 유지하지 않습니다. 공개 헤더는 MLX 타입을 노출하지 않습니다.

Metal 는 애플 실리콘에서 빌드되며, CUDA 소스 구성은 Linux / NVIDIA 를 대상으로 합니다. MLX 의 명시적 CPU 백엔드는 진단, 호스트 리드백, 그리고 옵트인 동시 CPU / GPU 선형 분할을 지원하며, 자동 선택은 여전히 GPU 를 필요로 합니다. RAM 스테이지는 MLX 복사 및 스트림을 사용하며, 표준 C++ 스레드/퓨처 조정과 추가 런타임 의존성이 없습니다. 첫 번째 네이티브 컴포넌트는 `LinearLayer` 로, 실제 SD 1.5 CLIP 레이어와 PyTorch 에 대해 검증되었습니다. 전체 확산 파이프라인 어셈블리는 독립적인 Python 오라클에 속합니다. [hardware-compute.md](hardware-compute.md)를 참조하세요.

`cmake/IildMlx.cmake` 핀은 버전 및 SHA-256별로 종속성을 가져왔습니다.

|의존성|버전|라이센스|사용|
|---|---|---|---|
|MLX, 번들 JACCL 코드 포함| 0.32.2 | MIT |네이티브 계산/런타임|
|애플 메탈-CPP| 26 | Apache-2.0|Metal   C++  바인딩,  Metal  빌드만|
|nlohmann/json| 3.11.3 | MIT |프라이빗 MLX safetensors 메타데이터 및 번들 코드|
|fmt| 12.1.0 | MIT |비공개 MLX 형식 지정|
| NVIDIA CCCL | 3.1.3 |Apache-2.0(상위 공급 측 예외 포함)|CUDA 장치/JIT 헤더|
| NVIDIA NVTX | 3.1.1 |Apache-2.0(LLVM 예외 포함)|CUDA 프로파일링 주석|
|NVIDIA cuDNN 프런트엔드| 1.16.0 | MIT |CUDA 신경 연산 설명자|
|NVIDIA CUTLASS 헤더| 4.4.2 |BSD-3-Clause|CUDA 매트릭스 커널/JIT 헤더|

사용되지 않는 백엔드 의존성은 선언되지만 가져오지 않습니다. MLX   상위 공급 측 의 테스트, 예제, 벤치마크,  Python  바인딩, 및  GGUF  로딩은 비활성화됩니다. Metal   JIT 는 계속 활성화됩니다. 공유  런타임 와  `mlx.metallib` 는 라이브러리와 함께 설치되며,  CUDA 빌드 또한 필요한  JIT 헤더를 설치합니다. 완전한  상위 공급 측 통지는  `share/licenses/iiLocalDiffusion/` 하위에 설치됩니다. CCCL 헤더 릴리스는 전체 라이선스 텍스트를 생략하므로, 정확한 버전의 문서가 고정된  SHA-256 와 함께 별도로 가져옵니다. CUTLASS 의  Python   DSL 는 여기에서 빌드되거나 사용되거나 배포되지 않습니다.

Metal 컴파일러/프레임워크 및  CUDA 툴킷, 드라이버,  cuDNN 라이브러리, 및  BLAS / LAPACK 는 계속 플랫폼 의존성으로 남아있습니다. 프로젝트는 허용적인 소스 헤더 라이선스가  NVIDIA 의 바이너리 SDK 나 모델 가중치를 포함한다고 암시하지 않습니다. 물리적  Metal 작동 및 이동은  M1 Max 호스트에서 확인되었으며,  CUDA 컴파일 및 실행은  NVIDIA 하드웨어에서 실행되지 않았습니다.

<a id="apple-core-ml-components"></a>

### 애플 Core ML 부품

Apple 는 추가로 시스템 Foundation 과 Core ML 프레임워크를 사내 구현 의존성으로 구축하며, `IILD_ENABLE_COREML` 에 의해 독립적으로 제어됩니다. 공개 C++ 헤더에는 Objective-C/프레임워크 타입이 포함되어 있지 않습니다. 공개 컴퓨팅 장치 및 컴퓨팅 계획 API 는 ANE 발견과 예상 배치 기능을 제공하며, 사내 `_ANE` API, 커스텀 신경망 커널, 또는 제 3 자 전체 확산 런타임 를 사용하지 않습니다. 검토된 SDK 는 macOS 14.4 API 를 노출합니다. 프레임워크는 Apple 의 SDK 조항 하에 운영 시스템 의존성으로 유지되며 설치에 복사되지 않습니다.

오프라인 변환의 경우에만 `reference/coreml/requirements.txt`는 coremltools 9.0(BSD-3-Clause), NumPy 2.3.5(BSD-3-Clause), safetensors를 고정합니다. 0.6.2(Apache-2.0) 및 ml_dtypes 0.5.3(Apache-2.0). 애플의
[coremltools](https://github.com/apple/coremltools) 는 유지 관리되며 변환/컴파일 및 계획 API 를 제공합니다. 안정적인 9.0 가 검토된 9.1 개발 릴리스보다 선택되었습니다. 그것의 CPython 3.13 macOS 휠과 작은 변환 전용 의존성 집합은 Torch / TensorFlow 를 이 환경에 추가하거나 Python 3.14 Diffusers 설치를 변경하는 것을 피합니다. 별도의 환경은 `build/reference/coreml-venv/` 하에 유지되며, NumPy 도 C++ 런타임 의존성이 아닙니다. 예측과 수치 검증은 C++ Core ML 브릿지를 사용하며, 독립적인 NumPy 오라클을 사용하여 Python 예측 버퍼 브릿지를 대신합니다. [변환 노트](../reference/coreml/README.md) 와 [하드웨어 제한 사항](neural-accelerators.md)을 참조하십시오.

Tensor 코어 감지 및 정밀도 선택은 MLX/cuBLAS 및 이미 고정된 PyTorch API를 재사용합니다. 추가 NVIDIA SDK 또는 사용자 정의 CUDA 코드는 도입되지 않습니다.

<a id="optional-amd-rocm--libtorch"></a>

### 옵션 AMD ROCm / LibTorch

`IILD_ENABLE_LIBTORCH=ON` 는 외부 설치된 LibTorch 2.x SDK (>= 2.9) 를 사용하며, 2.13.0 는 현재 호스트에서 컴파일되고 실행되었습니다. PyTorch 는 적극적으로 유지 관리되며, BSD 스타일의 라이선스를 사용하며 번들된 제 3 자 공지사항을 포함하고 있으며 이미 독립적인 오라클의 텐서 연산을 제공합니다. 그의 C++ API 는 커스텀 HIP 텐서 할당자 또는 행렬 커널을 도입하는 것을 피합니다. 의존성은 크기가 크므로 선택 사항이며 자동으로 다운로드되거나 번들되지 않습니다. 실제 Radeon 실행은 일치하는 벤더 HIP 빌드, GPU / OS 및 AMD 드라이버가 지원되며 CPU / NVIDIA 배포판은 ROCm 으로 보고되지 않습니다. 구식 C++ ABI = 0 배포판은 거부됩니다.

브릿지는 `at::Context::hasROCM()` 와 Torch 의 GPU 카운트를 사용하며 HIP PyTorch 에 의해 사용되는 CUDA 네임스페이스 ATen 연산을 사용합니다. 공개되는 것은 컴포넌트 수준의 호스트 값과 메타데이터만이며 Torch 타입이나 Python 인터프리터는 노출되지 않습니다. 설치 후에도 외부 SDK 라이브러리 경로가 필요하며 iiLocalDiffusion 를 이동하면 LibTorch /ROCm 이 이동하지 않습니다. 커널 라이브러리, 드라이버 및 SDK 컴포넌트는 벤더 라이선스와 배포 요구사항을 유지합니다. [상위 공급 측 C++ 설정](https://docs.pytorch.org/cppdocs/installing.html) 와 [HIP 의미론](https://docs.pytorch.org/docs/stable/notes/hip.html) 이 검토되었습니다.

MLX는 필수 AMD 백엔드를 제공하지 않으며 LibTorch에는 이에 상응하는 네이티브 safetensors 로더가 없습니다. 따라서 선택적 경로는 다음의 작고 종속성이 없는 C++ 판독기를 사용합니다.
[safetensors-cpp](https://github.com/syoyo/safetensors-cpp)는 커밋 `af90b6c3006cdcecf8b7d7254f5f32d301728acc`(2025-12-27), 아카이브 SHA-256 `f978132be070d6e0ae0be097c6cd5b65edeedf19f78c57158b2c43ffa412323d`에 고정되어 있다. PyTorch보다 작은 커뮤니티 프로젝트이다. 문서화된 불완전한 형태 검증은 전체 페이로드 범위 확인과 겹치는 범위 거부를 포함한 형태 곱 및 바이트 범위 검사에 한계가 설정된을 적용하고 잘못된 파일의 회귀 테스트를 추가하여 보완한다. 부동소수점 변환 도우미는 사용하지 않는다. LibTorch가 FP16/BF16 변환을 담당한다. 보안 샌드박스는 아니다. 리더는 MIT이며 내장 고지는 MIT에 따른 minijson/nlohmann-json, Grisu2, 메모리 매핑과 AMD 파생 코드 및 CC0에 따른 사용하지 않는 FP16 도우미를 포함한다. 고정된 전체 헤더와 라이선스는 네이티브 설치에 동봉한다. 상위 공급 측 README는 과거 Apache-2.0 파싱 코드도 나열하므로 고정 버전을 바꿀 때 상위 공급 측 고지를 유지한다. [Radeon 설정](radeon-rocm.md)을 참고한다.

<a id="reference-only-python-dependencies"></a>

## 참조 전용 Python 종속성

`reference/diffusers/requirements.txt` 직접 오라클 의존성을 유지합니다. Apache-2.0 또는 이와 유사하게 허용적인 라이브러리이지만, iiLocalDiffusion 에 링크되거나 호출되지 않습니다. 설치는 옵트인이며 `reference/diffusers/.venv/` 디렉토리에 계속 유지됩니다. Torch 에 있는 변경되지 않은 미삭제 핀이 이제 `requirements-common.txt` 에 있습니다. `requirements-rocm.txt` 에는  macOS   Torch 버전을  AMD 의 벤더 휠에 강제로 적용하지 않는 핀들을 포함합니다. 일치하는 HIP 휠을 먼저 설치한 후 Python 의존성을 해결한 후 다시 유효성을 검사하세요; 런타임 프로브는 모델 로딩 전에 호환되지 않는 CPU / NVIDIA 교체본을 거부합니다.

이 검증된 환경에 대한 수정된 Python 버전은 macOS arm64의 CPython 3.14입니다. 패키지 버전과 하드웨어는 생성된 모든 참조 이미지 옆에 기록됩니다.

오라클은 또한 `huggingface_hub`가 참조 모델의 대용량 파일을 전송하는 데 사용하는 Hugging Face Xet 1.6.0(Apache-2.0)를 고정합니다. 해당 청크 캐시는 `build/reference/huggingface-xet`로 리디렉션되므로 모델 다운로드 시 시스템 볼륨 캐시를 사용하지 않습니다.

PEFT 0.20.0(Apache-2.0)는 Diffusers를 통한 런타임 LoRA 어댑터 주입 및 활성화를 위해 고정되어 있다. Python 검증 기준에만 필요하며 C++ 라이브러리에 링크하지 않는다. 모델/VAE 단일 파일 로딩은 이미 고정된 Diffusers 및 safetensors 패키지를 사용하며 추가 제3자 런타임 의존성을 도입하지 않는다. Diffusers는 체크포인트 배치 변환과 신경망 구성 요소 생성을 담당하고 프로젝트 코드는 인수 해석, 구성, 파일 식별과 출처를 담당한다.

선택적 ControlNet  생성은 Diffusers   0.40.0  ( Apache-2.0 ) 과 기존 PyTorch , Accelerate,  safetensors ,  huggingface_hub , 및 Pillow  핀을 재사용합니다. 유지되는 상위 공급 측   ControlNet  모델/파이프라인 구현은 신경 실행 및 이미지 리사이징을 담당하며, Pillow 는 정적 이미지 디코딩, EXIF  방향, 및 RGB  변환을 처리합니다. 프로젝트 코드는 명시적 선택, 패밀리 호환성, 구성, 해시, 및 기원을 담당합니다. 네이티브 단일 파일은 임시 Diffusers  컴포넌트 패키지로 준비되며, 지원되는 원본 SD / SDXL  파일은 그 단일 파일 변환기를 사용합니다. 이는 런타임  런타임 의존성, C++  바인딩, 또는 독립적으로 유지되는 신경 구현을 추가하지 않습니다. 자동 조건 감지자, OpenCV , 또는 깊이/포즈 모델은 도입되지 않으며, 호출자가 준비된 이미지를 제공합니다. [ControlNet  입력](controlnet.md)을 참조하세요.

선택적 Hires Fix는 동일한 유지되는 Diffusers , PyTorch , Accelerate 및 Pillow  의존성을 사용합니다. Pillow 는 인접한, 양선형, 삼각형 및 란코즈 RGB 크기 조정 기능을 제공합니다; Diffusers 는 선택된 기본 모델로 이미지 간 VAE 준비, 일정 및 노이즈 제거를 수행하며, VAE , LoRA 및 선택적 ControlNet 구성 요소를 포함합니다. 프로젝트 코드는 2 단계를 조정하며 그 출처를 기록합니다. 학습된 업스케일러 패키지, 추가 모델 다운로드, 텐서 커널 또는 C++ 바인딩은 도입되지 않습니다. 기존 의존성 라이선스와 가중치별 약관은 계속 적용됩니다. 보라
[고용 수정](hires-fix.md).

고정된 FLUX ControlNet img2img 호출은 기존 생성 인터페이스에서 지원되는 부정 조건화 인수가 누락되어 있습니다. 범위 호환성 어댑터는 상위 공급 측 FLUX img2img 잠재 표현/일정 보조 기능을 사용하여 상위 공급 측 ControlNet 노이즈 제거 루프와 연결되며, 실제 CFG 와 부정 임베딩을 유지합니다. 그것은 중복 신경 루프를 유지하지 않으며, 이 API  호환성 지점은 회귀  업그레이드 시 Diffusers  검사가 필요합니다.

텍스트 인버전은 고정된 Diffusers , Transformers , PyTorch 및 safetensors 의존성을 재사용합니다. 그들의 토크나이저 및 임베딩 테이블 API 는 학습된 토큰 등록 및 텐서 저장을 처리하며, 프로젝트 코드는 로컬 파일 동일성, 인코더 호환성 및 토큰 충돌을 확인하고 출처를 기록합니다. 이 기능은 학습 런타임 , 새 패키지, 번들 임베딩 가중치 또는 자동 다운로드를 추가하지 않습니다. 그의 학습된 벡터는 런타임 라이선스와 독립적으로 자체 약관을 가집니다. 보라
[학습된 텍스트 임베딩](text-embeddings.md).

CPU 프롬프트 인코딩과  RAM 오프로드 는 이미 고정된  PyTorch ,  Diffusers , 및 Accelerate 패키지를 재사용합니다. Accelerate 가 오프로드 후크 수명 주기를 소유하므로, 사용자 정의 후크, 디스크 스왑 레이어, 외부 스케줄러, 또는 추가 의존성이 도입되지 않습니다.  상위 공급 측 메모리 가이드는 연결됩니다
[hardware-compute.md](hardware-compute.md#cpugpu-cooperation-and-ram-storage).

로컬 모델,  VAE ,  LoRA , 및  ControlNet 입력은  `.safetensors` 와  `.safetensor` 를 받습니다. 단수형 철수는 임시 표준  `.safetensors` 심링크를 사용하여  Diffusers 의  safetensors 브랜치를 유지하며, pickle-weight 형식은 거부됩니다. 원격 어댑터 파일명은 여전히  `.safetensors` 를 요구합니다. 구성 스냅샷은 불변의 수정에서 명시적으로 해결되어 단일 파일 로더로 로컬 디렉토리로 전달되므로 오프라인 작업이  Diffusers 의 자동 구성 다운로드  대체 경로 에 의존하지 않습니다. 상세 내용은
[model-inputs.md](model-inputs.md).

<a id="reviewed-future-inference-backends"></a>

## 향후 추론 백엔드 검토

|후보|검토된 릴리스|라이센스|결정|
|---|---:|---|---|
|[ONNX 런타임](https://github.com/microsoft/onnxruntime)| 1.29.0 | MIT |추천되는 후기 크로스 플랫폼 백엔드; 버전화된  ONNX 내보내기 파이프라인을 요구합니다|
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) |롤링 커밋 릴리스| MIT |파이프라인 의미 체계를 소유하기 때문에 비교/스파이크만|
|[ml-안정-확산](https://github.com/apple/ml-stable-diffusion)| 1.1.1 | MIT |전체  Core ML / Swift 확산 배포 참조; 구현된 네이티브 컴포넌트 브리지와 별개입니다|

이러한 대안은 연결되어 있지 않습니다. MLX의 현재 구현은 일반 텐서/백엔드 인터페이스를 동결하거나 파이프라인 소유권을 외부 전체 Stable Diffusion 구현으로 이전하지 않습니다.

<a id="model-and-project-licensing"></a>

## 모델 및 프로젝트 라이선스

추론 라이브러리 라이선스는 모델 가중치를 포함하지 않습니다. 고정된 Stable Diffusion  1.5 미러는  CreativeML   OpenRAIL -M 을 선언합니다. 고정된 SDXL Base 1.0 저장소는 CreativeML Open RAIL ++-M 을 선언하며, 이는 별도의 SDXL 0.9 연구 용어와 혼동되어서는 안 됩니다. 고정된 FLUX.1 -schnell 저장소는 Apache-2.0 를 선언하며, FLUX.1 -dev 의 별도의 비상업적 라이선스와 혼동되어서는 안 됩니다. 모델 정체성, 수정본, 라이선스 메타데이터, 안전 검사기 상태, 그리고 워터마커 상태는 최종 생성 결과에 반드시 포함되어야 합니다.

원본 코드와 iiLocalDiffusion 에 있는 문서는 GNU Affero General Public License 버전 3.0 만 ( `SPDX-License-Identifier: AGPL-3.0-only` ) 배포되며, 전체 라이선스 텍스트는 [LICENSE](../LICENSE) 에서 확인하세요. 제 3 자 코드, 라이브러리, 도구, 및 모델 가중치는 각각의 라이선스를 유지하며, [THIRD_PARTY_NOTICES .md](../THIRD_PARTY_NOTICES.md)에서 확인하세요.

<a id="temporal-video-dependency-decision"></a>

## 일시적인 비디오 종속성 결정

비디오 백엔드는 Diffusers 0.40.0 `LTXConditionPipeline` 와 유지되는 LTX 트랜스포머, 시간적 VAE , T5 및 플로우 매칭 스케줄러를 재사용합니다. 기존 PyTorch / Transformers /Accelerate 스택은 신경 추론 및 오프로드를 담당하며, 기존 Pillow / FFmpeg 유틸리티는 이미지 준비 및 검증된 MP4 게시를 담당합니다. `requirements-video.txt` 는 `protobuf==7.36.1` 의 Transformers ' 게시된 T5 SentencePiece 토크나이저 변환을 추가합니다. 이 유지되는 BSD-3 -Clause 패키지는 작은 네이티브 휠을 제공하고 커스텀 토크나이저 구현을 피하며, [고정된 릴리스](https://pypi.org/project/protobuf/7.36.1/)를 참조하세요. Tokenizer 로딩은 다중 GB 모델 가중치를 읽기 전에 확인됩니다. 이전에 검증된 LTX Video 2B 0.9.5 가중치는 별도의 Open RAIL-M 용어를 가지며, 호출자는 호환 가능한 로컬 가중치를 제공해야 합니다. 새로운 모델 버전은 다른 용어를 가집니다. 참고 워크플로우, 선택 근거 및 범위에 대한 [비디오 계약](temporal-video.md) 을 참조하십시오.
