<a id="standalone-generation-verification"></a>

# 독립형 생성 검증

<a id="sdxl-mps-attention-precision--2026-09-11"></a>

## SDXL MPS 주의 정밀도 — 2026-09-11

Dreamscapes 가 `Cannot preview non-finite denoising latents` 를 100 이미지들을 큐잉하는 동안 `A beautiful girl` 와 비율 3:4로 보고했습니다. 현재 512px-범위 애플리케이션은 해당 비율을 384×512 로 해결했으며, 애플리케이션은 별도의 이미지 요청을 순차적으로 처리합니다. `redLilyIllu_v10.safetensors` 와 20 단계 및 명시적 시드 0로, 이전 기본값 MPS FP16 슬라이스 주의 경로는 UNet 호출에서 NaN 를 생성했습니다. 슬라이싱을 비활성화하면 UNet 출력의 유한한 20 단계를 모두 완료하며, 명시적으로 슬라이스된 주의 점수를 업캐스팅해도 20 단계를 모두 완료합니다. 별도의 유한 입력 주의 탐지는 소프트맥스 전에 FP16 점수 오버플로를 재현했습니다.

SDXL 는 기본적으로 MPS 에서 PyTorch SDPA 를 유지합니다. 명시적 슬라이싱은 FP16 UNet / ControlNet 주의에 Diffusers ' FP32 점수 누적 을 사용합니다. 모델 가중치는 FP16 로 유지되며, 거주 파이프라인이 재사용되고 무효 미리보기/출력 확인이 계속 활성화됩니다. 증거 및 설치된 런타임 런타임 검증은 `build/nonfinite-latents/REPORT.md` 에 기록됩니다.

<a id="sdxl-rectangular-decoding--2026-09-11"></a>

## SDXL 직사각형 디코딩 — 2026-09-11

Dreamscapes 는 이제 1024px에서 짧은 쪽을 유지하므로, 9:16 요청은 잠재 그리드로 반올림한 후 1024×1824 됩니다. VAE 타일링이 없는 초기 실제 앱 시안은 첫 번째 미리보기에 대략 13 초에서 후기 미리보기 사이 대략 130 초로 느려졌고 5 단계 후 취소되었습니다. 프로세스 샘플은 MPS 업샘플링에 의해 지배되었고, 미리보기 경로는 각 단계마다 풀 해상도 VAE 이미지를 디코딩했으며, 시스템 스왑 사용량은 대략 17.7 GB 에 도달했습니다. 최종 이미지가 게시되지 않았습니다.

SDXL 가속기 기본값은 기존 Diffusers VAE 타일링 경로를 활성화합니다. 번들된 VAE 는 1024×1024 에서 풀 디코딩을 유지하며 더 큰 영역을 위해 겹치는 타일을 사용합니다. 모델 배치는 계속 유지되며, 명시적 비활성화 및 CPU 기본값은 보존됩니다. SDK 빌드와 모든 79 CTest 항목이 통과했습니다. 실제 앱 증거와 최종 이미지 검증 상태는 Dreamscapes `build/quickgenerate-short-side/REPORT.md` 에 보관됩니다.

업데이트된 Dreamscapes 앱은 1024×1824 에서 실제 MPS FP16 생성을 완료했으며, 220.789 초에 20 미리보기 이벤트를 처리했습니다 (전경 준비 제외). 저장된 RGB PNG 차원과 표시된 완료된 이미지가 확인되었습니다. 생성은 준비된 모델을 0 파이프라인 로드로 재사용하거나, 전체 모델 해시 또는 장치 배치를 사용했습니다. 이는 중립적인 주전자 이미지 하나였으며, 100-이미지 실행이나 실제 모바일 장치 테스트가 아닙니다.

<a id="standalone-routing--2026-09-08"></a>

## 독립형 라우팅 — 2026-09-08

이전 Dreamscapes 요청이 `Local image runtime is missing. Run reference/setup_comfyui.py once.`로 인해 실패했습니다. 이제 자동 단일 파일 디스패치가 `standalone_image`를 호출합니다. 이는 SDK의 Diffusers/PyTorch 프로세스를 사용하고 번들로 제공됩니다. SD1/SDXL 구성/토크나이저. ComfyUI는 명시적으로 선택된 옵션입니다.

<a id="verified-results"></a>

## 검증된 결과

- SDK 네이티브 빌드 및 CTest: **72/72가 `build/standalone-validation/build`에서**를 통과했습니다.
- Python 런타임 제품군: **744 실행, 742 통과, 2 환경 건너뛰기**, 오류 없음.
- Dreamscapes 는 `build/` 에서 재구성되었으며, GUI 및 생성 스위트: **2/2 는**를 통과했습니다.
- CMake 소비자를 설치하고 재배치된 실행기/리소스 검증을 통과했습니다.
- 런타임 -만 설치가 `~/.local/SDK/iiLocalDiffusion`로 업데이트되었습니다; 3 기존 네이티브 라이브러리 해시가 변경되지 않았습니다.
- 실제 Dreamscapes는 `redLilyIllu_v10.safetensors`가 완료된 작업 `4e4f253e-33e2-4ee8-8c0e-89c176883dc8`와 함께 작업을 생성합니다: **512×512, 20 단계, MPS FP16**. 네이티브 결과 화면에 생성된 티포트 이미지가 표시되었습니다. PNG의 SHA-256는 `cb354b8a781c3de796edf0e4235ab78ed119e001e5cd5fa00daed3745035c161`입니다.
- 설치된 Deforum 백엔드는 동일한 실제 SDXL 체크포인트와 번들 구성을 사용하여 **512×512, 3-프레임 GIF를 MPS FP32에 생성했습니다 (8 FPS 요청; GIF 지연은 센티초 사용)**.
- 설치된 LTX 백엔드가 생성하고 보간한 **64×64, 9프레임, 24 FPS H.264 MP4** 를 작은 로컬로 초기화된 LTX 테스트 모델에서 가져왔습니다. FFprobe 가 프레임 수/속도를 확인하고 생성자가 디코딩을 확인했습니다. 이는 파이프라인과 미디어 인코딩 증거이며, 전체 크기의 LTX 품질 벤치마크가 아닙니다.

<a id="random-seed-defaults"></a>

## 무작위 시드 기본값

Image/checkpoint, Deforum/Interpolator, LTX, 원격 이미지 및 관리된 워크플로우 요청은 이제 누락된 시드를 새 무작위 32비트 기본 시드로 해결합니다. 명시된 시드, 0를 포함한 시드는 그대로 유지됩니다. 배치 스트라이드, 애니메이션 프레임/샷 정책 및 HiRes 시드 상속은 보존됩니다. 해결된 구성 및 생성 메타데이터는 재생을 위해 실제 시드를 유지합니다. 일반 Diffusers 는 계속 파이프라인의 기존 확률적 기본값을 사용합니다.

설치된 런타임 는 실제 CPU 추론을 기존 작은 로컬 SD15 와 LTX 픽스처 와 함께 수행했습니다. 2 이미지 요청은 다른 시드와 픽셀 해시를 가졌습니다; 첫 번째 기록된 시드를 재생하면 정확한 픽셀 해시를 재현했습니다. 2 LTX 요청은 다른 시드와 프레임 해시를 가졌으며, 확인된 9프레임, 24 FPS H.264 디코딩을 포함합니다. 이 픽스처 는 시드 처리 및 실행을 검증하며, 모델 품질을 검증하지 않습니다. 증거는 `build/standalone-validation/random-seed/verification.json` 에 있습니다. 동일한 설치된 독립형 체크포인트 라우트는 실제 Society `redLilyIllu_v10.safetensors` 에 대해서도 해결되었으며, 2 누락된 시드는 다르고 명시된 0 는 0로 유지되었습니다. 이 구성 확인은 큰 모델을 다시 샘플링하지 않았습니다. 런타임 설치에서는 모든 3 기존 네이티브 라이브러리가 유지되었습니다.

구조화된 보고서 및 복사된 Dreamscapes PNG/출처는 `build/standalone-validation/verification.json` 및 `build/standalone-validation/dreamscapes-*`에 있습니다. 네이티브 빌드, CTest, Python 테스트, 설치된 런타임, 이미지 및 비디오 로그가 옆에 보관됩니다.

<a id="boundaries"></a>

## 경계

호스트의 누락된 Metal 컴파일러는 기존 선택형 네이티브 MLX Metal 백엔드를 구성하는 것을 방지합니다. 별도의 검증 빌드는 선택형 MLX / LibTorch / CoreML 을 비활성화하며, 활성 SDK 에 적용된 것은 오직 런타임 설치 구성 요소뿐입니다. Python MPS 추론은 실제 GPU 에서 실행되었습니다.

명시적인 256×256 FP16 Deforum 탐지는 유한하지 않은 디노이징 잠재 표현을 보고하고 완료된 비디오를 게시하지 않았습니다. 512×512 FP32 탐지는 3 프레임 전체를 완료했습니다. 해당 모델의 검증된 애니메이션 설정을 `--dtype float32` 를 사용하세요. 이미지 경로의 기본 MPS FP16 은 512×512에서 성공했습니다. 이는 모든 체크포인트/정밀도/차원 조합에 대한 지원을 확립하지 않습니다.

모델 가중치가 다운로드되거나 번들로 제공되지 않았습니다. iOS/Android 네이티브 추론 및 임의 GGUF/사용자 지정 노드 모델은 독립 실행형 데스크톱 경로 외부에 유지됩니다.
