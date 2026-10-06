<a id="civitai-compatibility-validation--2026-09-04"></a>

# Civitai 호환성 검증 — 2026-09-04

구현에서는 다운로드한 파일 검사, 관리형 로컬 이미지 생성, 명명된 이미지 사전 설정, 일반 내장 Diffusers 파이프라인 및 `reference/generate.py`를 통한 명시적 로컬 ComfyUI 워크플로를 노출합니다. [usage 및 전체 카탈로그](civitai-models.md)를 참조하세요.

<a id="verified-environment-and-checks"></a>

## 검증된 환경 및 점검

- CMake 구성과 `cmake --build build --parallel 4`: 통과했다.
- `ctest --test-dir build --output-on-failure`: **52/52가 합격**.
- Python 단위 테스트 발견: **539 사례, 537가 통과했으며 2 옵트인 사례가**를 건너뛰었습니다. 옵트인 실제 Torch 변환 스위트는 설치된 ML 환경에서도 실행되었습니다: **22/22가**를 통과했으며, 해당 2 사례도 포함됩니다.
- `git diff --check`: 통과했습니다.
- Diffusers 0.40.0, PyTorch 2.13.0, SentencePiece 0.2.2.
- 관리형 ComfyUI 0.34.0와 고정된 ComfyUI - GGUF 확장은 별도의 환경에 설치되어 있습니다. 실제 MPS 이미지 추론 및 클린 프로세스 종료는 공개 CLI를 통해 검증되었으며, 여기에는 일반 시스템 Python 호출자가 포함됩니다.
- 라이브 노드 스키마 검증: **152 그래프** 를 **52 Civitai 이미지 레이블** 에 대해 **25 아키텍처**에서 수행합니다. 체크포인트, 분할 및 GGUF 로더 스키마는 포함되며, 인벤토리 파일명은 픽스처 이므로 이는 텐서 로딩을 증명하지 않습니다.
- 오프라인 설치된- 런타임 감사: 모든 105 카탈로그 행에 대해 수행됩니다. **64 파이프라인 클래스 사용 가능, 15 워크플로우 필수, 24 호스팅됨, 2 알 수 없는**입니다. 고정된 SentencePiece 를 설치한 후 Kolors 라우트가 누락된 상태로 남아있는 Diffusers 라우트는 없습니다. 클래스 사용 가능 여부는 모델 가중치나 추론을 검증하지 않습니다.

<a id="actual-generation-evidence"></a>

## 실제 생성 증거

|실행|증거 및 범위|
| --- | --- |
|유명한 이름의 경로, 128×128 EPS|캐시된 **원래 SDXL를 사용하여 통합 CLI를 통해 생성됨 1.0** 무게. 이는 Illustrious에서 학습한 가중치가 아닌 라우팅 및 호환 가능한 계약을 검증합니다.|
|NoobAI v-pred 경로, ControlNet + HiRes 128→192|원본 SDXL 가중치와 합성 ControlNet를 사용하였다. 두 단계 모두 v-prediction/0-SNR와 유한한 잠재 표현을 사용하였다. 이는 구성에 관한 증거이며 NoobAI 이미지 품질 검증은 아니다.|
|FLUX dev/Krea 유도 경로|safetensors에서 저장하고 다시 로드하는 작은 합성 유도 베어링 변압기; 안내를 변경하면 생성된 픽셀이 변경됩니다. 전체 Krea 12B 테스트가 아닙니다.|
|일반 FLUX, DDPM 및 SD3|2개 이미지 출력 배치 및 구성 요소/출력 해시를 포함한 작은 모델을 사용한 실제 CPU 추론입니다.|
|비디오/오디오 내보내기|합성 디코딩된 프레임 및 스테레오 WAV 인코딩이 확인되었습니다. 전체 비디오/오디오 모델 추론은 수행되지 않았습니다.|
|관리형 ComfyUI, 일러스트레이스 라벨|기존 6.94 GB 원본 SDXL 체크포인트 → 자동 검사/그래프 → MPS 추론 → 1개 128×128 PNG. 소스/출력 SHA256, 디코딩된 픽셀 및 프로세스 종료가 확인되었습니다.|
|관리형 ComfyUI, NoobAI 라벨|일반 `python3` → 관리되는 Python → v-prediction + 0-SNR를 사용하는 원본 SDXL 체크포인트 → 2개의 128×128 PNG이다. 이는 실행과 묶음 처리를 검증하며 NoobAI로 학습한 가중치나 시각적 품질의 검증은 아니다.|
|레거시 체크포인트 변환|실제 zip 및 이전 비zip Torch 체크포인트가 텐서/dtype/키 보존을 통해 safetensors로 변환되었습니다. 악성 축소 픽스처가 해당 마커를 실행하지 않고 거부되었습니다. Torch >=2.10가 필요합니다.|
|로컬 HTTP 계약|픽스처는 대기열 오류, 노드/유형/입력 오류, 아티팩트 다운로드, 리디렉션, 루프백 경계 및 게시 오류를 다룹니다.|

기계가 읽을 수 있는 증거와 자세한 로그는 `build/`에 남아 있습니다.

- `build/reference/civitai-validation.json`
- `build/reference/civitai-runtime-audit.json`
- `build/reference/model-families/sdxl-fixture-cli-smoke.json`
- `build/reference/model-families/tiny-flux-dev-smoke.json`
- `build/reference/generic-sd3-smoke/generation/generation.json`
- `build/reference/generic-diffusers-smoke/generation.json`
- `build/reference/generic-media-smoke/ddpm-generation/generation.json`
- `build/reference/generic-media-smoke/encoding-validation.json`
- `build/reference/local-image-smoke/verification.json`
- `build/reference/local-image-smoke/illustrious-sdxl-fixture/local-image.json`
- `build/reference/local-image-smoke/noobai-sdxl-fixture/local-image.json`
- `build/workflow-source-review/live-schema-validation.json`
- `build/reference/checkpoint-conversion-smoke/tiny-conversion.json`
- `build/reference/checkpoint-conversion-smoke/malicious-checkpoint.json`
- `build/civitai-ctest.log`, `build/civitai-python-tests-final.log`

구현은  **로**  모든  79  로컬에 사용 가능한 카테고리, 모든 체크포인트 형식, 커스텀 노드, 어댑터 또는 하드웨어 조합에 대해 성공적인 생성을 확립하지 않습니다. 나머지 모델별 계약은 실제 가중치, 일치하는 구성/컴포넌트, 사용 가능한  런타임  또는 워크플로우, 그리고 성공적으로 생성된 아티팩트를 요구합니다.  런타임  감사 및 저장된 출처는 카탈로그 커버리지를 보장으로 제시하지 않으면서 해당 검사를 반복 가능하게 만듭니다.

알려진 파일 수준의 격차: 설치된 Kolors 파이프라인은 전체  Diffusers  디렉토리를 로드하지만, 원본 Civitai 체크포인트에 대한  `from_single_file`  인터페이스가 없습니다.  HunyuanDiT  와  HiDream-O1  자동 그래프는 현재 번들된 체크포인트를 필요로 합니다. 이미지 조건부, 오디오, 비디오 및  3D 작업은 적절한 입력과 실행 경로를 필요로 하며, 컴포넌트 가중치만으로는 이미지 생성기로 사용될 수 없습니다.
