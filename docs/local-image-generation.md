<a id="standalone-local-image-generation"></a>

# 독립형 로컬 이미지 생성

`iild-generate --model-path /models/checkpoint.safetensors`는 아키텍처별로 SDK Python 프로세스에서 네이티브 엔진 또는 Diffusers/PyTorch를 선택합니다. 참조
FLUX.2, Z-Image, Krea 2, SD3와 분리된 텍스트 인코더/VAE 가중치의 [백엔드 라우팅](backend-routing.md)이다. ComfyUI 소스, 인터프리터, HTTP 서버, 워크플로, 사용자 정의 노드 또는 ComfyUI 캐시는 필요하지 않다. Dreamscapes도 동일한 `--backend local` 경로를 사용한다.

앱 큐의 경우 SDK 의 [거주 추론 작업자](inference-worker.md)를 사용하세요. 이는 요청 간 Python 초기화 및 호환 로드 가중치를 유지하며, 파일 시스템 식별자가 변경되지 않으면 전체 모델 해시를 재사용합니다. 모델 구성, 파싱된 구성 및 장치 배치 는 메모리에 유지됩니다. 샘플링 옵션은 모델을 재구성하거나 배치 를 반복하지 않고 변경할 수 있으며, 실행 배치 만 변경하면 이미 로드된 구성 요소를 재사용합니다.

<a id="offline-checkpoint-configuration"></a>

## 오프라인 체크포인트 구성

SDK 는 약  4.8 의  MB   SD   1.x 와  SDXL 구성/토크나이저 리소스를  `reference/diffusers/configs/` 아래에 탑재합니다. 정확한 상위 공급 측 수정본, 파일 해시 및 로컬 수정은 `configs/manifest.json` 에 기록됩니다. 이 리소스에는 모델 가중치가 포함되어 있지 않습니다. SDXL 파생물(예: Illustrious) 은 SDXL 텐서 아키텍처를 사용하며, 선택적인 Civitai 사이드카 또는 `--base-model` 는 더 구체적인 식별자를 유지합니다. 파일명은 아키텍처를 선택하지 않습니다.

완전한 SD 1.x 또는 SDXL 체크포인트는 디노이저, VAE 및 텍스트 인코더를 제공합니다. 번들된 구성은 구성 요소 정의, 스케줄러 및 CLIP 어휘/병합을 제공합니다. `--model-config` 는 명시적인 로컬 구성으로 이를 덮어쓸 수 있습니다. 모델/구성 로딩은 오프라인으로 유지됩니다. 누락된 동반 가중치는 추론 전에 보고되며 다운로드되거나 대체되지 않습니다.

```bash
iild-generate --model-path /models/illustration.safetensors \
  --prompt 'a red ceramic teapot on a wooden table' \
  --width 512 --height 512 --steps 20 --device mps \
  --output-dir /existing/parent/new-result
```

`--backend local` 도 LoRA, ControlNet, 텍스트 인버전, HiRes, 스케줄러, 데이터 타입 및 오프로드에 대한 기존 프리셋 옵션을 모두 받습니다. 예측 메타데이터는 명시적으로 덮어쓰이지 않는 한 유지됩니다. `.safetensor` 와 대문자 접미사는 원래 파일을 이름 변경하거나 복사하지 않고 확인된 임시 `.safetensors` 별칭을 사용합니다.

구성된 FLUX.1 단일 파일에는 여전히 로컬 보조 가중치와 `--model-config`가 필요하다. 그 밖의 지원되는 Diffusers 파이프라인은 완전한 로컬 모델 디렉터리 또는 호환되는 단일 파일 구성과 함께 `--backend diffusers`를 사용한다. SD2, SD3, 편집/refiner 및 양자화 모델을 SDXL로 잘못 취급하지 않는다. 선택적 레거시 변환/GGUF/그래프 백엔드는 `--backend comfyui-local`로 명시적으로 선택한다. [관리형 ComfyUI](managed-comfyui-image.md)를 참고한다. 해당 백엔드로 자동 대체 경로는 없다.

`--seed` 를 생략하면 요청마다 새 무작위 기본 시드가 선택됩니다. `--seed 0` 를 포함한 명시적인 시드는 유지됩니다. 해결된 요청 및 이미지별 메타데이터는 실제 시드를 기록하며, 이미지 배치들은 `--seed-stride` 를 사용하여 후속 시드를 유도합니다. Deforum/Interpolator 와 LTX 도 요청마다 기본 시드를 무작위화하면서 구성된 프레임/샷 시드 정책을 유지합니다.

<a id="consumer-paths-and-publication"></a>

## 소비자 경로 및 출판

`--preview-dir`는 단일 패스 이미지 요청을 위해 [실시간 노이즈 제거 미리 보기](live-previews.md)를 활성화합니다. 각 실제 샘플링 단계는 최종 출력 게시와 독립적으로 VAE로 디코딩된 PNG 및 플러시된 `IILD_PREVIEW` JSON 이벤트를 게시합니다.

`--output-dir` 는 새 디렉토리 또는 빈 디렉토리를 받습니다. 이미지와 그 기원이 성공적인 추론, 디코딩 및 해시 검증 후 인접한 임시 디렉토리에 준비되어 함께 게시됩니다. `generation.json` 는 스키마  `iild-standalone-image-v1` , 백엔드  `diffusers` , 상태  `complete` , 최종 출력 경로 및 이미지별 전체 메타데이터를 사용합니다. 실패 시 출력 디렉토리가 비게 되며 기존 결과를 유지합니다. `--output` 는 프리셋의 직접적인 이미지/사이드카어 출력 계약을 유지하며  `--output-dir` 와 상호 배타적입니다.

`--work-dir` 는 선택적으로 해결된 요청을 새 또는 빈 실제 디렉토리에 저장합니다. Dreamscapes 는  Society 바깥에 앱 소유의 임시 작업 디렉토리를 제공하고 작업/세션 종료 시 이를 제거합니다. `--cache-dir` 는  런타임 /weight-alias 캐시를 유지하며  Dreamscapes 는 이를 같은 임시 작업 디렉토리에 배치하고  Society 에 영구 저장하지 않습니다. 리디렉션되거나 비어 있지 않은 출력/작업 디렉토리는 거부됩니다. 기존  `--startup-timeout` 는 이전 호출자에게 허용되지만 서버는 시작되지 않으며  Dreamscapes 는 더 이상 이를 제공하지 않습니다. Dreamscapes 는 지속성 또는 재시작 복구를 없이 앱 메모리에서만 큐 및 작업 상태를 유지합니다. 그것은 엔진 출력과 기원을 임시 작업 디렉토리에 배치한 후 검증된 이미지만을 직접  Society   `Generation History/` 로 게시합니다. 기록에는 앱/작업 하위 폴더, 요청  JSON ,  런타임 캐시 또는 미리보기가 없습니다. 생성된 이미지는 `Asset Library/` 에 자동으로 등록되지 않습니다.

`--print-config` 및 `--validate-only`는 파이프라인을 로드하거나 샘플링하지 않고 요청을 검증합니다. 그것은 추론 증거가 아닙니다.

<a id="standalone-video"></a>

## 독립형 비디오

LTX 비디오는 이미 Diffusers / PyTorch 와 FFmpeg /FFprobe 를 통해 직접 실행됩니다. 완전한 로컬 LTX 모델 디렉토리가 필요합니다. Deforum/Interpolator 는 동일한 이미지 모델 로더를 사용하며 이제 호환되는 체크포인트 파일에 대해 번들된 SD1/SDXL 구성을 자동으로 선택합니다.

```bash
iild-generate --backend video --model-path /models/local-ltx \
  --prompt 'a slow camera move through a forest' --frames 9 --fps 24 \
  --output /existing/parent/forest.mp4

iild-generate --backend deforum --model-path /models/illustration.safetensors \
  --prompt 'a red ceramic teapot' --width 512 --height 512 --dtype float32 \
  --frames 4 --fps 8 \
  --output /existing/parent/teapot.mp4
```

선택 사항인 Deforum 의존성은 OpenCV 로 유지됩니다. 새로운 추론 패키지나 유료 서비스가 도입되지 않았습니다. Diffusers ( Apache-2.0 ) 과 PyTorch 는 유지되는 상위 공급 측 추론 구현으로 남아 있으며, 작은 번들 리소스는 상위 공급 측 모델 라이선스와 고지를 그대로 유지합니다. 참조
[타사 통지](../THIRD_PARTY_NOTICES.md) 및 공식
[Diffusers 단일 파일 로더](https://huggingface.co/docs/diffusers/api/loaders/single_file).

<a id="verification"></a>

## 검증

`StandaloneImageTests` 는 서버 생성 없이 라우팅, 번들 리소스 무결성, 모델 패밀리 및 누락된 구성 요소 거부, 대문자 접미사, 이미지 애니메이션 구성 재사용 및 실패한 출력 보존을 다룹니다. 이 변경에 대한 빌드 및 설치된 런타임 추론 증거는 `build/standalone-validation/` 에 기록되어 `standalone-validation.md` 에서 설명됩니다.

<a id="sdxl-color-preserving-hires-configuration"></a>

### SDXL 색상 보존 HiRes 구성

외부 SDXL 호환 단일 VAE 는 `--vae /local/vae.safetensors` 로 제공할 수 있습니다. 동일한 구성 요소는 기본 디코딩과 HiRes 인코딩/디코딩에 유지됩니다. SDXL 패밀리 프리셋의 기본 HiRes 강도는 0.25 이며, 전체 일정이 확장되어 최소 10 유효한 노이즈 제거 단계를 유지합니다. 명시적인 HiRes 일정은 결코 길어지지 않습니다. `--hires-save-base` 는 첫 번째 패스 이미지를 직접 검증하기 위해 저장하며, VAE 유한성만은 색상 재구성 테스트가 아닙니다.
