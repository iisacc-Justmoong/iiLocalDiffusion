<a id="model-families-and-local-checkpoint-generation"></a>

# 모델 패밀리 및 로컬 체크포인트 생성

`reference/diffusers/generate.py` 는 모델의 이름과 추론 아키텍처를 분리합니다. Illustrious, NoobAI 및 Pony XL 은 SDXL 파이프라인을 사용하며, FLUX.1 dev 와 Krea 는 가이드선스-distilled FLUX 를 사용합니다. 원래 `sd15`, `sdxl-base` 및 `flux1-schnell` 프리셋은 엄격한 참조 스냅샷 계약을 유지합니다.

|사전 설정|건축|기본 모델 또는 필수 입력|교육 설정|
| --- | --- | --- | --- |
| `sd15-compatible` | SD1.x |고정된 SD1.5 ; 선택된 SD1 .x 파생어를 수락합니다|로드된 호환 스케줄러를 유지합니다|
| `flux1-schnell-compatible` |FLUX.1 빠르게|고정된 FLUX.1 schnell; 선택된 schnell 파생어를 수락합니다|로드된 흐름 스케줄러를 유지합니다; 텐서 삽입에 대한 안내가 없습니다.|
| `sdxl` | SDXL |고정된 SDXL 베이스; 선택된 SDXL 파생어를 수락합니다|로드된 스케줄러를 유지합니다.|
| `illustrious` | SDXL |고정된 OnomaAI Illustrious XL 초기 릴리스|로드된 스케줄러를 유지합니다.|
| `noobai` | SDXL |고정된 Laxhar NoobAI XL 1.0 EPS|로드된 스케줄러를 유지합니다.|
| `noobai-v-pred` | SDXL |고정된 Laxhar NoobAI XL V-Pred 1.0|Euler, `v_prediction` , 0-터미널- SNR 베타 리스케일링|
| `pony` | SDXL |명시적 로컬 체크포인트 또는 선택된 Diffusers 디렉터리 필요|SDXL 구성은 변환일 뿐입니다. 대체 경로|
| `flux1-dev` | FLUX.1|고정된 Black Forest Labs FLUX.1 dev|안내 임베딩, 기본 안내 3.5, 512 T5 토큰|
| `flux1-krea-dev` | FLUX.1|고정된 Black Forest Labs FLUX.1 Krea dev|안내 임베딩, 기본 안내 4.5, 512 T5 토큰|

프리셋은 아키텍처와 기본값을 식별하며, 마케팅 이름을 가진 모든 체크포인트를 포함하지 않습니다. 예를 들어, SD1 .x 기반 Pony 버전은 SD1 .x 아키텍처를 필요로 하며, NoobAI 의 EPS 와 V-pred 가중치는 별도의 예측 설정이 필요합니다. 파일 이름을 단순히 변경하는 것만으로는 모델이 호환되지는 않습니다.

<a id="local-files-and-directories"></a>

## 로컬 파일 및 디렉터리

완전한 Diffusers 디렉토리는 저장된 구성 요소와 스케줄러를 사용하여 로드됩니다. 원격 오버라이드는 여전히 불변 40자리의 커밋 SHA 이 필요합니다. 단일 파일 모델은 로컬 `.safetensors` (또는 `.safetensor` ) 파일이어야 합니다. Pickle 체크포인트는 활성화되지 않습니다. 체크포인트/디노이저 파일과 VAE / LoRA 파일은 별도의 입력 역할, 파일 크기 및 SHA-256 출처와 전후 동일성 확인을 유지합니다.

```sh
reference/diffusers/.venv/bin/python reference/diffusers/generate.py --model-config /absolute/path/model-config \
  --preset illustrious --model /absolute/path/illustrious.safetensors \
  --prompt 'a blue ceramic teapot on a wooden table' \
  --output build/reference/illustrious.png

reference/diffusers/.venv/bin/python reference/diffusers/generate.py --model-config /absolute/path/model-config \
  --preset noobai-v-pred --model /absolute/path/noobai-vpred.safetensors \
  --prompt 'a blue ceramic teapot on a wooden table' \
  --output build/reference/noobai-vpred.png

reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset flux1-krea-dev --model /absolute/path/flux1-krea-dev.safetensors \
  --model-config /absolute/path/complete-flux-dev-diffusers-package \
  --local-files-only --prompt 'a blue ceramic teapot on a wooden table' \
  --output build/reference/krea.png
```

단일 파일 변환은 일치하는 토크나이저/구성 요소 구성이 필요합니다. Full SDXL 체크포인트는 인식된 CLIP 인코더와 VAE 를 모두 포함해야 하며, `--vae` 가 제공하는 경우를 제외합니다. 디노이저 전용 파일은 선택된 구성 패키지의 인코더와 VAE 를 사용하므로 해당 패키지는 이러한 가중치를 포함해야 합니다. 모든 모델, 구성 및 보조 가중치는 이미 로컬에 존재해야 합니다. 생성 작업은 항상 로컬 전용 로딩을 사용합니다. 저장소 선택이나 대체 경로 를 다운로드하지 않으면 누락된 구성 요소는 실패합니다.

전용 프리셋이 없는 V-pred 파생형의 경우 `--preset sdxl --prediction-type v_prediction` 를 사용하며, 모델의 훈련 레시피가 0-SNR 스케일링을 필요로 하는 경우 `--scheduler-config '{"rescale_betas_zero_snr":true}'` 를 사용합니다. `--prediction-type auto` 는 문서화된 훈련 기본값을 제외하고 로드된 값을 유지합니다. 충돌하는 명시적 `--prediction-type` 와 스케줄러 JSON 값은 아무런 알림 없이 재순열 대신 실패합니다. 스케줄러 구성 및 적용된 프리셋 기본값은 출력 메타데이터에 기록됩니다.

NoobAI V-pred 모델 카드는 오일러, V-예측 및 0-SNR 스케일링을 명시적으로 규정하며, 확인된 스케줄러 구성은 `epsilon` 을 선언합니다. 전용 프리셋은 로드 후 문서화된 레시피를 적용합니다. 사용자가 제공하는 스케줄러 또는 스케줄러 구성은 프리셋 기본값을 의도적으로 덮어쓸 수 있으며, 이는 해당 레시피의 품질을 설정하지 않습니다.

<a id="compatibility-boundaries-and-verification"></a>

## 호환성 경계 및 검증

SDXL 패밀리 검증은 내장 호환성 확산 스케줄러와 다양한 샘플 크기, 프롬프트 제로잉 및 VAE 업캐스트 정책을 허용합니다. 그것은 여전히 잘못된 UNet 입력/출력 채널, SDXL 리파이너/인페인팅 형식, 텍스트 인코더 너비, 잠재 표현 차원/스케일 및 사용자 정의 구성 요소 선언을 거부합니다. 플로우 스케줄러는 SDXL 확산 스케줄러로 사용할 수 없습니다. FLUX dev/ Krea 는 가이드 임베딩 텐서를 필요로 하며, 해당 가중치에 schnnell 구성을 적용하는 것은 로더가 추가 텐서를 삭제하기 전에 거부됩니다.

`tests/ModelFamilyTests.py` 는 패밀리 라우팅, 스케줄러 학습 기본값, 구성 허용 목록, 교차 패밀리 거부 및 안전 선택 경계를 포함합니다. ControlNet 파이프라인 변환은 기본 패밀리를, 가이드 및 스케줄러 학습 설정을 보존하며, 증강 파이프라인 유효성 검사도 다른 아키텍처에서 ControlNet 를 거부합니다. `ModelLoadingTests.py` 와 `GenerationSchedulerTests.py` 는 원래 로더와 스케줄러 회귀 회귀 범위를 유지합니다. 합성 미니 모델은 실제 로딩, 노이즈 제거 및 가이드 배관 구조를 확인할 수 있지만, 전체 크기 커뮤니티 모델의 시각적 품질이나 완전성을 증명하지는 않습니다.

SD1 .x 호환성 프리셋은 4채널 UNet 와 768-wide 텍스트/교차 주의력 계약을 유지하며, SD2 .x 와 인페인팅 아키텍처를 거부합니다. 스케줄러가 다른 경우에도 여전히 가이드 증류된 dev/ Krea 가중치를 거부합니다.

2026-09-04의 로컬 통합 실행에서는 캐시된 원본 SDXL 1.0 fp16 가중치와 공개 `reference/generate.py` 진입점도 사용하였다. `illustrious` 라우팅은 128x128 EPS 이미지를 생성하였고 합성 ControlNet를 사용한 `noobai-v-pred` 라우팅은 192x192 HiRes 이미지를 생성하였다. 두 V-pred 단계는 모두 0-SNR 재조정과 유한한 잠재 표현을 기록하였다(기본 2단계와 개선 한 단계). 로컬 보고서는 `build/reference/model-families/sdxl-fixture-cli-smoke.json`이다. 이는 SDXL 픽스처로 라우팅/구성을 검증하며 학습된 Illustrious/NoobAI/Pony 가중치나 그 이미지 품질을 검증하는 것은 아니다. 해당 실행의 NoobAI EPS 및 Pony 경로는 구성 검사만 수행하였다.

이 네이티브 어댑터는 Civitai 엔트리의 모든 것, 작업 유형, 양자화 컨테이너 또는 커스텀 노드 워크플로우를 모두 지원하지 않습니다. 양자화 변형 및 기타 아키텍처는 자체 검증된 런타임/컨테이너 지원이 필요합니다. 추론 호환성 또한 모델 재배포 또는 상용 사용 권한을 부여하지 않으며, 선택된 체크포인트의 라이선스를 독립적으로 확인해야 합니다.

<a id="primary-references"></a>

## 주요 참고문헌

이러한 상위 공급 측 모델 카드/구성을 확인하고 2026-09-04에 Diffusers 0.40.0를 설치했습니다.

- [일러스트 XL 모델 카드](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0)
- [NoobAI V-pred 모델 카드 및 Diffusers 레시피](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/66aa55e3469c27c29a89813cd35dd95fb7485fa1/README.md)
- [NoobAI V-pred 스케줄러 구성](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/66aa55e3469c27c29a89813cd35dd95fb7485fa1/scheduler/scheduler_config.json)
- [Pony XL 작성자 체크포인트 릴리스](https://huggingface.co/AstraliteHeart/pony-diffusion-v6/commit/5ec9c05863255568f1b59753e3838107befaa712)
- [FLUX.1 Krea 개발자 모델 카드](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)
- [Black Forest Labs 모델 사양](https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py)
- [Diffusers FLUX 파이프라인](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/flux/pipeline_flux.py)
