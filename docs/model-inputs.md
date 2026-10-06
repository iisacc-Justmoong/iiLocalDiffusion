<a id="model-and-vae-input-contract"></a>

# 모델 및 VAE 입력 계약

<a id="scope"></a>

## 범위

The Python generation oracle 는 독립적으로 선택된 모델, VAE, 및 LoRA 가중치를 수락합니다. 원래 엄격한 프리셋 `sd15`, `sdxl-base`, 및 `flux1-schnell` 는 호환 가능한 SD1/SDXL/FLUX 프리셋과 Illustrious, NoobAI, Pony, FLUX 개발 및 Krea 변형과 함께 계속 사용 가능합니다. 참조하세요
[모델 제품군](model-families.md). 다른 가중치를 선택한다고 해서 아키텍처가 변경되거나 제품군 간 호환성을 의미하지는 않습니다.

주요 모델 입력은 이제 3 위치를 가지며: `--model-path` (구형 `--model`), `--model-api`, 또는 `--model-cloud` 와 `--model-provider` 입니다. 참조
원격 실행을 위한 [3소스 계약](model-sources.md), JSON/Python 값, 자격 증명 및 공급자 경계. 아래의 로컬 구성 규칙은 로컬 경로 소스에만 적용됩니다. 보조 VAE/LoRA/ControlNet는 로컬 입력으로 유지됩니다.

이 인터페이스는 `reference/diffusers/generate.py`에 속합니다. Python 검사 명령은 여전히 ​​Diffusers 패키지를 허용하며, C++ 검사기는 텐서 가중치를 구문 분석하거나 실행하지 않고도 패키지 메타데이터의 유효성을 검사합니다.

<a id="arguments"></a>

## 인수

|인수|허용되는 입력|목적|
|---|---|---|
| `--model-path` / `--model` |로컬 Diffusers 디렉터리 또는 로컬 단일 파일|전체 로컬 모델 패키지, 체크포인트 또는 노이즈 제거기 선택|
| `--model-api` |직접 추론 엔드포인트 URL|배포된 모델을 원격으로 평가|
| `--model-cloud` |공급자 모델 ID, `--model-provider`|원격으로 클라우드 모델 평가|
| `--model-config` |단일 파일 모델에 필요한 로컬 Diffusers 디렉터리|공급 구성 및 보조 구성요소|
| `--vae` |로컬 `.safetensors` 또는 `.safetensor` 파일|`AutoencoderKL` 가중치 교체|
| `--lora` |로컬 파일 또는 디렉터리|파이프라인 검증 후 하나의 선택적 어댑터 적용|

각 로컬 생성 요청은 로컬 모델을 요구합니다. 프리셋은 아키텍처와 샘플링 기본값을 제공하며, 기본 저장소나 자동 가중치 다운로드가 없습니다. 누락된 경로와 허브 ID 는 로딩 전에 거부되며, 선택적 ControlNet 및 어댑터 입력에도 해당됩니다. 비-null 구형 리비전 인수는 거부됩니다. VAE 또는 LoRA 을 생략하면 모델의 VAE 가 보존되거나 어댑터가 비활성화됩니다. [로컬 모델 생성](local-model-generation.md), [LoRA](lora.md) 와
[ControlNet](controlnet.md).

로컬 가중치 파일은 소문자 `.safetensors` 또는 `.safetensor` 확장자를 사용하여 일반 파일 심볼릭 링크를 통해 도달할 수 있는 비어 있지 않은 일반 파일이어야 합니다. 레거시 체크포인트와 GGUF는 별도의 관리형 로컬 이미지 백엔드를 사용합니다.

<a id="model-file-roles"></a>

## 모델 파일 역할

로더는 파이프라인을 조립하기 전에 safetensors 키를 검사합니다.

|사전 설정 및 파일 역할|`--model`에서 사용되는 가중치|`--model-config`에서 사용되는 가중치 및 메타데이터|
|---|---|---|
|SD 1.5 / SDXL 원본 형식 체크포인트|UNet, 텍스트 인코더 및 재정의되지 않는 한 내장형 VAE|구성 요소 구성, 토크나이저, 스케줄러 및 선택된 보조 구성요소|
|SD 1.5 / SDXL 독립형 UNet|UNet 만|설정, 토크나이저, 텍스트 인코더(s), 스케줄러 및 VAE 는 오버라이드되지 않는 한|
|FLUX.1 schnell/dev/ Krea 독립형 트랜스포머|트랜스포머 만; 가이드런스 가중치는 선택된 패밀리 변형과 일치해야 합니다.|설정, 토크나이저, CLIP, T5, 스케줄러 및 VAE 는 오버라이드되지 않는 한|

SD / SDXL 풀 체크포인트 분류는 원래 디노이저 키와 Diffusers 가 인식하는 모든 필요한 임베딩 텍스트 인코더에 대한 정확한 CLIP 센티넬 키가 필요합니다. 필수 텍스트 인코더가 존재해야 합니다; `--model-config` 에서 완전히 분류된 체크포인트에 대해 유사하게 이름이 붙은 가중치로 절대로 되돌아가지 않습니다. 이것은 부분적으로 임베딩된 체크포인트가 유일한 신경 가중치 소스로 보고되는 것을 방지합니다. 누락된 임베딩 VAE 에는 명시적인 `--vae` 가 필요합니다. 독립형 네이티브 Diffusers UNet 상태 사전과 원본 형식 UNet 전용 파일은 해당 Diffusers 컴포넌트 로더에 의해 지원됩니다. FLUX 는 네이티브 또는 지원된 원본 형식 가중치에 대해 자신의 변환기 컴포넌트 로더를 사용합니다. 모델 파일은 어댑터 전용 또는 VAE 전용 파일로 대체될 수 없습니다.

`--model-config` 는 단일 파일 `--model` 와 함께 필수이며 기본값이 없습니다. 명시적으로 제공된 로컬 소스는 허용된 컴포넌트 선언과 필수 컴포넌트 구성을 포함하여 `model_index.json` 에서 일치하는 Diffusers 파이프라인을 기술해야 합니다. 독립형 JSON / YAML 구성 파일은 이 인터페이스가 아닙니다. 임베딩되지 않은 신경 컴포넌트가 필요한 경우, 소스는 또한 해당 가중치를 제공해야 합니다. 모든 필수 보조 가중치가 해당 위치에 사용 가능하지 않은 경우, 보조기 전용 모델에 대한 구성 전용 디렉터리는 충분하지 않습니다. 비 NULL IP -어댑터 또는 `image_encoder` 선언은 해당 보조 모델이 이 생성 인터페이스 외부에 있기 때문에 거부됩니다. 구성 소스는 선택된 프레셋을 추가 모델 컴포넌트로 아무런 알림 없이 확장할 수 없습니다.

가속기 실행 시, 로컬 Diffusers 소스는 컴포넌트 파일 이름이 해당 변형을 선언할 때만 프레셋의 `fp16` 변형만 사용합니다. SD 1.5 안전성 체크서는 자신의 로컬 하위 디렉터리를 독립적으로 확인합니다. 단일 파일 가중치는 선택된 런타임 데이터 형식으로 로드됩니다.

예를 들어, 모든 3 SDXL 가중치 입력을 명시적인 로컬 구성 소스로 교체합니다.

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --model-config /absolute/path/model-config \
  --preset sdxl-base \
  --model /absolute/path/to/sdxl-checkpoint.safetensors \
  --vae /absolute/path/to/sdxl-vae.safetensors \
  --lora /absolute/path/to/sdxl-style.safetensors \
  --lora-scale 0.75
```

FLUX 보조 구성 요소의 소스로 명시적인 로컬 패키지를 사용합니다.

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py \
  --preset flux1-schnell \
  --model /absolute/path/to/flux-schnell.safetensor \
  --model-config /absolute/path/to/flux-schnell-package \
  --vae /absolute/path/to/flux-vae.safetensors \
  --local-files-only
```

<a id="vae-replacement-and-ordering"></a>

## VAE 교체 및 주문

VAE 는 모델 패키지의 `AutoencoderKL.from_single_file` 와 `vae/` 설정 또는 단일 파일 모델의 경우 `--model-config` 로 로드됩니다. 텐서 파일은 스케일링, 시프트 또는 아키텍처 설정을 독립적으로 결정하지 않습니다. 네이티브 Diffusers 와 지원되는 원본 VAE 레이아웃은 Diffusers 에 위임되며, 설정과 텐서 모양은 일치해야 합니다. 양자화 및 대체 VAE 아키텍처는 이 API 에 의해 추가되지 않습니다.

VAE 는 파이프라인 생성기로 전달되므로, 대체된 VAE 을 먼저 로드할 필요가 없습니다. 조립된 파이프라인은 LoRA 로딩 및 활성화 전에 사전 설정 계약을 충족해야 하며, 그 후 장치 배치/오프로드 및 생성이 수행됩니다. 모든 3 프로필은 RGB 입력/출력과 4 VAE 레벨을 8x 공간 다운샘플링에 요구합니다. SD 1.5 은 4 잠재 표현 채널과 `0.18215` 스케일을 사용하고, SDXL 는 네 개와 `0.13025` 를 사용하고, FLUX 는 16, `0.3611` 스케일, 그리고 `0.1159` 이동을 사용합니다. SDXL 와 FLUX 는 추가 프로필 검사를 유지합니다.

<a id="safe-loading-offline-operation-and-provenance"></a>

## 안전한 로딩, 오프라인 운영 및 출처

제공된 로컬 설정 디렉토리는 `from_single_file` 에 직접 전달됩니다. 기본, 보조 및 LoRA 로딩은 항상 `local_files_only=True` 를 사용합니다. 누락된 설정, 토크나이저 또는 가중치는 대체물을 다운로드하지 않으면 실패합니다. 독립형 모델 파일은 여전히 로컬 설정과 모든 외부 신경 구성 요소를 필요로 합니다.

각 로컬 모델/ VAE / LoRA 파일은 제공된 절대 경로, 해결된 타겟, SHA-256 및 바이트 크기에 의해 식별됩니다. 로딩 전후로 동일성이 확인됩니다. 단수형 `.safetensor` 철수는 구성된 캐시 하위에 `.safetensors` 로 끝나는 임시 심링크를 사용하여 선택된 바이트를 복사하거나 재작성하지 않고 Diffusers 를 안전 로더 브랜치에 유지합니다. 별명은 이후 제거되며 심링크 지원이 없는 파일 시스템은 표준 확장명을 사용해야 합니다. 이러한 확인은 악의적인 동시 파일 시스템 변경에 대한 일반적인 샌드박스입니다.

Diffusers' 저용량 메모리 단일 파일 로딩이 사용되며 나머지 메타 텐서는 거부됩니다. 구성 요소 구성 및 모양은 여전히 ​​검증됩니다. 성공적인 safetensors 구문 분석은 호환성을 보장하지 않습니다. 이 기능에는 새로운 타사 종속성이 도입되지 않습니다.

결과 메타데이터는 모델 선택, 감지된 `weights_role` , 구성 선택 및 로컬 디렉토리, 선택적 VAE 파일 식별자, `model.loading` 하위의 `component_sources` 를 기록합니다. 컴포넌트 기원 라벨은 `model` , `model_config` , 및 `vae_override` 입니다. 최상위 `vae` 객체 또한 오버라이드되었는지 여부, 소스, 파일 식별자, 잠재 채널, 스케일링/이동 인자, 및 공간 다운샘플링 인자를 기록합니다. LoRA 식별자와 활성화는 별도의 `adapters` 배열에 유지됩니다. 명시적인 `--model` 은 기본 출력 줄에 `-custom` 을 추가하고, `--vae` 는 `-vae` 을 추가하며, `--lora` 는 `-lora` 를 순서대로 추가합니다. 결합된 SDXL 기본값은 `sdxl-base-red-cube-custom-vae-lora.png` 입니다. 이는 커스텀 실행을 정통 픽스처 픽스처와 분리하며, 기존 파일을 대체하는 경우에도 `--overwrite` 가 필요합니다. 출력 경로를 생략하면 충돌 시 사용되지 않은 번호가 매겨진 실행 이름을 선택합니다. 선택적 초기 잠재/임베딩 파일 포함 모든 생성 값은 [생성 매개변수](generation-parameters.md)에 설명됩니다.

<a id="boundaries"></a>

## 경계

이는 SDXL Refiner, FLUX.1-dev, 임의의 FLUX 파생물, 양자화 체크포인트, 계열 간 변환 또는 C++ 추론을 추가하지 않는다. 파일 출처는 학습 출처, 이미지 품질, 상업적 권리 또는 안전성을 입증하지 않는다. 선택한 각 체크포인트, 보조 모델, VAE와 LoRA는 독립된 라이선스 조건을 가진다.

기본 형식 어댑터는 다음 문서에 설명된 고정된 Diffusers API입니다.
[단일 파일 로드](https://huggingface.co/docs/diffusers/v0.40.0/api/loaders/single_file). 프로젝트 선택/구성은 `presets.py` 및 `model_loading.py`에 있습니다. 로컬 ID 및 임시 정식 경로는 `weight_files.py`에 있습니다.
