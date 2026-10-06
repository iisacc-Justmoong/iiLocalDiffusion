<a id="generic-diffusers-generation"></a>

# 일반 Diffusers 생성

`reference/diffusers/generate_any.py` 는 설치된 Diffusers   런타임 에 내장된 공개 파이프라인 클래스를 실행합니다. 이는 완전한 Diffusers 디렉터리로 배포된 모델에 대한 3-프리셋 제한을 제거하면서, 더 구체적인 구성 요소 검증 및 생성 확장을 위한 기존 프리셋 오라클을 유지합니다. 현재 의존성 핀은 Diffusers   0.40.0입니다.

Civitai 기본 모델 라벨은 모델 가족을 식별하며, 범용 파일 형식이 아닙니다. 예를 들어, SDXL 에서 파생된 가족은 SDXL 파이프라인 아키텍처를 공유할 수 있지만, 다른 스케줄러, 예측 유형 또는 조건부 구성을 필요로 합니다. 일반 러너는 제공된 모델 구성과 실제 파이프라인 호출 서명을 사용합니다. `--base-model` 는 Civitai 카탈로그 라벨을 선택된 파이프라인 아키텍처와 대조합니다. SD/SDXL/SD3/Flux /Flux2/Qwen 작업 형제들은 명시적으로 그룹화되어 편집, 인페인팅 및 ControlNet 변형이 사용 가능하게 유지되도록 하며, 가중치 로딩 전에 가족 간 불일치가 거부됩니다. 다른 일반적인 파이프라인은 설치된 아키텍처 네임스페이스를 사용합니다. Diffusers 파이프라인 클래스가 없는 카탈로그 항목은 `declared-only` 로 표시됩니다. 알 수 없는 이름과 호스트 전용 라벨은 거부됩니다. 이는 파이프라인 아키텍처만 설정합니다: 아키텍처 내 SD1 와 SD2 를 구별하거나 특정 미세 조정 작업을 식별할 수 없습니다. 텐서 아키텍처를 변경하지도 않고 아무런 알림 없이 모델 가중치를 변환하지도 않습니다.

<a id="model-loading"></a>

## 모델 로딩

로컬 Diffusers 디렉터리에는 `model_index.json`·구성요소 설정·토크나이저·필수 모델 가중치 전체가 있어야 한다. 선언된 파이프라인 클래스는 자동으로 선택하며, `--pipeline-class`로 이미지 편집이나 인페인팅 등 설치된 다른 호환 작업 파이프라인을 명시적으로 선택할 수 있다. 공개 내장 Diffusers 파이프라인 클래스와 Diffusers/Transformers 구성요소만 허용한다. 사용자 지정 파이프라인 스크립트·임의 가져오기 경로·동적 원격 코드는 제외한다.

```sh
reference/diffusers/.venv/bin/python reference/diffusers/generate_any.py \
  --model /Volumes/Storage/Workspace/Models/sd35-medium \
  --pipeline-class StableDiffusion3Pipeline \
  --prompt 'a glass observatory above a cloud sea' \
  --width 1024 --height 1024 --steps 28 --guidance-scale 4.5 \
  --seed 42 --dtype bfloat16 --device cuda --offload model \
  --output-dir build/reference/sd35-run
```

`--model`는 기존 로컬 경로여야 합니다. 허브 ID, URL 및 null이 아닌 개정은 거부됩니다. 다운로드 단계가 없습니다. 모델 로드는 항상 `local_files_only=True`, `use_safetensors=True` 및 `trust_remote_code=False`를 사용합니다. [로컬 모델 생성](local-model-generation.md)를 참조하세요.

로컬 `.safetensors` 체크포인트는 `--model-config` 가 **로컬** Diffusers 구성 디렉토리를 가리키고 체크포인트에 없는 `model_index.json`, 토크나이저 및 모든 구성 요소를 포함해야 합니다. 선택된 클래스는 Diffusers ' `from_single_file` 을 구현해야 합니다. 파이프라인의 디렉토리 형식을 지원하는 것이 모든 단일 파일 형식을 로드한다는 것을 의미하지는 않습니다. 특히, 트랜스포머 전용 파일은 필요한 VAE 와 텍스트 인코더가 없으면 완전한 파이프라인이 되지 않습니다. 클래스에 단일 파일 로더가 없으면 완전한 Diffusers 디렉토리를 제공하거나 워크플로 백엔드를 사용해야 합니다.

상위 `reference/generate.py` 는 `--model-config` 가 있는 경우, 기존 다운로드된 체크포인트 `--model` 가 이름으로 지정된 경우를 포함하여 이 일반 러너를 자동으로 선택합니다. 명시적인 `--backend` 와 `--preset` 선택은 우선순위를 유지합니다. `--pipeline-class` 와 `--pipeline-inputs` 도 자동 로컬 파일 경로 전에 일반 러너를 선택하므로 이미지 투 이미지, 인페인팅 및 리파이너 입력이 요청된 작업 파이프라인에 도달합니다.

```sh
reference/diffusers/.venv/bin/python reference/diffusers/generate_any.py \
  --model /Volumes/Storage/Workspace/Models/model.safetensors \
  --model-config /Volumes/Storage/Workspace/Models/model-config \
  --pipeline-class StableDiffusion3Pipeline \
  --prompt 'a small architectural model in daylight' \
  --device mps --dtype float32 \
  --output-dir build/reference/single-file-run
```

체크포인트 `.ckpt` , 피클 `.pt` / `.pth` / `.bin` , GGUF , 및 커스텀 Python 모델 패키지는 이 러너에 의해 읽히지 않습니다. 단일 파일 구성 추가 항목은 피클 형식 가중치를 포함해서는 안 됩니다. Diffusers 의 0.40 내부 단일 파일 추가 항목 로더는 `use_safetensors` 를 모든 컴포넌트로 전달하지 않으므로 해당 파일들은 명시적으로 거부됩니다. 양자화된 GGUF 모델과 설치된 Diffusers 에 의해 구현되지 않은 계열의 경우, 일치하는 로컬 워크플로우/ 런타임 와 함께 상위 워크플로우 백엔드를 사용하세요.

<a id="kolors-and-task-specific-checkpoints"></a>

## Kolors 및 작업별 체크포인트

Kolors 는 Kolors UNet , VAE , ChatGLM 텍스트 인코더, 토크나이저 및 스케줄러를 갖춘 완전한 로컬 Diffusers 디렉터리로 지원됩니다. 공식 배포판은 내장 `KolorsPipeline` 를 식별하며, 내장 `KolorsImg2ImgPipeline` 는 이미지 투 이미지 생성을 위해 해당 컴포넌트를 재사용할 수 있습니다.

```sh
reference/diffusers/.venv/bin/python reference/generate.py \
  --model /Volumes/Storage/Workspace/Models/Kolors-diffusers \
  --base-model Kolors --prompt 'a glass observatory above a cloud sea' \
  --width 1024 --height 1024 --steps 50 --guidance-scale 5 \
  --device cuda --dtype float16 --offload model --local-files-only \
  --output-dir build/reference/kolors-image
```

**다운로드된 Kolors 단일 파일 체크포인트는 이 일반 러너에 의해 로드 가능하다고 설정되지 않습니다.** 설치된 Diffusers 0.40.0 와 공식 파이프라인 소스에 대한 검사 결과, Kolors 파이프라인 중 어느 것도 `from_single_file` 를 갖지 않는다는 것이 확인됩니다. `--model-config` 는 해당 누락된 로더를 추가할 수 없습니다. `UNet2DConditionModel` 는 컴포넌트 수준의 단일 파일 로더를 노출하지만, 그것만으로는 Kolors 체크포인트의 투영 가중치, ChatGLM 조건부, VAE 및 토크나이저의 변환을 설정하지 않습니다. 이러한 파일은 Diffusers 디렉토리로의 완전한 변환 또는 호환 가능한 명시적인 워크플로우를 필요로 합니다. 런너는 선택된 체크포인트를 무시하면서 구성 디렉토리의 기본 가중치를 로드하는 대신 구체적인 에러를 발생시킵니다. 이 검사 동안 완전한 Kolors 모델이 다운로드되거나 생성되지 않았습니다.

단일 파일 로드를 구현하는 파이프라인의 경우 작업과 일치하는 구성과 필수 입력을 제공합니다. 이 예에서는 SDXL 인페인팅 체크포인트와 로컬 마스크를 사용합니다.

```sh
reference/diffusers/.venv/bin/python reference/generate.py \
  --model /Volumes/Storage/Workspace/Models/sdxl-inpaint.safetensors \
  --model-config /Volumes/Storage/Workspace/Models/sdxl-inpaint-config \
  --pipeline-class StableDiffusionXLInpaintPipeline \
  --prompt 'a copper dome above the observatory' \
  --pipeline-inputs '{"image":{"image_path":"/Volumes/Storage/Workspace/Assets/source.png"},"mask_image":{"image_path":"/Volumes/Storage/Workspace/Assets/mask.png","mode":"L"},"strength":0.8}' \
  --device mps --dtype float32 --local-files-only \
  --output-dir build/reference/inpaint-image
```

SDXL 이미지-이미지 또는 리파이너 체크포인트의 경우 `StableDiffusionXLImg2ImgPipeline`, 해당 체크포인트 구성, `image` 입력, `strength` 를 사용하며 그 클래스는 마스크가 필요하지 않습니다. Kolors 이미지-이미지의 경우 `KolorsImg2ImgPipeline` 를 완전한 Kolors 디렉토리와 동일한 타입의 이미지 입력 문법으로 선택합니다. 이러한 라우팅 예시는 임의의 다운로드된 가중치나 작업별 시각적 품질을 인증하지 않습니다.

출처: [official Kolors Diffusers 사용법](https://github.com/Kwai-Kolors/Kolors#using-with-diffusers),
[Kolors 파이프라인 구현](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/kolors/pipeline_kolors.py),
[Kolors 이미지 대 이미지 구현](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/kolors/pipeline_kolors_img2img.py),
[단일 파일 모델 변환](https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/single_file_utils.py).

<a id="pipeline-inputs-and-defaults"></a>

## 파이프라인 입력 및 기본값

`--pipeline-inputs` 는 JSON 객체 또는 `@/absolute/path/inputs.json` 를 받습니다. 객체는 선택된 파이프라인의 `__call__` 에 명시적인 인수로 전달됩니다. 각 제공된 키는 해당 서명에서 이름으로 나타날 수 있어야 하며, 제한되지 않은 `**kwargs` 파라미터는 알 수 없는 값의 침묵스러운 수락을 허용하지 않습니다. 필수 입력은 모델 가중치가 로드되기 전에 확인됩니다. 중복 JSON 키, 비유한 JSON 숫자, JSON 생성자 객체, 타입이 지정되지 않은 잠재 출력 요청, `return_dict=false` 는 거부됩니다.

`--prompt`, `--negative-prompt`, `--width`, `--height`, `--steps`, `--guidance-scale` 는 선택 사항입니다. 제공된 경우 해당 JSON 값을 덮어쓰며, `--steps` 는 `num_inference_steps` 에 매핑됩니다. 제외된 경우 런너는 각 파이프라인의 기본값을 그대로 둡니다. 명시적인 빈 프롬프트와 가이드 `0` 는 유지됩니다. `--seed` 는  CPU   `torch.Generator` 를 생성하며, 생략은 파이프라인의 확률적 기본값을 보존합니다. `--print-config` 는 로컬 경로를 해결하고  JSON 를 가져오지 않고  Torch ,  Diffusers ,  Pillow , 또는  NumPy 를 가져오지 않으며, 가중치를 생성하거나 다운로드하지 않습니다. 파이프라인 호환성을 증명하지 않습니다.

비디오, 오디오, 편집 및 모델별 옵션은 JSON 개체에 속합니다. 예를 들어 비디오 파이프라인은 일반적으로 `num_frames`를 허용하고 편집 파이프라인은 `strength`를 허용할 수 있습니다. 이는 파이프라인별 이름의 예이며 모든 클래스가 해당 이름을 허용한다는 약속은 아닙니다.

```json
{
  "prompt": "a slow orbit around a ceramic sculpture",
  "num_frames": 17,
  "image": {"image_path": "/Volumes/Storage/Workspace/Assets/sculpture.png"}
}
```

이미지를 포함하는 인자는  `{"image_path":"/absolute/file.png","mode":"RGB"}` 형식의 로컬 타입 객체를 지원합니다. 모드는  `RGB` (기본값),  `RGBA` , 또는  `L` 입니다; 마스크는  `L` 를 요청할 수 있습니다. 지원되는 인자 이름은  `image` ,  `images` ,  `mask_image` ,  `control_image` ,  `control_images` ,  `reference_image` ,  `reference_images` ,  `ip_adapter_image` ,  `conditioning_image` ,  `video` ,  `frames` ,  `last_image` ,  `start_image` ,  `end_image` , 그리고  `image_2` 에서  `image_4` 까지입니다. 리스트는 이미지 배치나 명시적인 비디오 프레임을 나타낼 수 있습니다. 입력 파일은 디코딩 전에 해시되며 이후 다시 확인됩니다. 애니메이션 이미지 파일은 명시적인 프레임 목록이 필요합니다. 텐서 변환은 명시적인  safetensors 설명자만 사용하며,  Python 콜백, 직렬화된 생성자, 사용자 코드는  JSON 를 통해 허용되지 않습니다.

<a id="diffusion-flow-autoregressive-and-hybrid-interchange"></a>

## 확산, 흐름, 자동 회귀 및 하이브리드 교환

`--generation-architecture` 는  `diffusion` ,  `rectified-flow` ,  `flow-matching` ,  `autoregressive` ,  `hybrid` ,  `auto` , 또는  `unspecified` (기본값) 를 받습니다. 이것은 출처선언이며, 모델 변환이나 스케줄러 오버라이드가 아닙니다. `auto` 와 `unspecified` 은 `not-inferred` 로 기록됩니다: 런너는 클래스 이름에서 훈련 목표를 추측하지 않습니다. 실행은 설치된 Diffusers 파이프라인과 그 설정을 계속 사용합니다. Transformers 전용 모델은 여전히 별도의 로드된 모델 생성 어댑터가 필요합니다; 여기에서 `autoregressive` 를 선택한다고 해서 Diffusers 로더가 생성되지 않습니다.

숫자 상태는 픽셀이나 부동 소수점 토큰 ID로 변환되지 않고 생성 단계를 교차할 수 있습니다. 목록이나 중첩된 개체 내부를 포함하여 명명된 파이프라인 입력에 명시적 설명자를 사용합니다.

```json
{
  "image": {
    "tensor_path": "/absolute/previous/images.safetensors",
    "key": "images",
    "semantic": "latents",
    "layout": "BSC",
    "representation_space": "my-model-packed-vae-v1",
    "dtype": "bfloat16",
    "shape": [1, 256, 64],
    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  }
}
```

첫 5 속성은 필수입니다. `dtype`, `shape`, 그리고 `sha256` 은 선택 사항이며 제공될 때 엄격한 동일성 검사가 됩니다. 위 해시는 예시이며 실제 파일 해시로 교체되어야 합니다. 비어 있지 않은 로컬 `.safetensors` 파일과 정확한 선택된 키만 읽힙니다; 피클이나 전체 파일 텐서 로딩은 발생하지 않습니다. 파일 식별자는 읽기 전과 후에 확인되며, 텐서는 소유된 CPU 저장소에 복제됩니다. 정수와 부동소수점 데이터 타입, bfloat16포함은 `--dtype` 로 캐스팅 없이 유지됩니다. 부동 값은 유한해야 합니다. 레이아웃은 차원마다 대문자 축 문자 하나를 사용하며, BCHW 또는 BSC 와 같이; 그 랭크는 텐서와 일치해야 합니다. 파이프라인은 후속 장치 배치, dtype 요구 사항, 및 모델별 형식 검증을 소유합니다.

`semantic` 값들은 `latents`, `embeddings`, `token-ids`, `logits`, `continuous-state`, `noise`, `epsilon`, `v-prediction`, `velocity`, `attention-mask`, 및 `tensor` 입니다. 확산 에파손과 v-예측은 유속 필드와 구별되며, 이 예측 매개변수화 중 어느 것도 자동으로 변환되지 않습니다. 잠재 표현, 임베딩, 로그 확률, 연속 상태, 노이즈, 및 예측 필드는 부동 소수점 값을 요구합니다. 토큰 ID 는 음이 아닌 정수 S 또는 BS 텐서를 요구합니다. 표현 레이블은 호출자의 정확한 토크나이저, VAE, 임베딩, 또는 상태 공간을 식별하며, 일치하는 레이블은 모델 호환성을 독립적으로 증명하지 않습니다. 특히, 소음 제거된 잠재는 자동으로 유효한 초기 노이즈 잠재가 아니며, 언패킹, VAE 스케일링, 디코딩, 토크나이징, 또는 학습된 투영은 결코 추론되지 않습니다.

`--tensor-outputs`를 통해 텐서 출력 필드를 선언하고 JSON 객체 또는 `@JSON-file`를 허용합니다. 요청된 모든 필드는 실제로 반환되어야 합니다. 잠재 요청에는 명시적인 잠재 계약이 필요합니다. 반환된 잠재 보유 필드는 픽셀 인코더를 통과할 수 없습니다.

```sh
reference/diffusers/.venv/bin/python reference/diffusers/generate_any.py \
  --model /absolute/flux-directory --prompt 'a red cube' \
  --generation-architecture rectified-flow \
  --pipeline-inputs '{"output_type":"latent"}' \
  --tensor-outputs '{"images":{"semantic":"latents","layout":"BSC","representation_space":"my-model-packed-vae-v1"}}' \
  --device cpu --output-dir build/reference/flux-latents
```

출력은 `images.safetensors` 이며, `images` 키 아래의 정확한 텐서입니다. 그의 아티팩트 엔트리는 `generation.json` 에 포함되어 있으며, 다음 호환성 파이프라인 입력으로 복사될 수 있는 완전한 설명자 `tensor_input` 를 포함합니다. 이것은 dtype, shape, layout, semantic, representation label, 및 SHA-256 을 유지합니다. The safetensors 헤더 또한 semantic, layout, representation label, output field, 및 iiLocalDiffusion 스키마 마커를 기록합니다. 이 메타데이터로 텐서를 읽는 것은 모든 선언을 저장된 값과 비교하고 재표시된 잠재 표현 또는 semantics 를 거부합니다. 이 프로듀서 메타데이터가 없는 외부 텐서는 호출자의 명시적 설명자를 사용하며, 누락된 메타데이터는 추론되지 않습니다.

이름이 `sequences` 와 `token_ids` 로 지정된 출력은 필드당 하나의 safetensors 아티팩트를 자동으로 생성하여, 개별 토큰 값을 반복하는 대신 그들의 완전한 S 또는 BS 텐서를 유지합니다. 그들의 기본 표현 레이블은 `unspecified` 입니다; `--tensor-outputs` 를 `token-ids` 의미론으로 선언하여 정확한 토크나이저 또는 코드북 공간을 기록합니다. `text` 와 `texts` 은 단일 UTF-8 문자열 또는 문자열의 일괄 처리를 받습니다. 이 필드들과 명시적으로 선언된 숫자 텐서는 반환된 이미지, 비디오, 오디오 필드와 함께 저장되므로, 혼합된 출력은 동일한 생성 보고서에 유지됩니다. 프로세스는 다른 모델을 자동으로 연결하거나 토큰 ID 를 텍스트로 해석하지 않습니다. 불투명한 `past_key_values` , 캐시 객체 및 기타 모델별 런타임 상태는 직렬화되지 않습니다. 그들은 `--tensor-outputs` 를 통해 요청할 수 없으며, 지원된 필드와 함께 존재하는 경우 무시됩니다.

구현은 기존 Torch 와 safetensors 의존성을 재사용하며, 새로운 런타임 의존성이 필요하지 않습니다. 상위 공급 측 프로젝트는 적극적으로 유지 관리되며 BSD 스타일 ( PyTorch ) 과 Apache-2.0 ( safetensors ) 라이선스를 사용합니다. 이 인터치레이어는 그들의 텐서 저장소나 생성 구현을 대체하지 않습니다.

공식 참조: [FLUX 파이프라인 입력 및 잠상 처리](https://huggingface.co/docs/diffusers/api/pipelines/flux),
[흐름 일치 스케줄러 입력](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete),
[Transformers 생성 시퀀스 및 모델별 캐시](https://huggingface.co/docs/transformers/internal/generation_utils),
[safetensors 안전 텐서 로딩](https://huggingface.co/docs/safetensors/index),
[PyTorch 라이센스](https://github.com/pytorch/pytorch/blob/main/LICENSE),
[safetensors 라이센스](https://github.com/huggingface/safetensors/blob/main/LICENSE).

<a id="hardware-and-artifacts"></a>

## 하드웨어 및 아티팩트

`--device auto` 는 공유 가속기 정책을 사용하며:  CUDA /ROCm 또는 Apple  MPS 가 사용 가능해야 합니다. 그것은 결코  아무런 알림 없이  CPU 로 되돌아가지 않습니다.  `--device cpu` 는 명시적으로  CPU 생성을 활성화합니다.  `--device metal` 는  MPS 의 별칭이며,  `--device rocm` 는  HIP -활성화된  PyTorch   런타임 를 필요로 합니다. 런너는 공유 산술  사전 검사 를 실행하며 생성 전후로 파이프라인 실행 장치를 확인합니다.

`--dtype` 는 기본값으로  `float32` 입니다;  `float16` 와  `bfloat16` 는 하드웨어 및 모델에 따라 사용 가능 여부와 수치적 동작이 명시적인 대안입니다.  `--offload` 는 기본값으로  `none` 입니다;  `model` 와  `sequential` 는 선택된 실행 장치로  Diffusers ' 내장 오프로드 방법을 호출합니다.  CPU 오프로드는 명시적인  CPU 생성에 대해 거부됩니다.

출력 디렉터리에는 첫 번째 배치 요소뿐 아니라 생성된 모든 항목이 포함됩니다.

|파이프라인 출력|아티팩트|
| --- | --- |
| `images` | `image-0001.png`, `image-0002.png`, ... |
| `frames` | `video-0001/frame-000001.png`, ... |
|`audios` 또는 `audio`|`audio-0001.wav` , ...  PCM16 로|
|명시적인  `--tensor-outputs` 필드|`<field>.safetensors` 및  `generation.json` 의 재사용 가능한 설명자|
|`sequences` 또는 `token_ids`|`sequences.safetensors` 또는  `token_ids.safetensors` , 정수 데이터 타입을 보존하며|
|`text` 또는 `texts`|`text-0001.txt` / `texts-0001.txt`, ... UTF-8|

`[0, 1]` 에서 지원되는  Pillow 이미지, 디코딩된 uint8 배열, 및 유한한 디코딩된 부동소수점 픽셀입니다. Torch 출력은  CPU 로 이동되어 내보내집니다. 비디오는 프레임 시퀀스이며, 코덱이나 추측된 재생 프레임 속도가 부과되지 않습니다. NumPy 비디오 출력은 기본값으로  `[batch, frames, height, width, channels]` ( BFHWC )이며,  Torch 출력은 기본값으로  Diffusers 의  VideoProcessor  `[batch, frames, channels, height, width]` ( BFCHW )입니다. `--video-layout bfhwc|bfchw|bcfhw` 는 명시적으로 다른 레이아웃을 선택합니다. 런너는 작은 프레임 수에서 프레임/채널 축을 추측하지 않습니다. 오디오 샘플 레이트는 실제 파이프라인의 vocoder/VAE/config 에서 읽거나 `--audio-sample-rate` 에 의해 제공되며, 누락된 샘플 레이트는 런너가 추측하지 않습니다. 오디오 부동소수점 진폭은 `[-1, 1]` 로 인코딩될 때 PCM16 로 잘립니다. 지원되지 않는 미디어, 텍스트, 토큰 또는 명시적으로 선언된 텐서 출력이 없는 결과는 실패하며, 메시와 임의의 객체는 생성된 미디어로 표현되지 않습니다.

`generation.json` 는 요청된 입력, 동일성 레이블, 선택된 및 실제 클래스, 컴포넌트 파일 SHA-256 해시와 바이트 크기, 단일 체크포인트 동일성, 디코딩된 입력 동일성, 모든 출력 해시, 의존성 버전, 장치 사전 검사 , 데이터 타입, 오프로드 모드, 아키텍처 레이블 유효성 검사 및 제공된 경우 실제 안전 검사 결과 필드를 기록합니다. 누락된 안전 결과는 여전히 누락되며, 검사기 누락은 성공적인 스크리닝으로 표현되지 않습니다. 이미지 레벨 NSFW 플래그는 이미지 배치 길이와 일치해야 합니다. 모델 동일성은 로딩 후와 생성 후 다시 확인됩니다. 각 출력 파일은 원자적으로 게시됩니다. 기존 비어 있지 않은 출력 디렉토리는 `--overwrite` 가 명시적이지 않은 경우 거부되며, 그렇지 않더라도 관련 없는 파일은 유지됩니다. 어떤 미디어를 대체하기 바로 전에, 기존 `generation.json` 는 고유한 `generation.previous-*.json` 히스토리 파일로 원자적으로 이동됩니다. 모든 아티팩트와 결과 메타데이터 검사가 성공한 후에만 새로운 현재 보고서가 나타납니다. 내보내기 중 실패는 완료된 미디어에 현재 보고서가 없게 남을 수 있으므로, 소비자는 작업을 완료로 간주하기 전에 유효한 `generation.json` 를 요구해야 합니다. 아카이브된 리포트는 이전 미디어 바이트를 보존하지 않고 오래된 기원을 보존하며, 해당 해시값은 이후 경로가 대체되더라도 이전 파일을 식별합니다.

<a id="validation-scope"></a>

## 검증 범위

다음을 사용하여 종속성 없는 계약 제품군을 실행하세요.

```sh
python3 tests/GenericDiffusersTests.py
python3 tests/GenericIOTests.py
reference/diffusers/.venv/bin/python tests/GenericIOTests.py
```

의존성 없는 테스트는 중립적 생략, 명시적 오버라이드, 원격 모델 소스 거부, 안전한 구성 요소 선택, 단일 파일/구성 경계, 단일 파일 로더가 없는 파이프라인일 때의 거부, Kolors 작업 계열 네임스페이스 유효성 검사, 알 수 없는 인자 거부, 입력 미디어 디코딩 경계, 파일 변경 감지, 완전한 이미지/비디오 배치, 샘플 레이트 해상도, 하드웨어/오프로드 확인, 덮어쓰지 않는 게시, 부분적 오버라이드 실패 주입, 유지된 안전 플래그, 아키텍처 레이블 거부, 및 짧은 비디오 프레임/채널 축 처리를 포함합니다. 일반적인 I/O 계약 테스트는 명시적 텐서 지정, 타입화된 잠재적 게이트, 혼합 텍스트/미디어 출력, 및 불투명한 캐시 거부를 추가로 확인합니다. Torch 과 safetensors 가 설치되어 있으면, 런타임 테스트는 실제 bfloat16 safetensors 라운드 트립, 2 ^ 53보다 큰 int64 ID, 정확한 키/해시/타입/형식 실패, 파일 변경 감지, 유효하지 않은 토큰 및 잠재적 값, 및 원자적 텐서 게시를 확인합니다. 런타임 전용 사례는 의존성 없는 호스트 인터프리터에서 건너뛰고 기존 참조 환경에서 실행됩니다.

검증된 작은 FLUX CPU 스모크는 이미 캐싱된 `hf-internal-testing/tiny-flux-pipe` 스냅샷을 커밋 `a98bc4ec6a80c47c477ed22d7e81c79872d9c723`, 64×64, 한 단계, 시드 42, 가이드 0, 및 `max_sequence_length=64` 에서 사용했습니다. 그 리포트는 `build/reference/generic-diffusers-smoke/generation.json` 에서 정확한 테스트된 파일과 불균일한 64×64 RGB 출력을 식별합니다. 두 번째 실제 CPU 생성은 프롬프트 없이 로컬로 초기화된 작은 `DDPMPipeline` 를 사용하여 2-이미지 NumPy 배치 를 내보냈으며, 그 보고서는 `build/reference/generic-media-smoke/ddpm-generation/generation.json` 입니다. 합성 미디어 내보내기는 Torch uint8 이미지, 채널 우선 비디오 텐서, 그리고 입체 PCM16 WAV 헤더를 `build/reference/generic-media-smoke/encoding-validation.json` 에서 별도로 검증하며, 이는 인코딩 증거이지 비디오/오디오 모델 생성 증거가 아닙니다. 세 번째 실제 CPU 생성은 로컬로 초기화된 작은 `StableDiffusion3Pipeline` 에 `SD3Transformer2DModel`, 2 작은 CLIP 투영 인코더와 캐시된 작은 T5/VAE /토크나이저를 사용하여 2 32×32 이미지를 생성했습니다. 그 빌더는 `build/reference/generic-sd3-smoke/run.py` 이며 보고서는 `build/reference/generic-sd3-smoke/generation/generation.json` 입니다; 큰 모델이 다운로드되지 않았습니다. 작은 모델은 러너 실행만 설정하며, 전체 크기 모델 품질, 모든 패밀리 호환성, 라이선싱 또는 GPU 성능을 설정하지 않습니다. 계약 테스트를 위해 큰 모델 다운로드가 필요하지 않습니다.

일반적인 텐서 I/O 스모크는 기존 로컬 작은 가이드 활성화 `FluxPipeline`, CPU 부동소수점32, 64×64, 한 단계, 시드 42, 가이드 1, 그리고 `max_sequence_length=16` 를 사용했습니다. 그것은 `build/reference/generic-io-smoke/latent/generation.json` 에서 실제 `[1, 1024, 4]` 잠재 출력을 생성했습니다. 그 방출된 설명자는 이후 일반적인 일반 입력 경로를 통해 로드되어 저장된 텐서와 정확히 비교되었으며, `build/reference/generic-io-smoke/round-trip-validation.json` 레코드는 데이터 형식, 형상, SHA-256, 그리고 비트 단위 동등성을 포함합니다. 이는 이러한 정확한 작은 가중치에 대한 로컬 생성과 손실 없는 상호 교환을 증명하며, 디노이즈된 잠재가 다른 모델이나 초기 노이즈와 상호 교환 가능하다고 주장하지 않습니다.

설치된 파이프라인은 실행 가능한 모델 지원을 결정합니다. 공식 [Diffusers 파이프라인 개요](https://huggingface.co/docs/diffusers/api/pipelines/overview) 와 [단일 파일 로더 문서](https://huggingface.co/docs/diffusers/api/loaders/single_file)를 참조하세요. Civitai 카탈로그 경로 및 유효한 CLI 구성은 호환성 경로이며, 해당 계열의 모든 체크포인트가 성공적으로 생성되었다는 증거가 아닙니다.
