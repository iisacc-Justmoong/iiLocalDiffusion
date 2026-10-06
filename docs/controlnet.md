<a id="controlnet-generation-contract"></a>

# ControlNet 생성 계약

독립적인 Python 생성기는 선택 가능한 ControlNet 하나와 `sd15` , `sdxl-base` 또는 `flux1-schnell` 를 포함한 준비된 로컬 조건부 이미지를 하나 받습니다. `--controlnet` 를 생략하면 원래 생성 경로를 유지합니다. 기본적으로 ControlNet 모델 또는 이미지가 선택되지 않습니다. 이 문서는 독립적인 Python 경로를 설명합니다. 별도의 프로세스 내 C++ 고급 경로는 이제 최대 64 Canny/Tile ControlNets 를 SD 1.5 및 SDXL 에 연결하며, [네이티브 이미지 매개변수](image-parameters.md)를 참조하십시오.

<a id="generate-with-a-conditioning-image"></a>

## 컨디셔닝 이미지로 생성

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py --model /absolute/path/image-diffusers \
  --preset sd15 \
  --controlnet /absolute/path/to/sd15-controlnet-package \
  --control-image /absolute/path/to/prepared-canny.png \
  --controlnet-scale 0.8
```

이미지는 해당 모델이 예상하는 입력을 이미 나타내야 하며, 예를 들어 Canny 에지, 깊이 맵, 또는 포즈 가이드와 같은 것들입니다. 생성기는 하나의 정적 이미지를 디코딩하여 EXIF 방향을 적용하고, RGB 로 변환한 후 Diffusers 가 `--width` 와 `--height` 로 리사이즈하도록 합니다. 사진에서 에지, 깊이, 또는 포즈를 계산하지 않습니다. 애니메이션 이미지와 디코딩 실패는 거부됩니다. `--num-images` 내의 모든 이미지에 동일한 조건부 이미지가 사용됩니다.

|사전 설정|필수 ControlNet 구성 요소|파이프라인|내부적으로 사용되는 이미지 인수|
|---|---|---|---|
| `sd15` | `ControlNetModel` | `StableDiffusionControlNetPipeline` | `image` |
| `sdxl-base` | `ControlNetModel` | `StableDiffusionXLControlNetPipeline` | `image` |
| `flux1-schnell` | `FluxControlNetModel` | `FluxControlNetPipeline` | `control_image` |

클래스 정체성과 텐서 인터페이스 구성은 첨부 전에 선택된 베이스와 비교됩니다. SDXL ControlNet 는 SD 1.5 와 함께 사용될 수 없으며, 둘 다 `ControlNetModel` 를 사용한다고 해서 그렇습니다. 구조적으로 호환되는 FLUX ControlNet 는 여전히 Schnell 에 적합한 가중치가 필요합니다. 아키텍처를 수용한다고 해서 훈련 호환성이나 이미지 품질을 확립하지 않습니다.

<a id="arguments-and-defaults"></a>

## 인수 및 기본값

|인수|생략된 값 / 허용된 입력|
|---|---|
| `--controlnet` |비활성화됨; 그렇지 않으면 로컬 Diffusers 구성요소 디렉토리 또는 로컬 safetensors 파일|
| `--control-image` |ControlNet를 선택한 경우 필수입니다. 비어 있지 않은 로컬 정적 이미지|
| `--controlnet-config` |단일 파일 구성 요소 구성: 기본적으로 형제 `config.json`, 그렇지 않으면 명시적 로컬 디렉터리|
| `--controlnet-variant` |패키지 파일명 변형이 없으며, `fp16`와 같은 명시적 값은 `--weight-variant`와 독립적입니다.|
| `--controlnet-scale` / `--controlnet-conditioning-scale` |1.0를 선택하면; 유한하고 비음수인 0는 0의 컨디셔닝 강도를 제공합니다.|
| `--control-guidance-start` |0.0 선택 시|
| `--control-guidance-end` |선택 시1.0; `0 <= start < end <= 1`가 필요합니다|
| `--guess-mode` / `--no-guess-mode` |선택된 경우 False; 활성화하려면 SD / SDXL가 필요합니다|
| `--control-mode` |없음; FLUX Union 모델에 대해 해당 모델의 비음수 모드 인덱스를 사용하여 필요합니다|

가이드 시작/종료는 디노이징 단계의 분율을 지정합니다. 0 스케일은 선택된 ControlNet 를 계속 로드하고 실행하며, `--controlnet` 를 생략하여 기능을 비활성화합니다. 종속 제어는 명시적으로 기본값으로 설정되어 있더라도 `--controlnet` 를 필요로 합니다. FLUX 유니온 모드 ID 는 모델별로 특정되며 `num_mode` 와 비교됩니다. 유니온이 아닌 모델은 `--control-mode` 를 거부합니다.

<a id="weight-and-configuration-sources"></a>

## 무게 및 구성 소스

구성 요소 패키지에는 예상되는 ControlNet 클래스와 표준 Diffusers safetensors 가중치를 선언하는 루트 `config.json`가 있습니다. 기본 파이프라인 패키지와 ControlNet 구성 요소 패키지는 별도의 로컬 선택입니다. 호환되는 로컬 모델을 제공하십시오.

```sh
reference/diffusers/.venv/bin/python reference/diffusers/generate.py \
  --preset sdxl --model /absolute/path/sdxl-diffusers \
  --controlnet /absolute/path/sdxl-controlnet \
  --control-image /absolute/path/conditioning.png
```

로컬 단일 파일은 `.safetensors` 및 `.safetensor`를 허용합니다. 형제 `config.json`는 존재하는 경우 구성 요소 구성을 제공합니다. 그렇지 않으면 명시적인 로컬 `--controlnet-config`를 전달하세요. 구성 소스는 기본 파이프라인의 `model_index.json`가 아닌 루트에 해당 구성 요소 `config.json`를 포함해야 합니다.

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py --model /absolute/path/image-diffusers \
  --preset flux1-schnell \
  --controlnet /absolute/path/to/flux-controlnet.safetensor \
  --controlnet-config /absolute/path/to/flux-controlnet-config \
  --control-image /absolute/path/to/prepared-canny.png \
  --local-files-only
```

네이티브 Diffusers 텐서 이름은 모든 3 계열에 대해 허용됩니다. Diffusers 에는 구성된 캐시 하에 임시 컴포넌트 패키지로 제시되며, 정상적인 저메모리 로더와 구성 확인을 유지합니다. 지원되는 원본 형식 SD / SDXL ControlNet safetensors 는 `ControlNetModel.from_single_file` 에 위임됩니다. 원본 형식 FLUX ControlNet 변환은 제공되지 않습니다. Pickle `.ckpt`, `.bin`, 및 `.pt` 파일은 거부되며, 단일 안전 파일 확장명은 임시 정통 별칭을 사용합니다. 구성 옵션은 단일 파일에만 적용되며, 패키지 변형은 로컬 디렉토리에만 적용됩니다.

패키지의 경우 선택된 변형의 인덱스는 레거시 샤드 이름을 포함하여 어떤 로컬 safetensors 샤드가 선택되고 해시되는지 정확하게 결정합니다. 색인이 없으면 해당 변형의 표준 단일 가중치 파일만 사용됩니다. 동일한 로컬 디렉터리의 다른 변형은 로드되지 않습니다.

원본 형식 변환은 고정된 상위 공급 측 컨버터를 사용합니다. 그 네이티브 텐서 키는 로드된 컴포넌트의 상태 사전과 정확히 일치해야 하며, `--no-low-cpu-mem-usage` 가 선택된 경우에도 마찬가지입니다. 누락된 가중치는 무작위 초기화로 대체할 수 없으며, 예상치 못한 변환된 텐서는 거부됩니다.

ControlNet 중량 및 구성 요소 구성은 로컬에 존재해야 합니다. 로드에서는 항상 `local_files_only=True`를 사용합니다. 대체 경로가 없으면 누락된 파일이 실패합니다. 널이 아닌 베이스, ControlNet 및 구성 개정 인수는 거부됩니다.

<a id="json-python-and-composition"></a>

## JSON, Python 및 구성

CLI, `--config`, `--print-config`, 및 Python `resolve_request()` 는 동일한 스키마를 공유합니다. JSON 는 `controlnet_scale` 를 정통 스케일 키로 사용하며, 긴 철수는 CLI 별칭입니다. JSON 파일 내 경로는 해당 파일의 디렉토리에서 해결됩니다. 생략된 값과 JSON null 은 중립 기본값을 보존합니다. 예를 들어, `models/` 및 `inputs/` 디렉토리와 함께 배치된 구성은 다음을 포함할 수 있습니다:

```json
{
  "preset": "sd15",
  "controlnet": "./models/sd15-controlnet",
  "control_image": "./inputs/canny.png",
  "controlnet_scale": 0.8,
  "control_guidance_start": 0.0,
  "control_guidance_end": 1.0
}
```

Python 호출자는 `resolve_request(values)` 에 동일한 키를 전달할 수 있습니다. ControlNet 는 기존 베이스 모델, VAE, LoRA, 스케줄러, 초기 잠재 표현, 및 임베딩 입력과 함께 구성됩니다. 기본 검증은 ControlNet 애착 전에 수행되며, 선택적 LoRA 활성화 및 CPU 프롬프트 인코딩은 GPU 배치 또는 모델/순차 오프로드 전에 발생합니다. ControlNet 는 동일한 선택된 장치와 데이터 타입을 사용합니다. Diffusers /Accelerate 는 오프로드 후크를 소유합니다. 파이프라인 변환은 Diffusers '  `from_pipe`  기본값인 float32보다 명시적으로 요청된 dtype 을 유지합니다.

파이프라인들은 nonzero  `--guidance-rescale` 를 구현하지 않으므로, 아무런 알림 없이  무시되는 대신 거부됩니다. 기존 가족별 제한 사항이 여전히 적용됩니다.  `--guess-mode` 는  SD / SDXL  옵션이며,  FLUX  Union 은  `--control-mode` 대신 사용합니다. Multi- ControlNet , 자동 조건 감지기, 외부 이미지-이미지 입력, 그리고 인페인팅은 이 인터페이스의 범위를 벗어납니다.

선택된 [Hires Fix](hires-fix.md) 는 모든 3   ControlNet 경로에 적용됩니다. 선택된 ControlNet 와 준비된 조건부 이미지는 기본 패스와 두 번째 img2img 확산 패스 동안 계속 활성화됩니다. Diffusers 는 선택된 스케일, 가이드 간격, 및 가족별 모드로 각 단계의 해상도에서 해당 조건을 준비합니다. 확대된 기본 이미지는 img2img 입력이며, 별도의  ControlNet 조건을 대체하지 않습니다. LoRA , 대체  VAE ,  CPU 프롬프트 인코딩 및 배치화는 여전히 조합 가능합니다.

FLUX   ControlNets 는 입력 힌트 블록 샘플  VAE 잠재 표현을 조건부 이미지에서 샘플링하기 전에 초기 분화 노이즈를 샘플링합니다. 이는 구성된 이미지별 생성기들을 모두 소모하며, ControlNet 가 없는 생성과 동일한 노이즈 시퀀스를 의미하지 않습니다.  CPU 사전 계산된 프롬프트 임베딩은 기존 배치 확장 정책을 유지합니다.

<a id="output-identity-and-verification"></a>

## 출력 신원 및 검증

기본 출력 줄임말은  `-controlnet`  다음에  `-custom` ,  `-vae` , 및  `-lora`  수정자를 추가합니다.  SD   1.5   ControlNet -만 실행하는 경우 따라서  `build/reference/sd15-red-cube-controlnet.png` 에서 시작합니다. 충돌 처리 및 배치 번호 매기는 기존 출력 규칙을 따르며 이전 증거를 보존합니다. Hires Fix 는  `-hires` 를  `-controlnet` 다음에 추가하며; 선택적 저장 기본 이미지는 최종 이미지 줄임말에  `-base` 를 추가합니다. 두 단계의 출력 경로 및 사이더카는 충돌 검사에 참여합니다.

사이더카는  ControlNet 로컬 소스 및 파일 식별자, 모델 클래스, 구성 소스, 가중치/구성 파일 경로,  SHA-256 해시 및 바이트 크기, 패키지 변형, 스케일, 가이드 간격, 추측/연합 모드, 및 조건부 이미지의 식별자, 방향성 차원,  RGB 변환, 및 요청된 출력 크기를 기록합니다.  LoRA 활성화는 별도로 기록됩니다. 완전한 해결된 요청은  `parameters` 에서 재실행을 위해 유지됩니다.

회귀 커버리지는 요청 기본값 및 오류, 원격 소스 거부, 단일 파일 구성,  JSON / CLI 우선순위, 가족별 호출 인자, 파일 이름, 로딩 식별자, 및 호환성을 다룹니다. 작은 실제  Diffusers 스모크 실행은 이러한 요청 테스트와 별도로 조건부 및 생성을 연습합니다. 합성 가중치 또는 작은 파이프라인은 전체 모델 시각 품질, 임의의  ControlNets 적합성, 또는 모든 가속기에서의 작동을 확립하지 않습니다.

구현은 고정된 Diffusers 0.40.0, PyTorch, Accelerate, safetensors, Hugging Face Hub, 및 Pillow 의존성을 재사용하며, 런타임 감지기나 새로운 의존성이 도입되지 않습니다. 모델 가중치는 Diffusers 의 Apache-2.0 라이선스와 독립적으로 자체 용어를 유지합니다. 가중치가 번들되지 않습니다. API 계약은 상위 공급 측 에 의해 문서화됩니다.
[SD ControlNet API](https://huggingface.co/docs/diffusers/v0.40.0/en/api/pipelines/controlnet),
[SDXL ControlNet API](https://huggingface.co/docs/diffusers/v0.40.0/en/api/pipelines/controlnet_sdxl),
[FLUX ControlNet API](https://huggingface.co/docs/diffusers/v0.40.0/en/api/pipelines/controlnet_flux)및 [단일 파일 로더](https://huggingface.co/docs/diffusers/v0.40.0/en/api/loaders/single_file).
