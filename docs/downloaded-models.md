<a id="downloaded-model-inspection"></a>

# 다운로드된 모델 검사

`reference/diffusers/downloaded_model.py` 는 생성 백엔드가 선택되기 전에 로컬 모델을 식별합니다. 검사기는 텐서 이름, 차원 및 모델 버전 메타데이터를 읽습니다. 파일명에서 모델 계열을 선택하지 않습니다.

```python
from downloaded_model import inspect_downloaded_model

identity = inspect_downloaded_model("/models/download.safetensors")
# 별도로 저장한 Civitai 모델 버전 응답도 받는다:
identity = inspect_downloaded_model(
    "/models/download.safetensors", info_path="/models/model-version.json"
)
```

결과는 JSON 호환 사전입니다.

|필드|의미|
| --- | --- |
| `format` |`safetensors`, `gguf` 또는 `pytorch` 컨테이너|
| `base_model` |메타데이터의 정확한 카탈로그 검증 Civita 카테고리 또는 null|
| `architecture` |Tensor 파생 아키텍처 또는 명시적으로 선언된 카테고리 패밀리|
| `preset`, `pipeline_class` |호환되는 초기 생성 경로 또는 경로가 설정되지 않은 경우 null|
| `role` |`checkpoint`, `lora`, `vae`, `embedding`, `controlnet`, 다른 선언된 구성요소 유형, 또는 `unknown`|
| `weights_role` |`denoiser` 생성 모델에 내장된 VAE 또는 텍스트 인코더 가중치에 대한 증거가 부족한 경우 그렇지 않으면 구성 요소 역할|
| `available_components`, `missing_components` |이 컨테이너에서 디노이저, VAE 및 텍스트 인코더 가중치가 발생한다는 증거입니다. 이는 완전한 구성 요소 무결성 검사가 아닙니다.|
| `prediction_type` |선언 시 명시적 엡실론, v-예측, 샘플 또는 흐름 매개변수화|
| `task` |필수 작업으로, 감지된 입력 채널/리파이너에 대한 `inpainting` 및 `image-to-image` 를 포함합니다.|
| `confidence` |`exact` 는 제공된 SHA256 가 일치함을 의미하며, `metadata` 는 해시 없이 카테고리가 선언됨을 의미하고, `architecture` 는 구조가 인식됨을 의미하며, `unknown` 는 라우트가 설정되지 않음을 의미합니다.|
| `evidence`, `role_guidance` |구성 요소에 대한 검사 기준 및 적절한 첨부 인수|

`exact` 는 제공된 메타데이터로 설명된 로컬 파일을 식별합니다. 메타데이터 발행자를 인증하거나 성공적인 생성을 증명하지 않습니다. VAE /text 인코더 텐서 이름의 존재가 모든 필수 가중치, 토크나이저 또는 스케줄러 구성이 포함되었음을 확립하기에 충분하지 않습니다.

공용 명령은 생성을 시작하지 않고 검사도 노출합니다 런타임 :

```sh
python3 reference/generate.py --inspect-model --model /models/download.safetensors
```

생성을 위해, `reference/generate.py --model /models/download.safetensors` 는 기존 가중치 파일에 대한 로컬 이미지 런타임 를 자동으로 선택합니다. `model_index.json` 를 포함하는 완전한 로컬 디렉토리는 Diffusers 를 선택합니다. `--model-config` 를 제공하는 것은 또한 일반적인 Diffusers 단일 파일 경로를 선택하며, 선택된 파이프라인은 실제로 `from_single_file` 를 구현해야 합니다. Kolors 예시에서 누락된 로더를 제공할 수 없습니다.
[generic-diffusers.md](generic-diffusers.md#kolors-and-task-specific-checkpoints) 가 설명합니다. 명시적인 `--backend`, `--preset`, `--pipeline-class` 및 `--workflow` 선택은 우선순위를 유지합니다. `--backend local` 는 로컬 런타임 를 직접 선택합니다. 로컬 구성 요소 및 런타임 옵션 (`--model-info`, `--components`, `--text-encoder*`, `--model-type`, `--runtime-*`) 또한 자동으로 선택합니다.

검사기는 SD1, SD2, SDXL base/refiner, SD3 및 FLUX.1 시그니처를 인식한다. SD1/2/XL 라우팅은 UNet 교차 어텐션 폭을 확인한다. FLUX dev/schnell 라우팅은 guidance embedding 가중치를 확인한다. FLUX 편집 변형, Chroma와 Flux2를 일반 FLUX.1 텍스트에서 이미지 생성 프리셋으로 아무런 알림 없이 라우팅하지 않는다. 원본 이미지가 필요한 것으로 식별된 작업은 여전히 호출자가 명시적인 워크플로를 제공해야 한다.

Illustrious, Pony, NoobAI 및 Krea 는 아키텍처가 확인된 후 카테고리별 라우팅 결정입니다. 그 이름들은 텐서 형식으로는 신뢰할 수 있게 기본 모델과 구별할 수 없습니다. NoobAI v-예측은 명시적인 예측 메타데이터 (`predictionType: "v_prediction"`, safetensors, `modelspec.prediction_type` 또는 `ss_v_parameterization: "true"`) 또는 명시적인 생성 프리셋이 필요합니다. 단순히 카테고리 NoobAI 를 이름만 지어도 그 훈련 파라미터화를 식별하지 못합니다.

<a id="civitai-model-version-metadata"></a>

## Civitai 모델 버전 메타데이터

검사관은 인접한 `download.civitai.info` 및 `download.safetensors.civitai.info` 이름을 자동으로 확인합니다. 둘 다 존재하는 경우 `info_path`를 사용하여 명시적으로 하나를 선택하십시오. 메타데이터는 저장된 Civitai 모델 버전 API 응답일 수 있습니다.

```json
{
  "baseModel": "Illustrious",
  "model": { "type": "Checkpoint" },
  "files": [
    {
      "name": "original-download.safetensors",
      "hashes": {
        "SHA256": "64 hexadecimal characters from the actual download"
      }
    }
  ]
}
```

SHA256 값이 존재할 때, 선택된 파일은 하나와 일치해야 합니다. 이는 로컬 이름 변경을 허용하면서 관련 없는 모델 버전 사이드카를 거부합니다. 유효하지 않은 해시, 알 수 없는 카테고리 이름, 호스팅된 카테고리 및 모델 유형, 아키텍처 또는 내장 메타데이터 간의 충돌은 모델 로딩 전에 에러를 유발합니다. 해시가 없는 저장된 JSON 는 사용 가능하지만 파일 신원은 명시적으로 검증되지 않았습니다. 네트워크 요청, 모델 다운로드 또는 원격 코드 실행은 검사에 포함되지 않습니다.

LoRA, LyCORIS/LoCon, VAE, 학습된 임베딩 및 ControlNet 파일은 구성 요소입니다. 이들은 첨부 지침을 반환하며 독립적인 체크포인트 파이프라인을 반환하지 않습니다. 예를 들어, LoRA 는 일치하는 기본 모델과 함께 `--lora` 에 속해야 합니다. Civitai `Checkpoint` 사이카는 선택된 파일이 LoRA 임을 나타내는 텐서 증거를 덮어쓸 수 없습니다. 알 수 없는 구성 요소 유형은 기본적으로 SD1 로 변경되지 않고 알 수 없는 상태로 유지됩니다.

<a id="containers-and-dependencies"></a>

## 컨테이너 및 종속성

Safetensors 검사에는 표준 라이브러리를 사용하여 최대 100 MB 의 JSON 헤더를 읽습니다. 이는 중복 키, 지원되는 데이터 타입, 형상, 바이트 범위, 중첩, 절단 및 인덱싱되지 않은 후미 바이트를 디코딩하지 않고 검증합니다. Safetensors 페이로드 해석은 생성 백엔드에서 사용되는 유지 관리 `safetensors` 및 Diffusers 패키지의 책임으로 남아 있습니다.

GGUF v2/v3 검사는 게시된 형식을 사용하여 한계가 설정된 메타데이터 및 텐서 설명자를 읽습니다. 이는 `general.architecture` 를 읽으며 형상 비교를 위해 GGUF 차원 순서를 반전시키고 정렬/데이터 오프셋을 확인합니다. 그것은 양자화된 페이로드를 디코딩하거나 유효성을 증명하지 않습니다. 유지 관리 `gguf` Python 패키지는 이 경계를 위해 평가되었습니다: 전체 리더는 텐서 데이터를 매핑하고 의존성 없는 검사 경로에 NumPy 를 가져옵니다. 따라서 양자화 로딩은 여기에 재구현되는 대신 선택된 백엔드로 위임됩니다. 빅엔디안 GGUF 및 인식되지 않은 형식/타입 수정은 호환 가능한 백엔드 또는 업데이트된 검사기가 필요합니다.

`.ckpt`, `.pt`, `.pth` 및 `.bin` 검사에는 안정적인 PyTorch 2.10.0 또는 그 이상의 런타임 가 필요하며, `torch.load(weights_only=True, map_location="meta")` 를 사용합니다. 지원되지 않는 객체는 오류를 발생시키며, 안전하지 않은 pickle 대체 경로 는 없습니다. PyTorch 가 누락되면 파일은 열리지 않으며 명시적으로 제공된 메타데이터만 경로를 설정할 수 있습니다. 생성 환경은 생성 전에 유지되는 PyTorch 로더와 모든 필요한 구성 요소를 제공해야 합니다.

공공 `reference/generate.py --inspect-model` 명령은 이러한 레거시 컨테이너에 대해 설치된 관리 런타임 를 자동으로 재사용하므로 호스트 Python 는 Torch 가 필요하지 않습니다. `--runtime-python /absolute/path/to/python` 는 다른 설치된 검사 환경을 선택합니다. 재실행은 해당 인터프리터가 이미 실행 중일 때 중지됩니다. Safetensors 와 GGUF 검사는 호스트의 표준 라이브러리 경로에 유지되며 관리 런타임 를 시작하지 않습니다.

모듈에는 새로운 필수 런타임 종속성이 없습니다. 회귀 테스트는 실제 바이너리 safetensors/GGUF 픽스처 컨테이너를 사용하고 충돌하는 메타데이터, 잘못된 해시, 잘못된 형식의 헤더 및 우발적인 구성 요소를 체크포인트 경로로 거부합니다.

```sh
python3 tests/DownloadedModelTests.py
```

테스트 컨테이너는 구문 분석 및 디스패치 정확성을 설정합니다. 여기에는 합성 텐서가 포함되어 있으며 전체 Civitai 체크포인트에 대한 시각적 품질이나 성공적인 추론을 설정하지 않습니다.

<a id="upstream-references"></a>

## 상위 공급 측 참고자료

- [Safetensor 형식 및 무결성 규칙](https://github.com/huggingface/safetensors/blob/main/README.md#format)
- [Diffusers 체크포인트 서명 및 단일 파일 로딩 구현](https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/single_file_utils.py)
- [ComfyUI 모델 감지](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_detection.py)
- [GGUF 파일 형식](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [유지보수된 GGUF Python 리더](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py)
