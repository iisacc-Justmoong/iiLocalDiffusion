<a id="generic-diffusers-generation"></a>

# 일반 Diffusers 생성

`reference/diffusers/generate_any.py` 는 설치된 Diffusers 런타임에 내장된 공개 파이프라인과 프로젝트에 고정된 MOSS-Image 파이프라인을 실행한다. 이는 완전한 Diffusers 디렉터리로 배포된 모델에 대한 3-프리셋 제한을 제거하면서, 더 구체적인 구성 요소 검증 및 생성 확장을 위한 기존 프리셋 오라클을 유지한다. 현재 의존성 핀은 Diffusers 0.40.0이다.

고정된 Krea 2, FLUX.1 및 MOSS 리소스와 기존 전처리기를 결합하는 명시적 호출 예제는 [로컬 ControlNet 리소스 실행 조합](controlnet-resource-recipes.md)을 참조한다.

공통 런처는 내장 MOSS 이미지 패키지를 C++ 구성 요소 카탈로그와 원본 텐서 서명으로 Diffusers에 연결한다. 원본 샤드 인덱스와 체크섬을 확인해 변환기와 필요한 인코더·VAE만 캐시에 준비하며, 사용자가 CLI나 설정 파일에서 지정한 구성 요소를 우선한다. 여러 후보가 있는 패키지에서는 명시된 `main` 또는 `--package-model`로 변환기를 고르고, 중복 인코더나 VAE는 `--package-components` 또는 외부 경로로 지정한다. 결과 기록에는 원본 패키지와 준비된 변환기 및 명시된 구성 요소를 남긴다. 자동과 명시적 Diffusers 경로 모두 `--config`를 지원하고 CLI 옵션이 우선한다. `--inspect-model`은 가중치 복원 없이 구성 요소를 보고하고, 패키지 `--print-config`는 복원을 미룬다는 안내를 반환한다. 합성 패키지와 대체 엔진 테스트는 사전 학습 모델의 실제 생성 검증이 아니다.

Diffusers의 JSON 설정은 모델 및 로컬 인코더 경로를 설정 파일의 위치를 기준으로 해석한다.
`pipeline_inputs`와 `tensor_outputs`는 JSON 객체로, `sigmas`는 배열로 지정할 수 있다.
해당 백엔드에 없는 옵션이나 표현할 수 없는 불리언 값은 무시하지 않고 거부한다.
MOSS 수학·런타임 합성 테스트는 임시 테스트 모듈만 복원하여 같은 프로세스에서
PyTorch의 지연 로딩 모듈과 네이티브 연산자 등록이 유지되도록 한다.

Civitai 기본 모델 라벨은 모델 가족을 식별하며, 범용 파일 형식이 아닙니다. 예를 들어, SDXL 에서 파생된 가족은 SDXL 파이프라인 아키텍처를 공유할 수 있지만, 다른 스케줄러, 예측 유형 또는 조건부 구성을 필요로 합니다. 일반 러너는 제공된 모델 구성과 실제 파이프라인 호출 서명을 사용합니다. `--base-model` 는 Civitai 카탈로그 라벨을 선택된 파이프라인 아키텍처와 대조합니다. SD/SDXL/SD3/Flux /Flux2/Qwen 작업 형제들은 명시적으로 그룹화되어 편집, 인페인팅 및 ControlNet 변형이 사용 가능하게 유지되도록 하며, 가중치 로딩 전에 가족 간 불일치가 거부됩니다. 다른 일반적인 파이프라인은 설치된 아키텍처 네임스페이스를 사용합니다. Diffusers 파이프라인 클래스가 없는 카탈로그 항목은 `declared-only` 로 표시됩니다. 알 수 없는 이름과 호스트 전용 라벨은 거부됩니다. 이는 파이프라인 아키텍처만 설정합니다: 아키텍처 내 SD1 와 SD2 를 구별하거나 특정 미세 조정 작업을 식별할 수 없습니다. 텐서 아키텍처를 변경하지도 않고 아무런 알림 없이 모델 가중치를 변환하지도 않습니다.

<a id="model-loading"></a>

## 모델 로딩

로컬 Diffusers 디렉터리에는 `model_index.json`·구성요소 설정·토크나이저·필수 모델 가중치 전체가 있어야 한다. 선언된 파이프라인 클래스는 자동으로 선택하며, `--pipeline-class`로 이미지 편집이나 인페인팅 등 설치된 다른 호환 작업 파이프라인을 명시적으로 선택할 수 있다. 공개 내장 Diffusers 파이프라인 및 아래 명시한 고정 MOSS 구현과 해당 구성요소만 허용한다. 사용자 지정 파이프라인 스크립트·임의 가져오기 경로·동적 원격 코드는 제외한다.

```sh
reference/diffusers/.venv/bin/python reference/diffusers/generate_any.py \
  --model /Volumes/Storage/Workspace/Models/sd35-medium \
  --pipeline-class StableDiffusion3Pipeline \
  --prompt 'a glass observatory above a cloud sea' \
  --width 1024 --height 1024 --steps 28 --guidance-scale 4.5 \
  --seed 42 --dtype bfloat16 --device cuda --offload model \
  --output-dir resources/outputs/reference/sd35-run
```

`--model`는 기존 로컬 경로여야 합니다. 허브 ID, URL 및 null이 아닌 개정은 거부됩니다. 다운로드 단계가 없습니다. 모델 로드는 항상 `local_files_only=True`, `use_safetensors=True` 및 `trust_remote_code=False`를 사용합니다. [로컬 모델 생성](local-model-generation.md)를 참조하세요.

Diffusers 디렉터리에서 선언된 Diffusers·Transformers 구성 요소의 표준 PyTorch 단일 가중치 또는 분할 인덱스만 있는 경우, 기존 제한된 weights-only 변환기로 필요한 구성 요소를 Safetensors 캐시에 준비한다. 원본 구성과 토크나이저를 포함한 임시 로딩 디렉터리를 사용하고 원본은 수정하지 않는다. 같은 구성 요소와 variant에 Safetensors 가중치가 이미 있으면 이를 우선한다. `--variant`는 내장 Diffusers 및 고정 MOSS 파이프라인의 명시적인 가중치 variant를 지정하며, 단일 체크포인트에 딸린 보조 구성 요소에도 적용한다. 디렉터리 모델에서 생략하면 기본 가중치를 우선하며, 기본 가중치가 부족하고 실제로 사용하는 구성 요소의 공통 variant가 하나뿐일 때 자동으로 선택한다. 후보가 여러 개이거나 일치하지 않으면 사용자가 선택하도록 보고한다. 외부 인코더나 VAE 등으로 대체된 내부 구성 요소는 variant 선택과 변환에서 제외한다. 인코더와 VAE의 variant 분할 인덱스는 가중치가 존재하는 것으로 감지하며, 불완전한 인덱스를 다른 기본 구성 요소로 조용히 대체하지 않는다. MOSS에서 해당 variant가 없는 구성 요소는 기본 가중치를 사용하며, 파일 variant만을 근거로 생성 단계 수나 샘플링 설정을 바꾸지 않는다. 입력과 변환 결과는 로딩 및 출력 게시 전까지 다시 검사하고, 임시 디렉터리는 로딩 이후 제거한다. `--dequantize-pytorch`를 명시하면 주 체크포인트와 디렉터리의 PyTorch 구성 요소에서 지원되는 Torch 양자화 텐서를 float32로 복원하며, 기본값은 거부이다. 동일한 옵션은 JSON `dequantize_pytorch` 불리언으로 지정할 수 있고 CLI가 우선한다. 이 옵션은 GGUF·bitsandbytes·임의의 packed 객체를 변환하지 않는다. 디렉터리의 구조와 텐서 변환은 모델 아키텍처 지원이나 생성 품질을 보증하지 않는다. 선언된 내장 변환기 또는 UNet 디렉터리에 GGUF 가중치가 있으면 표준 가중치와 명시적 variant의 우선순위를 지켜 선택하고, 후보가 여러 개면 임의로 선택하지 않는다. 원본 식별자를 로딩과 결과 게시까지 확인한다.

파일과 디렉터리 심볼릭 링크를 지원하고 순환 링크는 거부한다. 임시 로딩 디렉터리에서 심볼릭 링크를 사용할 수 없을 때는 검증한 복사본을 사용한다. 연결된 원본 디렉터리와 파일의 식별자는 로딩과 출력 게시 전후에 검사한다. 유지 중인 파이프라인의 원본이 바뀌면 현재 요청을 중단하고 메모리 캐시를 폐기하여 다음 요청이 모델을 다시 읽도록 한다. 변환 캐시는 원본이나 연결된 모델 디렉터리 안에 둘 수 없으며 출력은 그 디렉터리나 변환 캐시와 겹칠 수 없다.

단일 체크포인트에 제공한 Diffusers 설정 디렉터리도 선언된 보조 구성 요소의 PyTorch 가중치를 동일하게 변환한다. 표준 구성 요소 이름에 `.bin` 대신 `.pt`, `.pth`, `.ckpt`를 사용한 단일 파일과 분할 인덱스도 인식한다. 구성 요소 파일이나 인덱스의 확장자가 Safetensors여도 실제 내용이 PyTorch이면 내용에 맞춰 변환한다. 같은 우선순위에 후보가 여러 개면 임의로 선택하지 않는다. 주 체크포인트가 Safetensors 분할 인덱스이면 확인된 샤드를 병합해 일반 단일 파일 로더에 전달하며 MOSS 전용 인덱스 로더는 기존 직접 로딩을 유지한다. Transformers 보조 구성 요소는 Diffusers 단일 파일 로더가 표준 Transformers 가중치를 찾을 수 있도록 임시 설정에 연결하며 체크포인트에 포함된 구성 요소의 우선순위는 유지한다. 체크포인트에 포함되지 않고 별도 로드되는 Diffusers 보조 구성 요소는 필요할 때 Safetensors 샤드를 제한된 메모리로 결합해 단일 파일 로더에 제공한다. 선택한 variant가 있으면 이 보조 구성 요소에도 적용한다. 외부에서 대체한 구성 요소와 단일 체크포인트에서 직접 로드한 MOSS·Krea 변환기는 변환하지 않는다. 단일 파일 로더가 임의 pickle을 읽지 못하도록 임시 설정 디렉터리에는 원본 PyTorch 가중치와 인덱스를 노출하지 않는다. 선언된 구성 요소 밖의 미확인 pickle은 기존처럼 거부한다. 모델과 설정 원본은 그대로 유지한다.

명시적 또는 기존 리소스에서 선택한 대체 VAE는 로딩 진단의 누락·미사용·크기 불일치 텐서와 오류를 검사하고, 하나라도 있으면 생성을 중단한다. 빠른 `IILD_MODEL_VALIDATION=metadata` 모드에서도 작은 VAE 설정 파일은 SHA-256으로 확인하고 로딩 및 생성 전후 변경을 검사한다. 이 모드의 대형 가중치는 여전히 파일 메타데이터로 확인하므로 가중치 전체 해시를 확인한 것으로 표시하지 않는다. 모델 자체에 포함된 VAE는 해당 파이프라인의 로딩 계약을 따른다. 명시적 VAE는 디렉터리 외에 별도의 Safetensors·PyTorch 파일이나 샤드 인덱스도 지원하며, 선택한 로컬 설정과 텐서 전체가 일치해야 한다. PyTorch 입력은 설정 조회 시 역직렬화하지 않고 로딩할 때 검증된 weights-only 변환기로 준비하며, 원본과 샤드의 식별자를 유지한다. 별도 VAE의 지원되는 Torch 양자화 텐서는 일반 실수 VAE로 자동 복원하고, 사용한 정책과 텐서 개수는 로딩 보고서와 변환 manifest에 남긴다. VAE의 `vae.`·`first_stage_model.` 접두사와 `AutoencoderKL`·`AutoencoderKLFlux2`의 기존 LDM 이름은 설정과의 일대일 대응을 검증해 임시 파일에서 변환한다. 분할 인덱스의 하위 경로와 누락된 선택적 메타데이터는 임시 로딩 경로에서 정규화하며 원본은 수정하지 않는다.

`tests/VaeDefaultsRuntimeTests.py`는 로컬에 배치된 Qwen Image, FLUX.1 및 FLUX.2 대체 VAE의 safetensors 헤더를 실제 Diffusers 클래스의 메타 장치 구조와 대조한다. 가중치 전체를 리소스 매니페스트의 SHA-256과 비교하려면 `IILD_VERIFY_BUNDLED_VAE_PAYLOAD=1`을 설정해 같은 테스트 파일의 `VaeDefaultsRuntimeTests.test_existing_local_vae_payloads_match_the_resource_manifest`를 실행한다. 없는 리소스는 건너뛰기로 표시된다. 이 검사는 파일 식별 및 구조 검증이며 이미지 생성이나 모든 텐서의 수치 품질 검증이 아니다.

## 범용 러너의 ControlNet 구성 요소

범용 러너는 일반 텍스트-이미지 `StableDiffusionPipeline`, `StableDiffusionXLPipeline`, `StableDiffusion3Pipeline`, `FluxPipeline`, `QwenImagePipeline`에 별도로 선택한 로컬 ControlNet 구성 요소 디렉터리 또는 단일 safetensors 파일을 연결한다. FLUX.1 dev, Schnell과 Krea-dev는 모두 `FluxPipeline` 계열이지만 가중치는 해당 기본 모델과 호환되어야 한다. `--controlnet`과 `--control-image`를 함께 지정하고, 이미지에는 기존 전처리기에서 미리 만든 로컬 힌트를 제공한다. 모델 구성의 클래스, 실제 ControlNet과 베이스 네트워크의 텐서 구성, 로컬 파일을 확인한 뒤에만 해당 ControlNet 파이프라인으로 전환한다. 샘플링 완료 후에도 모델 및 ControlNet 파일 식별자를 재확인한다.

```sh
iild-generate --backend diffusers --model /absolute/path/flux-diffusers \
  --controlnet /absolute/path/flux-controlnet --control-image /absolute/path/canny.png \
  --controlnet-scale 0.8 --control-guidance-start 0 --control-guidance-end 0.8 \
  --prompt 'a city at night' --output-dir resources/outputs/reference/flux-control
```

제어 이미지, 강도 및 구간을 `--pipeline-inputs`에도 중복 지정하면 오류이다. `--controlnet-variant`는 구성 요소 가중치 변형을 선택한다. FLUX Union은 모델별 `--control-mode`를 사용하며, `--guess-mode`는 SD/SDXL에서만 허용한다. 입력은 기존 범용 이미지 로더가 RGB로 디코딩한다. `--control-preprocess canny`를 선택하면 별도 구현을 근사하지 않고 iiLocalDiffusion 네이티브 Canny 함수를 호출한다. 네이티브 경로와 동일하게 RGB 사본에 high/low `0.08`, weak `0.8`, strong `1.0`, inverse `false`를 적용하며 각 변은 최대 2048픽셀이다. SDK 빌드 또는 설치 라이브러리를 찾을 수 없다면 `--control-preprocessor-library`나 `IILD_NATIVE_LIBRARY`로 명시한다. 라이브러리와 원본 입력의 식별자는 결과에 기록된다.

설치된 러너는 자체 설치 접두사의 `lib/` 또는 `lib64/`(Windows는 `bin/`)에서 Canny 공유 라이브러리를 우선 찾는다. 소스 실행은 기존 `build/`를 사용하며, 명시 경로와 환경 변수는 자동 탐색보다 우선한다. Windows에서는 로딩 중 라이브러리 디렉터리와 유효한 PATH 디렉터리를 DLL 탐색 경로에 추가하고 이후 해제한다. 표준 위치와 다른 사용자 지정 라이브러리 디렉터리는 명시 경로로 지정한다.

범용 러너의 파일 이미지 입력은 디코딩 후 EXIF 방향을 적용하고 요청된 RGB·RGBA·L 형식으로 변환한다. 제어 이미지뿐 아니라 색상 맵, 인페인트 원본과 마스크, 영역 맵, 일반 참조 이미지에도 같은 규칙이 적용된다. Krea 2 자동 크기 선택과 Canny·깊이 전처리는 방향을 적용한 이미지의 크기와 픽셀을 사용하며, 입력 기록에는 적용 사실과 변환 후 크기가 남는다. EXIF 방향이 있는 파일은 이를 무시하던 이전 버전과 결과가 달라질 수 있다.

단일 파일 `.safetensors` 또는 `.safetensor`는 인접 `config.json`을 사용하거나 `--controlnet-config /absolute/path/to/config-directory`로 별도의 로컬 구성을 지정한다. `--controlnet-variant`는 디렉터리 전용이다. SD/SDXL 원본 형식은 기존 프리셋 러너의 엄격한 변환기를 재사용하며, SD3·FLUX.1·Qwen Image 파일은 네이티브 Diffusers 텐서 이름이 필요하다. 가중치와 설정의 신원은 각각 기록하고 생성 전후로 확인한다. 임시 패키지와 `.safetensor` 별칭은 실행 후 제거된다. 별칭과 임시 패키지는 파일시스템의 심볼릭 링크 지원이 필요하다. 소형 실제 FLUX ControlNet을 디렉터리와 단일 `.safetensor`로 각각 로드한 수치 테스트에서 동일한 시드의 출력과 임시 패키지 정리를 확인했다.

Stable Diffusion 3은 로컬 `SD3ControlNetModel`과 RGB 제어 이미지를 사용한다. VAE 잠재 채널과 Transformer 차원, 제어 블록의 사용 여부를 검사하고 추가 인페인트 채널이나 입력 임베딩이 없는 별도 구조는 거부한다. 기존 Canny 또는 상대 깊이 전처리도 선택할 수 있다. Diffusers의 SD3 ControlNet 파이프라인은 제어 이미지 VAE 표본에 기본적으로 전역 난수를 사용한다. 범용 러너에서 `--seed`를 지정하면 생성기 상태와 SD3 전용 식별자에서 SHA-256으로 파생한 별도 CPU 생성기로 그 표본을 만든다. 생성 노이즈와 난수열이 다르고 원래 생성기의 상태와 전역 난수 상태는 건드리지 않는다. 이 v1 정책은 제어와 생성에 같은 시드를 사용하던 이전 구현과 결과가 다르다. 샘플링 완료나 오류 후 VAE 메서드를 복원한다. 공개 [InstantX SD3 Canny 구성](https://huggingface.co/InstantX/SD3-Controlnet-Canny/blob/main/config.json)을 구조 참고에 사용했으며, 소형 무작위 가중치 테스트는 전역 난수 상태가 달라도 같은 시드의 잠재 출력이 동일한지 확인한다. 하드웨어나 런타임을 넘어 비트 단위 동일성을 보장하지 않으며 사전 학습 가중치의 실제 이미지 생성과 품질은 검증하지 않았다.

FLUX.1은 VAE로 힌트를 인코딩하는 InstantX 계열과 RGB 힌트 인코더를 내장한 XLabs 계열의 Diffusers `FluxControlNetModel`을 구분한다. 후자는 인코더 출력에 2×2 패킹을 사용하므로 ControlNet의 `patch_size=2`와 베이스 Transformer의 `patch_size=1`이 서로 달라도 정상이다. 입력 힌트 채널과 패킹 후 베이스 입력 채널이 맞는지 확인한다. [XLabs 공개 Diffusers 구성](https://huggingface.co/XLabs-AI/flux-controlnet-canny-diffusers/blob/main/config.json)의 구조를 참고했으며, 작은 무작위 가중치로 로딩과 힌트 영향을 검증했다. 원본 XLabs 체크포인트를 자동 변환하지 않으며 공개 사전 학습 가중치의 생성 품질은 검증하지 않았다. 가중치의 라이선스는 별도 확인해야 한다. 예를 들어 [XLabs Canny v3 원본](https://huggingface.co/XLabs-AI/flux-controlnet-canny-v3)은 비상업 라이선스를 명시한다.

FLUX.1의 CLIP 토크나이저가 인코더의 위치 임베딩보다 큰 최대 길이를 선언하면, 러너는 실제 로드된 CLIP 인코더의 위치 수로 실행 중 토크나이저 길이를 제한한다. 예를 들어 기존 ComfyUI CLIP 토크나이저의 8192와 CLIP 인코더의 77이 함께 사용될 때 77로 제한하여 인코더 입력 오류를 방지한다. 더 짧게 설정된 토크나이저는 그대로 유지하며, 토크나이저 파일은 수정하지 않는다. 제한이 적용되면 `model.flux_clip_token_limit`에 원래 값과 적용 값을 기록한다. FLUX.2와 다른 모델 계열에는 적용하지 않는다. 이 처리는 누락된 CLIP·T5 인코더 가중치를 제공하지 않는다.

FLUX.1 모델이 호환 `CLIPTokenizer` 또는 `T5Tokenizer`를 선언하지만 해당 디렉터리가 비어 있거나 없을 때, 범용 러너는 기존 ComfyUI에서 가져온 `resources/text_encoders/flux1-clip/` 및 `flux1-t5/` 리소스를 각각 사용한다. 로컬 파일의 크기와 SHA-256을 매니페스트와 대조하고, 등록되지 않은 추가 파일도 거부한다. 디렉터리에 파일이 일부라도 있으면 모델 구성 요소를 우선하여 로더에서 불완전한 모델을 감지한다. 두 프롬프트 임베딩을 모두 제공하고 별도 부정 텍스트 인코딩이 필요하지 않으면 누락 토크나이저를 로드하지 않는다. 각 선택은 `request.flux_tokenizers`에 기록된다.

모델이 FLUX.1 CLIP·T5 인코더를 선언하지만 해당 가중치가 없으면 로컬 `fallback_flux_text_weights` 카탈로그의 원본 CLIP과 스칼라 배율 FP8 T5를 각각 선택한다. 단일 베이스 파일에 포함된 인코더나 모델 디렉터리의 일부라도 존재하는 가중치는 대체하지 않는다. 프롬프트 임베딩이 충분하면 누락 인코더를 로드하지 않으며, 명시적인 인코더 옵션은 항상 우선한다. 고정된 대체 가중치는 모델의 일반 메타데이터 전용 검증 옵션과 무관하게 SHA-256을 대조한다. 리소스를 준비하려면 프로젝트 루트에서 `python3 scripts/provision_flux1_text_encoders.py`를 실행한다. 원격의 고정 리비전을 이어받아 전체 크기와 SHA-256이 맞을 때만 최종 경로에 둔다. `--check`는 다운로드 없이 두 파일을 검증한다. 가중치는 Git에 포함되지 않으며 CLIP 약 246MB, T5 약 5.16GB의 로컬 공간이 필요하다. 리소스가 아직 없으면 해당 상태를 보고하고 모델 로더가 누락 구성 요소를 처리한다. 실행 중 자동 다운로드하지 않는다.

2026-10-31 작업공간에 준비된 두 파일은 전체 크기와 SHA-256을 확인했다. 실제 T5 파일의 스칼라 배율 168개는 유한한 음수가 아닌 값이고 두 임베딩 텐서의 전체 바이트가 일치했다. 이는 파일과 구조의 확인이며 사전 학습된 전체 이미지 모델의 실행이나 품질 검증은 아니다.

FLUX.1의 분리된 로컬 CLIP 및 T5 safetensors 가중치는 `--flux-clip-encoder /absolute/path/clip_l.safetensors`와 `--flux-t5-encoder /absolute/path/t5xxl_fp16.safetensors`로 지정한다. 명시된 파일이 해당 모델의 인코더보다 우선한다. 각 모델의 `text_encoder/config.json` 또는 `text_encoder_2/config.json`을 우선하고, 없으면 리소스 매니페스트에 고정된 원본 CLIP 또는 T5 설정을 사용한다. 파일의 모든 텐서 이름·크기와 설정을 검증하고, T5의 `shared.weight`와 `encoder.embed_tokens.weight`가 같은지 확인한다. CLIP의 `text_model.` 접두사도 지원한다. 비양자화 F16·BF16·F32와 별도 배율이 없는 E4M3 FP8을 읽는다. T5는 공개 ComfyUI 구형 `scaled_fp8` 표지 및 레이어별 `.scale_weight`가 있는 스칼라 배율 E4M3 FP8도 지원한다. 각 FP8 값과 배율을 선택한 계산 정밀도로 변환해 곱하고 유한성을 확인한다. 블록 배율, 입력 배율, 다른 표지와 양자화 형식은 지원하지 않는다. FP8 네이티브 행렬 연산 대신 메모리에 일반 가중치를 구성하므로 실행 정밀도의 메모리가 필요하며 기존 FP8 커널과 비트 단위 결과가 같다는 의미는 아니다. 설정과 파일 식별자는 `request.flux_text_encoders`에, 로딩 형식과 배율 적용 레이어 수는 `model.flux_text_encoder_loading`에 기록한다.

선택적으로 `--flux-t5-fp8-storage cpu-stream`을 지정하면 FP8 T5 선형 가중치를 CPU에 압축 상태로 두고 호출하는 계층만 계산 정밀도로 복원해 실행 장치에 전달한다. 별도 명시 파일과 자동 대체 파일은 스칼라 배율 FP8 형식을 사용하며, 공개 Comfy 통합 단일 파일의 내장 T5는 별도 배율이 없는 E4M3 형식을 사용한다. 통합 파일에서 T5가 감지되면 해당 텐서를 직접 사용하고 공유 임베딩 별칭을 복원한다. 명시적 `--flux-t5-encoder`가 있으면 그 파일이 우선한다. 모든 경우 `--offload none`이 필요하다. 별도의 로컬 LoRA는 모든 텐서 키가 `transformer.` 또는 `text_encoder.`로 시작하는 Diffusers 형식일 때 함께 사용할 수 있다. 이 둘은 각각 베이스 트랜스포머와 CLIP을 의미하며 스트리밍 T5를 변경하지 않는다. T5 대상, 접두사가 없거나 Kohya 등 다른 이름 체계의 LoRA와 T5를 수정할 수 있는 파이프라인 로더는 명시적으로 거부한다. T5 임베딩·정규화 가중치, 다른 구성 요소와 현재 복원 중인 계층은 별도 메모리를 사용하며 반복 전송과 복원으로 속도가 저하될 수 있다. 보고서에는 압축 가중치의 바이트 수와 최대 단일 계층의 복원 바이트 수가 기록되지만 전체 프로세스·GPU 메모리 실측치는 아니다. 이 임시 실행 모델의 선형 계층에는 계산 자료형을 표시하는 자리표시자 가중치가 있으므로 불완전한 체크포인트가 저장되지 않도록 `state_dict` 직렬화를 거부한다. 소형 합성 T5와 FLUX ControlNet은 CPU와 사용 가능한 MPS에서 일반 복원 방식과 잠재 출력을 대조한다. 트랜스포머·CLIP LoRA를 함께 적용한 합성 테스트도 두 방식의 출력을 비교한다. 사전 학습 모델의 실제 메모리 사용량이나 이미지 품질은 검증하지 않았다.

내장 T5 스트리밍은 `text_encoders.t5xxl.transformer.` 네임스페이스와 해당 형식을 엄격히 검사한다. 범위 밖의 모호한 양자화 메타데이터나 FP8 선형 가중치가 없는 파일은 거부한다. 선택 경로는 `request.flux_text_encoders.text_encoder_2.status`의 `embedded`로 기록한다.

FLUX·MOSS 트랜스포머, FLUX T5, Krea 2 FP8의 스트리밍 선형 계층은 `load_state_dict`로 베이스 `weight`를 교체하는 작업도 거부한다. 실제 계산에 쓰이는 CPU 보관 가중치 또는 FP8 배율이 별도로 유지되므로 일반 밀집 가중치만 교체하면 모델 상태와 계산 결과가 달라질 수 있다. 베이스 가중치를 바꾸려면 원하는 원본 체크포인트로 모델을 다시 로드한다. 베이스 `weight`를 포함하지 않는 어댑터 상태 로딩은 이 제한의 대상이 아니다. 상위 모델 전체의 `load_state_dict`는 트랜잭션이 아니므로 거부되기 전에 다른 계층이 변경되었을 수 있다.

`--flux-transformer-storage cpu-stream`은 별도의 원본 FLUX.1 단일 파일 트랜스포머 선형 가중치를 CPU에 원본 정밀도로 유지하고 계층별로 실행 장치와 계산 정밀도에 옮긴다. 원본 이중·단일 블록의 Q·K·V 분할과 최종 변조 순서를 실제 Diffusers 구조에 맞춰 검증한다. 비배율 E4M3 FP8과 비양자화 F16·BF16·F32를 지원하며, 배율이 있는 FP8 또는 모호한 양자화 메타데이터는 거부한다. `--flux-t5-fp8-storage cpu-stream`과 함께 사용할 수 있고 준비된 또는 전처리한 ControlNet 힌트도 그대로 적용한다. 이 방식은 `--offload none`을 요구하며 현재 별도 LoRA는 지원하지 않는다. LoRA가 기본 리소스에서 자동 선택된다면 `--no-default-modifiers`로 비활성화해야 한다. 가중치 전송과 정밀도 변환은 반복되므로 느릴 수 있으며, VAE·ControlNet·텍스트 인코더·활성값과 메모리 매핑된 원본은 별도 메모리를 사용한다. 보고서의 스트리밍 바이트와 최대 계층 바이트는 전체 프로세스·장치 메모리 실측값이 아니다. 스트리밍 계층의 `state_dict` 저장은 불완전한 체크포인트를 막기 위해 거부된다.

```sh
python reference/diffusers/generate_any.py --model /absolute/path/flux1-schnell.safetensors \
  --model-config-preset flux1-schnell --flux-transformer-storage cpu-stream \
  --flux-t5-fp8-storage cpu-stream --controlnet /absolute/path/flux-controlnet \
  --control-image /absolute/path/photo.png --control-preprocess canny --no-default-modifiers \
  --prompt 'a city at night' --seed 42 --output-dir resources/outputs/reference/flux-stream
```

공개 FLUX.1 Schnell BF16 트랜스포머 전용 파일과 FP8 통합 파일의 헤더는 실제 Diffusers 모델 구성 및 기존 변환기와 대조했다. 소형 원본 형식 체크포인트의 밀집 로딩과 스트리밍은 내장 T5·VAE, 기존 FLUX.1 대체 VAE, Canny·소형 깊이 전처리 및 ControlNet을 조합해 비교한다. 소형 모델의 상위 Diffusers 변환기만 고정된 폭 3072를 시험용 폭으로 조정한다. 전체 사전 학습 체크포인트를 실행하거나 전체 메모리 사용량과 이미지 품질을 검증한 것은 아니다.

`python3 scripts/provision_flux1_schnell.py`는 위에서 검사한 공개 FP8 통합 체크포인트를 `resources/models/flux1-schnell-fp8.safetensors`에 준비한다. 17,236,328,572바이트가 필요하며 중단된 다운로드 이어받기와 SHA-256 검증 후 게시를 지원한다. `--check`는 기존 파일만 검사한다. 체크포인트는 `--model`로 명시해야 하며, ControlNet은 별도 준비해야 한다. 준비 및 헤더 검사는 해당 ControlNet의 학습 대상 호환성이나 생성 품질을 증명하지 않는다. 출처와 라이선스는 [리소스 문서](../resources/README.md)를 참조한다.

공개 Schnell 사용 사례가 있다고 해서 해당 ControlNet이 Schnell로 학습된 것은 아니다. 예를 들어 [MFlux의 Schnell Canny 설정](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/common/config/model_config.py)은 InstantX의 Dev Canny를 선택하며, [InstantX 제작자](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny/discussions/3)는 Schnell에서 성능을 보장하지 않는다고 설명한다. 해당 [ControlNet의 배포 라이선스](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny)는 비상업적 조건이다. 베이스 체크포인트의 Apache-2.0 표기가 별도 제어 가중치에 자동으로 적용되지는 않는다.

이 특정 가중치를 사용할 권한이 있다면 `python3 scripts/provision_flux1_canny.py`로 InstantX Dev Canny의 Diffusers 설정과 가중치를 `resources/models/flux1-instantx-canny`에 별도 준비할 수 있다. `--check`는 두 파일의 전체 크기와 SHA-256을 검증한다. 이 디렉터리를 `--controlnet`으로 명시해야 하며 자동 선택하지 않는다. 공개 헤더의 188개 텐서 형태는 실제 Diffusers ControlNet과 Schnell 베이스 구조에 대조했고, 2026-10-31에는 실제 어댑터 전체를 로드하여 모든 매개변수의 유한성과 Schnell 구조 호환성도 확인했다. 합성 소형 파이프라인은 guidance가 필요한 제어 모델과 guidance가 없는 Schnell 베이스에 각각 올바른 입력이 전달되고 네이티브 Canny 힌트에 따라 잠재 결과가 변하는지 검사한다. 이는 특정 사전 학습 조합의 생성 품질이나 Schnell에 대한 제작자의 보증을 뜻하지 않는다.

내장 T5 경로의 통합 회귀 테스트는 공개 Schnell 트랜스포머 형상을 축소한 원본 형식 FP8 텐서와 소형 합성 CLIP·T5를 하나의 체크포인트에 넣고 실제 Diffusers `FluxPipeline.from_single_file`로 로드한다. 범용 러너가 준비한 토크나이저와 기존 FLUX.1 VAE를 주입하여 Canny·소형 깊이 전처리 및 ControlNet을 연결한다. 별도의 통합 파일 시험은 공개 원본 형상의 채널을 축소한 VAE 가중치도 내장하여 대체 VAE 대신 실제 단일 파일 로더로 읽는다. 밀집 T5는 단일 파일 로더가 읽고 스트리밍 T5는 범용 러너가 해당 로더에 주입하며, 두 결과를 대조한다. 트랜스포머는 실제 단일 파일 로더와 상위 Diffusers 변환기를 통과한다. 다만 상위 변환기에 고정된 3072 폭 분할 크기만 시험 중 소형 모델의 32 폭에 맞게 조정하며, 변환한 가중치를 직접 주입한 참조 모델과 두 전처리의 잠재 출력을 각각 비교한다. 공개 크기 트랜스포머 변환 계약은 별도의 실제 헤더 기반 구조 검사에서 검증한다. 이 시험은 사전 학습 체크포인트의 전체 로딩 또는 이미지 품질을 검증하지 않는다.

FLUX.1 `FluxPipeline` 단일 파일에 원본 VAE가 포함되고 명시적 VAE 대체가 없으면, 가중치를 로드하기 전에 로컬 VAE 설정과 safetensors 헤더만으로 Diffusers의 원본 VAE 변환을 메타 장치에서 검증한다. 변환된 각 텐서가 설정된 VAE의 이름·형상과 정확히 일치해야 하며 누락, 불일치, 원본 네임스페이스 혼용은 오류이다. 이 검사는 텐서 값, 파일 전체의 무결성이나 이미지 품질까지 검증하지 않는다. 누락된 가중치를 임의 초기화하거나 대체 VAE로 숨기지 않는다.

```sh
iild-generate --backend diffusers --model /absolute/path/flux-model \
  --flux-clip-encoder /absolute/path/clip_l.safetensors \
  --flux-t5-encoder /absolute/path/t5xxl_fp16.safetensors \
  --controlnet /absolute/path/flux-controlnet --control-image /absolute/path/photo.png \
  --control-preprocess canny --prompt 'a city at night' \
  --output-dir resources/outputs/reference/flux-split-text
```

고정된 [공개 FLUX 인코더 파일](https://huggingface.co/comfyanonymous/flux_text_encoders/tree/3bcfe977a9a2617a0b9025076030e217795fe028)의 CLIP·T5 FP16·비배율 및 스칼라 배율 T5 FP8 헤더는 실제 Transformers 메타 모델의 이름·크기와 비교했다. 배율 FP8의 공개 헤더에 있는 배율 168개는 각각 대응하는 가중치와 대조한다. 소형 합성 인코더를 사용한 실행 테스트는 디렉터리와 단일 파일 경로, 스칼라 배율 FP8과 같은 정밀도로 복원한 일반 가중치의 제어 출력 일치를 확인한다. 이후 CLIP과 스칼라 배율 FP8 T5 대체 가중치는 작업공간에서 전체 SHA-256을 확인했다. 다른 공개 인코더 파일 전체나 사전 학습 이미지 결과를 검증했다는 뜻은 아니다.

전체 범용 러너 테스트는 소형 합성 FLUX.1 트랜스포머·ControlNet과 사전 계산된 임베딩을 사용하면서 기존 FLUX.1 VAE를 자동 선택하고, 네이티브 Canny와 소형 Depth Anything 출력에 대한 제어 변화 및 동일 시드 반복을 확인한다. 추가로 기존 ComfyUI CLIP·T5 토크나이저와 소형 합성 텍스트 인코더로 문자열 프롬프트의 변화와 반복, CLIP 길이 제한 및 토크나이저 자동 대체 전후의 출력 일치를 검사한다. 사전 학습된 CLIP·T5 인코더, 베이스 또는 제어 모델의 생성 품질 검증은 아니다.

Qwen Image는 `QwenImageControlNetModel`을 사용하며 기존 Qwen Image RGB VAE의 16채널 잠재 공간과 2×2 패치를 요구한다. 제어 모델과 베이스의 차원·어텐션·위치 인코딩을 대조하며, 미사용 제어 블록이 생기는 배치, 인페인팅 전용 추가 채널과 별도의 시간 조건을 가진 베이스 변형은 거부한다. Qwen Image에서도 준비된 힌트 또는 `--control-preprocess canny`를 사용할 수 있다. `--guess-mode`와 FLUX Union `--control-mode`는 적용되지 않는다. [InstantX Qwen Image ControlNet Union](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union) 리비전 `b13036f066d6dee7c20513e263d3d673055e9de8`의 공개 설정과 가중치 헤더는 입력 64채널과 제어 블록 5개의 계약을 확인하는 데 사용했다. 소형 실제 Qwen 파이프라인 테스트는 제어 이미지의 영향과 동일 시드 재현을 확인한다. 전체 공개 가중치 로딩이나 품질 검증을 의미하지 않는다.

원본 사진에서 상대 깊이를 계산하려면 `--control-preprocess depth --control-depth-model /absolute/path/depth-model`을 지정한다. `--control-image`는 이때 원본 사진이다. 로컬 Transformers 형식의 상대 Depth Anything 모델과 DPT 이미지 프로세서, safetensors 가중치가 필요하며 다운로드하지 않는다. Krea 2, FLUX.1/2, MOSS-Image 및 위 일반 ControlNet 경로에서 동일하게 선택할 수 있다. 깊이 모델 정밀도는 기본 FP32이며 `--control-depth-dtype float16`으로 FP16을 선택할 수 있다. 모델의 부동소수점 입력에도 선택한 정밀도를 적용하고 정수 입력은 정수형을 유지한다. 깊이 예측은 FP32로 변환해 출력 이미지 크기로 쌍선형 보간한 뒤 최솟값과 최댓값으로 정규화한다. 가까울수록 밝은 값이며, 모델 파일 식별자와 설정을 결과에 기록하고 실행 후 재검사한다. Krea 2의 기본 깊이 Control-LoRA 경로는 깊이 추정 전에 이미지를 출력 크기에 맞춰 중앙 크롭하고, 원본 구현처럼 정규화된 FP32 깊이를 세 채널로 복제해 VAE에 직접 전달한다. 따라서 그 경로에서 깊이 값을 8비트로 양자화하지 않는다. Krea 2 색상·깊이 어댑터와 일반 ControlNet, FLUX.2 및 MOSS 경로는 RGB 8비트 제어 이미지 계약을 사용한다.

`python3 scripts/provision_depth_anything_v2.py --variant small` 또는 `--variant large`로 고정된 모델 구성·이미지 프로세서·가중치를 각각 `resources/depth-estimators/depth-anything-v2-small` 또는 `depth-anything-v2-large`에 준비한 후 해당 디렉터리를 `--control-depth-model`에 지정한다. 각 파일은 전체 크기와 SHA-256 확인 뒤 게시되며 `--check`는 세 파일을 모두 검증한다. Small은 Apache-2.0, Krea 깊이 어댑터 원본이 사용하는 Large는 CC-BY-NC-4.0이다. Small은 상업적 사용을 고려한 대안이지만 Large와 같은 깊이 결과를 보장하지 않는다. 리소스 준비가 원본 워크플로의 비트 단위 재현이나 생성 품질을 증명하지 않는다.

소형 무작위 Depth Anything 모델의 실제 출력을 Krea 2 깊이 Control-LoRA, FLUX.1/2 및 MOSS의 소형 제어 파이프라인에 연결한 테스트는 서로 다른 깊이 힌트의 영향과 동일 시드 반복을 확인한다. 이는 전처리부터 샘플링까지의 데이터 연결 검증이며 사전 학습 모델의 시각적 품질이나 공개 워크플로와의 이미지 일치를 입증하지 않는다.

깊이 예측과 리사이즈 결과가 유한한지 검사한다. 유한한 최솟값과 최댓값의 차이만 FP32 범위를 넘어서는 경우에는 값을 먼저 스케일해 정규화한다. 리사이즈나 정규화 결과가 비유한 값이면 오류로 중단하여 잘못된 깊이 힌트를 생성에 넘기지 않는다. 일반적인 유한 범위에서는 기존 FP32 정규화 순서를 유지한다.

생성 보고서의 `control_preprocessor.output_identity`는 전처리 직후, 모델별 리사이즈와 VAE 인코딩 전 조건의 크기·형식·SHA-256을 기록한다. RGB 출력은 행 우선 RGB8 바이트, Krea 2의 부동소수점 깊이는 행 우선 리틀엔디언 FP32 바이트를 해시한다. 원본 파일과 전처리 리소스가 같더라도 이 값을 비교하면 실제 생성에 전달된 전처리 결과가 같은지 확인할 수 있다. 조건 이미지 자체를 별도 파일로 저장하거나 다른 장치 간 동일성을 보장하지는 않는다.

Krea 2 기본 깊이 및 Krea 2·MOSS 색상/깊이 어댑터는 추가로 `control_execution.vae_input_hints`에 모델별 크롭·리사이즈와 채널 변환 이후, VAE의 `[-1,1]` 정규화 및 정밀도 변환 이전 입력의 크기·형식·SHA-256을 기록한다. 키는 사용한 `color`와 `depth`이며 미사용 맵은 없다. 이 값은 준비된 맵을 직접 제공할 때도 기록되므로 서로 다른 리사이즈 설정의 실제 픽셀을 비교할 수 있다. 부동소수점 깊이는 정규화된 `[0,1]` FP32를 해시하며, VAE 잠재값이나 전체 생성 이미지의 일치를 의미하지는 않는다.

```sh
iild-generate --backend diffusers --model /absolute/path/flux-diffusers \
  --controlnet /absolute/path/flux-depth-controlnet --control-image /absolute/path/photo.png \
  --control-preprocess depth --control-depth-model /absolute/path/depth-anything \
  --prompt 'a city at night' --output-dir resources/outputs/reference/flux-depth
```

공개 Krea 2 구현은 [Depth-Anything-V2-Large-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)와 FP16 추론을 사용한다. 이 사전 학습 가중치는 기본 배포에 포함되지 않으며 위의 `--variant large` 공급 명령으로 별도 준비한다. 이 모델의 가중치는 CC-BY-NC-4.0이므로 상업적 이용 가능 여부는 모델 라이선스를 따로 확인해야 한다. `--control-depth-dtype float16`으로 원본의 모델 정밀도를 선택할 수 있으나 장치, 런타임, VAE 난수 정책의 차이로 비트 단위 동일성이 보장되지는 않는다. 다른 가중치도 원본 모델과 같은 결과를 보장하지 않는다. 포즈·세그멘테이션 자동 추정, 다중 ControlNet과 캐시 유지형 foreground preparation은 제공하지 않는다. 전체 사전 학습 체크포인트나 출력 품질을 검증했다는 의미는 아니다.

러너의 합성 통합 테스트는 깊이 및 네이티브 Canny 전처리 결과가 Krea 2, FLUX.1, FLUX.2, MOSS 각 실행 경로에 전달되고 MOSS의 별도 색상 입력이 보존되는지 확인한다. 합성 대체 파이프라인을 사용한 연결 검증이며, 사전 학습 모델의 생성 결과 검증은 아니다. ControlNet이 포함된 foreground preparation 요청은 입력 이미지의 디코딩이나 깊이 모델 실행 이전에 명시적으로 거부한다.

추가 수치 테스트는 실제 빌드된 네이티브 Canny가 만든 빈 맵과 사각형 에지 맵을 소형 무작위 가중치의 Krea 2 Control-LoRA, FLUX.1 ControlNet, FLUX.2 Fun 및 MOSS 색상·깊이 경로에 각각 전달한다. 동일 에지와 시드는 같은 잠재 출력을, 빈 맵과 에지 맵은 서로 다른 출력을 만들어야 한다. 여기서 어댑터 가중치는 합성된 것이므로 실제 깊이 전용 가중치가 Canny 맵을 이해한다는 뜻이나 사전 학습 모델의 품질 증거는 아니다. 네이티브 공유 라이브러리와 수치 테스트용 Python 의존성이 없으면 해당 검증은 생략된다.

범용 러너의 전체 호출을 검증하는 추가 통합 테스트는 로컬의 소형 합성 Krea 2, FLUX.2 Fun, MOSS-Image 패키지와 제어 가중치를 사용한다. 실제 네이티브 Canny와 소형 무작위 Depth Anything을 각각 입력 사진에 적용하고 모델 로딩, 제어 적용, 잠재 출력과 생성 보고서 저장까지 실행한다. 세 경로에서 입력을 바꾸면 잠재 출력이 달라지고 같은 입력과 시드는 반복됨을 확인한다. Krea 2는 기존 토크나이저 대체 리소스를 사용하고 깊이 힌트를 FP32로 보존한다. FLUX.2는 합성 사전 계산 텍스트 임베딩을 사용하고, MOSS는 작은 무작위 Qwen3.5와 시험용 토크나이저로 실제 프롬프트 인코딩을 거친다. MOSS에서는 프롬프트 변경과 CFG 적용이 출력에 반영되고 같은 시드가 반복되는지도 확인한다. 기존 Qwen Image VAE와 FLUX.2 VAE 파일이 있으면 VAE를 생략한 Krea 2, FLUX.2, MOSS 패키지에서 각각 해당 리소스를 자동 선택하여 Canny 및 깊이 힌트를 실제로 인코딩하고 제어 결과의 변화와 시드 반복을 확인한다. 이 추가 시험에서도 트랜스포머와 제어 가중치는 합성이다. FLUX.2와 MOSS의 깊이 힌트는 RGB8이다. 이 검증은 사전 학습 모델의 이미지 품질이나 공개 워크플로와의 이미지 일치를 입증하지 않는다.

별도의 실제 Diffusers 런타임 테스트는 무작위 소형 Krea 2 모델을 저장할 때 토크나이저를 생략하고, 범용 로더가 기존 토크나이저를 주입해 프롬프트 인코딩과 Control-LoRA 후크를 포함한 잠재 출력을 계산하는지 확인한다. 이 시험은 소형 합성 가중치를 사용하며 공개 체크포인트 전체의 로딩, 이미지 품질 또는 원본 결과의 재현을 입증하지 않는다.

Krea 2의 Control-LoRA와 MOSS-Image 색상·깊이 어댑터는 위의 FLUX ControlNet 구성 요소와 다른 결합 계약을 가진다. Krea 2 경로는 [공개 Control-LoRA](https://github.com/Tanmaypatil123/Krea-2-controlnet)의 safetensors 파일을 `--controlnet`으로 받으며, 입력 투영과 모든 블록의 A/B 텐서 이름·크기를 실제 베이스 트랜스포머와 대조한다. Qwen VAE로 조건 이미지를 인코딩하고 2×2 패치로 묶은 잠재 표현을 확장 입력 투영에 공급한다. 원본과 같은 RGB 힌트를 출력 크기로 중앙 자른다. 픽셀 정규화와 VAE 잠재 표현의 평균·표준편차 정규화는 FP32로 수행하고, 각각 VAE 및 트랜스포머에 전달할 때 선택 정밀도로 변환한다. `--seed`에 따른 생성기 상태에서 용도를 구분한 SHA-256 값으로 제어 VAE 전용 난수 생성기를 파생하므로 원래 생성기의 상태와 전역 난수 상태는 보존되고, 생성 노이즈와 제어 표본이 같은 난수열을 사용하지 않는다. 이 v1 정책은 이전 생성기 복제 방식과 결과가 다르다. 공개 구현은 VAE 표본에 전역 난수를 사용하므로, 이 별도 시드 정책은 공개 구현과의 비트 단위 동일성을 의미하지 않는다. 기본 모델, 어댑터 및 입력의 파일 식별자를 기록하고 어댑터를 실행 후 다시 검사한다. 요청 이후 임시 후크는 제거한다.

`python3 scripts/provision_krea2.py --role turbo`와 `--role depth`는 각각 공개 Krea 2 Turbo 스칼라 배율 FP8 체크포인트와 깊이 Control-LoRA를 고정된 출처에서 준비한다. 각각 약 13.14GB와 862MB이며 이어받기와 SHA-256 검증을 지원한다. 파일은 모델과 제어 리소스로 각각 명시적으로 선택하며 `--check`로 다운로드 없이 검증할 수 있다. [어댑터 제작자](https://huggingface.co/Patil/Krea-2-depth-controlnet)는 Raw로 학습한 어댑터를 Turbo에서도 사용할 수 있다고 설명한다. 둘 다 Krea 2 Community License의 적용을 받으며, 현재 SDK에서 전체 사전 학습 조합의 출력 품질을 검증한 것은 아니다.

다운로드 후 선택적 Krea 2 페이로드 검사는 깊이 어댑터의 전체 해시·텐서 기술 정보·모든 값의 유한성을 검사한다. Turbo 검사는 전체 텐서 기술 정보와 FP8 배율, 선택된 텐서 행을 확인하며 전체 해시도 선택할 수 있다. 실행 방법과 검사 범위는 [리소스 문서](../resources/README.md)를 참조한다. 이 검사는 추론이나 이미지 품질 검증이 아니다.

이 Krea 2 어댑터에서 `--controlnet-scale`은 공개 구현처럼 블록 LoRA에만 적용되고 학습된 입력 투영은 항상 전체 강도이다. 따라서 0도 조건을 끄지 않는다. 이 경로는 현재 `--offload none`, 전체 디노이징 구간, 명시적 시드를 요구한다. 별도의 스타일 LoRA는 `--lora`와 `--lora-scale`로 함께 선택할 수 있다. 입력 투영에 적용된 PEFT 스타일 잔차도 제어 투영 위에 더해지며, `--controlnet-scale`은 제어 블록에만 적용된다. 소형 무작위 Krea 2와 합성 가중치를 사용하는 실제 PEFT 로더 테스트는 입력 투영과 어텐션 투영 각각에서 두 LoRA의 영향, 같은 시드 반복 및 예외 후 복원을 확인한다. 개별 체크포인트의 호환성과 사전 학습 모델의 이미지 품질은 별도 검증이 필요하다. 같은 방식으로 별도 색상·깊이 어댑터도 스타일 LoRA와 결합할 수 있지만, AI Toolkit 포즈 어댑터에는 이 조합을 적용하지 않는다. 깊이는 준비된 힌트 또는 위의 로컬 상대 깊이 추정 모델을 사용한다. `--control-preprocess canny`는 Canny로 학습한 호환 Control-LoRA를 위한 옵션이지 depth 모델의 전처리가 아니다. 포즈 힌트가 입력으로 받아들여진다는 사실만으로 다른 포즈 모델 가중치와 호환되는 것은 아니다. 합성 모델과 가중치 테스트는 사전 학습된 특정 가중치의 결과와 시각 품질을 보증하지 않는다.

[Krea 2 Turbo OpenPose 어댑터](https://huggingface.co/thedeoxen/Krea-2-pose-controlnet)는 위 깊이 Control-LoRA와 다른 AI Toolkit 이미지 편집 형식이다. `--controlnet-format krea2-ostris-edit`를 지정하고 `--controlnet`에 해당 로컬 safetensors, `--control-image`에 준비된 OpenPose/DWPose 맵을 넣는다. 포즈 이미지는 기존 Qwen3-VL 비전 인코더와 Qwen VAE 양쪽에 전달한다. 비전 입력은 최대 384² 면적으로 줄이고 이미지 토큰과 Krea 텍스트 템플릿을 함께 인코딩한다. VAE 참조는 최대 1024² 면적과 16픽셀 배수에 맞추고 posterior mean을 정규화·패킹하여 생성 토큰 뒤에 추가한다. 공개 워크플로처럼 CFG의 긍정·부정 분기 모두에 참조를 전달하며 위치 축 0의 인덱스 1과 0 시점 변조를 사용한다. AI Toolkit의 LoRA 이름은 Diffusers 계층으로 대조해 병합하지 않고 적용하며 `--controlnet-scale`은 LoRA 잔차를 조절한다. 0일 때도 이미지 조건 자체는 유지된다.

`python3 scripts/provision_krea2.py --role pose`는 이 포즈 어댑터의 고정된 공개 파일을 별도 준비하며 `--check`로 전체 크기와 SHA-256을 검증할 수 있다. 어댑터는 Apache-2.0으로 표시되어 있지만 베이스 모델에는 별도의 Krea 2 Community License가 적용된다. 입력은 이미 준비된 포즈 맵이어야 하며 자동 포즈 추정 기능은 제공하지 않는다. 공급이나 구조 검사는 사전 학습 모델의 생성 결과를 검증하지 않는다.

텍스트 인코더가 베이스 모델에 없으면 기존 `fallback_llms`의 Qwen3-VL 가중치를 언어·비전 분기 모두에 로드한다. 모델이 자체 인코더를 제공하면 해당 인코더에 비전 가중치가 있어야 한다. 공개 포즈 워크플로의 참조 전용 어텐션에 맞춰 기본 `--krea2-edit-reference-attention isolated`에서는 참조 토큰이 다른 참조만 보도록 마스크한다. 실제 KV 캐시는 만들지 않고 참조를 각 단계에서 다시 계산한다. 비격리 학습을 한 다른 AI Toolkit 편집 LoRA는 `--krea2-edit-reference-attention joint`를 선택한다. 격리 마스크는 배치별 총 8천만 원소로 제한한다. 명시적 시드와 `--offload none`이 필요하며 부분 적용 구간, 다른 LoRA 및 영역별 프롬프트와의 조합은 지원하지 않는다. 자동 포즈 추출은 제공하지 않으며 호환되는 준비된 포즈 맵이 필요하다.

```sh
iild-generate --backend diffusers --model /absolute/path/krea2-turbo \
  --controlnet-format krea2-ostris-edit --controlnet /absolute/path/krea2-pose.safetensors \
  --control-image /absolute/path/openpose.png --controlnet-scale 0.9 \
  --prompt 'a dancer in a sunlit studio' --seed 42 --output-dir resources/outputs/reference/krea2-pose
```

이 명령은 변경 사항이 포함된 SDK 설치본을 사용한다. 소스 작업 트리에서는 필요한 Python 패키지가 있는 환경으로 `python reference/diffusers/generate_any.py`를 직접 실행하고 위 옵션 중 `--backend diffusers`만 생략할 수 있다.

공개 포즈 파일 헤더의 256개 LoRA 대상과 실제 Diffusers 트랜스포머의 형태, 기존 Qwen3-VL 리소스의 713개 언어·비전 텐서 형태를 대조했다. 공개 포즈 파일 전체의 SHA-256과 512개 텐서 값, 기존 Qwen3-VL의 비전 텐서 315개 값의 유한성도 확인했다. 소형 무작위 Qwen·VAE·Krea 모델과 합성 LoRA를 함께 사용하는 테스트는 이미지 조건, 0 시점 참조, CFG, 시드 반복과 후크 복원을 확인한다. 격리 참조의 두 블록 출력은 참조 전용 별도 실행과 대조했다. 공개 포즈 가중치를 사전 학습 베이스 모델에 결합한 이미지 생성, 품질 및 원본 출력 재현은 검증하지 않았다.

`--krea2-control-auto-bucket`은 공개 깊이 파이프라인처럼 입력 이미지의 종횡비와 로그 거리로 가장 가까운 해상도를 선택한다. 후보는 1024×1024, 896×1152, 1152×896, 832×1216, 1216×832, 768×1344, 1344×768, 704×1472, 1472×704이다. 선택은 깊이 추정과 중앙 자르기 전에 이뤄지고 해당 크기에 맞게 샘플링 설정도 갱신된다. 이 옵션은 `--width`, `--height` 또는 `--pipeline-inputs`의 해당 값과 함께 쓸 수 없다. 선택하지 않으면 기존 기본값과 명시 크기를 유지한다. `--print-config`는 입력을 디코딩하지 않으므로 크기를 미확정으로 표시하며, 실제 실행 결과에는 입력과 선택된 출력 크기가 기록된다.

별도의 [Krea 2 색상·깊이 어댑터](https://huggingface.co/darask0/Krea-2-color-depth-control)는 `--controlnet-format krea2-color-depth`로 명시적으로 선택한다. `--controlnet`은 해당 Diffusers 형식 safetensors, `--control-image`는 근거리일수록 밝은 깊이 맵, `--control-color-image`는 색상 영역 맵이며 둘 중 하나 이상이 필요하다. `--control-preprocess depth`는 깊이 입력을 기존 로컬 깊이 모델로 생성한다. 색상은 최근접, 깊이는 선형 보간으로 출력 크기에 맞추며 Qwen Image VAE의 posterior mean을 정규화·패킹한다. 제어 맵별 종료값은 `--control-color-end`, `--control-depth-end`로 지정한다. LoRA 강도는 `--controlnet-scale`이고 확장 입력 투영은 항상 전체 강도이다. 명시적 시드와 `--offload none`이 필요하다. `--lora`와 `--lora-scale`로 별도 스타일 LoRA를 함께 사용할 수 있다. PEFT 래퍼를 원래 선형 계층으로 식별하고 입력 투영의 스타일 잔차도 유지한다. 소형 무작위 모델과 합성 가중치에 대한 실제 PEFT 로더 테스트는 입력 투영과 어텐션 투영을 각각 또는 동시에 수정한 스타일 LoRA에 대해 두 어댑터의 영향, 시드 반복과 오류 후 복원을 확인한다. 개별 사전 학습 가중치 조합의 호환성과 이미지 품질을 보증하지는 않는다. 단일 깊이 Control-LoRA와는 텐서 이름, 입력 투영 크기 및 VAE 표본 방식이 다르므로 형식을 바꿔 지정할 수 없다.

공개 모델 카드의 263개 선형 계층, rank 64와 입력 투영 `[6144, 192]` 설명은 실제 Diffusers Krea 2 모델을 메타 장치에 구성하여 합성 형상 계약으로 대조한다. 이 검사는 공개 가중치 파일의 개별 텐서 이름이나 값을 검증하지 않는다. 해당 저장소가 인증 없이 파일 헤더 요청에 401을 반환하여 그 파일 자체의 로딩과 전체 이미지 생성은 검증되지 않았다.

```sh
python reference/diffusers/generate_any.py --model /absolute/path/krea2-diffusers \
  --controlnet-format krea2-color-depth --controlnet /absolute/path/krea2-control.safetensors \
  --control-image /absolute/path/depth.png --control-color-image /absolute/path/color.png \
  --control-depth-end 0.75 --control-color-end 0.5 --seed 42 \
  --prompt 'A sunlit studio' --output-dir resources/outputs/reference/krea2-control
```

이 경로는 공개 모델 카드의 3배 입력 투영, 색상·깊이 잠재값 및 Diffusers 계층별 LoRA 계약을 구현한다. 자동 버킷을 선택하면 깊이 이미지가 있으면 그것을, 없으면 색상 이미지를 기준으로 크기를 선택한다. 소형 무작위 Krea 2 트랜스포머·VAE와 합성 어댑터를 이용한 수치 테스트는 제어 적용, 같은 시드의 반복, 입력별 적용 구간 및 실행 후 후크 복원을 확인한다. 공개 어댑터 파일을 현재 환경에서 가져올 수 없어 실제 체크포인트의 키·메타데이터와 사전 학습 모델의 이미지 생성 품질은 아직 확인하지 못했다. 선택한 파일의 모든 계층 이름과 크기는 로드 시 실제 트랜스포머와 엄격하게 대조한다.

Krea 2에서 `--krea2-regions /absolute/path/regions.json`을 사용하면 아래 MOSS 예제와 같은 `map`, `regions`, `isolate_until`, `layout: "native"`, `control` JSON을 읽는다. JSON `control`의 어댑터와 색상·깊이 경로는 명시적 명령행 옵션보다 낮은 우선순위로 선택되며, 색상·깊이 형식이 자동 선택된다. 전역 프롬프트와 영역 프롬프트는 각각 기존 Qwen 텍스트 인코더로 인코딩하고, 텍스트 리파이너와 이미지·텍스트 공동 어텐션에 지역 마스크를 적용한다. 각 이미지 토큰은 전역 및 자기 영역 텍스트에 직접 연결되며 초기 `isolate_until` 구간에는 이미지 영역 사이의 직접 어텐션도 제한된다. 이후 이미지 간 직접 어텐션을 복원한다. Raw의 CFG 부정 프롬프트는 별도로 인코딩하고 같은 시퀀스 길이로 패딩하며 지역 마스크를 적용하지 않는다. 직접 제공한 임베딩, 배치 문자열 프롬프트, 영역별 LoRA, `native` 외 레이아웃은 이 경로에서 지원하지 않는다. 마스크는 8천만 원소로 제한한다. 이 구현은 공개 JSON 형식을 수용하지만 Hayate 원본과 동일한 결과를 보장하지 않는다.

```sh
iild-generate --backend diffusers --model /absolute/path/krea2-diffusers \
  --krea2-regions /absolute/path/regions.json --prompt 'two people in a park' \
  --seed 42 --output-dir resources/outputs/reference/krea2-regions
```

소형 실제 Diffusers Krea 2 트랜스포머와 합성 색상·깊이 어댑터를 사용한 테스트는 Turbo와 CFG Raw의 영역 라우팅, 다른 영역 문구의 직접 격리, 색상 맵 변경, 시드 재현, 부정 프롬프트 패딩 및 후크 복원을 확인한다. 사전 학습 가중치로 동일한 이미지를 생성한 검증은 아니다.

Krea 2 모델에 텍스트 인코더 가중치가 없으면 범용 러너는 기존 `fallback_llms`의 Qwen3-VL-4B safetensors와 고정된 공식 설정을 선택한다. `model.language_model.*` 텐서만 `Qwen3VLTextModel`에 마운트하여 프롬프트 인코딩에 필요 없는 비전 가중치를 적재하지 않는다. 토크나이저가 없으면 기존 ComfyUI의 Qwen 토크나이저에서 가져온 세 리소스를 `Qwen2Tokenizer`로 로컬 로드한다. 어휘는 공식 Qwen3-VL과 바이트 단위로 같고 병합 순서와 26개 추가 토큰도 일치한다. 공식 토크나이저와 영어·한국어·일본어·이모지·특수 토큰 예시의 실제 토큰 ID를 대조했으며 Krea 2 접두부는 34토큰이다. 토크나이저 설정의 채팅 템플릿은 다르지만 Krea 2 파이프라인은 자체 고정 템플릿을 사용한다. 모델에 구성 요소가 있으면 이를 우선하며 불완전한 구성 요소를 대체품으로 숨기지 않는다. 완전한 사전 계산 임베딩이 있으면 대체 인코더와 토크나이저가 필요하지 않다. 언어 텐서 전체의 이름·크기·유한값 및 선택 정밀도를 확인하고 리소스와 설정의 식별자를 출력에 기록한다. 무결성 모드가 `metadata`이면 가중치 해시를 검증했다고 표시하지 않는다. 소형 Qwen3-VL 전체 모델과 이 로더의 텍스트 전용 모델에서 모든 은닉 상태가 정확히 일치했으며 기존 리소스 헤더의 398개 언어 텐서도 공식 모델 구조와 일치했다. 이는 전체 사전 학습 가중치로 이미지 생성한 검증이 아니다. 인코더와 토크나이저의 설치 출처는 [리소스 문서](../resources/README.md)에 있다.

추가 오프라인 비교 테스트는 고정된 공식 MOSS 파이프라인 원본과 내장 구현에 같은 소형 무작위 가중치와 색상·깊이 제어 어댑터를 적용한다. 네이티브 Canny 또는 소형 Depth Anything 힌트, CFG 켜짐·꺼짐 및 제어 종료 구간을 조합한 CPU FP32 조건에서 최종 잠재 출력이 비트 단위로 같고 제어 단계가 일치함을 확인한다. 비교는 영역 마스크 없는 경로에 한정되며 사전 학습 모델, 다른 장치나 생성 이미지의 일치를 보증하지 않는다. 원본 리비전과 SHA-256은 `resources/test-fixtures/moss-image-pipeline-upstream-85975793.py` 및 테스트에 기록한다.

MOSS-Image는 공식 Apache-2.0 구현을 `reference/diffusers/vendor_moss/`에 고정하여 실행한다. MPS가 float64를 지원하지 않으므로 시간 임베딩의 float64 삼각함수는 공식 MPS 구현과 같이 CPU에서 계산하며, 선택적인 영역 어텐션 마스크 지원도 추가되어 있다. 공식 모델 디렉터리의 `pipeline_moss_image.py`는 실행하지 않으며, 트랜스포머·스케줄러·토크나이저·텍스트 인코더·VAE 선언이 검증된 계약과 일치해야 한다. 누락된 VAE에는 기존 FLUX.2 VAE 리소스 정책을 적용한다. MOSS 전용 로더가 선택된 VAE를 FP32로 한 번만 로드하고 채널 및 배치 정규화 계약을 검증한다. 다운로드된 패키지의 임의 Python 코드를 신뢰하는 일반적인 custom-pipeline 경로는 추가하지 않는다.

MOSS 스케줄러는 자체 시프트와 학습 격자에 대한 최근접 스냅을 적용한 뒤 결정적 Euler 갱신을 사용한다. 스케줄러 설정에서 동적 시프트, 시그마 역전, 종료값 재조정, Karras·지수·베타 시그마 재구성 또는 확률적 샘플링이 활성화되어 있으면 대용량 모델을 로드하기 전에 거부한다. Diffusers가 이 하위 클래스의 미지원 설정을 무시한다고 경고하더라도 내부 설정에 남은 값이 상위 스케줄러의 동작을 바꿀 수 있기 때문이다. MOSS 자체 `shift`와 `training_shift` 설정은 그대로 사용한다. 검사는 지원되는 합성 파이프라인과 공식 기본 설정을 대상으로 하며 사전 학습 이미지의 품질을 입증하지 않는다.

MOSS는 공식 Diffusers 디렉터리 외에 부동소수점 또는 Comfy INT8 tensorwise ConvRot 트랜스포머 단일 safetensors를 `--model`로 읽는다. `--model-config` 또는 고정된 `--model-config-preset moss-image-sft`에는 MOSS 인덱스와 변환기·스케줄러 설정이 필요하다. 합쳐진 블록 Q·K·V의 가중치와 행별 스케일을 순서대로 나누고 이름·크기·양자화 메타데이터 및 스케일 값을 검증하며 변환된 체크포인트를 디스크에 만들지 않는다. INT8 가중치는 CPU에 보관하고 활성값을 ConvRot 및 양자화해 선형 연산을 수행한다. 다른 양자화 형식은 명확히 거부한다.

파일에 `comfy_attention.config`가 있으면 해당 레이어와 제한된 JSON을 검사한다. Comfy의 어텐션 커널 선호는 가중치 양자화와 별개이며, 이 러너는 해당 커널이 없으므로 PyTorch SDPA를 사용한다. 경고와 `model.single_file_loading.attention`에 요청된 방식·레이어 수·선택된 대체 방식을 기록한다. 따라서 선호 커널을 사용한 출력과 수치적으로 같다고 주장하지 않는다. [ComfyUI 양자화 명세](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md)는 사용 불가능한 선호 방식에 대해 일반 어텐션을 유지하도록 설명한다.

별도 Society 리소스 카탈로그에 고정 설정 디렉터리 자체가 없으면 설치된 SDK의 동일한 해시 고정 설정을 사용한다. Society에 MOSS Qwen3.5 인코더·구성·토크나이저 등록이 아예 없는 경우에는 설치된 SDK 카탈로그에서 같은 구성 요소를 찾고 그 크기와 해시를 검증한다. Society에 해당 항목이 등록되었지만 파일이 없거나 잘못되었으면 SDK 항목으로 덮어쓰지 않는다. 설정만으로 모델이 완성되지는 않는다. Qwen3.5 텍스트 인코더와 토크나이저 및 호환 FLUX.2 VAE가 로컬 카탈로그 또는 명시적 모델 구성에 필요하며, 누락된 인코더는 가중치를 로드하기 전에 별도 오류로 보고한다.

MOSS의 각 디노이징 단계가 완료되면 `iild-diffusers-progress-v1` 이벤트를 출력한다. 일반 및 rapid 경로 모두 실제 완료 단계만 보고하며, 이벤트는 작업의 미리보기 이미지나 모델 로딩 중의 세부 진행률을 의미하지 않는다.
일반 회귀 검사는 합성 가중치와 입력을 사용한다. 로컬에 사전 학습 FLUX.2 VAE 파일이 있다는 이유만으로 이를 실행하지 않으며, 해당 별도 검사에는 `IILD_VERIFY_PRETRAINED_VAE=1`을 명시해야 한다.

프로젝트 루트에서 `python3 scripts/provision_moss_transformer.py`를 실행하면 고정된 공개 MOSS SFT BF16 원본을 `resources/models/moss_image_sft_bf16.safetensors`에 준비할 수 있다. 이어받기를 지원하며 전체 크기와 SHA-256이 확인된 파일만 게시한다. `--check`는 다운로드 없이 기존 파일을 검증한다. 약 10.16GB가 필요하고 파일은 Git에 포함되지 않으며 생성 중 자동 다운로드되지 않는다. 이 파일은 사용자가 명시적으로 `--model`로 선택해야 한다. 텍스트 인코더·VAE·제어 어댑터는 별도의 리소스이며, 파일 준비만으로 사전 학습 이미지 출력이나 하드웨어 적합성이 검증되는 것은 아니다. 출처와 라이선스는 [리소스 문서](../resources/README.md)에 기록한다.

색상·깊이 어댑터도 `python3 scripts/provision_moss_control.py`로 별도 준비하고 `--controlnet resources/models/moss_image_color_depth_control.safetensors`로 명시한다. 약 313MB이며 같은 방식으로 이어받기·전체 해시 검증·`--check`를 지원한다. [공개 모델 카드](https://huggingface.co/darask0/MOSS-Image-color-depth-control)는 이 어댑터가 MOSS Base로 학습되었고 SFT와도 사용되었다고 설명한다. 이는 현재 SDK에서 특정 출력의 품질이나 일치가 검증되었다는 뜻은 아니다.

선택적으로 `--moss-transformer-storage cpu-stream`을 지정하면 MOSS 단일 파일 트랜스포머의 이미지 블록과 텍스트 블록 선형 가중치를 원본 정밀도 그대로 CPU에 두고, 실행되는 계층만 입력 장치와 계산 정밀도로 옮긴다. Q·K·V 분할은 원본 텐서의 뷰로 유지한다. 나머지 트랜스포머 구성 요소, 텍스트 인코더, VAE, 제어 어댑터와 활성값은 별도 메모리를 사용하며, 반복 전송 때문에 느릴 수 있다. 이 옵션은 `--offload none`을 요구하고 디렉터리 체크포인트에는 적용되지 않는다. 스트리밍 계층은 불완전한 자리표시자 가중치가 저장되지 않도록 `state_dict` 직렬화를 거부한다. `request.moss_transformer_storage`와 `model.single_file_loading`은 선택 방식, 스트리밍 계층 수, 원본 바이트 및 최대 단일 계층 복원 바이트를 기록한다. 이는 총 프로세스·장치 메모리의 실측값이나 특정 메모리 용량에서의 실행 보장이 아니다.

```sh
python reference/diffusers/generate_any.py --model /absolute/path/moss-image-sft.safetensors \
  --model-config-preset moss-image-sft --moss-transformer-storage cpu-stream \
  --controlnet /absolute/path/moss-control.safetensors --control-image /absolute/path/depth.png \
  --seed 42 --prompt 'a sunlit studio' --output-dir resources/outputs/reference/moss-control
```

소형 원본 형식 트랜스포머·합성 제어 어댑터 및 텍스트 인코더 테스트는 밀집 방식과 스트리밍 방식의 잠재 출력을 CPU FP32·BF16과 사용 가능한 MPS FP32에서 대조한다. CPU에서는 합성 INT8 텍스트 인코더, 기존 FLUX.2 VAE 및 Canny·깊이 전처리 조합도 비교한다. 전체 크기 사전 학습 모델의 메모리 사용량, 실행 속도와 이미지 품질은 검증하지 않았다.

별도의 비양자화 Comfy-Org Qwen3.5 단일 파일은 `--moss-text-encoder /absolute/path/qwen3.5_9b_bf16.safetensors`로 지정한다. 텍스트 생성에 필요한 언어 분기의 텐서만 엄격히 로드하며, 파일의 비전·출력 헤드·MTP 텐서는 사용하지 않는다. 모델에 `text_encoder/config.json`이 없으면 고정된 공식 MOSS Qwen3.5 설정을 사용한다. 모델 토크나이저가 없으면 기존 ComfyUI Qwen3.5 토크나이저 리소스를 사용한다. 자체 파일이 일부라도 있는 토크나이저는 자동으로 덮어쓰지 않는다. 토크나이저의 기본 채팅 템플릿은 공식 파일과 다르지만 MOSS 구현이 명시하는 `enable_thinking=True`와 생성 프롬프트 설정에서 검증된 예제의 텍스트와 토큰 ID가 같다. 설정·토크나이저의 해시를 검증하고 선택 결과를 보고서에 기록한다. 이 옵션은 MOSS 텍스트 프롬프트용이며 Qwen3-VL 토크나이저를 Qwen3.5의 대체품으로 간주하지 않는다.

원본 MOSS-Image-SFT 트랜스포머 단일 파일에는 `--model-config`로 제공할 수 있도록 `resources/model-configs/moss-image-sft/`에 공식 모델 인덱스와 트랜스포머·스케줄러 설정을 포함한다. 설치본에서는 `share/iiLocalDiffusion/resources/model-configs/moss-image-sft/`를 사용한다. 이 디렉터리는 설정만 포함하며 텍스트 인코더나 모델 가중치를 포함하지 않는다. 다른 구조의 모델은 해당 모델에 맞는 설정을 명시해야 한다.

단일 파일에 `--model-config-preset moss-image-sft`를 지정하면 이 설정을 해시 검증 후 선택한다. Krea 2는 `--model-config-preset krea2-raw` 또는 `krea2-turbo`를 지정할 수 있다. `--pipeline-class Krea2Pipeline`과 명시적인 `--krea2-variant raw` 또는 `turbo`를 함께 주면 해당 Krea 2 설정이 자동 선택된다. FLUX.1 Schnell은 `--model-config-preset flux1-schnell`로 기존 CLIP·T5·VAE 구성과 호환되는 설정을 선택할 수 있다. Schnell 프리셋은 Diffusers 모델 기본 구조와 스케줄 계약을 기준으로 작성한 것으로, Dev나 다른 FLUX 모델에 사용해서는 안 된다. MOSS의 다른 변형이나 Krea의 Raw·Turbo를 모델 이름만으로 추측하지 않는다. 프리셋과 `--model-config`는 동시에 쓸 수 없으며, 보고서의 `model_config_status`에 선택 방식을 기록한다.

Schnell 프리셋은 원본 트랜스포머의 텐서 이름 776개와 형태를 로드 전에 검사하며, 별도 배율을 적용해야 하는 FP8 형식은 거부한다. 공개 Comfy-Org BF16 트랜스포머 단일 파일과 FP8 통합 단일 파일의 헤더를 이 계약에 대조했다. 전자에는 텍스트 인코더와 VAE가 없어 기존 리소스를 선택하고, 후자는 내장 구성 요소를 우선한다. 통합 단일 파일의 T5가 공유 임베딩만 저장한 경우 일부 Diffusers 버전은 인코더의 임베딩 별칭을 미할당 상태로 두므로, 이 별칭을 공유 임베딩에 연결하고 CLIP과 T5의 다른 미할당 텐서는 거부한다. 적용 결과는 `model.flux_t5_embedded_loading`에 기록된다. 헤더와 소형 합성 모델 검사는 전체 가중치의 수치 또는 사전 학습 이미지 출력 일치를 보증하지 않는다.

`--moss-text-encoder`는 Comfy-Org Qwen3.5 INT8 ConvRot 단일 파일도 받는다. 언어 분기의 INT8 가중치와 배율을 CPU에 유지하고, 임베딩은 요청된 토큰 행만 역회전한다. 선형층은 활성값을 회전하고 행별 INT8 양자화한 뒤 CPU에서 정수 행렬곱을 수행한다. 출력은 선택된 장치와 정밀도로 이동한다. 이 이식성 우선 경로는 가속 INT8 커널보다 느릴 수 있으며 `--offload none`이 필요하다. 원본 파일의 텍스트 분기에 있는 임베딩 1개와 선형층 200개에 대해 형태·행별 배율·메타데이터를 엄격히 검사하며 비전 분기는 읽지 않는다.

오프라인 ConvRot 수치 시험은 고정한 Apache-2.0 Comfy Kitchen 원본 eager 함수와 소형 합성 행렬의 FP32·FP16·BF16 결과를 비교한다. 선택적인 로컬 가중치 감사는 설치된 Qwen3.5 INT8 파일의 첫 MLP 투영에서 11개 실제 행과 배율을 읽어 같은 원본 함수와 비교한다. 원본 소스와 실행 조건은 [리소스 문서](../resources/README.md)에 기록한다. 이 검사는 선택한 행과 CPU 연산에 한정되며 사전 학습 인코더 전체, 다른 가속기 또는 MOSS 이미지 생성의 일치를 뜻하지 않는다.

패키지에 텍스트 인코더 가중치가 없다면, 별도로 확보한 리소스의 INT8 파일을 해시 검증 후 자동 선택한다. 명시적으로 지정한 파일과 모델 패키지의 가중치가 우선한다. 일부만 존재하는 패키지 가중치도 자동 교체하지 않는다. 실행 보고서의 `moss_text_encoder.status`는 `explicit` 또는 `fallback-int8-convrot`을 기록한다. 이 대체 경로는 양자화된 인코더를 사용하므로 BF16 인코더와 동일한 이미지를 보장하지 않는다. 파일은 자동 다운로드하지 않으며 `python3 scripts/provision_moss_text_encoder.py`로 확보하거나 `--check`로 확인한다. 소형 합성 가중치로 FP32·FP16·BF16 단위 연산, Qwen 모델 수치 및 범용 러너의 제어 어댑터 결합을 검사했다. 공개 대형 가중치를 이용한 전체 이미지 생성과 다른 구현의 최종 이미지 일치는 아직 검증하지 않았다.

```sh
iild-generate --backend diffusers --model /absolute/path/moss_image_sft_bf16.safetensors \
  --model-config /absolute/path/to/resources/model-configs/moss-image-sft \
  --moss-text-encoder /absolute/path/qwen3.5_9b_bf16.safetensors --dtype bfloat16 \
  --controlnet /absolute/path/moss-control.safetensors --control-image /absolute/path/depth.png \
  --seed 42 --prompt 'a sunlit studio' --output-dir resources/outputs/reference/moss-single
```

공개 MOSS SFT 단일 파일의 379개 텐서와 42개 결합 QKV는 실제 MOSS 모델의 463개 텐서 크기와 대조했다. 또한 공개 단일 파일과 공식 분할 파일의 첫 블록 Q·K·V에서 각각 첫 4KiB가 같은 순서와 값인지 대조했다. 공개 Qwen3.5 단일 파일의 언어 텐서 426개도 실제 텍스트 모델의 이름과 크기에 대조했다. 소형 합성 모델은 디렉터리·단일 파일 간 제어 출력과 전체 Qwen3.5·언어 전용 인코더의 텍스트 출력을 CPU 및 사용 가능한 MPS에서 비교한다. 2026-10-31에는 실제 SFT 트랜스포머 전체를 BF16 CPU 스트리밍으로 로드해 262개 스트리밍 계층과 모든 가중치의 유한성을 확인했다. 실제 INT8 언어 인코더, 토크나이저, VAE 및 스케줄러를 해당 트랜스포머와 한 파이프라인에 함께 로드했고 실제 어댑터의 262개 대상 연결 계획도 검사했다. 별도 검사에서는 합성 잠재값과 실제 베이스·어댑터를 사용해 실행 후크 등록과 예외 시 복구를 확인했다. 사전 학습 모델의 전방 연산 및 이미지 품질은 검증하지 않았다.

[MOSS 색상·깊이 어댑터](https://huggingface.co/darask0/MOSS-Image-color-depth-control)는 `--controlnet`으로 로컬 safetensors를 선택하고, `--control-image`에 준비된 깊이 맵, `--control-color-image`에 색상 영역 맵을 넣는다. 둘 중 하나 이상이 필요하다. 깊이 맵은 가까울수록 흰색인 회색조여야 한다. 기본 `semantic` 전처리는 깊이 입력을 회색조로 변환해 RGB 세 채널에 복제한다. 컬러 영상을 회색조로 바꾸는 것은 깊이 추정이 아니다. 제어 픽셀은 FP32에서 [-1, 1]로 정규화한 다음 VAE 정밀도로 변환한다. FLUX.2 VAE의 posterior mean을 2×2 패킹과 배치 정규화한 후 이미지 잠재 표현에 이어 붙인다. 기본 모드는 색상 맵은 색상 ID를 보존하도록 최근접, 깊이 맵은 선형 보간으로 출력 크기에 맞춘다. `--control-guidance-start`, `--control-color-end`, `--control-depth-end`는 맵별 적용 구간을 정하며 지정하지 않은 종료값은 `--control-guidance-end`를 따른다. 학습된 입력 투영은 전체 구간에 유지되고 `--controlnet-scale`은 LoRA에 적용된다. 어댑터 메타데이터의 `moss-base`와 `moss-sft`를 지원하고 실제 선언을 실행 결과에 기록한다. 실행에는 명시적 `--seed`와 `--offload none`이 필요하며, 다른 LoRA의 동시 적용은 현재 지원하지 않는다.

`--control-map-preprocess training`은 [공개 학습 코드의 `load_pixels`](https://huggingface.co/darask0/MOSS-Image-color-depth-control/blob/357d7cc1eefd746de28ed74741c79464e3d82daa/training/moss_control.py)에 맞춰 색상·깊이 입력을 각각 RGB로 유지하고, 출력 크기와 비율에 맞게 중앙 크롭하면서 Lanczos 보간한다. 이는 학습 코드의 이미지 공간 처리에 관한 선택이며 Hayate의 추론 구현이나 사전 학습 모델의 이미지 출력을 재현했다는 뜻은 아니다. 색상 경계가 섞일 수 있고 RGB 깊이 이미지는 자동 회색조 변환되지 않으므로 준비된 회색조 깊이 맵을 권장한다. 영역 프롬프트를 함께 쓰면 영역 ID는 같은 중앙 크롭 기하를 최근접 보간으로 적용해 색상 ID를 보존한다. 실행 보고서에 선택된 방식이 기록된다.

결과의 `control_execution.control_activity`에는 색상과 깊이 조건이 각각 적용된 0부터 시작하는 단계 번호가 기록된다. 같은 단계에서 CFG의 양·음성 분기나 여러 배치가 제어 투영을 반복 호출해도 단계 번호는 한 번만 기록한다. 이 기록은 제어 맵 적용 여부이며 입력 투영과 LoRA의 비활성화를 뜻하지 않는다. 네이티브 Canny의 Python 연결도 실제 공유 라이브러리와 합성 사각형으로 에지·RGB 채널·반복성을 확인한다.

범용 MOSS 로더는 현재 FLUX.2 VAE를 FP32로 로드한다. BF16 VAE를 직접 결합한 회귀 테스트에서도 픽셀의 FP32 정규화 후 BF16 변환 순서를 확인하며, 이는 현재 범용 로더가 BF16 VAE를 선택한다는 뜻은 아니다.

공개 [rapid-moss 4단계 어댑터](https://huggingface.co/darask0/MOSS-Image_4-step_adapter_lora)는 MOSS-Image-SFT와 함께 `--moss-rapid-adapter /absolute/path/rapid-moss`로 선택한다. 디렉터리에는 원본 `lora.safetensors`, `time_embedding.safetensors`, `dataspace.safetensors`가 모두 필요하며 파일 식별자를 기록하고 실행 전후로 확인한다. 원본처럼 LoRA를 병합하지 않고 FP32 잔차로 더하고 시간 임베딩과 두 변조 경로를 FP32로 교체한다. 고정 FP32 CPU 노이즈와 학습된 Q·step code로 데이터 공간 반복을 수행한다. 단계 수는 1–4이며 기본 4, guidance는 1이다. 명시적 시드, `--offload none`, BF16 또는 FP32 베이스가 필요하다. 네거티브 프롬프트, 별도 LoRA, SLG 및 중간 미리보기는 이 모드에서 사용할 수 없다. 권장 베이스는 SFT이며 선택한 로컬 체크포인트가 SFT인지 자동으로 입증하지 않는다. 색상·깊이 제어와 함께 사용하면 제어 구간은 이 반복 단계에 맞춰 적용된다. 실제 공개 가중치의 헤더 계약과 소형 합성 모델을 검사했으며, 사전 학습 가중치 조합의 시각적 재현을 의미하지 않는다.

소형 MOSS 트랜스포머와 합성 어댑터를 이용해 데이터 공간 반복식의 결과, 입력 변화, 시드 반복, 제어 종료 단계 및 요청 후 복원을 확인했다. 별도 BF16 테스트는 FP32 LoRA 잔차를 병합하지 않고 더하며 예외 발생 시에도 가중치와 시간 경로가 복원되는지 확인한다.

```sh
iild-generate --backend diffusers --model /absolute/path/moss-image-sft \
  --moss-rapid-adapter /absolute/path/rapid-moss --dtype bfloat16 \
  --controlnet /absolute/path/moss-control.safetensors \
  --control-image /absolute/path/depth.png --control-color-image /absolute/path/colors.png \
  --control-depth-end 0.5 --control-color-end 0.5 --seed 42 \
  --prompt 'a person in a sunlit studio' --output-dir resources/outputs/reference/moss-rapid-control
```

```sh
iild-generate --backend diffusers --model /absolute/path/moss-image-package \
  --controlnet /absolute/path/moss-control.safetensors \
  --control-image /absolute/path/prepared-depth.png \
  --control-color-image /absolute/path/region-colors.png \
  --control-depth-end 0.5 --control-color-end 0.5 --seed 42 \
  --prompt 'a person in a sunlit studio' --output-dir resources/outputs/reference/moss-control
```

입력 맵과 학습된 어댑터의 호환성, 라이선스, 사전 학습된 전체 모델의 생성 품질은 별도 검증이 필요하다. 작은 실제 FLUX.2 VAE 테스트는 컬러 깊이 입력과 그 회색조 변환이 같은 잠재 표현이 되며 색상 맵에는 RGB 차이가 유지되는지 확인한다. 로컬 Depth Anything 모델을 명시하면 원본 사진에서 깊이 맵을 계산할 수 있으며, 세그멘테이션 검출기는 포함하지 않는다. Canny는 에지 맵을 계산할 뿐 깊이 맵을 계산하지 않으므로, 이를 depth 입력에 사용하는 것은 공개 depth 어댑터의 학습 입력을 재현하지 않는다. 출처: [MOSS 공식 구현](https://github.com/OpenMOSS/MOSS-Image), [어댑터 모델 카드](https://huggingface.co/darask0/MOSS-Image-color-depth-control).

MOSS에서 `--moss-regions /absolute/path/regions.json`은 기존 색상 맵의 RGB ID와 영역별 프롬프트를 결합한다. JSON은 공개 모델 예제의 `map`, `regions`, `isolate_until`, `layout: "native"`, 선택적인 `control` 형식을 받는다. 상대 파일 경로는 JSON 디렉터리 기준이다. `control`의 `adapter`, `color_map`, `depth`, `strength`, `color_end`, `depth_end`는 대응하는 명령행 옵션이 없을 때 사용한다. 명령행 옵션이 우선한다. `loras`는 빈 배열만 허용하며, 영역별 LoRA나 다른 레이아웃은 거부한다. 하나의 전역 문자열 프롬프트, `--offload none`, 16의 배수 크기가 필요하다.

Krea 2와 기본 MOSS 영역 맵은 색상 제어와 동일하게 먼저 출력 크기로 최근접 리사이즈한 후 잠재 토큰 크기로 최근접 축소한다. MOSS `training` 모드에서는 같은 중앙 크롭 기하를 사용한다. 각 `#RRGGBB`와 정확히 일치하는 픽셀을 해당 영역으로 배정하며, 축소된 맵에 없는 영역 색상은 오류이다. 원본을 잠재 크기로 곧바로 축소하던 이전 방식과는 출력 크기가 다른 입력에서 선택된 픽셀이 달라질 수 있다. 이 정렬은 토큰의 색상 ID 선택에 관한 것이며 VAE의 공간적 혼합까지 없앤다는 뜻은 아니다. 전역 및 영역 텍스트는 각각 기존 MOSS 텍스트 인코더로 인코딩한다. 이미지 토큰은 전역 텍스트와 자기 영역 텍스트를 직접 참조하고 영역 텍스트는 자기 영역의 이미지 토큰을 직접 참조한다. 텍스트 자체의 어텐션은 프롬프트별로 분리된다. `isolate_until` 구간에는 서로 다른 이미지 영역의 직접 어텐션도 제한하고 이후에는 이미지 간 어텐션을 연다. 전역 텍스트를 통한 간접 영향까지 배제하는 보장은 없다. 일반 MOSS 및 rapid-moss 샘플러와 색상·깊이 제어를 함께 사용할 수 있으며 영역 맵과 JSON의 파일 변경 여부도 검사한다. 제곱 어텐션 마스크는 8천만 원소로 제한한다. 이 방식은 공개 예제의 입력 형식을 수용하는 iiLocalDiffusion 구현이며, 접근할 수 없는 Hayate 참조 구현과 동일한 알고리즘이나 이미지 출력을 보장하지 않는다. 소형 합성 모델 테스트는 마스크 연결, 초기 격리, CFG의 부정 분기, 같은 시드 반복, 4단계·제어 어댑터와의 조합을 확인한다.

```json
{
  "map": "colors.png",
  "regions": [
    {"color": "#ff0000", "prompt": "a person in a red coat"},
    {"color": "#00ff00", "prompt": "a person holding a camera"}
  ],
  "isolate_until": 0.2,
  "layout": "native",
  "control": {"adapter": "moss-control.safetensors", "color_map": "colors.png", "depth": "depth.png", "color_end": 0.5, "depth_end": 0.5}
}
```

위 JSON을 `regions.json`으로 저장한 뒤 MOSS 실행에 `--moss-regions /absolute/path/regions.json --prompt 'two people in a park' --seed 42`를 더한다. 픽스처와 소형 모델 테스트는 사전 학습 체크포인트의 시각적 품질이나 공개 예제 이미지의 재현을 입증하지 않는다.

FLUX.2 dev의 [Fun ControlNet Union](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union)은 FLUX.1과 다른 단일 safetensors 어댑터이다. `Flux2Pipeline`에서 `--controlnet`과 `--control-image`를 지정하면 기존 FLUX.2 VAE의 posterior mean을 2×2 패킹·정규화하고 마스크와 인페인트 잠재값을 결합한다. 두 입력을 생략하면 해당 조건은 비어 있다. 제어 이미지는 공개 구현과 같이 Diffusers `VaeImageProcessor`로 FP32 정규화한 다음 VAE 정밀도로 변환한다. 제어 브랜치는 Diffusers의 FLUX.2 블록을 사용하고 결과를 이중 블록 0, 2, 4, 6에 적용한다. 로컬 가중치의 이름과 크기는 실제 베이스 블록에 대해 전부 검사한다. `--control-preprocess canny`, `--controlnet-scale`, 시작·종료 구간을 지원하며 `--seed`와 `--offload none`이 필요하다. 시작·종료 구간은 다른 Diffusers ControlNet처럼 전체 단계 구간이 지정한 범위에 포함되는 단계에만 적용된다. 예를 들어 2단계 중 종료값이 0.4이면 첫 단계의 끝 0.5가 경계를 넘으므로 제어 단계가 없다. 참조 이미지는 `--width`와 `--height`를 명시하고 `--pipeline-inputs '{"image":{"image_path":"/absolute/path/reference.png","mode":"RGB"}}'`로 전달한다. 공개 구현처럼 생성 이미지의 제어 토큰 뒤에 참조 이미지용 0 토큰을 붙인다. 인페인트는 `--control-inpaint-image /absolute/path/source.png`와 `--control-mask-image /absolute/path/mask.png`로 지정한다. 흰색 마스크는 재생성 영역이며 검정색은 소스 이미지를 조건으로 제공하는 영역이다. 공개 구현처럼 역마스크와 마스크를 적용한 입력의 VAE 잠재값을 제어 조건에 결합한다. 두 입력은 각각 선택 사항이며 생략 시 공개 구현의 기본 마스크 및 빈 잠재값을 사용한다. 별도의 FLUX.2 Diffusers LoRA는 `--lora`와 `--lora-scale`로 베이스 Transformer에 함께 적용할 수 있다. 이 LoRA는 독립적인 제어 브랜치에는 적용되지 않는다. 소형 합성 가중치와 실제 PEFT 로더로 두 효과, 시드 반복과 예외 시 제어 후크 복원을 확인했으며 특정 사전 학습 LoRA 조합의 품질은 검증하지 않았다. 어댑터의 라이선스는 SDK 코드 및 베이스 모델과 별개이며, 위 공개 가중치는 비상업 라이선스를 표시한다.

참조 이미지, 제어 이미지 및 인페인트 마스크를 함께 사용하는 소형 무작위 FLUX.2 파이프라인 테스트는 같은 시드 반복, 제어 이미지와 마스크에 따른 출력 변화 및 후크 복원을 확인한다. 사전 학습된 전체 모델의 출력 품질 검증은 아니다.

FLUX.2 Klein의 참조 제어 LoRA는 `--controlnet-format flux2-klein-reference`로 선택한다. `--controlnet`에는 로컬 LoRA safetensors, `--control-image`에는 준비된 힌트 또는 전처리할 원본 사진을 지정하고, `--pipeline-inputs`의 `image`에는 별도의 RGB 참조 이미지 한 장을 지정한다. 러너는 제어 힌트와 참조 이미지를 이 순서로 `Flux2KleinPipeline`에 전달하고 `--controlnet-scale`을 LoRA 강도로 적용한다. `--control-preprocess canny`와 `--control-preprocess depth`는 첫 번째 이미지에만 적용한다. 다른 `--lora`, 부분 제어 구간 및 Fun 전용 인페인트 옵션은 이 경로에서 지원하지 않는다.

```sh
iild-generate --backend diffusers --model /absolute/path/flux2-klein-base-4b \
  --controlnet /absolute/path/flux2_klein_4b_refcontrol_depth.safetensors \
  --controlnet-format flux2-klein-reference --control-image /absolute/path/source.png \
  --control-preprocess depth --control-depth-model /absolute/path/depth-model \
  --control-reference-resize comfy-1mp \
  --pipeline-inputs '{"image":{"image_path":"/absolute/path/reference.png"}}' \
  --prompt 'refcontrol, a person in a garden' --controlnet-scale 0.9 --seed 42 \
  --output-dir resources/outputs/reference/klein-refcontrol
```

[공개 4B RefControl 모델 카드](https://huggingface.co/thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora)는 Base 4B, `refcontrol` 트리거와 깊이·참조 이미지의 순서를 명시한다. 고정한 공개 헤더의 160개 텐서는 설치된 Diffusers AI Toolkit 변환기와 실제 Klein 4B 구조에 대해 크기를 확인했다. 소형 무작위 Klein 파이프라인은 AI Toolkit 형식 LoRA, 네이티브 Canny와 소형 Depth Anything 힌트, 이미지 순서 및 같은 시드 반복을 검증한다. Diffusers의 참조 이미지 리사이즈와 깊이 전처리는 공개 ComfyUI 워크플로와 동일한 픽셀 결과를 보장하지 않으며, 사전 학습 가중치의 이미지 생성이나 품질도 검증하지 않았다. 어댑터별 학습 데이터와 베이스 호환성은 사용자가 확인해야 한다.

`--control-reference-resize comfy-1mp`는 전처리 후 두 RGB 이미지를 각각 `ImageScaleToTotalPixels` 워크플로의 1,048,576픽셀 기준, 원본 비율, 반올림 및 `nearest-exact` 규칙으로 크기 조정한다. 반올림 결과가 기준 면적을 조금 초과해도 Diffusers가 다시 Lanczos 축소하지 않도록 요청 범위에서만 막는다. FLUX.2 VAE의 참조 잠재 정규화도 기존 ComfyUI와 같은 배치 정규화 연산을 사용한다. 요청 종료나 오류 시 원래 메서드를 복원하며 보고서는 이미지의 전후 크기, 조정 직후 픽셀 해시와 정규화 방식을 기록한다. 생략하거나 `none`을 선택하면 기존 Diffusers 처리를 사용한다.

기존 FLUX.2 VAE는 배치 정규화와 2×2 잠재 패킹으로 압축 배율이 16이므로 ComfyUI와 Diffusers 모두 참조 이미지를 16의 배수로 중앙 크롭한다. 작은 FP32 VAE 테스트는 같은 RGB8 입력에서 크롭 픽셀의 일치와 정규화된 잠재값의 근접성, 같은 잠재값에 적용한 정규화 연산의 일치를 확인한다. 별도의 VAE 계산에는 작은 부동소수점 차이가 남을 수 있다. 테스트는 두 참조의 위치 인덱스 10·20도 확인한다. 이는 임의의 VAE나 장치에서 전체 생성 이미지가 같다는 증거는 아니다. 원본 깊이 검출과도 결과가 달라질 수 있다. 준비된 RGB8 깊이 맵을 입력하면 깊이 검출을 생략하고 해당 맵을 리사이즈한다.

공개 가중치의 safetensors 헤더는 Krea 2 depth 어댑터 리비전 `21889413aa7282a6e78bd510247cceccad034b24`의 224개 블록 대상, MOSS 색상·깊이 어댑터 리비전 `357d7cc1eefd746de28ed74741c79464e3d82daa`의 262개 대상, FLUX.2 Fun 2602 리비전 `b3dcd7836a0e926248dac3ccba8fc0853495764b`의 76개 텐서를 각각 구조 검사와 대조했다. Krea 2 depth의 전체 파일 SHA-256과 450개 텐서 값의 유한성을 확인했다. Krea 2 트랜스포머와 FLUX.2 제어 블록은 실제 Diffusers 클래스를 메타 장치에 구성해 공개 헤더의 크기와 대조했다. MOSS는 추가로 [Comfy-Org 배포본](https://huggingface.co/Comfy-Org/MOSS-Image/blob/f7a9772e063542422b66dd492c4e17aa64f7797b/loras/moss_image_color_depth_control.safetensors)의 전체 파일 SHA-256과 526개 텐서 값의 유한성을 확인하고, 메타 장치에서 만든 실제 MOSS 트랜스포머와 크기를 대조했다. 세 헤더의 고정 스냅샷은 `resources/test-fixtures/`에서 오프라인 회귀 테스트에 사용한다. 소형 무작위 트랜스포머·VAE·어댑터를 이용한 수치 테스트는 실행, 시드 반복 및 후크 복원을 확인한다. FLUX.2 Fun 2602의 전체 파일 SHA-256과 76개 텐서 값의 유한성도 확인했다. 사전 학습된 베이스 모델로 이미지를 생성한 검증은 아니다.

로컬 단일 파일 체크포인트에는 **로컬** Diffusers 구성 디렉터리 또는 위에서 설명한 명시적 내장 설정이 필요합니다. 필요한 구성 요소는 모델에서 제공하거나 위에 명시된 호환 대체 리소스로 충족해야 합니다. Krea 2와 MOSS의 지원되는 원본 트랜스포머 파일을 제외하면 선택된 클래스는 Diffusers의 `from_single_file`을 구현해야 합니다. 파이프라인의 디렉토리 형식을 지원하는 것이 모든 단일 파일 형식을 로드한다는 것을 의미하지는 않습니다. 특히, 트랜스포머 전용 파일은 필요한 VAE 와 텍스트 인코더가 없으면 완전한 파이프라인이 되지 않습니다. 클래스에 단일 파일 로더가 없으면 완전한 Diffusers 디렉토리를 제공하거나 워크플로 백엔드를 사용해야 합니다.

Krea 2는 Diffusers 파이프라인에 단일 파일 로더가 없어도 원본의 비양자화 트랜스포머와 Comfy-Org의 스칼라 배율 E4M3 FP8 트랜스포머 safetensors를 직접 읽을 수 있다. `--model-config` 디렉터리는 Krea 2 `model_index.json`, `transformer/config.json`, 스케줄러 및 필요한 구성 요소를 제공해야 한다. VAE, 텍스트 인코더와 토크나이저가 없다면 위에서 설명한 기존 Krea 2 대체 리소스를 사용한다. 원본 텐서 이름과 선택된 트랜스포머의 모든 이름·크기를 대조하고, 블록 변조 벡터를 변환한 뒤 메모리에 직접 로드한다. 선택한 정밀도로 변환 전후의 유한값을 확인하며 별도의 변환 체크포인트를 만들지 않는다. 공개 Comfy-Org E4M3 FP8 파일은 기본적으로 각 FP8 값과 스칼라 배율을 요청된 실행 정밀도로 변환해 일반 밀집 가중치로 로드한다. 선택적으로 `--krea2-fp8-storage cpu-stream`을 지정하면 트랜스포머 블록 및 텍스트 융합의 FP8 가중치를 CPU에 압축 상태로 유지하고, 각 선형층 실행 시 해당 계층만 계산 정밀도로 복원해 선택한 장치로 옮긴다. 이 방식은 `--offload none`을 요구하며 일반 LoRA만 함께 사용할 수 있다. FP8 가중치와 스케일이 별도로 보관되므로 불완전한 모델을 저장하지 않도록 이 스트리밍 모델의 `state_dict` 직렬화를 거부한다. 일반 LoRA 로딩 중 PEFT의 내부 상태 조회는 해당 모델과 호출 컨텍스트에 한해 허용하고, 성공이나 실패 후 다시 차단한다. 저장 가능한 모델이 필요하면 원본 체크포인트를 보존하거나 지원되는 밀집 로딩을 사용한다. VAE, 텍스트 인코더, 비양자화 가중치와 현재 계산 중인 선형층은 별도 메모리를 사용하며 반복 전송과 복원 때문에 느릴 수 있다. 네이티브 FP8 행렬 연산을 사용하지 않으므로 ComfyUI FP8 커널과의 비트 단위 결과 일치는 보장하지 않는다. 생성 보고서의 `model.single_file_loading`에 파일 형식, FP8 계층 수, 저장 방식, FP8 원본 바이트 수와 최대 단일 계층의 복원 바이트 수를 기록한다. 이 수치는 전체 프로세스나 GPU 메모리의 실측값이 아니다. MXFP8·NVFP4·INT8 및 다른 양자화 형식은 지원하지 않는다. 실제 Krea 2 베이스 가중치는 크므로 해당 정밀도의 모델과 나머지 구성 요소를 담을 충분한 메모리가 필요하다.

단일 파일용 설정으로 `resources/model-configs/krea2-raw/` 또는 `resources/model-configs/krea2-turbo/`를 `--model-config`에 지정할 수 있다. 이 설정은 공개 Diffusers Krea2 모델의 기본 구조와 공개된 지수 시프트 계약을 바탕으로 iiLocalDiffusion이 구성한 것으로, Krea가 배포한 원본 설정 파일이 아니다. 각 디렉터리는 Raw와 Turbo의 샘플링 변형을 명시하며 가중치는 포함하지 않는다. 선택한 트랜스포머의 모든 텐서 형태는 로드 시 검증한다. Turbo 공개 FP8 헤더와 설정 구조를 대조했지만 실제 Raw 파일이나 사전 학습 이미지 생성은 이 설정으로 검증하지 않았다.

Krea 2의 검증된 흐름 스케줄은 지수 시프트와 결정적 Euler 갱신을 사용한다. 패키지 스케줄러에 시그마 역전, 종료값 재조정, Karras·지수·베타 시그마 재구성 또는 확률적 샘플링이 활성화되어 있으면 생성 전에 해당 옵션을 명시하여 거부한다. 이 설정들은 스케줄이나 갱신식을 바꾸므로 같은 모델과 시드여도 원본 계약을 재현하지 않는다. `--krea2-mu`와 명시적 시그마 목록은 기존대로 사용할 수 있다. 기본 Raw·Turbo 스케줄은 공개 [깊이 제어 구현](https://huggingface.co/spaces/Patil/krea-2-depth-controlnet/blob/bf7716320a5b9864f06658dcbc494e098d8b4fc1/pipeline.py)의 공식과 수치 테스트로 비교한다. 이 비교는 사전 학습 모델 이미지의 일치를 입증하지 않는다.

```sh
iild-generate --backend diffusers --model /absolute/path/krea2_turbo_bf16.safetensors \
  --model-config /absolute/path/to/resources/model-configs/krea2-turbo --dtype bfloat16 \
  --controlnet /absolute/path/depth-control-lora.safetensors \
  --control-image /absolute/path/photo.png --control-preprocess depth \
  --control-depth-model /absolute/path/depth-anything --seed 42 \
  --prompt 'a sunlit studio' --output-dir resources/outputs/reference/krea-single
```

같은 구성과 제어 리소스로 압축 가중치를 스트리밍하려면 위 명령의 모델을 `krea2_turbo_fp8_scaled.safetensors`로 바꾸고 `--krea2-fp8-storage cpu-stream`을 추가한다.

공개 [Comfy-Org Krea 2 Turbo BF16 헤더](https://huggingface.co/Comfy-Org/Krea-2/blob/eb1eddd3983a54678545a9b2c178c5853b30f7be/diffusion_models/krea2_turbo_bf16.safetensors)의 430개 텐서 이름과 크기는 실제 Diffusers 모델을 메타 장치에 구성해 대조했다. 작은 무작위 원본 형식 가중치와 기존 Qwen VAE를 사용하는 오프라인 테스트는 범용 러너의 로딩, 네이티브 Canny와 소형 Depth Anything 전처리, 제어 결과 변화, 동일 시드 반복 및 변환 전 소형 모델과의 잠재 출력 일치를 확인한다. 추가로 공개 Turbo FP8 헤더의 256개 FP8 계층·배율과 430개 모델 텐서도 실제 모델 구조에 대조했다. 부분 범위 요청으로 읽은 공개 파일의 배율 값 256개는 모두 유한한 양수였다. 소형 합성 FP8 가중치는 스칼라 복원, 기본 깊이 및 색상·깊이 제어, 일반 스타일 LoRA와 AI Toolkit 제어 잔차를 밀집 복원본에 대조한다. CPU와 MPS에서 소형 모델을 시험했지만 공개 크기 모델의 실제 메모리 사용량과 성능은 측정하지 않았다. 이후 실제 13GB FP8 베이스 트랜스포머와 깊이·포즈용 보조 구성 요소를 각각 로드했으며, 실제 어댑터 후크의 등록과 복구도 합성 잠재값으로 확인했다. 26GB BF16 베이스 전체 로딩, 사전 학습 모델의 전방 연산과 이미지 생성은 검증하지 않았다. 세부 범위는 [리소스 실행 조합](controlnet-resource-recipes.md)을 참고한다.

사전 계산된 Krea 2 임베딩, 마스크와 잠재 입력은 자료형을 유지한 채 선택된 실행 장치로 이동한다. 이는 파일에서 읽은 CPU 텐서를 MPS 또는 CUDA 파이프라인에 전달할 때도 적용된다.

상위 `reference/generate.py` 는 `--model-config` 또는 `--model-config-preset`이 있는 경우, 기존 다운로드된 체크포인트 `--model` 가 이름으로 지정된 경우를 포함하여 이 일반 러너를 자동으로 선택합니다. `--controlnet-format`, `--control-preprocess`, 깊이·색상·마스크 전처리 옵션 및 Krea 2·MOSS 영역/어댑터처럼 범용 러너에만 있는 제어 옵션도 자동 선택에 반영됩니다. 단일 베이스 체크포인트에는 호환되는 로컬 설정이나 위의 내장 설정 선택이 필요합니다. 여러 백엔드가 공유하는 `--controlnet`만으로는 기존 라우팅을 바꾸지 않습니다. 명시적인 `--backend` 와 `--preset` 선택은 우선순위를 유지합니다. `--pipeline-class` 와 `--pipeline-inputs` 도 자동 로컬 파일 경로 전에 일반 러너를 선택하므로 이미지 투 이미지, 인페인팅 및 리파이너 입력이 요청된 작업 파이프라인에 도달합니다.

```sh
reference/diffusers/.venv/bin/python reference/diffusers/generate_any.py \
  --model /Volumes/Storage/Workspace/Models/model.safetensors \
  --model-config /Volumes/Storage/Workspace/Models/model-config \
  --pipeline-class StableDiffusion3Pipeline \
  --prompt 'a small architectural model in daylight' \
  --device mps --dtype float32 \
  --output-dir resources/outputs/reference/single-file-run
```

단일 베이스 파일이 PyTorch ZIP 또는 레거시 pickle 체크포인트이면 설치된 Torch 2.10.0 이상의 weights-only 로더와 safetensors를 사용해 생성 시 `--cache-dir/checkpoint-conversions/`에 변환한다. GGUF 가중치는 로컬 파이프라인 설정이 선언한 유일한 내장 변환기 또는 UNet 구성 요소에 Diffusers의 GGUF 양자화 로더로 연결한다. GGUF 로더에는 `gguf`와 해당 런타임이 지원하는 양자화가 필요하며, 구성 요소가 모호하거나 호환되지 않으면 실패한다. 로컬 구성과 호환되는 아키텍처 및 나머지 구성 요소가 필요하며 원본은 수정하지 않는다. 원래 경로는 재실행 설정에, 원본 및 변환본 식별자는 생성 보고서에 남기고 출력 게시 전후로 두 파일을 다시 확인한다. `--print-config`는 대용량 변환 대신 `--inspect-model`을 안내한다. 변환에는 모델 크기에 비례한 CPU 메모리와 디스크가 필요하고 지원하지 않는 객체나 양자화는 거부될 수 있다. 사용자 정의 Python 모델 코드는 실행하지 않는다. 예를 들어 `iild-generate --backend diffusers --model /models/flux.gguf --model-config /models/flux-pipeline --prompt 'a red cube'`는 로컬 구성에 맞는 GGUF 변환기를 사용한다. GGUF 파일을 선언된 `transformer` 또는 `unet` 디렉터리에서 직접 선택하면 상위의 유효한 파이프라인 설정도 자동으로 찾는다.

<a id="kolors-and-task-specific-checkpoints"></a>

## Kolors 및 작업별 체크포인트

Kolors 는 Kolors UNet , VAE , ChatGLM 텍스트 인코더, 토크나이저 및 스케줄러를 갖춘 완전한 로컬 Diffusers 디렉터리로 지원됩니다. 공식 배포판은 내장 `KolorsPipeline` 를 식별하며, 내장 `KolorsImg2ImgPipeline` 는 이미지 투 이미지 생성을 위해 해당 컴포넌트를 재사용할 수 있습니다.

```sh
reference/diffusers/.venv/bin/python reference/generate.py \
  --model /Volumes/Storage/Workspace/Models/Kolors-diffusers \
  --base-model Kolors --prompt 'a glass observatory above a cloud sea' \
  --width 1024 --height 1024 --steps 50 --guidance-scale 5 \
  --device cuda --dtype float16 --offload model --local-files-only \
  --output-dir resources/outputs/reference/kolors-image
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
  --output-dir resources/outputs/reference/inpaint-image
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
  --device cpu --output-dir resources/outputs/reference/flux-latents
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
# Embedded MOSS VAE

For MOSS Image, `--vae` accepts a Diffusers directory, Safetensors or PyTorch
file, or a validated shard index. PyTorch weights are converted only when the
VAE loads, using the restricted weights-only checkpoint conversion path.
The loader checks the VAE configuration, every tensor and the packed latent
contract. Legacy FLUX.2 names and 1×1 attention shapes are mapped without
changing tensor bytes; missing, duplicate and mismatched tensors are rejected.
The extracted single-file Safetensors MOSS path still requires the verified
FLUX.2 resource catalog and its reference VAE weights. Other explicit files use
their compatible local configuration or the pinned family fallback config.
A valid package header does not prove successful pretrained generation.

단일 파일 형식은 확장자보다 알려진 내용 서명을 우선해 식별한다. 범용 Safetensors와 GGUF 로더에는 원본 신원을 검사한 임시 표준 확장자 경로를 전달하며, 심볼릭 링크를 만들 수 없는 환경에서는 원본과 해시를 비교한 임시 복사본을 사용하고 완료 후 제거한다. PyTorch 컨테이너는 weights-only 검사를 거쳐 변환된 Safetensors만 전달하며, ONNX는 이 경로에 전달하지 않는다. `--num-images`는 단일 프롬프트의 이미지 수를 지정하고 출력 개수를 게시 전에 검사한다.

MOSS Image 범용 경로는 `--hires-fix`의 최종 크기를 16픽셀 배수로 검사하고 절반 크기의 기본 이미지를 생성한다. 이미지를 지정한 보간법으로 최종 크기에 맞춘 후 FLUX.2 VAE로 다시 인코딩하고, 정규화된 잠재값에 강도에 따른 노이즈를 추가하여 새 MOSS 스케줄러로 디노이징한다. 기본 강도는 0.35이며 `--hires-strength`와 `--hires-upscaler`로 지정할 수 있다. 보고서는 두 단계의 실제 디노이징 횟수를 기록하고 진행 이벤트는 두 단계를 합쳐 표시한다. 디노이징이 완료되지 않거나 결과 크기·개수가 일치하지 않으면 결과를 게시하지 않는다. 이 경로는 단일 프롬프트, PIL 이미지 출력, `--offload none`에 한정되고 ControlNet·영역 프롬프트·rapid 어댑터 조합은 거부한다. 다른 범용 파이프라인의 HiRes는 지원되지 않으며 조용히 단일 패스로 바꾸지 않는다. `--no-hires-fix`는 명시적인 단일 패스를 선택한다. 작은 무작위 가중치 테스트는 실제 상용 MOSS 모델의 생성 품질이나 성공을 증명하지 않는다.
