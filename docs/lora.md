<a id="lora-generation-contract"></a>

# LoRA 생성 계약

하나 이상의  LoRA 변경을 가중치에 융합하여 전체 체크포인트를 저장하려면  [모델 병합](model-merging.md)을 사용하세요.  `iild-merge --additional-model` 는 가중 합이나 직접적인 가중치 뺄셈을 위해 전체 체크포인트와 함께  LoRAs 를 받습니다. 다음 계약은 생성 중에 어댑터를 적용하는 방법을 설명합니다.

<a id="scope"></a>

## 범위

프리셋 및 독립 실행기는 SD1, SDXL와 FLUX.1 LoRAs를 지원한다. 일반 `--backend diffusers` 실행기는 Diffusers의 `load_lora_weights`, `set_adapters` 및 `get_list_adapters` API를 제공하는 모든 설치된 파이프라인과 동일한 로더, 강도 제어 및 구성 요소별 활성화 검사를 공유한다. 여기에는 SD2/SD3, 호환되는 FLUX2, Qwen Image와 다른 이미지 파이프라인이 포함된다. 지원하지 않는 로더는 기본 가중치를 할당하기 전에 실패한다. 사용자 정의 원격 파이프라인 코드나 새 추론 의존성은 도입하지 않는다.

`--lora`를 생략하면 공유 항목에서 일치하는 제품군 항목이 선택됩니다.
[생성 기본값은](generation-defaults.md)를 나타냅니다. 배포된 매니페스트는 대체 경로 목록이 비어 있으므로 `--lora` 를 생략하면 어댑터가 추가되지 않습니다. 유지된 `addDetailAesthetic_v20_32` 자산은 여전히 명시적으로 선택될 수 있습니다. 다른 계열은 자체 호환 어댑터가 필요하며, 계열 레이블, 키 또는 텐서 차원을 변경하여 SDXL LoRA 가 SD2, SD3 또는 FLUX LoRA 가 되지 않습니다. 명시적인 어댑터는 대체 경로 를 대체합니다. `--no-default-modifiers` 는 자동 가중치 없이 명시적인 기준선을 제공합니다.

네이티브 C++ 생성기는 또한 지원되는 모든 모델 계열에 대해 명시적인 LoRAs 를 엔진에 전달하며, 엔진이 로드된 모델을 식별한 후 계열 기본값을 해결합니다. 메타데이터 전용 C++ 매니페스트 검사기는 LoRAs 를 실행하지 않습니다. 기본 모델은 먼저 파이프라인 계약을 통과해야 하며, 어댑터는 장치 배치 또는 순차적 CPU 오프로드 후크가 설치되기 전에 로드되고 활성화됩니다. `--cpu-text-encoding` 가 활성화되면, 검증된 LoRA 활성화 후 CPU 프롬프트 인코딩이 실행되며, 해당 후크들 전에 실행됩니다. 따라서 텍스트 인코더 어댑터는 CPU 에서도 활성화되며, 결과적인 임베딩만 GPU 로 이동하여 생성됩니다. 모델 및 순차적 RAM 오프로드는 어댑터 선택과 독립적으로 유지됩니다. CPU 단계는 이후 원래 인코더 저장 정밀도를 복원하며, 인코딩이 실패하더라도 혼합 정밀도 어댑터 텐서/버퍼를 포함합니다. 그것은 선택된 어댑터 스케일을 융합하거나 비활성화하거나 변경하지 않습니다. 기본 가중치와 VAE 도 `--model` 와 `--vae` 를 통해 명시적으로 선택할 수 있으며, [모델 입력 계약](model-inputs.md)을 참조하십시오. 어댑터는 단순히 파일 확장자를 공유하는 것뿐만 아니라 결과 파이프라인과 일치해야 합니다. LoRA 도 선택된 [ControlNet](controlnet.md)와 함께 구성할 수 있습니다. ControlNet 컴포넌트는 LoRA 활성화 전에 연결되며, `--lora` 는 여전히 가족의 기존 디노이저/텍스트 인코더 어댑터 인터페이스를 타겟으로 유지하고 ControlNet 가중치 어댑터를 선택하거나 훈련하지 않습니다.

<a id="local-adapter"></a>

## 로컬 어댑터

정확한 safetensors 파일을 전달합니다.

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --model /absolute/path/image-diffusers \
  --preset flux1-schnell \
  --lora /absolute/path/to/style.safetensors \
  --lora-scale 0.75 \
  --output build/reference/flux-style.png
```

디렉토리는 정확한 파일 이름이 명시적으로 지정된 경우에만 허용됩니다.

```bash
reference/diffusers/.venv/bin/python \
  reference/diffusers/generate.py --model /absolute/path/image-diffusers \
  --preset sdxl-base \
  --lora /absolute/path/to/adapter-directory \
  --lora-weight-name style.safetensors \
  --lora-scale 0.8
```

SD2, SD3 및 기타 내장 이미지 파이프라인은 전체 로컬 Diffusers 디렉터리(또는 명시적 로컬 단일 파일 구성)를 사용합니다.

로컬 `model_index.json`의 명시적 `[null, null]` 구성 요소는 T5 없이 저장된 SD3 모델을 포함하여 일치하는 생성자 매개 변수에 `None`로 전달됩니다. 생략된 구성 요소는 생성되거나 가져오지 않습니다.

```bash
reference/diffusers/.venv/bin/python reference/generate.py \
  --backend diffusers --model /absolute/path/sd3-diffusers \
  --lora /absolute/path/sd3-style.safetensors --lora-scale 0.75 \
  --prompt 'a red cube' --output-dir build/reference/sd3-style
```

동일한 `--lora`와 `--lora-scale` 입력은 `--backend deforum`에서도 작동한다. 여기에는 SD, SDXL 및 FLUX.1의 텍스트에서 이미지 생성 이후 이미지에서 이미지 생성 프레임이 포함된다. 생성되는 모든 프레임은 프롬프트 인코딩 전, 장치 배치 후와 추론 후에 어댑터를 검증한다. `from_pipe`로 변환할 때 이를 유지해야 한다. 디노이징 강도가 0인 프레임은 이전 이미지를 워핑하고 `frames[].lora.applied: false`를 기록할 뿐이며 새로운 LoRA 추론을 수행했다고 주장하지 않는다.

선택된 로컬 파일은 존재해야 하며 비어있지 않고 `.safetensors` 또는 `.safetensor` 로 끝나야 합니다. 단수형 철자는 캐시 하에 있는 임시 표준 `.safetensors` 심링크를 통해 노출되므로 Diffusers 는 해당 철자를 위해 피클 로더를 선택하지 않습니다. 원래 경로와 해결된 타겟 SHA-256 및 바이트 크기가 기록되며, 로딩 전후로 동일성이 확인됩니다. 별명은 원래 가중치를 다시 작성하거나 복제하지 않으며 로딩 후 제거됩니다.

`--output`를 생략하면 `-lora`가 `-custom` 모델 및 `-vae` 접미사 뒤에 기본 출력 줄기에 추가됩니다. 따라서 결합된 모델/VAE/LoRA 입력은 아무런 알림 없이가 정식 베이스 픽스처의 파일 이름을 사용할 수 없습니다.

<a id="local-only-adapter-inputs"></a>

## 로컬 전용 어댑터 입력

어댑터는 기존 로컬 파일이나 디렉터리로 제공되어야 합니다. 디렉토리에는 `--lora-weight-name`가 필요합니다. 직접 파일은 그렇지 않습니다. 허브 ID 및 null이 아닌 `--lora-revision` 값은 거부됩니다. 모델 및 어댑터 로딩은 항상 `local_files_only=True`를 사용합니다. 생성 중에는 어댑터가 다운로드되지 않습니다.

<a id="runtime-semantics"></a>

## 런타임 의미론

고정된 내부 어댑터 이름은 `iild_lora` 입니다. Diffusers 는 `use_safetensors=True` 와 `low_cpu_mem_usage=True` 로 이를 로드한 후 `set_adapters("iild_lora", adapter_weights=scale)` 가 요청된 유한 스케일을 활성화합니다. 어댑터는 기본 가중치에 융합되지 않습니다. 음수, 0, 및 하나보다 큰 유한 스케일은 Diffusers 가 이를 지원하고 일부 어댑터가 0 에서 하나까지의 값에 의존하기 때문에 여전히 사용 가능합니다.

로더는 추론 시작 전에 어댑터가 적어도 하나의 모델 구성요소에 등록되어 있고 활성 목록에도 있는지 검증한다. SD 어댑터는 UNet과 CLIP 인코더를 대상으로 할 수 있고, SDXL 어댑터는 두 번째 CLIP 인코더도 대상으로 할 수 있으며, FLUX 어댑터는 transformer와 첫 번째 CLIP 인코더를 대상으로 할 수 있다. FLUX T5 인코더는 Diffusers 0.40.0에서 LoRA를 적재할 수 있는 구성요소가 아니다.

선택된 기본 아키텍처에 대해 어댑터가 훈련되었어야 합니다. 성공적인 다운로드가 형식 호환성, 의도된 트리거 단어, 출력 품질, 안전성, 소유권, 또는 상업적 권리를 설정하지 않습니다. 트리거 단어는 추론되거나 프롬프트에 삽입되지 않습니다. Diffusers 네이티브 safetensors LoRA 는 런타임 로 검증됩니다. 기타 safetensors 레이아웃은 Diffusers 0.40 의 형식 변환에 위임되며, Kohya, DoRA, LyCORIS 변형, Control LoRA, 퓨전, 그리고 동시 여러 어댑터는 이 인터페이스에 의해 독립적으로 검증되거나 명시적으로 거부되지 않습니다.

<a id="provenance"></a>

## 출처

프셋/Deforum JSON 는 `adapters` 배열을 포함하며, 일반 `generation.json` 는 `adapters.lora` (기본 전용 실행의 경우 null) 를 포함합니다. LoRA 실행은 적용 가능한 경우 소스, 정확한 파일, 요청된 리비전, 로컬 파일 해시 및 크기, 스케일, safetensors 형식, 고정 어댑터 이름, 실제 등록된 컴포넌트, 활성 어댑터 목록, 및 `fused: false` 를 기록합니다. 일반 리포트도 PEFT 의 설치 버전과 기본 선택 상태를 기록합니다. 캐시된 이미지 파이프라인은 구성 키에 어댑터 식별자와 강도를 포함하며: 어느 하나를 변경하면 파이프라인이 다시 로드되고, 변경되지 않은 요청은 등록된 어댑터를 재사용합니다. 파일은 추론 후 다시 확인됩니다.

`UniversalLoraTests`, `ModelPreparationCacheTests`, `DeforumRuntimeTests` 및 `NativeResultTests` 은 라우팅, 기본 선택, 우선순위, 캐시 변경 및 어댑터 손실을 포함합니다. `tests/UniversalLoraDiffusersSmoke.py` 는 작은 로컬로 초기화된 SD1, SD2, SDXL, SD3 및 FLUX 가중치를 사용하여 옵트인된 실제 Diffusers / PEFT 테스트입니다. 어댑터 0 강도, 기본 강도 및 반복 생성을 비교한 후 SD1/SDXL/FLUX Deforum 프레임 효과 및 비디오 디코딩을 확인합니다. 모델 다운로드 또는 훈련된 모델의 시각적 품질 설정을 수행하지 않습니다.

구현에서는 기존 Diffusers 0.40 / PEFT 0.20 스택과 해당 스택을 사용합니다.
[공식 LoRA 로더 인터페이스](https://huggingface.co/docs/diffusers/en/api/loaders/lora).

어댑터 라이센스는 기본 모델 및 Diffusers/PEFT 라이브러리 라이센스와 별개입니다. 운영자는 제품을 사용하기 전에 선택한 어댑터 저장소 또는 로컬 아티팩트 용어를 검토해야 합니다.
