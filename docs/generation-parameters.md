<a id="generation-values-and-neutral-defaults"></a>

# 생성 값과 중립적인 기본값

하나의 모델 위치를 선택하십시오: 로컬 `--model-path`(레거시 `--model`), 직접 `--model-api` 또는 `--model-provider`가 있는 원격 `--model-cloud`. 참조
원격 이미지/비디오 하위 집합 및 자격 증명 인수에 대한 [모델 소스](model-sources.md). 아래의 텐서, 스케줄러 및 어댑터 컨트롤은 로컬 실행을 설명합니다.

`--base-model` 는 `reference/generate.py` 를 통해 Civitai 가족을 선택하거나 프리셋 생성기를 사용합니다. 명시적인 `--preset` 는 `--base-model NoobAI --preset noobai-v-pred` 와 같은 호환 가능한 변형을 선택할 수 있으며, 충돌하는 가족은 실패합니다. `--prediction-type auto|epsilon|v_prediction|sample` 는 체크포인트 학습 매개변수화를 노출하며, `auto` 는 모델 구성 또는 이름 지정된 변형의 기본값을 유지합니다. 명시적인 `--scheduler-config` 값과의 충돌은 실패합니다. FLUX dev/ Krea 가이드는 schnell 의 0-가이드 요구사항과 구별됩니다. 참조
[모델 제품군](model-families.md) 및 [카탈로그](civitai-models.md).

독립적인 Python 생성기는 지원하는 SD 1.5, SDXL Base 및 FLUX.1-schnell의 텍스트에서 이미지 생성 값과 선택적 Hires Fix 값을 CLI 인수, 데이터 전용 JSON 파일과 동일한 Python 요청 해석기를 통해 제공한다. 추가 의존성은 도입하지 않는다. 인수 파싱과 JSON은 표준 라이브러리를 사용하며 로딩, 스케줄링, 인코딩과 텐서 연산은 기존 Diffusers/PyTorch API를 유지한다. 이는 네이티브 C++ 라이브러리에 완전한 확산 파이프라인을 추가하지 않는다.

<a id="optional-defaults-and-required-local-model-inputs"></a>

## 선택적 기본값 및 필수 로컬 모델 입력

명시적인 모델 소스가 필요합니다. 다른 생략된 값과 최상위 JSON `null` 는 문서화된 기본값을 사용합니다. 명시적인 `0`, `false` 및 빈 텍스트는 누락된 입력으로 오인되지 않고 유지됩니다. 알 수 없는 키, 잘못된 JSON 타입, 중복 키, 비유한 숫자 및 호환되지 않는 조합은 필요한 정보가 제공되는 모델 로딩 전에 안전하게 거부됩니다. 로딩된 모델에 의존하는 아키텍처, 스케줄러 및 텐서 형식 확인은 런타임 확인으로 유지됩니다.

우선순위는 **명시적인 CLI > JSON 설정 >**기본값입니다. 부울 제어는 `--name` 와 `--no-name` 를 모두 지원하며, `--no-cpu-text-encoding` 와 `--no-overwrite` 를 포함합니다. 모델 로딩은 항상 로컬이며, `--no-local-files-only` 는 거부됩니다. 설정 키는 CLI 목적지에서 snake_case 를 사용하며, 예를 들어 `num_images`, `guidance_scale`, `vae_tiling` 입니다. 파일-경로는 JSON 파일의 디렉토리에 대해 해결되며, CLI 경로는 작업 디렉토리를 사용합니다. 인용된 `~`  경로는 출력/캐시 생성 전에 확장되므로, 내보낸 경로와 실제 파일 시스템 타겟이 일치합니다. 설정 파일은 1   MiB  로 제한되며, 재귀적으로 다른 설정 파일을 로드할 수 없습니다.

Torch 가져오기, GPU 확인, 모델 다운로드, 캐시 생성 또는 생성 없이 해결된 전체 구성을 검사하거나 내보냅니다.

```bash
python reference/diffusers/generate.py --print-config --model /absolute/path/image-diffusers
python reference/diffusers/generate.py --preset sdxl-base --print-config --model /absolute/path/image-diffusers
python reference/diffusers/generate.py --preset flux1-schnell --print-config --model /absolute/path/image-diffusers
```

인쇄된 평면 JSON 는 저장되어 `--config` 에 공급될 수 있으며, 동적 `auto`  하드웨어 정책은 실제 실행까지 `auto`  유지됩니다. 사이드카는 또한 실제 해결된 런타임  값을 기록합니다. 명시적인 로컬 텐서 파일은 요청 해결 중에도 여전히 확인되고 해시됩니다.

예를 들어 부분 구성이면 충분합니다.

```json
{
  "preset": "sdxl-base",
  "prompt": "a ceramic cup on a plain table",
  "negative_prompt": "",
  "seed": 0,
  "num_images": 2,
  "width": null,
  "height": null,
  "guidance_rescale": 0.0,
  "vae_tiling": false
}
```

```bash
python reference/diffusers/generate.py --config build/generation.json --model /absolute/path/image-diffusers \
  --steps 24 --device rocm --cpu-text-encoding --offload model
```

최소한의 재사용 가능한 파일이 제공됩니다.
[`generation.example.json`](../reference/diffusers/generation.example.json). Python 진입점은 정확히 동일한 스키마를 확인합니다.

```python
# reference/diffusers가 sys.path에 있는 경우:
from generate import resolve_request, configuration_values

preset, request = resolve_request({
    "model": "/absolute/path/image-diffusers",
    "preset": "sdxl-base",
    "seed": 0,
    "width": None,
    "vae_tiling": False,
})
resolved_values = configuration_values(request)
# 로컬 모델 필드는 필수이며 생략한 다른 필드는 프리셋 기본값을 사용한다.
```

<a id="model-specific-defaults"></a>

## 모델별 기본값

|값| SD 1.5 |SDXL 기본|FLUX.1-schnell|
|---|---|---|---|
| `--preset` |`sd15`(전역 기본값)| `sdxl-base` | `flux1-schnell` |
| `--width`, `--height` | 512 × 512 | 1024 × 1024 | 1024 × 1024 |
| `--steps` | 20 | 20 | 4 |
| `--guidance-scale` | 7.5 | 5.0 |0.0 ; Schnell에는 개발 가이드 임베딩이 없습니다|
|자동 GPU dtype|float16|float16|bfloat16|
|자동 CPU/text dtype|float32|float32|bfloat16|
| `--max-sequence-length` |해당 없음|해당 없음|256; 명시적 범위 1–512|
|자동 중량 변형|fp16 GPU에서 사용 가능한 경우|GPU에서 사용 가능한 경우 fp16|없음|
|자동 GPU 오프로드|상주 또는 CPU 인코딩을 사용한 모델 오프로드|동일|순차|
|자동 GPU VAE 슬라이싱/타일링|꺼짐 / 꺼짐|꺼짐 / 켜짐|켜짐 / 켜짐|

SDXL 는 Diffusers   VAE 타일링을 MPS / CUDA 에서 가능하게 합니다. 포함된 SDXL   VAE 는 한 번의 패스로 1024×1024 를 디코딩하며, 더 큰 공간 범위는 오버랩 타일을 포함하여 사용합니다. 이것은 요청된 출력 크기 조정 없이 디코더 활성화 메모리를 제한하고, 주석 처리된 모델을 오프로드하지 않습니다. 타일 블렌딩은 전체 디코딩과 비교하여 작은 픽셀 차이를 생성할 수 있습니다. `--no-vae-tiling` 는 명시적인 전체 디코딩 옵션을 유지하며, CPU 기본값은 변경되지 않습니다.

이것들은 기존 호환성 참조 기본값이며, 보편적 최적 설정이 아닙니다. 단순한 대체 경로 프롬프트는 `a red cube on a white table` 입니다. 스타일/페르소나/아티스트 수정자가 없는 경우입니다. 음성 텍스트는 비어 있으며, LoRA 은 부재하고, ControlNet 과 Hires Fix 이 비활성화되어 하나의 이미지가 생성되며, 추가적인 리스케일링/조기 종료는 비활성화됩니다. 생략되거나 JSON -null 시드는 각 요청마다 새로운 무작위 32-비트 기본 시드를 선택합니다. 해결된 구성과 출력 메타데이터는 해당 시드를 유지하며, 요청을 재현하려면 명시적으로 전달합니다. 명시적 시드, 0를 포함하여 보존됩니다. 따라서 저장된 해결된 구성은 기록된 시드를 재사용하며, 새 무작위성을 위해 제거하거나 null 로 설정합니다.

<a id="prompt-sampling-and-batch-values"></a>

## 프롬프트, 샘플링 및 배치 값

|인수|기본/동작|
|---|---|
| `--prompt`, `--negative-prompt` |단순 큐브 프롬프트/빈 문자열|
| `--prompt-2`, `--negative-prompt-2` |SDXL/FLUX는 각각의 기본 텍스트를 상속합니다. 명시적인 값이 전달됩니다.|
| `--seed` |요청당 무작위; 명시적 값은 PyTorch의 범위를 지원합니다. [-2^63, 2^64−1]|
| `--num-images` / `--num-images-per-prompt` |1; 이 프롬프트에 대한 긍정적 이미지, 모두 저장됨|
| `--seed-stride` |1 ; 이미지 i 는 시드 + i × 스트라이드로 시드를 받으며, 0 는 의도적으로 반복합니다.|
| `--generator-device` |`cpu` ; `execution` 는 CUDA /ROCm 을 필요로 합니다.|
| `--guidance-rescale` |0 ; SD / SDXL 범위 [0,1]; 0 이 아닌 값은 ControlNet 와 함께 지원되지 않습니다.|
| `--eta` |0 ; 0 이 아닌 값은 eta 매개변수를 가진 스케줄러, 예를 들어 DDIM 를 필요로 합니다.|
| `--clip-skip` |없음; 로딩된 인코더 깊이에 대해 확인된 네이티브 CLIP 선택|
| `--true-cfg-scale` |1 ; FLUX 값 > 1 는 명시적 2-패스 CFG 와 음수 프롬프트를 활성화합니다.|
| `--scheduler` |`auto`; 그렇지 않으면 설치된 호환 가능한 Diffusers 스케줄러 클래스 이름|
| `--scheduler-config` |`{}` ; 엄격한 JSON 생성자 오버라이드|
| `--timesteps` |없음; 음수가 아닌 내림차순 SD/SDXL 정수 일정|
| `--sigmas` |없음; 하향식 유한 노이즈 스케줄, 시간 단계와 상호 배타적입니다.|
| `--denoising-end` |없음; SDXL 조기 정지 구간은 0와 1사이에 엄격히 포함됩니다.|
| `--cross-attention-kwargs` |`{}`; SD/SDXL 휴대용 통화 시간 LoRA `scale`|
| `--joint-attention-kwargs` |`{}`; FLUX 휴대용 통화 시간 LoRA `scale`|

스케줄러는 로드된 기본 계약이 통과된 후에만 교체됩니다. 교체는 해당 런타임 `compatibles` 클래스와 `from_config` 를 사용하며, 구성 키는 해당 생성자에 속해야 하며 임의의 Python 객체나 내부 메타데이터가 될 수 없습니다. 철자가 잘못된 키 또는 지원되지 않는 일정/ETA 는 아무런 알림 없이 무시되는 대신 거부됩니다. 예시:

```bash
python reference/diffusers/generate.py --scheduler DDIMScheduler --eta 0.2 --model /absolute/path/image-diffusers \
  --scheduler-config '{"timestep_spacing":"trailing"}' --steps 20
```

사용자 정의 일정이 단계 수 요청보다 우선합니다. 사이드카는 요청과 파이프라인 보고된 시간 단계 수, 그리고 스케줄러의 시간표 모두를 유지합니다. 조기 종료는 해당 시간표의 일부만 사용할 수 있습니다. 시그마 끝점의 의미는 스케줄러마다 다릅니다; 생성기는 명시적으로 제공된 일정을 제조하거나 재표본추출하지 않습니다.

주의 JSON 는 현재 이러한 파이프라인에서 지원되는 휴대용 유한 `scale` 필드만 허용하며, 선택된 LoRA 를 요구합니다. 이는 `--lora-scale` 와 구별되는 호출 시점 스케일이며, 둘 다 기록됩니다. CPU 텍스트 인코딩은 동일한 스케일을 전달합니다. 프로세서별 텐서, GLIGEN 와 IP -어댑터 설정은 이 JSON 를 허용함으로써 도입되지 않습니다.

FLUX의 `guidance_scale=0` 계약은 변경되지 않습니다. `true_cfg_scale`는 Dev 지침 임베딩이 아닌 별도의 선택적 두 번째 노이즈 제거 패스입니다. Schnell의 기본 256를 초과하는 시퀀스 길이는 새로 검증된 전체 모델 품질 설정이 아니라 명시적인 실험입니다.

설치된 Transformers 5.16.1 CLIP 는 최상위에서 최종 정규화를 노출하는 반면, Diffusers 0.40 의 SD 1.5 CLIP -skip 브랜치는 여전히 `text_model` 아래에서 조회합니다. 범위 지정된 비 모듈 뷰는 인코딩 중 동일한 정규화를 참조하며, 실패하더라도 제거됩니다. 중복 신경 모듈을 등록하지도 않고 가중치 키/수학을 변경하지도 않으며, 적용된 호환성 경로가 기록됩니다. 중첩된 CLIP 레이아웃과 SDXL 는 그대로입니다.

SDXL 에 대한 다음 크기 값은 `HEIGHT WIDTH` 순서를 사용하며 `--original-size`, `--target-size`, `--negative-original-size`, `--negative-target-size` 로 기본값을 설정합니다. `--crops-coords-top-left` 와 `--negative-crops-coords-top-left` 는 모두 `TOP LEFT` 를 사용하며 `0 0` 로 기본값을 설정합니다. 이 조건들은 모델을 조건화하며 저장된 PNG 를 자릅니다.

<a id="model-hardware-memory-and-output-values"></a>

## 모델, 하드웨어, 메모리 및 출력 값

기존 로컬 `--model`, `--model-config`, `--vae`, `--lora`, `--lora-weight-name`, `--lora-scale` 는 모든 3 입력 형식에서 사용 가능합니다. 로컬 모델은 필수이며 단일 파일 모델은 추가적으로 로컬 구성 소스가 필요합니다. VAE 는 선택된 소스에서 비롯되며 누락된 LoRA 는 비활성화됩니다. 선택된 LoRA 는 1.0스케일로 기본값을 설정합니다. 파일 레이아웃과 동일성을 확인하려면 [모델 입력](model-inputs.md) 를 참조하세요.

선택 가능한 `--controlnet` 와 `--control-image` 는 SD1/SDXL/FLUX 패밀리 프리셋에 호환되는 하나의 ControlNet 를 활성화합니다. 이미지는 해당 모델이 예상하는 조건을 이미 포함해야 하며 에지/깊이/포즈 감지기는 실행되지 않습니다. 선택될 경우 `--controlnet-scale` (별칭 `--controlnet-conditioning-scale`) 는 1.0로 기본값을 설정하며 `--control-guidance-start` / `--control-guidance-end` 는 0.0/1.0로, `--guess-mode` 는 false 로 기본값을 설정합니다. 이 옵션들은 `--controlnet` 를 필요로 하며 모두 누락되면 기능을 비활성화합니다. 영수가 아닌 `--guidance-rescale` 는 거부됩니다. SD / SDXL 는 추측 모드를 지원하며 FLUX 유니온 모델은 모델별 `--control-mode` 를 필요로 합니다. 가중치/구성 소스, 독립적인 변형 및 버전, 단일 파일, JSON 키, 그리고 출처는
[ControlNet 계약](controlnet.md).

|인수|기본/동작|
|---|---|
| `--device` |`auto`: 먼저 CUDA/ROCm과 호환되고 그 다음에는 Metal; 무음 없음 CPU 대체 경로|
| `--device-index` |0 에서 설명되며 경계 검사된 CUDA /ROCm 인덱스, CPU / Metal 는 0를 필요로 합니다.|
| `--dtype` |`auto` 또는 명시적인 부동소수점32/부동소수점16/bfloat16]|
| `--weight-variant` |`auto`, `none` 또는 명시적인 파일명 변형; 변환 데이터형과 독립적|
| `--low-cpu-mem-usage` |모델, VAE, LoRA 및 ControlNet 로더에 대해 참|
| `--cpu-text-encoding` |거짓|
| `--cpu-text-dtype` |`auto`; 명시적인 데이터형은 CPU 텍스트 인코딩이 필요함|
| `--cpu-threads` |설치된 런타임 기본값을 유지; 명시적인 개수는 양수여야 함|
| `--offload` |`auto`; 오버라이드는 없음/모델/순차적|
| `--cuda-tf32` |자동으로 자격 있는 NVIDIA 정책; 명시적인 TF32 스위치는 AMD 에서 거부됨|
| `--attention-slicing` |사전 설정/장치 정책; 호환되지 않는 FLUX 요청이 실패합니다.|
| `--attention-slice-size` |`auto`; 또한 최대 또는 양의 정수를 허용함|
| `--vae-slicing`, `--vae-tiling` |사전 설정/장치 정책, 명시적 활성화/비활성화|
| `--watermark` |SDXL 거짓; 명시적인 참은 선택적 설치된 의존성이 필요함|
| `--progress` |참; false는 생성 진행 상황을 숨깁니다.|
| `--cache-dir`, `--xet-cache-dir` |리포지토리 build/reference/huggingface 및 Huggingface-xet|
| `--local-files-only` |참; 비활성화가 거부되었습니다.|
| `--output` |build/reference 하의 사전설정 파일명; 사용자 정의/vae/lora/embedding/controlnet/hires 접미사가 구성됨|
| `--overwrite` |거짓|
| `--png-compress-level` |6, 범위 0–9; 무손실|
| `--png-optimize` |거짓; 무손실 스토리지 최적화|

CPU 텍스트 인코딩은 LoRA 활성화 후 GPU /오프로드 배치 전에 유지됨 SD/SDXL 파이프라인 호출 동안 하나의 프롬프트의 임베딩을 확장; FLUX 는 CPU 임베딩을 먼저 확장하고 호출 시점의 곱셈 인자가 하나임 이것은 누락된 이미지나 우연한 제곱 배치 크기를 방지함 VAE 슬라이싱/타일링 메타데이터는 실행된 정책의 증거가 아닌 활성화된 정책을 기록합니다. 명시적인 메모리/데이터형 변경은 여전히 RAM / VRAM 를 소진할 수 있으며, 기본값은 큰 모델이 맞다는 보장이 아닙니다.

명시적인 출력 경로가 없으면 충돌은 `-run-0002`, `-run-0003` 등 이전 증거를 보존합니다. 배치는 `-0001.png`, `-0002.png` 등을 받습니다. 각 PNG 는 고유한 JSON 사이드카, 이미지 시드 및 해시, 완전한 배치 소속, 해결된 `parameters`, 스케줄러 구성 및 텐서 입력 식별자를 갖습니다. 명시적인 출력 충돌은 오버라이드가 요청되지 않은 경우 의존성/모델 로딩 전에 실패합니다. 원자적 비-오버라이디트 게시는 같은 디렉토리의 하드 링크를 사용하며, 지원되지 않는 파일 시스템 또는 I/O 실패는 오버라이디트 권한이 아닌 명시적인 오류입니다. 다 파일 배치는 파일 시스템 트랜잭션이 아닙니다.

<a id="default-hires-fix"></a>

## 기본 고용 수정

`--hires-fix` 은 기본 생성을 활성화한 후 반복적인 RGB 업스케일링 및 img2img 정제와 SD1/SDXL/FLUX 패밀리 프리셋을 포함하여 그 ControlNet 변형을 수행합니다. `--hires-passes N` 는 기본 이미지 이후 N 개의 추가 정제 단계를 선택하며, 그 활성화된 기본값 1 은 원래 2단계 동작을 보존합니다. 명시적인 리사이즈 오버라이드가 없으면 너비/높이는 최종 목표이며 기본은 각 축의 절반을 사용하여 모델 그리드로 반올림됩니다. 반복된 패스는 해당 목표를 유지하며, `--no-hires-fix` 는 단일 단계를 선택합니다. 비활성화되면 종속 값은 비활성/무효이며 명시적인 종속 설정은 Hires Fix 를 요구합니다.

|인수|활성화 시 기본값/동작|
|---|---|
| `--hires-fix` |정지 이미지의 경우 True입니다. 절반 크기의 베이스에 이어 요청된 치수로 개선됩니다. 명시적 false는 단일 패스를 선택합니다.|
| `--hires-passes` |1 활성화되면; 기본 생성 후 리파인먼트 개수인 양의 정수이며 비활성화되면 null 입니다.|
| `--hires-scale` |기본값은 null 이며 1 보다 큰 명시적인 계수는 너비/높이를 기본 차원으로 하는 패스별 리사이즈 체인으로 전환합니다.|
| `--hires-width`, `--hires-height` |1차 정제 대상; 기본 종횡비에서 누락된 축을 추론한 다음 첫 번째 패스의 축별 요인을 반복합니다. 패밀리 차원 배수 적용|
| `--hires-upscaler` |`lanczos`; 가장 가까운/쌍선형/쌍입방/lanczos RGB 크기 조정|
| `--hires-denoising-strength` / `--hires-strength` | 0.35; `0 < value <= 1` |
| `--hires-steps` |`--steps` 를 상속하며 강도가 활성 부분을 선택하기 전까지 전체 일정 길이를 유지합니다; SD / SDXL 는 `int(steps * strength) >= 1` 를 필요로 하고, FLUX 는 자체 일정-꼬리 규칙을 사용합니다.|
| `--hires-seed` |`--seed` 상속; 모든 개선에서 동일한 시드를 반복하는 새로운 이미지당 생성기 및 `--seed-stride`|
| `--hires-guidance-scale` |기본 지침 상속, 패밀리 제한 유지|
| `--hires-true-cfg-scale` |기본 true CFG 를 상속하며 중립 값 1 보다 큰 값은 FLUX 를 필요로 합니다.|
| `--hires-scheduler` |`auto`, 첫 번째 단계의 실제 스케줄러 클래스/구성을 새로운 상태로 상속|
| `--hires-scheduler-config` |`{}`; 모든 개선 시마다 새로운 스케줄러 상태로 재사용되는 검증된 생성자 재정의|
| `--hires-save-base` |거짓이며 각 기본 PNG 와 그 JSON 사이카를 각각 저장합니다.|

명시적인 리사이즈 오버라이드가 있으면 각 패스는 이전 출력 이미지를 가져와 두 축을 유지하거나 확대하며 적어도 한 축을 확대합니다. `--hires-passes 2 --hires-scale 2` 가 있으면 512 × 512 기본이 1024 × 1024가 되고, 2048 × 2048가 됩니다. 명시적인 너비/높이 이름은 첫 번째 리파인먼트 타겟을 지정하며, 축별 계수는 이후 패스에 대해 반복되어 설정 차원의 배수로 반올림됩니다. 모델, VAE , LoRA , ControlNet , 프롬프트 및 임베딩은 모든 단계를 거쳐 구성됩니다. 각 리파인먼트는 SDXL 크기 조건을 위해 자체 해상도를 사용하며 새 일정을 시작하며, 기본 `--latents` , `--timesteps` , `--sigmas` , `--denoising-end` , `--guidance-rescale` 는 단계를 하나에만 적용됩니다. 최종 이미지와 선택적 기본 이미지만 저장됩니다. `hires_fix.stages` 는 정렬된 입력/업스케일/정제/출력 기록을 포함하며, `requested_passes` 와 `completed_passes` 는 정제 횟수를 기록합니다. 기존 `upscale` 와 `refinement` 메타데이터 필드는 마지막 단계를 설명합니다.

관리형 로컬 체크포인트 백엔드 또한 `--hires-fix`, `--hires-passes`, `--hires-scale`, `--hires-upscaler`, `--hires-steps` 및 `--hires-denoising-strength` / `--hires-strength` 을 수락합니다. 이는 `request.hires` 에 계획된 크기/설정을 기록하고 완전한 ComfyUI 워크플로우 및 최종 이미지를 확인합니다. 표에 있는 나머지 프리셋별 HiRes 옵션은 관리형 백엔드 옵션이 아닙니다. [Hires Fix](hires-fix.md) 를 참조하여 백엔드별 실행 증거, 예시, 출력 규칙 및 품질 제한을 확인하세요.

<a id="optional-initial-latents-and-precomputed-text"></a>

## 선택적 초기 잠재성 및 미리 계산된 텍스트

학습된 텍스트 반전 토큰은 아래 설명된 `--text-embedding`를 사용합니다. 이 섹션의 `--embeddings`는 완성된 프롬프트 텐서를 제공하고 텍스트 인코딩을 우회합니다. 학습된 토큰 파일과 결합할 수 없습니다.

`--latents FILE` 는 로컬 safetensors 파일을 수락합니다. `--latents-key` 는 `latents` 로 기본값입니다. 생략 시 새 노이즈가 구성된 생성자에서 샘플링됩니다. SD / SDXL 파일은 `[num_images, unet_channels, height/vae_factor, width/vae_factor]` 형식의 미확대 초기 노이즈를 포함합니다. FLUX 파일은 `[num_images, (height/(2*vae_factor))*(width/(2*vae_factor)), transformer_channels]` 형식의 이미 패킹된 노이즈를 포함합니다. 차원은 로드된 모델에서 가져오며 잘못된 형식은 리사이즈되지 않습니다.

`--embeddings FILE`는 `prompt_embeds` 및 해당하는 경우 `pooled_prompt_embeds`, `negative_prompt_embeds`, `negative_pooled_prompt_embeds`를 허용합니다. `--embedding-keys`의 기본값은 `{}`로, 이는 해당 표준 키를 의미하며 이를 다른 파일 키에 매핑할 수 있습니다.

```bash
python reference/diffusers/generate.py --embeddings /absolute/path/text.safetensors --model /absolute/path/image-diffusers \
  --embedding-keys '{"prompt_embeds":"positive","negative_prompt_embeds":"negative"}' \
  --latents /absolute/path/noise.safetensors --latents-key noise
```

프롬프트 텐서는 `[1 or num_images, tokens, embedding_dimension]` 형식을 가지며, 풀링된 텐서는 `[same_batch, projection_dimension]` 형식을 가집니다. 가이드는 일치하는 음의 텐서를 요구합니다. 사전 확장된 배치 는 다시 확장되지 않습니다. FLUX 단일 프롬프트 텐서는 명시적으로 반복됩니다. 외부 임베딩은 텍스트 인코딩을 대체하며 CPU 텍스트 인코딩 또는 CLIP 스킵과 결합될 수 없습니다. 텍스트 프롬프트 필드는 기록되지만 실행되지 않습니다. 임베딩 파일을 생략하면 기본값을 사용하는 일반 텍스트 인코더가 사용됩니다.

선택된 키만 구체화되며, 추론 전에 형식/데이터형/유한값 확인이 선행됩니다. 안전한 파일 식별자는 읽기 전과 후에 검증되며, 맵핑된 파일이 해제되기 전에 텐서가 복사됩니다. `.safetensors` 와 `.safetensor` 모두 허용됩니다. 잘못된 형식의 파일은 결코 제작된 0 임베딩이나 무작위 노이즈로 회귀하지 않습니다.

<a id="optional-learned-text-tokens"></a>

## 선택적 학습된 텍스트 토큰

`--text-embedding` 는 선택된 텍스트 인코더/토크나이저에 텍스트 인버전 벡터를 추가하고 정상 텍스트 프롬프트 인코딩을 유지합니다. SD 1.5 CLIP 에 모두 사용 가능하며, SDXL CLIP 인코더와 FLUX CLIP/T5 (그 중 ControlNet, CPU 텍스트 인코딩 및 Hires Fix 포함) 에 적용됩니다. 완전한 파일 레이아웃과 다중 벡터 계약은 [학습된 텍스트 임베딩](text-embeddings.md)에 있습니다.

|인수|기본/동작|
|---|---|
| `--text-embedding FILE [FILE ...]` / `--textual-inversion` |없음; 하나 이상의 로컬 `.safetensors` / `.safetensor` 학습 토큰 파일|
| `--text-embedding-token TOKEN [TOKEN ...]` |소스에서 추론합니다; 명시적 목록은 파일당 하나 이상의 비어있지하고 공백 없는 토큰을 포함합니다|
| `--text-embedding-encoder auto\|text_encoder\|text_encoder_2 [...]` |파일당 `auto` 개; 지원된 키/차원 또는 일치하는 인코더를 명시적으로 선택하여 추론합니다|

토큰과 인코더 제어는 파일이 필요하며, 제공된 목록은 파일 개수와 일치해야 합니다. 완료된 프롬프트 `--embeddings` 입력은 상호 배타적입니다. 로드된 벡터는 인코더 너비와 일치해야 하며, 유한한 부동소수점 값을 포함하고 토큰 충돌을 발생시키지 않아야 합니다. 토큰을 로드하는 것은 프롬프트에 삽입하지 않으며, 사용되지 않은 로드된 토큰은 유효합니다. 기본 출력 이름은 `-embedding` 를 추가하며, 메타데이터는 파일 해시, 토큰, 구성 요소, 토큰 ID 및 벡터 개수를 기록합니다.

<a id="deliberately-preserved-boundaries"></a>

## 의도적으로 경계를 보존함

선택적 샘플링 매개변수에 기본값이 적용됩니다. 이들은 하드웨어, 모델 파일 또는 라이선스를 제공하지 않습니다. 생성은 명시적인 로컬 모델 경로가 필요하며, LoRA 디렉토리도 정확한 파일 이름을 여전히 필요로 합니다. 누락된 로컬 파일은 다운로드나 대체 없이 실패합니다. [로컬 모델 생성](local-model-generation.md)을 참조하세요.

Safetensors 만 로드, 원격 코드 비활성화, 훈련된 아키텍처 검사, RGB   PNG  출력 및 추론 전용 실행은 계약으로 남아 있으며, 안전하지 않은 JSON  스위치가 아닙니다. 모델 구조 상수는 검증된 모델 구성에서 나옵니다. 콜백/코드 객체, 임의 주의 프로세서, IP -어댑터, Multi- ControlNet , 외부 이미지-이미지 입력 및 별도의 SDXL 리파이너 어셈블리는 추가되지 않습니다. Img2img 는 Hires Fix 내부에서 사용됩니다.

API 라우팅은 설치된 Diffusers 0.40.0 소스와 공식 [SD 파이프라인 문서](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)에 대해 확인되었습니다.
[SDXL 문서](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl),
[FLUX 문서](https://huggingface.co/docs/diffusers/api/pipelines/flux) 및
[스케줄러 가이드](https://huggingface.co/docs/diffusers/using-diffusers/schedulers). 단위 테스트는 시뮬레이션된 텐서/장치를 사용합니다. 실제 캐시된 모델 연기 실행은 이미지 품질이나 모든 하드웨어 증명이 아닌 별도의 실행 증거를 제공합니다.
