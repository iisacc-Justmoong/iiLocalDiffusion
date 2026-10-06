<a id="hires-fix-repeated-image-refinement"></a>

# Hires Fix: 반복적인 이미지 개선

기본 이미지 해상도는 최종 출력 크기이다. 네이티브 API, Python 프리셋·단일 체크포인트,
관리형 ComfyUI 이미지 경로는 1024×1024 요청을 512×512 1차 생성과 Hires 보정으로 완성한다.
Python에서는 SD/SDXL 8픽셀, FLUX 16픽셀 등 모델 배수에 맞게 절반 크기를 반올림한다.
최종 크기는 요청값을 유지하며, 작은 스텝 요청도 실제 보정이 최소 한 번 실행되도록
생략된 보정 스케줄을 보완한다. 기본 Lanczos·강도 0.35에는 새 의존성이 필요하지 않다.
`--hires-save-base`로 두 크기의 PNG와 실행 정보를 확인할 수 있다.

Python의 `--no-hires-fix`는 명시적인 단일 단계 실행이다. `--hires-scale` 또는
`--hires-width/height`를 직접 지정하면 기존 확대 체인을 선택하여 `--width/height`를
기준 이미지 크기로 해석한다. 이 크기 재정의가 없으면 `--hires-passes`를 늘려도
최종 목표 크기에서 보정을 반복한다(1–1000회). 설정 내보내기는 최종 요청 크기를
보존하므로 재실행 때 다시 절반으로 축소되지 않는다. 애니메이션·원격 생성·직접 지정한
범용 파이프라인 및 원시 ComfyUI 워크플로는 각 경로의 명시적 크기 계약을 따른다.

독립적인 Python 프리셋 생성기는 SD1/SDXL/FLUX 계열과 그들의 이름이 붙은 파생본을 지원하며, 각각 호환되는 ControlNet 이 있거나 없어도 됩니다. 모델/체크포인트 선택, VAE, LoRA, 학습된 텍스트 인버전 토큰, 배치 생성, 사전 계산된 임베딩, CPU 프롬프트 인코딩, 그리고 장치/오프로드 정책은 베이스 패스와 리파인먼트 패스를 가로지르며 구성됩니다.

<a id="explicit-resize-overrides"></a>

## 명시적 크기 조정 재정의

Hires Fix 는 먼저 `--width` / `--height` 에서 생성한 후, 결과 RGB 이미지를 리사이즈하고, 리사이즈된 크기로 이미지-이미지 확산을 수행합니다. `--hires-passes N` 는 베이스 이미지 이후 추가 리파인먼트 패스의 수를 선택합니다. 활성화되면 기본값은 1 로, 원래 베이스 플러스 하나 리파인먼트 동작을 유지합니다. 각 패스는 이전 패스의 디코딩된 이미지를 가져와 리사이즈하고, 선택된 VAE 로 인코딩한 후 리파인먼트 전에 노이즈를 추가합니다. 각 패스는 동일한 구성된 리사이즈 및 img2img 설정을 사용하며, 새로운 스케줄러와 생성기 상태를 사용합니다. 강도 0 는 거부됩니다.

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py --model /absolute/path/image-diffusers \
  --preset sd15 --width 512 --height 512 \
  --hires-fix --hires-passes 2 --hires-scale 2 --hires-upscaler lanczos \
  --hires-denoising-strength 0.35 --hires-steps 30 \
  --hires-save-base
```

이것은 512 × 512 베이스 생성, 1024 × 1024 첫 번째 리파인먼트, 2048 × 2048 두 번째 리파인먼트를 요청합니다. 스케일은 모든 패스에 적용되므로 추가적인 패스는 작업량과 최종 이미지 크기를 모두 증가시킵니다. 미세 조정 단계 설정은 img2img 강도가 활성 부분을 선택하기 전의 전체 일정 길이입니다. 예제의 일반적 1 차 일정으로, `int(30 * 0.35)` 은 10 개의 미세 조정 단계를 제공합니다; 이는 패스당 30 개의 활성 미세 조정 단계를 요청하지 않습니다. 런타임 스케줄러 메타데이터는 실행된 내용을 기록합니다.

<a id="parameters-and-neutral-defaults"></a>

## 매개변수 및 중립 기본값

아래 사전 설정 옵션은 CLI, JSON `--config` 및 Python `resolve_request()`를 통해 사용할 수 있으며 다른 옵션과 동일한 우선 순위 및 유효성 검사를 받습니다.
[생성 매개변수](generation-parameters.md). JSON 키는 snake_case를 사용합니다. 명시적으로 비활성화되면 종속 옵션은 비활성/null 상태로 유지되며 제공될 수 없습니다.

비활성화된 JSON 스타터의 경우 `hires_passes` 및 `hires_save_base`를 포함한 모든 종속 값에 대해 `"hires_fix": false` 및 `null`를 사용합니다. 명시적인 `"hires_save_base": false` 또는 `--no-hires-save-base`는 선택된 종속 설정이므로 여전히 Hires Fix가 필요합니다.

|인수|활성화 시 기본값/동작|
|---|---|
| `--hires-fix` / `--no-hires-fix` |정지 이미지에 대해 활성화됨; 절반 크기의 기본값으로 이어진 후 명시적으로 요청된 목표값; 명시적인 비활성화는 단일 패스를 선택합니다.|
| `--hires-passes` |1; 양의 정수 개수의 개선이 기본 이미지 이후에 전달됩니다. 비활성화된 경우 null|
| `--hires-scale` |Null: 요청된 목표 정책; 1 보다 큰 명시적인 계수는 레거시 패스별 스케일 체인을 선택합니다.|
| `--hires-width`, `--hires-height` |1 차 미세 조정 차원; 기본 가로 세로 비율에서 생략된 축을 추론한 후 1 차 패스의 축별 계수를 반복합니다.|
| `--hires-upscaler` |`lanczos`; 선택 `nearest`, `bilinear`, `bicubic`, `lanczos`|
| `--hires-denoising-strength` / `--hires-strength` |0.35 ; `0 < strength <= 1` 와 함께 유한한 값입니다.|
| `--hires-steps` |`--steps` 상속; 기본 대상 모드는 하나의 활성 단계에 대해 생략된 일정을 발생시킵니다. 명시적으로 빈 일정이 실패함|
| `--hires-seed` |`--seed` 를 상속; 각 미세 조정은 동일한 시드와 기존 `--seed-stride` 를 가진 이미지별 생성기를 새로 시작합니다.|
| `--hires-guidance-scale` |`--guidance-scale` 상속; 선택한 모델의 기존 제한 사항이 계속 적용됩니다.|
| `--hires-true-cfg-scale` |`--true-cfg-scale` 를 상속; 최소 1의 유한한 값이며, 1 보다 큰 값은 FLUX 에만 사용할 수 있습니다.|
| `--hires-scheduler` |`auto` : 1 단계의 실제 스케줄러 클래스와 구성을 상속하고, 새로 생성된 런타임 상태를 사용합니다.|
| `--hires-scheduler-config` |`{}`; 모든 구체화 스케줄러가 공유하는 검증된 생성자 재정의|
| `--hires-save-base` / `--no-hires-save-base` |거짓; 추가로 1 단계 PNG 와 그 사이카를 저장합니다.|

스케줄러 상속은 명시적 기본 재정의를 포함하여 실제로 실행된 클래스와 구성을 사용합니다. 미리 설정된 스케줄러 기본값과 원본 `--prediction-type`는 개선 시 다시 적용되지 않습니다. 명시적인 `--hires-scheduler-config`는 상속된 예측 구성을 재정의할 수 있습니다.

first-refinement dimension 과 함께 `--hires-scale` 를 지정하지 마십시오. 명시적 차원은 SD / SDXL 또는 FLUX 의 16 의 8 의 양의 배수여야 합니다. 계산된 차원은 필요한 가장 가까운 배수를 사용하며, 정확한 중간값은 위로 반올림됩니다. 각 패스에서 두 축은 모두 입력 축을 유지하거나 확대해야 하며, 적어도 하나는 증가해야 합니다. 두 차원을 모두 제공하는 경우 첫 번째 리파인먼트의 정확한 형식을 선택합니다. 그리고 나서 너비/베이스 너비 및 높이/베이스 높이 인자는 각 패스에서 독립적으로 반복되며, 패스마다 가족 반올림이 적용됩니다. 예를 들어, 512 × 512 에 `--hires-width 1024 --hires-height 768 --hires-passes 2` 를 사용하면 1024 × 768가 되고, 2048 × 1152가 됩니다. 따라서 명시적 차원은 리파인먼트 패스가 하나일 때만 최종 출력 크기를 이름으로 지정합니다.

예를 들어 첫 번째 미세 조정 너비를 선택하는 동안 종횡비를 유지합니다.

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py --model /absolute/path/image-diffusers \
  --preset sdxl-base --width 1024 --height 768 \
  --hires-fix --hires-passes 2 --hires-width 1536 \
  --hires-strength 0.4 --hires-steps 30 \
  --hires-guidance-scale 5 --hires-seed 1234
```

첫 번째 개선은 1536 × 1152입니다. 1.5 요소를 반복하면 2304 × 1728 최종 이미지가 생성됩니다. 첫 번째 미세 조정 너비 또는 높이가 명시적으로 선택된 경우 기본 업스케일 승수는 적용되지 않습니다.

JSON 또는 `resolve_request()`를 통해 동일한 요청을 제공할 수 있습니다.

```json
{
  "preset": "sdxl-base",
  "width": 1024,
  "height": 768,
  "hires_fix": true,
  "hires_passes": 2,
  "hires_width": 1536,
  "hires_denoising_strength": 0.4,
  "hires_steps": 30,
  "hires_guidance_scale": 5.0,
  "hires_seed": 1234
}
```

`hires_denoising_strength`는 표준 JSON/Python 키입니다. `--hires-strength`는 CLI 별칭입니다.

더 큰 디노이징 강도는 더 많은 노이즈를 추가하고 입력 이미지에서 더 많은 변화를 허용하며, 강도 1 는 전체 리파인먼트 일정을 사용합니다. 강도, 샘플링 단계 및 해상도의 선택은 모델 및 작업에 따라 달라집니다. 활성 카운트는 가족 및 스케줄러에 따라 달라집니다. SD / SDXL 는 `int(steps * strength) >= 1` 를 필요로 합니다. 고정된 FLUX img2img 구현은 `start = int(max(steps - min(steps * strength, steps), 0))` 를 사용하여 `start * scheduler.order` 에서 일정을 선택하며, 적어도 하나의 결과 단계가 남아 있어야 합니다. FLUX 의 기본값인 4 단계와 강도 0.35 는 따라서 2 개의 활성 세부 조정 단계를 선택합니다. 실제로 실행된 시간 단계는 기록되며, 일정 길이만으로는 실행 횟수가 아닙니다. 명시적인 `--hires-steps` 은 세부 조정의 양을 현저히 바꿀 수 있습니다.

<a id="composition-across-all-preset-stages"></a>

## 모든 사전 설정 단계에 걸친 구성

각 보정 패스는 덮어쓴 VAE, 활성 LoRA와 해당 강도, 선택적 ControlNet를 포함하여 선택한 신경망 구성요소를 재사용한다. 별도의 보정 체크포인트를 선택하거나 다운로드하지 않는다. 포지티브·네거티브 프롬프트 입력과 지원되는 어텐션 값은 계속 적용되며, 단계별 가이던스 덮어쓰기는 문서에 명시한 가이던스 값만 변경한다. 미리 계산한 임베딩은 기본 단계와 보정 단계의 가이던스 요구사항을 충족해야 한다. CPU 프롬프트 인코딩과 외부 임베딩은 기존 배치 정책을 유지하여 기본 이미지마다 보정 이미지 하나를 생성한다.

[학습된 텍스트 임베딩](text-embeddings.md) 는 `--text-embedding` 로 로드되어 모든 단계에서 등록된 토크나이저 항목과 인코더 벡터를 유지합니다. 다중 벡터 토큰은 각 인코더의 프롬프트 처리를 위해 확장되며, CPU 텍스트 인코딩을 포함합니다. 이러한 학습된 토큰은 텍스트 인코딩 경로를 사용하며, `--embeddings` 가 제공하는 완료된 프롬프트 텐서와 결합할 수 없습니다.

FLUX 에 대해, `--true-cfg-scale 1 --hires-true-cfg-scale 2 --negative-prompt blurry` 은 정제 단계에서만 부정적 조건화를 활성화할 수 있으며, `--negative-prompt-2` 도 함께 제공될 수 있습니다. 공유된 부정적 텍스트는 진술된 CFG 가 1보다 큰 경우에만 해당 단계로 전달됩니다. 기본 설정과 정제 설정이 모두 진술된 CFG 1를 가질 때 명시적 부정적 텍스트는 무효화됩니다. 각 저장된 단계의 `fixture.negative_prompt` 는 해당 단계가 부정적 조건화를 사용했는지 여부를 기록합니다.

ControlNet 는 각 단계의 해상도마다 동일한 준비된 조건화 이미지를 받습니다. 그 크기, 시작/종료 구간, 그리고 지원되는 추측/Union 모드는 모든 패스에서 계속 활성화되며, 구간은 각 단계의 활성 디노이징 일정에 적용됩니다. 각 정제는 이전 생성된 이미지를 img2img 입력으로 사용하며, 조건화 이미지는 별도의 ControlNet 입력으로 유지됩니다.

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py --model /absolute/path/image-diffusers \
  --preset flux1-schnell \
  --controlnet /absolute/path/to/schnell-controlnet-package \
  --control-image /absolute/path/to/prepared-condition.png \
  --controlnet-scale 0.8 \
  --hires-fix --hires-scale 1.5 --hires-upscaler bicubic \
  --hires-steps 8 --hires-strength 0.5 \
  --cpu-text-encoding --offload sequential
```

기존 `--latents` 입력은 기본 단계만 초기화합니다. 각 정제는 이전 단계의 리사이즈된 출력에서 이미지 잠재를 유도합니다. 동일하게, `--timesteps`, `--sigmas`, SDXL `--denoising-end` 는 기본 단계에만 적용됩니다. 각 정제는 `--hires-scheduler`, `--hires-scheduler-config`, `--hires-steps`, 및 강도 값에 의해 선택된 새로운 일정을 시작합니다. `--guidance-rescale` 도 첫 번째 단계 옵션으로 유지되며, 각 정제 패스는 0 가이드 리스케일링을 사용합니다. 그의 프롬프트 가이드는 대신 `--hires-guidance-scale` 로 선택할 수 있으며, FLUX, `--hires-true-cfg-scale` 에 대해서는 해당 패스의 높이/너비를 사용합니다. SDXL 의 정제 긍정 및 부정 크기 조건화는 기본 또는 첫 번째 정제 해상도를 유지하는 대신 해당 패스의 높이/너비를 사용합니다.

각 단계를 시작할 때 이미지별 생성기는 새로 시작됩니다. 이미지 `i` 은 첫 번째 단계에서 `seed + i * seed_stride` 를 사용하고 모든 개선 단계에서 `hires_seed + i * seed_stride` 를 사용합니다. 패스 인덱스는 이 시드를 자동으로 변경하지 않습니다. 동일한 단계 시드가 동일한 노이즈 텐서를 의미하지는 않습니다: img2img VAE 샘플링, 이미지 크기, 및 ControlNet 는 무작위 값을 다르게 소비할 수 있습니다. 재현 가능성은 단일 단계 생성과 마찬가지로 런타임 버전과 하드웨어에 의존합니다.

<a id="managed-local-checkpoint-backend"></a>

## 관리형 로컬 체크포인트 백엔드

공개 다운로드 파일 경로는 관리형 ComfyUI 이미지 워크플로를 통해 반복적인 HiRes 개선도 지원합니다.

```bash
python3 reference/generate.py \
  --model /absolute/path/to/illustrious.safetensors --base-model Illustrious \
  --width 1024 --height 1024 --steps 25 \
  --hires-fix --hires-passes 2 --hires-scale 1.5 \
  --hires-upscaler lanczos --hires-strength 0.35 --hires-steps 30 \
  --output-dir build/reference/illustrious-hires
```

이 백엔드는 `--hires-fix` / `--no-hires-fix`, `--hires-passes`, `--hires-scale`, `--hires-upscaler`, `--hires-steps`, `--hires-denoising-strength` / `--hires-strength`를 받는다. 활성화 기본값은 보정 1회·배율 2·Lanczos 크기 변경·강도 0.35이며, 보정 스텝은 기본 스텝 수를 상속한다. 각 패스는 직전 디코딩 이미지의 크기를 변경하고 선택한 VAE로 인코딩한 후 다른 보정 샘플러를 실행한다. 크기는 매 패스에서 이미지 계열이 요구하는 배수로 반올림한다. 네이티브 `SplitSigmasDenoise` 노드는 `round(steps * strength)`개의 보정 스텝을 유지하며, 최소 하나는 남아야 한다. 이는 관리되는 스케줄러 규칙으로 위의 SD/SDXL 및 FLUX 프리셋 규칙과 별개이다. 관리 경로는 `[1, 4096]`의 스텝을 검증하고 서버 시작 전에 계획한 이미지의 어느 축이든 네이티브 `ImageScale` 한계인 16384를 넘으면 거부한다.

관리 백엔드는 preset 의 `--hires-width`, `--hires-height`, `--hires-seed`, guidance/true- CFG /scheduler 오버라이드, `--hires-scheduler-config`, 또는 `--hires-save-base` 를 받지 않습니다. 이러한 지원되지 않는 옵션은 명시적으로 실패합니다. 최종 이미지를 저장하고 중간 개선 또는 추가 기본 이미지를 게시하지 않습니다. 지원되는 기본 모델, 구성 요소, 장치 및 출력 디렉토리 규칙은 관리 이미지 라우트의 규칙과 동일하게 유지됩니다.

`request.hires` 는 요청된 패스, 스케일, 업스케일러, 강도, 단계 및 정렬된 타겟 크기 계획을 기록합니다. 저장된 ComfyUI 그래프와 그 완전한 실행 결과가 리파인먼트 체인이 실행되었음을 확립하며, 최종 이미지 유효성 검사기는 제공된 아티팩트를 확인합니다. 계획된 단계 항목은 단계별 픽셀 해시 또는 Diffusers 디노이징 콜백 증거를 주장하지 않습니다.

<a id="preset-output-identity-and-execution-evidence"></a>

## 사전 설정된 출력 ID 및 실행 증거

명시적인 출력 경로가 없는 경우, `-hires` 는 `-custom`, `-vae`, `-lora`, `-embedding`, 및 `-controlnet` 수정자를 따릅니다. 따라서 SD 1.5 베이스 모델 Hires Fix 실행은 `build/reference/sd15-red-cube-hires.png` 을 작성합니다. `--output` 는 최종 결과를 이름 짓습니다. `--hires-save-base` 가 활성화되면 첫 번째 단계 파일은 `-base` 를 최종 스템에 추가합니다: `result.png` 는 `result-base.png` 를 가지며, 배치된 `result-0001.png` 는 `result-0001-base.png` 를 가집니다. 기본적으로 오직 최종 리파인먼트만 저장됩니다. `--hires-save-base` 는 또한 원래 베이스 이미지를 저장하며, 중간 리파인먼트 이미지는 메모리에서 소비되고 별도의 파일로 게시되지 않습니다.

모든 최종/베이스 PNG 와 그 JSON 사이카는 생성 전에 출력 충돌 검사에 참여합니다. 기본 경로 충돌은 사용하지 않는 런 접미사를 선택하며, 명시적 경로 충돌은 `--overwrite` 가 활성화되지 않는 한 실패합니다. 여러 출력 파일을 게시하는 것은 기존 파일별 원자적 쓰기 정책을 유지하며 파일 시스템 트랜잭션이 아닙니다.

사이카는 `parameters` 에서 해결된 요청을 유지합니다. `hires_fix.base` 는 첫 번째 단계 크기, 시드, 파이프라인/스케줄러, 실제 디노이징 단계, 타임스텝, 유한 잠재력 검사, 구성 요소 데이터 타입, 안전 상태 및 픽셀 해시를 기록합니다. `hires_fix.requested_passes` 와 `hires_fix.completed_passes` 는 요청된 리파인먼트 횟수와 완료된 리파인먼트 작업을 구별합니다. 정렬된 `hires_fix.stages` 목록은 각 리파인먼트의 `input`, `upscale`, `refinement` 및 `output` 을 기록합니다. 이것들은 이미지 식별자 및 크기, 보간 및 확대, 실제 노이즈 제거 단계/시간 단계, 유한 잠재력 확인 및 스케줄러 구성을 담고 있습니다. 각 항목의 입력은 기본 이미지로 시작하는 이전 출력입니다. 기존 `hires_fix.upscale` 및 `hires_fix.refinement` 필드는 여전히 사용 가능하며 마지막 개선 단계를 설명합니다. 최종 PNG 식별자는 `output` 에 있으며, 최종 안전성 및 구성 요소 데이터 타입은 `safety` 와 `runtime.component_dtypes` 에 있습니다. 선택적 기본 사이더는 `artifact_role: "hires_base"` 를 사용하며, 해당 최상위 필드에서 기본 이미지를 설명하고 완료된 이미지를 `final_output` 를 통해 연결합니다. 각 단계는 요청된 차원을 생성하고 비균일 RGB 출력 검사를 통과해야 합니다. 런타임 잠재력 콜백은 노이즈 제거가 실행되었음을 확립하고 비유한 잠재력 값을 거부합니다. 개선 시퀀스에서의 실패는 오류이며, 아무런 알림 없이 이전에 생성된 이미지나 확대 전용 이미지를 성공적인 Hires Fix 결과로 반환하지 않습니다.

테스트 및 연막 생성은 인수 라우팅, 실행된 확산 단계, 이미지 차원, 구성, 및 무효 출력 거부법을 확립합니다. 모든 프롬프트, 체크포인트, 또는 설정에 대한 지각적 품질을 보장할 수는 없습니다. Hires Fix는 반복 가능한 개선 절차와 그 실행 증거를 제공합니다; 의도된 모델 및 작업에 대한 시각적 평가는 여전히 필요합니다.

<a id="dependencies-and-scope"></a>

## 종속성 및 범위

Pillow 는 4 지원 RGB 리사이즈 방법을 수행합니다. 기존 Diffusers 와 PyTorch 는 VAE 인코딩, 노이즈, img2img 스케줄링 및 신경 실행을 제공하며, Accelerate 는 오프로드 후크의 소유권을 유지합니다. 반복된 패스를 위해 새로운 패키지, 학습된 업스케일러 가중치, 또는 커스텀 텐서/핵수 구현이 도입되지 않습니다. 관리된 백엔드는 기존 ComfyUI 이미지 리사이즈, VAE 및 샘플러 노드를 재사용합니다. 네이티브 C++ 는 핀 엔진의 Lanczos Hires 구현을 사용합니다; [네이티브 생성](native-image-generation.md)을 참조하세요. 이 인터페이스는 임의의 외부 img2img 입력, 잠재 공간 업스케일링 또는 SDXL 리파이너 어셈블리를 노출하지 않습니다.

상위 공급 측 계약은 다음과 같이 설명됩니다.
[Diffusers 이미지 간 가이드](https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/img2img.md) 및 [Pillow는 API](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize)크기를 조정합니다.

<a id="repeat-validation"></a>

## 검증 반복

현재 기본 경로의 오프라인 실행 검증은 다음 명령으로 재실행한다. 첫 명령은 작은
무작위 SD1/SDXL 모델로 두 단계의 실제 디노이징, 비정사각형 크기, 미리보기,
연속 요청의 모델 재사용을 검사한다. 두 번째는 지정한 로컬 체크포인트와 빌드된
C 브리지로 네이티브 생성을 실행한다. 결과와 JSON은 `build/half-resolution-*`에 남는다.

```sh
reference/diffusers/.venv/bin/python tests/HalfResolutionSmoke.py
IILD_NATIVE_DIAGNOSTICS=1 reference/diffusers/.venv/bin/python tests/HalfResolutionSmoke.py \
  --native-model /absolute/path/model.gguf
```

아래 기록은 명시적 확대 옵션을 사용한 기존 반복 체인의 검증이다.

반복 경로는 SD1.5, SDXL와 FLUX에 대해 각각 ControlNet가 있는 경우와 없는 경우에 3회 개선을 수행하여 실행하였다. ControlNet 사례는 LoRA, CPU 프롬프트 인코딩과 2개 이미지 배치도 유지한다. 이 합성/소형 모델은 임의 체크포인트의 시각적 품질이 아니라 실행과 구성을 검증한다.

- SD1.5/SDXL: 축당 `64 → 96 → 144 → 216` 픽셀.
- FLUX: `64 → 96 → 144 → 224`(16픽셀 반올림 요구 사항 포함)
- 각 단계는 비어 있지 않은 디노이징, 유한한 잠재 표현, 균일하지 않은 디코딩 이미지와 이전 단계의 출력 해시와 동일한 입력 해시를 가지고 있었습니다.
- 이후 단계 실패는 부분 체인을 성공으로 반환하지 않고 전파됩니다.
- 설치된 ComfyUI 노드 스키마는 152 반복 워크플로우 변형을 허용했으며, 해당 검사에서의 픽스처 인벤토리명은 모델 가중치 추론을 증명하지 않습니다.

증거는 `build/reference/repeated-hires/six-paths/verification.json` 와 `build/workflow-source-review/hires-repeat-live-schema-validation.json` 에 유지됩니다. 공공 관리된 CLI 도 기존 원본 SDXL 체크포인트를 통해 `128 → 192 → 288 → 432` 를 거쳐 비균일한 432×432 PNG 를 생성했으며, 3 개의 별도 리파인먼트 샘플러를 사용했습니다. 원본/단계별 파일 식별자와 출력 SHA256 가 검증되었으며, 소유된 엔진 프로세스가 성공적으로 종료되었습니다. 이는 `build/reference/repeated-hires/managed-sdxl/local-image.json` 에 기록되어 있으며, 해당 픽스처 에 명시된 Illustrious 카테고리는 Illustrious 학습된 가중치의 증거가 아닙니다. CMake 빌드 및 53 CTest 경우가 통과했습니다. Python 발견은 572 경우를 실행했으며: 570 가 통과했고, 2 기존 옵트인 체크포인트 변환 경우는 건너뛰었습니다. `build/hires-repeat-ctest.log` 와 `build/hires-repeat-tests.log` 는 로그를 유지합니다.
