<a id="automatic-local-image-workflows"></a>

# 자동 로컬 이미지 워크플로

`reference/diffusers/comfyui_image_workflow.py`는 Civitai 기본 모델 이름, 등록된 로컬 가중치 이름, 프롬프트, 명시적 생성 입력과 실행 중인 서버의 `/object_info`로 ComfyUI API 그래프를 조립한다. 이 레시피에서 사용자가 워크플로를 작성하거나 내보낼 필요는 없다. 빌더는 추가 Python 의존성이 없고 파일 시스템/네트워크 I/O를 수행하지 않으며 Torch 또는 ComfyUI를 가져오지 않는다. 런타임 설정, 파일 등록, 제출과 아티팩트 공개는 호출자의 책임이다.

```python
from comfyui_image_workflow import build_workflow

graph = build_workflow(
    base_model="Illustrious",
    model_name="my-illustrious.safetensors",  # 정확한 서버 인벤토리 이름
    components={},
    prompt="a lighthouse above a quiet bay, painted illustration",
    object_info=client.json("/object_info"),
    negative_prompt="blur",
    width=1024, height=1024, seed=42, steps=25, cfg=7.0,
)
```

`workflow_requirements(base_model)` 는 서버를 시작하지 않고 레시피, 기준 해상도, 샘플링 기본값, 텍스트 로더 유형, 및 필수 분할 컴포넌트 역할을 노출하며, 이는 레시피를 설명하는 것이며 추론 결과가 아닙니다. Civitai 라벨은 의도된 아키텍처를 선언하고, ComfyUI 의 텐서 로더는 제출된 그래프가 실행될 때 파일들을 확인합니다.

<a id="weight-roles"></a>

## 가중치 역할

A `checkpoint` 는 `CheckpointLoaderSimple` 출력 MODEL, CLIP 및 VAE 를 사용합니다. 명시적 구성 요소는 체크포인트의 디노이저를 대체하지 않고 인코더/ VAE 출력을 대체합니다. A `diffusion_model` 는 `UNETLoader` 를 사용하며 별도의 인코더 및 VAE 파일이 필요합니다. `model_type="auto"` 는 라이브 재고 양쪽을 모두 확인하며, 이름이 두 재고 모두에 나타나면 호출자는 `checkpoint` 또는 `diffusion_model` 를 지정해야 합니다.

`components`는 다음과 같은 정확한 키를 허용합니다.

|키|역할|
| --- | --- |
| `vae` |디코더 VAE ; 스테이블 캐스케이드용으로 이것은 A 단계입니다.|
| `text_encoder` |첫 번째 인코더 또는 단독 인코더|
| `text_encoder_2` |두 번째 인코더|
| `text_encoder_3` |세 번째 인코더|
| `text_encoder_4` |네 번째 엔코더|
| `model_negative` |Ideogram4의 별도 무조건 디노이저|
| `decoder` |안정적인 캐스케이드의 B단계 잡음 제거기|

인코더 키는 연속적이어야 한다. Flux.1은 CLIP-L + T5XXL을 사용하며 SDXL은 CLIP-L + CLIP-G을 사용한다. SD3은 상위 공급 측의 단일/이중/삼중 로더를 통해 하나, 2개 또는 3개의 일치하는 인코더를 받는다. HiDream-I1은 상위 공급 측의 단일/이중/사중 형식을 받는다. 전체 형식은 CLIP-L, CLIP-G, T5XXL와 Llama 3.1이다. 축소 형식도 각 역할에 맞는 올바른 인코더 텐서가 필요하다.

GGUF 디노이저는 `UnetLoaderGGUF` 를 선택합니다. GGUF 텍스트 인코더는 일치하는 `CLIPLoaderGGUF`, `DualCLIPLoaderGGUF`, `TripleCLIPLoaderGGUF` 또는 `QuadrupleCLIPLoaderGGUF` 를 선택합니다. 누락된 로더 또는 지원되지 않는 아키텍처/열거형은 제출 전에 실패합니다. 파일 확장자만은 선택적 GGUF 로더의 텐서 호환성을 증명하지 않습니다. 빌더는 커스텀 노드를 설치하지 않습니다.

<a id="recipes-and-baseline-defaults"></a>

## 레시피 및 기준 기본값

|베이스 제품군|이미지 잠재/특수 동작|베이스라인|
| --- | --- | --- |
| SD1 |4채널 SD 잠재|512, 20 단계, CFG 7|
| SD2 |SD 잠재 표현 ; `768` 카테고리는 768를 사용합니다|512 또는 768, 20, CFG 7|
|SDXL, Pony, Illustrious, NoobAI|SDXL 잠재 및 번들 또는 듀얼 인코더| 1024, 25, CFG 7 |
| SD3 / 3.5 |16채널 잠재; 싱글/듀얼/트리플 SD3 인코더| 1024, 28, CFG 5 |
|Flux.1 dev / Krea|16채널 잠재성; 별도의 내장 지침 3.5| 1024, 20, CFG 1 |
|Flux.1 Schnell|내장된 안내 없음| 1024, 4, CFG 1 |
|Flux.2 dev|128채널 Flux2 잠재 및 `Flux2Scheduler`; 내장형 안내 4| 1024, 20, CFG 1 |
|Flux.2 Klein|Flux2 잠재/스케줄러; 증류수와 기본 설정이 다름|증류수 4 / CFG 1; 베이스 20 / CFG 5|
|AuraFlow / 포니 V7|4채널 잠재성, T5-XL 인코더가 상위 공급 측| 1024, 30, CFG 3.5 |
|크로마|T5 패딩 비활성화됨, 1이동, 베타 일정| 1024, 26, CFG 3.5 |
|PixArt 알파 / 시그마|Alpha는 추가로 해석을 인코딩하고, Sigma는 인코딩하지 않습니다.| 1024, 30, CFG 4.5 |
| HunyuanDiT |번들 체크포인트 및 BERT/MT5 프롬프트 인코딩| 1024, 30, CFG 6 |
| HiDream-I1 |16채널 잠재 및 네이티브 인코더는| 1024, 50, CFG 5 |
| HiDream-O1 |번들 체크포인트, 네이티브 픽셀 공간 잠재/변환 VAE| 2048, 40, CFG 5 |
|Qwen 이미지|16채널 잠재, Qwen 인코더 및 시프트 3.1| 1328, 20, CFG 4 |
|Lumina|Lumina2 인코더 및 해당 `superior` 시스템 프롬프트| 1024, 30, CFG 4 |
|ZImageBase / Turbo|Lumina2 로더는 Qwen3 인코더를 감지합니다. 16채널 잠재|기본 25 / CFG 4; 터보 8 / CFG 1|
|Anima|Qwen3-0.6B 인코더가 상위 공급 측에 의해 감지되었으며, 이미지 VAE| 1024, 30, CFG 4 |
|Ernie|Ministral 인코더 및 Flux2 latent/VAE| 1024, 20, CFG 4 |
|Boogu|Boogu 인코더와 네이티브 디노이저; 베이스라인은 Turbo입니다.| 1024, 4, CFG 1 |
| Krea 2 |Krea2 인코더와 네이티브 디노이저; 베이스라인은 Turbo입니다.| 1024, 8, CFG 1 |
|렌즈|Flux2 잠재/VAE, 크기 인식 Flux 시프트, 사전 CFG 표준| 1440, 20, CFG 5 |
| MageFlow |네이티브 인코더는 조정 및 매칭을 모두 제공합니다 잠재 표현| 1024, 30, CFG 5 |
|이데오그램4|조건부/무조건 모델 및 네이티브 스케줄러를 분리하십시오.| 1024, 20, CFG 7 |
|안정적인 캐스케이드|C 단계 샘플링 → B 단계 컨디셔닝/샘플링 → A 단계 VAE| 1024; C 20 / CFG 4; B 10 / CFG 1 |

이 설정들은 모든 파인튜닝, 증류 변형 또는 Civitai 카테고리를 공유하는 모델 버전의 보장을 위한 것이 아니라 편집 가능한 기준선입니다. 특히, LCM, Turbo, Lightning, Hyper, HiDream fast/dev 및 Krea2 raw 변형은 체크포인트별 샘플링 계약을 갖습니다. 명시적 인수를 통해 정확한 모델 카드 설정이 우선순위를 가집니다. 여기서 `Lumina` 는 Lumina2 를 의미하며, Lumina1 번드도 자체 호환 파이프라인이 필요합니다. HunyuanDiT 의 BERT/MT5 라우트는 현재 내장된 `DualCLIPLoader` 가 `hunyuan_dit` 열거형을 노출하지 않기 때문에 광고되지 않습니다. HiDream-O1 는 체크포인트 주입 토크나이저와 픽셀 공간 변환 VAE 를 사용합니다.

<a id="validation-and-explicit-controls"></a>

## 검증 및 명시적 통제

빌더는 라이브 `/object_info` 와 연결 출력 타입, 포함된 노드 제외, 필수 입력, 모델 및 샘플러 열거형, 스칼라 타입/범위를 포함한 모든 생성된 노드/입력에 대해 확인합니다. 알 수 없는 구성 요소 키와 소모되지 않은 옵션은 안전하게 거부됩니다. 잠재 표현 노드를 구성하기 전에 차원 나누어 떨어지는지 확인하며, 요청된 이미지 크기를 아무런 알림 없이 아무런 알림 없이 반올림하지 않습니다.

선택적 인수는 `negative_prompt`, `width`, `height`, `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `batch_size`, `prediction_type`, `model_type`, `guidance`, `sampling_shift`, `zsnr`, 및 `clip_skip` 입니다. 제외된 값은 레시피 또는 체크포인트 기본값을 유지하며, `seed` 를 제외하면: 제외는 그래프에 대한 새 랜덤 32비트 시드를 선택합니다. 시드 0를 포함한 명시적 시드는 보존되며, 모든 정제 패스는 해당 그래프의 시드를 공유합니다. `guidance` 는 Flux 의 내장 가이드이며 CFG 와는 별개입니다. Flux2 와 Ideogram4 는 네이티브 이름 지정 스케줄을 요구합니다. `prediction_type` 는 `epsilon`, `v_prediction`, `sample`, 및 `lcm` 을 SD1/2/SDXL 에만 지원하며, `zsnr=True` 는 명시적 예측 타입을 요구합니다. NoobAI 은 범주 때문에 오직 V-예측으로만 가정되지 않습니다. `clip_skip=0` 는 최종 CLIP 레이어를 의미하며, `clip_skip=1` 는 바로 앞의 레이어를 선택합니다.

빌더는 텍스트-이미지 작업 능력을 요구합니다. 이미지 편집, 업스케일링, 비디오, 오디오 및 3D 리소스는 각각 자체 입력/출력 워크플로우를 필요로 합니다. 호스팅된 및 알 수 없는 범주는 명시적으로 실패합니다. Kolors 는 이 모듈에서 검증된 네이티브 ComfyUI 레시피가 없으며, 일치하는 로컬 Diffusers 파이프라인을 통해 계속 이용 가능합니다. LoRA , 임베딩 또는 VAE 파일은 구성 요소이며, 완전한 노이즈 제거기/체크포인트가 아닙니다.

`tests/ComfyUIImageWorkflowTests.py` 는 인코더의 아리티, 구성 요소 교체, 잠재 표현 선택, 샘플/조건부 흐름, NoobAI 예측 오버라이드, Flux2 일정, 2-단계 캐스케이드, Ideogram4 조건부 없는 모델, GGUF 노드 경계 및 스키마 오류를 다운로드하거나 가중치를 로드하지 않고 확인합니다. 이 테스트들은 그래프 계약을 수립하며, 실제 생성 또는 시각적 품질 증거가 아닙니다.

ComfyUI 커밋에 대해 소스 확인
[`e80c1570b6b44a2557d5d8e341e05782d18c9bbb`](https://github.com/Comfy-Org/ComfyUI/tree/e80c1570b6b44a2557d5d8e341e05782d18c9bbb):
[로더](https://github.com/Comfy-Org/ComfyUI/blob/e80c1570b6b44a2557d5d8e341e05782d18c9bbb/nodes.py),
[텍스트 인코더 파견](https://github.com/Comfy-Org/ComfyUI/blob/e80c1570b6b44a2557d5d8e341e05782d18c9bbb/comfy/sd.py),
[모델 계약](https://github.com/Comfy-Org/ComfyUI/blob/e80c1570b6b44a2557d5d8e341e05782d18c9bbb/comfy/supported_models.py),
[네이티브 이미지/샘플러 노드](https://github.com/Comfy-Org/ComfyUI/tree/e80c1570b6b44a2557d5d8e341e05782d18c9bbb/comfy_extras)및 [공식 워크플로 템플릿](https://github.com/Comfy-Org/workflow_templates/tree/main/templates).
