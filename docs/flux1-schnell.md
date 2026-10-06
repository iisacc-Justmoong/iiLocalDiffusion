<a id="flux1-schnell-reference-contract"></a>

# FLUX.1-schnell 참조 계약

<a id="canonical-reference"></a>

## 정식 참조

FLUX 오라클은 Black Forest Labs의 공식 Hugging Face 저장소입니다.
불변 개정판의 [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell):

```text
741f7c3ce8b383c54771c7003378a50191e9efe9
```

해당
[`741f7c3` 스냅샷 트리](https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/741f7c3ce8b383c54771c7003378a50191e9efe9)는 이 프로필의 메타데이터 소스입니다.

저장소는 Apache-2.0을 선언한다. 그러나 Hugging Face에서 접근 승인이 필요하므로 공식 가중치를 다운로드하기 전에 저장소 약관에 동의하고 인증해야 한다. 이 프로필은 FLUX.1-dev를 받지 않는다. Dev는 트랜스포머에 guidance를 내장하며 라이선스와 스케줄러 의미가 다르다.

아래 값은 공식을 따릅니다.
[`FluxPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/flux) 및
[`FluxTransformer2DModel`](https://huggingface.co/docs/diffusers/api/models/flux_transformer) 계약 및 고정된 저장소 메타데이터.

허용되는 Diffusers 레이아웃은 다음과 같습니다.

```text
model-root/
├── model_index.json
├── tokenizer/
├── tokenizer_2/
├── text_encoder/
│   ├── config.json
│   └── *.safetensors
├── text_encoder_2/
│   ├── config.json
│   └── *.safetensors
├── transformer/
│   ├── config.json
│   └── *.safetensors
├── vae/
│   ├── config.json
│   └── *.safetensors
└── scheduler/
    └── scheduler_config.json
```

Tokenizer 1 는 `tokenizer_config.json`, `vocab.json`, 및 `merges.txt` 를 필요로 합니다. Tokenizer 2 는 `tokenizer_config.json`, `spiece.model`, 및 `tokenizer.json` 를 필요로 합니다. 분할된 safetensors 는 정형화된 인덱스 JSON 를 필요로 하며, `weight_map` 는 완전한 연속적인 분할 집합을 참조해야 합니다. 모든 필요한 신경 구성 요소는 비어 있지 않은 정규 아티팩트 하나 이상을 가져야 합니다.

패키지 인덱스 이름은 토크나이저 2 `T5TokenizerFast`입니다. Transformers 5.x는 런타임 클래스 이름 `T5Tokenizer`에 대한 구현 별칭을 지정하고 4.x는 `T5TokenizerFast`를 보고합니다. Python 오라클은 런타임 철자를 허용합니다.

<a id="component-metadata"></a>

## 구성요소 메타데이터

C++ `Flux1Schnell` 프로필은 이 프롬프트-이미지 계약을 검증합니다.

|컴포넌트|클래스|검사자가 사용하는 계약|
|---|---|---|
| Tokenizer 1 | `CLIPTokenizer` |최대 시퀀스 길이 77|
| Tokenizer 2 |`T5TokenizerFast` 패키지 인덱스에서|T5 런타임 토크나이저; 최대 시퀀스 길이 512|
|텍스트 인코더 1| `CLIPTextModel` |숨겨진/투영 너비 768; 중간 너비 3,072; 12 헤드; 12 레이어; 어휘 49,408|
|텍스트 인코더 2| `T5EncoderModel` |모델 너비 4,096; `gated-gelu`를 사용한 피드포워드 폭 10,240; 64 헤드; 24 레이어; 어휘 32,128|
|디노이저| `FluxTransformer2DModel` |패킹형 채널 64; 19 듀얼 스트림 및 38 단일 스트림 블록; 24 너비 128 헤드; 조인트 폭 4,096; 안내 임베딩 없음|
|이미지 코덱| `AutoencoderKL` |RGB 입력/출력; 잠재 채널 16; 샘플 크기 1,024; 0.3611 규모; 0.1159 이동; 4개의 인코더/디코더 레벨|
|스케줄러| `FlowMatchEulerDiscreteScheduler` |1,000 훈련 단계; 시퀀스 범위 256 ~ 4,096; 기본/최대 이동 0.5/1.15; 고정 시프트 1.0|

정규 변환기의 회전 축은 `[16, 56, 56]` 입니다. Diffusers 은 소스 JSON 에서 이를 생략하더라도 이 생성자 기본값을 구현할 수 있으므로, C++ 패키지 인스펙터는 값이 존재할 때 해당 값을 검증하고, 로드된 Python 오라클은 유효한 런타임 값을 검증합니다. 변환기 출력 채널도 JSON 에서 부재하거나 null 일 수 있으며, 유효한 값은 반드시 64이어야 합니다.

<a id="tensor-flow"></a>

## 텐서 흐름

프롬프트 배치 `B`, 이미지 배치 `E`, 이미지 크기 `H` 및 `W`의 경우:

```text
CLIP token IDs                                  [B, 77]
T5 token IDs                                    [B, <=256]
pooled CLIP conditioning                        [E, 768]
T5 token conditioning                           [E, <=256, 4096]
VAE latent                                      [E, 16, H/8, W/8]
2 x 2 packed latent sequence                    [E, (H/16)*(W/16), 64]
transformer noise prediction                    [E, (H/16)*(W/16), 64]
unpacked latent                                 [E, 16, H/8, W/8]
VAE-decoded image                               [E, 3, H, W]
```

각 토큰은 16채널 VAE 잠재의 2 영역별로 2를 팩하기 때문에 변환기는 64 채널을 사용합니다. 결과적으로 지원되는 이미지 너비와 높이는 16의 양의 배수여야 합니다.

<a id="fixed-reference-fixture"></a>

## 고정 참조 픽스처

```text
프롬프트:               a red cube on a white table
부정 프롬프트:          전달하지 않음
시드:                   42
해상도:                 1024 x 1024
추론 단계:              4
Guidance scale:         0.0
최대 프롬프트 길이:     256
스케줄러:               고정된 저장소의 FlowMatchEulerDiscreteScheduler
```

Schnell 은 4 개의 디노이징 단계에서 고품질 생성을 위해 최적화되어 있으며, 4 는 견고한 파이프라인 한계가 아닌 재현 가능한 픽스처 선택입니다. 오라클은 BF16 를 CPU 와 가속기 로딩에 모두 사용하며, 존재하지 않는 `fp16` 저장소 변형을 요청하지 않습니다. MPS 와 CUDA 에서 순차적 CPU 오프로드 및 VAE 슬라이싱과 타일링을 적용하며, 전체 파이프라인을 하나의 거주 할당으로 가속기로로 이동하지 않습니다. 이것들은 숨겨진 상수가 아닌 기본값이며, 데이터 타입, 토큰 제한, 메모리 정책, 시드/배치 및 스케줄러 값은 CLI / JSON 를 통해 공급할 수 있습니다. 선택적 `--true-cfg-scale > 1` 는 Schnell `guidance_scale=0` 계약 변경 없이 별도의 부정 프롬프트 CFG 패스를 가능하게 합니다. 보라
값 및 대체 경로 규칙에 대한 [생성 매개변수](generation-parameters.md)입니다.

<a id="compatibility-boundary"></a>

## 호환성 경계

`Flux1Schnell`는 정규 Schnell 패키지 메타데이터와 필수 아티팩트 경로가 일치함을 뜻한다. safetensors 본문을 파싱했거나, 인증된 공식 가중치를 다운로드했거나, 추론을 실행했다는 뜻은 아니다. GGUF, 단일 파일 체크포인트, 양자화 파생물, ControlNet, FLUX.1-dev와 임의의 제3자 FLUX 배치는 이 C++ 패키지 프로필의 범위 밖에 있다. LoRA도 마찬가지로 C++ 패키지 프로필의 범위 밖에 있다.

독립적인 Python 생성 오라클은 `--model` 를 통해 호환되는 Schnell 변환기 파일을, 별도의 `AutoencoderKL` 파일을 통해 `--vae` 를, 그리고 선택적인 `--lora` 를 하나 받습니다. 그의 `--model-config` 소스는 CLIP/T5 인코더, 토크나이저, 스케줄러, 구성 요소 설정 및 대체되지 않으면 기본 VAE 를 공급합니다. 단일 변환기 파일은 자체 완결형 FLUX 파이프라인이 아닙니다. 네이티브 Diffusers 와 지원되는 원본 형식 변환기 가중치는 고정된 Diffusers 단일 파일 로더를 사용하며, `.safetensors` 와 로컬 `.safetensor` 파일명은 허용됩니다. 모든 조립된 구성 요소는 LoRA 활성화 및 생성 전에 Schnell 계약을 반드시 만족해야 합니다. 이는 Dev 가이드 임베딩, 양자화된 가중치 또는 대체 잠재 표현 계약을 허용하지 않습니다. 소스 선택 및 출처에 대해서는 [model-inputs.md](model-inputs.md) 를 참조하십시오.

Python 오라클은 실제 가중치를 로드하여 유효한 구성 요소 클래스와 핵심 설정을 확인하고 RGB PNG 를 생성하는 더 강력한 검사를 수행합니다. FLUX.1 -schnell 은 안전 검사기나 워터마커를 포함하지 않으며, 패키지 호환성을 제품 안전 정책으로 제시해서는 안 됩니다.

현재 32 GiB M1 Max 검증 호스트는 고정된 작은 FLUX 픽스처 를 사용하여 동일한 API 경로를 실행했지만, 저장소가 인증 게이트 액세스를 필요로 하므로 공식 12B 가중치를 로드하지 않았습니다. 작은 실행은 프레임워크 및 MPS 호출 호환성만 증명하며, 공식 모델의 메모리 적합성, 수치적 정확성, 이미지 품질 또는 프로덕션 준비 상태를 증명하지 않습니다.
