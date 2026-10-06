<a id="stable-diffusion-xl-base-reference-contract"></a>

# 안정 확산 XL 기본 참조 계약

<a id="canonical-reference"></a>

## 정식 참조

SDXL 오라클은 공식 Hugging Face 저장소입니다.
불변 개정판의 [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0):

```text
462165984030d82259a11f4367a4eed129e94a7b
```

저장소는 CreativeML Open RAIL++-M 라이선스를 선언합니다. 개정 핀은 오라클을 재현 가능하게 만듭니다. 모델 라이센스를 양도하거나 확대하지 않습니다. SDXL 0.9는 액세스 및 라이센스 조건이 다르며 이 프로필이 아닙니다.

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
├── unet/
│   ├── config.json
│   └── *.safetensors
├── vae/
│   ├── config.json
│   └── *.safetensors
└── scheduler/
    └── scheduler_config.json
```

각 토크나이저 디렉토리는 자신의 `tokenizer_config.json`, `vocab.json`, 및 `merges.txt` 을 유지해야 합니다. 2 토크나이저는 고정된 저장소에서 어휘 내용을 공유하지만 패딩 구성이 다르므로 추론 구현은 이를 하나의 구성된 객체로 축소해서는 안 됩니다.

<a id="component-metadata"></a>

## 구성요소 메타데이터

C++ `StableDiffusionXLBase` 프로필은 공식 Base 1.0 프롬프트 이미지 패키지 계약을 검증합니다.

|컴포넌트|클래스|검사자가 사용하는 계약|
|---|---|---|
| Tokenizer 1 | `CLIPTokenizer` |최대 시퀀스 길이 77|
| Tokenizer 2 | `CLIPTokenizer` |최대 시퀀스 길이 77|
|텍스트 인코더 1| `CLIPTextModel` |숨겨진 너비 768; 투사 폭 768; 최대 위치 77|
|텍스트 인코더 2| `CLIPTextModelWithProjection` |숨겨진 및 투영 너비 1,280; 최대 위치 77|
|디노이저| `UNet2DConditionModel` |잠재 표현 채널 4 ; 샘플 크기 128 ; 교차 주의 폭 2,048 ; `text_time` 추가 컨디셔닝|
|이미지 코덱| `AutoencoderKL` |RGB 입력/출력; 잠재 채널 4; 샘플 크기 1,024; 0.13025 규모; 강제 업캐스트|
|스케줄러| `EulerDiscreteScheduler` |1,000 훈련 단계; 베타 0.00085 ~ 0.012; 확장된 선형 일정; 엡실론 예측; 선행 시간 간격|

UNet의 `addition_time_embed_dim`는 256이며, `projection_class_embeddings_input_dim`는 2,816입니다. 이 값들은 2 교차 구성 요소 불변량을 인코딩합니다:

```text
768 + 1280 = 2048
6 * 256 + 1280 = 2816
```

첫 번째는 연결된 토큰 조절 너비입니다. 두 번째는 6개의 마이크로 컨디셔닝 값을 텍스트 인코더 2의 풀링된 투영과 결합합니다. 또한 검사관은 파이프라인 인덱스의 `force_zeros_for_empty_prompt=true`와 UNet의 `use_linear_projection=true`를 요구합니다.

<a id="tensor-flow"></a>

## 텐서 흐름

프롬프트 배치 `B`, 유효 이미지 배치 `E` 및 분류자 없는 안내 요소 `G`(활성화된 경우 `2`, 그렇지 않은 경우 `1`)의 경우:

```text
각 토크나이저의 토큰 ID                          [B, 77]
텍스트 인코더 1의 끝에서 두 번째 은닉 상태       [B, 77, 768]
텍스트 인코더 2의 끝에서 두 번째 은닉 상태       [B, 77, 1280]
이어 붙인 프롬프트 임베딩                        [G*E, 77, 2048]
텍스트 인코더 2의 풀링된 임베딩                  [G*E, 1280]
미세 조건 ID                                    [G*E, 6]
초기 잠재 표현                                  [E, 4, H/8, W/8]
UNet 잠재 표현 입력과 노이즈 예측                [G*E, 4, H/8, W/8]
guidance가 적용된 잠재 표현                      [E, 4, H/8, W/8]
VAE로 디코딩한 이미지                           [E, 3, H, W]
```

6 미조절 값은 원래 높이, 원래 너비, 상단 자르기, 좌측 자르기, 목표 높이, 및 목표 너비입니다. 정준 1,024 by 1,024 해상도에서 잠재 공간 크기는 128 by 128입니다. VAE 디코드는 잠재를 0.13025 로 나누고 `force_upcast` 가 참이므로 부동소수점32 을 사용합니다.

<a id="fixed-reference-fixture"></a>

## 고정 참조 픽스처

```text
프롬프트:           a red cube on a white table
부정 프롬프트:      빈 문자열
시드:               42
해상도:             1024 x 1024
추론 단계:          20
Guidance scale:     5.0
스케줄러:           고정된 저장소의 EulerDiscreteScheduler
워터마커:           명시적으로 비활성화
```

20단계는 Diffusers 호출 기본값이 아니라 iiLocalDiffusion 연기/참조 선택입니다. 빈 부정 프롬프트도 의도적인 것입니다. SDXL에서는 고정된 파이프라인이 프롬프트가 없는 경우에만 부정 임베딩을 0으로 설정하기 때문에 `None`와 동일하지 않습니다.

<a id="compatibility-boundary"></a>

## 호환성 경계

`StableDiffusionXLBase` 는 정준 Base 1.0 패키지 메타데이터와 필요한 아티팩트 경로가 일치한다는 것을 의미합니다. safetensors 본문이 파싱되었거나 일관된 fp16/fp32 변형이 선택되었거나 추론이 실행되었다는 것을 의미하지는 않습니다. 단일 파일 체크포인트, ONNX, OpenVINO, Flax, 보조 VAE, SDXL 리파이너, img2img, 및 임의의 파생 계약은 이 C++ 메타데이터 프로파일에 포함되지 않습니다.

Python 오라클은 더 강력한 다음 검사를 수행합니다: Diffusers 는 실제 가중치를 로드하고 구성 요소 클래스 및 중요한 텐서 구성을 확인하며, 1,024 를 1,024 MPS 픽스처 로 생성할 수 있습니다. SDXL Base 에 안전 검사기가 없고 어떤 워터마커가 있었는지 여부를 기록하며, 수행되지 않은 안전 단계를 주장하지 않습니다. 독립적인 단일 파일 모델인 VAE 와 LoRA 조합 경계는 문서화되어 있습니다.
[model-inputs.md](model-inputs.md).
