<a id="stable-diffusion-v1-reference-contract"></a>

# Stable Diffusion v1 참조 계약

<a id="canonical-reference"></a>

## 정식 참조

초기 오라클은 Hugging Face 저장소입니다.
불변 개정판의 [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5):

```text
451f4fe16113bff5a5d2269ed5ad43b0592e9a14
```

저장소는 더 이상 사용되지 않는 RunwayML 저장소의 미러로 자신을 식별하고 CreativeML OpenRAIL-M 아래에 가중치를 배포합니다. 개정 핀은 오라클을 재현 가능하게 만듭니다. 모델 라이센스를 양도하거나 확장하지 않습니다.

첫 번째로 허용되는 패키지 레이아웃은 다음과 같습니다.

```text
model-root/
├── model_index.json
├── tokenizer/
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
├── text_encoder/
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

다른 메타데이터와 선택적 구성 요소가 있을 수 있습니다. 알 수 없는 키는 패키지에 의해 보존되며 현재 검사기에서는 무시됩니다.

<a id="component-metadata"></a>

## 구성요소 메타데이터

고정된 파일은 다음을 선언합니다.

|컴포넌트|클래스|검사자가 사용하는 계약|
|---|---|---|
| Tokenizer | `CLIPTokenizer` |최대 시퀀스 길이 77|
|텍스트 인코더| `CLIPTextModel` |숨겨진 너비 768; 최대 위치 77|
|디노이저| `UNet2DConditionModel` |입력/출력 채널 4; 샘플 크기 64; 교차 주의 폭 768|
|이미지 코덱| `AutoencoderKL` |RGB 입력/출력; 잠재 채널 4; 샘플 크기 512|
|스케줄러| `PNDMScheduler` |1,000 훈련 단계; 확장된 선형 베타 일정; PRK 단계를 건너뛰었습니다.|

고정된 VAE 구성에는 `scaling_factor` 가 포함되지 않습니다. Diffusers 는 `AutoencoderKL` 를 통해 런타임 기본 `0.18215` 를 제공합니다. 이는 모델 패키지에서 단독으로 증명된 값이 아닌 참조 런타임 기본값이며, 향후 C++ VAE 구현은 이를 명시적으로 테스트해야 합니다.

<a id="tensor-flow"></a>

## 텐서 흐름

배치 크기 `B`, 높이 `H` 및 8로 나눌 수 있는 너비 `W`의 경우:

```text
positive prompt -> token ids                 [B, 77]
negative prompt -> token ids                 [B, 77]
CLIP outputs, each                           [B, 77, 768]
classifier-free guidance context             [2B, 77, 768]
initial latent                               [B, 4, H/8, W/8]
UNet latent input under guidance             [2B, 4, H/8, W/8]
UNet noise prediction                        [2B, 4, H/8, W/8]
guided latent after scheduler steps          [B, 4, H/8, W/8]
VAE-decoded image                            [B, 3, H, W]
```

정규 512 x 512 해상도에서 잠재 공간 형상은 64 x 64입니다. 위의 레이아웃 표기는 PyTorch 의 논리적 NCHW 관례를 따릅니다. 향후 백엔드는 내부적으로 텐서를 다르게 저장할 수 있지만, 구성 요소 경계는 논리적 계약을 유지해야 합니다.

<a id="fixed-reference-fixture"></a>

## 고정 참조 픽스처

```text
프롬프트:           a red cube on a white table
부정 프롬프트:      빈 문자열
시드:               42
해상도:             512 x 512
추론 단계:          20
Guidance scale:     7.5
스케줄러:           고정된 저장소의 PNDMScheduler
```

시드는 크로스 백엔드 노이즈 규정이 아닙니다. PyTorch, MLX 및 기타 런타임은 다른 무작위 알고리즘을 사용할 수 있으며, GPU 결과는 하드웨어와 릴리스에 따라 달라질 수 있습니다. 구성 요소 동등성 테스트는 결국 safetensors 또는 NPY 형식의 실제 초기 잠재를 형상, 데이터형, 레이아웃, 해시 및 수치 허용 오차와 함께 교환해야 하며, 최종 PNG 는 충분한 진단이 아닙니다.

<a id="inspector-guarantee"></a>

## 검사관 보증

C++ 검사기는 메타데이터를 검증하고 비어 있지 않은 safetensors 파일을 찾는다. safetensors 직렬화, 텐서 이름, 형태, 해시 또는 수치는 검증하지 않는다. 따라서 `StableDiffusionV1`는 메타데이터 호환성을 뜻하며 가중치가 Stable Diffusion 1.5이거나 실행 가능하다는 근거가 아니다.
