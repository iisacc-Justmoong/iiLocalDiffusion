<a id="learned-text-embeddings-textual-inversion"></a>

# 학습된 텍스트 임베딩: 텍스트 반전

Python 생성기는 `--text-embedding` 를 통해 로컬 텍스트 인버전 임베딩을 수락합니다. 이 파일들은 선택된 토크나이저와 텍스트 인코더에 학습된 토큰을 추가하며, 프롬프트는 여전히 텍스트 인코딩을 통과합니다. 이는 기존 `--embeddings` 옵션과 다릅니다. 기존 옵션은 완료된 프롬프트/풀링된 텐서를 공급하고 텍스트 인코딩을 우회합니다.

|입력|내용|사용방법|
|---|---|---|
| `--text-embedding` / `--textual-inversion` |학습된 토큰 벡터|토큰을 등록한 다음 일반 프롬프트 텍스트에서 사용|
| `--embeddings` |완성된 `prompt_embeds` 및 관련 텐서|프롬프트 텍스트 인코딩을 제공된 조건 텐서로 교체|

2 입력 모드는 하나의 요청에서 결합할 수 없습니다. `--text-embedding` 를 생략해도 [번들로 제공된 부정 기본값](generation-defaults.md)가 여전히 로드됩니다. 명시적인 `--no-default-modifiers` 는 이를 포함하지 않는 기준선을 선택합니다. 이 기능은 기존 `sd15`, `sdxl-base`, `flux1-schnell` 프리셋을 지원하며, ControlNet 와 Hires Fix 생성을 포함합니다. 텍스트 인버전 학습을 추가하지 않습니다. 네이티브 C++ 이미지 생성은 동일한 번들로 제공된 기본값을 사용합니다.

<a id="cli-and-token-selection"></a>

## CLI 및 토큰 선택

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py --model /absolute/path/image-diffusers \
  --preset sd15 \
  --text-embedding /absolute/path/to/paint-style.safetensors \
  --text-embedding-token '<paintstyle>' \
  --prompt 'a ceramic cup in <paintstyle> style'
```

`<` 또는 `>` 를 포함하는 토큰을 인용하여 쉘이 이를 텍스트로 취급하게 하십시오. 선택된 토큰은 비어있지 않고 공백이나 제어 문자를 포함하지 않아야 합니다. 해당 토큰을 인코더가 받아야 하는 프롬프트에 사용하십시오. 임베딩을 단순히 로드하는 것만으로는 해당 토큰을 프롬프트에 삽입하는 것이 아니며, 사용하지 않는 임베딩은 허용됩니다.

|인수|기본/동작|
|---|---|
| `--text-embedding FILE [FILE ...]` |비활성화됨; 하나 이상의 로컬 `.safetensors` / `.safetensor` 파일|
| `--textual-inversion FILE [FILE ...]` |`--text-embedding`의 별칭|
| `--text-embedding-token TOKEN [TOKEN ...]` |소스의 토큰 ID를 추론합니다. 명시적 값은 프롬프트에 사용되는 이름을 선택합니다.|
| `--text-embedding-encoder auto\|text_encoder\|text_encoder_2 [...]` |각 파일에 대해 `auto`; 지원되는 텐서 키와 로드된 인코더 차원을 사용하여 라우팅합니다.|

토큰 또는 인코더 목록이 제공되면 각 목록은 임베딩 파일마다 정확히 하나의 항목을 포함해야 하며, 동일한 순서를 따라야 합니다. 종속 옵션은 최소 하나의 임베딩 파일을 요구합니다. 명시적인 인코더 선택은 의도된 구성 요소를 선택하며, 호환되지 않는 벡터는 크기 조정되거나 관련 없는 인코더에 할당되는 대신 검증에 실패합니다.

토큰 재정의를 생략하면 토큰 이름이 다음 순서로 확인됩니다.

1. 파일의 safetensors 메타데이터 `token`, 그 다음 `name`입니다.
2. 단일 일반 학습 토큰 텐서에 대한 텐서 키입니다.
3. `emb_params` 또는 명명된 인코더 텐서의 파일 이름 어간입니다.

유추된 이름은 동일한 토큰 유효성 검사를 통과해야 합니다. 파일 이름이나 저장된 이름이 프롬프트 텍스트에 적합하지 않은 경우 명시적 재정의를 사용합니다.

예를 들어, 고유하게 학습된 스타일과 객체 토큰을 로드합니다.

```bash
reference/diffusers/.venv/bin/python reference/diffusers/generate.py --model /absolute/path/image-diffusers \
  --preset sd15 \
  --text-embedding /absolute/path/to/style.safetensors /absolute/path/to/object.safetensors \
  --text-embedding-token '<style>' '<object>' \
  --text-embedding-encoder text_encoder text_encoder \
  --prompt 'a photograph of <object> with <style> lighting'
```

<a id="file-layouts-and-encoder-compatibility"></a>

## 파일 레이아웃 및 인코더 호환성

로컬 safetensors 파일만 허용됩니다. 원격 저장소, 디렉토리, pickle `.pt` / `.bin` / `.ckpt` 파일, 임의 Python 객체 또는 자동 모델 다운로드가 이 인터페이스에 의해 선택되지 않았습니다. 단수형 로컬 `.safetensor` 표기는 기존에 확인된 안전 파일 경로 처리를 사용합니다.

지원되는 파일에는 단일 학습 토큰 텐서인 `emb_params` 텐서 또는 지원되는 명명된 인코더 텐서가 포함됩니다. 각 텐서는 부동 소수점 벡터 `[embedding_dimension]` 또는 다중 벡터 행렬 `[number_of_vectors, embedding_dimension]`입니다. 너비는 로드된 대상 인코더의 입력 임베딩 테이블과 일치해야 합니다.

|사전 설정|사용 가능한 대상|명명된 다중 인코더 레이아웃|
|---|---|---|
| SD 1.5 | `text_encoder`: CLIP | `text_encoder` |
|SDXL 베이스|`text_encoder`: CLIP; `text_encoder_2`: 두 번째 CLIP|`clip_l` + `clip_g` 또는 정식 구성 요소 키|
|FLUX 슈넬| `text_encoder`: CLIP; `text_encoder_2`: T5 |`clip_l` + `t5xxl` 또는 정식 구성 요소 키|

`auto` 호환되는 목적지를 형식과 텐서 너비에서 해결합니다. 모호한 목적지는 명시적인 인코더 선택이 필요합니다. 텐서 차원을 매칭하면 로드 호환성이 확립되지만, 이 특정 모델에 대해 가중치가 훈련되었거나 의도된 시각적 효과를 달성한다는 것을 의미하지는 않습니다.

명명된 다중 인코더 파일은 `auto`를 사용하여 각 항목의 대상을 유지합니다. 명시적 인코더는 이러한 파일에 대한 필터가 아닙니다. 충돌하는 명명된 항목은 거부됩니다. 예를 들어, `clip_l` + `clip_g`는 단지 인코더 옵션 변경만으로 `text_encoder`로 제한될 수 없습니다.

멀티 벡터 임베딩의 각 벡터는 토크나이저 엔트리를 받습니다. 프롬프트 처리는 학습된 토크ンを 해당 인코더에 맞는 벡터 토크넬로 확장합니다. 프롬프트의 논리적 토크인을 사용하세요; 내부 벡터 접미사를 수동으로 반복하지 마십시오. 확장 후에도 기존 인코더 컨텍스트 길이 제한이 여전히 적용됩니다.

<a id="composition-and-input-forms"></a>

## 작성 및 입력양식

학습된 토크인은 프롬프트 인코딩 및 가속기/ RAM 오프로드 배치 전에 등록됩니다. 모델/체크포인트 교체, VAE 교체, LoRA, ControlNet, 배치, CPU 프롬프트 인코딩 및 Hires Fix 는 기존 역할을 유지합니다. 학습된 토크인 로딩은 토크나이저 어휘와 인코더 입력 임베딩을 변경하며, 선택된 디노이저나 VAE 를 대체하지 않습니다.

2 차 및 부정 프롬프트 필드는 해당 인코더 경로에 학습된 토크인을 사용합니다. SDXL / FLUX 에서 생략되거나 빈 2 차 프롬프트는 확장을 위한 2 번째 인코더에 대해 해당 원래 1 차 프롬프트를 상속하며, Diffusers 의 대체 경로 동작과 일치합니다. FLUX 에 대해 부정 텍스트는 실제 CFG 가 1보다 큰 단계에서만 사용됩니다. Hires Fix 는 두 단계 모두에 대해 등록된 토크나이저와 인코더 구성 요소를 유지하며, 멀티 벡터 확장을 포함합니다. 따라서 임베딩은 단계의 프롬프트/가이드 설정에 따라 기본 패스, 리파인먼트 패스, 또는 둘 다에 참여할 수 있습니다.

CLI, JSON, `--config`, 및 Python `resolve_request()` 는 동일한 스키마를 공유합니다. JSON 경로는 해당 구성 파일에 대해 상대적이며, CLI 경로는 작업 디렉토리에 대해 상대적입니다. JSON 요청은 파일에 대한 배열과 해당 선택적 토크인/인코더 목록을 사용합니다.

```json
{
  "preset": "sdxl-base",
  "text_embedding": ["./embeddings/portrait.safetensors"],
  "text_embedding_token": ["<portrait>"],
  "text_embedding_encoder": ["auto"],
  "prompt": "a portrait of <portrait>",
  "hires_fix": true,
  "hires_scale": 1.5
}
```

부재 목록 및 최상위 JSON `null` 기능을 비활성화된 상태로 두거나 문서화된 기본값을 선택합니다. `--print-config`는 재생 가능한 요청을 노출합니다. 추론 전에 로드된 구성 요소에 대해 인코더 호환성 및 토크나이저 충돌이 확인됩니다.

<a id="validation-and-output-identity"></a>

## 검증 및 출력 ID

인코더 dtype 으로 변환하는 동안 유효하지 않은 텐서 랭크, 차원, dtype, 빈 벡터, 유한하지 않은 값, 및 오버플로가 거부됩니다. Tokenizer 충돌은 요청된 논리적 토큰과 생성된 다중 벡터 토큰 이름을 포함합니다. 충돌은 기존 어휘 항목을 덮어쓰도록 요청하는 것이 아니라 오류입니다. 로컬 파일 식별자는 로딩 중에 확인되고 provenance 에 유지됩니다.

기본 출력 스템은 `-lora` 수정자 뒤, `-controlnet` / `-hires` 앞에 `-embedding`를 추가합니다. 따라서 학습된 토큰 SD 1.5 실행은 기본적으로 `build/reference/sd15-red-cube-embedding.png`로 설정됩니다. 기존 충돌, 배치 번호 지정 및 선택적 고용 수정 기본 출력 규칙이 적용됩니다.

사이드카의 `text_embeddings` 배열은 각 소스 파일과 SHA-256 해시, 요청된 토큰/인코더, 및 해당 `registrations` 를 식별합니다. 각 등록은 해결된 논리적 `token`, 확장된 `tokens`, 인코더 `component`, `token_ids`, `source_key`, `source_dtype`, 및 벡터 `shape`, 그리고 로딩된 `dtype` 와 `vector_count` 를 기록합니다. `embedding_rows_before` / `embedding_rows_after` 값은 어휘 저장 크기를 기록하며, 사용하지 않는 패딩 행은 유지됩니다. 성공적인 CLI 로딩은 각 해결된 토큰, 인코더 및 벡터 개수를 출력하여 생성 전에 추론된 토큰 이름이 가시화됩니다. 원본 요청과 생성 입력은 재생을 위해 사용 가능합니다. 작은 실행 테스트는 로딩, 토큰 확장 및 파이프라인 구성을 확립하며, 임의의 제 3 자 임베딩 가중치의 시각적 품질이나 학습 호환성을 확립하지 않습니다.

<a id="dependencies"></a>

## 종속성

구현은 고정된 Diffusers, Transformers, PyTorch 및 safetensors 패키지를 재사용합니다. 기존 토크나이저 및 임베딩 테이블 작업은 어휘/가중치 저장을 소유하며, 프로젝트 코드는 명시적 선택, 호환성, 구성 및 provenance 를 소유합니다. 의존성, 학습된 가중치, 학습 루틴 또는 자동 다운로드기는 추가되지 않습니다. 임베딩 가중치는 런타임 패키지와 독립적으로 자체 용어를 유지합니다.

상위 공급 측는 토큰 로딩 및 다중 벡터 처리를 설명합니다.
[텍스트 반전 로더](https://huggingface.co/docs/diffusers/api/loaders/textual_inversion) 및 CLIP/T5 포함 파일
[FLUX 고급 교육 예시](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/README_flux.md).
