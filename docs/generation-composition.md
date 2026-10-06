<a id="generation-architecture-io-and-composition"></a>

# 생성 아키텍처의 입출력과 구성

iiLocalDiffusion 는 3 상호 보완 인터페이스를 제공합니다.

* [Native C++ 호스트 텐서 계약](generation-io-native.md)는 호출자가 제공한 백엔드 콜백을 통해 명명된 스테이지를 검증하고 실행합니다.
* `reference/diffusers/generation_composition.py`는 이미 로드된 Python 런타임을 타이핑된 호스트 텐서 및 텍스트 포트를 사용하여 연결합니다. `generation_adapters.py`는 Diffusers 파이프라인을 호출하고, Transformers `generate()`를 호출하며, 토큰화 디코딩을 합니다.
* [일반 Diffusers 파일 I/O](generic-diffusers.md#diffusion-flow-autoregressive-and-hybrid-interchange)는 이미지, 비디오, 오디오 및 텍스트와 함께 safetensors 디스크립터를 저장/재로드합니다.

확산, 교정된 흐름, 흐름 매칭, 그리고 자기회귀 생성은 명시적인 단계 선언입니다. 여러 가지 종류는 하이브리드 파이프라인을 형성합니다. 내부 혼합 생성을 가진 모델은 또한 `hybrid` 를 Python, API 또는 일반적인 CLI 에서 선언할 수 있으며, C++ 는 해당 구성 단계를 나타냅니다. 아키텍처 선언은 샘플러를 선택하거나, 훈련을 검사하거나, 가중치를 변환하거나, 모델을 인증하지 않습니다. 교정된 흐름과 흐름 매칭은 서로 다른 훈련 목표를 사용하면서 흐름 스케줄러를 공유할 수 있습니다. 확산 `v_prediction` 와 흐름 `velocity` 는 서로 다른 포트 의미론이며, 명시적인 변환 없이 함께 연결될 수 없습니다.

기존 모델 로딩, 장치 선택, 및 기본값 설정은 변경되지 않았습니다. 이 API 는 패키지 의존성을 추가하지 않습니다. 유지된, 이미 고정된 Diffusers 와 Transformers 는 추론과 생성을 공급하며, safetensors 는 타입화된 저장소를 공급하고, NumPy / Torch 는 검증과 호스트 복사를 공급합니다. 그들은 이 프로젝트에서 모델 수학, 토크나이제이션, 또는 안전하지 않은 텐서 역직렬화 구현을 피합니다. 패키지 버전, 라이선스 및 배포 비용은 [의존성 결정](dependencies.md)에 남아있습니다.

<a id="python-contract"></a>

## Python 계약

`Port.tensor(dtype, shape, layout, semantic, representation_space)` 는 하나의 값을 설명합니다. `None` 는 독립적인 와일드카드 차원이며, 모든 구체적인 Python 차원은 양수여야 합니다. `Port.text()` 는 하나의 문자열을 허용하며, `Port.text(batch=True)` 는 비어 있지 않은 문자열 배치입니다. 네이티브 API 는 추가로 0-크기의 텐서를 허용하며, `-1` 와일드카드를 사용하며, Python 객체나 JSON ABI 의존성이 없습니다.

텐서 페이로드 는 반드시 일반 NumPy 배열 또는 CPU Torch 텐서 이어야 합니다. 데이터 타입은 정확히 확인되며, BF16 와 정수 토큰 ID 를 포함합니다. 토큰 ID, 잠재 표현 정규화, 전치 또는 리쉐입에 대한 자동 부동소수점 변환이 발생하지 않습니다. 포트는 데이터 타입, 랭크, 레이아웃, 의미론 및 표현 공간이 일치해야 합니다. 동적 프로듀서 차원은 구체적 소비자 형상과 다시 확인됩니다. 부동소수점 값은 유한하고 토큰 ID 는 음수가 아니어야 합니다. 어텐션 마스크 길이와 같은 어텐션 마스크 길이와 같은 교차 입력 관계와 같은 어휘 경계와 같은 모델 확인 사항으로 남아 있습니다.

실제 토크나이저/인코더/ VAE 버전 및 정규화 관례를 포함하는 공간 정체성을 사용해야 합니다. `auto`, `unspecified` 및 `unknown` 는 예약되어 있으며 구성 호환성을 설정할 수 없습니다. 라이브러리는 텐서 형식이나 호출자 제공 레이블에서 유효한 좌표계 정체성을 유도할 수 없습니다.

`Stage` 는 이름, 아키텍처, 입력/출력 포트 맵, 바인딩 및 실행기를 가집니다. `Source.input(name)` 는 외부 입력을 바인딩하며, `Source(stage, port)` 는 이전 단계의 이름 지정된 출력을 바인딩합니다. 각 입력은 정확히 하나의 바인딩이 필요합니다. 누락, 초과, 중복 또는 순방향 연결은 실행 전에 실패합니다. 맵을 내보낼 때 여러 개의 최종 또는 중간 결과를 유지할 수 있습니다. 콜백은 반드시 명시된 출력 맵을 정확히 반환해야 합니다. `architecture="adapter"` 은 명시적인 디코더/인코더/변환 브릿지를 식별하며 결과의 아키텍처 목록에서 제외됩니다. 브릿지 출력은 모델 출력과 동일한 값 검증을 받습니다.

실행자 경계는 외부 입력, 콜백 입력, 콜백 출력 및 내보낸 결과의 스냅샷을 소유합니다. 콜백의 원위치 작성은 다른 단계의 유지된 값이나 호출자의 원래 입력을 수정할 수 없습니다. 이는 호스트 메모리/복사를 소모하며 0-복사 장치 그래프가 아닙니다. 실행은 순차적이며 실패는 후속 단계를 중단시키고 백엔드 측 부작용을 되돌리지 않습니다.

<a id="autoregressive-to-flow-through-an-explicit-text-bridge"></a>

## 명시적 텍스트 브리지를 통과하는 자동 회귀

다음 통합 단편은 `language_model`, `tokenizer`, `flow_pipeline` 가 이미 로드되고 검증되어 애플리케이션에 배치되었다고 가정합니다. 호출자는 실제 토크나이저 공간 정체와 선택된 모델에 맞는 올바른 흐름 출력 형식/데이터 타입을 대치해야 합니다. 어댑터는 가중치를 다운로드하지 않습니다. 언어 모델은 `eval()` 모드가 되어야 합니다.

```python
from generation_composition import Port, Source, Stage, GenerationPipeline
from generation_adapters import TransformersAdapter, TokenDecodeAdapter, DiffusersAdapter

tokens = Port.tensor("int64", (1, None), "BS", "token_ids", "my-tokenizer:revision")
texts = Port.text(batch=True)
pixels = Port.tensor("float32", (1, None, None, 3), "BHWC", "pixels", "rgb:0-1")

stages = [
    Stage("language", "autoregressive", {"input_ids": tokens}, {"tokens": tokens},
          {"input_ids": Source.input("tokens")},
          TransformersAdapter(language_model, {"tokens": "sequences"},
                              {"max_new_tokens": 32, "do_sample": False}, device="cpu")),
    Stage("decode", "adapter", {"tokens": tokens}, {"text": texts},
          {"tokens": Source("language", "tokens")}, TokenDecodeAdapter(tokenizer)),
    Stage("image", "rectified-flow", {"prompt": texts}, {"pixels": pixels},
          {"prompt": Source("decode", "text")},
          DiffusersAdapter(flow_pipeline, {"pixels": "images"},
                           {"output_type": "np", "num_inference_steps": 4}, device="mps")),
]
pipeline = GenerationPipeline({"tokens": tokens}, stages,
    {"tokens": Source("language", "tokens"), "pixels": Source("image", "pixels")})
result = pipeline.run({"tokens": input_ids_on_cpu})
assert result.hybrid
```

동일한 API는 확산→흐름, 흐름→자동회귀 또는 인터리브 블록을 연결합니다. 토큰, 임베딩 또는 잠재 공간이 다른 실제 모델별 브리지를 제공합니다. ID 콜백은 호환되지 않는 표현을 동일하게 만들 수 없습니다.

`DiffusersAdapter` 는 명시적인 파이프라인 호출 매개변수와 구조화된 결과를 요구합니다. `output_type="np"` 를 사용하거나 파이프라인의 문서화된 잠재 표현 출력 모드와 일치하는 출력 포트를 사용하여 타입화된 픽셀을 사용하세요. 그의 `outputs` 맵은 정확한 결과 필드를 선택합니다. `TransformersAdapter` 는 정수 ID 를 캐스팅하지 않고 `sequences` 를 내보내며 `logits` 단계의 원본을 내보낼 수 있고 `[B,V]` 단계를 `[B,S,V]` 로 스택합니다. 로그 확률은 샘플링된 토큰 ID 가 아닙니다. 빔 검색 시퀀스는 지원되지만 `num_beams != 1` 를 가진 원본 로그 확률은 빔 조상 관계가 별도의 어댑터를 필요로 하므로 거부됩니다. 모델별 KV 캐시 및 처리된 점수는 이동 가능한 출력 포트가 아닙니다. 기타 다중 모달 필드에는 명시적인 애플리케이션 어댑터 및 계약이 필요합니다.

어댑터 `device` 는 입력 배치만 제어합니다. 로드된 모델을 이동하거나 CPU 에서 재시도하지 않습니다. RNG, 스케줄, 생성 길이 및 샘플링 기본값은 호출자의 런타임 설정에 속하며, 어댑터는 아무런 알림 없이 대체하지 않습니다.

<a id="reusing-saved-tensors"></a>

## 저장된 텐서 재사용

일반 출력의 `generation.json.outputs[].tensor_input` 는 경로, 정확한 키, SHA-256, 데이터형, 형상, 의미, 레이아웃 및 표현 공간 라벨을 포함합니다. `Port.from_tensor_descriptor(descriptor)` 는 해당 구체적 파일 설명자를 컴포지션 포트에 매핑합니다. `generic_io.load_tensor_input(descriptor, generate_any.file_identity)` 로 실제 데이터를 별도로 로드하고 인증한 후 `GenerationPipeline.run()` 에 전달합니다. 로더는 내보낸 파일 메타데이터를 설명자와 비교하므로 VAE 공간의 재라벨링으로 변환할 수 없습니다. `unspecified` 공간의 자동 토큰 내보내기에는 `--tensor-outputs` 를 사용하여 내보내기 전에 실제 토크나이저 정체성을 선언해야 합니다.

일반 `latents`, `embeddings`, `token-ids`, `v-prediction`, `continuous-state` 및 `attention-mask` 는 Python, `latent`, `embedding`, `token_ids`, `v_prediction`, `sample` 및 `mask` 에 매핑됩니다. 일반 `tensor` 의미는 `tensor` 로 유지됩니다. C++ 는 문서화된 열거형과 소유한 벡터를 사용하며, 이 파일 설명자를 파싱하거나 백엔드 텐서를 노출하지 않습니다. 애플리케이션 어댑터가 그 경계를 수행합니다.

<a id="verification-and-upstream-contracts"></a>

## 검증 및 상위 공급 측 계약

CTest를 통해 경량 계약을 실행하세요. 실제 오프라인 런타임 확인의 경우:

```sh
PYTHONPYCACHEPREFIX=build/pycache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  reference/diffusers/.venv/bin/python tests/GenerationAdapterTests.py
PYTHONPYCACHEPREFIX=build/pycache \
  reference/diffusers/.venv/bin/python tests/GenericIOTests.py
```

테스트는 로컬 설정에서 작은 GPT-2 및 DDPM 를 구성하고 설치된 흐름 스케줄러를 실행하며 명시적인 브릿지를 확인하고 유효하지 않은 출력에서 중지합니다. 변이 격리, 동적 형식 불일치, 원본-logit/빔 경계, BF16, 그리고 큰 정수 safetensors 왕복도 포함합니다. 이는 I/O 와 작은 런타임 실행을 확립하며, 큰 모델 품질이나 범용 체크포인트 지원을 확립하지 않습니다. 일반 가이드에 기록된 작은 FLUX 잠재 표현 왕복 변환 도 참조하세요.

공식 참조: [Transformers 생성 출력 형식](https://huggingface.co/docs/transformers/internal/generation_utils),
[FLUX 파이프라인 입력/출력 옵션](https://huggingface.co/docs/diffusers/api/pipelines/flux)및 [FlowMatchEuler 스케줄러 계약](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete).
