<a id="native-generation-io-contracts"></a>

# 네이티브 생성 I/O 계약

`Generation/GenerationIO.hpp` 은 확산, 정적분 흐름, 흐름 매칭, 자동회귀 생성, 그리고 해당 단계들의 순서 조합에 대해 C++20 호스트 데이터 경계를 제공합니다. 그것은 호출자가 제공하는 `GenerationExecutor` 를 통해 **실제 타입화된 텐서 데이터** 를 검증하고 라우팅합니다. 그것은 공유 라이브러리로 설치되며 MLX, LibTorch, Python, 또는 다운로드된 모델 없이 작동합니다.

API 는 모델, 토크나이저, 수치 샘플러, 디코더, 또는 백엔드 텐서 런타임 를 구현하지 않습니다. 기존 Python 생성은 별도의 런타임 로 남아있습니다. 해당 작업을 위해 기존 백엔드를 사용하며 입력/출력 텐서를 선언된 호스트 포트에 맵핑합니다. 이 경계에는 추가적인 제삼자 의존성이 도입되지 않습니다: 컨테이너, 확인된 차원, 그리고 실행 콜백은 C++ 표준 라이브러리를 사용합니다.

<a id="architecture-and-prediction-semantics"></a>

## 아키텍처 및 예측 의미론

각 `model` 단계는 명시적으로 하나의 `GenerationArchitecture` 를 선언해야 합니다: `diffusion`, `rectifiedFlow`, `flowMatching`, 또는 `autoregressive`. `bridge` 단계는 아키텍처 선언이 없습니다. 이 선언들은 호출자의 모델 계약을 설명하며, API 는 스케줄러, 클래스 이름, 파일 확장자, 또는 텐서 형식에서 학습 목표를 추론하지 않습니다. 아키텍처를 선언하는 것이 백엔드가 그것을 구현한다는 증거는 아닙니다. 완전한 하이브리드 모델은 대신 명시적으로 정렬된 구성 단계로 표현될 수 있습니다.

포트는 `GenerationSemantic`를 사용하여 페이로드의 의미를 유지합니다.

|의미 체계|페이로드 의미|
| --- | --- |
| `sample` |선언된 표현 공간의 연속 샘플/상태|
| `epsilon` |소음 예측|
| `vPrediction` |확산 v-매개변수화 예측|
| `velocity` |흐름 벡터 필드 예측|
| `logits` |백엔드의 선언된 어휘 공간에 대한 부동 점수|
| `tokenIds` |선언된 토크나이저 공간의 음수가 아닌 정수 식별자|
| `embeddings` |선언된 인코더 공간에 있는 부동 벡터|
| `pixels` |선언된 레이아웃 및 값 공간 ID가 있는 픽셀 텐서|
| `timestep` |백엔드별 시간 또는 시간 단계 값|
| `attentionMask` |백엔드별 마스크 값|

`vPrediction`와 `velocity`는 별개로 유지됩니다. 예측은 결코 아무런 알림 없이를 샘플로 취급하지 않으며 로짓은 아무런 알림 없이를 토큰으로 변환하지 않습니다. 백엔드는 정규화, 토큰 어휘 범위, 마스크 및 예약을 소유합니다.

<a id="tensor-contract"></a>

## 텐서 계약

각 `GenerationTensorSpec` 는 정확한 dtype, 랭크/형식, 레이아웃, 의미론, 그리고 비어 있지 않은 `representationSpace` 를 요구합니다. 마지막 필드는 잠재 스케일링/이동 또는 토큰 어휘수 개정과 같은 세부 사항이 포함된 실제 VAE /토크나이저/인코더 좌표계를 식별해야 합니다. 예를 들어, 서로 다른 VAE 에서의 동일한 `[1,4,64,64]` 잠재 형식은 직접적인 와이어링을 허용하지 않습니다. 애플리케이션은 신뢰할 수 있는 공간 식별자에 대해 책임을 져야 하며, 라이브러리는 텐서 바이트에서 모델 신원을 확인할 수 없습니다.

정확한 대문자 감별 값 `auto`, `unspecified`, 및 `unknown` 는 예약된 센티넬이며 표현 신원으로서 거부됩니다. 일치하는 알 수 없는 공간은 호환성을 확립할 수 없으므로, 포트 연결이나 계획 실행 전에 실제 공유 좌표계 신원을 공급해야 합니다.

`-1` 는 포트 계약에서 독립적인 와일드카드 차원입니다. 구체적인 페이로드 는 모든 와일드카드를 음이 아닌 크기로 대체해야 합니다. 0 차원은 빈 토크인 접두사와 빈 배치 를 허용하며, 빈 형식은 하나의 스칼라를 의미합니다. 레이아웃은 `NCHW`, `BS`, `BSC`, 또는 `scalar` 와 같은 불투명하고 정확한 식별자입니다; 라이브러리는 축 의미를 전치하거나 추론하지 않습니다. 와일드카드 는 별도의 축 또는 포트 간에 기호적 등식을 생성하지 않습니다. 백엔드는 일치하는 토크인과 마스크 시퀀스 길이와 같은 추가 관계를 유효성 검사해야 합니다.

지원되는 dtype 및 저장소 대안은 다음과 같습니다.

|D타입|`GenerationTensorData` 스토리지|
| --- | --- |
| `float16`, `bfloat16` |네이티브 16비트 비트 패턴을 포함하는 `std::vector<std::uint16_t>`|
| `float32` | `std::vector<float>` |
| `float64` | `std::vector<double>` |
| `int32` | `std::vector<std::int32_t>` |
| `int64` | `std::vector<std::int64_t>` |
| `uint8` | `std::vector<std::uint8_t>` |

유효성 검사기는 요소 개수, 요소 개수와 바이트 크기 오버플로우, 저장 유형, 구체적 형식, 및 유한 부동 값을 포함하여 half-precision NaN 와 무한 비트 패턴을 확인합니다. 토큰 식별자는 `int32` 또는 `int64` 저장 공간을 필요로 하며 음수가 될 수 없습니다. 연속 상태, 예측, 로짓, 임베딩은 부동소수점 데이터 타입을 필요로 합니다. 다른 데이터 타입은 이 호스트 경계로 들어오기 전에 명시적인 백엔드 변환이 필요합니다. 암시적 캐스팅, 리쉐이핑, 스케일링, 또는 정밀도 변환은 발생하지 않습니다.

<a id="named-composition-and-execution"></a>

## 명명된 구성 및 실행

`GenerationPlan.inputs` 는 모든 외부 입력을 선언합니다. 각 단계 입력은 정확히 하나의 `GenerationBinding` 를 가지며, 빈 소스 단계는 외부 입력을 참조하고, 다른 소스 단계는 계획에서 더 이전에 나타납니다. 중복 이름, 누락된 입력, 누락된 소스 포트, 순방향 참조, 그리고 사이클은 어떤 실행자 호출 전에 실패합니다. 소스는 후속 소비자로 분산될 수 있습니다. 이름이 지정된 계획 출력은 최종 또는 중간 텐서를 노출할 수 있습니다.

직접 바인딩은 호환 가능한 형상, 데이터 타입, 레이아웃, 의미, 및 표현 공간이 필요합니다. 토큰을 흐름 모델에 연결하려면, 자기회귀 단계를 선언하고, 토큰 입력과 잠재 표현/임베딩 출력을 가진 브릿지를 선언한 다음, 흐름 단계를 선언합니다. 브릿지 실행자는 실제 디코더/인코더나 다른 변환을 실행하여 약속된 텐서를 반환해야 합니다. 브릿지를 단순히 라벨링하는 것만으로는 입력 또는 출력의 검증을 비활성화하지 않습니다. 모델 단계 계약은 자체 입력과 출력 사이에서 자연스럽게 다를 수 있습니다. 예를 들어, 노이즈 제거기의 샘플 입력과 에파실론 출력과 같은 경우입니다.

`executeGenerationPlan` 는 전체 정적 계획과 외부 페이로드를 검증한 다음, 실행 전에 각 단계의 구체적 입력을 검증하고, 전달하기 전에 모든 반환된 출력을 검증합니다. 와일드카드 소스는 고정된 소비자와 정적으로 호환될 수 있으며, 실제 생성된 형상은 런타임 에서 그 소비자를 만족해야 합니다. 추가 콜백 출력은 누락된 출력과 마찬가지로 거부됩니다. 콜백 예외는 계획을 중지하고 스테이지 식별자로 보고되며, 이전에 실행된 외부 백엔드 사이드 이펙트는 되돌려지지 않습니다.

호스트 API 는 이름 지정된 입력을 라우팅하고 결과를 내보낼 때 벡터를 소유하고 페이로드를 복사하며, 이는 포터블한 정확도 경계이며 0-복사 장치 그래프나 백엔드 네이티브 텐서 저장의 대체물이 아닙니다.

```cpp
#include <Generation/GenerationIO.hpp>

const iild::GenerationTensorSpec state{
    iild::TensorDType::float32, {1, 2}, "BC",
    iild::GenerationSemantic::sample, "vae:example-v1"};
auto prediction = state;
prediction.semantic = iild::GenerationSemantic::epsilon;

const iild::GenerationPlan plan{
    {{"state", state}},
    {{"denoiser", iild::GenerationStageRole::model,
      iild::GenerationArchitecture::diffusion,
      {{"sample", state}}, {{"prediction", prediction}},
      {{"sample", {"", "state"}}}}},
    {{"prediction", {"denoiser", "prediction"}, prediction}}};

// 실제 추론 백엔드를 사용하여 구현한다. 콜백은 다음을 반환해야 한다:
// 해당 단계가 선언한 출력 이름과 구체적인 텐서 명세가 정확히 일치하는 맵.
iild::GenerationExecutor backend = myBackendExecutor;
const auto result = iild::executeGenerationPlan(
    plan, {{"state", {state, std::vector<float>{0.1F, 0.2F}}}}, backend);
```

`GenerationIOError` 는 `invalidContract` , `incompatibleBinding` , `invalidTensor` , 및 `executionFailed` 오류 코드를 노출하며, `GenerationIOTests` 는 실제 토큰-잠재 표현 브릿지 이후 플로우 매칭, 동적 및 빈 입력, 예측 구분, 데이터 타입/저장 오류, 유한 값 확인, 오버플로우, 무효 그래프, 및 실행기 실패를 포함한 모든 4 아키텍처 선언을 수행하며, `InstallConsumerTests` 는 재설치된 설치 접두어에서 공개 API 를 별도로 컴파일, 링크, 및 실행합니다.
