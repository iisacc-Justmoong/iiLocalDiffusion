<a id="local-model-merging"></a>

# 로컬 모델 병합

<a id="preflight-and-saved-output-verification"></a>

## 사전 검사 및 저장된 출력 검증

검사는 `base_profile`, 리소스별 `profile`, 및 `preflight` 를 반환합니다. 프로파일이 헤더 증거에서 생태계를 식별하고 별도로 UNet, DiT, VAE, 텍스트 인코더 및 분류되지 않은 텐서 (개수, 모양, 데이터 형식 및 한계가 설정된 예시) 를 나열합니다. 분류되지 않은 텐서는 아무런 알림 없이 소음 제거기로 불리지 않습니다. 리소스는 `compatible`, `conditional` 또는 `incompatible` 상태를 가집니다: 생태계 일치만 정확한 구조적 일치가 아닙니다. `lora_targets` 는 모든 승인된 투영에 대한 기본 텐스와 행 범위를 나열하며; 거부된 어댑터는 실제 대상 또는 구성 실패를 제외 이유로 유지합니다.

사전 검사 는 유지/제외된 리소스, 유효한 계수, 투영된 텐서 개수 및 수치/품질 위험을 설명합니다. 가중 산술은 기본 텐서 계약을 유지하며, `common-layer` 하에 자격 있는 좌표를 적합시킵니다. Unified 는 독립적인 순차적 이미지 개선 패키지이며, **가 아닌** 단일 네트워크 변환입니다. 적합 또는 성공적인 검증을 통해 생성된 이미지가 인증되지 않습니다.

가중 출력을 게시하기 전에 저장된 모든 텐서는 다시 열려 계산된 값, 모양 및 데이터 형식과 정확히 비교됩니다. 부동 출력은 유한해야 하며, 독립적으로 관찰된 변경된 텐서 개수는 계산과 일치해야 합니다. `output_verification` 는 샘플이 아닌 전체 범위를 보고합니다. 유니파이드 출력이 재개되고 모든 패키징된 체크포인트 바이트가 계획된 멤버에 대해 해시되며; LoRA -퓨즈된 멤버도 텐서 검증을 받습니다. 복사된 유니파이드 멤버는 바이트가 보존되며 수치적으로 수리되지 않습니다. 검증은 전체 출력 읽기 I/O 를 추가하며 소스 모델을 수정하지 않습니다.

최선의 노력 산술은 빈 기본 텐서를 보존하고 빈/무효 양자화 스케일을 가진 기여를 생략합니다. 동일한 FP8 LoRA 별칭은 FP32 를 통해 비교되어 지원되지 않는 저장 데이터 타입 비교 연산자를 피합니다. 완전히 사용 불가능한 재료 집합은 재료 가중치가 혼합되었다는 주장을 대신하여 명확하게 보고된 보존되거나 수리된 기본을 제공합니다. 읽을 수 없는 기본 구조, 불충분한 저장, 프로세스 종료 및 실패한 출력 무결성 검사도 여전히 출판을 중단하며; 그들은 정직하게 성공적인 병합으로 표시될 수 없습니다.

이종 SD 1.5, SDXL, DiT 및 편집 체크포인트는 먼저 `iild-convert`와 하나의 Tsubaki DiT 텐서 계약으로 정규화될 수 있습니다. 참조
[DiT -표준 모델 변환](model-conversion.md). 동일한 대상 템플릿을 사용하는 변환된 출력은 일반적인 호환 DiT 체크포인트이며 아래 가중치 모드를 사용할 수 있습니다. 이는 독립적인 아키텍처를 순차적인 이미지 공간 연쇄로 보존하는 `unified` 와 구별됩니다.

<a id="forced-base-layout-fitting"></a>

## 강제 베이스 레이아웃 피팅

`--checkpoint-policy base-layout` 는 기본 텐서 이름, 형식 및 데이터 타입을 유일한 가중치 출력 계약으로 만듭니다. Society 는 가중치 합과 차이를 위해 이 정책을 선택합니다. 읽을 수 있는 재료는 생태계 또는 좌표 불일치로 인해 제외되지 않습니다. SDK CLI 기본값은 `common-layer` 로 유지되며; 유니파이드는 기존 선택 규칙을 가진 별도의 모델 연쇄로 유지됩니다.

체크포인트 매핑은 정확한 이름, 정규화된 이름, 역할/깊이 일치도를 시도한 다음 결정론적으로 구성 역할 간에도 가장 가까운 사용 가능한 소스를 선택합니다. 좌표는 전치, 리쉐이프 또는 가장 가까운 리샘플링되어 기본에 맞춥니다. 배운 소스 좌표가 없을 때, 배운 기본 텐서는 0 재료 텐서를 받습니다. 합계는 요청된 계수를 곱한 0 를 포함하며, 차이는 0를 뺍니다. 기본 일정, 빈 텐서, 정수 상태 및 양자화 보조 요소는 기본 정의에 유지됩니다. 수치 복구 및 저장된 출력 검증이 적용됩니다.

정책은 합성 LoRA 목표 선택 및 자르기/0패딩 맞춤을 가능하게 합니다. 혼합 체크포인트/어댑터 또는 지원되지 않는 어댑터가 표준 LoRA 로 해석될 수 없으면, 읽을 수 있는 텐서들은 체크포인트 좌표로 투영되며, `resource_compatibility[].raw_tensor_projection` 가 이를 명시적으로 기록합니다. 이러한 재료들은 체크포인트 계수를 소모하며, LoRA 델타 계수가 아닌 것입니다. 일반적인 LoRAs 는 델타 의미를 유지합니다. 포함된 모든 체크포인트 가중치는 기존 합계 정규화에 참여하며, 이는 기본 계수 0를 만들 수 있습니다.

보고서는 수락한 강제 재료를 `conditional`로 표시하고 생태계 증거를 유지하며 `base-layout-force-fit-zero-fill-v1`, `forced_source_tensors`, `normalized_name_tensors`, `shape_changed_tensors`, `zero_filled_tensors`, `zero_fill_targets`를 기록한다. 이 결정적 좌표 맞춤은 학습되지 않았으며 의미적 호환성이나 이미지 품질을 입증하지 않는다. 손상되거나 누락된 파일은 계속 명시적으로 제외한다. 읽을 수 없는 기준 모델과 저장 출력 검증 실패는 여전히 실패한다. `ModelMergeBaseLayoutTests`는 계열 간 포함·독립 연산·LoRA 크롭/패딩·혼합 어댑터·누락 좌표·결정성·원본 보존·변경되지 않은 공통 레이어 경계를 검사한다.

<a id="base-scoped-compatibility-selection-common-layer-strict-and-unified"></a>

## 기본 범위 호환성 선택(공통 계층, 엄격 및 통합)

명시적인 `base-layout` 가중 정책을 제외하면 병합은 선택한 기본 모델을 호환성 경계로 삼는다. 계수를 해석하거나 출력을 작성하기 전에 SDK가 모델 메타데이터와 텐서 구조에서 생태계 근거를 식별한다. 체크포인트는 기본 생태계를 공유하거나(예: FLUX.2와 Krea 2또는 SDXL와 Illustrious), 분류되지 않은 체크포인트가 기본 모델과 정확히 같은 텐서 배치를 가질 때 유지한다. LoRAs는 엄격한 대상 및 형태 규칙에 따라 전체 대상 집합을 기본 체크포인트에 대해 해석할 수 있을 때만 유지한다.

호환되지 않는 재료는 부분적으로 유효한 요청을 중단하는 대신 개별적으로 제외됩니다. 20 재료로 구성된 요청은 따라서 11 를 병합하고 9 를 `excluded_sources` 에 보고하며, 감지된 생태계와 이유를 포함합니다. 명시적인 가중치는 해당 재료에 따르며, 제외된 계수는 제거되고 유지된 체크포인트에서 가중 합산 기본 계수가 다시 계산됩니다. 기본 공통 레이어 산술에서 사용 불가능한 추가 파일도 제외됩니다. 무엇이든 기여할 수 없는 경우, 기본 전용 출력이 생성되어 `result_kind: base-fallback`, `no_effect: true` 로 명시적으로 보고되며, 유효한 블렌드로는 보고되지 않습니다. 엄격한 산술과 통합 패키징은 여전히 포함된 재료가 필요합니다. 검사 및 완료된 보고서 모두 `resource_compatibility`, `included_material_count`, `excluded_material_count` 를 노출합니다.

<a id="single-safetensors-output"></a>

## 단일 세이프텐서 출력

단일 파일 체크포인트인 경우 기본 `weighted-sum` 및 명시적 `weighted-difference` 모드는 새 Safetensors 파일을 작성합니다. 입력은 원래 경로에 유지되며 기존 출력은 절대 대체되지 않습니다. Society 는 또한 통합 `.iildmodel` 캐스케이드 형식을 지원하며 사용자 입력 출력 모델 이름을 요구합니다.

동일한 Anima 체크포인트 백본 네임스페이스 ( `model.diffusion_model.*` ,  `diffusion_model.*` ,  `net.*` , 선택적 중첩 `net.*` , 및 언래핑 이름) 은 메모리에서 정렬됩니다. 두 입력 모두 Anima LLM -어댑터 서명을 포함해야 하며, 접두어 정렬 후 동일한 명확한 완전한 텐서 목록을 가져야 합니다. 형상, 데이터 타입, 아키텍처 메타데이터 및 예측 마커는 여전히 일치해야 합니다. 임베딩된 인코더/ VAE 이름은 정확히 유지되며, 누락되거나 초과된 구성 요소는 오류입니다. 출력은 모든 기본 텐서 이름과 데이터 타입을 유지하며, 소스 해시와 키 정책은 Safetensors 병합 메타데이터에 기록됩니다. 이것은 다른 아키텍처를 변환하거나 소스 파일을 이동, 이름 변경, 재작성 또는 삭제하지 않습니다.

엄격한 정책 회귀 적용 범위에는 LoRA를 사용한 산술 모드, 네임스페이스 변형, 기본 이름 및 소스 바이트 보존, 누락/추가/중복/모양 거부가 모두 포함됩니다.

<a id="cross-architecture-common-layer"></a>

### 아키텍처 간 공통 계층

엄격한 가중치 병합은 동일한 텐서 목록, 형상, 예측 마커, 런타임 자산 및 아키텍처 메타데이터를 요구합니다. Flux2 및 Krea2 는 그 계약을 만족하지 않습니다: Krea2 Turbo INT8 는 `blocks.*.attn.wq/wk/wv/wo` 텐서와 `weight_scale` 를 사용하며, Flux 체크포인트는 다른 트랜스포머 네임스페이스를 사용하며 종종 다른 차원을 사용합니다. 이 불일치가 산술 연산이 아닌 출력이 생성되기 전에 거부된 이유였습니다.

기본 스코프 생태계 선택 후, `--checkpoint-policy common-layer` (기본값) 은 해당 생태계*내의 텐서 레이아웃 계약 *의 적용을 의도적으로 완화합니다. 기본 체크포인트는 출력 형식 및 실행 가능한 텐서 목록으로 유지됩니다. 각 학습된 기본 텐서에 대해, SDK 는 정확한 주소를 사용하고, 고유한 정규화된 이름을 사용하며, 같은 구성 요소/같은 역할/같은 종류 소스를 사용합니다. 알려진 래퍼 접두사 ( `module`, `_orig_mod`, `model.diffusion_model`, `diffusion_model`, `model` ) 및 언더스코어 구분자는 정규화됩니다. 블록 깊이는 블록 번호 대신 역할 내의 상대적 위치를 사용합니다. 패키지 단계, 텍스트 인코더, VAEs, 디노이저, Q/K/V 역할 및 편향/가중치 종류는 별도의 매칭 풀을 가집니다.

정책 `base-layout-normalize-project-flatten-v3` 는 좌표를 다음 순서로 적합합니다: 변경되지 않은 형상; 뒤집힌 2-D 형상 전치; 동일한 요소 개수의 리쉐이프; 랭크 변경 평탄화/재샘플링; 아니면 정규화된 좌표에 대한 축별 최근접 샘플링. 스칼라 목표는 첫 번째 소스 좌표를 선택하며, 스칼라 소스는 확장됩니다. 축을 줄이는 것이 축을 늘리는 것보다 우선하며, 평탄화된 인덱스는 최대 1,048,576 개의 좌표로 구성된 덩어리로 구성됩니다. 빈 텐서와 누락된 역할은 발명된 값을 패딩하는 대신 기본값을 보존합니다.

INT8/UINT8 및 인접한 `weight_scale` 또는 모듈 `scale_weight`가 있는 FP8는 산술 연산 전에 역양자화한다. 선택적 `weight_zero_point` 또는 `weight.zero_point`를 먼저 뺀다. 스케일은 스칼라, 출력 행별, 브로드캐스트 가능 형태, 축별로 나누어떨어지는 블록 또는 나누어떨어지는 연속 평탄 블록일 수 있다. 선행 차원과 일치하는 1차원 스케일은 정방 행렬을 포함하여 출력 행 스케일을 뜻한다. 양자화된 기본 모델은 디코딩하여 병합한 뒤 자체 매개변수로 인코딩한다. 정수 저장 값은 반올림하며 허용 범위에서 포화시킨다. FP8 색인과 유한성 검사는 지원되는 FP32 산술을 사용하고 FP64 소스는 정밀도를 유지한다. 기본 모델의 scale/0-point 텐서는 평균하지 않는다.

Scheduler 좌표 (예: `denoiser.sigmas`), 예측 마커, 실행 통계, 일반 정수/부울 버퍼 및 매핑되지 않은 텐서는 소스에 해당 항목이 포함되어 있더라도 기본값을 유지합니다. 합산에서 누락된 재료의 비율은 해당 텐서**의 기본값 **으로 되돌아갑니다; 차감에서는 0를 기여합니다. 이전에는 누락된 텐서의 기본 대체값이 기본값의 일부에서 잘못 뺐습니다. 0-기여 텐서는 병합된 것으로 간주되는 대신 보존됩니다. 외부 전용 텐서는 생략되며; 출력 이름, 형식 및 저장 데이터 타입은 항상 기본값을 따릅니다.

이 정책은 결정적이며 FLUX.2 및 Krea 2처럼 생태계가 호환되는 구현 사이에서 사용자가 요청한 실험을 완료하도록 설계되었다. 아키텍처 변환, 증류 또는 의미적 레이어 동등성의 근거가 아니다. 이미지 품질은 낮거나 무의미할 수 있다. 보고서와 내장 병합 메타데이터는 `checkpoint_policy: common-layer`, `semantic_equivalence: false`를 설정하고 정확/투영/보존 개수, `transform_counts`, 한계가 설정된을 적용한 예시, 역양자화 개수 및 런타임 보존 이유를 기록한다. 잘못된 양자화기(빈 스케일, 유한하지 않거나 양수가 아닌 스케일, 나누어떨어지지 않는 블록)는 영향을 받는 기본 텐서만 보존하고 이유를 첨부한다. `numeric_normalization`는 기본 양자화기 실패와 재양자화된 텐서를 기록한다. 헤더 검사는 스케일 값을 로드하지 않으므로 런타임 유효성을 보장할 수 없다. 배치 불일치가 하나라도 작업을 중단해야 한다면 `--checkpoint-policy strict`를 사용한다. 공통 레이어 체크포인트 재료에서 NaN와 양/음의 무한대는 전체 병합을 중단할 이유가 아니라 누락된 **좌표**로 취급한다. 역양자화와 형태 투영 후 잘못된 좌표는 가중 합에서 기본값을 사용하고 가중 차에서 0을 사용한다. 유한한 좌표와 다른 모든 재료는 여전히 정상적으로 기여한다. 재료 텐서 전체가 잘못되었고 다른 기여도 적용되지 않으면 기본 텐서를 정확히 복사하며 병합된 것으로 세지 않는다. 원본 파일은 제자리에서 정리하지 않는다.

보고서 및 내장 메타데이터는 `nonfinite_material_policy` 를 선포하고 `numeric_normalization.nonfinite_material_values`, `nonfinite_material_tensors`, 최대 64 개의 `nonfinite_material_examples` 를 기록합니다. 각 예시는 레시피의 소스 인덱스, 소스/대상 텐서, NaN /+Inf/-Inf 개수, 대체 규칙, 및 **투영된 기본** 형상의 최대 8 개의 좌표를 식별합니다. 따라서 개수는 산술을 위해 복구된 좌표를 지칭하며, 반드시 원본 소스 요소는 아닙니다. 검사는 헤더 전용으로 남아 있으며 유한 텐서 값을 인증하지 않습니다.

<a id="best-effort-numerical-recovery"></a>

### 최선의 수치 복구

기본 일반 레이어 가중 산술은 `repair_policy: best-effort-v1` 을 선포합니다. 비유한 기본 값, 포함하여 보존된 부동 상태는 산술 전에 0 로 바뀝니다. 비유한 LoRA 인자는 축적 전에 0 로 바뀝니다; 남은 비유한 델타 좌표는 0로 바뀝니다. 실패한 LoRA 축소는 이유와 함께 생략되어 다른 델타와 체크포인트를 유지합니다. 추가 체크포인트 좌표는 여전히 합산에는 정제된 기본값을 사용하고 차이에 0 를 사용합니다.

출력 NaNs 는 정제된 기본 좌표로 되돌아갑니다. 무한하거나 범위 밖의 결과는 캐스팅 전에 기본 저장소 데이터 유형의 유한 범위로 포화되며, FP8 와 FP16 를 포함합니다. 이는 원래 학습된 값의 복구가 아닌 결정론적 손실 제한입니다. 입력은 절대 다시 쓰여지지 않습니다.

비어있거나 부재하거나 잘못 형성되거나 불완전한 **추가** 리소스는 제외되며; 읽을 수 없는 추가 텐서는 중립적인 기여를 합니다. 모두 제외된 요청은 기본 레이아웃을 생성합니다. 만약 기본 수치 수정만 값이 변경되었다면 보고서는 `result_kind: repaired-base` 를 사용하며; 값이 변경되지 않았다면 `base-fallback` 와 `no_effect: true` 를 사용합니다. `numeric_normalization` 레코드는 `base_nonfinite_values`, `lora_nonfinite_values`, `lora_delta_nonfinite_values`, `output_nonfinite_values`, `output_clipped_values`, `skipped_lora_deltas` 및 `unreadable_material_tensors` 와 같은 카운트를 기록하며, 주소, 적용 가능한 소스 인덱스, 그리고 이유를 포함하여 최대 64 개의 `repair_events` 를 포함합니다.

기본값은 여전히 읽을 수 있고 비어 있지 않은 텐스 인벤토리를 노출해야 합니다: 누락되거나 잘린 기본값은 출력 계약을 정의할 수 없습니다. 소스 변형, 출력 충돌, 쓰기 실패, 사용 불가능한 의존성 및 할당 실패는 여전히 실제 실패입니다. 엄격한 정책은 수치 거부를 유지하며 이러한 수정을 활성화하지 않습니다. 산술 없이 복사된 통합 멤버는 원래 바이트를 유지하며; 이 정책으로 인해 그 내용은 전역적으로 정제되지 않습니다.

<a id="predictable-problems-and-regression-coverage"></a>

### 예측 가능한 문제 및 회귀 적용 범위

|조건|공통층 처리|
| --- | --- |
|누락된 키 또는 다른 내보내기 접두사|이름을 정규화한 다음 동일한 역할 매핑. 사용할 수 없는 경우 보존|
|다양한 블록 수|결정적 타이 브레이킹을 사용한 상대적 블록 깊이|
|전치된 행렬/동일 요소 수|근사 리샘플링 전 전치/재형성|
|다양한 순위, 커널, 너비 또는 스칼라 레이아웃|평면화 또는 축 정규화 최근접 리샘플링|
|빈 텐서 또는 예측 마커|기본 바이트와 형태를 보존하십시오|
|스케줄러 누락 또는 상태 값 다름|두 산술 모드 모두에서 기본 상태 유지|
|혼합 FP8/FP16/BF16/FP32/FP64|지원되는 누적 dtype; 기본 저장소 dtype 유지|
|스케일된 INT8/UINT8/FP8 및 0 포인트|기본 양자화기에서 디퀀타이즈하고, 병합하고, 재인코딩하십시오.|
|잘못된 양자화자|그 텐서를 보존하고 런타임 이유를 첨부하십시오.|
|차이의 재료 누락|베이스를 빼는 대신 기여도0|
|프로젝션 메모리 스파이크|평면 인덱스를 청크하고, 확장하기 전에 축을 축소합니다.|
|관련되지 않은 구성 요소 또는 패키지 단계|교차 구성 요소 없음 대체 경로|
|NaN/Inf 체크포인트 재료 좌표|합계: 기본 사용; 차이점: 0을 사용합니다. 유한한 좌표를 유지하고 수리를 보고합니다|
|비유한 기본 또는 LoRA 요인/델타|0-잘못된 좌표를 채우고; 실패한 LoRA 축약을 생략하고 이유를 포함하십시오.|
|부동 출력 오버플로/출력 NaN|유한 dtype 범위로 포화/삭제된 기본 좌표 사용|
|비어 있거나 없거나 손상되었거나 불완전한 추가 리소스|해당 리소스를 제외합니다. 다른 기여 유지|
|유효한 재료가 남아 있지 않습니다.|효과적인 혼합을 주장하지 않고 명시적인 베이스-대체 경로 또는 수리된 베이스 출력을 게시합니다.|
|체크포인트 합계 계수가 1보다 큽니다.|해당 비율을 유지하고 총 1로 정규화합니다. LoRA 계수를 변경하지 않고 그대로 둡니다.|
|읽을 수 없는 기본, 소스 변형, 동시 출력, 저장/할당 실패|무결성 검사 및 no-clobber 게시를 유지합니다. 정직하게 실패하세요|

회귀 coverage 는 64-쌍 스칼라/1-D/2-D/3-D/4-D 형상 행렬, 청크 경계, 양자화된 베이스/재료, 유효하지 않은 스케일, FP64 정밀도, 누락된 키 차이, 그리고 정확한 예상 수치 출력을 포함합니다. `tests/verify_model_merge_samples.py BASE MATERIAL` 는 전체 헤더 계획과 한계가 설정된 실시간 텐서 증거를 `build/real-merge-*` 에 씁니다. 그것은 최대 64 학습된 텐서를 독립적인 가중 합산 산술과 비교하고 원래 파일을 재작성하지 않고 보고된 누락된 스케줄러를 확인합니다.

`tests/verify_model_merge_nonfinite.py --base BASE --material MATERIAL` 는 보고된 CLIP 레이어-11 `mlp.fc1.weight` 실패를 설치된 CLI 에 대해 재연합니다. `--material` 를 반복하여 여러 실제 내보내기를 포함합니다. 그것은 원래 무한대 텐서와 유한한 레이어-10 탐구로 두 산술 모드를 모두 테스트하고, 수정된 출력이 유한하며 탐구가 여전히 변경됨을 확인하고 보고/샘플 아티팩트를 `build/nonfinite-real-*` 에 씁니다. 유닛 커버리지 또한 부분적 NaN /+Inf/-Inf 마스크, 완전히 유효하지 않은 텐서, 여러 재료, 형상 투영, FP8 저장, 및 한계가 설정된 좌표 진단을 포함합니다. `--repair-base` 를 사용하여 실제 무한대 베이스와 유한한 재료로 베이스 0채우기를 동일한 설치된 CLI 와 독립적인 출력 오라클을 통해 재연합니다.

한계: 리쉐이핑은 학습된 뉴런 정렬이 아닙니다. Fused QKV 패킹과 불투명 또는 패킹된 INT4 /아키텍처별 양자화기는 전용 디코더가 필요합니다; 일반적으로 재구성되지 않습니다. 알 수 없는 구성 요소는 유지될 수 있습니다. 명시적인 계수 정규화는 좌표 투영과 별도로 보고됩니다. 전체 출력 샤드가 직렬화 동안에도 여전히 메모리를 차지하므로, 차단된 투영은 완전한 아웃 오브 코어 직렬화기 가 아닙니다. 기본 재고 유지가 추론 백엔드가 출력을 로드할 수 있음을 증명하거나 생성된 이미지가 유용함을 증명하지는 않습니다.

<a id="unified-model-objects"></a>

## 통합 모델 객체

`--mode unified --output NAME.iildmodel` 는 ZIP64 저장 패키지 파일 하나를 생성하며, 이 파일에는 `model_index.json`, 소스 출처, 그리고 독립적인 체크포인트 멤버가 포함됩니다. 압축은 의도적으로 비활성화되어 큰 텐서 멤버를 스트리밍하고 추가 압축 해제 표현 없이 접근할 수 있습니다. 그것은 단일 네트워크 가중치 평균이나 증류가 아닌 **순서 이미지 개선 연쇄**입니다. 기본 범위의 선택이 여전히 적용됩니다: 새로운 패키지로 들어오는 것은 기본 생태계의 체크포인트와 LoRAs 에 대한 것뿐입니다. 실제 추론은 각 멤버가 설치된 네이티브 백엔드에 의해 지원되고 필요한 텍스트 인코더와 호환되는 VAE 를 포함해야 합니다. 알 수 없는 가족은 패키징될 수 있습니다; 이는 런타임 지원의 증거가 아닙니다. 아키텍처 식별은 조언적입니다: 인식되지 않은 하이브리드 서명과 마커 저장 관행은 독립적으로 복사된 멤버에서 그대로 유지됩니다.

첫 번째 체크포인트는 이미지를 생성합니다. 각 후속 체크포인트는 RGB 이미지를 자체 VAE 로 인코딩하고 이를 개선합니다. 모델은 기존 네이티브 컨텍스트 캐시를 사용하여 하나씩 실행됩니다. 체크포인트 단계는 독립적인 잠재 표현 공간과 텐서 레이아웃을 유지하며, 선택적 LoRA 적응은 아래에서 설명됩니다. 순서가 중요하며, 실행 비용이 누적되고 시각적 품질은 실제 생성물에 대해 평가되어야 합니다. 후기 단계는 네이티브 반크/Hires 정책을 유지합니다. 이 명시 모드에서의 체크포인트 가중치는 `[0,1]` 의 개선 강도이며, 기본값은 `0.35` 입니다; 0 는 해당 단계를 건너뜁니다. LoRA 강도 기본값은 `1` 입니다. 이 모드는 가중 뺄셈 해석을 갖지 않습니다.

엄격한 정책 하에서 LoRAs 는 **모든** 대상 이름, 모양, 랭크 및 알파 계약을 일치시킵니다. 각 어댑터는 가장 가까운 이전 호환 체크포인트에 융합되거나, 유일한 후기 호환 체크포인트에 융합됩니다. 어댑터를 의도된 모델 직후에 배치합니다. 동등한 Anima 네임스페이스에는 `diffusion_model.*` , `net.*` , `model.diffusion_model.*` 와 완전한 내보내기의 `model.diffusion_model.net.*` 가 포함됩니다. 기본 엄격한 정책 하에서 모호한 별칭과 호환되지 않는 모양은 오류로 남습니다.

<a id="legacy-lora-compatibility-objects"></a>

### 레거시 LoRA 호환성 객체

브리지 API 는 직접 검사용으로 계속 사용 가능하지만, 병합 진입점은 이제 일치하지 않는 LoRA 를 구하거나 삽입하는 외국 체크포인트를 더 이상 발견하지 않습니다. 기본 범위의 선택은 해당 전체 어댑터를 먼저 제외합니다. 아래 세부사항은 유지된 레거시 API 를 설명하며, 현재 병합 라우팅이 아닙니다.

요청한 체크포인트 중 어느 것도 LoRA를 수용하지 않으면, 통합 모드는 이제 요청한 체크포인트 폴더의 바로 옆 safetensors 파일 중에서 **실제 호환 체크포인트**를 찾는다. 모든 어댑터 대상·형태·rank·alpha를 검사한다. 패키지 내부로 재귀 탐색하거나 기기 전체를 스캔하거나 모델을 다운로드하거나 파일명을 호환성 증거로 사용하지 않는다. 무관한 잘못된 파일은 무시한다.

`LoraCompatibility.inspect(checkpoint, adapter)` 는 재사용 가능한 구조적 결과 ( `compatible` , `architecture` , `target_count` , `embedded_components` , `reason` ) 를 노출합니다. `LoraCompatibilityBridge.discover(checkpoints)` 는 로컬 후보를 나열하며, `LoraCompatibilityBridge.resolve(adapter, candidates)` 는 하나를 선택하고 `checkpoint` , `refinement_strength` 및 `as_dict()` 를 포함하는 불변 브릿지를 반환합니다. 이 메서드들은 `model_merge_compatibility.py` 에 존재하며, 전체 체크포인트 텐서를 로드하거나 체크포인트 해시를 계산하지 않습니다. 레거시 입력 검사에는 필요할 때 기존 안전한 변환을 여전히 사용합니다.

아니마의 경우, 내장된 텍스트 인코더와 VAE 를 가진 일치하는 체크포인트가 데노이저 전용 내보내기보다 우선순위가 높습니다. 컴포넌트 존재는 추론 인증서가 아닌 구조적 증거입니다. 여러 개의 동등하게 적합한 체크포인트가 남아있다면, 선택은 실패하며 그들의 경로를 표시합니다. 원하는 체크포인트를 일반 재료로 추가하거나 `--compatibility-model PATH` 를 전달하세요. 이 옵션을 반복하여 다른 폴더에서 후보 풀을 제공합니다. `--no-auto-compatibility` 는 대체 경로 를 비활성화합니다. 상응하는 Python 인수는 `compatibility_models` 입니다: `None` 는 자식을 검색하며, 시퀀스는 후보를 제한하고, `[]` 는 대체 경로 를 비활성화합니다.

선택된 체크포인트는 LoRA 와 융합되어 캐스케이드의 해당 어댑터 위치에 삽입됩니다. 추가 호환 가능한 어댑터는 동일한 단계를 재사용할 수 있습니다. 기존 RGB 핸드오프는 해당 체크포인트의 자체 VAE 로 이전 이미지를 다시 인코딩하며 각 네트워크는 자신의 잠재 차원과 예측 설정을 유지합니다. 브리지 리파인먼트 기본값은 `0.35` 입니다; `--compatibility-strength` / `compatibility_strength` 는 LoRA 델타 강도와 독립적으로 `[0,1]` 로 설정합니다. 요청된 재료 인덱스와 가중치는 그대로 유지되며, 추가된 브리지 소스는 별도의 출처 인덱스와 해시를 받습니다. 검사 보고서에는 `resolved_sources` 가 포함되고, 단계에는 `compatibility_bridge` 가 포함되며, 완료된 보고서에는 `compatibility_bridge_count` 가 포함됩니다. 총 단계 수는 64 로 제한됩니다.

예를 들어, SDXL + Noob + Anima LoRA 는 로컬 호환성 있는 완전한 Anima 체크포인트를 자동으로 추가할 수 있습니다. 결과 객체는 3 개의 독립적인 네트워크를 포함합니다. 이는 Anima 가중치를 SDXL 로 변환하는 것이 아닙니다. 기반 네트워크가 누락된 LoRA 는 누락된 기본 네트워크를 재현할 수 없습니다: 일치하는 체크포인트가 없으면 작업이 여전히 실패하며, 가중치 합/차 모드도 엄격하게 유지됩니다. 임의의 텐서 리사이징, 부분 어댑터 제거, 가족 간 LoRA 투영 또는 재학습은 주장되지 않습니다.

```python
from model_merge import merge_models
from model_merge_compatibility import LoraCompatibility, LoraCompatibilityBridge

compatibility = LoraCompatibility.inspect("/models/anima-complete.safetensors", "/models/style.safetensors")
bridge = LoraCompatibilityBridge.resolve("/models/style.safetensors", ["/models/anima-complete.safetensors"])
report = merge_models("/models/sdxl.safetensors", "/models/style.safetensors",
                      mode="unified", compatibility_models=[bridge.checkpoint],
                      compatibility_strength=0.35, output="/models/compatible.iildmodel")
```

```bash
iild-merge --base-model /models/sdxl.safetensors \
  --additional-model /models/noob-vpred.safetensors \
  --additional-model /models/anima.safetensors \
  --additional-model /models/anima-lora.safetensors \
  --mode unified --weights 0.35 0.35 1 --output /models/combined.iildmodel
iild-generate --model-path /models/combined.iildmodel --prompt 'a mountain lake' \
  --output-dir /images/combined
```

네이티브 C++ 이미지 진입점은 `.iildmodel` 패키지 파일을 `modelPath` 로 받으며, CLI 는 `unified` 백엔드를 자동으로 선택합니다. 로더는 압축, 암호화, 중복, 리디렉션 및 과도한 크기의 항목을 거부하고, 매니페스트 계약, CRC, 멤버 크기와 게시된 SHA-256 신원을 확인하며, 출처 신원 범위에 국한된 로컬 캐시를 안전하게 생성합니다. 패키지는 독립적인 복사본을 포함하며 원래 경로가 이동한 후에도 사용 가능하게 유지됩니다. 현재 캐스케이드는 단일 파일 체크포인트와 표준 지원 LoRAs ; Diffusers 체크포인트 디렉토리를 먼저 내보내야 합니다. 다단계 통합 객체를 하나의 체크포인트로 평탄화하지 않으며, 대기 중인 LoRAs 없이 유효한 단일 단계 `.iildmodel` 는 가중치 병합 기준 또는 재료로 전체로 수용됩니다. 레거시 디렉토리 형식 `.iildmodel` 객체는 생성을 위해 읽을 수 있지만 새로운 통합 병합은 항상 단일 파일을 게시합니다. 아래의 Safetensors/레거시 변환 및 어댑터 제한 사항이 여전히 적용됩니다. 취소, 오류 또는 변경된 멤버는 중간 이미지를 폐기합니다. 네이티브 로더는 멤버 경로를 제한하고 크기를 유효성 검사하며 실행 내내 파일 신원을 확인합니다; CLI 도 기록된 SHA-256 해시와 SDK 의 변경되지 않은 파일 해시 캐시를 확인합니다.

어댑터가 없는 멤버는 이미 소스에 존재하는 비유한 값을 포함하여 바이트 단위로 보존됩니다. 이를 패키징하는 것은 수치적 건강 상태를 인증하거나 런타임 가 사용할 수 없는 레이어의 값을 대체하지 않습니다. 보고서는 이 유효성 검사 범위를 명시적으로 명시합니다. 어댑터가 있는 체크포인트는 게시 전에 일반적인 유한 입력/유한 출력 산술 검사를 통과합니다.

`--inspect` 는 구조적 검사를 수행하고 전체 체크포인트 해싱 또는 출력 생성 전에 해결된 강도, 아키텍처 증거 및 어댑터 라우팅을 보고합니다. 레거시 pickle 입력은 여전히 기존 안전한 변환이 필요합니다. 일반 병합도 사전 검사 전에 수행합니다. `--print-config` 는 인수만 계약하는 계약을 유지합니다. 비어 있는 `v_pred` / `ztsnr` 텐서는 예측 마커이며, 학습 가능한 가중치가 아닙니다: 일치하는 마커는 유지되고, 다른 마커는 통합 모드가 필요합니다. 그들을 단순히 삭제하면 결과 모델의 해석이 변경됩니다.

검증: `ModelMergeCompatibilityTests.py` 는 중첩된 내보내기 이름, 공개 호환성 객체, 자동/명시/비활성 선택, 모호성, 형식 거부, 소스 보존, 정렬, 공유 브릿지 및 실제 델라프 산술을 확인합니다. `UnifiedModelMergeTests.py` 는 실제 텐서, 마커 보존, 초기 실패, 어댑터 라우팅, 유한 산술, 바이트 보존 및 게시를 확인합니다. `NativeResultTests` 와 `NativeMobileResultTests` 는 제어된 엔진 결과에 대해 생산 연쇄 어댑터를 실행하여 패키지 파일 로딩, RGB 전달, 0강도 스킵, 경로 제한 및 취소를 확인합니다. `UnifiedImageTests.py` 는 또한 저장된 아카이브 물질화를 확인하고, SHA-256 검증, 자동 라우팅 및 압축과 탐색 거부를 확인합니다. 이 픽스처 는 모든 이름된 모델 가족에 대한 이미지 품질을 인증하지 않습니다.

`iild-merge` 와 Python `merge_models()` 는 로컬 체크포인트와 LoRAs 를 결합합니다. **기본 모델과 첫 번째 추가 모델은**필수입니다. 가중치, 추가 모델, 모드, 출력 및 변환 캐시는 선택 사항입니다. 기본기는 전체 체크포인트이며, 첫 번째 및 이후 물질은 각각 체크포인트 또는 LoRA 일 수 있습니다. 병합은 기존 Python PyTorch / safetensors 런타임, `iild-generate` 를 함께 사용하며, C++ 텐서 연산이나 추론 워커 요청이 아닙니다.

<a id="arithmetic-and-defaults"></a>

## 산술 및 기본값

`A`를 기본 모델로 하고 `B_i`를 추가 전체 체크포인트로 둡니다. 연산은 순차적 재혼합 없이 일치하는 모든 부동 소수점 텐서에 적용됩니다.

|모드|공식|생략된 가중치|
| --- | --- | --- |
|`weighted-sum`(기본값)| `(1 - sum(w_i)) A + sum(w_i B_i)` |모든 모델은 `N` 추가 모델에 대해 `1 / (N + 1)`를 받습니다.|
| `weighted-difference` | `A - sum(w_i B_i)` |각 추가 모델의 `0.5`를 빼십시오; 기본 계수는 `1`를 유지합니다.|

2 모델의 기본값은 `(A + B) / 2` 및 `A - 0.5 B`입니다. 3 모델은 합계 가중치 `[0.2, 0.3]`를 사용하여 `0.5 A + 0.2 B + 0.3 C`를 생성합니다. 동일한 차이 가중치가 `A - 0.2 B - 0.3 C`를 생성합니다. 차이는 직접 가중 뺄셈을 의미합니다; 암묵적 참조 모델이나 `A + w(B - C)`는 존재하지 않습니다.

LoRAs는 해당 체크포인트 결과에 대한 변경 사항을 제공합니다. `D_j`를 자체 알파/순위 스케일링을 포함한 각 어댑터의 델타로 설정하고 `s_j`는 요청된 강도로 설정합니다.

|LoRAs가 포함된 모드|공식|
| --- | --- |
| `weighted-sum` | `(1 - sum(w_i)) A + sum(w_i B_i) + sum(s_j D_j)` |
| `weighted-difference` | `A - sum(w_i B_i) - sum(s_j D_j)` |

LoRAs 기본 계수를 소모하지 않습니다. 각 누락된 LoRA 강도는 **1** 모드 중 어느 모드든 동일하며, 체크포인트 기본값은 기본 및 전체 체크포인트만 세웁니다. 따라서 기본 + 하나 LoRA 기본값은 `A + D` 또는 `A - D` 이고, 기본 + 체크포인트 + LoRA 기본값은 `0.5 A + 0.5 B + D` 또는 `A - 0.5 B - D` 입니다. 여러 어댑터는 서로 다른 랭크를 가질 수 있습니다. 그 델타는 동일한 체크포인트 블렌드에 누적됩니다.

모든 추가 재료로 브로드캐스트할 하나의 가중치 또는 동일한 순서대로 추가 모델마다 정확히 하나의 가중치를 공급합니다. 값은 유한한 음이 아닌 수여야 합니다. 합계 모드에서 전체 체크포인트 가중치의 나머지는 기본에 속합니다. 일반 레이어 합계가 `1` 보다 크면 `1` 로 비례하여 정규화되며, 기본 가중치는 0 입니다. `weight_normalization` 는 요청된 계수와 유효한 계수를 기록합니다. 요청된 가중치를 더해도 오버플로우가 발생하더라도 이 계산은 안정적으로 유지됩니다. 엄격한 합계 모드에서는 체크포인트 합계가 `1` 보다 크거나 같을 수 없습니다. LoRA 강도와 차이 가중치는 `1` 를 초과할 수 있으며 정규화되지 않습니다. 부울, 음수, NaN /무한대 및 불일치된 가중치 목록은 텐서 로딩 전에 실패하며, 체크포인트 합계는 재료 분류 후에 확인됩니다. 0 가중치는 여전히 입력 구조를 검사하지만, 일반 레이어 체크포인트 산술은 0-가중치 재료 값을 읽는 것을 건너뜁니다.

## CLI

일반적인 [installation](installation.md)후 명령은 다음과 같습니다.

```bash
iild-merge --base-model /models/base.safetensors \
  --additional-model /models/style.safetensors \
  --output /models/merged.safetensors

iild-merge --base-model /models/base.safetensors \
  --additional-model /models/style.safetensors \
  --additional-model /models/detail.safetensors \
  --weights 0.2 0.3 --output /models/blended.safetensors

iild-merge --base-model /models/base.safetensors \
  --additional-model /models/style.safetensors \
  --mode weighted-difference --weight 0.25 \
  --output /models/subtracted.safetensors

# 필수 추가 모델 자체가 LoRA일 수도 있다.
iild-merge --base-model /models/base.safetensors \
  --additional-model /models/style-lora.safetensors \
  --weight 0.8 --output /models/fused.safetensors

# 순서가 있는 재료 목록에서 체크포인트와 LoRA를 교차 배치할 수 있다.
iild-merge --base-model /models/base-diffusers \
  --additional-model /models/style-lora.safetensors \
  --additional-model /models/other-diffusers \
  --additional-model /models/peft-adapter \
  --weights 0.8 0.25 0.3 --mode weighted-difference \
  --output /models/subtracted-diffusers
```

결제 시 `iild-merge` 대신 `reference/diffusers/.venv/bin/python reference/merge.py`를 사용하세요. 설치된 명령은 `IILD_PYTHON_EXECUTABLE` 및 연결된 Diffusers 환경을 생성 실행 프로그램과 공유합니다.

추가 모델마다 `--additional-model` 를 반복합니다. `--weights` 와 `--weight` 는 별칭입니다. `--print-config` 는 Torch , 텐서 읽기, 해싱, 출력 생성 또는 다운로드 없이 경로와 요청된 가중치를 확인합니다. 그것은 누락된 가중치와 `base_weight` 를 `null` 로 보고하며, `coefficient_resolution: after-input-inspection` 에서 유효한 기본값은 각 재료가 체크포인트 또는 LoRA 키를 포함하는지에 따라 달라집니다. 완료된 병합 보고서는 해결된 계수와 종류를 포함합니다. 구성 미리보기는 모델 호환성을 설정하지 않습니다. 누락된 출력은 `build/reference/merged/<base-name>-<mode>.safetensors` 로 기본값이 되거나, Diffusers 패키지의 확장 없는 디렉토리 이름을 사용합니다. 기존 목적지는 거부되며, 다른 결과에 대해 다른 `--output` 를 선택해야 합니다.

## Python API

체크아웃의 `reference/diffusers` 디렉터리 또는 설치된 `share/iiLocalDiffusion/reference/diffusers` 디렉터리를 Python의 모듈 경로에 추가합니다.

```python
from model_merge import merge_models

# 이 두 모델 인수만 필수이다.
report = merge_models("/models/base.safetensors", "/models/style.safetensors")

# 같은 필수 위치에 LoRA를 사용할 수 있으며 기본 강도는 1이다.
report = merge_models("/models/base.safetensors", "/models/style-lora.safetensors")

report = merge_models(
    "/models/base.safetensors",
    "/models/style.safetensors",
    additional_models=["/models/detail.safetensors"],
    weights=[0.2, 0.3],
    mode="weighted-difference",
    output="/models/subtracted.safetensors",
)

# 호환되는 생태계 변형을 베이스 배치에 매핑한다.
report = merge_models(
    "/models/flux.safetensors",
    "/models/krea-int8.safetensors",
    checkpoint_policy="common-layer",
    output="/models/flux-krea-experimental.safetensors",
)
```

`resolve_merge_request()` 는 `model_merge_options` 에서 동일한 경량 검증과 불변 해결된 요청을 노출하며, `merge_models()` 는 정렬된 소스 SHA-256 식별자, 재료 종류, 계수, 런타임 버전, 출력 해시 및 텐서/버퍼 개수가 포함된 JSON 준비된 보고서를 반환합니다. CLI 는 이 보고서를 인쇄합니다.

<a id="formats-and-compatibility"></a>

## 형식 및 호환성

2 가중산 모드의 경우, 전체 체크포인트는 모두 기본 저장 형식인 단일 파일 또는 Diffusers 디렉터리를 사용해야 합니다. LoRA 파일/디렉터리는 두 형식 중 하나로 혼합할 수 있습니다:

- `.safetensors` 및 `.safetensor` 파일은 기존 안전 로더로 읽습니다. 텐서 이름과 도형은 입력 간에 정확히 일치해야 합니다.
- `.ckpt`, `.pt`, `.pth` 및 `.bin` 단일 체크포인트는 기존 [가중량 전용 변환](checkpoint-formats.md)를 재사용합니다. 그것의 Torch 버전 게이트와 위험한 피클 로딩 재시도 거부가 여전히 효력을 유지하고 있습니다. 변환된 파일은 `build/reference/model-merge-cache/`를 사용하며, `--cache-dir` 로 재정의할 수 있습니다.
- Diffusers 디렉토리는 `model_index.json` 와 safetensors 가중치가 필요합니다. 텐서 이름은 구성 디렉토리 내에서 매칭되므로 입력은 다른 샤드 레이아웃을 사용할 수 있습니다. 샤드 인덱스는 실제 키와 파일 이름에 대해 검증됩니다. 출력은 기본 파일 이름, 샤드, 구성, 토크나이저 자산 및 기타 가시적인 보조 파일을 유지합니다. 파일은 심볼릭 링크일 수 있으며, 중첩된 디렉토리 심볼릭 링크와 숨겨진 캐시/버전 관리 디렉토리는 지원되는 패키지 레이아웃에서 제외됩니다.
- 디렉토리 런타임 구성/토크나이저 자산은 일치해야 합니다. JSON 비교는 `_name_or_path`, `_diffusers_version`, `transformers_version`, `_commit_hash`, `_use_default_values` 및 `torch_dtype` 만 무시합니다. 샤드 인덱스 레이아웃과 이전 `merge.json` 는 일치할 필요가 없습니다. 다른 아키텍처, 예측 설정 또는 토크나이저는 병합 전에 호환 가능한 입력을 필요로 합니다. 피클, GGUF, ONNX 또는 기타 가중치 형식을 가진 패키지는 먼저 단일 safetensors 변형으로 준비되어야 합니다. 중복 텐서 키/변형은 거부됩니다.
- 유전 디렉토리 형태의 `.iildmodel` 는 가중치 산술 중 패키지로 남아 있으므로 `.iildmodel` 출력 경로를 필요로 합니다. 각 구성원은 요청된 산술로 다시 작성되며, SHA-256 단계와 바이트 크기는 `model_index.json` 에서 새로고침되고 패키지는 디렉토리로 원자적으로 게시됩니다. 기본 텐서 이름, 형상 및 데이터 타입이 보존되므로 동일한 또는 거의 동일한 디스크 크기가 예상됩니다; `changed_tensor_count` 와 출력 해시, 즉 파일 크기가 아닌 것이 수치 값이 변경되었는지 증명합니다.
- Float8 checkpoint 텐서는 유한 값 확인과 가중치 산술을 위해 일반 Float32 누적 데이터 타입으로 승격된 후, 기본 체크포인트의 Float8 저장 데이터 타입으로 다시 변환됩니다. 이는 Float8 텐서에 대해 직접 구현되지 않은 PyTorch 연산을 호출하지 않고도 FP8/INT8 공통 레이어 프로젝션을 허용합니다.

`--checkpoint-policy strict` 아래에서는 정수/부울 버퍼는 동일한 dtype, 형식 및 값을 가져야 하며 베이스에서 복사됩니다. 부동 소수점 입력은 FP8 , FP16 , BF16 , FP32 및 FP64 을 지원하며, 혼합 정밀도를 포함하며 출력은 베이스 dtype 을 유지합니다. FP8/FP16/BF16 는 FP32 에 누적되며, FP64 입력은 FP64 를 유지합니다. 엄격하게는 비유한 재료를 거부하며, 스케일된 정수 저장의 양자화를 해제하지 않습니다. 기본 공통 레이어 정책은 위에서 설명된 매핑, 양자화 및 비유한 좌표 대체 경로 대신 보고된 매핑을 사용합니다. 두 정책 모두 소스 무결성 검사를 유지하고 복잡한 가중치를 거부합니다. Strict 는 부동 출력 오버플로우를 거부하며, common-layer 는 이를 포화시키고 수리 기록을 합니다.

호환성 검사는 구조적/구성 일관성을 확립하며, 시각적 품질이나 독립적으로 훈련된 체크포인트가 유용한 가중치 공간을 공유하는 것을 의미하지 않습니다. Strict 정책 하에 존재하는 경우, 명시된 `modelspec.architecture`, `modelspec.implementation` 및 `modelspec.prediction_type` 헤더 힌트는 일치하는 구성 요소 내에서 서로 일치해야 합니다. 단일 파일은 완전한 토크나이저/스케줄러 구성을 포함하지 않으며, 호출자는 호환 가능한 계열에서 체크포인트를 공급해야 합니다. 결과는 기존 `iild-generate --model-path` 워크플로우에 전달될 수 있으며, 해당 워크플로우의 일반적인 계열/구성 인수를 사용합니다. 모델 라이선스는 입력과 결과 가중치에 계속 적용됩니다.

<a id="lora-materials"></a>

## LoRA 재료

LoRA 입력은 파일 이름이 아닌 텐서 키로 식별됩니다. 지원되는 입력에는 안전한 텐서 파일, 동일한 안전한 변환기를 통한 레거시 텐서 체크포인트, PEFT 어댑터 디렉토리 (`adapter_config.json` 및 safetensors), 그리고 Diffusers /Kohya safetensors 어댑터 내보내기 디렉토리를 포함합니다. 각 재료는 하나의 어댑터를 포함해야 하며, 완전한 다운/업 쌍과 관련 없는 훈련된 텐서를 포함하지 않아야 합니다. 명시적으로 선택된 `adapter_model.safetensors` 파일의 경우, 그 형제 파일 `adapter_config.json` 도 읽히고 해시되며 재검증됩니다.

지원되는 파라미터화에는 표준 선형 LoRA 와 그룹화되지 않은 Conv1d/2d/3d LoRA (공간적 다운 투영과 1x1 업 투영을 포함) 이 있습니다. 선형 타겟의 경우, `D = (alpha / rank) * (B @ A)` 입니다. 레이어별 `.alpha` 텐서는 PEFT `lora_alpha` / `alpha_pattern` 보다 우선하며, alpha 가 누락되면 실제 랭크로 기본값이 적용됩니다. PEFT `r` / `rank_pattern` 는 실제 텐서 랭크와 일치해야 합니다. `use_rslora` 는 `alpha / sqrt(rank)` 를 사용하며, `fan_in_fan_out` 는 선형 타겟에 대해 지원됩니다. 텐서 전용 PEFT 내보내기는 alpha 가 랭크와 다를 때 설정을 유지해야 합니다.

인식된 이름에는 PEFT `lora_A` / `lora_B` (선택적 단일 어댑터 이름과 `base_model.model.` 접두사 포함), Diffusers `lora.down` / `lora.up`, 레거시 주의 처리기/ `lora_linear_layer` 쌍, 그리고 `lora_down` / `lora_up` 모듈 이름과 `lora_unet_`, `lora_te_`, `lora_te1_` 또는 `lora_te2_` 가 포함됩니다. 표준 Diffusers 구성/모듈 이름도 트랜스포머 가중치에 타겟팅할 수 있습니다.

타겟은 실제 기본 키에 대해 해결되며, 명칭이 없는 이름은 구성 간에 고유해야 합니다. SD 1.x/2.x/ XL 원래 UNet 이름은 고정된 Diffusers LDM 키 매핑을 사용하며 (표준 2-잔여 레이어-블록 레이아웃입니다). SD 1.x/ XL 의 첫 CLIP 와 SD 2.x/ XL OpenCLIP 이름이 지원되며, 패킹된 Q/K/V 행 업데이트와 전치된 OpenCLIP 텍스트 투영을 포함합니다. 기본 레이아웃은 출력에서 유지됩니다. 모호한 평탄화된 이름, 알 수 없는 타겟, 호환되지 않는 형식, 불완전한 쌍 및 지원되지 않는 텐서는 해당 전체 어댑터를 호환성 선택 시 제외하거나 (또는 엄격한 선택 시 사용 가능한 자료가 없으면 실패합니다). 수치 LoRA NaN /Inf 값은 위의 공통 레이어 복구를 사용하며, 엄격한 산술은 이를 거부합니다. SGM /Kohya 블록 이름은 또한 Diffusers UNet 패키지를 타겟할 수 있으며, 해당 `unet/config.json` `layers_per_block` 와 고정된 어댑터 이름 변환기를 사용합니다.

DoRA, LyCORIS / LoHa / LoKr, 임베딩 LoRA, 그룹 컨볼루션, 편향/모듈 저장 업데이트 및 수정된 기본 초기화를 가진 어댑터는 이 오프라인 병합기에 의해 지원되지 않습니다. 아키텍처별 패킹된 트랜스포머 어댑터 형식은 기본과 일치하는 타겟을 가진 표준 Diffusers / PEFT 내보내기가 필요합니다. 이는 별도의 [생성 시간 LoRA 로더](lora.md) 와 그 백엔드 기능을 변경하지 않습니다.

저장된 결과는 기존 텐서에 델타가 융합된 전체 체크포인트입니다. 생성 시 LoRA 입력이 필요하지 않습니다. 출처는 각 어댑터의 소스 해시, 확인된 대상 및 알파/순위 규모와 요청된 강도를 기록합니다.

<a id="storage-memory-and-verification"></a>

## 저장, 메모리 및 검증

병합은 CPU 가중치 준비 작업이며 추론 파이프라인을 로드하거나 GPU 를 예약하지 않습니다. 입력 safetensors 은 게으르게 열립니다; 하나의 출력 샤드가 유지되고 추가 텐서는 개별적으로 읽힙니다. 메모리는 출력 샤드, 작업 텐서, 매핑된 입력 페이지 및 직렬화기 버퍼를 포함합니다. 단일 파일 모델은 하나의 샤드이므로 해당 출력을 위한 충분한 RAM 가 여전히 필요합니다.

원본 파일은 절대 작성되지 않습니다. 캐시된 모델 해시는 기존 파일 동일성 확인이 변경되지 않은 동안만 재사용됩니다. 모든 소스는 게시 전에 재검증됩니다; 변경된 파일 또는 변경된 패키지 목록은 병합을 중단시킵니다. 임시 출력이 목적지 옆에 생성되고 실패 시 제거됩니다. 단일 파일 결과는 원자적, 덮어쓰지 않는 하드 링크 게시를 사용하며, 완전한 패키지는 디렉토리 이름 변경을 사용합니다. 파일 출력은 따라서 파일 시스템 하드 링크 지원이 필요합니다. 입력 패키지 내의 기존 출력 및 목적지는 거부됩니다.

각 safetensors 헤더는 `iild_merge` provenance 를 저장하며 원래 헤더 메타데이터를 `iild_merge_base_metadata` 하에 중첩합니다. 합쳐진 파일의 식별자로 이전 모델 해시를 재사용하지 않습니다. Diffusers 패키지에는 출력 해시를 포함한 `merge.json` 도 포함됩니다. 성공적인 병합 보고서에는 이미지 품질 주장은 내장되어 있지 않습니다.

기존 런타임를 사용하여 소규모 실제 텐서 테스트를 실행한 다음 네이티브 제품군을 실행합니다.

```bash
reference/diffusers/.venv/bin/python tests/ModelMergeTests.py
reference/diffusers/.venv/bin/python tests/ModelMergeLoraTests.py
reference/diffusers/.venv/bin/python tests/ModelMergeDiffusersSmoke.py
reference/diffusers/.venv/bin/python tests/ModelMergeLoraSmoke.py
cmake -S . -B build
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

`ModelMergeTests` 는 항상 인수를 확인하며 의존성 없는 도움말/구성을 확인합니다. 선택된 인터프리터가 Torch / safetensors 를 갖추지 못하면 실제 텐서 케이스는 건너뛰며 명시적인 첫 번째 명령은 기존 생성 환경에서 실행합니다. `ModelMergeLoraTests` 는 실제 작은 텐서를 사용하여 어댑터 산술, 혼합 재료 기본값, 매핑, 랭크/알파, 컨볼루션, 정밀도, 실패 및 소스 보존을 확인합니다. 의존성이 이용 불가능하면 런타임 케이스도 건너뜁니다. `InstallConsumerTests` 는 이동된 `iild-merge` 명령과 구성을 확인합니다. `ModelMergeDiffusersSmoke.py` 는 로컬에서 2 tiny DDPM 모델을 초기화하고 모든 재로딩 파라미터에 대해 두 산술 모드를 검증한 후 각 병합 모델을 SDK 생성기를 통해 실행합니다. 그림과 `verification.json` 를 `build/reference/model-merge-smoke/` 아래에 씁니다. `--merge-entry` 와 `--generate-entry` 를 전달하여 이동된 설치된 명령도 확인합니다. 네트워크 요청을 하지 않으며 전체 크기 모델 품질을 설정하지 않습니다.

`ModelMergeLoraSmoke.py` 는 2 nonzero PEFT 어댑터를 생성하며 서로 다른 랭크의 어댑터는 주의와 컨볼루션 타겟을 모두 가지며 2 tiny DDPM 체크포인트와 결합합니다. 합과 뺄셈은 모든 모델 파라미터에 대해 PEFT 의 융합되지 않은 전방 통과와 공식 `merge_and_unload()` 결과와 비교된 후 SDK 이미지 생성을 통해 실행됩니다. 보고서 및 이미지는 `build/reference/model-merge-lora-smoke/` 보다 아래에 유지됩니다. 동일하게 설치된 진입점 오버라이드를 허용하며 완전히 오프라인으로 실행됩니다.

<a id="foreground-preparation-of-a-packaged-checkpoint"></a>

### 패키지 체크포인트의 포그라운드 준비

단일 단계 통합 패키지는 완전한 체크포인트 (원래의 디노이저, 텍스트 인코더 및 VAE 포함) 를 하나의 `model.safetensors` 멤버로 내장할 수 있습니다. 매니페스트는 상대 경로, 바이트 크기 및 SHA-256 를 기록하며 패키지는 하나의 카탈로그 객체입니다. 로더는 `model_index.json` 를 식별하며 디렉토리 접미사를 식별하지 않습니다.

통합 프론트그라운드 준비는 NativeEngine 를 InferenceSession 에 유지하며 성공적인 준비를 기록하며 단일 파일 백엔드와 동일합니다. 캐시는 매니페스트 및 모든 멤버를 추적하며 변경된 멤버는 준비 상태를 무효화합니다. 반복된 준비는 엔진을 재사용하며 출력 이미지를 생성하지 않습니다. UnifiedImageTests 은 준비, 재사용 및 동일 크기 멤버 교체와 함께 패키지 유효성을 포함합니다.

<a id="installed-python-and-legacy-compatibility-registry"></a>

## Python 및 레거시 호환성 레지스트리 설치

설치된 실행기는 모델 모듈을 가져오기 전에 Python >= 3.10을 검증한다. `IILD_PYTHON_EXECUTABLE`가 `reference/runtime-python.json`보다 우선하며 해당 파일의 형태는 `{"python":"/absolute/venv/bin/python"}`이다. 그 밖의 경우에는 번들된 `.venv` 또는 실행기의 인터프리터를 고려한다. 잘못되었거나 오래된 환경은 어노테이션 traceback 대신 조치할 수 있는 진단을 제공한다.

통합 브리지 발견은 또한 설치 로컬 `reference/merge-compatibility.json` 를 읽으며 `{"checkpoints":["/absolute/full-checkpoint.safetensors"]}` 입니다. 상대 경로는 해당 레지스트리 디렉토리에서 해결됩니다. 레지스트리는 인접한 체크포인트 발견을 보강하며 단일 체크포인트 래퍼를 포함하는 범주를 포함합니다. 명시적 호환성 후보는 발견을 오버라이드합니다. 모든 타겟/형상이 유효성 검사되며 누락되거나 모호하거나 호환되지 않는 베이스는 여전히 실패합니다. 검사에서는 모델을 다운로드하거나 가중치 산술을 실행하지 않습니다. 다른 네트워크는 순차적 이미지 정제로 구성되며 정규화는 임의의 네트워크 텐서를 상호 교환 가능하게 만들지 않습니다.

PythonLauncherTests는 환경 선택, 명시적 재정의, 항목 가져오기 전 이전 버전 거부 및 손상된 구성을 확인합니다. ModelMergeCompatibilityTests는 래핑된 입력 및 출력 없는 입력 검사에서 등록된 브리지 검색을 다룹니다.

<a id="lora-alias-normalization"></a>

### LoRA 별칭 정규화

LoRA 투영 쌍은 형식, 랭크, 알파 및 방향성 확인 후 체크포인트의 실제 텐서 주소로 정규화됩니다. 일부 내보내기에는 동일한 주소에 대한 SGM 와 Diffusers 이름이 모두 포함될 수 있습니다. 동일한 형식, 스케일 및 방향성을 가진 동일한 투영 텐서는 한 번 적용됩니다. 구별되는 투영 쌍은 해당 주소에 대한 가산 델타로 유지되며, 각 쌍의 자체 알파/랭크 스케일을 포함합니다. 대상은 아무런 알림 없이 삭제되거나 임의로 재형상되지 않습니다. 보고서는 `lora_alias_policy` 를 `identical-projections-once; distinct-projections-additive` 로 노출합니다. 입력 검사 는 동등성을 확립하기 위해 중복 LoRA 프로젝션을 읽을 수 있지만, 전체 체크포인트 텐서를 구체화하거나 병합된 모델을 생성하지 않습니다.

LoRA 회귀 제품군은 중복된 별칭이 한 번 적용되고, 고유한 투영이 올바르게 합산되고, 다양한 알파 값이 작은 픽스처 텐서를 사용하여 개별 스케일링을 유지하는지 확인합니다.

<a id="synthetic-cross-family-lora-adaptation"></a>

## 합성 제품군 LoRA 적응

공통 계층 및 엄격한 검사점 정책에 따라 사전 검사에는 엄격한 기본 대상이 필요하며 이 어댑터가 실행되기 전에 일치하지 않는 LoRA를 제외합니다. `base-layout` 가중치 적용 정책은 해당 선택을 재정의하고 다음 결정적 어댑터를 활성화합니다.

`--lora-policy synthetic` ( Python   `lora_policy="synthetic"` ) 의 적응기 변환은 의도적으로 훈련되지 않은 적응기 변환을 가능하게 합니다. 기본 `strict` 는 기존 호환성 검사를 유지합니다. 정확한 호환 가능한 타겟은 정상적인 계산을 유지합니다. 일치하지 않는 모듈은 구성 요소 (디노이저/텍스트/ VAE ), 투영 역할, 블록/레이어 인덱스, 평탄화된 형상 거리, 그리고 마지막으로 사전적 텐서 주소를 통해 결정론적으로 부동점수 행렬/합성곱 가중치로 매핑됩니다. 동일한 구성 요소/역할 타겟이 존재하지 않는 경우, 다음으로 가까운 후보가 사용되며 리매핑이 보고됩니다. 새로운 네트워크 레이어는 발명되지 않으며 모든 기본 텐서 이름, 형상 및 데이터 형식은 변경되지 않습니다.

일반적인 랭크/알파 스케일링 후, 소스 델타는 출력 행과 입력 열로 평탄화됩니다. 그의 왼쪽 위 교차점은 유지되고, 초과 행/열은 잘리고, 누락된 값은 **0** 로 채운 후 목표 텐서로 다시 형성됩니다. 이는 결정론적이며 다운로드나 훈련이 필요하지 않으며, 완전한 두 번째 모델이 아닌 현재 소스/타겟 델타만 할당합니다. 하나의 타겟으로 매핑된 여러 개의 다른 델타는 요청된 강도로 합쳐집니다. 이는 업데이트를 확대할 수 있으므로 시각적 품질을 평가할 때 작은 강도로 시작합니다. 형식이 잘못된/짝이 안 맞는 인자, 지원되지 않는 적응기 변형 및 무한이 아닌 산술 연산은 여전히 오류입니다.

유니파이드 모드에서 정확한 호환성 있는 제공된 체크포인트가 우선순위를 가집니다. 그렇지 않으면 매칭되지 않는 어댑터는 호환성 체크포인트를 삽입하는 대신 가장 가까운 이전 제공된 체크포인트에 합성적으로 융합됩니다. 이 정책은 멤버 병합으로 전달됩니다. 독립적인 체크포인트 아키텍처는 여전히 기존 RGB 정제 연쇄를 사용합니다. 이 옵션은 전체 네트워크를 변환하지 않습니다.

검사 및 출력 출처 기록 정책 `role-depth-zero-pad-crop-v1`, 소스 모듈, 타겟 텐서, 원본/타겟 형상, 유지/0-채워진/잘라낸 값 개수와 `semantic_equivalence: false`. 검사는 구조적 정보만 포함하며, 실행은 소스 해시와 유한한 값을 검증합니다. 인공적 제로는 누락된 차원을 공급하며 학습된 특징이 아니므로, 성공적인 산술 연산이 원본 LoRA 스타일 또는 생성된 이미지 품질의 보존을 인증하지 않습니다.

```sh
iild-merge --base-model anima.safetensors --additional-model sdxl-style.safetensors \
  --lora-policy synthetic --weights 0.1 --output adapted.safetensors --inspect
# 새 출력을 작성하려면 --inspect를 제거한다. 기존 파일은 덮어쓰지 않는다.
```

`ModelMergeSyntheticTests`는 패딩, 자르기, 빼기, 알파/강도 산술, 정확한 이름 차원 불일치, 소스 보존, 결정론적 검사, 기형/무한 거부 및 실제 통합 멤버 융합을 확인합니다.

<a id="package-local-vae-components"></a>

### 패키지-로컬 VAE 구성 요소

순서 있는 연쇄 단계를 통해 `"vae": "vae/anima/model.safetensors"` 를 지정할 수 있습니다. 이 선택적 상대 경로는 멤버 체크포인트에 내장된 VAE 를 명시적으로 덮어씁니다. 오래된 패키지는 내장된/ 대체 경로 선택을 유지합니다. 공유 디코더는 한 번 저장되고 여러 단계에서 참조됩니다. 패키지 리더는 포함 여부, 비어 있지 않은 파일 및 상태 동일성만 확인합니다. 텐서 호환성은 준비/생성 시 네이티브 엔진에서 확인되며, 실패는 영향을 받은 연쇄 단계를 식별합니다. 디코더는 다른 잠재적 가족과 평균화되지 않습니다. 해독된 RGB 이미지는 단계 사이의 경계로 남습니다. 수집된 구성 요소는 `vae/catalog.json` 에 함께 제공되어야 합니다. 여기에는 소스 수정 버전, 다운로드 해시, 라이선스 및 가족 별칭이 포함됩니다. 저장 디코더를 추가하는 것은 디노이저를 추가하거나 지원되지 않는 모델 아키텍처를 실행 가능하게 만들지 않습니다. Civitai 생태계 이름만으로는 잠재적 호환성을 확립하지 않습니다. (예: Illustrious XL 와 Illustrious Lumina 는 구별됩니다.)

Decoder 검증은 denoiser 없이 GPU 를 사용하여 `build/NativeVaeFallbackTests --decode-component <sd15|sd2|sd3|sdxl|anima|flux1|flux2> <weights.safetensors> <build/output-directory>` 를 통해 실행될 수 있습니다. 이는 실제 텐서 마운팅 계약을 확인하고 누락된 텐서와 비유한 출력을 거부하는 64x64 RGB 네이티브 CPU 디코드를 실행합니다. 이는 완전한 캐스케이드의 이미지 품질 평가가 아닌 decoder 스모크 테스트입니다.

DiT 변환 회귀는 다른 템플릿 해시가 거부되었다고 주장할 때 명시적으로 `checkpoint_policy="strict"`를 선택합니다. 기본 공통 계층 정책은 실험적인 예측이며 엄격한 메타데이터 ID 계약을 부과하지 않습니다.
