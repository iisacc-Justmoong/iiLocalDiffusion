<a id="dit-standard-model-conversion"></a>

# DiT-표준 모델 변환

iiLocalDiffusion 는 실행 가능한 DiT 체크포인트에서 병합 입력을 표준화합니다. 첫 번째 구현은 Tsubaki 계열을 대상으로 합니다. SD, 1.5, SDXL 파생체 (Haruka, Hoshino 및 Illustrious 포함), 기존 DiT 체크포인트 및 별도의 Reference Pro 편집 계열은 따라서 일반 가중치 병합 전에 동일한 텐서 인벤토리, 형식 계약 및 데이터형 계약을 가진 체크포인트를 생성할 수 있습니다.

<a id="conversion-contract"></a>

## 전환계약

`--target-dit` 는 실제 Tsubaki/ DiT 체크포인트이며 결과의 실행 가능한 아키텍처입니다. 이는 파일명에서 추론된 것이 아닙니다. 그 텐서 키, 형식, 데이터형, 부동 소수점이지 않은 버퍼 및 아키텍처 메타데이터는 `iild-dit-standard-v1` 를 정의합니다. 모든 변환된 출력은 그 정확한 계약을 가지며, 동일한 템플릿에서 만들어진 다른 출력과 함께 `weighted-sum` 또는 `weighted-difference` 모드에서 `iild-merge` 에 전달될 수 있습니다.

정확한 대상 이름이 없는 소스 텐서에 대해 정책 `role-depth-stat-match-template-fill-v1` 는 구성 (디노이저/텍스트/ VAE ), 투영 역할, 블록 깊이 및 형식 거리를 기준으로 타겟을 선택합니다. 그는 앞쪽/출력 차원을 평평하게 만들고, 겹침을 잘라내며, 소스 RMS 를 대상 RMS 에 매칭하고, 겹침 바깥의 템플릿 값을 유지합니다. `--transfer-strength` 는 이 이식된 텐서를 DiT 템플릿에 블렌드하고 기본값은 `1` 입니다. 비부동 버퍼는 항상 타겟 템플릿의 것 그대로 유지됩니다.

이 정책은 결정론적이며 구조적 병합 부트스트래핑에 유용하지만 학습된 디스틸레이션은 아닙니다. 컨버터는 `semantic_equivalence: false` 를 기록하고 전체 매핑 보고서를 생성합니다. 이미지 품질과 소스 모델의 개념 보존은 실제 프롬프트로 평가해야 하며, 프로덕션 품질의 아키텍처 이식은 나중에 합성 매핑을 훈련된 Tsubaki 델타로 대체하되 동일한 출력 계약을 유지해야 합니다.

<a id="commands"></a>

## 명령

모델 페이로드를 해싱하거나 출력을 작성하지 않고 검사합니다.

```sh
iild-convert \
  --source /Models/illustrious.safetensors \
  --source-family illustrious \
  --target-dit /Models/tsubaki.safetensors \
  --output /Models/illustrious-tsubaki.safetensors \
  --inspect
```

2 정규화 체크포인트를 빌드하고 병합하세요:

```sh
iild-convert --source /Models/sd15.safetensors --target-dit /Models/tsubaki.safetensors \
  --output /Models/sd15-tsubaki.safetensors
iild-convert --source /Models/reference-pro.safetensors --source-family reference-pro \
  --target-dit /Models/tsubaki.safetensors --output /Models/reference-pro-tsubaki.safetensors
iild-merge --base-model /Models/sd15-tsubaki.safetensors \
  --additional-model /Models/reference-pro-tsubaki.safetensors --weight 0.25 \
  --output /Models/merged-tsubaki.safetensors
```

자동 감지는 SD 1.5, SDXL 및 지원되는 DiT 아키텍처에 대해 체크포인트 텐서 서명을 사용합니다. Haruka, Hoshino, Illustrious 및 Reference Pro 는 생태계 이름만으로는 신뢰할 수 있는 아키텍처 증거가 아니므로 명시적으로 선택할 수 있습니다. 개인 Tsubaki 내보내기가 식별 메타데이터를 누락한 경우 템플릿을 독립적으로 확인한 후 `--target-family tsubaki` 를 전달하세요.

<a id="safety-and-provenance"></a>

## 안전과 출처

입력은 safetensors 만 허용됩니다. 컨버터는 절대 pickle 을 로드하지 않으며, 입력을 절대 수정하지 않으며, 비유한이 전달되거나 템플릿 텐서를 거부하며, 두 입력을 해싱하고 게시 전에 재검증하며, 기존 경로를 덮어쓰지 않고 원자적으로 새 출력을 생성합니다. 출력은 소스/템플릿 SHA-256 값, 변환 정책, 패밀리 분류 및 전달 강도를 `iild_dit_conversion` 에 포함하며, 타겟의 로더 메타데이터는 유지됩니다.
