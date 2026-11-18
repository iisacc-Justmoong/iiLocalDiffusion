# 모델의 생성 유형 판별과 사용자 분류

`iiLocalDiffusion::ModelPackaging`의 C++23 API이다. Qt, PyTorch, Python 없이 사용할 수 있다. Checkpoint/LoRA/VAE 같은 저장 역할과 image/video/audio/language 생성 유형은 독립적이다.

`inspectModel(path)`는 디렉터리의 `model_index.json`, `config.json`, `transformer/config.json` 또는 단일 safetensors의 헤더와 공통 Society 메타데이터를 읽는다. 파일 이름에 LTX/Video가 있다는 이유만으로 비디오로 분류하지 않는다. LTX 1과 LTX 2의 파이프라인, AVTransformer3DModel, 통합 패키지 manifest와 오디오 텐서 구조를 판별한다. ImageToVideo의 Image는 입력 유형이며 출력은 video이다. LTX 2의 audio/video 출력은 image와 구별된다. GGUF v2/v3은 64 MiB 이내의 메타데이터와 텐서 설명자를 읽어 명시된 아키텍처 또는 LTX 오디오·투영·트랜스포머 서명으로 출력 유형을 판별한다. 미지의 양자화 형식은 이름 판별에서 보존하며 추론 가능하다고 단정하지 않는다.
MOSS Image는 명시된 파이프라인 클래스 또는 Safetensors의 같은 네임스페이스에 있는 임베더·출력·QKV 텐서 세 개와 서로 일치하는 2차원 크기를 근거로 image로 분류한다. 분할 인덱스에서도 동일하게 검사한다. 이름 일부만 있거나 크기가 맞지 않는 파일은 MOSS로 추정하지 않는다.
Wan, Hunyuan Video, CogVideoX, Mochi와 Stable Video Diffusion의 알려진 파이프라인·트랜스포머 클래스는 출력 유형을 video로 분류한다. Wan 체크포인트는 같은 네임스페이스의 patch embedding 및 cross-attention 텐서가 함께 있을 때도 인식한다. GGUF의 `general.architecture=wan`은 다른 아키텍처에서도 사용되므로 이것만으로 출력 유형을 추정하지 않는다.
기본 Diffusers Wan 가중치는 같은 접두사 아래의 네 가지 텐서 서명을 함께 확인하여 단일 파일과 샤드 인덱스에서도 비디오 모델로 식별한다. 일부 서명이나 파일명만으로는 판정하지 않는다.
iiLocalDiffusion 패키지의 주 모델이 GGUF 자산이면 패키지 내부 원본 GGUF의 제한된 헤더를 직접 검사한다. 선택된 주 자산의 안전한 경로·매핑·크기를 확인하고 다른 구성 요소와 GGUF의 가중치 본문을 읽지 않는다. 분류 시 저장된 체크섬을 전체 검증하지 않으며 실행 가능성을 보증하지 않는다. 모델 이름이나 보조 구성 요소의 아키텍처로 주 모델을 대신 판별하지 않는다.

독립된 `.safetensors.index.json` 모델은 인덱스와 참조된 모든 샤드를 하나의
모델로 검사한다. 인덱스의 각 텐서는 지정된 샤드에 정확히 한 번 존재해야 하며,
인덱스가 누락한 텐서, 중복 이름, 안전하지 않은 상대 경로, 잘린 헤더 및 유효하지
않은 데이터 범위는 거부한다. 검사·캐시 식별자는 인덱스와 각 샤드의 파일 상태를
포함하며, 한 샤드라도 변경되면 이전 결과를 재사용하지 않는다. 심볼릭 링크가
가리키는 대상도 식별자에 반영한다. 헤더 검사는 모델 실행 또는 출력 품질을 보증하지 않는다.

`probeModelTensorNames`, `modelContainsTensorNames`와 `findModelTensorNames`는 Safetensors, 표준 샤드 인덱스 또는 GGUF에 요청한 원본
텐서 이름이 모두 있는지 메타데이터만 읽어 확인한다. 손상된 파일과 검사 중 변경된
파일은 오류이며, 이름의 존재는 차원·양자화·실행 호환성을 보증하지 않는다.
CLI `contains-tensors --input MODEL --tensor NAME`과 Python `contains_tensors`,
`find_tensor_names`도 같은 검사를 제공하며 일치한 이름과 SDK 패키지 메타데이터 존재 여부를 반환한다.

일반 설정 JSON은 각각 1 MiB, 분할 인덱스와 각 safetensors 헤더는 64 MiB, 전체 인덱스·샤드 메타데이터는 256 MiB 이하로 제한한다. 가중치 본문을 읽거나 전체 SHA-256을 계산하지 않는다. 모델 및 메타데이터의 파일 식별자·크기·수정 시각으로 최대 128개 결과를 캐시한다. POSIX에서는 inode와 ctime도 비교한다. 반복 조회의 `cacheHit`는 true이고 `metadataBytesRead`는 0이다. 파싱 실패·로컬 미제공 상태는 `diagnostic`으로 반환하며, 미확인 모델을 확인된 아키텍처로 표시하지 않는다. `detectModelContainer(path)`와 CLI `probe`·Python `detect_container`는 파일 내용으로 safetensors 또는 GGUF 후보를 먼저 판별한다. 표준 `.safetensors.index.json` 이름은 인덱스 구조를 읽어 구분한다. 이 결과는 형식 후보이며 전체 헤더, 텐서, 추론 호환성 검증이 아니다.

자동 판별 결과는 `detectedModalities`, 사용자 선언을 적용한 결과는 `modalities`이다. 우선순위는 내장 메타데이터 → 확장자를 제거한 `.model.json` → 원래 파일명 뒤의 `.model.json`이다. 디렉터리는 `society.model.json`을 사용한다. 기존 `society.modality` 문자열을 지원하고, 복수 유형은 `society.modalities` 배열로 표현한다. safetensors의 문자열 메타데이터에 있는 `society.modality`도 지원한다.

```json
{
  "society": {
    "type": "Checkpoint",
    "modalities": ["image", "video"]
  }
}
```

`setModelModalities(path, {"video"})` 또는 `{"image", "video"}`로 저장한다. 빈 목록은 `modality: "auto"`로 자동 판별을 복원한다. 기존 라이선스·설명·저장 역할 등은 보존한다. 파일은 원래 이름 뒤의 `.model.json`, 디렉터리는 `society.model.json`에 기록하며 가중치 파일과 위치는 바꾸지 않는다. 파일 생성·동시 변경 검사는 iiFileProvider Native의 create/update가 담당한다.

```sh
iild-model-package inspect --input '/path/model.safetensors'
iild-model-package classify --input '/path/model.safetensors' --modalities video
iild-model-package classify --input '/path/model.safetensors' --modalities image,video
iild-model-package classify --input '/path/model.safetensors' --modalities auto
```

사용자 분류는 검색·선택 가능 유형에 대한 선언이다. 실제 추론 엔진이 지원하지 않는 아키텍처를 변환하지는 않으며, 실행 시에는 해당 엔진이 호환성을 다시 검증한다. 로컬에 없는 가중치를 분류 변경하기 위해 자동 다운로드하지 않는다.

## 이름이 변경된 분할 인덱스

표준 접미사가 없는 단독 인덱스도 JSON의 `weight_map`과 실제 참조 샤드의
Safetensors 콘텐츠로 판별한다. 인덱스의 선택 경로를 기준으로 상대 샤드를
해석하며, 이름만으로 PyTorch 샤드를 Safetensors로 취급하지 않는다. 인덱스
구조와 샤드 헤더·텐서 매핑의 전체 검증은 기존 검사기가 수행한다. C++, Python
및 네이티브 이미지·비디오 로딩은 같은 논리적 경로와 샤드 식별자를 사용하고,
Dreamscapes는 완전한 그룹을 모델 하나로 표시한다. 검사와 합성 픽스처를 이용한
로더 테스트는 가중치의 실행 가능성이나 실제 생성 품질을 보증하지 않는다.
Python 생성 워커는 `iild_image_container_probe_v1` C ABI로 동일한 판별기를
호출하며, CLI `probe`와 Python 패키징 API `detect_container`도 같은 결과를 쓴다.
Wan 디렉터리의 변환기·인코더·VAE 구성 요소도 같은 기준으로 분할 인덱스를
찾으며, 인덱스와 각 샤드를 실행 전후의 원본 식별 검사에 포함한다.

## PyTorch 체크포인트 후보

형식 후보 판별은 일반 ZIP과 구분되는 PyTorch 아카이브의 `data.pkl` 항목 또는
PyTorch 저장 형식의 pickle 매직을 살핀다. 접미사가 달라도 같은 후보로 식별하며,
이 단계에서는 pickle을 실행하거나 모델의 아키텍처와 출력 유형을 추정하지 않는다.
완전한 구조와 텐서는 생성 워커의 안전한 weights-only 검사 및 변환에서 검증한다.
ONNX는 이 모델 실행 경로에서 지원하지 않는다.
ZIP64도 확인하며 ZIP 꼬리는 최대 65,557바이트, 중앙 디렉터리는 최대 64 MiB와
100만 항목으로 제한한다. 검사는 파일 이름만 읽고 가중치 본문은 읽지 않는다.

## PyTorch 분할 인덱스

인덱스의 상대 경로와 샤드 서명을 검사하여 `pytorch-index` 후보를 식별한다.
인덱스와 모든 샤드의 논리 경로·실제 대상·식별자를 함께 보관하고 변경을 감지한다.
이 단계에서는 pickle을 해석하지 않으며 샤드의 텐서 내용이나 실행을 보증하지 않는다.
