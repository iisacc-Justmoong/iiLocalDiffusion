# 모델의 생성 유형 판별과 사용자 분류

`iiLocalDiffusion::ModelPackaging`의 C++23 API이다. Qt, PyTorch, Python 없이 사용할 수 있다. Checkpoint/LoRA/VAE 같은 저장 역할과 image/video/audio/language 생성 유형은 독립적이다.

`inspectModel(path)`는 디렉터리의 `model_index.json`, `config.json`, `transformer/config.json` 또는 단일 safetensors의 헤더와 공통 Society 메타데이터를 읽는다. 파일 이름에 LTX/Video가 있다는 이유만으로 비디오로 분류하지 않는다. LTX 1과 LTX 2의 파이프라인, AVTransformer3DModel, 통합 패키지 manifest와 오디오 텐서 구조를 판별한다. ImageToVideo의 Image는 입력 유형이며 출력은 video이다. LTX 2의 audio/video 출력은 image와 구별된다.

JSON 파일은 각각 1 MiB, safetensors 헤더는 64 MiB 이하로 제한한다. 가중치 본문을 읽거나 전체 SHA-256을 계산하지 않는다. 모델 및 메타데이터의 파일 식별자·크기·수정 시각으로 최대 128개 결과를 캐시한다. POSIX에서는 inode와 ctime도 비교한다. 반복 조회의 `cacheHit`는 true이고 `metadataBytesRead`는 0이다. 파싱 실패·로컬 미제공 상태는 `diagnostic`으로 반환하며, 미확인 모델을 확인된 아키텍처로 표시하지 않는다.

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
