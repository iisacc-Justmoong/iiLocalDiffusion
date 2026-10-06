<a id="local-comfyui-generation"></a>

# 로컬  ComfyUI  생성

`reference/generate.py --backend comfyui`  는 기존 루프백  ComfyUI  서버에 대해 명시적인  API  워크플로우를 실행합니다. 모델 아키텍처, 양자화/  GGUF  로더, 분할 컴포넌트, 비디오, 오디오 및 메시는 설치된 노드에 의해 처리됩니다. 클라이언트는 추가 패키지 의존성을 추가하지 않으며 노드나 가중치를 설치하지 않습니다.

ComfyUI **파일 → 내보내기(API)**를 사용하여 워크플로를 내보냅니다. `nodes`/`links`가 포함된 비주얼 편집기 JSON가 거부되었습니다. API JSON는 노드 ID를 `class_type` 및 `inputs`에 매핑합니다. 대기열 없이 이용 가능 여부를 확인하세요.

```bash
python3 reference/generate.py --backend comfyui \
  --workflow /absolute/path/workflow-api.json --validate-only
```

워크플로는 모든 구성 요소, 작업 입력, LoRA, 샘플링 및 내보내기를 결정합니다. 노드 ID로 명시적으로 입력을 설정합니다. 프롬프트 노드는 절대 추측되지 않습니다:

```bash
python3 reference/generate.py --backend comfyui --base-model Illustrious \
  --workflow /absolute/path/workflow-api.json \
  --workflow-inputs '{"6":{"text":"a lighthouse at sunrise"},"3":{"seed":42}}' \
  --output-dir build/reference/illustrious-workflow
```

ID `6`/`3`는 제공된 워크플로에 있어야 하는 예입니다. `--workflow-inputs @/path/values.json`는 파일에서 동일한 객체를 읽습니다. 기존 입력만 재정의할 수 있습니다. 로컬 모델 이름은 서버의 인벤토리와 일치해야 합니다. 이미지/비디오 입력은 해당 로컬 서버에 이미 존재해야 합니다.

`--print-config` 는 오프라인 상태입니다. `--validate-only` 는 라이브 노드 가용성, 필요한 입력, 모델/열거형 선택, 그래프 링크 및 사이클을 확인합니다. `/prompt` 는 서버의 최종 검사를 수행합니다. 성공적인 사전 검사 는 생성 결과가 아닙니다. ComfyUI 로 식별된 호스팅된 API 노드는 거부됩니다. 엔드포인트는 localhost/루프백으로 제한되며, 프록시 변수와 리디렉션은 모델 프롬프트를 리디렉션할 수 없습니다.

`submission.json` 는 프롬프트 ID, 패치된 워크플로우 및 그 SHA-256 를 성공적으로 완료된 `/history/{prompt_id}` 엔트리가 대기하는 동안 기록합니다. 노드 오류, 빈 출력, 시간 초과 및 다운로드 오류는 실패합니다. 시간 초과가 잠재적으로 실행 중인 작업을 취소하거나 재제출하지는 않습니다; 기록된 프롬프트 ID 를 ComfyUI 에서 확인하세요.

출력 노드가 반환하는 이미지/비디오/오디오/메시 파일 설명자는 `/view` 를 통해 다운로드됩니다. 경로는 확인되며, 스트리밍은 크기 제한이 있고 파일은 원자적으로 게시됩니다. 모든 아티팩트가 도착한 후에만 `generation.json` 가 작성되며, 해시와 크기를 포함합니다. 새 또는 빈 출력 디렉토리는 기존 사용자 파일을 유지합니다.

기본값: `http://127.0.0.1:8188`, 3600초 작업 시간 초과, 1 초 폴링, 2 GiB 아티팩트당 제한. `--timeout`, `--poll-interval`, 및 `--max-artifact-bytes` 는 이러한 제한을 노출합니다. 요청은 한계가 설정된 에서 30 초 또는 더 짧은 작업 시간 초과입니다. 출력 저장소는 `build/` 아래 기본값입니다.

기본 모델 레이블은 출처로 선언되며, 클라이언트는 서버 측 텐서를 검사하거나 해당 이름의 모델을 로드한 워크플로우를 증명하지 않습니다. HTTP 통합 테스트는 픽스처 서버를 사용하여 사전 검사, 대기열, 완료, 바이트, 해시, 오류 및 기존 파일 보존을 확인합니다. 모든 모델 또는 플러그인에 대한 실제 생성 지원이 있음을 증명하지는 않습니다.

프로토콜 소스: [서버 경로](https://docs.comfy.org/development/comfyui-server/comms_routes) 및 [공식 API 예제](https://github.com/Comfy-Org/ComfyUI/tree/master/script_examples).
