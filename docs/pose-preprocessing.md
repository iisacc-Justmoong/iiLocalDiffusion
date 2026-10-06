<a id="native-pose-preprocessing"></a>

# 네이티브 포즈 전처리

Figma `237:6488` 프로세스=포즈 제어는 실제 이미지에서 포즈로 추론이 필요하며, 원래 RGB 이미지를 ControlNet 힌트로 전달하는 것이 아님. 이는 SDK 에 제품 리소스 선택, 불변 큐 스냅샷 및 ControlNet 전달과 연결된 네이티브 DWPose 경로가 포함되어 있음. 이는 최소한의 운영 통합이며, 실제 훈련된 모델의 품질과 상세한 동작은 유보됨.

<a id="pipeline-and-boundaries"></a>

## 파이프라인 및 경계

- `NativePoseSession`는 인라인 YOLOX-L 및 DWPose ONNX 파일과 2 CPU 추론 세션을 보유하고 있습니다. `process`는 동일한 크기의 소유된 RGB 포즈 힌트를 반환합니다.
- `PoseProcessing` 은 사적인, Qt / OpenCV / Python -없는 C++23 수치 경계입니다: 640x640 상단 왼쪽 YOLOX 레터박스, BGR   CHW 탐지기 입력, 사람 전용 그리드 디코딩 및 NMS , 사람별 1.25x 아핀 컷, 정규화된 RGB 포즈 입력, 133-관절 SimCC 디코딩, 합성 목과 OpenPose 신체 리매핑, 검은 캔버스 위의 컬러 신체/손 팔다리와 얼굴 랜드마크입니다.
- 감지기 출력은 `[1,8400,85]`이며, 8/16/32입니다. 사람 점수는 0.3보다 크며, IoU NMS 0.45를 포함합니다. 포즈 입력은 `[1,3,256,192]` 또는 `[1,3,384,288]`이며, 출력은 x/y SimCC 텐서 `[1,133,2*width]`, `[1,133,2*height]`이며, 분할 비율 2입니다.
- RGB 크롭 규칙은 MMPose의 `bgr_to_rgb=True` 학습 전처리기를 따르며, 이는 `[123.675,116.28,103.53]`, 표준 `[58.395,57.12,57.375]`입니다. 이는 원래 BGR 예제와 RGB Gradio 호출자에서 발생하는 색상 규칙이 일관되지 않는 문제를 해결합니다. 실제 체크포인트 패리티는 아직 확립되지 않았습니다.
- 검출이 없을 경우 검은 힌트가 생성되며, 조작된 전체 이미지 사람이 아닙니다. 최대 64 명의 사람과 한 변당 최대 2048 개의 이미지가 허용되며, 이 한도를 초과하면 아무런 알림 없이 people 를 삭제하는 대신 실패합니다. 검출자/사람 순서는 결정론적으로 유지됩니다. 발은 렌더링되지 않으며 DWPose의 ControlNet 시각화와 일치하며, 신체, 손과 얼굴은 렌더링됩니다.
- 래스터화는 OpenCV의 폴리곤 루틴이 아니라 잘라낸 분석 타원/원 및 세그먼트 거리를 사용합니다. 색상/토폴로지 계약은 테스트됩니다; OpenCV 렌더링과의 정확한 픽셀 패리티는 주장되지 않습니다.

<a id="memory-and-lifetime"></a>

## 기억과 수명

모델 파일은 표준적인 로컬 정규 `.onnx` 파일이며, 취소 가능한 조각으로 완전히 읽혀 소유된 익명 바이트 배열에 저장되고, 전후 메타데이터 동일성 검사가 수행됨. ONNX 세션은 **해당 배열**에서 생성되며, 파일 이름이 아님. 배열과 세션은 호출과 일시 정지 동안 소유자에서 계속 유지됨. 메모리 압박/비활성 제거, mmap, 모델 다운로드, 변환 캐시, 또는 직렬화된 최적화 모델이 도입되지 않음. OS 는 일반적으로 익명 메모리를 압축/페이지할 수 있음.

`OnnxInlineModel` 는 세션을 생성하기 전에 한계가 설정된 프로토obuf 와이어 필드를 순회하며, 초기화자, 속성, 중첩 그래프, 함수, 희소 텐서 및 학습 그래프에서 외부 데이터/데이터 위치 필드를 거부함. 절대 경로가 존재하더라도 외부 사이드카 텐서는 금지되며, 배열 기반 세션만 해당 불변식의 증거로 의존되지 않음. ONNX 런타임 는 여전히 의미론적 그래프 유효성 검사를 수행함. 내선 모델 파일은 비어있지 않고 2 GiB 미만이어야 함.

`processNativePose` 는 절대 모델 쌍과 스레드 수에 의해 키링된 프로세스-런타임 캐시에 세션을 유지하며, 명시적 `releaseNativePoseCache`, `releaseNativeDiffusionCache` 또는 해체 때까지 유지됨. 캐시 히트는 소스 파일을 다시 통계하거나 읽지 않으며, 동일한 경로에서 파일을 교체하려면 명시적 해제 필요. 비활성화, 일시정지 및 취소는 로드된 모델을 제거하지 않습니다. 호출은 취소 가능한 잠금 장치를 통해 직렬화됩니다. `NativeExecutionControl` 는 실행/작물 작물 사이에 주차하며, 이미 활성화된  ONNX 실행은 일시정지 시 완료됩니다. 취소 요청은  ORT  RunOptions 종료를 수행하며, 취소된 호출은 세션을 파괴하지 않습니다. 세션 초기화는  ORT 내부에서 중단될 수 없으며, 파일 읽기 주변과 동안 취소가 확인됩니다.

<a id="build-and-execution-provider"></a>

## 빌드 및 실행 공급자

`IILD_ENABLE_POSE=ON` 는 선택적  CPU   ONNX 경로를 활성화하며, 기본값은  OFF 입니다. 현재  macOS   ARM64 빌드는  ONNX   런타임   1.30.0 를  `build/_deps` 로 다운로드하며,  SHA-256 로 고정됩니다. 다른 타겟은 명시적인 타겟 네이티브  `IILD_ONNXRUNTIME_ROOT` 를 필요로 하며,  iOS 와 기타 패키징된 플랫폼은 검증되지 않았습니다. 시스템/전역  런타임 가 설치되어 있지 않습니다. ONNX 제공자 선택은 명시적  CPU 이며, 확산 엔진의  Metal 백엔드에서 무음  대체 경로 이 아닙니다. 네이티브  CPU 인트라-오프 연산 스레드는 하드웨어 동시성으로 기본이며, 인터-오프= 1 ; 비활성화 스레드 회전은 비활성화됩니다. CoreML 컴파일된 모델 캐시는 사용되지 않습니다.

```
env -u CPATH -u CPLUS_INCLUDE_PATH cmake -S . -B build -DIILD_ENABLE_POSE=ON
env -u CPATH -u CPLUS_INCLUDE_PATH cmake --build build --target iiLocalDiffusion PoseProcessingTests NativePoseTests NativePoseDisabledTests -j 4
ctest --test-dir build -R '^(PoseProcessingTests|NativePoseTests|NativePoseDisabledTests)$' --output-on-failure
```

이전 오프라인 빌드에 첫 번째 종속성 다운로드가 필요한 경우 임시로 `FETCHCONTENT_FULLY_DISCONNECTED=OFF`를 구성한 다음 채운 후 복원합니다. 패키지 소비자는 ORT 공유 라이브러리 및 라이센스 고지 사항을 휴대해야 합니다. 빌드 가용성은 패키지 또는 설치된 앱 업데이트의 증거가 아닙니다.

<a id="verification-scope-and-remaining-work"></a>

## 검증 범위 및 남은 작업

수치적 테스트는 색상 정렬, 레터박스, 모든 검출기 스트라이드, 다중 사람 NMS, 좌표 변환, SimCC 신뢰도, 목/몸체 재매핑, 손/얼굴, 검은 배경 및 손상된/무한 입력을 다룹니다. 네이티브 테스트는 알려진 텐서 출력을 가진 작은 ONNX 그래프를 구성하고 실제 C++ ORT API 를 실행합니다. 그들은 2 독립적으로 위치된 사람, 형상 거부, 취소, 일시정지/재개 및 픽스처 모델 파일 두 개가 모두 제거된 후 반복 추론을 확인합니다. 이 픽스처 는 **학습된 모델이 아니며** 검출기 정확도, 생성 품질, 실제 모델 처리량 또는 설치된 앱 동작을 증명하지 않습니다.

2026-09-29에서 공유 SDK 및 모든 3 포즈 테스트 타겟이 성공적으로 구축되었습니다. 26 선택된 네이티브 스위트 (포즈, 매개변수, 메모리 정책, ControlNet, IP -어댑터, 리파이너, 디테일러, 업스케일링 및 조건부) 는 24.40 초에 통과했습니다. 증거: `build/pose-red.log`, `build/pose-build.log`, `build/pose-tests.log` 및 `build/pose-native-regression.log` 입니다. 구현 전에 빨간 수치 계약이 실패했습니다. SDK 설치/준비 또는 Dreamscapes 재설치가 수행되지 않았습니다.

최소 통합은 `poseDetector` / `poseModel` 리소스, 원자적 제품 선택, 큐 포워딩, 런타임 유지 및 기존 ControlNet API 에 전달된 포즈 힌트를 추가합니다. 테스트는 소스 제거 후 캐시 재사용, 명시적 해제, 큐 리소스 포워딩 및 피커 취소를 다룹니다. 실제 YOLOX/DWPose 체크포인트 정확도, 이미지 동등성 및 엔드투엔드 학습된 모델 생성은 검증되지 않았으며 사용자의 최소 작업 범위 하에서 연기됩니다.

<a id="minimum-integration-verification-2026-09-29"></a>

### 최소 통합 검증(2026-09-29)

공유 SDK 및 영향 받은 테스트 대상이 성공적으로 재구성되었습니다. 6 선택된 스위트는 28.78 초 내에 통과했습니다: PoseProcessing, NativePose, 비활성화된 Pose, 매개변수, 네이티브 결과 전달 및 그 모바일 계약 변형. SDK 는 제품 빌드를 위해 `build/install` 하에 준비되었으며, ONNX 런타임 및 라이선스를 포함합니다; 전역 SDK 설치에는 변경이 없습니다. 로그: `build/pose-integration-build.log`, `build/pose-integration-tests.log`, `build/pose-integration-stage.log`.

<a id="provenance"></a>

## 출처

Tensor/pose 규약은 [DWPose의 ONNX 구현](https://github.com/IDEA-Research/DWPose/tree/3dca5db79d9f9ffdd378753ddf6ec66535aace88/ControlNet-v1-1-nightly/annotator/dwpose) 및 [MMPose 구성](https://github.com/IDEA-Research/DWPose/blob/3dca5db79d9f9ffdd378753ddf6ec66535aace88/mmpose/configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py)을 참조합니다. 수정된 네이티브 구현; 원래 귀속/라이선스는 `docs/licenses/DWPose.txt` 에 유지됩니다. ONNX 런타임 의 라이선스 및 제 3 자 고지사항은 활성화될 때 공유 라이브러리와 함께 설치됩니다. 모델 가중치 권리는 소스/ 런타임 라이선스와 별개이며, 여기에 사전 학습된 가중치가 포함되지 않습니다.
