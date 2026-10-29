# 자동 Pose 모델 경로 폐기

ONNX 기반 자동 포즈 추론과 런타임 다운로드·연결을 제거하였다. `nativePoseAvailable()`은
`false`를 반환하며 `NativePoseSession`과 `processNativePose`는 실행 전에 명시적 폐기
오류를 반환한다. 모델 파일을 읽거나 다른 엔진으로 암묵적으로 우회하지 않는다.

공개 인터페이스는 기존 소비자의 실패 처리와 향후 독립 구현을 위해 유지한다.
포즈 좌표 변환·letterbox·NMS·SimCC 해석과 RGB 힌트 그리기의 순수 C++ 계산은
`PoseProcessing`으로 분리되어 있다. 이 계산만으로 학습된 모델 추론을 제공하지 않는다.
기존 DWPose 파생 계산의 출처·라이선스는 `docs/licenses/DWPose.txt`에 보존한다.

`NativePoseTests`는 가용성이 false인지와 모델 요청이 파일 접근 전에 거절되는지를
확인한다. `PoseProcessingTests`는 독립적인 수치 계산 규약을 검사한다.
