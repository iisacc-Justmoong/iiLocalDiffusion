# Windows 검증

Windows에서는 네이티브 파일 핸들로 모델 소스의 최종 경로를 구한다. 심볼릭 링크의 메타데이터 대신 대상 파일의 크기와 수정 시각을 검사하며, 패키지에는 사용자가 선택한 논리적 파일 이름을 유지한다. 패키지 생성·검증·추출 테스트는 실제 파일 링크를 사용하여 내용과 추출 이름을 확인한다.

현재 검증 구성은 MSVC x64 Ninja와 같은 컴파일러의 vcpkg 의존성을 사용하고 `build/`에 빌드한다. MLX CPU 구성과 이번 실행 범위는 WindowsBuildAndRun.md에 기록한다. `ctest --test-dir build --output-on-failure`로 네이티브 엔진, 모델 포맷과 설치 소비자 검증을 실행한다.

최신 구성·실행 결과와 복구 조치는 [Windows 빌드 및 실행](WindowsBuildAndRun.md)에 기록한다.
