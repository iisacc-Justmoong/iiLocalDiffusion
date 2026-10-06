# Windows 실행 검증

패키지 검증과 원본 스캔에 동일한 Windows 핸들 기반 최종 경로 해석을 적용한다. 일반 파일과 심볼릭 링크 모두 패키지 작성 후 바이트 검증 및 추출 테스트를 수행한다.

Windows에서는 iild-generate.exe, iild-merge.exe, iild-convert.exe가 함께 설치된다. 네이티브 런처가 UTF-16 인수, 공백 및 따옴표, 표준 스트림과 자식 종료 코드를 보존하여 Python 엔트리를 실행한다. WindowsPythonLauncherTests가 이를 실제 자식 프로세스로 검증한다. IILD_PYTHON_EXECUTABLE로 로컬 Python 실행 파일을 선택할 수 있다. 패키징 변경 감지는 파일 동일성, 크기, 수정 시간을 구분하여 진단한다.

네이티브 런처는 Python을 suspended 상태로 생성하여 Windows Job Object에 먼저 배정한다. 런처 종료 시 남은 자식 프로세스도 함께 종료된다.

CPU 네이티브 checkpoint 엔진은 IILD_ENABLE_NATIVE_DIFFUSION=ON으로 빌드한다. 고정된 stable-diffusion.cpp 및 ggml을 사용하며 Qt MinGW 헤더에 없는 Windows 10 thread power ABI를 해당 백엔드에만 제공한다. WindowsThreadPowerTests는 ABI 크기와 실제 SetThreadInformation 호출을 검증한다. 실제 모델 가중치가 필요한 추론 테스트와 모델 없는 엔진 계약 테스트를 구분한다.

VAE 설정의 캐시 동일성에는 검증된 JSON 내용의 해시를 포함한다. 같은 크기와 수정 시간을 유지한 설정 변경도 NativeVaeConfigTests에서 검증한다. 네이티브 입력·ControlNet fixture의 경로는 canonical 표현으로 비교하여 Windows 구분자 차이를 제거한다.

병렬 로그 fixture는 thread-local 정수 식별자와 호출별 지역 문자열로 8개 스레드의 16,000개 메시지를 검사한다. MinGW의 동적 thread-local 문자열 수명에 대한 의존을 제거하면서 실제 로그 callback의 메시지 일관성 검사를 유지한다.
