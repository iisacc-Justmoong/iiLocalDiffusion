# Windows 모델 패키지 검증

`iild-model-package`의 inspect/classify와 materialize는 다른 명령과 마찬가지로 최종
JSON 보고서를 명시적으로 flush한다. 파이프로 실행할 때 종료 코드가 0이어도 최종
보고서가 비어 있던 현상을 방지한다. 패키지 검사는 보고서가 없을 때 종료 코드와
표준 출력을 함께 진단하며, 동일 패키지를 16회 materialize·inspect하여 성공 결과가
매번 완전한 JSON인지 확인한다. 모델 바이트·체크섬·캐시 복구 검사를 유지한다.

40GiB 헤더 전용 검사는 Windows에서도 실제 sparse 파일을 사용한다.
`FSCTL_SET_SPARSE`를 파일 크기 확장 전에 호출하고, 0으로 채우는 CRT truncate 대신
`SetFilePointerEx`·`SetEndOfFile`로 끝을 확장한다. sparse 속성·40GiB 논리 크기와
1MiB 미만의 실제 할당량을 검사한다. 따라서 검사 자체가 일반 파일 공간을 할당하지 않으면서,
검사기의 메타데이터 읽기 한도가 4096바이트 미만임을 그대로 확인한다.
[Microsoft의 sparse 파일 제어 계약](https://learn.microsoft.com/en-us/windows/win32/api/winioctl/ni-winioctl-fsctl_set_sparse)을 따른다.

POSIX SIGTERM 취소 검사 하나는 기존대로 Windows에서 실행하지 않는다.
이 항목은 CTest 성공 개수와 별도로 실행 범위를 기록한다.
이미지와 동영상 Python 어댑터는 같은 네이티브 DLL 로더를 사용한다. Windows에서는
Python의 기본 ctypes 로더가 PATH를 자동으로 따르지 않으므로 SDK가 구성한 DLL
디렉터리를 로드 구간에 명시적으로 등록하고 반환 시 해제한다. 실제 C ABI 호출 검사는
이미지 처리 패키지 Pillow도 설치한 상태에서 실행한다. 대형 모델 추론 완료와는 구분한다.

새 일괄 이미지 출력 경로는 추론 전에 생성하지 않는다. 비공개 stage에서 PNG·해시·보고서가
모두 확인된 뒤 완성된 폴더를 한 번에 게시하며, 실패하면 공개 출력 경로도 생성하지 않는다.
이 Windows 검증은 새 출력 경로를 사용한다. 기존 비어 있지 않은 사용자 폴더는 계속 거절한다.
이미지 fixture의 임시 폴더 정리는 provider를 사용하여 비동기 미리보기 작업이 파일 핸들을
반환한 뒤 삭제한다.
BackendRegistry의 동반 모델 변경 실패 검사도 새 출력 폴더가 게시되지 않는지 확인한다.
모델 캐시 변경 식별은 설치된 제공자의 NTFS 변경 순번을 사용한다. 같은 크기·같은
파일 시각의 빠른 외부 덮어쓰기를 놓치지 않는 회귀 검사와 함께 검증한다.
