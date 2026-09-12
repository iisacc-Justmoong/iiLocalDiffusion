# 모델 리소스

이 디렉터리의 `.safetensors`와 `.pt` 원본은 Git LFS로 관리한다. 모델 파일을 Git의 일반 blob에 넣지 않으며, 저장소를 받은 뒤 다음 명령으로 원본을 내려받는다.

```sh
git lfs install --local
git lfs pull
git lfs fsck
```

`.gitattributes`가 리소스 확장자별 추적 규칙을 정의한다. LFS 포인터에는 각 원본의 SHA-256과 바이트 크기가 들어 있다. 이 파일들은 선택 가능한 모델 리소스이며 CMake가 라이브러리 또는 소비자 앱에 자동으로 포함하지 않는다. 추론 호출 시 필요한 파일을 명시적으로 선택한다.
