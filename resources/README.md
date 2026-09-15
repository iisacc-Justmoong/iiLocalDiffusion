# 모델 리소스

이 디렉터리의 `.safetensors`와 `.pt` 원본은 Git LFS로 관리한다. 모델 파일을 Git의 일반 blob에 넣지 않으며, 저장소를 받은 뒤 다음 명령으로 원본을 내려받는다.

```sh
git lfs install --local
git lfs pull
git lfs fsck
```

`.gitattributes`가 리소스 확장자별 추적 규칙을 정의한다. LFS 포인터에는 각 원본의 SHA-256과 바이트 크기가 들어 있다. `generation-defaults.json`에 지정된 LoRA와 네거티브 임베딩은 이제 전역 이미지 생성 기본값이며 CMake 설치에 포함된다. SDXL에서는 임베딩 7개와 기본 강도 1.0의 폴백 LoRA를 사용한다. 자세한 우선순위와 소비자 배포는 [전역 기본값 문서](../docs/generation-defaults.md)에 있다.

다른 모델 계열의 LoRA 기본값은 `fallback_loras` 배열에 호환 파일·강도·계열·해시·크기를 등록한다. 기존 SDXL용 `fallback_lora` 항목과 함께 사용할 수 있다. 이미지와 Deforum, 네이티브 런타임이 동일한 계열 선택 규칙을 사용한다. SDXL 가중치의 계열 이름만 바꾸는 것은 변환이 아니며, 각 기반 모델에 맞는 실제 LoRA 파일이 필요하다.

`embeddings/`는 `.pt` 원본 3개를 텐서 전용 safetensors로 보존한 사본이다. 원본 변경 시 `scripts/prepare_generation_defaults.py`로 다시 만들고 `--check`로 벡터 동일성과 해시를 검증한다. 원본 `.pt` 파일도 계속 보존한다.

`vae/`에는 Qwen Image RGB, SDXL, FLUX.1, FLUX.2의 원본 VAE와 설정을 포함한다. `fallback_vae`(기존 Qwen)와 `fallback_vaes` 배열이 고정 리비전·SHA-256·크기·호환 계열을 기록한다. 네이티브/Diffusers는 VAE가 빠진 모델에 해당 계열의 VAE를 연결하며 기본 LoRA를 꺼도 유지한다. 내장 VAE가 우선하며 서로 다른 잠재 공간을 혼용하지 않는다. [기본값 문서](../docs/generation-defaults.md#vae-자동-폴백)에 재다운로드와 검증 명령이 있다. `.cache`는 Git/설치에서 제외한다. Qwen/FLUX는 Apache-2.0 원문을 보관하고 SDXL은 MIT를 선언한 공식 모델 카드를 보존한다. FLUX.1 원본과 공개 사본의 동일성 근거는 `vae/flux1/NOTICE.md`에 있다.
