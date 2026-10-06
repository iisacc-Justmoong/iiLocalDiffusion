<a id="live-denoising-previews"></a>

# 실시간 노이즈 제거 미리보기

`--preview-dir /absolute/empty/temporary-directory` 를 로컬 체크포인트, 사전 설정 또는 일반 Diffusers 이미지 런너로 전달합니다. 추가 의존성이나 모델이 필요하지 않습니다: 기존 Diffusers 콜백, Torch, VAE 및 Pillow 는 실제 디노이징 단계마다 미리판을 생성합니다. 원래 샘플러 잠재 표현과 RNG 상태는 보존됩니다. 이는 추가 VAE 디코딩을 수행하므로 생성 시간을 추가합니다.

SDXL 가속기 기본값은 미리판과 최종 디코딩 모두에 기존 Diffusers VAE 타일링 경로를 활성화합니다. 번들된 VAE 타일 차원은 1024px 를 초과하여 디코더 활성화 메모리를 제한하며, 이는 샘플러의 잠재 표현이나 요청된 출력을 리사이즈하지 않습니다. 명시적 `--no-vae-tiling` 는 여전히 지원됩니다.

디렉터리는 최종 출력 디렉터리 외부에 있는 새 디렉터리이거나 비어 있어야 합니다. 프레임은 줄 바꿈으로 구분된 stdout 이벤트가 플러시되기 전에 `step-000001.png`로 원자적으로 게시됩니다.

```text
IILD_PREVIEW {"schema": "iild-preview-v1", "step": 1, "total_steps": 20, "image": "step-000001.png"}
```

스텝은 1부터 시작하며 `total_steps`는 실제 스케줄러 타임스텝을 세므로 요청한 샘플링 스텝과 다를 수 있다. 각 RGB 프리뷰는 종횡비를 유지하며 최대 변 길이는 512픽셀이다. 배치는 첫 이미지를 프리뷰한다. 이벤트는 일반 로그와 섞이므로 소비자는 부분 줄을 버퍼링하고 버전이 지정된 이벤트 접두사만 수락해야 한다. 파일명은 요청 안에서 모두 고유하므로 이미지 캐시가 연속 프레임을 숨길 수 없다. 프리뷰 정리는 호출자가 소유하며, Dreamscapes는 작업별 임시 디렉터리를 사용하고 성공·실패·취소 뒤에 제거한다. 프리뷰는 생성 성공을 뜻하지 않으며 최종 자산 매니페스트에 포함되지 않는다.

디코더는 공간형 AutoencoderKL 잠재 표현(SD/SDXL 스케일링, 평균/표준편차 정규화와 VAE 업캐스팅 포함)과 파이프라인 언패커 및 VAE 시프트를 사용하는 FLUX.1 패킹된 잠재 표현을 처리한다. 파이프라인은 `latents` 및 VAE 디코더와 함께 `callback_on_step_end`을 제공해야 한다. 지원하지 않는 파이프라인은 명시적인 오류를 보고한다. 원격 요청, 애니메이션과 여러 패스의 HiRes는 이 단일 패스 이미지 미리보기 옵션에서 지원하지 않는다.

`GenerationPreviewTests` 는 실제 텐서, 단계당 불변 PNG, 잠재/ RNG 격리, 정규화, FP16 복원, FLUX 언패킹 및 유효하지 않은 경로를 확인합니다. 설치된 Torch / Diffusers Python 환경으로 실행하세요. 소비자는 여전히 최종 프로세스 종료, 이미지 및 생성 목록을 별도로 확인해야 합니다.

`Cannot preview non-finite denoising latents` 는 샘플러가 미리보기 디코딩 전에 이미 NaN 또는 무한대를 생성했음을 의미합니다. 그것은 여전히 치명적인 실패이며, 미리보기를 건너뛰거나 유효하지 않은 값을 대체하는 것은 실패한 생성을 숨깁니다. SDXL 에서 MPS 에 대해 기본 처리기는 PyTorch SDPA 입니다. FP16 가 QK 점수를 잘라내어 넘칠 수 있기 때문입니다. 명시적인 잘라기는 점수 누적 과정을 Diffusers 를 통해 업캐스팅하면서 FP16 가중치를 유지합니다. `GenerationPreviewTests` 는 CPU 에서 유한한 FP16 입력으로 오버플로우를 재현하며, 사용 가능한 경우 MPS 를 사용하고 정밀도 정책이 적용된 후 유한한 확률을 확인합니다. 동일한 SDXL / MPS 슬라이싱 실패는
[Diffusers 문제 11229](https://github.com/huggingface/diffusers/issues/11229); SDK 정책은 로컬 텐서 회귀와 `docs/standalone-validation.md`의 실제 체크포인트 확인에 의해 적용됩니다.

콜백 계약은 공식 계약을 따릅니다.
[Diffusers 파이프라인 콜백](https://huggingface.co/docs/diffusers/using-diffusers/callback) 문서; 최종 VAE 스케일링은 설치된 Diffusers 파이프라인 코드를 따릅니다.
