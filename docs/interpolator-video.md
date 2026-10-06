<a id="interpolator-video-generation"></a>

# 보간기 비디오 생성

`iild-generate --backend interpolator` 는 텍스트 조건부 및 2 엔드포인트 간의 초기 노이즈를 보간하여 H.264 MP4 또는 애니메이션 GIF 를 생성합니다. 동등한 프리셋 러너/ Python 설정은 `animation_mode="Interpolator"` 입니다. 모든 프레임, 양쪽 엔드포인트 포함, 텍스트에서 이미지로의 확산 패스를 완전히 받습니다. 이는 프롬프트/시드 보간이며 광학 흐름 프레임 삽입이나 이미지 크로스 페이드가 아닙니다. Deforum 의 카메라/이전 프레임 피드백은 별도의
[2D 모드](deforum-video.md).

보간기는 **FPS <= 12 또는 GIF 출력**로 제한됩니다. 기본값은 12 FPS입니다. 기타 자동 비디오 요청은 24 FPS에서 로컬 LTX를 사용합니다. 참조
[공유 비디오 선택 및 GIF 규칙](local-model-generation.md#images-and-animations-from-the-same-image-model).

이 가이드는 독립형 **이미지 모델 프롬프트/시드 애니메이션**를 다룹니다. 일반 LTX 비디오는 이미 생성된 시간적 시퀀스에 프레임을 삽입하기 위해 [포스트 LTX 프레임 인터폴레이터](temporal-video.md#ltx-followed-by-frame-interpolation) 를 사용합니다. 두 번째 단계는 FFmpeg 운동 보상을 사용하며 LTX 소스 프레임만 필요로 하고 최종 FPS 12이상에서 실행됩니다. 이 가이드의 이미지 모델 파이프라인을 로드하지 않습니다.

이 방법은 공식의 프롬프트/시드 의미를 따릅니다.
[DiffusionBee 인터폴레이터](https://github.com/divamgupta/diffusionbee-stable-diffusion-ui/blob/master/backends/stable_diffusion/applets/frame_interpolator.py)입니다. `N` 의 `i` 프레임에 대해 `t = i / (N - 1)` 입니다. 모든 활성 긍정, 부정 및 풀링 프롬프트 텐서는 `(1-t)*A + t*B` 를 사용합니다. 초기 노이즈는 `sqrt(1-t)*A + sqrt(t)*B` 를 사용하여 독립적 가우시안 엔드포인트를 위한 단위 기대 분산을 보존합니다. 동일한 노이즈는 변하지 않게 유지되어 시드가 모두 같을 때 진폭 급증을 피합니다. 프레임 제로와 프레임 `N-1` 는 정확한 엔드포인트 입력을 유지합니다. 독립적으로 렌더링된 이미지입니다; 시드 변경은 구성을 상당히 바꿀 수 있으며 운동 연속성을 보장하지 않습니다.

<a id="usage"></a>

## 사용법

기존 고정된 Diffusers Python 환경에 FFmpeg 와 FFprobe 를 PATH 에 사용하세요. FFmpeg 는 `libx264` 를 노출해야 하며, MP4 에 대한 GIF 인코더와 팔레트 필터를 GIF 에 사용해야 합니다. 인터폴레이터는 새로운 Python 의존성, OpenCV 및 인터폴레이션 모델 다운로드가 필요 없습니다.

```sh
reference/diffusers/.venv/bin/python reference/generate.py \
  --backend interpolator --preset sd15-compatible \
  --model /absolute/path/to/diffusers-model --local-files-only \
  --prompt 'a city at sunrise' --end-prompt 'a forest at sunrise' \
  --negative-prompt 'blur, low quality' \
  --seed 42 --end-seed 43 --max-frames 120 --fps 12 --steps 20 \
  --output build/interpolator.mp4
```

프롬프트 전용 변경을 위해 양쪽 끝에서 동일한 시드를 사용하거나 `--end-seed` 를 생략하세요. 시드 전용 변경을 위해 양쪽 끝에서 동일한 프롬프트를 사용하거나 `--end-prompt` 를 생략하세요. 양쪽 끝 값을 의도적으로 생략하면 결정론적 스케줄러를 가진 일정한 시퀀스가 생성됩니다. `--max-frames` 는 양쪽 끝점을 포함하며 2 와 1,000,000사이여야 합니다. 지속 시간은 `max_frames / fps` 입니다; 마지막 프레임은 `(max_frames - 1) / fps` 에서 시작합니다.

|옵션|기본값|의미|
|---|---|---|
| `end_prompt`, `end_negative_prompt` |시작 텍스트|최종 양수 및 음수 텍스트|
| `end_prompt_2`, `end_negative_prompt_2` |최종 기본 텍스트 또는 명시적으로 제공된 시작 보조 텍스트|최종 SDXL/FLUX 보조 인코더 텍스트|
| `end_seed` | `seed` |최종 초기 노이즈 시드는 `[-2^63, 2^64-1]` 에 있습니다.|
| `max_frames` / `frames`, `fps` | 120, 12 |MP4: 양수 FPS 최대 12; GIF: `100/65535` ~ 100|
| `duration` |없음|프레임 수로 반올림된 초; 프레임 수와 배타적입니다.|
| `video_crf`, `video_preset` | 18, `medium` |H.264 품질 및 인코딩 속도|
| `ffmpeg`, `ffprobe` |실행 파일 이름|미디어 도구 경로 재정의|
| `encoding_timeout` | 300 |인코딩/디코드 확인 시간 초과(초)|

명시적인 빈 기본 부정 프롬프트는 빈 상태를 유지합니다. 빈 2 차 문자열은 기존 Diffusers 대체 경로 를 기본 텍스트로 유지합니다. JSON 와 Python 값은 CLI 인수의 동일한 엄격한 스키마를 사용하며, CLI 는 JSON 를 덮어씁니다. `--print-config` 는 Torch 또는 미디어 패키지를 가져오지 않고 가중치를 로드하거나 FFmpeg 를 실행하지 않고 모든 해결된 끝점 값을 검증하고 내보냅니다.

```sh
python3 reference/generate.py --backend interpolator --model /absolute/path/image-diffusers \
  --config reference/diffusers/interpolator.example.json --print-config
```

Python 에서 `generate.resolve_request({...})` 를 사용하여 설정을 검증/해결하세요. `prepare_pipeline_with_adapters()` 는 실행 후크 전에 2 조건부 끝점을 계산하고 첨부합니다. `interpolator_runtime.render_interpolator_frames()` 는 그 파이프라인/요청과 프레임 작성 콜백을 받습니다. 일반적인 CLI 는 이 수명 주기를 수행하고 비디오 번들을 자동으로 게시합니다.

<a id="runtime-and-compatibility"></a>

## 런타임 및 호환성

기존 SD 1.x, SDXL/Illustrious/NoobAI/Pony 및 FLUX.1 프리셋 계열은 기존 모델 로더를 사용한다. 로컬 safetensors 체크포인트에는 여전히 적절한 모델 구성/추가 항목이 필요하다. 모델/VAE 재정의, LoRA, textual inversion, 정적 ControlNet, 스케줄러 구성과 CPU/Metal/CUDA/ROCm 실행은 기존 계약을 유지한다. 일반 비디오 모델, 원시 GGUF와 ComfyUI 워크플로는 Interpolator 파이프라인으로 변환하지 않는다.

텍스트 임베딩과 LoRA를 설치한 후 가속기/오프로딩 훅을 연결하기 전에, 두 끝점을 CPU에서 한 번씩 인코딩한다. `cpu_text_encoding`는 true로 결정되며, 이 모드에서 명시적인 `--no-cpu-text-encoding`는 거부한다. 디노이징과 디코딩은 선택한 실행 장치를 사용한다. 네거티브 텐서와 풀링 텐서는 포지티브 임베딩과 함께 혼합한다. FLUX 네거티브 프롬프트에는 `true_cfg_scale > 1`이 필요하며, FLUX 노이즈는 해당 파이프라인 고유의 2x2 latent 배치로 패킹한다. 프롬프트/노이즈 보간은 float32 중간값을 사용하며, 유한성 검사를 수행하면서 선택한 추론 dtype으로 변환한다.

스케줄러는 각 프레임마다 다시 생성됩니다. 추가 확률적 샘플러 노이즈는 각 프레임마다 시작 시드로 독립적인 생성기 리셋을 사용하여 생성 순서가 공유 랜덤 스트림을 진행하는 것을 방지합니다. 결과적으로, 확률적 샘플러에서 생성된 종료 프레임은 초기 잠재 표현과 조건부가 정확함에도 불구하고, 종료 시드로 생성된 독립적인 이미지와 픽셀이 완전히 일치할 필요는 없습니다. 하드웨어/ 런타임 변경 사항도 픽셀을 변경할 수 있습니다. 결정론적 샘플러는 더 부드러운 전환을 위해 선호됩니다.

모드는 `num_images=1`, 완전한 디노이징 및 `.mp4` 또는 `.gif` 출력을 필요로 합니다. HiRes  수정, 외부 잠재 표현/임베딩 파일,  SDXL  조기 종료 및 Deforum 전용 카메라/프롬프트 일정/씨드 정책은 거부됩니다. ControlNet 가 아닌 비영도 방향의 재조정 거부가 거부됩니다. 2 엔드포인트를 가로지르는 모든 다른 생성 설정은 고정됩니다. 이 모드는 2 개의 엔드포인트를 제공하며, 멀티 키프레임 키프레임, 이미지 반전, 오디오 동기화 또는 재시도 렌더링이 아닙니다. 완전한 추론은 Python 동안 실행됩니다; C++ API 는 기존 계약과 컴퓨팅 구성 요소 책임을 유지합니다.

<a id="artifacts-and-validation"></a>

## 아티팩트 및 검증

`build/interpolator.mp4` 은 `interpolator.json` 과 `interpolator-frames/frame-000000.png` 이후로 함께 제공됩니다. 각 프레임은 디스크로 스트리밍되며, 엔드포인트 텐서와 현재 프레임만 메모리에 유지되고, 압축된 프레임별 기원이 JSON 보고서에 축적됩니다. 보고서는 엔드포인트 구성, 모델/어댑터 식별자, 텐서 형식 및 표준 부동소수점32 해시, 보간 비율, 실제 디노이징 시간 단계, 유한한 잠재 상태, PNG 해시 및 디코딩된 MP4/GIF 속성을 기록합니다.

공유된 `animation_video.py` 는 FFprobe 로 디코딩하여 H.264/yuv420p, 차원, 프레임 수, FPS 및 지속 시간을 확인합니다. GIF 디코딩은 Pillow 를 사용하여 프레임, 차원, 루프 및 10 ms 시간 해상도로 지속 시간을 확인합니다. 완료 보고서는 마지막에 게시되며, 일반 샘플링/인코딩/게시 실패 시 이전 출력을 복원합니다. 충돌 시 기본 출력 이름에 번호가 매겨진 접미사가 추가되며, 명시적인 경로는 `--overwrite` 를 사용하여 대체해야 합니다. 관리되지 않는 프레임 디렉터리와 심볼릭 링크는 덮어쓸 수 없습니다. 두 모드 모두 독점 출력 잠금 장치를 공유하며, 기존 프레임 번들은 동일한 애니메이션 모드에 속해야 합니다. 강제 프로세스 종료/전원 손실은 설명된 대로 스테이지/락 파일을 남길 수 있습니다.
[Deforum 아티팩트 계약](deforum-video.md#artifacts-and-verification).

`InterpolatorOptionsTests`, `InterpolatorRuntimeTests` 및 `InterpolatorMediaTests` 는 CTest 에 등록되며, 텐서 테스트는 런타임 런타임 생성이 필요하고, 미디어 테스트는 Pillow / FFmpeg 를 필요로 합니다. 설치된 하드웨어에서 실제 추론 스모크를 실행합니다:

```sh
reference/diffusers/.venv/bin/python -m unittest discover -s tests -p 'Interpolator*Tests.py'
reference/diffusers/.venv/bin/python tests/InterpolatorDiffusersSmoke.py --device cpu
reference/diffusers/.venv/bin/python tests/InterpolatorDiffusersSmoke.py --device mps
reference/diffusers/.venv/bin/python tests/InterpolatorAdapterSmoke.py --device mps
reference/diffusers/.venv/bin/python tests/InterpolatorAdapterSmoke.py --device mps --dtype float16 --offload model
```

스모크는 로컬로 생성된 작은 무작위 SD / SDXL 호환 safetensors 를 사용하여 ControlNet 를 포함하고 포함하지 않은 상태로 실행됩니다. 그것은 변경되는 임베딩/노이즈/픽셀, 엔드포인트 해시, 실제 유한 확산 단계 및 비디오 디코딩을 확인합니다. 그것은 훈련된 모델의 시각적 품질, FLUX 추론 또는 CUDA /ROCm 하드웨어 실행을 확립하지 않습니다. 아티팩트와 로그는 `build/interpolator-smoke` 에 있습니다. 어댑터 스모크는 합성 비영수 LoRA, 학습된 2-벡터 토큰, 정적 ControlNet 및 선택 가능한 정밀도/오프로드를 추가합니다. 텍스트 인버전 벡터는 엔드포인트 인코딩 전에 검증되며, 오프로드가 그들을 메타 텐서로 변환한 후 텍스트 인코더 가중치를 읽지 않고 프레임 루프는 캐시된 조건을 사용합니다.

테스트된 M1 Max 런타임 ( PyTorch   2.13.0,  Diffusers   0.40.0, Accelerate  1.14.0 )에서 결합된 어댑터 픽스처 는 FP16 와 순차적 오프로드를 포함하여 무한이 아닌 소음 제거 값을 생성했으며, 후크를 새로고친 후에도 마찬가지입니다. FP32 순차 오프로드와 FP16 모델 오프로드 또는 레지던트 가중치가 완료된 경우 이는 해당 조합의 측정된 수치 제한이며, 훈련된 모델 호환성 주장이 아닙니다. 이 경우 `--dtype float32` 또는 다른 오프로드 정책을 사용하세요. 유한 값 감사 (audit) 는 실패한 프레임을 거부하고 기존 출력을 그대로 두며, 아무런 알림 없이 정밀도를 대체하거나 NaNs 를 수리하거나 유효하지 않은 비디오를 게시하지 않습니다.
