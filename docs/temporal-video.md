<a id="temporal-video-generation"></a>

# 시간적 비디오 생성

`iild-generate --backend video`는 Diffusers의 `LTXConditionPipeline`, 비디오 확산 트랜스포머와 시간축을 압축하는 비디오 VAE를 사용하여 공간과 시간에 걸쳐 하나의 샷을 함께 생성한다. 텍스트, 이미지 키프레임 및 카메라 설명이 모델에 조건을 제공한다. 각 샷은 한 번의 파이프라인 호출로 비디오 잠재 표현 시퀀스를 샘플링한다. 일반 비디오는 이어서 **두 번째 단계의 프레임 Interpolator**를 거친 뒤 샷들을 조립하여 검증된 H.264 MP4를 만든다.

<a id="ltx-followed-by-frame-interpolation"></a>

## LTX 다음에 프레임 보간

12위의 최종 FPS의 경우 생성은 항상 다음 순서로 실행됩니다.

1. 호출자의 로컬 LTX 모델은 소스 프레임의 시간 시퀀스를 생성합니다.
2. 프레임 보간기는 누락된 출력 프레임을 각 샷 내에 삽입합니다.
3. 완성된 시퀀스는 MP4 검증을 위해 인코딩 및 디코딩됩니다.

`--fps`, `--frames`, `--duration`는 보간 프레임을 포함한 **최종 출력**을 설명한다. `--interpolation-factor` 기본값은 **2**이며 2–8의 정수를 받는다. 해당 출력 타임라인에서 LTX 원본 프레임의 일반적인 간격을 설정하며, 요청한 최종 FPS나 재생 시간을 곱하지 않는다. 시작/끝 프레임과 시간 지정 이미지 조건 전체는 규칙적인 샘플 사이에 있어도 항상 LTX 원본 기준점에 포함한다. 원본 샘플링 FPS는 샷의 첫 시각부터 마지막 시각까지의 범위를 유지하며, 추가한 기준점 때문에 `final_fps / factor`와 약간 달라질 수 있다. 보간기는 원본 프레임을 배치할 때 기록한 출력 타임스탬프를 사용한다.

예를 들어, 5초의 24 FPS 출력에는 120 프레임이 포함되어 있습니다. 기본 플랜은 61 LTX 소스 프레임을 유지하며, 65 프레임으로 시간적 추론을 채우고 VAE 의 두 번째 단계에 59 프레임을 삽입합니다. 마지막 프레임 포함 모든 유지된 소스 PNG 는 변경 없이 최종 PNG 시퀀스에 복사됩니다. 매우 짧은 또는 키프레임이 적용된 키프레임이 적용된 샷은 0 누락 프레임을 가질 수 있으며, 보고서는 추가 프레임이 합성되었다고 주장하는 대신 이를 기록합니다.

**FPS <= 12 및 GIF 예외는 기존 독립형 Deforum 또는 프롬프트/시드 인터폴레이터 라우트**를 유지합니다. 명시적 저-FPS LTX 요청은 단일 LTX 단계를 유지합니다. `--backend interpolator` 에 대한 고-FPS 제한은 독립형 이미지 모델 애니메이션에 관한 것이며, 비디오 백엔드의 사후 처리 단계를 의미하지 않습니다.

포스트 프로세서는 기존 FFmpeg를 사용합니다.
[`minterpolate` 운동 보상 필터](https://ffmpeg.org/ffmpeg-filters.html#minterpolate) 를 CPU 에 적용합니다. 이미지 모델, 인터폴레이션 모델, 다운로드, OpenCV 또는 새 Python 의존성이 필요하지 않습니다. 필터 가용성은 LTX 가중치가 로드되기 전에 확인됩니다. 이는 [인터폴레이터 애니메이션](interpolator-video.md)에서 설명된 프롬프트/노이즈 인터폴레이션과 구별되는 LTX 의 손실 없는 PNG 시퀀스 위에서의 프레임 인터폴레이션입니다. FFmpeg 가 운동을 추정하며, 가림과 복잡한 운동은 여전히 아티팩트를 생성할 수 있습니다.

각 샷은 독립적으로 처리되므로 스토리보드 컷을 가로지르는 합성 전환이 도입되지 않습니다. 경계 패딩은 인터폴레이션 미리보기를 공급하고 버려집니다. LTX 파이프라인 참조는 CPU 사후 처리 전에 해제됩니다. 두 번째 단계의 실패는 완료되지 않은 첫 번째 단계를 게시하는 대신 전체 새 번들을 중단시킵니다.

<a id="local-model-and-dependencies"></a>

## 로컬 모델 및 종속성

로컬 생성에는 `model_index.json`, 트랜스포머, 시간축 VAE, 텍스트 인코더, 토크나이저와 스케줄러가 들어 있는 `--model-path /absolute/path/to/local-ltx-model`(레거시 `--model`)가 필요하다. 출력이 GIF인 경우를 제외하면 12 FPS를 넘거나 FPS를 생략한 자동 요청(기본값 24)의 기본 비디오 아키텍처는 LTX이다. 기본 모델 경로나 자동 모델 다운로드는 없다. 원격 실행은 `--model-family ltx`와 함께 `--model-api` 또는 `--model-cloud` 및 `--model-provider`를 통해 명시적으로 선택한다. [모델 소스](model-sources.md)를 참고한다. 로컬 소스에서는 CLI, JSON의 `model_path`/`model` 및 Python 요청이 동일한 로컬 디렉터리 계약을 사용한다. [로컬 모델 생성](local-model-generation.md)을 참고한다.

기존 Diffusers / 0.40.0 / PyTorch / Transformers / Accelerate 패키지는 신경 모델, 시간적 주의력, VAE , 스케줄러 및 RAM 오프로드를 제공합니다. 추가적인 Python 추론 스택 또는 유료 API 는 필요하지 않습니다. FFmpeg 와 FFprobe 는 이미 애니메이션 백엔드에 의해 사용되며, 결과를 인코딩하고 검증합니다. 선택된 FFmpeg 는 또한 일반 2단계 비디오를 위해 `minterpolate` 와 `tpad` 를 제공해야 합니다. 기존 외부 실행 가능 라이선스는 변경되지 않습니다.

이전에 검증된 모델은
[`Lightricks/LTX-Video-0.9.5`](https://huggingface.co/Lightricks/LTX-Video-0.9.5)는 `e58e28c39631af4d1468ee57a853764e11c1d37e` 에 있습니다. 호환 가능한 로컬 스냅샷은 `--model` 로 제공될 수 있으며, 암묵적으로 선택되는 경우는 절대 없습니다. 버전별 오픈 RAIL-M 라이선스는 제한 사항에 따라 상업적 사용을 허용합니다. 가중치는 별도로 다운로드되며 SDK 에 재분배되지 않습니다. 더 새로운 LTX 버전은 다른 라이선스를 가지며 상당한 더 많은 메모리가 필요할 수 있습니다; [상위 공급 측 모델과 라이선스 카탈로그](https://huggingface.co/Lightricks/LTX-Video)를 참조하세요.

<a id="generate-a-video"></a>

## 비디오 생성

기존 관리형 Python 환경과 함께 비디오 토크나이저 종속성을 설치합니다([environment setup](../reference/diffusers/README.md)참조).

```sh
uv pip install --python reference/diffusers/.venv/bin/python \
  -r reference/diffusers/requirements-video.txt
```

이 핀은 모델의 SentencePiece 토크나이저를 변환하는 데 필요한 `protobuf`를 핀합니다. Tokenizer 검증은 큰 모델 가중치를 읽기 전에 실행됩니다. 관리형 Python 환경으로 실행하거나 해당 환경을 선택하는 `IILD_PYTHON_EXECUTABLE`와 함께 설치된 `iild-generate` 실행기를 사용합니다.

```sh
reference/diffusers/.venv/bin/python reference/generate.py --model /absolute/path/ltx-diffusers \
  --backend video \
  --prompt 'A glass bottle on a stone table, warm sunlight glinting through amber liquid.' \
  --camera dolly-in --duration 5 --fps 24 --interpolation-factor 2 \
  --output build/reference/bottle.mp4
```

이미지에서 비디오를 생성하려면 `--first-frame /absolute/path/keyframe.png`를 추가한다. 선택적 `--last-frame /absolute/path/end.png`는 마지막 출력 프레임에 조건을 제공한다. 입력은 정적 로컬 이미지로 디코딩하고 EXIF 방향을 적용한 뒤 RGB로 변환하여 출력 종횡비에 맞게 중앙을 잘라낸다. 키프레임은 모델 생성에 영향을 주며 끝점의 픽셀 일치나 정체성 보존을 보장하지 않는다.

```sh
reference/diffusers/.venv/bin/python reference/generate.py --model /absolute/path/ltx-diffusers \
  --backend video --first-frame /absolute/path/keyframe.png \
  --prompt 'The subject turns slowly toward the window in warm afternoon light.' \
  --camera pan-left zoom-in --duration 3 --seed 7 \
  --output build/reference/keyframe-video.mp4
```

`--camera`는 호환되는 움직임을 최대 3개 받는다. 편집 가능한 프롬프트 설명은 `--backend video --list-camera-motions`로 나열한다. 지원하는 선택에는 dolly·pan·tilt·orbit·crane·zoom·handheld·tracking·dolly zoom이 있다. `none`은 장면 프롬프트를 유지하고 `static`은 고정된 삼각대를 요청한다. 중립·정지·중복·직접 반대되는 움직임은 섞을 수 없다. 카메라 제어는 학습된 텍스트 조건이며, 측정되거나 보장된 3D 카메라 경로가 아니다. 움직임 설명은 장면 캡션보다 앞에 온다. 선택한 토크나이저 한계를 넘는 요청은 조용히 잘라내어 카메라나 장면 지시를 누락하는 대신, 조치 가능한 오류로 실패한다.

<a id="shot-plans-and-reference-continuity"></a>

## 촬영 계획 및 참조 연속성

```sh
reference/diffusers/.venv/bin/python reference/generate.py --model /absolute/path/ltx-diffusers \
  --backend video --storyboard reference/diffusers/video-storyboard.example.json \
  --output build/reference/story.mp4
```

스토리보드는 JSON 객체로 `shots` 배열을 가집니다. 샷 필드는 전역 기본값을 덮어씁니다: `prompt`, `negative_prompt`, `camera` (배열), `frames` 또는 `duration`, `seed`, `first_frame`, `last_frame`, `continue_previous`, `conditions` 입니다. 샷 내 경로는 스토리보드 파일에 상대적입니다. 시간 기반 이미지 조건을 사용하려면:

```json
{"image": "detail.png", "frame": 24, "strength": 0.8}
```

`conditions` 배열은 최대 32 개의 고유 프레임 인덱스 엔트리를 허용합니다. 강도는 `[0.001,1]` 에 있습니다. `continue_previous: true` 조건은 이전 샷의 생성된 마지막 프레임을 0 에 적용하며 다른 프레임0 참조와 충돌할 수 없습니다. 각 샷은 프롬프트, 카메라, 시드 및 참조 메타데이터를 유지합니다. 샷은 컷에서 만나며 이는 순차적 샷 생성으로 하나의 네이티브 멀티샷 모델 호출이 아닙니다. 의미적 연속성은 여전히 프롬프트와 가중치에 의존합니다.

<a id="configuration-and-execution"></a>

## 구성 및 실행

JSON 는 밑줄로 구분된 동일한 인수 이름을 사용합니다. 명시적인 CLI 값은 JSON 값을 덮어씁니다. `--config reference/diffusers/video.example.json` 는 `backend: "video"` 를 통해 비디오 백엔드를 선택합니다. `--print-config` 는 Torch 를 가져오거나 가중치를 다운로드하지 않고 구성을 해결하고 유효성을 검사합니다. 생략되거나 JSON -null 시드는 요청마다 무작위입니다. 해결된 구성과 샷 메타데이터는 실제 시드를 기록합니다. 명시적인 시드, 0포함은 유지됩니다.

||설정 기본값 및 계약|
| --- | --- |
| `width`, `height` |704 × 480 ; 32의 배수, 32에서 4096까지|
| `duration` / `frames` |5 초 또는 명시적인 프레임 수; 상호 배타적|
| `fps` |24; 보간 후 최종 출력 속도|
| `interpolation_factor` |2, 정수 2 – 8 ; 12위에 있는 FPS의 소스 프레임 간격|
| `steps`, `guidance_scale` | 30, 3 |
| `seed` |비디오당 무작위 32-비트 기본 시드; 이후 샷은 별도 지정이 없는 한 이를 증가시킵니다.|
| `max_sequence_length` |256 ; T5 프롬프트 제한, 최대 512까지 구성 가능|
| `device` |GPU-필수 자동 또는 명시적 CPU/mps/metal/cuda/rocm|
| `dtype` |자동 사용은 CPU에 float32를 사용하고, 가속기에서는 bfloat16를 사용합니다.|
| `offload` |자동 사용 모델 RAM는 GPU에서 하역하고, CPU에서는 제외합니다.|
| `cpu_text_encoding` |False; CPU에 모든 캡션을 인코딩하고 샘플링 전에 T5를 해제하도록 선택하십시오.|
| `vae_tiling` |참|
| `decode_timestep`, `decode_noise_scale` | 0.05, 0.025 |
| `image_cond_noise_scale` | 0 |
| `video_crf`, `video_preset` |18, 중간|
| `encoding_timeout` |300 초|

지속 시간은 가장 가까운 출력 프레임으로 반올림됩니다. 샘플러는 각 샷의 **LTX 소스 카운트** 를 `8k+1` 프레임으로 패드한 다음 초과된 꼬리 프레임만 잘라냅니다. 인터폴레이터는 남은 출력 위치를 채웁니다. 요청된 지속 시간, FPS 및 프레임 카운트는 명시적으로 유지됩니다. 각 샷은 2 – 4097 출력 프레임을 수락하며, 스토리보드는 최대 256 샷을 수락합니다. 큰 요청은 더 많은 메모리가 필요합니다. 추론 오류는 아무런 알림 없이 CPU 또는 이미지 애니메이션 대체 경로 없이 보고됩니다.

로컬 실행을 위해 `--model-path` / `--model` 는 완전한 LTX Diffusers 디렉토리가 필요합니다. 비-null 수정어거먼트는 거부됩니다. 내장 컴포넌트와 safetensors 만 로드되며, 누락된 리소스는 다운로드 없이 실패합니다. `--cache-dir` 는 임시 작업 저장소를 제어하며, 모델 선택이 아닙니다. 모델 신원 기록에는 컴포넌트 파일 해시와 크기가 포함되며, 로드와 추론은 소스 파일 변경을 확인합니다.

<a id="outputs-and-verification"></a>

## 출력 및 검증

각 완료된 런은 `name.mp4`, `name.json`, `name-frames/` 를 게시합니다. JSON 는 모델 파일/수정어거먼트, 요청된 설정, 실제 캡션, 카메라 제어, 키프레임 해시, 컷 위치, 디노이징 텐서 형식과 장치, 개별 PNG 해시, 그리고 디코딩된 MP4 의 개수, 지속 시간, FPS, 해상도와 SHA-256 를 포함합니다. 비 유한 잠재 표현 또는 디코딩된 픽셀은 거부됩니다.

`stages` 는 LTX 와 인터폴레이터를 별도로 기록합니다. `source_frames` 는 `name-frames/ltx/shot-NNNN/` 하의 LTX PNG 를 식별하며, 해시와 출력 위치를 포함합니다; `frames` 는 완전한 출력 시퀀스를 식별합니다. 각 삽입된 프레임은 인접한 소스 인덱스와 인터폴레이션 분수를 기록합니다. `interpolation` 는 FFmpeg 필터, CPU 실행, 팩터, 실제 삽입된 개수, 샷별 검증과 타이밍을 기록합니다. 소스 앵커 해시는 변경 없이 살아남아야 합니다.

로딩, 샘플링, 보간, 인코딩 또는 게시가 실패하면 기존 번들이 그대로 유지됩니다. 또한 일반적인 독점 잠금 또한 동시적으로 같은 목적지에 애니메이션 및 비디오 작업을 게시하는 것을 방지합니다. 기존 명시적 출력은 `--overwrite` 를 필요로 합니다; 생성된 기본 이름은 새 런 접미사를 받습니다.

`VideoOptionsTests.py` 은 계획 및 라우팅을 담당하며, `VideoRuntimeTests.py` 는 텐서/미디어/게시 계약서를 담당합니다. `VideoInterpolationTests.py` 은 타이밍, 운동 삽입, 소스 해시, 그리드 밖 앵커, 컷 경계 및 실패를 확인합니다. `VideoDiffusersSmoke.py` 는 텍스트, 엔드포인트-키프레임 및 체인샷 요청에 대해 LTX 파이프라인을 실제로 재귀적으로 실행하며, 두 단계가 기본적으로 활성화되어 CPU 또는 MPS 에서 실행됩니다; `--fps 8` 는 저-FPS 예외를 확인합니다. 작은 무작위 픽스처 는 실행을 증명하며, 훈련된 모델의 시각적 품질을 증명하지 않습니다. 네이티브 C++ SDK 의 기존 매니페스트/계산 경계는 변경되지 않으며, 이 기능은 설치된 Python 생성 진입점을 통해 실행됩니다.

## 데스크톱 앱의 작업 진행률

`IILD_WORKER_PROGRESS=1`인 설치된 워커는 `IILD_VIDEO_PROGRESS` 뒤에 `iild-video-progress-v1` JSON을 출력한다. `stage`는 loading, encoding, denoising, decoding, interpolating, encoding-video, complete이며 `step`/`total`은 실제 단계 경계와 유한 잠재 텐서 검증을 통과한 샘플링 진행을 나타낸다. 단순 heartbeat를 진행으로 취급하지 않는다. Dreamscapes는 이 이벤트를 작업 화면과 정체 감시 타이머에 사용한다. 모델은 실행 전에 로컬 LTX 패키지로 준비되어 있어야 한다.
