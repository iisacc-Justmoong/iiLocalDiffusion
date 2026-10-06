<a id="generate-from-local-model-arguments"></a>

# 로컬 모델 인수에서 생성

이 페이지에서는 **local-path** 소스를 설명합니다. 다른 모델 위치는
[직접 API 와 클라우드 모델 ID](model-sources.md)은 모두 원격으로 평가됩니다. iiLocalDiffusion 의 로컬 라우트는 호출자 제공 파일과 패키지를 사용하여 생성됩니다. `--model-path` (레거시 `--model`) 는 로컬 이미지, Deforum, Interpolator 및 시간적 비디오 요청, `--print-config` 를 포함하여 필요합니다. 프리셋은 아키텍처와 샘플링 기본값을 선택하며, 모델 가중치를 선택하거나 다운로드하지 않습니다. 허브 저장소 ID, URL 및 누락된 경로는 모델 로딩 전에 실패하며, 불변의 리비전이 제공된 경우에도 마찬가지입니다.

기존 Diffusers / PyTorch 구현은 독립적인 추론을 제공합니다. ComfyUI 는 명시적으로 선택된 워크플로 백엔드에만 필요합니다. 새로운 추론 라이브러리나 유료 서비스가 도입되지 않습니다. 런타임 설치 는 별도의 준비 단계입니다. 생성은 로컬 전용 모델 로딩을 사용하며, 누락된 구성, 토크나이저, 인코더 및 가중치는 SDK 번들 내에 존재하거나 추론 중에 다운로드되지 않고 로컬로 제공되어야 합니다.

<a id="model-inputs"></a>

## 모델 입력

|생성 경로|필수 로컬 입력|
|---|---|
|자동 이미지 경로|다운로드된 체크포인트 파일 또는 전체 Diffusers 모델 디렉터리|
|프리셋 이미지, Deforum, Interpolator|호환 SD/SDXL/FLUX.1 Diffusers 디렉토리; SD1/SDXL 체크포인트는 번들 오프라인 구성을 사용하며, 기타 단일 파일은 로컬 `--model-config` 추가 파일이 필요합니다.|
|일반 Diffusers|완전한 모델 디렉토리; 지원되는 단일 파일 파이프라인은 로컬 `--model-config` 디렉토리가 필요합니다.|
|임시 비디오|변환기, 임시 VAE, 텍스트 인코더, 토크나이저 및 스케줄러를 포함하는 호환 가능한 LTX Diffusers 디렉터리|
|명시적 ComfyUI 워크플로|루프백에 이미 설치된 모델을 참조하는 로컬 API 워크플로 런타임|

선택 사항 `--vae`, `--lora`, `--controlnet`, 구성 요소 구성 및 텍스트 임베딩 또한 로컬 파일이나 디렉토리를 참조합니다. 독립적인 노이즈 제거기는 별도로 제공된 VAE/text-encoder 가중치를 여전히 필요로 할 수 있습니다. 로컬 가용성은 아키텍처 호환성이나 시각적 품질을 함의하지 않습니다.

구체적 버전 인수에는 생성 경로가 없습니다. 비어 있지 않은 버전은 거부됩니다. `--local-files-only` 는 기존 로컬 명령에 대해 계속 허용되며 항상 활성화되어 있으며, 이를 비활성화하려는 요청은 거부됩니다. JSON 경로는 해당 설정 파일에 대해 상대적으로 해결됩니다. CLI 경로는 작업 디렉토리에 대해 상대적으로 해결됩니다.

<a id="images-and-animations-from-the-same-image-model"></a>

## 동일한 이미지 모델의 이미지 및 애니메이션

유니파일 런처의 비디오 기본값은 **로컬 LTX 에서 24 FPS**입니다. 그것은 `.mp4` / `.gif` 출력, FPS, 프레임 수 또는 지속 시간에서 비디오 요청을 인식합니다. 자동 선택에서 **FPS <= 12 OR GIF 출력** 은 Deforum 을 사용하거나, 종료 프롬프트/시드 입력이 있는 경우 인터폴레이터를 사용합니다. 명시적 Deforum/인터폴레이터 요청은 12 FPS 로 기본값이며, 12 FPS 보다 높으면 거부되지만 출력 확장자가 `.gif` 인 경우 예외입니다. 정책은 직접 프리셋 런처와 Python 요청에도 적용됩니다.

|요청|자동 백엔드|필수 로컬 모델|
|---|---|---|
|MP4, FPS 생략 또는 12|LTX → 프레임 보간기(`video`), 최종 24 생략된 경우 FPS|LTX 디렉토리만 해당|
|MP4, FPS 최대 12|Deforum|호환 이미지 모델|
|GIF, FPS 포함 12|Deforum|호환 이미지 모델|
|종료 프롬프트/시드가 있는 적격 애니메이션|보간기|호환 이미지 모델|

`--backend interpolator` 를 사용하여 인터폴레이션을 명시적으로 선택합니다. 명시적 `--backend video` 와 LTX -특정 키프레임 /카메라/스토리보드 요청은 LTX 를 유지하며, FPS 에서도 포함됩니다. 금지된 이미지 애니메이션 요청은 아무런 알림 없이 다른 모델로 대체하거나 컨트롤을 버리는 대신 실패합니다. `--model` 는 항상 호출자의 호환 가능한 로컬 가중치를 이름 짓습니다. 런처는 백엔드를 선택할 때 가중치를 가져오거나 대체하지 않습니다. 이미지 요청은 기존 라우팅을 유지하며, 명시적 일반 파이프라인/워크플로우는 자체 계약을 유지합니다.

`--frames` 와 `--max-frames` 는 JSON 를 포함하여 별칭입니다. `--duration` 는 프레임 카운트로만 독점되며 `seconds * fps` 를 출력 프레임으로 반올림합니다. CLI 와 FPS 는 JSON 를 라우팅하기 전에 덮어씁니다; 출력 형식은 확장자에 따라 대문자/소문자 구분 없이 결정됩니다.

```sh
reference/diffusers/.venv/bin/python reference/generate.py \
  --backend preset --preset sdxl \
  --model /absolute/path/sdxl-diffusers --prompt 'A glass bottle on a table.' \
  --output build/bottle.png

reference/diffusers/.venv/bin/python reference/generate.py \
  --backend deforum --preset sdxl \
  --model /absolute/path/sdxl-diffusers --prompt 'A glass bottle on a table.' \
  --max-frames 48 --fps 12 --zoom 1.01 --output build/bottle-camera.mp4

reference/diffusers/.venv/bin/python reference/generate.py \
  --backend interpolator --preset sdxl \
  --model /absolute/path/sdxl-diffusers \
  --prompt 'A blue glass bottle.' --end-prompt 'An amber glass bottle.' \
  --max-frames 48 --fps 12 --output build/bottle-transition.mp4
```

이러한 모드는 동일한 이미지 가중치를 재사용합니다. Deforum은 이전 프레임 피드백을 사용합니다. 보간기는 프롬프트/시드 조건을 혼합합니다. 그들은 이미지 모델을 시간적 움직임에 대해 훈련된 모델로 바꾸지 않습니다.

24 FPS GIF 에 대해서는 `--fps 24 --output build/bottle.gif` 와 동일한 이미지 가중치를 사용합니다. GIF 출력은 새로운 의존성 없이 FFmpeg 팔레트 생성/인코딩과 Pillow 디코딩을 사용합니다. 그것은 무한히 반복합니다. GIF 지연은 10 ms 해상도를 가지며: 인코딩된 지속 시간은 이에 따라 반올림되고 리포트에는 요청된 FPS 와 실제 `encoded_fps` 가 모두 포함됩니다. 지원되는 GIF 타이밍 범위는 `100/65535 <= fps <= 100` 입니다. H.264 CRF /preset 옵션은 MP4 에만 적용됩니다. 두 형식 모두 PNG 프레임, 기원 리포트 및 거래적 게시 확인을 유지합니다.

<a id="temporal-video-from-a-local-video-model"></a>

## 로컬 비디오 모델의 시간 비디오

일반적인 LTX 비디오는 **LTX 생성 후 프레임 보간**를 사용합니다. 두 번째 단계에는 추가 모델이 필요 없습니다. `--fps` 는 최종 비율이며 `--interpolation-factor` 는 2 로 기본값으로 설정됩니다 (허용 2 – 8 ). LTX 는 키프레임 를 보존한 소스 프레임을 생성한 다음, 비디오 길이 변경이나 컷 교차 없이 나머지 출력 위치를 채우는 운동 보간기를 사용합니다. [2단계를 거친 계약의](temporal-video.md#ltx-followed-by-frame-interpolation) 를 참조하여 소스 프레임 일정을 확인하고, CPU 후처리 및 출처 정보를 확인하세요. LTX 를 명시적으로 FPS <= 12 로 설정하면 후처리를 건너뛰고, 독립형 GIF /low- FPS 애니메이션은 변경되지 않습니다.

```sh
reference/diffusers/.venv/bin/python reference/generate.py \
  --backend video --model /absolute/path/ltx-diffusers \
  --prompt 'A red boat moves across blue water.' \
  --camera dolly-in --duration 5 --fps 24 --interpolation-factor 2 --output build/boat.mp4
```

첫 번째/마지막 이미지 조건, 카메라 설명, 촬영 계획 및 이전 촬영 연속성은 계속 사용할 수 있습니다. 모델의 선언된 로컬 아키텍처는 선택한 백엔드와 일치해야 합니다. 임의의 모델 파일 이름은 LTX로 해석되지 않습니다.

Python 호출자는 동일한 명시적 모델 필드를 사용합니다.

```python
from generate import resolve_request

preset, request = resolve_request({
    "model": "/absolute/path/sdxl-diffusers",
    "preset": "sdxl",
    "prompt": "A glass bottle on a table.",
    "animation_mode": "2D",
    "max_frames": 48,
    "output": "build/bottle.mp4",
})
```

이렇게 하면 요청이 해결됩니다. 선택된 런타임가 추론을 실행합니다. 도움말, 카탈로그, 카메라 목록 및 런타임 검사 명령에는 생성 가중치가 필요하지 않습니다.
