<a id="deforum-2d-video-generation"></a>

# Deforum 2D 비디오 생성

`iild-generate --backend deforum` 는 순차적인 2D 확산 애니메이션을 실행하고 H.264 MP4 또는 애니메이션된 GIF, 손실 없는 PNG 프레임과 JSON 기원 보고서를 작성합니다. 동등한 프리셋 러너 옵션은 `--animation-mode 2D` 입니다. 프레임 0 은 텍스트에서 이미지로, 또는 선택적인 로컬 초기화 이미지를 사용합니다. 각 후속 프레임은 **이전 생성된 프레임**을 왜곡한 다음 이를 이미지에서 이미지 조건으로 사용합니다. 모델 가중치는 한 번 로드되며, img2img 파이프라인은 해당 구성 요소를 공유합니다.

Deforum은 **FPS <= 12 또는 GIF 출력**로 제한됩니다. 기본값은 12 FPS입니다. 기타 자동 비디오 요청은 24 FPS에서 로컬 LTX를 사용합니다. 참조
[공유 비디오 선택 및 GIF 규칙](local-model-generation.md#images-and-animations-from-the-same-image-model).

이는 기존 Diffusers 런타임 를 통해 [Deforum 2D 피드백과 키프레임 방법](https://github.com/deforum/sd-webui-deforum/wiki/Animation-Settings) 을 구현합니다. 그것은 AUTOMATIC1111 확장 프로그램의 설치도 아니며 임의의 Deforum 설정 파일을 가져오지 않습니다. 3D 깊이 왜곡, 광학 흐름, 템포/이중 프레임, 비디오 입력, 오디오 동기화 및 재개는 구현되지 않았습니다. 프롬프트 임베딩 블렌딩은 별도의 [인터폴레이터 모드](interpolator-video.md)에서 사용할 수 있습니다. Deforum 프롬프트 키프레임 는 지정된 프레임에서 텍스트를 변경하며, 숫자 앵커는 보간합니다. C++ 라이브러리는 여전히 기존 계약과 컴퓨팅 구성 요소를 제공하며, 완전한 비디오 추론은 Python 에서, 완전한 이미지 추론과 동일하게 실행됩니다.

<a id="runtime-and-usage"></a>

## 런타임 및 사용법

기존 Diffusers 환경과 선택적인 유지 관리 카메라 의존성을 사용하세요. FFmpeg **와 FFprobe** 은 PATH 에서 사용 가능해야 하며, FFmpeg 의 `libx264` 인코더를 MP4 에, 또는 GIF 인코더와 팔레트 필터를 GIF 에 사용해야 합니다. 생성기는 모델 가중치를 로드하기 전에 이를 사전 준비합니다.

```sh
uv pip install --python reference/diffusers/.venv/bin/python \
  -r reference/diffusers/requirements-deforum.txt

reference/diffusers/.venv/bin/python reference/generate.py \
  --backend deforum --preset sd15-compatible \
  --model /absolute/path/to/diffusers-model --local-files-only \
  --animation-prompts '{"0":"a city at sunrise","60":"a forest at sunrise"}' \
  --max-frames 120 --fps 12 --steps 20 \
  --strength-schedule '0:(0.35), 119:(0.45)' \
  --zoom '0:(1.01)' --angle '0:(0.2*sin(2*pi*t/fps))' \
  --output build/deforum.mp4
```

`--preset sdxl`, `illustrious`, `noobai`, `noobai-v-pred`, `pony`, `flux1-dev`, `flux1-krea-dev` 또는 해당 가중식과 호환되는 기존 프레셋을 사용하세요. `--base-model` 는 카탈로그의 정확한 패밀리 라우팅을 유지합니다. 로컬 체크포인트는 기존 `--model-config` 동반 디렉토리를 사용할 수 있습니다. Deforum 은 프레셋 로더를 사용하므로, 원시 GGUF / ComfyUI 워크플로우와 관련 없는 일반적인 아키텍처는 지원되지 않는 애니메이션 모델로 변환되지 않습니다.

모델/ VAE 교체, LoRA, 텍스트 인버전, 정적 ControlNet 이미지, CPU 프롬프트 인코딩, CPU / Metal / CUDA /ROCm 실행 및 기존 오프로드 정책은 여전히 사용 가능합니다. 기존 FLUX ControlNet 리파인먼트 어댑터는 부정 조건화 경로를 유지합니다. 각 프레임은 자신의 스케줄러 상태와 생성기를 다시 생성하며, CPU 프롬프트 임베딩은 현재 프레임의 텍스트를 위해 새로 갱신됩니다.

동일한 옵션이 형식화된 JSON `--config` 및 Python `generate.resolve_request({...})`에서 작동합니다. 명시적 CLI 값은 JSON 값을 재정의합니다. `--print-config`는 Torch/OpenCV를 가져오거나 가중치를 다운로드하거나 FFmpeg를 실행하지 않고도 재생 가능한 매개변수를 검증하고 내보냅니다. 참조
[`deforum.example.json`](../reference/diffusers/deforum.example.json). `deforum_runtime.render_deforum_frames()`는 또한 이 프레임 루프를 Python에 삽입하기 위해 이미 준비된 파이프라인과 출력 콜백을 허용합니다.

```sh
python3 reference/generate.py --backend deforum --model /absolute/path/image-diffusers \
  --config reference/diffusers/deforum.example.json --print-config
```

<a id="options-and-schedule-semantics"></a>

## 옵션 및 일정 의미

|옵션|2D 모드의 기본값|의미|
|---|---|---|
| `max_frames` / `frames` | 120 |출력 프레임 수, 1 ~ 1,000,000|
| `duration` |없음|프레임 수로 반올림된 초; 프레임 수와 배타적입니다.|
| `fps` | 12 |MP4: 양수 및 최대 12; GIF: `100/65535` ~ 100|
| `init_image` |없음|정적 로컬 이미지, EXIF 수정 및 요청된 크기로 크기 조정|
| `animation_prompts` | `{"0": prompt}` |표준 0기반 프레임 번호로 키가 지정된 양수 텍스트|
| `animation_negative_prompts` | `{"0": negative_prompt}` |부정적인 텍스트 키프레임|
| `animation_prompts_2`, `animation_negative_prompts_2` |주 스케줄|SDXL / FLUX 의 2 차 인코더 스케줄; 명시적인 2 차 텍스트는 스케줄되지 않는 한 고정됩니다.|
| `zoom` | `0:(1)` |프레임당 배율; 위의 1 확대, `(0,100]`|
| `angle` | `0:(0)` |프레임당 시계 반대 방향 회전(도)|
| `translation_x`, `translation_y` | `0:(0)` |프레임당 픽셀 변환; 긍정적인 움직임 이미지 내용을 오른쪽/아래로 이동|
| `strength_schedule` | `0:(0.35)` |`[0,1]` 의 Diffusers 디노이징 강도; 더 높은 값은 이전 이미지의 더 많은 부분을 변경합니다.|
| `cfg_scale_schedule` | `0:(guidance_scale)` |각 프레임의 가이드스케일; FLUX schnell 은 0를 필요로 합니다.|
| `noise_schedule` | `0:(0)` |255에 대한 독립적인 가우시안 픽셀 노이즈 표준편차, `[0,1]` 단위.|
| `contrast_schedule` | `0:(1)` |잡음 제거 전 음이 아닌 픽셀 승수|
| `seed_behavior` | `fixed` |`fixed`, `iter`(`seed_stride` 사용) 또는 재현 가능한 해시 기반 `random`|
| `border` | `replicate` |OpenCV `replicate`, `reflect` 또는 `wrap` 이미지 테두리|
| `color_coherence` | `none` |`RGB`는 첫 번째 출력 프레임의 채널당 히스토그램과 일치합니다.|
| `video_crf`, `video_preset` | 18, `medium` |FFmpeg libx264 품질 `[0,51]` 및 인코딩 속도 사전 설정|
| `ffmpeg`, `ffprobe` |PATH 의 실행 가능 이름|경로로 어느 실행 가능 항목을든 덮어쓰세요.|
| `encoding_timeout` |300 초|인코딩 및 디코딩 확인 시간 초과|

수치 스케줄은 상수 ( `"1.02"` ) 또는  `"0:(1), 48:(1.02)"` 를 허용합니다. 수치 앵커는 선형적으로 보간되며, 유지된 엔드포인트 값을 사용합니다. 현재 프레임에서 다음  키프레임 까지 평가되는 `t` 를 사용하는 값 표현식입니다. `max_f` 는  **번째 마지막 프레임 인덱스**,  `max_frames - 1` ;  `fps` ,  `pi` 와  `e` 도 사용할 수 있습니다. 표현식은  `+ - * / % **` , 단항 기호, 괄호 및  `sin` ,  `cos` ,  `tan` ,  `sqrt` ,  `exp` ,  `log` ,  `floor` ,  `ceil` ,  `abs` ,  `min` ,  `max` 를 지원합니다. 그들은  한계가 설정된 산술 해석기를 사용하며, 절대  Python   `eval` 를 사용하지 않습니다. 표현식은  1024 문자/128 문법 노드로 제한되며, 지수는  `[-32,32]` , 스케줄은  64   KiB 와 중간 값은 유한한  `+/-1e12` 로 제한됩니다. 중복, 소수점, 음수 또는 범위 밖의 프레임 키는 거부됩니다.

프롬프트 스케줄은  `"0"` 를 포함해야 하며, 각 프롬프트는 다음 키까지 유지됩니다. 인라인 Deforum  `--neg` /가중 프롬프트 표현식은 파싱되지 않으며: 명시적인 부정 스케줄을 사용하세요.  FLUX 부정 프롬프트는  `true_cfg_scale > 1` 를 필요로 합니다.  2 프롬프트 스트림과 모든 기본값은 내보낸 설정에서 유지되므로, 재생이 2 차 인코더  대체 경로 동작을 보존합니다.

`strength_schedule=0` 는 입력 이미지가 있는 경우 확산을 명시적으로 건너뜀; 리포트는 해당 프레임을  `warp-only` 에  0 샘플링 단계로 표시합니다. 초기 이미지가 없는 경우, 첫 번째 프레임은 항상 완전한 텍스트-이미지 패스를 사용합니다. 양의 강도는 최소 하나의 디노이징 단계를 생성해야 합니다. 가중치가 로드되기 전에 전체 타임라인이 확인되며, 이후의 표현식 실패도 포함됩니다.

애니메이션은  `num_images=1` 와  `.mp4` 또는  `.gif` 출력이 필요합니다. HiRes  수정된 잠재 표현/임베딩, 사용자 지정 시간 단계/시그마 배열 및 SDXL  조기 종료는 애니메이션과 결합할 수 없습니다. SD   1.5  및 ControlNet  img2이미지는 0이 아닌 `guidance_rescale`  값을 허용하지 않습니다. 이러한 조합은 명시적으로 실패합니다. 해상도, LoRA , 스케줄러, 샘플링 단계 및 정적 ControlNet  설정은 시퀀스에 의해 공유되며 프레임별 스케줄이 아닙니다.

<a id="artifacts-and-verification"></a>

## 아티팩트 및 검증

`build/deforum.mp4`  에 대한 번들에는 `deforum.mp4` , `deforum.json`  및 `deforum-frames/frame-000000.png`  이후의 항목이 포함됩니다. 카메라 변환 및 픽셀 노이즈는 CPU 에서 실행되며 디노이징은 선택된 장치에서 실행됩니다. 이전/참조 이미지만 메모리에 유지되며 PNG 는 임시 디렉토리로 스트리밍됩니다. FFprobe 는 인코딩된 비디오를 디코딩하고 게시 전에 H.264/yuv420p, 해상도, FPS , 지속 시간 및 정확한 프레임 수를 확인합니다. 보고서는 모델/로딩 식별자, 어댑터 출처, 초기 이미지 식별자, 각 프레임의 프롬프트, 시드, 소스/출력 픽셀 해시, PNG  해시, 실제 샘플링 시간 단계, 유한한 잠재 표현 및 출력 해시를 기록합니다. GIF 는 Pillow 로 디코딩되어 10  밀isecond 시간 분해능으로 정확한 프레임 수, 차원, 루프 및 지속 시간을 확인합니다. 시드는 하드웨어 또는 런타임  버전 간에 동일한 픽셀을 보장하지 않습니다.

기존 명시적 출력 경로는 `--overwrite` 가 제공되지 않는 한 실패합니다. 충돌 시 기본 경로는 새 번호가 붙은 접미사를 받습니다. 오버라이트는 오직 iiLocalDiffusion 애니메이션 번들로 표시된 프레임 디렉토리를만 바꿉니다. 샘플링 또는 인코딩 실패는 이전 번들을 그대로 유지하며, 파일 시스템 오류가 정상적으로 발생하면 출판 롤백은 이전 아티팩트를 복원합니다. 완성 보고서는 마지막에 게시됩니다. 프로세스 잠금 (@number@) 은 동일한 출력에 대한 동시 작성자를 방지합니다. 강제 프로세스 종료/전원 손실은 잠금 또는 임시 디렉토리를 남길 수 있습니다; 프로세스가 종료되었음을 확인한 후에만 해당 디렉토리를 제거해야 합니다. 이것은 재개 가능한 렌더링 형식이 아니거나 ACID 크로스 파일 시스템 트랜잭션이 아닙니다.

`DeforumOptionsTests`, `DeforumRuntimeTests` 및 `DeforumMediaTests`는 CTest에 등록되어 있습니다. 미디어 테스트는 선택 사항인 OpenCV/Pillow/NumPy 또는 FFmpeg 없이 통역사에서 건너뜁니다. 생성 환경을 사용하여 명시적으로 실행합니다.

```sh
reference/diffusers/.venv/bin/python tests/DeforumMediaTests.py
reference/diffusers/.venv/bin/python tests/DeforumDiffusersSmoke.py --device cpu
reference/diffusers/.venv/bin/python tests/DeforumDiffusersSmoke.py --device mps
```

연기는 로컬에서 작은 무작위 SD / SDXL 호환 safetensors 를 생성하고 ControlNet 를 포함하고 포함하지 않는 경우 모두 공개 CLI 를 사용합니다. 실제 샘플링, 피드백 해시 및 비디오 디코딩을 확인합니다. 훈련된 모델의 시각적 품질이나 CUDA /ROCm 하드웨어 지원을 확인하지 않고 실행을 확인합니다.
