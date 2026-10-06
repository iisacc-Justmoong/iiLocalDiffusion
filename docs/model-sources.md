<a id="three-model-input-locations"></a>

# 3 모델 입력 위치

Python 생성 진입점(`reference/generate.py`, `iild-generate`로 설치)은 정확히 하나의 모델 소스를 허용합니다. 모델 위치와 생성 방법은 별도의 선택입니다. 클라우드 모델은 해당 공급자에 의해 원격으로 평가됩니다. 해당 가중치는 로컬 실행을 위해 다운로드되지 않습니다.

|입력| CLI |JSON / Python 값|실행|
|---|---|---|---|
|로컬 경로| `--model-path PATH` |`model_path`(표준 `model`)|기존 로컬 체크포인트/Diffusers 런타임|
|직접 API| `--model-api URL` | `model_api` |제공된 추론 엔드포인트에 POST|
|클라우드 모델| `--model-cloud OWNER/MODEL --model-provider PROVIDER` | `model_cloud`, `model_provider` |Hugging Face 추론 공급자가 이름으로 명시적으로 선택됨|

`--model` 는 **로컬** `--model-path` 의 별칭으로 남아 있습니다. 누락된 경로가 클라우드 모델인 것으로 추측하는 일은 절대 없습니다. 2 원격 입력은 로컬 모델 경로를 허용하지 않습니다. CLI 와 config 간의 혼합과 같은 혼합 소스 타입은 거부되며, 동일한 필드의 CLI 값은 config 를 덮어쓸 수 있습니다. 로컬 config 경로는 설정 디렉토리에 대해 해결되며, API URL 과 클라우드 ID 는 불투명한 식별자로 남아 있습니다.

`model_sources.ModelInput` 는 `kind` ( `local` , `api` , `cloud` ), `location` , `provider` , `token_env` , `family` , 그리고 `timeout` 를 기록합니다. C++ 메타데이터 인스펙터와 컴포넌트 실행기는 기존 계약을 유지하며, 이 기능은 네이티브 C++ HTTP 추론을 추가하거나 Python 모델 실행을 라이브러리로 이동시키지 않습니다.

<a id="commands"></a>

## 명령

```sh
# 기존에 로컬로 설치한 이미지 모델.
reference/diffusers/.venv/bin/python reference/generate.py \
  --model-path /absolute/path/image-diffusers --prompt 'A glass bottle.'

# 아래 계약을 구현하여 배포한 엔드포인트.
reference/diffusers/.venv/bin/python reference/generate.py \
  --model-api https://inference.example/image --model-token-env IMAGE_API_TOKEN \
  --prompt 'A glass bottle.' --width 512 --height 512 --output build/api-image.png

# OWNER/MODEL에는 선택한 공급자와 이미지 작업에 대한 실제 매핑이 있어야 한다.
reference/diffusers/.venv/bin/python reference/generate.py \
  --model-cloud OWNER/MODEL --model-provider fal-ai \
  --prompt 'A glass bottle.' --output build/cloud-image.png

# LTX를 먼저 실행한 뒤 기존 로컬 프레임 Interpolator를 사용하며 최종 결과는 24 FPS이다.
reference/diffusers/.venv/bin/python reference/generate.py \
  --model-api https://inference.example/ltx --model-token-env VIDEO_API_TOKEN \
  --model-family ltx --prompt 'A slow camera orbit around a glass bottle.' \
  --frames 49 --fps 24 --width 704 --height 480 --output build/api-video.mp4
```

위 도메인과 모델 ID 는 프로비저닝된 서비스가 아닌 플레이스홀더입니다. 클라우드 LTX 에 대해서는 `--model-api` 를 `--model-cloud OWNER/LTX-MODEL --model-provider PROVIDER` 로 교체하고 `--model-family ltx` 를 유지합니다. `text-to-video` 에 대해 해당 모델을 실제로 서비스해야 합니다. 카탈로그 항목이나 승인된 모델 ID 는 라이브 배포를 수립하지 않습니다.

토큰은 `--model-token-env NAME` 에서 실행 시 읽힙니다. 클라우드 입력 기본값은 `HF_TOKEN` 입니다. 직접 API 입력은 토큰 변수가 지정되지 않는 한 인증되지 않았습니다. 지정되었지만 부재/빈칸인 변수는 제출 전에 실패합니다. Config 와 사이더카는 변수 이름을 저장하며, 절대 그 값을 저장하지 않습니다. HTTPS 사용; 로컬 서버 및 테스트에는 HTTP 를 지원합니다. 엔드포인트 URL 의 자격 증명/쿼리 문자열은 거부됩니다. `--model-timeout` 는 86400 초까지의 유한한 양의 양수입니다 (기본값 300 ). `--print-config` 는 서비스 연결, 토큰 읽기, 텐서 로드 또는 가중치 다운로드 없이 유효성을 검사합니다.

동등한 이미지 JSON(`generate.resolve_request(values)`에서도 허용됨):

```json
{
  "model_cloud": "OWNER/MODEL",
  "model_provider": "fal-ai",
  "model_token_env": "HF_TOKEN",
  "prompt": "A glass bottle.",
  "width": 512,
  "height": 512,
  "output": "./cloud-image.png"
}
```

`generate.resolve_request`는 로컬 이미지의 경우 `(preset, args)`를 반환하고 원격 이미지의 경우 `(None, args)`를 반환합니다. 후자는 `remote_generation.generate_image(args)`를 통해 실행됩니다. 비디오 값은 `video_options.build_parser().parse_values(values)` 및 `resolve_options`를 사용합니다. `generate_video.py`는 로컬 또는 원격 평가자를 선택합니다. 해결된 구성은 동일한 스키마를 사용하여 재생할 수 있습니다.

<a id="api-and-provider-adapters"></a>

## API 및 공급자 어댑터

직접 API 입력은 임의 공급업체 REST API가 아닌 동기식 Hugging Face 스타일 추론 엔드포인트 계약을 구현합니다. 이미지/샷당 하나의 JSON POST를 보냅니다.

```json
{
  "inputs": "the prompt, including camera text for video",
  "parameters": {
    "width": 704,
    "height": 480,
    "num_inference_steps": 30,
    "guidance_scale": 3,
    "negative_prompt": "blurry",
    "seed": 42,
    "num_frames": 25,
    "frame_rate": 12
  }
}
```

`num_frames` / `frame_rate` 는 비디오에만 포함되며 **LTX 소스** 요청을 설명하며, 최종 보간 프레임 수/ FPS 를 포함하지 않습니다. 엔드포인트는 HTTP 200 와 원본 `image/png` 또는 `video/mp4` 바이트를 반환해야 하며, 최대 512 MiB 입니다. 작업- ID 폴링, 리디렉션, URL 결과 엔벨로프 및 OpenAI 스타일 이미지 API 는 이 프로토콜이 아닙니다. 실패하거나 시간 초과된 직접 요청은 자동으로 재제출되지 않습니다.

클라우드 입력은 기존 고정된 `huggingface-hub==1.29.0`를 사용합니다.
[`InferenceClient`](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client), 제공자 매핑 및 `text_to_image` / `text_to_video` 방법. 제공자 및 허브 모델 ID 는 별도로 제공되며, `auto` 제공자 선택은 사용되지 않습니다. 사용 가능한 작업, 네이티브 해상도, 프레임 수 제한 및 스케줄링은 배포에 따라 다릅니다. 표준화 비디오 클라이언트는 너비/높이/프레임 속도 매개변수가 없습니다: 요청과 일치하는 네이티브 해상도를 가진 배포를 선택하세요. 반환된 소스 프레임은 iiLocalDiffusion 의 요청된 타임라인에 배치되며, 제공자의 컨테이너 재생 속도는 최종 출력 속도가 아닙니다. 잘못된 차원 또는 소스 프레임 수를 가진 응답은 아무런 알림 없이 리사이징, 프레임 복제 또는 완료된 결과 게시 없이 실패합니다. 제공자 청구는 iiLocalDiffusion 외부이며, 자동화된 테스트는 유료 클라우드 생성을 제출하지 않습니다.

새 라이브러리가 추가되지 않습니다: 직접 HTTP 은 Python 의 표준 라이브러리를, 클라우드 추론은 `requirements-common.txt` 에 이미 고정된 유지 관리 중인 Apache-2.0 Hugging Face 클라이언트를 재사용하며, 디코딩/인코딩은 Pillow 와 FFmpeg 을 재사용합니다. 제공자별 오류는 인증 정보나 응답 바디를 되새기지 않고 요약됩니다. 설치된 클라이언트의 어댑터 이상의 제공자 커버리지가 암시됩니다.

<a id="generation-rules-and-verified-boundaries"></a>

## 생성 규칙 및 검증된 경계

이미지는 프롬프트, 부정 프롬프트, 해상도, 단계, 가이드, 시드 배치 및 PNG 출력을 지원합니다. 원격 호출은 로컬 스케줄러, 어댑터, 잠재 표현, 텍스트 인코더 또는 하드웨어 제어를 노출하지 않으며, 명시적으로 제공된 지원되지 않는 옵션은 추론 전에 실패합니다. 원격 모델은 소스 인수로 선택되므로 `--base-model` 와 로컬 프리셋 식별자는 이를 대체하지 않습니다.

기본 비디오 경로는 LTX 이어 최종 24 FPS 에서 프레임 인터폴레이터로 유지됩니다. 원격 비디오는 `--model-family ltx` 를 필요로 합니다: 이는 호출자의 선언이며 공급자 가중치의 로컬 확인이 아닙니다. 각 샷은 LTX 의 8n+1 소스 길이를 요청하고 결과를 디코딩하며, 계획된 소스 프레임을 유지하고 로컬 LTX 와 동일한 FFmpeg 운동 인터폴레이션 및 출력 검증을 적용합니다. 독립적인 텍스트 스토리보드 샷과 카메라 프롬프트 제어가 지원됩니다. 키프레임, 이전 샷 이미지 연속성과 잠재 표현 제어는 공급자 어댑터가 해당 정확한 계약을 준수할 때까지 로컬 LTX 평가자가 필요합니다.

FPS <= 12 또는 GIF 는 여전히 로컬 Deforum(또는 프롬프트/시드 인터폴레이터) 로 자동으로 라우팅됩니다. 이러한 이미지 애니메이션 모드에 원격 소스를 제공하는 것은 텍스트로 이미지를 만드는 API 가 그들의 잠재 표현 제어를 노출하지 않기 때문에 제출 전에 실패합니다. 명시적 `--backend video` 은 낮은 FPS 에서 LTX 을 유지하고 두 번째 단계를 건너뜁니다. GIF 는 LTX 출력입니다. 이 규칙들을 우회하기 위해 원격 제공자/모델이 대체되지 않으며, Higgsfield 와 제거된 호스팅된 Seedance 라우트는 여전히 제거됩니다.

출력 사이드카는 소스와 실제 디코딩된 미디어를 식별하며, 원격 가중치가 로컬에서 검증되지 않았음을 명확히 표시합니다. 비디오 번들은 수신된 소스 클립, 디코딩된 PNG 해시, 앵커, 샷 경계 및 최종 인코딩 검사를 보존합니다. 잘못된 응답이나 실패는 기존 출력을 보존합니다. 테스트는 실제 루프백 HTTP 이미지/비디오 전송, 인증 헤더, 실패, 원자적 게시 및 실제 FFmpeg 보간을 다룹니다. 클라우드 테스트는 시뮬레이션된 제공자 응답을 사용한 클라이언트 배포를 확인하며, 라이브 제공자/모델 가용성, 이미지 품질 또는 성공적인 청구된 추론을 설정하지 않습니다.
