<a id="optional-managed-comfyui-checkpoint-backend"></a>

# 선택적 관리형 ComfyUI 체크포인트 백엔드

Dreamscapes 등 소비 앱은 `--work-dir <앱 임시 위치>/<job>/runtime`을 지정하여 작업 로그·워크플로·모델 링크·서버 중간 결과를 Society 밖에서 잠시 사용하고 정리한다. 큐와 실행 상태는 앱 메모리에만 둔다. 없거나 비어 있는 실제 디렉터리만 받으며 기존 내용과 리디렉션된 경로는 거부한다. 최종 실행기 결과는 기존 `--output-dir`로 따로 지정한다. Dreamscapes는 앱 전용 임시 위치에서 결과를 검증하고 이미지 파일만 `Generation History/` 바로 아래에 저장한다. 앱별 하위 폴더나 Asset Library 등록은 만들지 않는다. 옵션을 생략하면 기존 SDK `build/reference/local-image-jobs/` 배치를 사용한다. 원본 모델은 복사하지 않고 작업용 링크를 사용하며, `.safetensor`와 대문자 확장자는 이 링크에서 `.safetensors`로 정규화한다. 모델 원본 이름과 해시 검증은 유지한다. `LocalImageTests`가 Society 작업 경로·기존 내용 보존·링크 거부·확장자 정규화를 검사한다.

`reference/generate.py --backend comfyui-local --model /absolute/path/download.safetensors` 는 기존 로컬 가중치 파일을 명시적으로 검사하고 격리된 ComfyUI 프로세스를 시작합니다. 모델 패밀리에서 이미지 워크플로우를 조립하여 실행하고 디코딩된 이미지를 저장하며 원래 파일 해시를 기록합니다. 지원되는 자동 레시피를 위해 워크플로우 편집 또는 별도로 시작된 서버는 필요하지 않습니다.

<a id="install-once"></a>

## 한 번 설치

`--cache-dir`는 실행기 캐시 경로이다. Dreamscapes는 앱 전용 임시 작업 위치의 `cache/`를 전달한다. Python bytecode·런타임 캐시·작업 로그·중간 결과는 작업 종료와 함께 정리하며 Society에 영구 저장하지 않는다. 지정하지 않으면 기존 SDK 캐시 경로를 사용한다.

```bash
reference/diffusers/.venv/bin/python reference/setup_comfyui.py \
  --python reference/diffusers/.venv/bin/python
```

설치 프로그램에는 Git와 `uv`가 필요하다. ComfyUI `e80c1570b6b44a2557d5d8e341e05782d18c9bbb`와 ComfyUI-GGUF `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`를 `build/reference/`에 체크아웃하고, 의존성을 `build/reference/comfyui-venv`에 설치하며, 해석된 패키지를 `build/reference/comfyui-installed.lock`에 기록한다. 소스 리비전은 고정하고 lock은 플랫폼별 의존성 해석 결과를 기록한다. 기존의 다른 출처 또는 수정된 런타임 체크아웃은 덮어쓰지 않고 거부한다. 새 체크아웃은 fetch 성공 뒤에만 게시하므로, 다운로드 실패를 불완전한 런타임 소스 디렉터리 없이 재시도할 수 있다. 원래 Diffusers 환경과 C++ 라이브러리는 독립적이다.

전체 엔진은 HTTP 클라이언트보다 의존성이 훨씬 많으며 Torch, 비전/오디오 패키지, 토크나이저와 공식 프런트엔드를 포함한다. 가중치는 로컬에서 공급하며 이 명령은 소프트웨어 패키지만 다운로드한다. ComfyUI는 GPL-3.0이고 ComfyUI-GGUF는 Apache-2.0이다. 이들의 소스와 의존성은 라이브러리 패키지 밖에 유지한다.

<a id="bundled-checkpoint"></a>

## 번들 체크포인트

```bash
reference/diffusers/.venv/bin/python reference/generate.py --backend comfyui-local \
  --model /models/illustration.safetensors --base-model Illustrious \
  --prompt "a lighthouse above a calm ocean" --device mps \
  --output-dir build/reference/lighthouse
```

SD1/SD2/SDXL/SD3/FLUX 텐서 서명은 `--base-model` 없이도 아키텍처를 추론할 수 있습니다. 사이드카 `download.civitai.info` 는 Civitai 신원을 정확히 유지할 수 있으며, 필요시 SHA256 로 선택적으로 확인할 수 있습니다. 파일명은 신원을 증명하지 않습니다. 텐서 레이아웃이 인식되지 않는 아키텍처에 대한 카테고리를 제공하십시오. 감지된 텐서와 상충하는 명시적 카테고리는 거부됩니다.

NoobAI v-예측 체크포인트의 경우, 내장/사이드카 예측 메타데이터가 자동으로 사용됩니다. 그렇지 않으면  `--prediction-type v_prediction` 를 전달하세요; 모델의 샘플링 레시피가  0-종단  SNR 를 필요로 할 때는  `--zsnr` 를 전달하세요.  `--guidance-scale` 는 분류자 없는 가이드를 제어합니다.  FLUX dev/ Krea 의 별도의 내장 가이드는  `--embedded-guidance` 입니다 (기본값  3.5 ).  `--clip-skip` 는 기존 관행을 따릅니다:  0 는 마지막 레이어를 선택하며, 나머지 하나는 바로 앞 레이어입니다.

레거시 `.ckpt`, `.pt`, `.pth` 및 `.bin` 컨테이너는 로드하기 전에 캐시된 safetensors로 안전하게 변환됩니다. [변환 제한 사항](checkpoint-formats.md)를 참조하세요.

`--seed`를 생략하면 각 요청에 대해 새로운 무작위 기본 시드가 선택됩니다. 0을 포함한 명시적 시드는 보존됩니다. `local-image.json`는 선택한 시드를 `request.seed`에 기록하고 워크플로에서는 HiRes 패스에 동일한 시드를 사용합니다.

<a id="split-or-quantized-model"></a>

## 분할 또는 양자화 모델

디노이저에만 일치하는 텍스트 인코더와 VAE가 필요합니다. GGUF는 양자화를 유지하면서 설치된 네이티브 GGUF 로더를 선택합니다. 예를 들면:

```bash
reference/diffusers/.venv/bin/python reference/generate.py --backend comfyui-local \
  --model /models/krea.gguf --base-model 'Flux.1 Krea' \
  --vae /models/ae.safetensors \
  --text-encoder /models/clip_l.safetensors \
  --text-encoder-2 /models/t5xxl.gguf \
  --prompt "a lighthouse above a calm ocean" --device cuda \
  --output-dir build/reference/krea-lighthouse
```

`--components '{"vae":"...","text_encoder":"..."}'` 는 또한 명시적인 로컬 경로를 수락합니다. 인코더 입력은 `text_encoder_4` 까지 연속적입니다. `decoder` 는 Stable Cascade의 B 단계 노이즈 제거기를 공급하며, `model_negative` 는 Ideogram4 의 조건부 없는 노이즈 제거기를 공급합니다. 경로는 각 작업의 개인 모델 인벤토리에 있는 심링크를 통해 등록됩니다. 어떤 구성 요소 다운로드나 파일명 매칭도 수행되지 않습니다. [아키텍처 레시피 테이블](comfyui-image-workflows.md)을 참조하십시오.

`--model-type checkpoint|diffusion_model` 는 번들 파일과 분리된 파일에 대한 검사 중첩을 상쇄합니다. 명시적 `--backend preset` 은 원래 이미지 오라클의 LoRA, ControlNet, 텍스트 반전 및 HiRes 입력을 유지합니다. 명시적 `--pipeline-class` / `--pipeline-inputs` 는 일반적 Diffusers 을 선택하며, `--workflow` 는 기존 로컬 API 워크플로우를 선택합니다. 지원되지 않는 인수는 오류입니다.

<a id="evidence-and-execution-limits"></a>

## 증거 및 실행 한계

`--print-config` 다운로드만 검사합니다. `--validate-only` 엔진을 시작하고 샘플링 없이 노드/입력 가용성을 검증합니다. 추론을 증명하지 않습니다. 성공적인 샘플링은 이미지  `generation.json` 와  `local-image.json` 를 새 또는 빈 출력 디렉토리에 씁니다. 후자는 원래 모델과 구성 요소 해시, 변환 매니페스트, 감지된 아키텍처, 완전한 그래프, 런타임@, 장치 정보 및 이미지 차원을 기록합니다. 실행 후 해시 값이 다시 확인되며, 엔진에 실제로 전달된 스테이지된 심링크 타겟도 함께 확인됩니다. 공용 명령은 필요할 때 관리된 해석자로 위임합니다; 호스트 Python 는 Torch 나 Pillow 가 필요하지 않습니다. 로그, 노드 스키마 및 준비된 경로는 `build/reference/local-image-jobs/` 에 유지됩니다.

관리 프로세스는 루프백에서만 경청하며, 호스팅된 API 노드를 비활성화하고 선택된 GGUF 확장만 로드합니다. 완료 또는 실패 시 중지됩니다. `--device auto` 는 가속기가 필요합니다. CPU 는 명시적으로 선택되어야 합니다. `--runtime-source` 와 `--runtime-python` 는 독립적으로 설치된 호환 런타임 를 허용합니다. `--timeout` 와 `--startup-timeout` 는 바운드 실행/시작을 허용합니다.

자동 빌더는 문서화된 텍스트-이미지 아키텍처를 모두 포함합니다. 이미지 조건부 모델은 입력 이미지와 해당 파이프라인 또는 워크플로우가 필요합니다. LoRA, VAE, 임베딩, ControlNet 및 업스케일러는 구성 요소이며, 오디오와 3D 가중치는 독립적인 텍스트-이미지 생성자가 아닙니다. 카테고리, 유효한 컨테이너 또는 기존 노드는 다운로드된 모든 가중치와 올바른 샘플링을 보장하지 않습니다. 실제 검증 증거](civitai-validation.md)를 [하세요.

출처: [ComfyUI 소스](https://github.com/Comfy-Org/ComfyUI/tree/e80c1570b6b44a2557d5d8e341e05782d18c9bbb),
[GGUF 로더 소스](https://github.com/city96/ComfyUI-GGUF/tree/6ea2651e7df66d7585f6ffee804b20e92fb38b8a),
[공식 설치](https://docs.comfy.org/installation/manual_install).
