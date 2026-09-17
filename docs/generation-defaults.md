# 전역 이미지 생성 기본값

`resources/generation-defaults.json`이 네이티브 C++ 및 Stable Diffusion 계열 Python 생성 경로의 공통 기본값이다. SDXL 계열에서 LoRA를 따로 선택하지 않으면 `addDetailAesthetic_v20_32.safetensors`를 강도 **1.0**으로 실제 적재·활성화한다. 커스텀 LoRA를 지정하면 해당 LoRA가 기본 LoRA를 대체한다. 네거티브 문장은 아래 학습 임베딩 토큰과 합쳐지며, 빈 네거티브 문장에도 기본 임베딩이 적용된다. 재실행 가능한 JSON을 다시 읽어도 토큰을 중복 추가하지 않는다.

| 원본 | 토큰 | 학습 벡터 수 | 인코더 |
| --- | --- | --- | --- |
| `(negative_v2 Color_Balance_Calibration_0.8).safetensors` | `iild_negative_color_balance` | 7 | CLIP-L / CLIP-G |
| `NDXL.safetensors` | `iild_ndxl` | 10 | CLIP-L / CLIP-G |
| `NegativeDynamics-neg.pt` | `iild_negative_dynamics` | 10 | CLIP-L |
| `negativeXL_D.safetensors` | `iild_negative_xl` | 16 | CLIP-L / CLIP-G |
| `negative_hand-neg.pt` | `iild_negative_hand` | 8 | CLIP-L |
| `negdetface-neg.safetensors` | `iild_negative_face` | 2 | CLIP-L / CLIP-G |
| `realisticvision-negative-embedding.pt` | `iild_negative_realisticvision` | 75 | CLIP-L |

네이티브와 프리셋/standalone/Deforum 경로의 SDXL에서는 7개·128개 벡터를 모두 사용한다. CLIP-L만 있는 3개는 원래 768차원 학습값을 그대로 사용하고 CLIP-G의 같은 토큰 위치를 0으로 채운다. 이는 고정 stable-diffusion.cpp의 동작과 같으며 새로운 CLIP-G 학습값을 만들어내는 변환이 아니다. SD 1.x에는 호환되는 CLIP-L 임베딩 3개만 적용한다. 범용 Diffusers 경로도 모델별 기본 LoRA를 선택한다. 다만 범용 경로의 임의 입력 계약에는 학습 네거티브 임베딩을 자동 주입하지 않으며, 이 기본 임베딩을 사용할 때는 프리셋/standalone/Deforum 경로를 사용한다. 외부 ComfyUI/원격/시간축 비디오 경로에는 이 기본값을 주입하지 않는다.

128개 벡터는 CLIP의 한 문맥을 넘는다. 네이티브와 Python은 75개 내용 토큰에 BOS/EOS를 붙인 문맥으로 나누고, 양·음성 조건의 문맥 수를 맞추어 이어 붙인다. SDXL pooled 조건은 첫 문맥을 사용한다. 마지막 임베딩을 포함한 학습 벡터가 토큰 잘림으로 사라지지 않는다. Python은 등록·토큰 확장·CLIP 실행에 기존 Diffusers, Transformers, PyTorch를 사용한다. 별도 추론 라이브러리는 추가하지 않았다.

## 호출과 우선순위

기존 단일 `fallback_lora` 항목을 유지하면서 `fallback_loras` 배열에 모델별 항목을 추가할 수 있다. 배열만 있는 명세도 지원한다. 각 항목은 `file`, `scale`, `families`, `sha256`, `size`를 가진다. 파일은 명세 디렉터리 안의 상대 경로이며 각 계열에는 기본값 하나만 지정한다. 같은 계열의 중복 지정, `*` 지정, 0 또는 비유한 기본 강도는 오류이다. 명시적으로 선택한 LoRA에는 비교용 강도 0을 허용한다.

| 계열 값 | 기본값 선택 기준 |
| --- | --- |
| `sd15` (`sd1`) | SD 1.x, 범용 경로는 CLIP hidden_size 768 확인 |
| `sd2` | SD 2.x, 범용 경로는 CLIP hidden_size 1024 확인 |
| `sdxl-base` (`sdxl`) | SDXL 생성·img2img·인페인팅 파이프라인 |
| `sd3` | SD3/3.5 파이프라인의 실제 구조와 호환되는 LoRA |
| `flux1` | FLUX.1, 프리셋의 `flux1-schnell`/`flux1-dev`와 같은 선택 키 |
| `flux2`, `qwen-image`, `z-image` | 해당 파이프라인/네이티브 엔진의 모델 계열 |
| `hunyuan-dit`, `chroma` | 해당 범용 Diffusers 파이프라인 |

계열 선택은 가중치 호환성의 보증이 아니다. 같은 계열 안에서도 모델 크기·변형별 LoRA는 실제 로더의 구조 검증을 통과해야 한다. SD1/SD2의 인코더 설정이 없으면 임의로 SD1이라고 추정하지 않는다. Python의 `generation_defaults.fallback_lora_status`는 `selected`, `explicit-override`, `disabled`, `not-configured-for-family`를 구분한다. 호환 기본값이 없는 모델에 SDXL 파일을 대신 주입하지 않는다.

현재 번들에는 SDXL에서 학습된 `addDetailAesthetic_v20_32` 하나가 있다. 이 파일을 SD/FLUX 전용 파일로 변환하거나 새로 학습한 것으로 취급하지 않는다. 다른 계열의 호환 LoRA를 명세에 등록하면 이미지 및 해당 프리셋의 Deforum에서 같은 기본값으로 사용한다. `--lora`를 직접 전달하는 경로도 동일하게 지원한다.

기존 `generateNativeImage*` 함수는 새 기본값을 자동 사용한다. 기존 요청·결과 구조체의 ABI는 유지한다. 커스텀 네거티브 문장·LoRA·리소스 위치는 별도 옵션 진입점으로 전달한다.

```cpp
iiLocalDiffusion::NativeGenerationOptions options;
options.negativePrompt = "blurred text";
// options.loras = {{"/absolute/path/custom.safetensors", 0.75f}};
auto result = iiLocalDiffusion::generateNativeImageWithOptions(request, options, cancelled);
```

Python의 기존 `--lora` / `--lora-scale`은 기본 LoRA보다 우선한다. 별도의 `--text-embedding`은 기본 임베딩과 함께 등록한다. `--negative-prompt-2`에 별도 문장을 지정해도 기본 토큰을 포함한다. `--default-modifiers`의 기본값은 true이다. 비교 실험에서만 `--no-default-modifiers` 또는 `NativeGenerationOptions::defaultModifiers = false`로 기본 가중치 없는 기준 출력을 명시적으로 선택할 수 있다. 이미 계산한 `--embeddings`는 텍스트 인코딩을 건너뛰므로 이 명시적 설정이 필요하다.

네이티브 캐시는 기본 리소스·커스텀 LoRA의 파일 정체성이 바뀌면 무효화한다. Python의 단일 이미지 캐시는 LoRA 강도와 임베딩 파일/토큰 선택을 구성 키에 포함하고, 준비된 어댑터를 중복 등록하지 않는다. 커스텀 네거티브 문장과 샘플링 파라미터만 바뀌면 준비된 모델을 재사용한다.

## 패키지와 재현

### VAE 자동 폴백

VAE를 명시하지 않고 모델에도 VAE 가중치가 없으면 아래의 호환 VAE를 자동 선택한다. 우선순위는 명시적 `--vae` → 모델 내장/로컬 설정에 동봉된 VAE → 계열별 기본 VAE이다. `--no-default-modifiers`나 네이티브 `defaultModifiers=false`에서도 필수 디코더 폴백은 유지한다.

| 모델 계열 | 기본 VAE 출처 | Diffusers 클래스 | 잠재 공간 | 라이선스 |
| --- | --- | --- | --- | --- |
| Qwen Image RGB | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image/tree/75e0b4be04f60ec59a75f475837eced720f823b6/vae) | AutoencoderKLQwenImage | 16채널 RGB | Apache-2.0 |
| SDXL 및 호환 체크포인트 | [stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae/tree/6f5909a7e596173e25d4e97b07fd19cdf9611c76) | AutoencoderKL | 4채널, scale 0.13025 | 모델 카드의 MIT 선언 |
| FLUX.1 schnell/dev/Krea/Fill/Control/Kontext | [diffusers/FLUX.1-vae](https://huggingface.co/diffusers/FLUX.1-vae/tree/da548cfb003bdeebaff6da0211fc8fbc67cb563a) | AutoencoderKL | 16채널, scale 0.3611, shift 0.1159 | 원본 FLUX.1 schnell Apache-2.0 |
| FLUX.2 및 Klein | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294/vae) | AutoencoderKLFlux2 | 32채널, 2×2 패치와 batch normalization | Apache-2.0 |

FLUX.1 공개 사본의 가중치 SHA-256과 설정 Git blob은 BFL `FLUX.1-schnell` 리비전 `741f7c3ce8b383c54771c7003378a50191e9efe9` 원본과 동일하다. 각 VAE의 고정 리비전·파일 크기·가중치 및 설정 SHA-256은 `resources/generation-defaults.json`에 기록한다. 기존 단일 `fallback_vae`는 Qwen에 유지하며, `fallback_vaes` 배열에 SDXL/FLUX.1/FLUX.2를 등록한다. 중복 계열·지원하지 않는 클래스·잠재 공간 불일치는 오류이다. 이 VAE의 라이선스와 별도로 사용자가 선택한 디노이저 모델의 라이선스가 적용된다.

```sh
# Diffusers 디렉터리: --vae 없이 계열을 자동 판별한다.
iild-generate --backend diffusers --model /path/to/sdxl-or-flux-directory --prompt "a red cube"
# SDXL 체크포인트에 텍스트 인코더는 있고 VAE만 없는 경우도 자동 폴백한다.
iild-generate --backend diffusers --preset sdxl --model /path/to/sdxl.safetensors --prompt "a red cube"
```

범용 로더의 명시적 교체는 `--vae /path/to/vae-directory`이며 `config.json`과 `diffusion_pytorch_model.safetensors`가 필요하다. 기존 SD/SDXL/FLUX 프리셋은 `--vae /path/to/vae.safetensors` 계약을 유지한다. 프리셋·standalone SDXL 체크포인트·범용 디렉터리/단일 파일에서 같은 기본 레지스트리를 사용한다. FLUX 단일 디노이저 파일은 텍스트 인코더·토크나이저 등 다른 필수 컴포넌트가 들어 있는 로컬 `--model-config`가 필요하다. VAE 폴백은 다른 누락 컴포넌트를 다운로드하지 않는다.

설정만 있는 `vae/` 폴더는 폴백을 사용한다. 일부 가중치나 샤드 인덱스가 있으면 해당 VAE 로딩 오류를 드러내고 다른 파일로 덮지 않는다. 내장 VAE가 있으면 기본 리소스 폴더가 없어도 내장 파일을 사용한다. Qwen Layered RGBA, SD1/2/SD3, Wan, 미확인 후속 파이프라인에는 다른 계열의 VAE를 주입하지 않는다. FLUX.1과 FLUX.2를 구분하며 채널 수만으로 호환성을 추측하지 않는다.

Python 로더의 부분 파일 오류 정책은 위와 같다. 네이티브에서는 전체 VAE 파라미터의 이름·형상을 실제 엔진과 대조하고, 불완전하거나 호환되지 않는 내장 VAE를 같은 계열의 기본 VAE로 대체한다. 단일 VAE 텐서 존재만으로 정상 마운트로 간주하지 않는다.

네이티브 `auto-mount-v2`는 입력 safetensors/GGUF를 읽어 모델 계열과 VAE 계열을 분리해서 판정한다. 상태는 정상 내장·누락·구조 불일치·지원 범위 밖으로 구분한다. 원본에서 Q8 사본을 준비했다면 실제 사용할 GGUF를 검사한다. `redLilyIllu_v10`처럼 완전한 SDXL 내장 VAE가 있으면 그 가중치를 사용하고, 없거나 일부가 손상되었으면 SDXL 기본 VAE를 선택한다. 모델 파일명이나 모델 목록의 표시 이름은 선택 기준이 아니다.

Anima는 `qwen-image`, Z-Image는 `flux1` VAE 계약으로 연결한다. 이 공유 관계는 [Anima 제작자 모델 카드](https://huggingface.co/circlestone-labs/Anima) 및 [고정 엔진의 Z-Image 문서](https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/docs/z_image.md)에 근거하며 기존 동봉 가중치를 재사용한다. 다른 모델에 이 별칭을 확대 적용하지 않는다. 이 네이티브 지원 범위가 Python Diffusers 파이프라인의 지원 범위를 바꾸지는 않는다.

선택한 외부 VAE는 컨텍스트를 만들기 전에 별도로 읽어 필요한 모든 파라미터를 검증한다. 외부 파일에 빠진 텐서를 모델의 내장 VAE로 보충하여 통과시키지 않는다. 설정 파일도 같은 패키지 안에 있어야 하고 클래스·RGB 채널·잠재 채널·SDXL/FLUX.1 스케일과 시프트·Qwen 평균과 표준편차·FLUX.2 패치와 정규화 설정을 검사한다. SD3와 FLUX.1처럼 파라미터 형상이 같은 경우도 있으므로, 형상 검증만으로 계열 간 호환성을 판정하지 않는다. 설정 파일의 정체성도 컨텍스트 캐시와 작업 종료 시 변경 검사에 포함한다.

SD1/2/SD3의 정상 내장 VAE는 검증 후 사용한다. 이 계열의 VAE가 누락되었으면 현재 패키지에 해당 기본 VAE가 없다는 오류를 반환한다. 미확인 구조와 Qwen Layered는 `unsupported`로 기록하며 기존 엔진의 로딩 경로를 유지하고 임의의 RGB VAE를 주입하지 않는다. VAE 선택이 성공해도 별도 텍스트 인코더 등 다른 필수 모델 구성요소가 없으면 생성은 시작할 수 없다.

네이티브 기존 함수는 공개 요청/옵션 구조체를 변경하지 않는다. stable-diffusion.cpp가 실제 safetensors/GGUF 텐서 메타데이터로 계열과 VAE 누락을 판별하고, 컨텍스트 생성 전에 해당 `vae_path`를 연결한다. 모델 정체성이 같으면 판별을 재사용한다. VAE 정체성은 캐시 키와 생성 후 변경 검사에 포함한다. Diffusers는 VAE 파일과 설정을 함께 추적하고 결과 메타데이터에 `fallback` 및 SHA-256을 기록한다. 두 런타임은 설치된 원본 가중치를 오프라인으로 읽는다. 외부 ComfyUI 사용자 워크플로우는 해당 노드 구성 규칙을 따른다.

`VaeDefaultsTests`, `NativeVaeFallbackTests`, `NativeVaeConfigTests`, `NativeResultTests`와 설치 소비자 검사는 누락·내장 우선순위·계열별 잠재 공간·외부 VAE 불완전·설정 불일치·리소스 변경·네이티브 이름 변환·재배치한 패키지 해시를 검증한다. `tests/FamilyVaeDiffusersSmoke.py` 및 `tests/QwenVaeDiffusersSmoke.py`는 작은 무작위 디노이저와 합성 조건을 사용해 실제 학습된 기본 VAE로 64×64 RGB를 생성하고, 생략/명시 실행의 이미지 SHA-256 일치를 검사한다. 이는 전체 학습된 디노이저의 이미지 품질 검증과 별개이다. `NativeVaeFallbackTests ... --decode`는 SDXL·Qwen·FLUX.1·FLUX.2·Anima·Z-Image의 실제 네이티브 VAE 디코딩을 검사한다. `NativeVaeFallbackTests --inspect /absolute/model.safetensors`는 실제 입력 파일의 모델 계열·VAE 계열·상태를 출력한다(0: 지원 범위 밖, 1: 정상 내장, 2: 누락, 3: 구조 불일치).

리소스 저장소가 아직 동기화되지 않은 경우에는 디렉터리·명세의 일반 파일 여부와 크기를 먼저 검사한다. 빈 디렉터리 또는 누락된 `generation-defaults.json`을 `file_size()`에 바로 전달하지 않는다. 필수 VAE 누락 오류에는 `qwen-image` 같은 필요한 계열과 리소스 설치·동기화 안내를 포함한다. `defaultModifiers=false`인 Anima 통합본의 정상 내장 VAE 경로는 리소스 명세를 요구하지 않는다. `NativeVaeConfigTests`는 누락 폴더·빈 폴더·명세 대신 디렉터리가 있는 경우를, `NativeResultTests`와 `NativeMobileResultTests`는 Anima 통합본 성공·외부 VAE 누락 시 추론 전 실패를 검사한다.

재다운로드는 다음 명령을 사용한다. 실행 시 자동 다운로드하지 않으며 패키지를 준비할 때만 수행한다.

```sh
reference/diffusers/.venv/bin/hf download Qwen/Qwen-Image \
  vae/config.json vae/diffusion_pytorch_model.safetensors LICENSE \
  --revision 75e0b4be04f60ec59a75f475837eced720f823b6 --local-dir resources/vae/qwen-image
reference/diffusers/.venv/bin/hf download stabilityai/sdxl-vae \
  config.json diffusion_pytorch_model.safetensors README.md \
  --revision 6f5909a7e596173e25d4e97b07fd19cdf9611c76 --local-dir resources/vae/sdxl
reference/diffusers/.venv/bin/hf download diffusers/FLUX.1-vae \
  config.json diffusion_pytorch_model.safetensors \
  --revision da548cfb003bdeebaff6da0211fc8fbc67cb563a --local-dir resources/vae/flux1
reference/diffusers/.venv/bin/hf download black-forest-labs/FLUX.2-klein-4B \
  vae/config.json vae/diffusion_pytorch_model.safetensors LICENSE.md README.md \
  --revision e7b7dc27f91deacad38e78976d1f2b499d76a294 --local-dir resources/vae/flux2
reference/diffusers/.venv/bin/python scripts/prepare_generation_defaults.py --check
reference/diffusers/.venv/bin/python tests/FamilyVaeDiffusersSmoke.py
reference/diffusers/.venv/bin/python tests/QwenVaeDiffusersSmoke.py
build/NativeVaeFallbackTests build/family-vae/native resources/vae --decode
```

### 배포 리소스

CMake 설치는 `share/iiLocalDiffusion/resources`에 명세·LoRA·safetensors 임베딩을 함께 설치한다. 네이티브는 라이브러리를 기준으로 설치 디렉터리와 Apple 앱 리소스 위치를 찾는다. `nativeGenerationResourceDirectory()`로 실제 위치를 조회할 수 있다. Python도 설치된 reference 코드에 상대적인 같은 리소스를 찾는다. 공통 환경 변수 `IILD_GENERATION_RESOURCES`와 호출별 `resourceDirectory` / `--generation-resources`로 위치를 지정할 수 있다. 필수 리소스가 없거나 파일 크기/형식이 다르면 조용히 누락시키지 않고 실패한다. Python은 명세의 SHA-256까지 검증한다.

설치된 CMake 패키지는 `iiLocalDiffusion_GENERATION_RESOURCES` 변수와 `iiLocalDiffusion_deploy_generation_resources(appTarget)` 함수를 제공한다. 데스크톱 실행 파일 또는 Apple 앱 번들에 리소스를 복사한다. Android에서는 APK assets를 앱 전용 파일 저장소에 복사한 뒤 해당 경로를 `resourceDirectory`로 지정한다. 네이티브 엔진에는 실제 파일 경로가 필요하며 APK 가상 경로를 파일 경로로 취급하지 않는다.

`.pt` 원본은 실행하지 않는다. `scripts/prepare_generation_defaults.py`는 기존 PyTorch의 `torch.load(..., weights_only=True, map_location="cpu")`로 텐서만 읽어 `resources/embeddings`에 safetensors를 만든다. 원본과 변환 파일의 해시를 명세에 기록한다. 다음 명령은 원본 텐서와 변환 결과가 정확히 같은지 검증한다.

```sh
reference/diffusers/.venv/bin/python scripts/prepare_generation_defaults.py --check
```

기본값 회귀 검사는 `GenerationDefaultsTests`, `LongClipConditioningTests`, `NativeResultTests`에 포함한다. `DefaultModifierDiffusersSmoke.py`는 실제 기본 임베딩과 작은 무작위 CLIP/UNet을 사용해 모든 토큰 전달 및 마지막 벡터가 조건·denoiser 출력을 바꾸는지 검사한다. 이 검사는 학습된 모델의 이미지 품질 평가와 구분한다.

기존 의존성을 재사용한다: [Diffusers Textual Inversion](https://huggingface.co/docs/diffusers/en/using-diffusers/textual_inversion_inference), [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp). 각 가중치의 원래 라이선스는 SDK 코드 라이선스와 별개이다.

## Transformers의 flat CLIP LoRA 호환

Transformers의 flat `CLIPTextModel`에는 `text_model` 하위 객체가 없지만 기존 SDXL LoRA에는 `text_encoder.text_model.encoder.*` 텐서와 alpha 키가 남아 있다. 이 조합은 Diffusers의 rank 검색 결과를 비워 `IndexError: list index out of range`로 로딩을 중단했다. SDK의 `lora_encoder_compatibility`는 해당 파이프라인의 로딩 호출 안에서만 텐서와 alpha의 `text_model` 접두사를 제거한다. 인코더 트리·가중치 값·alpha·LoRA 강도는 바꾸지 않으며, 여전히 중첩 구조인 `text_encoder_2`는 유지한다. 키 충돌은 오류로 처리하고 실패해도 로더를 복원한다. `EncoderCompatibilityTests`는 두 구조·alpha 보존·충돌·예외 복원을 검사한다. 실제 기본 LoRA의 메모리 절약형 SDXL 모듈 로딩 검증은 `build/quickgenerate-lora-meta-fixed.log`에 기록한다.
## 체크포인트 예측 방식과 프리뷰 단계

독립 SDXL 체크포인트는 JSON 메타데이터 없이 빈 `v_pred`와 `ztsnr` 텐서로 예측 방식을 표시할 수 있다. 로더는 검증한 텐서 헤더에서 `v_pred`를 읽어 v-prediction을 선택하고, `ztsnr`가 있으면 zero terminal SNR와 마지막 타임스텝에서 시작하는 trailing 간격을 적용한다. 충돌하는 예측 메타데이터는 거부하고 사용자가 명시한 스케줄러 설정은 유지한다. 파일명으로 예측 방식을 추측하지 않는다. 표시는 [ComfyUI SDXL 로더](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/supported_models.py), 노이즈 일정은 [Diffusers 스케줄러 안내](https://huggingface.co/docs/diffusers/main/en/using-diffusers/scheduler_features)를 따른다.

Diffusers와 네이티브 프리뷰 이벤트 모두 계속 증가하는 `sequence`를 포함한다. `step`과 `total_steps`는 현재 기본 생성 또는 Hires 보정 구간의 단계이며, 보정 구간에서 `step`이 1로 돌아가도 프리뷰를 버리지 않는다. Img2img의 scheduler에는 전체 일정이 남아 있을 수 있으므로 `total_steps`는 파이프라인의 실제 `num_timesteps`를 우선한다. 기본 10스텝과 strength로 선택한 보정 3스텝을 구분하여 모든 프레임을 수신한다.
