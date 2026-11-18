<a id="architecture-aware-generation"></a>

# 아키텍처 인식 생성

공공 `iild-generate` 런처는 모델 컨테이너와 요청에서 실행 경로를 선택합니다. 로컬 체크포인트 텐서는 값 로딩 없이 검사되며, Diffusers 디렉토리는 `model_index.json` 를 통해 검사됩니다. 파일명은 아키텍처 증거가 아닙니다. Civitai 메타데이터는 변형을 이름 붙일 수 있지만, 인식된 텐서와 일치해야 합니다. 헤더 검사는 완전한 가중치 호환성 확인이 아니며, 성공적인 생성으로 보고되지 않습니다.

`--inspect-model`은 원본 또는 `model.diffusion_model.`, `diffusion_model.`,
`transformer.` 접두어 중 같은 이름 공간에 속한 MOSS Image 변환기 형상만 인식해 `moss-image`와
`MOSSImagePipeline`을 보고한다. 이는 모델 계열의 증거이며 네이티브 실행 계획이 아니다.
변환기 단일 파일에 텍스트 인코더와 VAE가 없으면 누락된 구성 요소로 표시한다.
네이티브 이미지 엔진은 MOSS를 실행하지 않는다. `--backend auto`는 이 서명이 확인된
로컬 체크포인트를 MOSS Diffusers 경로로 선택하고 고정된 `moss-image-sft` 설정을
제공한다. 명시적인 엔진·워크플로 선택은 유지하며, 호환되지 않는 네이티브 전용 옵션은
조용히 버리지 않고 거부한다. 알 수 없는 구조는 기존 경로의 판단을 유지한다.
MOSS 로더는 부동소수점 및 명시적으로 검증된 Comfy INT8 tensorwise ConvRot 변환기를
지원한다. 헤더와 양자화 메타데이터의 일치는 텍스트 인코더·토크나이저·VAE의 준비나
전체 가중치의 유효성, 실제 생성 성공을 증명하지 않는다.

C++의 `inspectImageModel`은 제한된 헤더와 필요한 패키지 인덱스를 읽어 구조·역할·양자화 표시·내장 구성 요소와
병합 출처의 합성 매핑 수를 별도 값으로 반환한다. 잘린 파일은 오류를 내고, 64 MiB보다
큰 헤더 등 판단 불가 상태는 `Unknown`으로 남긴다. 구성 요소의 존재 또는 인식은 실행
가능 여부를 뜻하지 않는다. 실제 양자화 알고리즘과 스케일은 워커가 별도로 검증한다.
모델 파일이나 병합 메타데이터는 수정하지 않는다.

독립된 safetensors 샤드 인덱스는 인덱스와 모든 구성 샤드의 텐서 매핑을
검증한다. 네이티브 엔진이 지원하는 이미지 체크포인트는 인덱스를 그대로 전달하며,
MOSS Image 서명이 확인된 변환기는 전용 Diffusers 경로에서 각 샤드를 읽는다.
MOSS의 FP 및 지원되는 INT8 ConvRot 텐서는 단일 파일과 동일하게 검사하며 열린
샤드 파일 수는 제한한다. 모델 캐시·준비 및 결과 게시 전 검사에는 샤드별 파일
식별자가 모두 포함된다. 원본은 수정하지 않는다. 다른 아키텍처를 명시적인
Diffusers 단일 파일 경로로 요청하면 검증한 샤드를 캐시에 병합하고 로컬 구성으로 로드한다.
비디오 워커도 동일한 인덱스 검사를 사용한다. LTX 체크포인트에 엔진이 사용하는
LLM 텐서 이름들이 이미 있으면 모델을 그대로 전달하고, 없으면 명시된 텍스트
인코더 또는 패키지에 포함된 구성 요소를 사용한다. SDK 패키지에서는 평탄화된
텐서 일부를 자체 완결된 모델로 오인하지 않고 패키지 인코더를 우선 복원한다.
독립된 인덱스에 내장 인코더가
없으면 호환 인코더를 명시해야 한다. 확인 시 표준 이름과 엔진이 변환하는 GGUF식
이름을 각각 인식하며 두 표기가 섞인 경우도 처리한다. 엔진은 전체 텐서 계약을 다시 검증한다.
모델과 인코더가 각각 인덱스인 경우 모든 샤드를 검증하고
총 바이트 수를 메모리 배치에 반영한다. 분류나 로딩 경로 제공은 모델의 실제 영상
생성 성공을 보증하지 않는다.

패키지 내 모델은 원본 경로와 텐서 매핑이 같은 구성 요소의 구조 서명을 확인한 후에만
선택한다. 인식된 패키지 자체를 단일 변환기 파일로 전달하지 않는다. 선택된 구성 요소는
원본 해시를 검증하여 별도 캐시에 복원하며, 중복되거나 모호한 후보는 명시적으로 거부한다.
새 패키지는 샤드 인덱스로 구성 요소를 묶는다. 기존 패키지에서는 내장 인덱스의
체크섬을 확인해 그룹을 복구하고, 복원 시 원본 샤드와 텐서 체크섬을 검사한다.
헤더 검사와 캐시 복원은 사전 학습 모델의 실제 생성 성공을 의미하지 않는다.

일반 이미지 체크포인트 패키지도 원본 구성 요소별 텐서 이름과 형상으로 판별한다.
다른 구성 요소의 LoRA나 ControlNet 텐서는 체크포인트 판별에 섞지 않으며,
분할 체크포인트는 인덱스가 지정한 샤드만 하나의 후보로 취급한다. 식별된 단일
후보는 원본과 체크섬을 확인해 캐시에 복원한 뒤 기존 로컬 실행 경로로 전달한다.
여러 체크포인트 후보가 있으면 임의로 선택하지 않으며, 일반 네이티브 체크포인트는
`--package-model`에 원본 구성 요소 또는 샤드 인덱스 경로를 지정하여 고를 수 있다.
패키지 작성자가 `main`으로 명시 지정한 후보가 정확히 하나이고 그 후보가 실제
체크포인트 서명과 일치하면, 별도 인자 없이 해당 후보를 사용한다. 자동으로 정해진
`main`이나 체크포인트가 아닌 파일은 여러 후보 중 하나를 고르는 근거가 아니다.
패키지에 여러 Anima 또는 MOSS 변환기가 있더라도 카탈로그는 각 원본을 독립적으로
검사한다. 명확하지 않은 인코더·VAE는 임의로 고르지 않고, 호출자가 명시한
구성 요소를 사용한다.
별도의 인코더와 VAE는
기존 명시적 구성 요소 설정을 유지하며, 형상 검사만으로 호환성이나 생성 성공을
보장하지 않는다.
일반 네이티브 이미지 패키지의 VAE는 체크포인트에 완전한 VAE가 없고 호출자가
VAE를 지정하지 않은 경우에만 자동 검색한다. 후보 헤더를 네이티브 엔진이 요구하는
전체 VAE 텐서 이름과 차원에 대조하며, 호환 후보가 정확히 하나일 때만 원본을
체크섬 검증 후 복원한다. 둘 이상이면 호출자가 원본 경로를 명시해야 하고,
계약을 알 수 없거나 일치 후보가 없으면 기존 백엔드 판단에 맡긴다. 이 검사는
임시 희소 파일에 구성 요소의 메타데이터와 필요한 양자화 서술만 복사해 수행하고
검사가 끝나면 임시 파일을 지운다. 희소 파일을 만들 수 없는 Windows 파일시스템에서는
대용량의 0 바이트를 할당하지 않고 자동 선택을 생략한다. 가중치 값, 텍스트 인코더
또는 실제 생성 품질의 검증은 아니다.
분리된 CLIP-L, CLIP-G, T5-XXL 인코더는 지원하는 SD1/2, SDXL, SD3,
FLUX.1 모델에서 네이티브 엔진의 슬롯별 전체 텐서 계약에 맞춰 검사한다. 슬롯에
완전한 인코더가 이미 있거나 호출자가 해당 슬롯을 지정한 경우 자동으로 교체하지
않는다. 한 슬롯에 호환 후보가 여러 개면 임의로 고르지 않고 원본 경로를 요구한다.
자동 검색은 패키지 카탈로그에서 표준 CLIP 또는 T5 텐서 서명이 확인되는
safetensors 구성 요소와 샤드 그룹에 한정한다. T5 자동 연결은 엔진의 표준 XXL
구조를 확인하며, 내장된 GGUF 등의 다른 자산은 기존의 명시적 구성 요소 지정으로
사용할 수 있다. 실제 가중치 값이나 토크나이저, 생성 결과의 품질은 별도 검증이다.
MOSS 자동 라우팅은 JSON 설정 파일의 모델·생성 옵션을 유지한다. 설정 파일에
명시된 모델 구성은 기본 프리셋으로 덮어쓰지 않으며, 해당 엔진에서 사용할 수 없는
네이티브 전용 옵션은 오류로 반환한다.

`--inspect-model`의 `embedded_package.component_paths`는 패키지 안의 원본
가중치 및 분할 인덱스 경로를 나열한다. 패키지 안에 별도로 들어 있는 네이티브
인코더와 VAE는 다음처럼 명시할 수 있다. 각 구성 요소는 원본 체크섬을 확인한 뒤
캐시에 복원된다. 같은 슬롯을 `--components`나 `--vae`로 중복 지정하면 오류다.
이 옵션은 로컬 네이티브 이미지 및 MOSS Diffusers 경로에서 사용할 수 있으며,
검사 전용 요청에서는 파일을 추출하지 않는다. MOSS에서 `llm`은 해당 텍스트 인코더를 뜻한다.
생성 결과의 `image_package.selected_components`에는 캐시 경로 대신 사용자가 지정했거나
자동으로 선택한 원본 패키지 내부 경로가 기록된다.

```sh
iild-generate --model /models/pack.safetensors --engine native \
  --package-components '{"clip_l":"Encoders/clip-l.safetensors","clip_g":"Encoders/clip-g.safetensors","vae":"VAE/vae.safetensors"}' \
  --prompt 'A cabin beside a lake'
```

<a id="local-image-backends"></a>

## 로컬 이미지 백엔드

`--backend local --engine auto` 는 기존 Diffusers SD1/SDXL 프레셋을 유지하며, Diffusers 구성이 요청되지 않은 경우 아래 다른 계약에 대해 프로세스 내 네이티브 엔진을 선택합니다. `--engine native` 는 명시적으로 모든 지원된 계약, SD1 및 SDXL 를 포함하여 네이티브 추론을 사용합니다. 명시적인 Diffusers 구성/패키지는 Diffusers 를 계속 사용합니다.

|건축|네이티브 텍스트 구성 요소 슬롯|VAE 계약|
| --- | --- | --- |
| SD 1.x | `clip_l` | SD1 |
| SD 2.x |`clip_l`(일치하는 OpenCLIP 가중치)| SD2 |
|SDXL, 일러스트리어스, NoobAI, 포니| `clip_l`, `clip_g` | SDXL |
| SD3 / SD3.5 | `clip_l`, `clip_g`, `t5xxl` | SD3 |
|FLUX.1 개발 / 슈넬 / Krea| `clip_l`, `t5xxl` | FLUX.1|
|FLUX.2 dev|`llm`(미스트랄과 일치)| FLUX.2|
|FLUX.2 Klein|`llm`(Qwen3 크기와 일치)| FLUX.2|
|Z 이미지 베이스/터보|`llm` (Qwen3)| FLUX.1|
|Qwen 이미지|`llm`(Qwen VL와 일치)|Qwen 이미지 RGB|
|크로마| `t5xxl` | FLUX.1|
| Krea 2 |`llm` (Qwen3-VL)|Qwen 이미지 RGB, 네이티브 Wan 텐서 표현|
|Anima| `llm` (Qwen3-0.6B-Base) |애니마 / Qwen 이미지 RGB|

이 어댑터는 고정된 stable-diffusion.cpp 의존성에 있는 실제 아키텍처 구현을 사용합니다. 호환되지 않는 텐서를 리사이즈하거나 모든 모델을 SDXL 로 변환하지 않습니다. 완전한 체크포인트는 임베딩된 인코더와 VAE 를 사용할 수 있습니다. safetensors / GGUF 파일은 명시적인 동반 파일을 받습니다. GGUF 양자화는 네이티브 엔진에 의해 디코딩되며, PyTorch 레거시 컨테이너는 생성 시 weights-only 변환 후 실행하며 원본과 변환본의 식별자를 확인합니다. Anima 디노이저는 원본, `net.`, `model.diffusion_model.`, `diffusion_model.`, `transformer.` 이름 공간을 정규화하고 실제 블록 수와 INT8 ConvRot 메타데이터를 유지합니다. 충돌하는 텐서 별칭은 거부합니다. 별도 인코더는 명시적 패키지 구성 요소 또는 `fallback_llms`의 Anima 항목에서 선택하며 전체 텐서 이름과 형상을 검증합니다. Qwen3-0.6B-Base 인코더와 Qwen 이미지 VAE가 분리된 모델에 필요하고, 리소스 카탈로그 항목 추가가 가중치의 자동 제공을 의미하지는 않습니다. GGUF 인코더와 확장자가 다른 safetensors 헤더는 파일 내용으로 판별합니다. Python 네이티브 진입점도 제한된 헤더를 우선 확인하여 확장자가 다르더라도 safetensors나 GGUF를 안전한 원래 리더로 전달하고, 레거시 체크포인트에는 기존 weights-only 검사를 유지합니다. macOS Metal에 INT8 tensorwise·ConvRot 행렬 커널이 없는 경우 해당 노드는 CPU 스케줄러로 넘기고 나머지 지원 연산은 Metal에 남긴다. macOS Python 워커는 실제 로드된 네이티브 라이브러리를 확인하고 다른 SDK 사본으로 리디렉션되면 경로와 DYLD 설정을 포함해 오류를 반환한다.

```sh
iild-generate --list-backends
iild-generate --inspect-model --model /models/z-image.safetensors
iild-generate --model /models/z-image.safetensors \
  --base-model ZImageTurbo \
  --components '{"llm":"/models/qwen3-4b.safetensors","vae":"/models/ae.safetensors"}' \
  --prompt 'A cabin beside a lake' --output-dir /outputs/cabin
```

`--inspect-model --components '{...}'` 는 추론을 시작하지 않고 선택된 네이티브 백엔드, 아키텍처, 내장/누락된 구성 요소 슬롯 및 기본값을 보고합니다. `--validate-only` 는 모델을 로드하지 않고 모든 로컬 파일 식별자를 해결합니다. 전경 작업자 준비는 실제 네이티브 컨텍스트를 로드하고 유효성을 검사합니다. 생성은 결과 RGB 이미지를 유효성 검사하고 게시합니다.

동일한 텐서 레이아웃은 Base/Turbo 또는 디스틸드 변형을 나타낼 수 있습니다. 아키텍처 서명만으로는 훈련 레시피를 확립할 수 없습니다. 가능한 경우 일치하는 `--base-model` 메타데이터를 제공하거나 명시적인 `--steps`, `--guidance-scale` 및 `--embedded-guidance` 를 제공하십시오. 네이티브 `guidance-scale` 는 표준 CFG 입니다; 내장 가이드는 별도의 플로우 모델 입력입니다. Krea 의 2 의 게시된 Diffusers 가이드는 `cond + scale*(cond-uncond)` 를 사용하므로, 네이티브 표준 CFG 는 그 값에 하나를 더한 것입니다. 명시적인 `--krea2-variant raw|turbo` 계약은 Raw 를 52 단계/네이티브 CFG 4.5 로, Turbo 를 8 단계/네이티브 CFG 1로 해결합니다. 알 수 없는 변형은 파일명에서 추론되지 않습니다. [Krea 2 컨트롤](krea2.md) 를 정밀도, 플로우 시프트, 시그마 일정표 및 네이티브 V3 를 위해 참조하십시오.

SD1/2/XL 는 `--prediction-type epsilon|v_prediction` 를 지원합니다; 인식된 예측 메타데이터는 네이티브 컨텍스트로 전파됩니다. 예측 유형을 변경하면 컨텍스트가 재구성됩니다. 플로우 아키텍처는 자체 네이티브 예측 계약을 유지합니다. Krea 2 의 `text_encoder_select_layers` 파이프라인 구성은 레이어 인덱스로 유효성 검사되며, 가져올 수 있는 모델 구성 요소로 오해받지 않습니다.

네이티브 정지 이미지 경로는 기본적으로 절반 크기 생성 뒤 Hires 보정을 수행하고 엔진 정렬 후 요청된 최종 크기로 자른다. Python의 `--no-hires-fix` 또는 C++ 컴포넌트의 `hires=false`는 명시적인 단일 패스를 선택한다. C V2/V3의 `hires` 값은 호출자가 지정하며 영으로 초기화된 구조체는 단일 패스다. 네이티브 워커에서 명시적인 `--hires-upscaler`와 `--hires-strength`는 보간 공급자와 보정 강도로 전달된다. 요청 가능한 출력은 축당 64~4096픽셀·8픽셀 배수이며 모델과 기기의 실제 메모리 성공을 보장하지 않는다. 지원되지 않는 편집, 인페인팅 및 RGBA /레이어 요청은 이미지/마스크 입력과 일치하는 Diffusers 파이프라인이 필요합니다; 이들은 아무런 알림 없이 텍스트로 이미지로 해석되지 않습니다.

<a id="pipeline-packages-and-output-publication"></a>

## 파이프라인 패키지 및 출력 게시

`model_index.json` 로 선택된 로컬 디렉토리는 설치된 내장 Diffusers 클래스를 선택합니다. 이는 기존 타입화된 입력/출력 시스템을 통해 추가 이미지, 비디오, 오디오 및 텐서 파이프라인을 포함합니다. `--inspect-model` 도 이러한 디렉토리를 받습니다. 패키지 검사 보고서는 구성 증거를 보고하며, 실제 클래스 가용성, 필요한 입력, 동반 텐서 및 출력 타입은 로더에 의해 검증됩니다. 사용자 다운로드 Python 코드는 실행되지 않습니다.

네이티브 워커는 PNG 파일, 이미지별 기원 및 `iild-standalone-image-v1` `generation.json` 를 게시합니다. 기원은 아키텍처, 백엔드 계획, 컴포넌트 식별자, 시드, CFG , 증류된 가이드, Hires 모드, 타이밍, 차원 및 출력 SHA-256 를 포함합니다. 최종 디렉토리는 전체 배치의 출력 검사가 통과한 후에만 게시됩니다. 기존 Diffusers 이미지/비디오/오디오/텐서 매니페스트는 기존 스키마 및 미디어 계약을 유지합니다.

<a id="c-and-c-interfaces"></a>

## C++ 및 C 인터페이스

`NativeModelComponents` 및 `generateNativeImageWithComponents` 는 명시적 CLIP-L /OpenCLIP, CLIP-G, T5, LLM 및 VAE 파일, CFG /증류된 가이드, CPU /자동 배치 및 준비만 실행을 추가합니다. 기존 C++ 구조 레이아웃 및 진입점은 변경되지 않습니다. `iild_native_request_v2` 는 V1 이미지 요청을 내장하고 이러한 옵션을 추가합니다. `iild_native_generate_v2` 는 V1 와 동일한 소유 결과, 메타데이터, RGB, 진행, 미리보기, 취소 및 자유 함수를 사용합니다.

모든 제공된 파일은 거주성 신원 확인 및 생성 후 변이 검사에 참여합니다. 동반자 바인딩을 변경하면 이전 컨텍스트가 무효화됩니다. 알려진 VAE 계약은 로딩 전에 검증되며, 자동 마운트 계약이 없는 새로운 가족은 명시적인 VAE 이 필요하고 전체 네이티브 컨텍스트 로더는 그 텐서를 검증합니다. 임의의 값으로 동반자가 대체되지 않습니다.

<a id="validation"></a>

## 검증

`BackendRegistryTests` 는 한계가 설정된 실제 safetensors 픽스처 픽스처를 원본과 Diffusers 텐서 이름, 구성 요소 완전성, 메타데이터 충돌, 재생 및 Python /C V2 경계용으로 사용합니다. `NativeResultTests` 와 `NativeMobileResultTests` 는 제어된 엔진 픽스처 에 대해 실제 C++ 어댑터를 컴파일하여 구성 요소 포워딩, 샘플링 매개변수, 캐시 무효화, 차원, 취소, 소유권 및 V1/V2 ABI 거부를 확인합니다. 이 테스트는 사전 학습된 모델의 품질을 측정하지 않습니다. 새 런타임 스모크 아티팩트는 `resources/outputs/verification/`에 유지되며, 모든 가능한 체크포인트가 실행되었다고 주장하는 대신 사용된 특정 가족과 가중치를 기록합니다. `NativeAnimaNamesTests`는 여러 이름 공간, 블록 수, 양자화 표시 및 인코더 텐서 계약을 safetensors와 Q8 GGUF 합성 메타데이터로 검사하며 선택적으로 실제 Anima 헤더를 읽을 수 있습니다. 실제 JANIMA29B V1.1 INT8 원본, Qwen3-0.6B-Base, Qwen 이미지 VAE로 512×512·1단계 네이티브 생성을 실행해 RGB PNG를 저장한 결과와 파일 불변 기록은 `resources/outputs/anima-native-verification-6ewcwvbo/`에 있다. 이 단일 단계 결과는 이미지 품질이나 다른 상용 모델의 호환성을 검증하지 않는다. 성공 실행 이후 확장자 판별 수정은 별도 회귀 테스트로 확인했다.

`tests/BackendRuntimeSmoke.py` 는 SD3 , FLUX.1 , FLUX.2 클라인, Z-Image, Qwen Image 및 Krea 2 를 공개 라우터를 통해 실제 작은 무작위 초기화 Diffusers 네트워크와 명시적인 조건부 텐서를 사용하여 실행합니다. 그것은 텐서 가족 검사, 실제 디노이징/디코딩, 상수 아님 64x64 RGB PNG 및 다운로드 없이 출력 매니페스트를 확인합니다. 현재 출력 경로 `resources/outputs/verification/backend-runtime-smoke/results.json`은 계산/내보내기 증거이며, 사전 학습된 품질이나 네이티브 엔진 벤치마크가 아닙니다.

참고 자료: [고정 네이티브 백엔드](https://github.com/leejet/stable-diffusion.cpp/tree/d04e8950c1ec8d30248cbe996682b3182fb1adf6),
[Z-이미지 파이프라인](https://huggingface.co/docs/diffusers/main/api/pipelines/z_image),
[Krea 2 파이프라인 및 CFG 규칙](https://huggingface.co/docs/diffusers/main/api/pipelines/krea2),
[FLUX.2 파이프라인](https://huggingface.co/docs/diffusers/main/api/pipelines/flux2).

MOSS 자동 라우팅은 확장자보다 파일 내용과 검증된 텐서 형상을 우선한다. 이름이 바뀐 safetensors와 PyTorch 체크포인트도 같은 MOSS Diffusers 경로로 선택하되 PyTorch는 안전한 weights-only 메타 검사를 거친다. 명시적인 백엔드 선택은 유지하며 GGUF는 safetensors로 취급하지 않는다. GGUF에 로컬 `model_config`를 제공한 요청은 CLI와 JSON 설정 모두 Diffusers 변환기 로더를 선택하며, 선언된 변환기 폴더의 GGUF를 직접 선택하면 상위 파이프라인 설정을 찾고, 해당 구조가 없는 기존 자동 요청은 네이티브 선택을 유지한다. 레거시 체크포인트의 전경 준비 단계에서는 대용량 변환을 시작하지 않고 생성 작업까지 미룬다. 형식 판별과 라우팅은 실제 추론 성공의 증거가 아니다.

워커는 C++가 판별한 내장 이미지 패키지의 구성 요소와 원본 경로를 보존한다. MOSS 변환기는 자동 선택 또는 명시적 Diffusers 경로로, Anima 변환기는 로컬 네이티브 경로로 보내며 필요한 구성 요소만 체크섬을 확인해 작업 캐시에 복원한다. CLI 또는 설정 파일에 명시된 구성 요소가 우선한다. 원본 파일의 식별자가 준비 또는 생성 중 바뀌면 출력 게시 전에 실패한다. `--inspect-model`은 인덱스와 구성 요소 경로를 보고하고, 패키지의 `--print-config`·`--validate-only` 및 전경 준비는 대용량 복원을 시작하지 않는다. 자동 MOSS 라우팅과 명시적 Diffusers 경로 모두 `--config`를 지원하며 CLI 값이 우선한다. C 인터페이스의 취소와 워커 진행률을 검사하며, 합성 패키지의 첫 작업 실패 후 다음 작업이 이어지는지 대체 엔진으로 검증한다. 이 검사는 사전 학습 모델의 실행이나 모든 형식의 호환성을 보증하지 않는다.
