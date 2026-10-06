<a id="architecture-aware-generation"></a>

# 아키텍처 인식 생성

공공 `iild-generate` 런처는 모델 컨테이너와 요청에서 실행 경로를 선택합니다. 로컬 체크포인트 텐서는 값 로딩 없이 검사되며, Diffusers 디렉토리는 `model_index.json` 를 통해 검사됩니다. 파일명은 아키텍처 증거가 아닙니다. Civitai 메타데이터는 변형을 이름 붙일 수 있지만, 인식된 텐서와 일치해야 합니다. 헤더 검사는 완전한 가중치 호환성 확인이 아니며, 성공적인 생성으로 보고되지 않습니다.

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
|Anima| `llm` |애니마 / Qwen 이미지 RGB|

이 어댑터는 고정된 stable-diffusion.cpp 의존성에 있는 실제 아키텍처 구현을 사용합니다. 호환되지 않는 텐서를 리사이즈하거나 모든 모델을 SDXL 로 변환하지 않습니다. 완전한 체크포인트는 임베딩된 인코더와 VAE 를 사용할 수 있습니다. safetensors / GGUF 파일은 명시적인 동반 파일을 받습니다. GGUF 양자화는 네이티브 엔진에 의해 디코딩되며, 레거시 pickle 컨테이너는 먼저 기존 안전한 변환 경로를 사용해야 합니다.

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

새로운 네이티브 계약은 요청된 출력 크기에서 단일 디노이징 패스를 사용합니다 (엔진 캔버스 그리드로 반올림된 후 정확한 요청된 크기로 로크됩니다). 기존 완전 Anima V1 라우트는 이전 Hires 동작을 유지합니다. C++ V2 컴포넌트 API 는 Hires 를 명시적으로 제어합니다. 지원되지 않는 편집, 인페인팅 및 RGBA /레이어 요청은 이미지/마스크 입력과 일치하는 Diffusers 파이프라인이 필요합니다; 이들은 아무런 알림 없이 텍스트로 이미지로 해석되지 않습니다.

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

`BackendRegistryTests` 는 한계가 설정된 실제 safetensors 픽스처 픽스처를 원본과 Diffusers 텐서 이름, 구성 요소 완전성, 메타데이터 충돌, 재생 및 Python /C V2 경계용으로 사용합니다. `NativeResultTests` 와 `NativeMobileResultTests` 는 제어된 엔진 픽스처 에 대해 실제 C++ 어댑터를 컴파일하여 구성 요소 포워딩, 샘플링 매개변수, 캐시 무효화, 차원, 취소, 소유권 및 V1/V2 ABI 거부를 확인합니다. 이 테스트는 사전 학습된 모델의 품질을 측정하지 않습니다. 실제 런타임 런타임 연기 아티팩트는 `build/` 에 유지되며, 모든 가능한 체크포인트가 실행되었다고 주장하는 대신 사용된 특정 가족과 가중치를 기록합니다.

`tests/BackendRuntimeSmoke.py` 는 SD3 , FLUX.1 , FLUX.2 클라인, Z-Image, Qwen Image 및 Krea 2 를 공개 라우터를 통해 실제 작은 무작위 초기화 Diffusers 네트워크와 명시적인 조건부 텐서를 사용하여 실행합니다. 그것은 텐서 가족 검사, 실제 디노이징/디코딩, 상수 아님 64x64 RGB PNG 및 다운로드 없이 출력 매니페스트를 확인합니다. 그것의 `build/backend-runtime-smoke/results.json` 는 계산/내보내기 증거이며, 사전 학습된 품질이나 네이티브 엔진 벤치마크가 아닙니다.

참고 자료: [고정 네이티브 백엔드](https://github.com/leejet/stable-diffusion.cpp/tree/d04e8950c1ec8d30248cbe996682b3182fb1adf6),
[Z-이미지 파이프라인](https://huggingface.co/docs/diffusers/main/api/pipelines/z_image),
[Krea 2 파이프라인 및 CFG 규칙](https://huggingface.co/docs/diffusers/main/api/pipelines/krea2),
[FLUX.2 파이프라인](https://huggingface.co/docs/diffusers/main/api/pipelines/flux2).
