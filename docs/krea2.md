<a id="iisacc-krea-2-model-ecosystem"></a>

# iisacc Krea 2 모델 생태계

Krea   2 Raw 와 Turbo 는 새로운 iisacc 이미지 워크플로우의 표준 생태계입니다. 이는 아키텍처 및 실행 계약이며, 기존 SDXL /Illustrious 병합을 Krea   2 로 다시 표시하거나 호환되지 않는 텐서를 Krea 가중치로 투영하지 않습니다. 기존 모델 계열은 기존 호출자에게 여전히 사용 가능합니다.

<a id="variant-and-precision-policy"></a>

## 변형 및 정밀도 정책

출판된 Raw 품질 예시는 52 단계와 Krea 가이드 3.5를 사용하며, Turbo 는 8 단계와 Krea 가이드 0를 사용합니다. Raw 는 상위 공급 측 추천이며 LoRA 훈련에, Turbo 는 추론에 사용됩니다. 이들은 기본값이며 잠금된 매개변수가 아닙니다. [공식 구현](https://github.com/krea-ai/krea-2) 을 참조하십시오.
[Diffusers 계약](https://huggingface.co/docs/diffusers/main/en/api/pipelines/krea2).

|제어|Diffusers 패키지|네이티브 체크포인트|
| --- | --- | --- |
|변형|`model_index.json` boolean `is_distilled` 또는 부재 시 명시적인 `--krea2-variant`|명시적인 `--krea2-variant raw\|turbo`; 자동은 Raw 품질 샘플링 프리셋을 사용함|
|원시 기본값|52 단계, 안내 3.5|52 단계, 표준 CFG 4.5|
|터보 기본값|8 단계, 안내 0|8 단계, 표준 CFG 1|
|정밀도|`--dtype float32` 기본값 또는 `bfloat16`; FP16 거부됨|소스 체크포인트 정밀도, 지원되는 양자화 포함|
|해상도|16의 배수, 16 – 8192 계약 한계; 장치 용량도 여전히 적용됨|8px 출력 그리드, 64 – 2048; 내부 캔버스는 64px로 반올림됨|
|샘플러| FlowMatchEulerDiscreteScheduler |`--native-sampler euler\|heun`; 자동은 Euler 로 해결됨|
|추가 노이즈 제거 통과|암시적 없음 HiRes 통과|암시적 HiRes 통과 없음|

Krea 가이드는 `cond + g * (cond - uncond)`; 네이티브 표준 CFG 은 `g + 1` 입니다. 0 Krea 가이드는 부정 조건화를 사용하지 않음. 부정 프롬프트를 탐색할 때 명시적인 nonzero Krea 가이드를 설정해야 함; 이는 출판된 Turbo 기본값에서 벗어남. 파일명 하위 문자열은 증류의 증거가 아님. 충돌하는 명시적인 변형과 패키지 메타데이터는 추론 전에 실패함

FP32 는 런타임 산술을 설명하며, 이미 INT8/FP8/GGUF 가중치에서 손실된 정보의 복구를 의미하지 않음. BF16 는 FP16 보다 더 넓은 지수 범위를 유지하지만 FP32 보다 더 적은 마니사 비트를 가짐. 높은 충실도 비교는 원래 양자화되지 않은 가중치, 동일한 시드, 조건부, VAE, 차원 및 샘플링 일정을 사용해야 함. 수치 확인이나 작은 픽스처 는 사전 학습된 이미지 품질을 보장하지 않음.

<a id="conditioning-and-vae"></a>

## 컨디셔닝 및 VAE

모델은 Qwen3-VL 텍스트 조건화와 16채널 Qwen Image RGB VAE, 팩터 8, 패치 크기 2 ( 64 패킹된 트랜스포머 입력 채널)을 사용합니다. 공식 인코더 선택은 12 탭된 레이어를 사용하며, 미세 조정 시 훈련된 선택과 인코더를 유지해야 합니다. 로더는 선택된 레이어 수를 트랜스포머와 인덱스를 인코더 깊이에 대해 확인합니다. 명시적 임베딩은 BSLC (배치, 시퀀스, 선택된 레이어, 채널)이며, 일치하는 부울 BS 마스크를 가집니다.

SDXL / FLUX VAE 은 이 VAE 를 대체할 수 없습니다. 네이티브 엔진은 공유 Qwen/Wan 텐서 표현을 `wan` 로 명명하며, 이는 임의의 Wan VAE 를 사용할 수 있는 권한이 아닙니다. 네이티브 엔진은 로딩 시 동반 텐서 호환성을 유효성 검사합니다. 네이티브 분할 가중치는 `--components` 와 `llm` 및 `vae` 를 사용합니다. Diffusers 는 기존 `--vae` 선택을 수락하며, 해당 경로는 Qwen Image RGB 유효성 검사기를 사용합니다. SDXL 미삭제 미결 LoRA 는 Krea 2에 주입되지 않습니다. 호환 가능한 명시적 어댑터는 기존 `--lora` / `--lora-scale` 로더와 활성화 검사를 사용합니다. 임의 SDXL, FLUX.1 Krea 또는 FLUX.2 어댑터는 Krea 2 어댑터가 아닙니다. 지원되지 않는 편집, ControlNet 및 인페인팅 입력은 실제 호환되는 파이프라인이 필요하며 텍스트로 이미지로 변환하는 컨트롤로 취급되지 않습니다.

<a id="flow-shift-and-schedules"></a>

## 흐름 교대 및 일정

`--krea2-mu`는 지수 이동(유한 0–4)을 재정의합니다. 이것이 없으면 Turbo는 1.15를 사용하고 Raw는 `image_tokens = (width / 16) * (height / 16)`와 함께 `0.5 + (image_tokens - 256) * 0.65 / (6400 - 256)`를 계산합니다. Diffusers에는 게시된 동적 지수 스케줄러 구성이 필요합니다. 관련되지 않은 스케줄러 재정의가 실패합니다.

`--sigmas '[1.0,0.75,0.4,0.1]'` 는 ( 0,1]) 내의 엄격히 감소하는 유한한 **** 시그마를 지정합니다. 그들의 개수는 명시적으로 동일한 단계 수가 제공되지 않는 한 단계를 설정합니다. 네이티브 CLI 는 `exp(mu)*s/(1+(exp(mu)-1)*s)` 를 적용하고 종단 0을 추가합니다. Diffusers 는 그 변환을 스케줄러에서 수행합니다. 네이티브 Krea 2 기본값 또한 명시적인 이동된 `1, …, 1/steps, 0` 일정을 제공하므로 상위 공급 측 이산 스케줄러 기본값은 아무런 알림 없이 변경할 수 없습니다. JSON 구성 재생은 변형, 뮤, 시그마 목록 및 네이티브 샘플러를 보존합니다.

추가된 C++ `NativeSamplingControls` / `generateNativeImageWithSampling`와 C `iild_native_request_v3` / `iild_native_generate_v3`는 V1/V2 ABI를 보존한다. V3 샘플러 값은 0=자동, 1=Euler, 2=Heun이며, `flow_shift=+infinity`는 상위 공급 측 기본값을 유지한다. **직접 전달하는 V3 사용자 지정 sigma는 이미 이동된 값**이며, 항목 수는 `steps + 1`이고 마지막 값은 0이다. 잘못된 크기·포인터·enum·NaN·순서·스텝 수는 거부한다. 스케줄은 요청별이며 가중치를 다시 적재하지 않는다.

<a id="examples"></a>

## 예

완전한 로컬 Diffusers 패키지:

```sh
iild-generate --model /absolute/local/krea2-turbo-package \
  --krea2-variant turbo --dtype float32 --device cpu \
  --prompt 'a vivid red flower in daylight' --seed 42 \
  --width 1024 --height 1024 --output-dir /absolute/empty/output
```

네이티브 호환 분할 체크포인트(실제 일치하는 동반 파일 선택):

```sh
iild-generate --model /absolute/local/krea2-turbo.safetensors \
  --engine native --krea2-variant turbo --native-sampler euler \
  --components '{"llm":"/absolute/local/qwen3-vl.safetensors","vae":"/absolute/local/qwen-image-vae.safetensors"}' \
  --prompt 'a vivid red flower in daylight' --seed 42 \
  --width 1024 --height 1024 --output-dir /absolute/empty/output
```

<a id="evidence-and-limits"></a>

## 증거와 한계

Diffusers 는 유한 입력 임베딩/잠재 표현, 로드된 정밀도, 마스크 및 잠재 패킹을 확인한 후 NaN /Inf 를 갖는 모든 소음 제거 콜백을 확인합니다. 실패 시 게시가 중단되며, 실패 상황에서도 임시 스케줄러/콜백 후크는 복원됩니다. `generation.json` 는 변형, 요청된 정밀도, 실제 구성 요소 데이터 유형, 해결된 mu, 실제 스케줄러 시그마 및 실행 단계 수를 기록합니다. 네이티브는 해결된 샘플링 요청과 소스 정밀도 정책을 기록하며, 상위 공급 측 인터페이스는 모든 잠재 텐서를 노출하지 않으므로 **** 단계별 유한 유효성 검사 주장을 `finite_latents_required=false` 하지 않습니다.

`Krea2ContractTests.py` 는 사전 검사 /기본 계약을 확인하고, `Krea2RuntimeTests.py` 는 실제 작은 무작위 Krea2 트랜스포머/ VAE /스케줄러 구성 요소를 사용하여 시드 반복성, 원본 음의 조건부, 유효한 mu 변경, 유효하지 않은 입력, NaN 거부 및 후크 복원을 확인합니다. `BackendRegistryTests.py` 는 네이티브 해결 및 재생을 확인하고, `NativeResultTests.cpp` 는 실제 어댑터에서 엔진 V3 매개변수 전달 및 V1/V2 동작을 확인합니다. `BackendRuntimeSmoke.py --families krea2` 는 실제 PNG 를 통해 공개 라우터를 확인합니다. 이 테스트는 전체 사전 학습 모델의 시각적 품질을 유효성 검사하지 않습니다.

<a id="embedded-comfyui-encoder-compatibility"></a>

## 내장형 ComfyUI 인코더 호환성

완전한 Krea2 체크포인트는 `text_encoders.qwen3vl_4b.transformer.*` 를 사용할 수 있습니다. 사전 검사 는 이 정확한 네임스페이스를 Krea2 의 임베디드 LLM 로 인식하며, 네이티브 로더는 이를 일반 텍스트, 비전 및 양자화 규모 변환 전에 `text_encoders.llm.*` 로 매핑합니다. 기존 파일은 다시 작성되지 않습니다. `BackendRegistryTests` 와 `NativeKrea2NamesTests` 는 두 경계를 모두 포함하며, 유사하지만 관련 없는 네임스페이스 이름은 이 임베디드 구성 요소로 인정되지 않습니다.

Bare native checkpoint 자동 모드 는 Raw 품질 샘플링 프리셋을 사용하므로 데스크톱 요청은 CLI 만 플래그를 필요로 하지 않습니다. 이는 훈련 변형 감지가 아닙니다: `variant_source=native-quality-default` 는 결정을 기록합니다. 명시적 Turbo 는 여전히 사용 가능하지만 파일명은 이를 선택하지 않으며, Diffusers 패키지는 엄격한 `is_distilled` 메타데이터 계약을 유지합니다.

ComfyUI Qwen 인코더는 확장된 FP8 E4M3/E5M2 가중치를 포함할 수 있습니다. Metal 는 이를 위한 저장과 변환을 지원하지만 고정 엔진에는 직접적인 FP8 행렬 커널이 없습니다. 기능 쿼리는 이제 직접적인 FP8 곱셈을 거부하여 Linear 가 BF16 로 캐스팅하고 저장된 스케일을 적용하게 합니다. `NativeFp8MetalTests` 는 실제 Linear 그래프를 통해 FP8 형식 모두를 실행하고 실제 Metal 백엔드에서 확장된 출력 값을 확인합니다.

<a id="quickgenerate-aspect-ratios"></a>

## QuickGenerate 종횡비

네이티브 Krea 2 는 Dreamscapes QuickGenerate 의 기존 출력 크기를 받습니다: 1024×1024, 1368×1024, 1024×1368, 1824×1024, 그리고 1024×1824입니다. 네이티브 엔진은 각 내부 캔버스 축을 64 픽셀로 올림하며, 리샘플링 없이 디코딩된 RGB 를 요청된 출력으로 중앙 자릅니다. 9:16 에 대해 이는 1024×1856 캔버스로 1024×1824 (각 수직 가장자리에서 16 픽셀) 으로 자릅니다. 해상도 의존적 Raw mu 는 내부 캔버스 토큰 수를 사용합니다. 보고서는 `output_size` , `canvas_size` , 그리고 `output_transform` 를 기록하며, 재생은 원래 출력 차원을 유지합니다. Diffusers 패키지 차원은 별도의 16px 계약을 유지합니다.

<a id="native-inference-diagnostics"></a>

## 네이티브 추론 진단

`IILD_NATIVE_TELEMETRY_DIR` 를 쓰기 가능한 디렉토리로 설정하여 네이티브 호출마다 하나의 `inference-*.jsonl` 트레이스를 유지합니다. Dreamscapes 는 연결된 Society 저장소에 `Models/.society-runtime/iiLocalDiffusion/diagnostics` 로 이를 설정합니다. 이 파일들은 임시 작업 정리와 워커 종료 후에도 생존하며, C 브릿지의 결과 메타데이터에는 `telemetry_path` 가 포함되어 Python 이미지 리포트에도 유지됩니다. 트레이스는 프롬프트나 이미지 픽셀을 기록하지 않으며, 모델 파일명과 엔진 진단 경로만 나타날 수 있습니다.

`iild-native-telemetry-v1` 는 모델 로드, 텍스트 인코딩, 노이즈 제거, VAE 디코딩, 포스트프로세스 경계를 기록하며, 완료된 단계/총 단계, 경과 시간 및 단계/단계 시간을 포함합니다. 새 가중치 세그먼트 로딩은 `weight-load` 작업 **으로** 현재 단계 내에서 수행되며, 노이즈 제거를 초기 모델 로딩으로 다시 라벨링해서는 안 됩니다. 각 가중치 로딩은 엔진이 반환할 때 시작/종료 시간, 경과 시간 및 바이트를 기록합니다. CPU 매개변수 스테이지는 실행 버퍼로 별도로 `weight-transfer` 로 표시되며, 경과 시간과 타겟 백엔드를 포함합니다. 5초 하트비트는 콜백이 완료되지 않더라도 마지막 관찰된 단계를 보존합니다. 그것들은 진행이 아닙니다. `step` 완료된 단계 수를 세는 경우: 단계가 3 로 변경되지 않은 다음 단계는 여전히 비행 중입니다. 지속 시간은 엔진 경계 사이의 호스트 벽 시간으로, 대기 및 전송을 포함하며 GPU 커널 프로파일링은 아닙니다.

`backend` 는 엔진의 실제 런타임 백엔드에서 얻은 해결된 모듈 런타임 백엔드입니다 (워밍 캐시 재사용 포함). 그의 범위는 `module-runtime-placement` 입니다; 그것은 해당 장치에서 실행된 모든 연산자가 **** 아님을 주장하지 않습니다. 자동 맞춤 저장소 결정과 관련 엔진 진단은 증거로 유지됩니다. 배치 누락은  `unobserved` 이며, 가정된  Metal  성공이 아닙니다. Op-level  대체 경로 /이동 횟수와  GPU  활용도는 측정되지 않습니다.

메모리 필드는 프로세스  RSS , 프로세스 수명 최대치  RSS , 현재 물리적 발자국 및 이 호출에 대한 Apple 의 샘플링된 최대 발자국과 사용 가능한 경우 시스템 전체 스왑 사용량을 구분합니다. 이들은 독점적인 모델/ GPU  할당 사항이 아니며, 단독으로 스래싱을 확립할 수 없습니다. 누락된 플랫폼 측정값은 생략됩니다. 느린 단계의 원인을 파악하려면 반복된 프로세스 샘플과 저장소 I/O 를 비교하십시오.

`native-completed` 는  RGB  생성/자르기 완료, 즉 발행을 의미하지 않습니다.  Python  워커는 소스 검증과  PNG  저장 ( `publication-start` ,  `publication-failed` ,  `output-published` ) 을 통해 동일한 트레이스를 계속합니다.  `performance.publication_ms`  리포트는 해당 구간을 측정합니다. 종료된 워커는 종료 이벤트가 없을 수 있으며, 이는 중단된/불완전한 트레이스이지 성공적인 이미지가 아닙니다. 기존 최종 크기/자르기 및 체크포인트 정밀도 계약은 변경되지 않았습니다.

검증:  `NativeTelemetryTests` ,  `NativeResultTests` ,  `NativeMobileResultTests` ,  `NativeImageTests` , 및  Dreamscapes ' 생성 워커 감시자  픽스처 는 트레이스  JSON , 단계/가중치 로딩 분리, 실제 배치 레이블, 실패, 발행 및 심박만 스토프를 포함합니다. 이들은 전체 사전 학습된  1024px   Krea2  생성을 벤치마크하지 않습니다.


<a id="weight-io-and-desktop-resource-policy"></a>

## 가중치 I/O 및 데스크탑 리소스 정책

구성요소 인식 Dreamscapes 작업자는 이제 추론 전에 소스를 익명 메모리에 스테이징합니다. [상주 작업자 실행](resident-worker-performance.md)를 참조하세요. 아래의 파일 지원 대량 읽기 동작은 레거시/비상주 API에 대해 유지됩니다.

네이티브 C++ 모델 참조는 이제 메타데이터 동일성 (경로, 크기, 타임스탬프, 파일/장치 동일성) 으로 기본값이며, 첫 번째 사용 전에 전체 체크포인트 해시를 피합니다. `IILD_MODEL_VALIDATION=content` 는 오프라인 무결성 워크플로우를 위한 전체 콘텐츠 유효성 검사를 유지하며, `metadata` 는 인터랙티브 기본값을 명시적으로 선택합니다. 생성 전/후 동일성 확인 및 헤더/텐서 유효성 검사는 활성화되어 있습니다. 이는 일반 편집/대체를 감지하며, 메타데이터 동일성은 콘텐츠 체크섬 또는 인증 보증이 아닙니다.

유닉스에서, 읽기 전용 맵된 가중치는 한계가 설정된 8 MiB `pread` 호출을 통해 모든 맵된 페이지를 `memcpy` 방식으로 요청하는 대신 복사됩니다. 텐서는 파일 오프셋 순서로 예약됩니다. Metal 공유 가중치 할당량은 CPU -맵핑된 매개변수 단계를 포함하여 직접 바이트를 받으며, 텐서 크기의 호스트 업로드 복사 없이 수행됩니다. CPU 변환 및 개인/장치 버퍼는 기존 변환/업로드 경로를 유지합니다. 쓰기 전용 개인 매핑은 memcpy 의미론을 유지하므로 메모리 내 LoRA 변경은 원래 파일 데이터에 의해 절대 덮어쓰여지지 않습니다. 짧은 읽기, 범위 실패 및 취소는 오류이며, 취소는 청크 간에 확인됩니다. 이 변경은 저장소 접근을 변경하며, 정밀도나 샘플링을 변경하지 않습니다.

macOS 는 사용 가능한 메모리에서  1   GiB 의 호스트 헤더룸을 뺀 값을 사용하며,  Metal 의 권장 작업 집합에서  256   MiB 를 뺀 값을 아래로 반올림하여  256   MiB 로 제한합니다. 엔진은 해당 할당량 내부에 컴퓨트 그래프 작업 공간을 예약합니다. 이는 사용 가능한  RAM 의 1/6 비율인 두 번째 비례 데스크톱 예약을 대체합니다. 이는 다른 앱의 메모리를 강제로 회수하거나, 보고된 제한을 초과하거나, 별도의  iOS   CPU-VAE 헤더룸 정책을 변경하지 않습니다. 과대확대 체크포인트는 여전히 분할 로딩을 사용합니다.

`NativeBulkReadTests` 는 여러 청크, 정렬되지 않은 범위, 경계, 사영 편집, 취소, 그리고 잘린 파일에 걸쳐 정확한 바이트를 확인합니다. `NativeMappedMetalStorage` 와 `NativeMappedPrivateStorage` 는 거주하는 17 MiB 가중치 그래프를 통해 Metal, CPU-to- Metal 스테이지 및 디스크 거주를 실행하며, 부분 스테이지 스왑 후 성공적인 재시도 후 취소도 포함합니다. `NativeWeightReadBenchmark FILE OFFSET BYTES bulk|mapped` 는 공유 Metal 저장소에 대한 실제 체크포인트 바이트에 대한 옵트인 프로브이며, 경과 시간, 처리량, 페이지 폴트 및 체크섬을 보고합니다. 그것은 엔드투엔드 생성 벤치마크가 아니며, 어떤 비교든 캐시 상태와 기타 I/O 를 보고해야 합니다.
