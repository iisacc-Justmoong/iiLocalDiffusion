# 로컬 ControlNet 리소스 실행 조합

이 문서는 `iiLocalDiffusion` 프로젝트 루트에서 세 모델 계열의 고정 리소스와 기존 전처리기를 명시적으로 조합하는 예제다. `resources/inputs/reference.png`는 사용자가 작업공간 안에 준비한 이미지로 바꾸고 출력 디렉터리는 실행마다 새로 지정한다. 다음 명령의 파일 경로는 리소스 공급이 완료된 경우에만 유효하다. 모델 가중치는 생성 중 자동으로 다운로드되지 않는다. 파일별 출처, 크기, 해시 및 공급 명령은 [리소스 문서](../resources/README.md)에 있다.

먼저 `IILD_PYTHON_EXECUTABLE`을 이 프로젝트의 Diffusers 런타임 의존성이 설치된 Python 실행 파일로 설정한다. 아래 명령은 시스템 `python`이나 `python3`가 해당 패키지를 갖고 있다고 가정하지 않는다. 예를 들어 2026-10-31 현재 이 작업공간에서 시험에 사용한 실행 파일은 `build/merge-python/bin/python`이다. 이 경로는 보존된 환경을 가리키는 로컬 링크이므로 다른 작업공간이나 클린 빌드 후에도 존재한다고 가정하지 말고 환경을 확인해 선택한다. 같은 시점의 `reference/diffusers/.venv`는 존재하지 않는 Python을 가리켰으므로 사용 가능한 것으로 간주하지 않았다.

```sh
export IILD_PYTHON_EXECUTABLE="$PWD/build/merge-python/bin/python"
```

아래 예제는 동일한 파일, 입력 이미지, 프롬프트, 시드와 설정을 다시 선택할 수 있게 하는 호출 명세다. 합성 모델과 실제 리소스 파일의 구조·로딩 검사를 바탕으로 작성했으며, 이 문서의 전체 사전 학습 조합으로 생성한 이미지의 품질이나 다른 구현·장치에서의 비트 단위 일치를 검증한 것은 아니다. BF16 지원, 메모리 사용량과 실행 속도는 장치에 따라 다르다. CPU 스트리밍은 전체 프로세스 메모리 한도를 보장하지 않는다.

2026-10-31 깊이 모델 입력 정밀도 변경 후 Krea 2, FLUX.1, MOSS 각각의 소형 합성 모델 통합 테스트를 다시 실행했다. 각 테스트에서 네이티브 Canny 또는 소형 Depth Anything 힌트의 변화가 제어 출력에 반영되고 같은 시드의 출력이 반복됨을 확인했다. 해당 테스트는 깊이 추정 모델에 FP32와 FP16을 각각 선택하는 경우를 포함한다. 이 검사는 사전 학습 가중치나 다른 장치에서 같은 이미지가 생성된다는 증거는 아니다.

## Krea 2 Turbo와 상대 깊이

Krea 2 Turbo FP8과 깊이 Control-LoRA, Qwen3-VL 텍스트 인코더 및 Qwen Image VAE를 준비한다. 아래 Large 깊이 추정 가중치에는 CC-BY-NC-4.0이 적용된다. 상업적 사용에 적합한 Apache-2.0 Small 모델을 선택할 수 있지만 Large와 다른 모델이므로 깊이 결과가 달라진다. Krea 베이스와 Control-LoRA에는 Krea 2 Community License가 적용된다.

```sh
"$IILD_PYTHON_EXECUTABLE" reference/diffusers/generate_any.py \
  --model resources/models/krea2_turbo_fp8_scaled.safetensors \
  --model-config-preset krea2-turbo --krea2-variant turbo --krea2-fp8-storage cpu-stream \
  --controlnet resources/models/krea2-depth-control-lora.safetensors \
  --control-image resources/inputs/reference.png --control-preprocess depth \
  --control-depth-model resources/depth-estimators/depth-anything-v2-large --control-depth-dtype float16 \
  --prompt 'a sunlit studio' --width 1024 --height 1024 --steps 8 --guidance-scale 0 \
  --seed 42 --dtype bfloat16 --offload none --no-default-modifiers \
  --output-dir resources/outputs/reference/krea2-depth
```

텍스트 인코더와 VAE를 별도로 지정하지 않고 패키지에서도 생략한 경우 기존 카탈로그의 호환 리소스로 대체한다. 생성 보고서에서 실제 선택과 파일 식별자를 확인한다. 깊이 추정 모델에 전달하는 부동소수점 입력은 `--control-depth-dtype`과 같은 정밀도를 사용하며 정수 입력은 정수형을 유지한다. 추정된 깊이 입력의 부동소수점 값은 8비트로 바꾸지 않고 제어 VAE에 전달한다.

Turbo FP8 체크포인트의 256개 양자화 계층 중 96개는 `full_precision_matrix_mult`를 명시하고 160개는 명시하지 않는다. 이 플래그의 `full precision`은 FP32 강제가 아니라 양자화된 가중치를 연산 정밀도로 복원해 행렬곱한다는 뜻이다. iiLocalDiffusion은 모든 FP8 계층에서 복원 후 연산하며, 플래그가 없는 계층에서 ComfyUI가 가속 FP8 연산을 선택한 경우와 수치가 다를 수 있다. 생성 보고서의 `model.single_file_loading`은 두 계층 수와 실제 연산 방식을 기록한다. 플래그 의미는 [ComfyUI 양자화 문서](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md)를 참고한다.

같은 Turbo 베이스에서 아래처럼 고정 포즈 LoRA를 선택할 수도 있다. `resources/inputs/openpose.png`에는 이미 준비한 포즈 맵이 필요하다. 이 경로는 자동 포즈 추출을 제공하지 않는다. 어댑터의 Apache-2.0 표기와 베이스 모델의 Krea 2 Community License를 각각 확인한다.

```sh
"$IILD_PYTHON_EXECUTABLE" reference/diffusers/generate_any.py \
  --model resources/models/krea2_turbo_fp8_scaled.safetensors \
  --model-config-preset krea2-turbo --krea2-variant turbo --krea2-fp8-storage cpu-stream \
  --controlnet-format krea2-ostris-edit --controlnet resources/models/krea2-turbo-openpose-controlnet.safetensors \
  --control-image resources/inputs/openpose.png --controlnet-scale 0.9 \
  --prompt 'a dancer in a sunlit studio' --width 1024 --height 1024 --steps 10 --guidance-scale 0 \
  --seed 42 --dtype bfloat16 --offload none --no-default-modifiers \
  --output-dir resources/outputs/reference/krea2-pose
```

2026-10-31 작업공간에서 Turbo 베이스 파일의 전체 SHA-256과 686개 텐서 기술 정보, 256개 FP8 배율 및 일부 텐서 값을 확인했다. 실제 Turbo 파일과 기존 깊이·포즈 어댑터 및 보조 리소스를 선택하는 사전 검사도 통과했다. 실제 Turbo 트랜스포머를 BF16 연산용 CPU 스트리밍으로 로드해 256개 양자화 계층과 변환된 가중치의 유한성을 검사하고 각 계층의 원본 FP8 가중치 전체를 바이트 단위로, 배율은 값으로 비교했다. 나머지 174개 파라미터도 필요한 형상·정밀도 변환 후 원본과 일치했다. 실제 텍스트 전용 Qwen·토크나이저·VAE·스케줄러를 함께 로드한 파이프라인에서 깊이 어댑터 224개, 포즈 어댑터 256개의 대상 형태도 확인했다. Diffusers는 텍스트 전용 Qwen 타입에 대해 경고를 출력했다. 별도로 포즈용 실제 Qwen3-VL 언어·비전 인코더 전체를 로드해 선택된 비전 가중치를 원본과 비교했고, 해당 인코더와 Turbo를 함께 로드한 포즈 파이프라인에서 256개 포즈 대상 형태를 다시 확인했다. 테스트용 잠재값을 사용한 별도 검사에서는 실제 깊이·포즈 어댑터 후크를 등록하고 예외 발생 후 모두 복원했다. 깊이 예측, 어댑터의 전방 연산, 전체 파이프라인의 추론이나 이미지 품질은 확인하지 않았다.

## FLUX.1 Schnell과 네이티브 Canny

Schnell FP8 통합 체크포인트와 InstantX Dev Canny ControlNet을 준비한다. 이 ControlNet은 **비상업적 라이선스**이며 Dev 기반 학습 가중치다. 제작자는 Schnell에 로드할 수 있으나 성능을 보증하지 않는다고 명시한다. 아래 `guidance-scale`은 guidance 입력을 요구하는 제어 모델에도 전달되며 Schnell 베이스에는 guidance 임베딩을 사용하지 않는다.

```sh
"$IILD_PYTHON_EXECUTABLE" reference/diffusers/generate_any.py \
  --model resources/models/flux1-schnell-fp8.safetensors --model-config-preset flux1-schnell \
  --flux-transformer-storage cpu-stream --flux-t5-fp8-storage cpu-stream \
  --controlnet resources/models/flux1-instantx-canny --control-image resources/inputs/reference.png \
  --control-preprocess canny --prompt 'a city at night' --width 1024 --height 1024 \
  --steps 4 --guidance-scale 3.5 --seed 42 --dtype bfloat16 --offload none --no-default-modifiers \
  --output-dir resources/outputs/reference/flux1-canny
```

네이티브 Canny 공유 라이브러리를 자동으로 찾지 못하면 `--control-preprocessor-library`로 빌드되었거나 설치된 iiLocalDiffusion 라이브러리를 지정한다. 단일 파일에 내장된 텍스트 인코더와 VAE를 우선 사용한다.

2026-10-31 작업공간에서는 InstantX 설정·어댑터의 전체 해시와 188개 텐서 기술 정보 및 일부 값을 검사했다. 실제 어댑터 전체를 Diffusers 구성 요소로 로드해 모든 매개변수의 유한성과 메타 장치의 Schnell 구조에 대한 호환성 검사도 통과했다. 실제 Schnell 단일 파일은 전체 SHA-256 검증을 통과했으며, 해당 파일과 Canny 어댑터 및 네이티브 전처리기를 선택하는 사전 검사도 통과했다. 실제 Schnell 트랜스포머는 BF16 연산용 CPU 스트리밍으로 로드해 모든 가중치의 유한성과 원본 대응값을 검사했다. 결합 Q/K/V·MLP 분할과 최종 변조 교환을 포함하여 스트리밍 가중치는 원본 FP8 바이트와, 나머지 매개변수는 BF16 변환값과 전부 비교했다. 실제 내장 T5·VAE와 Canny ControlNet을 포함한 파이프라인 결합도 통과했으며 서로 다른 guidance 설정을 확인했다. 실제 전처리·추론 또는 이미지 품질은 검증하지 않았다.

## MOSS-Image SFT와 색상·깊이

MOSS SFT 트랜스포머, INT8 ConvRot Qwen3.5 인코더, 색상·깊이 어댑터와 FLUX.2 VAE를 준비한다. 아래 예제에서는 같은 사진을 색상 원본과 깊이 추정 원본으로 사용한다. 두 입력은 서로 다른 파일로 지정할 수도 있다. 어댑터의 학습용 중앙 크롭·리사이즈에 맞추려면 `training`을 선택한다.

색상 맵만 사용하는 요청은 `--control-color-image`만 지정할 수 있다. Canny 또는 깊이 전처리를 선택하면 전처리할 원본을 `--control-image`로도 지정해야 하며, 색상 맵에는 해당 전처리를 적용하지 않는다. Krea 2의 색상·깊이 어댑터에도 같은 입력 규칙이 적용된다.

```sh
"$IILD_PYTHON_EXECUTABLE" reference/diffusers/generate_any.py \
  --model resources/models/moss_image_sft_bf16.safetensors --model-config-preset moss-image-sft \
  --moss-transformer-storage cpu-stream --moss-text-encoder resources/text_encoders/moss_qwen35_int8_convrot.safetensors \
  --controlnet resources/models/moss_image_color_depth_control.safetensors \
  --control-image resources/inputs/reference.png --control-color-image resources/inputs/reference.png \
  --control-preprocess depth --control-depth-model resources/depth-estimators/depth-anything-v2-small \
  --control-map-preprocess training --prompt 'a sunlit studio' --width 1024 --height 1024 \
  --steps 50 --guidance-scale 3 --seed 42 --dtype bfloat16 --offload none --no-default-modifiers \
  --output-dir resources/outputs/reference/moss-color-depth
```

기존 FLUX.2 VAE가 패키지에 없을 때 로컬 카탈로그 대체 리소스로 선택된다. INT8 텍스트 인코더는 BF16 인코더와 같은 출력을 보장하지 않는다. 생성 보고서의 `control_preprocessor.output_identity`와 `control_execution.vae_input_hints`는 각각 전처리 결과와 VAE에 전달된 조건을 비교하는 데 사용한다.

2026-10-31 작업공간에서는 MOSS SFT 파일의 전체 해시·텐서 기술 정보와 어댑터의 전체 해시·값을 각각 검사했고, 위 구성에 해당하는 실제 로컬 리소스 선택을 메타데이터 모드에서 확인했다. 실제 SFT 트랜스포머 전체도 CPU 스트리밍 로더로 구성해 모든 값의 유한성을 확인했고, 42개 이미지 블록의 결합 Q/K/V 분할을 포함한 모든 파라미터와 FP32 RoPE 버퍼를 원본과 비교했다. 스트리밍 가중치는 원본 표현과, 나머지 파라미터는 BF16 변환값과 일치했다. 별도의 실제 INT8 언어 인코더는 200개 선형 계층과 임베딩을 로드했다. 실제 트랜스포머·인코더·토크나이저·VAE·스케줄러를 한 파이프라인에 함께 로드하고 실제 색상·깊이 어댑터의 262개 대상 연결 계획도 확인했다. 테스트용 잠재값을 사용한 별도 검사에서는 실제 어댑터의 투영·LoRA 후크 263개와 스케줄러 후크를 등록하고 예외 발생 후 모두 복원했다. 깊이 예측, 어댑터의 전방 연산과 이미지 생성은 확인하지 않았다.
