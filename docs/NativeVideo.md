# 단일 LTX 2 패키지의 네이티브 비디오 실행

`generateNativeVideo` and `iild_native_video_generate_v1` forward video checkpoints and explicit text encoders to the native C++ engine. ONNX paths are rejected case-insensitively before execution. GGUF, safetensors and other paths are forwarded, with actual support depending on the engine and model architecture. The existing image ABI is unchanged. The native image entry points also reject ONNX for the checkpoint, component, LoRA and advanced model paths before engine execution, while leaving GGUF and other supported paths available.

한 요청의 모든 shot은 하나의 네이티브 context를 공유한다. 이미지 엔진과 같은 잠금을 사용하여 전역 콜백이 충돌하지 않도록 한다. 단계별 진행률과 취소를 지원하고 모든 종료 경로에서 프레임, context, 콜백을 해제한다. 비디오를 시작할 때 이전 이미지 캐시는 잠금 안에서 해제한다. 모델은 읽기 전용 mmap 및 메모리 예산을 적용하며 파일 전체를 익명 메모리에 복제하지 않는다.

`materializeModelPackageAsset` restores a packaged safetensors or GGUF encoder with its original bytes and format, validates its mapping, SHA-256 and file structure, and publishes it atomically. Existing cache entries are checked before reuse. Other weights are not extracted or hashed. An explicit GGUF or safetensors text encoder can also be supplied; packaging does not prove engine compatibility.

Python `generate_video.py`는 로컬 파일을 `native_video`로, 기존 디렉터리는 Diffusers 경로로 전달한다. native 경로는 출력 크기, shot별 prompt/negative prompt/seed, steps, CFG, 장치, 정밀도, CPU 텍스트 인코딩과 VAE tiling을 전달한다. 첫/마지막 키프레임과 이전 shot의 마지막 프레임 연결을 지원한다. 중간 이미지 조건과 서로 다른 첫/마지막 조건 강도는 조용히 생략하지 않고 실행 전에 오류로 알린다. Diffusers 전용 decode noise 조절 등은 네이티브 엔진의 별도 계약이 아니며, 네이티브 실행은 모델 자체의 sampler/VAE 기본값을 사용한다. 네이티브 경로는 현재 영상 RGB를 게시하며 LTX 2의 별도 오디오 생성 UI를 추가하지 않는다.

VAE가 요구하는 8n+1 샘플은 내부에서 생성한다. 첫/마지막 샘플을 포함하여 필요한 source frame을 선택한 후 기존 보간기로 정확한 출력 FPS·프레임 수를 만든다. 요청한 frame count, RGB 크기, 중복·누락을 검증하며 MP4는 기존 `iild-temporal-video-v1` 및 H.264 decode 검증을 유지한다. 패딩 프레임을 사용자 타임라인에 추가하지 않는다.

```sh
iild-generate --backend video --model-path '/path/LTX package.safetensors' \
  --prompt 'A camera moves through a studio' --width 512 --height 512 \
  --duration 2 --fps 24 --output '/path/video.mp4'
```

테스트는 가중치 다운로드 없이 실제 production adapter에 제어된 네이티브 결과를 주입하여 프레임 소유권, batch context, 연결, 실패와 취소를 검사한다. 테스트용 모델과 출력은 `resources/` 아래의 임시 디렉터리에 만들고 테스트 후 삭제한다. safetensors 텍스트 인코더의 실제 22B 모델 추론, 최종 영상 품질과 장치별 처리량은 이 테스트로 증명되지 않는다.

## 준비 진행률과 정지 진단

임베디드 텍스트 인코더의 추출 및 캐시 검증은 `Preparing`의 `step/total`로 실제 완료량을 전달한다. 기존 32비트 진행률 ABI를 유지하기 위해 단위는 MiB이며 전체량은 올림한다. 마지막 이벤트는 `step == total`이다. 파일 입력·출력 버퍼는 스트림을 열기 전에 1MiB로 설정하여 큰 자산을 작은 읽기·쓰기 호출로 분해하는 비용을 줄인다. SHA-256 검사와 원본 파일 변경 검사는 유지한다.

CPU 텍스트 인코딩 및 CPU 생성은 실제 완료된 그래프 묶음을 `Computing(step > 0, total = 0)`으로 전달한다. Python은 준비·계산 진행률을 기존 `IILD_NATIVE_PROGRESS` 프로토콜에도 보내 이미 실행 중인 데스크톱 클라이언트가 이를 인식할 수 있도록 한다. 반복된 동일 값이나 시간 기반 heartbeat는 작업의 진전으로 취급하지 않는다.

`IILD_NATIVE_TELEMETRY_DIR`을 지정하면 이미지 경로와 같은 `NativeTelemetry` JSONL을 생성한다. 준비, 모델 로딩, 텍스트 인코딩, 샘플링, 디코딩, 실제 메모리와 backend 배치를 기록하고 프로세스가 중단되어도 파일을 남긴다. heartbeat는 관측용이며 감시 시간 연장을 위한 진행률이 아니다. 자산 캐시의 수명은 호출자가 지정한 `assetCacheDirectory`가 결정한다. 재사용 가능한 캐시를 작업별 임시 디렉터리에 넣으면 매 작업마다 다시 추출되므로, 앱은 모델 저장소의 지속 캐시를 전달해야 한다.

엔진 경고와 오류는 상세 추적에 보존한다. 실패한 비디오 요청에는 최근 경고와 오류를 각각 제한된 크기로 구분해 반환하므로, 예를 들어 가중치 준비 실패 직전의 장치 메모리 경고도 보고서의 근거로 확인할 수 있다. 경고 자체는 생성 실패를 뜻하지 않는다. `NativeTelemetryTests`와 `NativeVideoResultTests`는 경고 기록, 성공과 실패 구분, 출력 크기 제한을 검증한다.

자산 임시 파일명에는 작성 프로세스 PID를 기록한다. 다음 준비 시 같은 자산의 임시 파일 중 작성 프로세스가 종료된 파일만 회수한다. 사용 중인 프로세스의 파일, 심볼릭 링크, 이름 규격이 다른 파일은 보존한다. 강제 종료로 RAII 정리가 실행되지 않아도 지속 캐시에 미완성 대용량 파일이 누적되지 않도록 한다.

CPU 텍스트 인코딩이나 명시적인 GPU 선택은 엔진의 자동 배치를 끈다. 이 경우 비디오
diffusion 가중치를 `diffusion=disk`로 지정해 사용하지 않는 구간을 실제 메모리 예산
안에서 축출하고 원본 매핑에서 다시 읽을 수 있도록 한다. CPU 전용 실행은 기존 CPU
배치를, 아무 배치도 지정하지 않은 실행은 엔진의 자동 배치를 유지한다. 이 정책은
요청한 해상도·프레임 수·모델 정밀도를 바꾸지 않는다. `NativeVideoResultTests`는
세 배치의 네이티브 컨텍스트 인자를 검증하며, 실제 대형 모델의 성공 여부와 처리량은
별도의 추론 결과로 검증해야 한다.

## BF16 가중치와 CPU 보조 연산

LTX AV의 FP32 활성값과 BF16 가중치가 CPU 보조 연산으로 배치될 수 있다. 고정 버전 ggml의 CPU 이항 연산에 `F32, BF16 -> F32` 조합을 추가하여 모델을 다른 정밀도로 변환하지 않고 계산한다. 덧셈·뺄셈·곱셈·나눗셈이 동일한 자료형 변환 경로를 사용하며 결과는 FP32로 유지한다. `NativeMixedPrecisionTests`는 연속·브로드캐스트·전치 입력 12개 조합의 실제 CPU 계산값을 FP32 기준값과 비교한다. 이 패치는 CPU 텍스트 인코딩 선택이나 모델 파일을 변경하지 않는다.
