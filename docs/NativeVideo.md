# 단일 LTX 2 패키지의 네이티브 비디오 실행

`generateNativeVideo` and `iild_native_video_generate_v1` forward video checkpoints and explicit text encoders to the native C++ engine. ONNX paths are rejected case-insensitively before execution. GGUF, safetensors and other paths are forwarded, with actual support depending on the engine and model architecture. The existing image ABI is unchanged. The native image entry points also reject ONNX for the checkpoint, component, LoRA and advanced model paths before engine execution, while leaving GGUF and other supported paths available.
For structurally recognized Wan checkpoints the external or packaged text encoder is routed to the engine's UMT5/T5 slot; LTX retains its LLM slot. A complete in-checkpoint T5 signature avoids unnecessary extraction. Text-encoder routing alone does not supply a missing Wan VAE, second expert, or vision encoder and is not evidence that every Wan configuration can execute. `NativeWanNamesTests` compares the SDK's synthetic tensor signature and T5 name conversion with the actual native model loader without performing inference.
The additive C++ component request and versioned C bridge accept explicit local video VAE, high-noise expert and vision-encoder paths, plus optional expert boundary, step and guidance settings. The existing v1 request layout and entry points remain unchanged. CLI flags `--video-vae`, `--high-noise-model`, `--vision-encoder`, `--moe-boundary`, `--high-noise-steps` and `--high-noise-guidance` expose the same native-file path. Unsupported runtime versions and non-native routes reject these controls instead of ignoring them. The caller must select compatible components; each file and referenced shard is checked for changes before and after generation, and the output cannot overwrite a model or component shard. Expert and vision inputs require a structurally recognized Wan checkpoint; expert controls require the high-noise model. When supplied, `--high-noise-steps` adds that many high-noise steps to `--steps`; omitted values retain the engine defaults. These SDK/CLI controls do not yet make Dreamscapes discover or select external components automatically, and do not establish pretrained model compatibility.
For a structurally recognized Wan iiLocalDiffusion package, included, unambiguous `video_vae`, `high_noise_model`, and `vision_encoder` components are materialized into the caller's asset cache, verified, and bound to their native slots automatically. Explicit paths take precedence. A component already marked duplicate of embedded model tensors is not mounted twice; multiple distinct candidates are rejected. Indexed shards are reconstructed by the package materializer. The original package is never changed, and packaged roles do not prove the weights are compatible with the model. Dreamscapes benefits when its worker uses an updated iiLocalDiffusion runtime; unrelated external files are not discovered automatically.
If the selected package's main model is a GGUF asset, native video reconstructs that exact main asset into the caller-owned cache, verifies its original checksum and format, and passes the restored GGUF to the engine. Package and restored-model identities are rechecked before accepting frames. A package encoder is preferred when present; otherwise a complete encoder embedded in the restored main GGUF can be used. This is separate from and does not imply pretrained inference validation.
When a Wan model has both bare `patch_embedding.weight` and `blocks.0.cross_attn.norm_k.weight`, its file is supplied to the native loader both in its original namespace and as a diffusion model. This preserves any embedded text encoder or VAE while exposing the bare diffusion weights under the engine's required prefix. The file is not rewritten; the loader reads its metadata for both mounts. Canonically prefixed Wan checkpoints do not use the second mount.
Wan 고노이즈 전문가에는 독립 가중치의 접두사 없는 이름과 `model.diffusion_model.`, `diffusion_model.`, `transformer.`, 고노이즈 네임스페이스를 허용한다. 네이티브 로더는 이들 가중치를 단일 고노이즈 네임스페이스로 연결하며, 함께 들어 있는 텍스트 인코더 등 보조 텐서를 전문가 가중치로 연결하지 않는다. Safetensors·샤드 인덱스·GGUF의 원본 가중치를 수정하지 않는다.
구조적으로 Wan인 패키지의 단일 `high_noise_model`과 `vision_encoder`도 검증된 캐시에서 복원해 각 엔진 슬롯에 연결한다. 명시적 경로가 패키지보다 우선하고 복수 후보는 임의 선택 없이 오류로 처리한다. 고노이즈 제어 인자는 패키지에 해당 전문가가 있거나 명시 경로가 있을 때 사용할 수 있다. Wan Diffusers 패키지의 `boundary_ratio`는 네이티브 전환점으로 전달하며 명시적 요청값이 우선한다. 이는 두 스케줄러의 추론 결과가 같다는 보장은 아니다.
고노이즈 파일이 기본 Diffusers Wan의 완전한 서명과 텐서 이름을 사용하면 로더가 원본을 변경하지 않고 네이티브 이름으로 연결한다. 변환은 지원하는 기본 이름에 한정되며 모든 Wan 파생 모델의 실행 가능성을 보장하지 않는다.
같은 기본 Diffusers Wan 서명의 단일 Safetensors·GGUF와 Safetensors 샤드 인덱스도 로더에서 텐서 바이트를 수정하지 않고 주 diffusion 네임스페이스로 연결한다. 텍스트 인코더 등 원래 별도 네임스페이스는 보존하며 이름 충돌은 명시적으로 실패한다.
패키지 메타데이터 조회는 일반 Safetensors의 텐서 dtype을 패키지 형식으로 검증하지 않는다. 일반 파일의 실행 가능 여부는 실제 네이티브 엔진이 판단한다.
로컬 `WanPipeline`과 `WanImageToVideoPipeline` 폴더는 `model_index.json`에 선언된 단일 또는 고·저노이즈 전문가, 텍스트 인코더, VAE와 선택적 이미지 인코더를 원본 그대로 네이티브 비디오 슬롯에 연결한다. 이미지-비디오 요청의 각 shot은 첫 프레임 또는 이전 shot 이어받기가 필요하며 마지막·중간 프레임 조건은 지원하지 않아 오류로 반환한다. `CLIPVisionModelWithProjection`과 `CLIPProcessor` 선언은 함께 있어야 하며, 두 선언이 모두 null인 구성도 허용한다. `iild::inspectWanPipelineDirectory`, `iild-model-package inspect-wan-directory --input DIR` 및 Python `inspect_wan_directory`가 두 구성을 검사하고 기존 `iild::inspectWanDirectory`는 텍스트 전용 계약을 유지한다. 기존 v1 텍스트 전용 검사 보고서는 새 필드가 없어도 사용할 수 있지만, 이미지-비디오 구성 정보가 없는 런타임은 업그레이드 오류를 반환한다. `model_index.json`은 최대 1 MiB까지 읽는다. 복수 후보, 잘못된 샤드, 지원하지 않는 선언 또는 불완전한 전문가 구성은 추측하지 않고 실패한다. 원래 경로와 파일 상태를 출력 직전까지 확인하고 모델 폴더 안에는 결과를 저장하지 않는다. 폴더용 외부 컴포넌트 재지정은 지원하지 않는다. 샤드 인덱스가 심볼릭 링크여도 선택한 인덱스 위치를 기준으로 샤드를 찾고 링크 대상이 바뀌면 실패한다. LTX 전용 옵션은 기본값을 허용하되 변경된 값은 무시하지 않고 거부한다. 12 FPS를 넘는 요청은 기존 비디오 경로와 마찬가지로 원본 프레임을 만든 뒤 보간한다. 네이티브 전처리와 스케줄러는 Diffusers와 다를 수 있어 보고서에 선택된 컴포넌트와 스케줄러 경로를 기록하며, 메타데이터 검사와 합성 테스트만으로 사전 학습 모델의 실제 생성이나 품질을 보장하지 않는다.

한 요청의 모든 shot은 하나의 네이티브 context를 공유한다. 이미지 엔진과 같은 잠금을 사용하여 전역 콜백이 충돌하지 않도록 한다. 단계별 진행률과 취소를 지원하고 모든 종료 경로에서 프레임, context, 콜백을 해제한다. 비디오를 시작할 때 이전 이미지 캐시는 잠금 안에서 해제한다. 모델은 읽기 전용 mmap 및 메모리 예산을 적용하며 파일 전체를 익명 메모리에 복제하지 않는다.

`materializeModelPackageAsset` restores a packaged safetensors or GGUF encoder, validates its mapping, SHA-256 and file structure, and publishes it atomically. An unambiguous indexed Safetensors encoder split across files is reconstructed as one file while preserving each tensor’s bytes, dtype and descriptor fields; this merged file is not byte-identical to the original shards. Existing cache entries are checked before reuse. An eligible Wan VAE is handled separately by the same verified materializer; unrelated weights are not extracted. An explicit GGUF or safetensors text encoder can also be supplied; packaging does not prove engine compatibility. `NativeVideoResultTests` checks the sharded path with synthetic tensors and a controlled native engine, not pretrained inference.

Python `generate_video.py`는 로컬 파일과 지원되는 Wan 디렉터리를 `native_video`로, 다른 디렉터리는 Diffusers 경로로 전달한다. native 경로는 출력 크기, shot별 prompt/negative prompt/seed, steps, CFG, 장치, 정밀도, CPU 텍스트 인코딩과 VAE tiling을 전달한다. 첫/마지막 키프레임과 이전 shot의 마지막 프레임 연결을 지원한다. 중간 이미지 조건과 서로 다른 첫/마지막 조건 강도는 조용히 생략하지 않고 실행 전에 오류로 알린다. Diffusers 전용 decode noise 조절 등은 네이티브 엔진의 별도 계약이 아니며, 네이티브 실행은 모델 자체의 sampler/VAE 기본값을 사용한다. 네이티브 경로는 현재 영상 RGB를 게시하며 LTX 2의 별도 오디오 생성 UI를 추가하지 않는다.

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
안에서 축출하고 원본 매핑에서 다시 읽을 수 있도록 한다. CPU 전용 실행은 CPU 계산을,
아무 배치도 지정하지 않은 실행은 엔진의 자동 배치를 유지한다. 이 정책은
요청한 해상도·프레임 수·모델 정밀도를 바꾸지 않는다. `NativeVideoResultTests`는
세 배치의 네이티브 컨텍스트 인자를 검증하며, 실제 대형 모델의 성공 여부와 처리량은
별도의 추론 결과로 검증해야 한다.

자동 장치 배치의 shot 계산 중 GPU 메모리 예산 부족이 보고되면 해당 shot을 CPU 계산과
디스크 기반 diffusion 가중치로 한 번 재시도한다. 명시적으로 선택한 장치는 바꾸지
않으며 취소 요청은 우선한다. CPU 계산 메모리의 스왑 여부는 운영체제가 결정한다.
디스크 오프로딩은 충분한 디스크 공간이나 실행 성공을 보장하지 않고 처리 시간이
늘어날 수 있다. 원본 모델과 이전에 전달된 프레임은 변경하지 않는다.

## BF16 가중치와 CPU 보조 연산

LTX AV의 FP32 활성값과 BF16 가중치가 CPU 보조 연산으로 배치될 수 있다. 고정 버전 ggml의 CPU 이항 연산에 `F32, BF16 -> F32` 조합을 추가하여 모델을 다른 정밀도로 변환하지 않고 계산한다. 덧셈·뺄셈·곱셈·나눗셈이 동일한 자료형 변환 경로를 사용하며 결과는 FP32로 유지한다. `NativeMixedPrecisionTests`는 연속·브로드캐스트·전치 입력 12개 조합의 실제 CPU 계산값을 FP32 기준값과 비교한다. 이 패치는 CPU 텍스트 인코딩 선택이나 모델 파일을 변경하지 않는다.

## 메모리 예산을 넘는 모델의 분할 적재

Low VRAM 비디오 실행은 텍스트 인코더와 VAE를 CPU로 배치하고 텍스트·diffusion·VAE 가중치를 읽기 전용 디스크에서 지연 적재한다. 비동기 선행 적재를 끄고 안전한 그래프 경계가 있으면 각 구간을 실행한 직후 가중치를 해제한다. VAE 타일링은 항상 활성화하며 모델 정밀도, 시드, 영상 크기와 길이는 변경하지 않는다. 사용 가능한 RAM, 디스크 공간 또는 분할할 수 없는 연산 작업 공간을 보장하지 않으며 속도 저하가 있을 수 있다. 기존 C ABI와 일반 실행은 그대로 유지하고 별도의 Low VRAM 진입점을 사용한다. Python 비디오 실행의 `--low-vram`은 해당 설정을 적용하고, `--offload sequential`도 네이티브 Low VRAM 경로를 선택한다. 앱은 `--low-vram`을 명시해 이전 워커에서 이 옵션이 조용히 무시되지 않게 한다.

명시적으로 연산 장치나 CPU 텍스트 인코딩을 선택한 비디오 요청에서 체크포인트 또는 별도 텍스트 인코더의 파일 크기가 모델 예산보다 크면 텍스트와 diffusion 가중치를 디스크 기반으로 지연 적재한다. 명시적인 CPU 이미지 생성에서도 체크포인트가 예산보다 큰 경우 같은 방식을 사용한다. 장치를 완전 자동으로 선택한 비디오와 이미지 요청은 기존 엔진의 자동 배치·메모리 예산 정책을 유지한다. 계산 그래프에 안전한 분할 지점이 있는 CPU 작업은 현재 구간의 가중치만 적재하고 실행한 다음 해제한다. 다음 구간에 필요한 값은 엔진의 그래프 경계 캐시로 전달한다. 같은 가중치를 이후 구간에서 다시 쓰면 원본에서 재적재한다. GPU의 기존 메모리 예산 기반 분할 정책과 사용자가 명시한 메모리 상주 모드는 유지한다. 파일 크기는 저장 정책을 선택하는 기준이지 추론 중 실제 메모리 사용량의 정확한 예측값은 아니다.

분할 지점은 모델 구조가 제공해야 하므로 모든 모델이나 단일 연산의 작업 공간까지 임의로 분할할 수는 없다. 텐서 한 개나 구간 자체가 사용 가능한 메모리를 넘으면 실패할 수 있으며 디스크 읽기가 늘어난다. 분할 실행은 모델 파일, 정밀도, 프레임 수와 출력 크기를 변경하지 않는다. CPU 디스크 분할은 각 구간 진입 전에 취소를 확인하고 오류나 취소 시 적재된 구간을 해제한다. 합성 그래프와 가중치 fixture 테스트는 각 구간의 실제 적재·해제, 반복 실행, 오류·취소 후 복구를 확인하며 대형 모델 추론의 성공을 보장하지 않는다.

## PyTorch 분할 비디오 가중치

PyTorch 단일 파일과 분할 인덱스는 모델·텍스트 인코더·VAE·전문가·비전 인코더의
후보로 받을 수 있다. 설정 출력은 원본 경로를 보존하며 변환하지 않는다. 실제 작업
준비 시 weights-only 검사와 순차 Safetensors 변환을 수행하고, 선택 원본과 변환본의
식별자는 네이티브 실행 및 결과 게시 전후에 검사한다. 인덱스의 모든 샤드는 출력
충돌 검사에도 포함한다. 이는 모델 아키텍처나 실제 비디오 생성의 호환성을 보장하지 않는다.
