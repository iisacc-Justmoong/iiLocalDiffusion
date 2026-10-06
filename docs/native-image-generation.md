# 네이티브 이미지 생성

메모리 매핑은 실행 백엔드가 실제 CPU 버퍼 형식을 지원할 때만 직접 사용한다.
Metal의 호스트 포인터 가져오기 기능은 일반 CPU 버퍼의 직접 실행을 뜻하지 않는다.
Metal에 배치한 가중치는 Metal 버퍼로 읽고, CPU에 둔 가중치는 매핑과 GPU 스테이징을 유지한다.
`NativeMappedMetalStorage`와 `NativeMappedPrivateStorage`는 작은 GGUF를 읽어 CPU 실행,
Metal 실행, CPU에서 Metal로 옮긴 실행의 정확한 산술 결과를 검증한다.

CPU 폴백이 섞이는 그래프에서는 제자리 연산과 뷰가 공유하는 원본 버퍼의 호환성도 검사한다.
호환되지 않는 실행 위치가 섞이면 해당 버퍼를 공유하는 연산들을 모두 지원하는 백엔드에 함께 배치한다.
입력 복사만으로는 제자리 출력의 버퍼가 이동하지 않는 ggml 스케줄러 결함을
`native-inplace-backend-compatibility.patch`로 보완하며, 이미 호환되는 배치는 유지한다.
`NativeMixedBackendStorage`는 Metal 가중치와 CPU 전용 연산 사이의 제자리 연산,
공유 뷰, 크기가 달라지는 그래프 재사용의 버퍼 호환성과 실제 계산값을 검증한다.

BF16 산술 미지원 기기에서도 가중치 저장과 reshape/view/transpose/permute는 실행 커널을 요구하지 않는다.
`native-metal-storage-ops.patch`는 이 메타데이터 연산을 산술 지원 검사보다 먼저 처리하여
Metal에 저장된 BF16 가중치를 CPU 폴백으로 복사할 수 있게 한다. BF16 GPU 산술은 계속 거부한다.
`NativeLegacyMetalStorage`는 테스트 프로세스에서만 A12Z의 관측된 기능 제한을 적용하고,
BF16 저장·뷰 지원과 CPU 행렬곱의 정확한 결과를 확인한다.

Flash Attention이 요청되었으나 주 GPU에서 미지원이면 CPU 지원 여부를 검사해 같은 연산을 유지한다.
`native-cpu-flash-attention.patch`는 이 경우 전체 어텐션 점수 행렬을 만들지 않도록 한다.
Anima의 1024×1856 내부 캔버스에서는 7,424개 토큰과 16개 헤드의 점수 행렬만 약 3.29 GiB이므로,
CPU 폴백에서도 메모리 절약형 연산을 유지해야 한다. 해상도·스텝·샘플러는 변경하지 않는다.
`NativeLegacyMetalStorage`는 CPU Flash Attention의 실제 결과를 확인하고, 같은 토큰 수의
그래프에 제곱 크기 중간 텐서가 생기지 않는지 검사한다. 네이티브 내부 헤더를 사용하는 소비자는
고정 엔진과 동일한 `GGML_MAX_NAME=160`을 전달받아 텐서 구조체의 레이아웃을 일치시킨다.

기존 호출도 [전역 네거티브 임베딩·폴백 LoRA](generation-defaults.md)를 자동 적용한다. SDXL은 임베딩 7개를 사용하고 LoRA는 자동 적용하지 않으며, `generateNativeImageWithOptions()`는 커스텀 네거티브 문장·LoRA·리소스 위치를 받는다. 기존 요청/결과 구조체 ABI는 유지한다.

기본 LoRA 선택은 SDXL에 고정하지 않는다. 엔진이 확인한 SD1/SD2/SD3/FLUX.1/FLUX.2/Qwen Image/Z Image 계열을 `fallback_lora` 또는 `fallback_loras` 명세와 대조한다. 명시적 `options.loras`가 있으면 기본값을 대체하며 각 LoRA의 강도를 그대로 엔진에 전달한다. 번들 명세는 빈 기본 LoRA 목록을 사용하며, 사용자 명세에 등록할 때는 계열별 호환 파일이 필요하다. `NativeResultTests`는 7개 계열의 기본값 전달·명시적 대체·중복 계열 거부를 실제 네이티브 어댑터와 C API 대역으로 검증한다.

## 출력 크기와 오류 처리

`NativeGenerationRequest.width/height`는 최종 출력 크기이다. 기존 생성 진입점과 V1 C 브리지는 각 변을 절반으로 나눈 값을 1차 생성에 전달한다. 예를 들어 1024×1024 요청은 512×512 생성 → Lanczos 확대 → VAE 재인코딩 → 강도 0.35의 재확산 → 1024×1024 디코딩으로 실행한다. 고정 stable-diffusion.cpp의 내장 Hires 기능을 사용하므로 별도 모델·패키지·네트워크가 필요하지 않다. 기본 보정 요청 스텝은 `max(1, floor(steps * 0.35))`이며 실제 샘플러 콜백이 실행 스텝을 보고한다. 로딩만 하는 `prepareNativeImageModel`은 두 생성 단계를 실행하지 않는다.

구성요소별 로딩을 지원하는 `generateNativeImageWithComponents`와 V2 C 브리지는 `hires`를 명시적으로 받으며 기본은 단일 패스이다. CLIP·T5·LLM·VAE 경로, CFG와 내장 guidance를 개별 지정한다. 모델별 자동 판별, 분리 가중치, 출력 명세와 사용법은 [백엔드 라우팅](backend-routing.md)에 정리되어 있다.

각 단계의 내부 캔버스는 엔진의 모델 배수에 맞게 올림된다. 따라서 1024×1368 요청의 1차 입력은 512×684이고 UNet 캔버스는 512×704이다. 최종 캔버스 1024×1408을 Hires로 완성한 후 상하 20픽셀씩 제거하여 정확히 1024×1368을 반환한다. 최소 64픽셀 요청 등에서는 내부 정렬 때문에 1차 캔버스가 수학적인 절반보다 클 수 있다. VAE 메모리 정책은 최종 캔버스를 기준으로 정한다. 최종 잘라내기 자체에는 추가 보간이 없다. 잘못된 크기·채널·개수·빈 결과, 보정 실패·취소·제한 시간 초과는 완성 이미지로 반환하지 않는다.

SDXL 내장 VAE의 `using Conv2D scale 0.031`은 엔진 경고이다. 경고만으로 생성 실패를 판정하거나 결과 검증 오류를 이 문구로 덮어쓰지 않는다. 실패는 해당 작업의 구체적인 사유와 실제 ERROR 수준의 엔진 기록을 제공하며, 크기 불일치는 예상·반환 크기를 표시한다. `IILD_NATIVE_DIAGNOSTICS=1`에서는 경고도 계속 진단 로그에 남는다.

`NativeResultTests`는 실제 SDK 어댑터를 제어 가능한 upstream C API fixture와 연결한다. 네 가지 비정사각형 기본 비율·정사각형·양축 올림의 정확한 RGB 픽셀과 행 간격, 경고 뒤 성공, 실패 사유, 잘못된 크기·채널·개수·빈 출력 거부, 결과 메모리 및 콜백 정리를 검사한다. 실제 모델 추론은 별도 실기 검증으로 수행한다.

### iOS VAE 렌더링

iOS 빌드는 고정 엔진의 모듈 배치 기능에 `backend="vae=cpu"`를 전달한다. VAE 연산과 가중치는 CPU에 두고 디노이저는 기존 자동 선택으로 Metal을 사용한다. VAE 타일 분할, 선택한 내장 또는 계열별 폴백 VAE, 스텝·출력 해상도·취소 계약은 유지한다. macOS의 모듈 배치는 변경하지 않는다. 별도 호출 인자나 새 라이브러리가 필요하지 않으며, CPU 디코딩은 기기에 따라 렌더 시간을 늘릴 수 있다.

명시적 백엔드는 upstream의 자동 메모리 배치를 비활성화하므로 `params_backend="te=disk,diffusion=disk,vae=cpu"`도 함께 지정한다. 텍스트 인코더와 디노이저의 가중치는 기존 mmap 파일에서 필요한 구간을 읽고 GPU 예산에 따라 회수·재적재할 수 있게 한다. 이를 생략하면 이전 구간의 GPU 가중치가 회수 불가능한 상태로 누적되어 디노이징 중 메모리 예산을 초과할 수 있다. 디스크 배치는 가중치를 재양자화하거나 새 파일에 쓰는 기능이 아니며 기존 Q8 사본 선택 계약과 독립적이다.

iOS에서는 GPU와 CPU가 메모리를 공유하므로 GPU 예산을 Metal 권장 작업 세트의 60% 이하로 제한한다. 앱의 남은 메모리에서도 최소 1.5GiB 또는 40% 중 큰 값을 남긴다. 두 제약 중 작은 값을 256MiB 단위로 내리며, 실제 남은 메모리가 0이면 계속 로딩을 거부한다. 관측된 iPhone 15 Pro Max의 자원 값에서는 3GiB가 된다. 이는 기기 모델명이나 고정 3GiB 상한에 의존하지 않으며 더 큰 자원을 가진 기기에서는 예산도 증가한다. 엔진의 구간 분할·가중치 회수가 이 예산을 따르고 이미지 크기·스텝·정밀도는 바꾸지 않는다. `NativeCachePolicyTests`는 관측된 기기 값·낮은 메모리·0 응답·상위 용량을 검사하고, `NativeMobileResultTests`는 이 예산이 실제 C API에 전달되는지 검사한다. 진단 로그의 정책 이름은 `ios-cpu-vae-headroom`이다.

이는 iPhone에서 관측된 VAE 디코딩 중 Metal 명령 버퍼 실패와 GPU 복구의 영향을 피하기 위한 실행 배치다. `command buffer ... status 5`는 Metal의 실패 상태이며, 그 숫자만으로 메모리 부족이나 GPU 타임아웃을 판정하지 않는다. VAE 이전의 Metal 실패나 시스템 전체 GPU 장애에 대한 재시도 기능은 아니다. 실패한 네이티브 컨텍스트는 기존 정책대로 폐기한다.

`NativeMobileResultTests`는 실제 어댑터를 iOS 정책으로 별도 컴파일하여 CPU VAE 배치·가중치 배치·타일링 및 모든 출력·취소·캐시·계열별 VAE 회귀를 검사한다. `NativeResultTests`는 데스크톱 자동 배치를 함께 검사한다. `NativeVaeFallbackTests <작업 폴더> <resources/vae> --decode-portrait`는 공식 SDXL VAE와 실제 CPU 엔진으로 1024×1856 타일 디코딩을 수행하고 완전한 유한 RGB 출력을 검사한다. 앱과 같은 엔진 초기값(타일 중첩 0.5)을 사용하며, 최대 메모리는 `/usr/bin/time -l` 등 해당 운영체제의 프로세스 측정 도구로 함께 확인할 수 있다. 합성 잠재값을 사용하는 이 검사는 실제 iPhone의 전체 모델 생성 검증과 구분한다.

근거: [고정 엔진의 모듈별 백엔드 배치](https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/docs/backend.md), [Apple 명령 버퍼 상태](https://developer.apple.com/documentation/metal/mtlcommandbuffer/status?language=objc).

## VAE 마운트 검증과 자동 최적화

네이티브 SDXL·Qwen Image RGB·FLUX.1·FLUX.2는 VAE 접두어 한 개만으로 마운트를 승인하지 않는다. 실제 엔진의 인코더·디코더 파라미터 전체와 파일의 텐서 이름·크기를 비교한다. 정상 내장 VAE는 유지하며, 누락·불완전·형상 불일치는 같은 모델 계열의 해시 검증된 기본 VAE로 교체한다. 다른 잠재 공간의 VAE로 대체하지 않는다. 검사는 가중치를 적재하지 않는 메타데이터 검사이며 파일 정체성이 바뀔 때 다시 수행한다. `NativeVaeFallbackTests`는 정상 전체 구조, 단일 텐서, 누락된 디코더, 잘못된 텐서 크기, 두 접두어와 네 계열을 검사한다.

`NativeVaePolicy.hpp`의 `adaptive-sdxl-v1` 정책은 SDXL 작업 예산의 1/3 이내에서 16~64 잠재 픽셀 타일을 고른다. 32×32에서 측정한 약 480 MiB 작업 공간을 512 MiB로 올림하고 면적 비례로 보수적으로 추산한다. 3 GiB 예산에서는 40×40, 2.25 GiB에서는 32×32이며 중첩 0.5를 유지한다. 작은 캔버스 전체가 들어가면 타일을 분할하지 않아 지역별 정규화와 attention 차이를 피한다. 이 추산은 전체 프로세스 RSS 한도나 iPhone에서의 측정값이 아니다. Qwen·FLUX 등에는 측정하지 않은 SDXL 메모리 모델을 적용하지 않는다.

SDXL의 Conv2D는 외부 VAE 파일이 float32여도 내부에서 fp16 활성값을 사용한다. 내장·외부 경로 모두 1/32 입력 스케일과 역스케일로 오버플로를 완화한다. 이 값은 SDXL 잠재 스케일 0.13025를 바꾸는 값이 아니다. VAE 입력의 NaN/Inf는 디노이징 오류로 즉시 중단하고, 출력의 NaN/Inf는 RGB 클램프 이전에 오류로 반환한다. 흰색·검은색·균일한 정상 이미지는 내용만으로 거부하지 않는다. `NativeVaeSafetyTests`와 `NativeVaePolicyTests`, 실제 어댑터 대역 테스트가 이 계약을 검사한다.

추가 모델이나 런타임 의존성을 도입하지 않고 고정 stable-diffusion.cpp와 기존 오프라인 Diffusers 검증 환경을 사용한다. float16 SDXL의 활성값 문제는 [SDXL VAE fp16 수정 모델의 원 저자 설명](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)에 근거하며, 정상 내장 가중치를 이 다른 모델로 일괄 교체하지 않는다.

다음 검사는 실제 체크포인트의 내장 VAE 가중치를 독립적인 float32 Diffusers 디코더에 적재하고 동일한 합성 잠재값의 네이티브 RGB를 비교한다. 생성 모델의 인물 형태나 프롬프트 품질을 검증하는 테스트와는 구분한다.

```sh
cmake --build build --target NativeVaeDecodeProbe
reference/diffusers/.venv/bin/python tests/NativeVaeParitySmoke.py \
  --model /absolute/path/redLilyIllu_v10.safetensors \
  --probe build/NativeVaeDecodeProbe --output-dir build/vae-parity \
  --tiles 0 32 40 48 64
```

## 실행 계약

`Generation/NativeDiffusion.hpp`는 같은 프로세스에서 로컬 체크포인트를 읽고 RGB 바이트를 반환한다. 모델 다운로드·기기 연결·동기화·출력 저장은 수행하지 않는다. 소비 앱이 작업 스레드에서 호출하고 결과를 자신의 저장소에 게시한다. 기존 데스크톱 Python worker는 별도로 유지한다.

`IILD_ENABLE_NATIVE_DIFFUSION=ON`은 stable-diffusion.cpp의 `d04e8950c1ec8d30248cbe996682b3182fb1adf6` 및 해당 ggml 커밋을 빌드한다. iOS의 새 구성에서는 기본 ON이다. 체크포인트/토크나이저/텐서 실행을 직접 구현하는 대신 이 라이브러리의 C API를 사용한다. upstream은 활발히 변경되므로 커밋을 고정했다. 두 프로젝트는 MIT이고 고지문을 SDK에 설치한다. 서버·Web UI·WebP/WebM 및 RPC는 제외한다. Apple은 내장 Metal 코드를 사용하고 나머지 환경은 CPU를 사용한다.

한 번에 한 이미지를 실행하고 라이브러리의 전역 콜백 때문에 네이티브 호출을 직렬화한다. 모델 원본을 mmap으로 읽고 지원되는 실행 경로에서 다음 구간 프리페치를 사용한다. macOS에서는 Metal 권장 작업 세트에서 256 MiB를 제외한 값과 현재 사용 가능 메모리에서 여유 공간을 뺀 값 중 작은 값을 256 MiB 단위로 내림하여 관리 GPU 예산으로 사용한다. macOS의 여유 공간은 최소 768 MiB 또는 사용 가능 메모리의 1/6이며, iOS는 앞 절의 CPU VAE 메모리 여유 정책을 적용한다. iOS는 프로세스의 남은 한도, macOS는 현재 free+inactive 페이지를 조회한다. 전체 CPU 코어를 사용하며 Apple 작업 스레드는 생성 중 USER_INITIATED 우선순위로 실행하고 이전 우선순위를 복원한다. 이 예산은 전체 앱 메모리 상한이나 특정 iPhone에서 임의의 SDXL 모델이 실행된다는 보장이 아니다. 메모리 한계와 지원 체크포인트는 실제 기기로 별도 검증해야 한다. Diffusers 디렉터리는 현재 네이티브 API의 입력이 아니다.

취소는 호출 전·엔진 대기·텐서 로딩 사이·연산 구간 사이에서 확인한다. 취소되거나 결과가 손상되면 RGB를 반환하지 않는다. 프로세스 수명보다 짧은 호출자 데이터는 콜백이 끝날 때까지 유지해야 한다. `NativeDiffusionTests`는 취소·잘못된 크기·제한 시간 값·누락/잘못된 모델 거부를 검사하며 실제 모델 추론 증거는 아니다.

공식 근거: [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), [고정 C API](https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/include/stable-diffusion.h), [MIT 라이선스](https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/LICENSE).

2026-09-10: macOS와 iOS arm64 네이티브 빌드, 네이티브 기능 ON/OFF 계약 검사, Dreamscapes 설치 소비자 빌드를 검증했다. 실제 macOS 소비자에서 `redLilyIllu_v10.safetensors`의 512×512·20스텝 결과를 로컬 컨테이너에 게시하고 PNG를 검사했다. iOS는 빌드·번들까지이며 연결된 iPhone에서의 모델 적재·생성·메모리 한계는 아직 검증하지 않았다. `Product/Dreamscapes/build/society-local-generation/REPORT.md`에 실행 증거를 기록했다.

2026-09-11: `generateNativeImageWithProgress`는 Loading/Encoding/Denoising/Decoding을 구분한다. 기존 두 정수 콜백에는 실제 Denoising 이벤트만 전달한다. `cmake/patches/native-progress-cancellation.patch`는 고정 upstream에 단계 콜백과 중단 검사를 추가하며, 텐서 로딩 및 연산 구간 사이에서 취소를 확인한다. SDK 구성은 패치 적용 가능 여부를 검사하고 FetchContent 소스 경로 override에도 적용한다. 원본/패치의 충돌은 구성 오류이다.

`timeoutMilliseconds`는 기본 900000(15분), 양수이며 엔진 대기까지 포함한다. 취소 가능한 timed mutex로 전역 콜백 사용을 직렬화한다. 제한 시간 초과는 오류, 사용자 취소는 cancelled로 반환한다. 진행 중인 단일 파일 읽기·GPU 연산을 강제로 죽이지는 않는다. `IILD_NATIVE_DIAGNOSTICS=1`은 기기 진단용 upstream 로그를 stderr에 출력한다. 모델 경로 등 실행 정보가 포함될 수 있어 기본은 꺼져 있다.

취소 시 텍스트 인코더가 빈 텐서를 단언하는 upstream 경로를 피하기 위해, 로딩 스레드를 모두 join한 뒤 호출 스레드에서 스택을 해제한다. SDK 경계가 이를 cancelled 또는 제한 시간 오류로 변환한다. 컨텍스트 초기화 도중 해제도 보장한다. `IILD_NATIVE_TEST_MODEL=<absolute checkpoint> NativeDiffusionTests <absolute scratch directory>`는 실제 모델의 로딩·노이즈 제거 진입 취소 및 제한 시간을 검사한다.

2026-09-11 최종 실기기 검증: iPhone 15 Pro Max의 동일 체크포인트, 512×512·20스텝 로컬 생성이 294.647초에 완료되고 이미지 표시·Society 파일 저장을 확인했다. 로딩 중 취소는 시작 3초 뒤 요청하여 3.162초에 종료되었다. 네이티브 SDK 79/79 CTest와 OFF 빌드의 계약 검사도 통과했다. 이는 검증한 모델·기기 조합의 관측 결과이다.

## 디스크 및 메모리 캐시

기본 SDK 요청은 원본 체크포인트를 직접 읽는다. `q8CacheDirectory`에 절대 경로를 지정하면 기존 upstream 스트리밍 변환기로 Q8_0 GGUF 사본을 준비하며 VAE는 F16으로 유지한다. Dreamscapes iOS QuickGenerate는 이 경로를 기본으로 사용한다. Society App Group의 원본은 보존하며 네트워크 다운로드나 원본 덮어쓰기는 없다. 첫 변환에는 추가 시간과 디스크 공간이 필요하고, 이후에는 검증된 사본을 재사용한다. Q8은 같은 seed에서도 원본 정밀도와 픽셀이 달라질 수 있다.

캐시 키에는 고정 변환기/규칙 버전과 원본 식별자가 들어간다. 별도 manifest의 전체 원본 식별자, 출력 크기·전체 파일 내용의 FNV-1a 지문과 GGUF 헤더를 확인한다. 크기·mtime·POSIX inode/ctime이 달라지면 전체 내용의 지문을 다시 계산한다. 이 파일 상태가 같으면 최대 128개 파일의 지문을 재사용해 취소·일시 정지 전에 매번 대형 가중치를 읽지 않는다. 권한·시각·iOS 데이터 보호 속성이나 동일 내용의 파일 교체는 결과를 폐기하는 근거가 아니며, 실제 바이트가 달라지면 같은 크기와 mtime을 유지했더라도 검출한다. 이 지문은 인증용 서명을 대신하지 않는다. 원본이 없거나 변경된 경우 이전 캐시를 선택하지 않는다. 고유 임시 디렉터리에서 변환 후 원본을 재확인하고 파일과 manifest를 원자적으로 게시한다. 취소·실패 시 임시 파일을 제거하며 불완전한 manifest는 재변환한다. 모델별 캐시는 앱 CacheLocation 아래에 있고 OS가 공간 압박 시 삭제할 수 있다. 이전 원본 버전의 캐시는 새로운 요청에 사용되지 않는다. `NativeCachePolicyTests`는 메타데이터 변경과 동일 내용 파일의 교체 허용, mtime을 보존한 내용 변경 검출을 검사한다. 진단 모드에서는 재계산한 파일 상태와 내용 지문을 함께 기록한다.

SDK는 성공한 최근 모델 하나의 컨텍스트·파일 매핑·백엔드가 유지하는 가중치와 실행 계획을 다음 생성에 재사용한다. 경로·크기·수정 시각과 POSIX 파일 식별자·변경 시각이 바뀌면 교체한다. 모델 전체를 매 요청 해시하지 않으며, 생성 중 파일 교체가 감지되면 결과를 폐기한다. 취소·시간 초과·실패는 부분 컨텍스트를 폐기한다. 서로 다른 요청은 기존 전역 엔진 잠금으로 직렬 실행한다.

`releaseNativeDiffusionCache()`는 명시적인 사용자 해제 또는 앱 런타임 종료 전용이다. 유휴 컨텍스트와 익명 메모리 원본 풀은 즉시 해제하며 실행 중이라면 현재 요청 종료 때 해제한다. 백그라운드 진입, 대기, UIKit 메모리 경고에는 호출하지 않는다. Resident 경로는 모델 원본을 런타임 수명 동안 유지하며, GPU 구간별 재배치는 이 메모리에서 수행한다. OS의 압축·스왑은 허용하지만 물리 RAM 고정이나 OS 강제 종료 후 복원은 보장하지 않는다. 상세 계약은 [runtime residency](resident-worker-performance.md)에 있다.

`native-mapped-upload.patch`는 dtype이 같은 mmap 텐서를 백엔드에 바로 전달하여 구간마다 대형 임시 CPU 버퍼를 할당·초기화·복사하지 않도록 한다. dtype 변환과 비매핑 입력은 기존 경로를 유지한다. 이 업로드 최적화 자체는 텐서 값을 바꾸지 않는다. 별도 Q8 준비 옵션 외 스텝·해상도·샘플러·VAE 타일 설정을 변경하거나 근사 스텝 캐시를 추가하지 않는다. 기존 MIT stable-diffusion.cpp/ggml 및 시스템 Metal/Foundation/pthread 외 새 의존성은 없다.

결과의 `modelCacheHit`, `memoryBudgetBytes`, `threads`, `modelLoadMilliseconds`, `generationMilliseconds`로 재사용과 실행 시간을 관측한다. load는 컨텍스트 준비, generation은 인코딩·샘플링·디코딩을 포함하며 파일 게시 시간은 앱 job 시각으로 별도 측정한다. 이 필드 추가 후 소비 앱도 재빌드해야 한다.

`NativeCachePolicyTests`는 메모리 여유·낮은 메모리와 동일 크기/mtime 파일 교체의 캐시 무효화를 검사한다. `NativeCacheIntegration <checkpoint>`는 실제 모델의 512×512·1스텝으로 균일색이 아닌 출력, 동일 seed의 첫 생성/캐시 재사용/해제 후 재생성 RGB 일치, 콜백 내부 해제의 비블로킹 동작과 다음 생성의 캐시 미적중을 검사한다. 실제 취소·제한 시간은 `IILD_NATIVE_TEST_MODEL`을 지정한 `NativeDiffusionTests`가 별도로 검사한다. 실제 모델이 필요한 수동 통합 실행이며 기본 CTest 숫자에 포함하지 않는다. 연결 iPhone 검증 증거는 Dreamscapes `build/quickgenerate-acceleration/REPORT.md`에 별도 기록한다.


버전 0.6.0은 Q8 요청·준비 단계·성능 필드를 포함하므로 ABI를 구분한다(SOVERSION 0.6). 이전 소비자는 기존 설치 이름을 유지하며 새 API를 사용할 앱은 0.6 헤더·라이브러리로 재빌드한다. iOS 소비자는 큰 체크포인트를 mmap하려면 Extended Virtual Addressing capability가 필요하다. Increased Memory Limit도 지원 기기의 한도를 늘릴 수 있지만, 런타임은 항상 실제 사용 가능 메모리를 조회한다. capability가 없거나 지원되지 않는 기기는 더 낮은 예산과 파일 읽기 경로를 사용한다.


`native-tensor-wakeup.patch`는 적재 진행률의 200ms 고정 폴링 대기를 작업 스레드 완료 알림으로 바꾼다. 긴 작업의 진행 갱신 빈도는 제한하지만 적재가 끝나면 즉시 진행한다. 수백 번의 짧은 구간 적재마다 200ms를 기다리지 않으며 취소 검사 전에 모든 읽기 스레드를 join하는 계약을 유지한다.

Apple의 upstream 코어 조회는 perflevel0만 반환하므로 A17 Pro에서 2개만 선택한다. SDK는 Apple에서 `hardware_concurrency()`도 반영하여 6개 CPU 코어를 작업에 제공한다. 실제 모델 통합 테스트의 선택적 두 번째 인수는 기존 RGB 기준 파일을 생성하거나 비교하여 최적화 전후 결과 일치까지 검사한다.

사용 가능 메모리가 0이라는 실제 iOS 응답은 미지원/알 수 없음과 구분하여 로딩을 거부한다. 종료 시 캐시는 나중에 초기화된 GPU 백엔드의 정적 소멸자보다 먼저 해제한다. 실제 모델 통합 테스트는 마지막 성공 컨텍스트를 유지한 채 종료하여 이 소멸 순서를 검사한다. `IILD_NATIVE_DIAGNOSTICS=1`은 캐시 해제와 모델 식별자 변경도 기록하며, 소비 앱의 메모리 경고 로그와 함께 캐시 미적중 원인을 구분할 수 있다.

`IILD_NATIVE_DIAGNOSTICS=1`과 `IILD_NATIVE_MAX_MEMORY_MIB=2048`을 함께 지정하면 새 컨텍스트의 관리 GPU 예산을 더 낮게 제한해 파일 페이지 캐시와의 경합을 비교할 수 있다. 실제 기기의 안전 예산을 초과할 수 없으며 512 MiB 미만·잘못된 입력은 무시한다. 기존 메모리 캐시에는 소급 적용하지 않으므로 별도 프로세스로 비교한다. 기본 실행에서는 이 진단 제한을 적용하지 않는다.

`native-metal-shared-upload.patch`는 공유 Metal 버퍼를 식별하는 작은 ggml 보조 함수를 제공한다. `native-mapped-upload.patch`의 같은 dtype 경로에서 공유 버퍼의 서로 다른 텐서 memcpy만 병렬 실행한다. GPU 전용 버퍼와 다른 백엔드는 기존 직렬 업로드를 유지한다. `NativeMetalSharedUpload`는 6개 작업자의 24개 텐서 쓰기·읽기 일치를, `NativeMetalPrivateBuffer`는 전용 GPU 버퍼가 병렬 경로에서 제외됨을 검사한다.

Q8 준비는 `Preparing` 단계이며 텐서 진행률을 denoising 스텝으로 표시하지 않는다. `native-conversion-cancellation.patch`는 변환 작업자의 텐서 전후에 중단을 검사하고 기존 실패 알림과 join 경로로 안전하게 종료한다. 요청 제한 시간에는 변환도 포함한다. `q8CacheUsed`, `diskCacheHit`, `modelBytes`, `preparationMilliseconds`로 첫 변환과 디스크 재사용을 구분하며 `modelCacheHit`는 메모리 컨텍스트 재사용만 의미한다. `NativeDiskCacheTests`는 캐시 적중, 손상, 원본 교체, 취소·실패 및 원자적 게시를 검사한다. 실제 변환 취소는 `IILD_NATIVE_TEST_MODEL` 계약에 포함하고, `IILD_NATIVE_Q8_CACHE=<absolute build/cache>`로 실제 모델 통합 검증의 Q8 경로를 선택한다. lazy GPU 백엔드는 첫 추론 중 초기화되므로 프로세스 종료 캐시 해제 등록은 첫 성공 추론 이후에 한다.

Q8 변환은 읽기 전용 mmap을 사용하고, 작업자들의 예약된 출력 바이트를 256 MiB로 제한한다(한 텐서가 더 크면 그 텐서만 처리). 이는 변환 전체 RSS 한도가 아니며 원본 페이지·float 변환 임시 버퍼·GGML 출력의 추가 메모리를 확보하기 위한 제한이다. 기존 1 GiB 출력 예약은 모바일에서 이 추가 메모리와 경합한다. mmap 실패 시 upstream 파일 읽기 경로를 유지한다.

`native-thread-safe-logging.patch`는 병렬 변환의 공유 정적 로그 버퍼를 호출별 버퍼로 바꾼다. ggml의 이미 포맷된 메시지는 `%s` 인수로 전달한다. `NativeLoggingTests`는 8개 작업자의 16000개 긴 로그가 서로 섞이지 않는지 검사한다. 변환 중 예외도 GGML 임시 컨텍스트를 RAII로 해제한다.

`NativePreparationTests`는 약 1 MiB의 유효 safetensors 행렬 fixture를 생성하고 실제 upstream Q8 변환의 서로 다른 텐서 진행 시점에서 세 번 취소한다. 학습된 이미지 모델이나 GPU 추론 없이 작업자 join·부분 파일 제거·원본 보존·준비 시간 계측을 기본 CTest에서 검증한다.

정확한 RGB 재현을 검사할 때는 진단용 메모리 예산을 고정한다. 실제 Mac Q8 검사에서 서로 다른 3.75/5 GiB 예산의 출력은 달랐고, 2 GiB로 고정한 첫 생성·메모리 재사용·해제 후 재생성은 RGB가 모두 일치했다. 실행 계획의 메모리 분할이 바뀌는 조건에서 픽셀 단위 재현을 보장하지 않는다.

Qwen Image RGB·SDXL·FLUX.1·FLUX.2·Anima·Z-Image 체크포인트에 VAE가 없거나 내장 구조가 맞지 않으면 SDK가 호환 VAE를 자동 연결한다. 기존 함수에 새 인자를 전달할 필요가 없으며 `defaultModifiers=false`에서도 유지한다. `auto-mount-v2`는 입력 파일의 텐서 메타데이터로 모델과 VAE 계열을 판별하고 정상 내장 VAE를 보존한다. 외부 VAE의 설정과 전체 텐서를 컨텍스트 생성 전에 검증하며 설정 변경도 캐시를 무효화한다. SD1/2/SD3는 정상 내장 VAE를 사용할 수 있고, 누락 시 해당 리소스가 없다는 오류를 반환한다. 지원 범위 밖 모델은 검사 완료와 구별하며 임의의 VAE를 주입하지 않는다. [VAE 기본값과 패키지 계약](generation-defaults.md#vae-자동-폴백)을 참고한다.

## 실행 일시 정지

`generateNativeImageWithExecutionControl()`은 선택적인 `NativeExecutionControl`을 통해 동일한 네이티브 요청을 일시 정지하고 재개한다. 기존 요청/결과 구조체와 함수 ABI는 유지하며 별도 진입점과 설치 헤더를 추가했다. 실행 제어는 Qt/UIKit을 참조하지 않는다.

소비자는 OS 실행 권한을 잃기 전에 `setPaused(true)`, 복귀하면 `setPaused(false)`를 호출한다. 엔진의 기존 텐서 적재·연산 구간 경계에서 조건 변수로 대기하므로 이미 제출한 GPU 호출이 반환되기까지는 시간이 필요하다. latent·모델 컨텍스트·seed를 유지하며 작업을 재시작하지 않는다. 정지 시간은 추론 제한 시간에서 제외하고, 여러 적재 스레드가 동시에 기다려도 한 번만 계산한다. 정지 중에도 기존 취소 토큰을 확인하므로 취소/소비자 파괴가 대기에 갇히지 않는다. OS의 프로세스 종료나 메모리 회수 이후 복원은 제공하지 않는다.

`NativeResultTests`는 실제 어댑터와 제어된 C API로 연산 경계 대기, 제한 시간보다 긴 정지 후 같은 요청 완료, 정지 중 취소와 자원 해제를 검증한다. 백그라운드 GPU 실행 허용 여부와 앱의 OS 작업 등록은 소비자가 담당한다.
<a id="anima-desktop-worker-bridge"></a>

# Anima 데스크톱 작업자 브리지

`NativeImageBridge.hpp`는 SDK Python 작업자를 위한 버전 지정 C 요청/결과 경계를 노출한다. `iild_native_generate_v1`는 기존 네이티브 생성기와 캐시를 감싸며, RGB 저장소와 JSON 진단 정보는 `iild_native_free_v1`를 호출할 때까지 라이브러리가 소유한다. C++ 예외는 이 경계를 넘지 않고, 호환되지 않는 요청 크기는 거부하며, 진행 콜백에서 취소를 요청할 수 있다. `prepareNativeImageModel`은 프롬프트 인코딩·샘플링·PNG 생성 없이 동일한 검증된 컨텍스트를 유지하며, 첫 생성에서는 텐서를 지연 배치한다. 기존 요청/결과 구조체와 생성 진입점은 ABI를 유지한다. 새로운 외부 의존성은 필요하지 않다. C 브리지의 `timeout_milliseconds` 기본값은 0이다. 대형 이미지를 디코딩하는 데스크톱 작업은 15분을 넘을 수 있기 때문이다. 0은 C++ API의 최대 정수 예산에 대응하며, 양수 제한은 명시적으로 유지하고 취소 동작은 바꾸지 않는다. 브리지 테스트는 단위 테스트에서 15분을 기다리지 않고, 제한이 0인 경우와 짧은 양수 마감 시간을 설정한 경우에 동일한 지연 연산을 비교한다.

`NativeResultTests`는 준비 단계에서 샘플링하지 않는지, 10회의 연속 브리지 호출이 준비된 컨텍스트를 재사용하는지, RGB 크기와 네거티브 프롬프트가 보존되는지, 취소 시 실행을 정리하는지, 잘못된 ABI 크기에 오류를 반환하는지 검증한다. `NativeImageTests`는 텐서에 따른 라우팅, 미지원 옵션, 누락된 구성요소, 포그라운드 명령 뒤에 이어지는 10개의 순서 있는 작업자 요청, 출처와 전체 성공 또는 전체 실패 방식의 배치 게시를 검사한다. `DownloadedModelTests`는 오해를 유발하는 파일명으로 된 완전한 Anima 모델과 디노이저 전용 Anima 모델, 실제 VAE 전용 파일을 다룬다.

<a id="explicit-cpu-execution-for-background-tasks"></a>

## 백그라운드 작업에 대한 명시적 CPU 실행

`generateNativeImageWithBackend(request, NativeComputeBackend::Cpu, cancelled, progress, control)`는 모든 연산 모듈과 해당 매개변수 저장소를 CPU에 배치한다. `Automatic` 옵션은 기존 플랫폼 배치를 유지한다. 이 진입점은 기존 요청/결과/옵션의 레이아웃을 변경하거나 OS 실행 시간을 부여하지 않는다. 소비자는 적절한 백그라운드 작업을 확보하고 취소·만료 처리에 협조해야 한다. CPU 배치는 모델 로딩 전에 고정하므로, CPU 전용 OS 실행 권한 아래에서 GPU 연산을 제출하지 않는다. 메모리 내 모델 식별자에는 선택한 백엔드가 포함되며, 이를 변경하면 호환되지 않는 상주 상태를 제거한다. 디스크 모델 캐시의 바이트는 백엔드와 무관하므로 계속 공유한다.

`NativeResultTests`와 `NativeMobileResultTests`는 제어된 상위 공급 측 C API 결과를 사용하여 제품 어댑터를 컴파일한다. 전체 CPU 연산/매개변수 선택, 준비된 CPU 상태의 재사용, 자동/CPU 전환 시 상주 상태 제거를 검증한다. 런타임 성능과 OS 백그라운드 실행 권한은 별도의 기기 실행으로 검증해야 한다.


<a id="shared-model-storage-061"></a>

## 공유 모델 스토리지(0.6.1)

애플리케이션은 모든 가중치를 저장소 소유자의 컨테이너에 보관할 수 있다. 절대 리소스 디렉터리를 `NativeGenerationOptions::resourceDirectory`에 전달하고 `generateNativeImageWithOptions(request, options, backend, ...)` 오버로드를 사용하여 명시적인 CPU 백그라운드 배치를 유지한다. 이 동작은 전역 환경 변수를 변경하거나 리소스를 애플리케이션에 복사하지 않는다. 선택적인 스타일 리소스가 설치되지 않았다면 `defaultModifiers`를 비활성화한다. 유효한 내장 VAE는 계속 사용할 수 있다. 필수 외부 VAE는 명시적으로 제공한 리소스 디렉터리에서 계속 선택하고 검증한다. 리소스가 누락되어도 다른 애플리케이션의 패키지로 전환하지 않는다.

`NativeDiffusionTests`는 명시적인 CPU 배치가 패키지 기본값을 요구하지 않으면서 호출자 옵션·취소·잘못된 모델 거부를 유지하는지 검사한다. 기존 API와 구조체 레이아웃은 변경하지 않는다.

### CPU 작업 내부의 진행과 정지

명시적 CPU backend는 upstream `sd_set_backend_eval_callback`으로 16개 노드마다 실제 연산 완료를 관측한다. `NativeGenerationStage::Computing`의 step은 완료된 그래프 배치의 누적 수이며 total=0은 전체 수가 아직 알려지지 않았음을 뜻한다. 기존 enum 값과 요청·결과 구조의 ABI는 유지한다. 소비자는 이 이벤트를 진행 보고와 비활동 제한 시간에 사용하되 denoising 스텝으로 표시하지 않는다. 이 경계에서 pause/cancel을 검사하고 작업 종료 시 전역 콜백을 제거한다. timer 기반의 가짜 진행 보고는 사용하지 않는다. NativeResultTests와 NativeMobileResultTests는 두 CPU 실행의 진행·캐시 재사용·콜백 해제를 검사한다.

## 메모리 상주 모델 로딩

`generateNativeImageWithResidentWeights()`는 Society 호스트용 진입점이다. Dreamscapes의 macOS/iOS/Android 생성 큐는 기본적으로 이 API를 사용한다. 대형 체크포인트를 사전에 해시하느라 같은 볼륨을 한 번 더 읽지 않도록 파일 메타데이터로 캐시 정체성을 판별하고, 큐가 모델 로딩 단계에 진입하면 원본 파일을 한 번 순차적으로 읽어 프로세스의 익명 메모리 버퍼로 전부 복사한다. 이후 텐서 로더는 그 버퍼만 참조한다. 복사가 끝나면 원본 파일 핸들을 닫으며 추론 중 원본 모델 파일을 다시 읽지 않는다. 이 경로는 Society 드라이브의 Q8 디스크 캐시를 생성·이전·참조하지 않는다. 익명 메모리는 OS가 메모리 압박에 따라 압축하거나 스왑할 수 있지만, 앱은 스왑 파일 위치나 정책을 강제하지 않는다. 스왑 공간이 부족하거나 OS 메모리 압박이 심하면 앱이 종료될 수 있으며, 대형 모델의 적재 가능성을 보장하지 않는다. 원본이 외장 볼륨에만 있다면 최초 RAM 적재 전송은 그 볼륨을 읽어야 하며, 이는 물리적으로 피할 수 없다.

이 모드는 Q8 디스크 캐시를 거부하고 ZIP 가중치 및 아카이브 `.iildmodel`의 디스크 추출, 익명 메모리 할당/읽기 실패 때 디스크 스트리밍으로 폴백하지 않는다. 아카이브 모델은 먼저 압축을 푼 디렉터리로 준비해야 한다. 일반 경로의 GPU 배치 메모리 예산이 낮아도 상주 모드의 적재를 사전 차단하지 않으며, 익명 메모리 할당/페이지 적재를 실제 시도한 뒤 실패를 반환한다. 익명 메모리는 OS가 압축하거나 스왑할 수 있고 앱은 별도 스왑 파일이나 외장 볼륨을 생성·지정하지 않는다. 일반 진입점의 기존 메모리 거부 정책은 유지된다. 대용량 모델은 전체 원본과 실행 중 필요한 가중치·작업 버퍼가 함께 메모리 압박을 일으킬 수 있으므로, OS 스왑에 의존하는 경우 생성 성능이 크게 낮아질 수 있다. 실제 대형 체크포인트 실행은 이 메모리 영향과 스왑 여유를 별도로 측정해야 한다.

`NativeBulkReadTests`는 resident 버퍼가 원본 파일 절단 및 삭제 이후에도 완전한 데이터를 제공하는지, `NativeResultTests`는 진입점의 플래그 전달과 Q8 디스크 캐시의 사전 거부를 검사한다.

## 실제 latent 기반 라이브 프리뷰

`generateNativeImageWithPreview`는 기존 요청·결과 구조와 함수 ABI를 유지하는 별도 진입점이다. 선택적 `NativePreviewCallback`은 실제 denoised latent의 RGB 투영을 전달하며 `PREVIEW_PROJ`를 사용해 매 단계 VAE 디코딩과 추가 모델 로딩을 피한다. 샘플링 데이터·난수·최종 이미지에는 영향을 주지 않는다. RGB를 소유하는 `NativeGenerationPreview`에는 최대 512×512 크기, 요청 전체의 sequence와 현재 pass의 step/total이 포함된다. Hires의 첫 단계에서도 sequence는 증가한다. 콜백은 작업 스레드에서 호출되며 엔진 전역 콜백은 실행 잠금 안에서 등록·해제한다.

C 경계 `iild_native_generate_with_preview_v1`의 RGB 버퍼는 콜백이 반환할 때까지만 유효하다. Python worker는 이를 복사해 원자적 PNG와 `IILD_PREVIEW` 이벤트를 발행하며, prepare-only 호출은 프리뷰와 이미지를 만들지 않는다. 모델에 대응하는 엔진 투영이 없으면 프리뷰를 지어내지 않는다. 기존 Diffusers VAE 프리뷰 경로는 유지한다. `NativeResultTests`와 `NativeMobileResultTests`는 기본/보정 pass의 실제 테스트 픽셀·콜백 수명과 CPU/GPU 선택을 검사한다. `NativeImageTests`는 프리뷰 파일과 단계 리셋을 검사한다.

데스크톱 worker에 `IILD_WORKER_PROGRESS=1`을 전달하면 최초 체크포인트 전체 해시 읽기가 `IILD_MODEL_PROGRESS`로 실제 읽은 바이트 수를 보고한다. 해시 캐시 적중 시 재읽기나 가상의 진행 이벤트를 만들지 않는다. `InferenceCacheTests`가 이 계약을 검증한다.


## Krea2 img2img와 자동 샘플링

단일 Krea2 참조 이미지는 `init_image`로만 전달한다. 일반 Krea2 체크포인트에서 참조 토큰을 동시에 주입하면 upstream의 `krea2_ostris_edit` 프리셋이 활성화되므로, 학습 방식이 확인되지 않은 베이스 체크포인트에는 이 경로를 사용하지 않는다. Krea2의 여러 입력 이미지는 현재 거부하며, 다른 검증된 편집 모델의 순서 있는 참조 계약은 유지한다.

네이티브 자동 스케줄은 Python worker와 동일하게 Euler, 해상도 기반 `mu = .5 + (W/16 * H/16 - 256) * .65/6144`, 지수 이동 sigma 및 마지막 0을 사용한다. W/H는 64px 정렬된 내부 추론 크기이며, 요청된 최종 출력 크기와 중앙 크롭 계약은 유지한다. 명시적 sigma, 샘플러 및 고급 스케줄러 선택은 보존한다.

`nativeImageParameterDefaults(modelPath)`는 가중치를 적재하지 않고 텐서 메타데이터를 검사한다. 일반 Krea2의 빠른 요청에는 Raw 품질 기본값 52 steps와 native CFG 7을 반환한다. CFG는 제품 설정으로 7을 사용하며, 단계 수는 [Krea 공식 추론 설정](https://github.com/krea-ai/krea-2/blob/main/README.md)을 기준으로 한다. 파일명으로 Raw/Turbo 학습 변형을 추정하지 않는다. 고급 요청은 사용자가 제출한 steps/CFG를 유지한다.

Krea2 분류는 메타데이터 검사와 적재된 런타임의 `sd_get_model_family` 양쪽에 적용한다. 런타임이 `other`를 반환하면 img2img 전용 분기와 자동 스케줄이 우회되므로, `NativeRefinerSamplingTests`는 실제 엔진 소스를 컴파일하여 두 분류가 일치하는지 가중치 없이 검사한다.

`NativeResultTests`와 `NativeMobileResultTests`는 Krea2 입력의 중복 편집 조건 방지와 해상도별 sigma 전달, 기존 편집 모델의 다중 참조 유지, 다른 모델로 기본값이 유출되지 않는지를 검사한다. 실제 체크포인트 검증에는 `NativeVaeRoundTripProbe`로 RGB→VAE→RGB를 분리하고 `NativeImg2ImgSmoke`로 인코더·확산·디코더 전체를 실행한다. 두 도구는 interleaved RGB 파일을 사용하며, 자동 CTest에 대형 모델 실행을 추가하지 않는다. 전체 추론 도구의 마지막 선택 인자 `prompt.txt`는 UTF-8 원래 프롬프트를 그대로 읽는다. 입력 RGB 크기와 출력 크기를 별도로 지정하여 원본 첨부의 네이티브 리사이즈 경로도 검증할 수 있다.
