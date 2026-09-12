# 네이티브 이미지 생성

## 출력 크기와 오류 처리

출력 너비·높이는 요청한 8픽셀 배수와 일치한다. 고정 upstream의 UNet은 VAE 8배와 UNet 8배를 합친 64픽셀 단위로 내부 캔버스를 올림하므로, SDK가 이 크기를 명시적으로 요청하고 완성 RGB의 초과 가장자리를 중앙 기준으로 잘라 반환한다. 예를 들어 1024×1368은 1024×1408에서 상하 20픽셀씩, 1024×1824는 1024×1856에서 상하 16픽셀씩 제거한다. 이미지 확대·축소나 보간은 없으며, 이미 64픽셀 배수인 출력의 픽셀은 그대로 복사한다. 요청한 내부 캔버스와 다른 크기, 빈 결과, 복수 이미지, RGB 이외 채널은 계속 오류로 처리한다.

SDXL 내장 VAE의 `using Conv2D scale 0.031`은 엔진 경고이다. 경고만으로 생성 실패를 판정하거나 결과 검증 오류를 이 문구로 덮어쓰지 않는다. 실패는 해당 작업의 구체적인 사유와 실제 ERROR 수준의 엔진 기록을 제공하며, 크기 불일치는 예상·반환 크기를 표시한다. `IILD_NATIVE_DIAGNOSTICS=1`에서는 경고도 계속 진단 로그에 남는다.

`NativeResultTests`는 실제 SDK 어댑터를 제어 가능한 upstream C API fixture와 연결한다. 네 가지 비정사각형 기본 비율·정사각형·양축 올림의 정확한 RGB 픽셀과 행 간격, 경고 뒤 성공, 실패 사유, 잘못된 크기·채널·개수·빈 출력 거부, 결과 메모리 및 콜백 정리를 검사한다. 실제 모델 추론은 별도 실기 검증으로 수행한다.

## 실행 계약

`Generation/NativeDiffusion.hpp`는 같은 프로세스에서 로컬 체크포인트를 읽고 RGB 바이트를 반환한다. 모델 다운로드·기기 연결·동기화·출력 저장은 수행하지 않는다. 소비 앱이 작업 스레드에서 호출하고 결과를 자신의 저장소에 게시한다. 기존 데스크톱 Python worker는 별도로 유지한다.

`IILD_ENABLE_NATIVE_DIFFUSION=ON`은 stable-diffusion.cpp의 `d04e8950c1ec8d30248cbe996682b3182fb1adf6` 및 해당 ggml 커밋을 빌드한다. iOS의 새 구성에서는 기본 ON이다. 체크포인트/토크나이저/텐서 실행을 직접 구현하는 대신 이 라이브러리의 C API를 사용한다. upstream은 활발히 변경되므로 커밋을 고정했다. 두 프로젝트는 MIT이고 고지문을 SDK에 설치한다. 서버·Web UI·WebP/WebM 및 RPC는 제외한다. Apple은 내장 Metal 코드를 사용하고 나머지 환경은 CPU를 사용한다.

한 번에 한 이미지를 실행하고 라이브러리의 전역 콜백 때문에 네이티브 호출을 직렬화한다. 모델 원본을 mmap으로 읽고 지원되는 실행 경로에서 다음 구간 프리페치를 사용한다. Apple에서는 Metal 권장 작업 세트에서 256 MiB를 제외한 값과 현재 사용 가능 메모리에서 여유 공간을 뺀 값 중 작은 값을 256 MiB 단위로 내림하여 관리 GPU 예산으로 사용한다. 여유 공간은 최소 768 MiB 또는 사용 가능 메모리의 1/6이다. iOS는 프로세스의 남은 한도, macOS는 현재 free+inactive 페이지를 조회한다. 전체 CPU 코어를 사용하며 Apple 작업 스레드는 생성 중 USER_INITIATED 우선순위로 실행하고 이전 우선순위를 복원한다. 이 예산은 전체 앱 메모리 상한이나 특정 iPhone에서 임의의 SDXL 모델이 실행된다는 보장이 아니다. 메모리 한계와 지원 체크포인트는 실제 기기로 별도 검증해야 한다. Diffusers 디렉터리는 현재 네이티브 API의 입력이 아니다.

취소는 호출 전·엔진 대기·텐서 로딩 사이·연산 구간 사이에서 확인한다. 취소되거나 결과가 손상되면 RGB를 반환하지 않는다. 프로세스 수명보다 짧은 호출자 데이터는 콜백이 끝날 때까지 유지해야 한다. `NativeDiffusionTests`는 취소·잘못된 크기·제한 시간 값·누락/잘못된 모델 거부를 검사하며 실제 모델 추론 증거는 아니다.

공식 근거: [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), [고정 C API](https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/include/stable-diffusion.h), [MIT 라이선스](https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/LICENSE).

2026-09-10: macOS와 iOS arm64 네이티브 빌드, 네이티브 기능 ON/OFF 계약 검사, Dreamscapes 설치 소비자 빌드를 검증했다. 실제 macOS 소비자에서 `redLilyIllu_v10.safetensors`의 512×512·20스텝 결과를 로컬 컨테이너에 게시하고 PNG를 검사했다. iOS는 빌드·번들까지이며 연결된 iPhone에서의 모델 적재·생성·메모리 한계는 아직 검증하지 않았다. `Product/Dreamscapes/build/society-local-generation/REPORT.md`에 실행 증거를 기록했다.

2026-09-11: `generateNativeImageWithProgress`는 Loading/Encoding/Denoising/Decoding을 구분한다. 기존 두 정수 콜백에는 실제 Denoising 이벤트만 전달한다. `cmake/patches/native-progress-cancellation.patch`는 고정 upstream에 단계 콜백과 중단 검사를 추가하며, 텐서 로딩 및 연산 구간 사이에서 취소를 확인한다. SDK 구성은 패치 적용 가능 여부를 검사하고 FetchContent 소스 경로 override에도 적용한다. 원본/패치의 충돌은 구성 오류이다.

`timeoutMilliseconds`는 기본 900000(15분), 양수이며 엔진 대기까지 포함한다. 취소 가능한 timed mutex로 전역 콜백 사용을 직렬화한다. 제한 시간 초과는 오류, 사용자 취소는 cancelled로 반환한다. 진행 중인 단일 파일 읽기·GPU 연산을 강제로 죽이지는 않는다. `IILD_NATIVE_DIAGNOSTICS=1`은 기기 진단용 upstream 로그를 stderr에 출력한다. 모델 경로 등 실행 정보가 포함될 수 있어 기본은 꺼져 있다.

취소 시 텍스트 인코더가 빈 텐서를 단언하는 upstream 경로를 피하기 위해, 로딩 스레드를 모두 join한 뒤 호출 스레드에서 스택을 해제한다. SDK 경계가 이를 cancelled 또는 제한 시간 오류로 변환한다. 컨텍스트 초기화 도중 해제도 보장한다. `IILD_NATIVE_TEST_MODEL=<absolute checkpoint> NativeDiffusionTests <absolute scratch directory>`는 실제 모델의 로딩·노이즈 제거 진입 취소 및 제한 시간을 검사한다.

2026-09-11 최종 실기기 검증: iPhone 15 Pro Max의 동일 체크포인트, 512×512·20스텝 로컬 생성이 294.647초에 완료되고 이미지 표시·Society 파일 저장을 확인했다. 로딩 중 취소는 시작 3초 뒤 요청하여 3.162초에 종료되었다. 네이티브 SDK 79/79 CTest와 OFF 빌드의 계약 검사도 통과했다. 이는 검증한 모델·기기 조합의 관측 결과이다.

## 디스크 및 메모리 캐시

기본 SDK 요청은 원본 체크포인트를 직접 읽는다. `q8CacheDirectory`에 절대 경로를 지정하면 기존 upstream 스트리밍 변환기로 Q8_0 GGUF 사본을 준비하며 VAE는 F16으로 유지한다. Dreamscapes iOS QuickGenerate는 이 경로를 기본으로 사용한다. Society App Group의 원본은 보존하며 네트워크 다운로드나 원본 덮어쓰기는 없다. 첫 변환에는 추가 시간과 디스크 공간이 필요하고, 이후에는 검증된 사본을 재사용한다. Q8은 같은 seed에서도 원본 정밀도와 픽셀이 달라질 수 있다.

캐시 키에는 고정 변환기/규칙 버전과 원본 식별자가 들어간다. 별도 manifest의 전체 원본 식별자, 출력 크기·mtime·POSIX inode/ctime, GGUF 헤더를 확인한다. 원본이 없거나 변경된 경우 이전 캐시를 선택하지 않는다. 고유 임시 디렉터리에서 변환 후 원본을 재확인하고 파일과 manifest를 원자적으로 게시한다. 취소·실패 시 임시 파일을 제거하며 불완전한 manifest는 재변환한다. 모델별 캐시는 앱 CacheLocation 아래에 있고 OS가 공간 압박 시 삭제할 수 있다. 이전 원본 버전의 캐시는 새로운 요청에 사용되지 않는다.

SDK는 성공한 최근 모델 하나의 컨텍스트·파일 매핑·백엔드가 유지하는 가중치와 실행 계획을 다음 생성에 재사용한다. 경로·크기·수정 시각과 POSIX 파일 식별자·변경 시각이 바뀌면 교체한다. 모델 전체를 매 요청 해시하지 않으며, 생성 중 파일 교체가 감지되면 결과를 폐기한다. 취소·시간 초과·실패는 부분 컨텍스트를 폐기한다. 서로 다른 요청은 기존 전역 엔진 잠금으로 직렬 실행한다.

`releaseNativeDiffusionCache()`는 호출자를 대기시키지 않는다. 유휴 컨텍스트는 즉시 해제하며 실행 중이라면 현재 요청 종료 때 해제한다. 앱은 실제 백그라운드 진입, UIKit 메모리 경고, 컨트롤러 종료에 이를 호출한다. 모든 가중치가 RAM에 상주한다는 뜻은 아니며 큰 모델은 예산 내에서 계속 구간별 적재한다.

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

## 실행 일시 정지

`generateNativeImageWithExecutionControl()`은 선택적인 `NativeExecutionControl`을 통해 동일한 네이티브 요청을 일시 정지하고 재개한다. 기존 요청/결과 구조체와 함수 ABI는 유지하며 별도 진입점과 설치 헤더를 추가했다. 실행 제어는 Qt/UIKit을 참조하지 않는다.

소비자는 OS 실행 권한을 잃기 전에 `setPaused(true)`, 복귀하면 `setPaused(false)`를 호출한다. 엔진의 기존 텐서 적재·연산 구간 경계에서 조건 변수로 대기하므로 이미 제출한 GPU 호출이 반환되기까지는 시간이 필요하다. latent·모델 컨텍스트·seed를 유지하며 작업을 재시작하지 않는다. 정지 시간은 추론 제한 시간에서 제외하고, 여러 적재 스레드가 동시에 기다려도 한 번만 계산한다. 정지 중에도 기존 취소 토큰을 확인하므로 취소/소비자 파괴가 대기에 갇히지 않는다. OS의 프로세스 종료나 메모리 회수 이후 복원은 제공하지 않는다.

`NativeResultTests`는 실제 어댑터와 제어된 C API로 연산 경계 대기, 제한 시간보다 긴 정지 후 같은 요청 완료, 정지 중 취소와 자원 해제를 검증한다. 백그라운드 GPU 실행 허용 여부와 앱의 OS 작업 등록은 소비자가 담당한다.
