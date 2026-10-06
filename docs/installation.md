<a id="local-installation"></a>

# 로컬 설치

`install.sh`는 `$HOME/.local/SDK/iiLocalDiffusion`에 라이브러리, 생성 및 모델 병합 진입점을 설치합니다. 다음을 사용하여 기존 결제에서 실행하세요.
[네이티브 빌드 요구 사항](../README.md#build-and-test) 사용 가능:

```bash
./install.sh
```

스크립트는 저장소의 `build/` 디렉토리에 릴리스 빌드를 구성하고 빌드한 다음 CTest 를 실행한 다음 패키지를 설치합니다. 기존 CMake 캐시 설정이 MLX, Core ML 및 선택적 LibTorch 에 대해 유지됩니다. 따라서 구성된 백엔드는 설치 후에도 활성화 상태로 남아 있으며, 해당 빌드에서 비활성화된 백엔드를 설치로 추가하지 않습니다. 새 Python 환경 또는 모델 다운로드가 수행되지 않습니다.

최종 세션 확인은 저장소의 SD 1.5 메타데이터 픽스처 를 명시적인 모델 소스로 제공하고 HiRes 구성을 출력합니다. 그것은 가중치를 로드하거나 이미지를 생성하지 않습니다. 설치자 계약 테스트는 스텁된 네이티브 빌드 명령과 실제 리로케이트된 Python 런처를 사용하여 이 마지막 단계를 확인하며, 공백이 포함된 설치 경로를 포함합니다. 그것은 Python 3.10+ CMake 에 의해 선택된 인터프리터를 리로케이트된 런처에 사용하며, PATH 에서 먼저 나타날 수 있는 더 오래된 시스템 `python3` 대신 사용합니다. 참조 소스 탐색은 그들로 내려가기 전에  `.venv` ,  `__pycache__`  및  `.git` 를 가지치기하므로, 기존 모델  런타임 (런타임) 는 설치 확인을 부풀리거나 임시 패키지를 증가시키지 않습니다.

접두사 및 빌드 동시성은 환경 변수로 제공될 수 있습니다.

```bash
IILD_INSTALL_PREFIX="$HOME/.local/SDK/iiLocalDiffusion" \
IILD_BUILD_JOBS=4 \
  ./install.sh
```

이는 기본값이기도 합니다. 설치된 사본을 업데이트하려면 라이브러리 또는 Python 소스를 변경한 후 명령을 반복하십시오.

<a id="checkout-relocation"></a>

## 결제 이전

체크아웃은  `Workspace/SDK/iiLocalDiffusion`  아래에 존재하며, 기본  CMake  접두사는 직접 구성할 때 포함하여  `~/.local/SDK/iiLocalDiffusion`  입니다. 설치 프로그램 인수는  CMake 에 전달됩니다. 구성된 체크아웃을 이동한 후, 임의의 백엔드와 다시 제공된  SDK  설정과 함께  `./install.sh --fresh` 를 사용하세요, 예를 들어  `-DIILD_ENABLE_LIBTORCH=ON -DTorch_DIR=/new/sdk/cmake/Torch` 와 같이. 다운로드된 모델과  Python 환경을 유지하되, 실행하기 전에 오래된 빌드 메타데이터를 재생성하고 이동된 환경 엔트리 포인트 경로를 업데이트하세요. 설치 프로그램 계약 테스트는 기본  SDK 와 사용자 정의 접두사를 모두 확인합니다.

<a id="installed-files-and-retained-runtimes"></a>

## 설치된 파일 및 유지된 런타임

접두사에는 다음 공개 진입점과 리소스가 포함됩니다.

|접두사|에 대한 상대 경로 목차|
| --- | --- |
| `bin/iild-run` |네이티브 구성 요소 계산 및 메타데이터 검사 CLI|
| `bin/iild-generate` |통합 Python 이미지 생성 실행기|
| `bin/iild-merge` |Python 가중 체크포인트/LoRA 합계/뺄셈 실행기|
| `lib/` |공유 라이브러리 및 활성화된 번들 네이티브 런타임 리소스|
| `lib/cmake/iiLocalDiffusion/` |CMake 패키지 구성 및 내보낸 대상|
| `include/` |공개 C++ 헤더|
| `share/licenses/iiLocalDiffusion/` |설치된 종속성 라이센스 공지|
| `share/doc/iiLocalDiffusion/` |README 및 문서|
| `share/iiLocalDiffusion/reference/` |Python 소스, JSON 구성 및 요구 사항 파일|

네이티브 실행 파일은 상대 런타임 검색 경로를 사용하여 설치된 라이브러리를 찾습니다. 활성화되면 MLX 리소스가 설치됩니다. 시스템 라이브러리와 선택적 LibTorch SDK는 외부 종속성을 유지합니다. 해당 백엔드를 구성하는 데 사용된 SDK를 유지합니다.

이는 기존 로컬 런타임을 재사용하는 개발자 설치입니다. 있는 경우 설치 프로그램은 기호 링크를 사용하여 이러한 소스 경로를 설치된 참조 트리에 연결합니다.

|설치된 링크|기존 소스 경로|
| --- | --- |
| `share/iiLocalDiffusion/reference/diffusers/.venv` |`reference/diffusers/.venv` 결제 중|
| `share/iiLocalDiffusion/build` |`build/` 결제 중|

두 번째 링크는 기존 개발 캐시와 선택적  ComfyUI 자산을 사용 가능하게 만듭니다. 독립형 이미지/비디오 추론에는  Diffusers 환경만 필요하며,  SD1/SDXL 구성과 토크나이저 리소스가 설치에 복사됩니다. Python 소스 파일들이 접두사로 복사됩니다. 체크아웃의 연결된 가상 환경과  `build/` 디렉토리를 유지하여 해당 런타임과 캐시들을 사용 가능하게 유지하세요. 이들을 이동하거나 삭제하면 해당 링크들이 끊깁니다. 체크아웃을 재배치하는 경우, 해당 의존성들을 복원한 후 링크들을 새 위치로 업데이트하세요. 설치기는 다른 곳에 지시하는 기존 런타임 목적지를 대체하지 않습니다.

<a id="use-from-cmake"></a>

## CMake에서 사용

애플리케이션은 내보낸 대상을 연결합니다.

```cmake
cmake_minimum_required(VERSION 3.31)
project(LocalDiffusionConsumer LANGUAGES CXX)

find_package(iiLocalDiffusion 0.3 REQUIRED CONFIG)
add_executable(consumer main.cpp)
target_compile_features(consumer PRIVATE cxx_std_20)
target_link_libraries(consumer PRIVATE iiLocalDiffusion::iiLocalDiffusion)
```

예를 들어, `main.cpp`는 로컬 Diffusers 패키지를 검사할 수 있습니다.

```cpp
#include <ModelManifest/DiffusionModelManifest.hpp>

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    const auto manifest = iild::loadModelManifest(argv[1]);
    (void)manifest;
}
```

소비자 프로젝트의 디렉터리에서 설치된 접두사를 사용하여 구성하고 빌드합니다.

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$HOME/.local/SDK/iiLocalDiffusion"
cmake --build build --parallel 4
./build/consumer /absolute/path/to/diffusers-package
```

macOS 에서, 패키지의 `lib/` 디렉토리가 `LIBRARY_PATH` 에 존재할 때, 내보낸 대상 또한 런타임 검색 경로를 제공합니다. CMake 는 해당 디렉토지를 암시적인 링커 위치로 취급하지만, 라이브러리의 `@rpath` 설치 이름에 대해 여전히 dyld 가 `LC_RPATH` 가 필요합니다. 애플리케이션은 위와 같이 표시된 대상 링크만 필요하며, `BUILD_RPATH` 또는 `DYLD_LIBRARY_PATH` 조정을 필요로 하지 않습니다. 재배치된 설치된 소비자 테스트는 이 유전 환경을 포함하여, 공백을 포함하는 설치 경로를 다룹니다.

C++ API는 구성 요소 계산 및 메타데이터 검사를 수행합니다. 전체 이미지 생성 및 [모델 병합](model-merging.md)는 별도의 Python 진입점을 사용합니다. 두 Python 명령 모두 아래의 인터프리터 선택을 공유합니다.

<a id="use-the-installed-commands"></a>

## 설치된 명령 사용

쉘의 `PATH`를 변경하지 않고 전체 명령 경로를 사용하십시오.

```bash
"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-run" devices
"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-run" compute
"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-run" inspect \
  /absolute/path/to/diffusers-package

"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-generate" --list-base-models
"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-merge" --help
"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-generate" --model /absolute/path/image-diffusers \
  --base-model Illustrious --print-config
```

다운로드한 로컬 체크포인트와 4개의 연속 HiRes 개선을 사용하여 생성하려면 다음을 수행하세요.

```bash
"$HOME/.local/SDK/iiLocalDiffusion/bin/iild-generate" \
  --model /absolute/path/to/illustrious.safetensors \
  --base-model Illustrious \
  --prompt "a red cube" --device mps \
  --hires-fix --hires-passes 4 --hires-scale 1.5 \
  --output-dir /absolute/path/to/output
```

이것은 모델의 동반 가중치와 구성된 로컬 생성 런타임 를 필요로 합니다. 모델 입력과 런타임 설정에 대해서는 [다운로드-이미지 생성](local-image-generation.md) 를 참조하십시오. 설치만으로는 모든 카탈로그 항목에 대한 호환성을 확립하거나 누락된 모델 구성 요소를 제공하지 않습니다.

HiRes 수정이 활성화되면, 생략된 반복 횟수는 **1**로 기본값으로 설정됩니다. `--hires-passes` 은 양의 정수를 수락하며, 각 패스는 동일한 설정으로 이전 패스의 출력을 정제합니다. HiRes 수정은 요청되지 않는 한 비활성으로 유지됩니다. 지원되는 옵션과 단계별 기원에 대해서는 [HiRes 수정](hires-fix.md) 을 참조하십시오.

런처는 연결된 Oracle 가상 환경을 사용할 수 있습니다. 다른 기존 Python 인터프리터를 선택하려면 해당 실행 파일을 명시적으로 제공하십시오.

```bash
IILD_PYTHON_EXECUTABLE=/absolute/path/to/venv/bin/python \
  "$HOME/.local/SDK/iiLocalDiffusion/bin/iild-generate" --check-runtime
```

해당 인터프리터에는 선택한 Python 백엔드에 대한 종속성이 포함되어 있어야 합니다. 관리형 로컬 엔진에는 로컬 생성 가이드에 문서화된 자체 환경도 있습니다.

<a id="manual-cmake-installation"></a>

## 수동 CMake 설치

이미 구성되고 테스트된 빌드의 경우 CMake는 직접 설치도 지원합니다.

```bash
cmake --install build --config Release \
  --prefix "$HOME/.local/SDK/iiLocalDiffusion"
```

이것은 네이티브 패키지와 도구 및 Python 소스를 설치하지만, 개발자 런타임 링크를 생성하거나 Python 의존성을 설치하지는 않습니다. 이 경로를 사용할 때 `IILD_PYTHON_EXECUTABLE` 를 기존 환경으로 설정하고 선택된 생성 백엔드의 런타임 경로를 구성하십시오.

<a id="deforum-video-runtime"></a>

## 디포럼 비디오 런타임

설치된 `iild-generate --backend deforum` 는 2D 일표와 프레임 피드백 런타임 를 포함합니다. 선택된 Python 환경에서 선택적 `reference/diffusers/requirements-deforum.txt` 를 설치하고 FFmpeg 와 FFprobe 을 PATH 에 제공하세요. 비디오 도구, OpenCV 휠 또는 가중치가 네이티브 라이브러리에 포함되지 않았습니다. `--backend deforum --max-frames 5 --print-config` 는 이러한 선택적 의존성 없이 작동합니다. 이동된 패키지 소비자 테스트는 설치된 모듈, 예: 의존성 파일, 문서 및 CLI 설정을 확인합니다. [Deforum 비디오 생성](deforum-video.md)을 참조하세요.

<a id="interpolator-video-runtime"></a>

## 보간기 비디오 런타임

설치된 `iild-generate --backend interpolator` 는 시작/종료 프롬프트와 시드를 받아 검증된 MP4 및 PNG 프레임과 메타데이터를 생성합니다. 기존 Diffusers 환경과 Deforum 과 동일한 외부 FFmpeg /FFprobe 도구를 사용하며 OpenCV 또는 추가 인터폴레이션 모델이 필요하지 않습니다. 공유 비디오 모듈, 엔드포인트 런타임 및 예제 JSON 는 Python 참조 파일과 함께 설치됩니다. `--backend interpolator --end-prompt forest --end-seed 43 --print-config` 도 추론 의존성 없이 작동합니다. [인터폴레이터 비디오 생성](interpolator-video.md)을 참조하세요.

<a id="updating-only-the-generation-runtime"></a>

## 생성 런타임 만 업데이트합니다.

`cmake --install build --component Runtime --prefix <existing-prefix>` 는 런처, Python 모듈, 번들 토크나이저/설정 파일 및 문서를 설치합니다. 설치된 네이티브 C++ 라이브러리 및 선택적 백엔드를 유지합니다. 정규 전체 설치에는 동일한 런타임 구성 요소가 포함됩니다. 선택적 Metal 컴파일러가 없는 머신에서 백엔드를 교체하지 않고 Python 추론 수정을 허용합니다. 존재하는 환경 또는 `IILD_PYTHON_EXECUTABLE` 를 사용하여 Python 의존성을 설치하지 않습니다. 위의 문서대로 사용하세요.
