include(FetchContent)
include(CheckLanguage)

option(IILD_ENABLE_MLX "Build the native MLX compute runtime" ON)
set(IILD_MLX_GPU_BACKEND "auto" CACHE STRING "MLX GPU backend: auto, metal, cuda, or none")
set_property(CACHE IILD_MLX_GPU_BACKEND PROPERTY STRINGS auto metal cuda none)

if(NOT IILD_ENABLE_MLX)
    return()
endif()

set(iild_gpu_backend "${IILD_MLX_GPU_BACKEND}")
if(iild_gpu_backend STREQUAL "auto")
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        set(iild_gpu_backend metal)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            set(iild_gpu_backend cuda)
        else()
            set(iild_gpu_backend none)
        endif()
    else()
        set(iild_gpu_backend none)
    endif()
endif()
if(NOT iild_gpu_backend MATCHES "^(metal|cuda|none)$")
    message(FATAL_ERROR "IILD_MLX_GPU_BACKEND must be auto, metal, cuda, or none")
endif()
if(iild_gpu_backend STREQUAL "metal")
    if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        message(FATAL_ERROR "The Metal backend requires Apple Silicon and macOS")
    endif()
    execute_process(COMMAND xcrun -sdk macosx metal --version
        RESULT_VARIABLE iild_metal_compiler_result OUTPUT_QUIET ERROR_QUIET)
    if(NOT "${iild_metal_compiler_result}" STREQUAL "0")
        message(FATAL_ERROR
            "The Metal compiler is missing. Install Xcode's Metal Toolchain; see docs/hardware-compute.md")
    endif()
elseif(iild_gpu_backend STREQUAL "cuda" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The native CUDA build currently targets Linux/NVIDIA")
endif()

# Pin the runtime and its fetched dependencies; no moving branches are used.
FetchContent_Declare(mlx
    URL https://codeload.github.com/ml-explore/mlx/tar.gz/1f8e74e3f12f31365464a6867c6579f0e9b29d85
    URL_HASH SHA256=cb988a5bdc38c798918d042b9b1c6edda3ccc5f23a2155138d3aa5c1b2acc301
    SYSTEM
)
FetchContent_Declare(metal_cpp
    URL https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip
    URL_HASH SHA256=4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4
)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_Declare(fmt
    URL https://codeload.github.com/fmtlib/fmt/tar.gz/refs/tags/12.1.0
    URL_HASH SHA256=ea7de4299689e12b6dddd392f9896f08fb0777ac7168897a244a6d6085043fea
    EXCLUDE_FROM_ALL
)
FetchContent_Declare(cccl
    URL https://github.com/NVIDIA/cccl/releases/download/v3.1.3/cccl-v3.1.3.zip
    URL_HASH SHA256=30f388ef784eb691d7de9d2cf918d53ab33464672500a17945b99d16af136cd6
)
FetchContent_Declare(nvtx3
    URL https://codeload.github.com/NVIDIA/NVTX/tar.gz/refs/tags/v3.1.1
    URL_HASH SHA256=30c9fe2034d5da90c7b2051570dd8b9d59e05b8b9037a1c80bb088dce9c1f5d9
    SOURCE_SUBDIR c EXCLUDE_FROM_ALL
)
FetchContent_Declare(cudnn
    URL https://codeload.github.com/NVIDIA/cudnn-frontend/tar.gz/refs/tags/v1.16.0
    URL_HASH SHA256=77ed731e0885ea2de1b6f272ad156b48292c2299a2571ae576280011fd5626bd
    EXCLUDE_FROM_ALL
)
FetchContent_Declare(cutlass
    URL https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.4.2
    URL_HASH SHA256=ef62816841b8cbcd0ed3ec45d3ab56cf67d569a0f39329b43eeeebea86f58473
    SOURCE_SUBDIR include EXCLUDE_FROM_ALL
)

block()
    set(BUILD_SHARED_LIBS ON)
    set(MLX_BUILD_TESTS OFF)
    set(MLX_BUILD_EXAMPLES OFF)
    set(MLX_BUILD_BENCHMARKS OFF)
    set(MLX_BUILD_PYTHON_BINDINGS OFF)
    set(MLX_BUILD_GGUF OFF)
    set(MLX_BUILD_SAFETENSORS ON)
    set(MLX_BUILD_CPU ON)
    set(MLX_BUILD_METAL OFF)
    set(MLX_BUILD_CUDA OFF)
    set(MLX_LOAD_CUDA_LIBS_FROM_PYTHON OFF)
    set(MLX_METAL_JIT ON)
    if(iild_gpu_backend STREQUAL "metal")
        set(MLX_BUILD_METAL ON)
    elseif(iild_gpu_backend STREQUAL "cuda")
        set(MLX_BUILD_CUDA ON)
    endif()
    FetchContent_MakeAvailable(mlx)
endblock()

set_property(GLOBAL APPEND PROPERTY JOB_POOLS iild_mlx_compile=4)
set_property(TARGET mlx PROPERTY JOB_POOL_COMPILE iild_mlx_compile)
if(APPLE)
    set_property(TARGET mlx PROPERTY INSTALL_RPATH "@loader_path")
else()
    set_property(TARGET mlx PROPERTY INSTALL_RPATH "$ORIGIN")
endif()

if(iild_gpu_backend STREQUAL "cuda")
    FetchContent_GetProperties(cccl SOURCE_DIR cccl_source)
    # Uninstalled native binaries also need CCCL for NVRTC. Installed headers
    # take precedence over this build-tree fallback in MLX's path resolver.
    target_compile_definitions(mlx_dirs PRIVATE MLX_CCCL_DIR="${cccl_source}/include")
    # The upstream header-only release zip omits the full license document.
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dependencies")
    set(cccl_license "${CMAKE_CURRENT_BINARY_DIR}/dependencies/cccl-3.1.3-LICENSE")
    file(DOWNLOAD https://raw.githubusercontent.com/NVIDIA/cccl/v3.1.3/LICENSE
        "${cccl_license}"
        EXPECTED_HASH SHA256=f96f51edda77fb9897de29d924d615e6bd153f4969db85fc6b7a22d7a624631e
        TLS_VERIFY ON)
    install(FILES "${cccl_license}"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/iiLocalDiffusion/cccl" RENAME LICENSE)
endif()

foreach(dependency IN ITEMS mlx metal_cpp json fmt nvtx3 cudnn cutlass)
    FetchContent_GetProperties(${dependency} SOURCE_DIR dependency_source POPULATED dependency_ready)
    if(dependency_ready)
        if(dependency MATCHES "^(metal_cpp|nvtx3|cudnn|cutlass)$")
            set(license_file LICENSE.txt)
        elseif(dependency STREQUAL "json")
            set(license_file LICENSE.MIT)
        else()
            set(license_file LICENSE)
        endif()
        install(FILES "${dependency_source}/${license_file}"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/iiLocalDiffusion/${dependency}")
    endif()
endforeach()
message(STATUS "iiLocalDiffusion native runtime: MLX 0.32.2 / ${iild_gpu_backend}")
if(iild_gpu_backend STREQUAL "none")
    message(STATUS "No GPU backend compiled: automatic compute will fail; CPU requires explicit selection")
endif()
