include(FetchContent)

option(IILD_ENABLE_LIBTORCH "Build the optional AMD ROCm / explicit CPU LibTorch backend" OFF)
if(NOT IILD_ENABLE_LIBTORCH)
    return()
endif()

# Supply a vendor-compatible SDK through Torch_DIR/CMAKE_PREFIX_PATH. Never
# download a generic NVIDIA/CPU package in place of an AMD HIP distribution.
find_package(Torch 2.9 CONFIG REQUIRED)
if(Torch_VERSION VERSION_GREATER_EQUAL 3.0)
    message(FATAL_ERROR "The LibTorch backend requires a reviewed PyTorch 2.x SDK (>=2.9)")
endif()
if(TORCH_CXX_FLAGS MATCHES "_GLIBCXX_USE_CXX11_ABI=0")
    message(FATAL_ERROR "Use a C++11-ABI LibTorch build; ABI=0 conflicts with the public C++ API")
endif()
separate_arguments(iild_torch_compile_options NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

# Retain the matching SDK's copyright/license for code instantiated from its
# C++ headers. Vendor runtime binaries themselves are still not redistributed.
set(IILD_LIBTORCH_LICENSE "" CACHE FILEPATH "Optional explicit path to the selected LibTorch LICENSE")
set(iild_torch_license "${IILD_LIBTORCH_LICENSE}")
if(NOT iild_torch_license)
    foreach(candidate IN ITEMS "${TORCH_INSTALL_PREFIX}/LICENSE"
            "${TORCH_INSTALL_PREFIX}/share/LICENSE")
        if(EXISTS "${candidate}")
            set(iild_torch_license "${candidate}")
            break()
        endif()
    endforeach()
endif()
if(NOT iild_torch_license)
    file(GLOB iild_torch_wheel_licenses
        "${TORCH_INSTALL_PREFIX}/../torch-${Torch_VERSION}*.dist-info/licenses/LICENSE"
        "${TORCH_INSTALL_PREFIX}/../torch-${Torch_VERSION}*.dist-info/LICENSE")
    list(LENGTH iild_torch_wheel_licenses iild_torch_license_count)
    if(iild_torch_license_count EQUAL 1)
        list(GET iild_torch_wheel_licenses 0 iild_torch_license)
    endif()
endif()
if(NOT iild_torch_license OR NOT EXISTS "${iild_torch_license}" OR IS_DIRECTORY "${iild_torch_license}")
    message(FATAL_ERROR "Provide the matching SDK LICENSE using IILD_LIBTORCH_LICENSE")
endif()
install(FILES "${iild_torch_license}"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/iiLocalDiffusion/libtorch" RENAME LICENSE)

FetchContent_Declare(iild_safetensors_cpp
    URL https://codeload.github.com/syoyo/safetensors-cpp/tar.gz/af90b6c3006cdcecf8b7d7254f5f32d301728acc
    URL_HASH SHA256=f978132be070d6e0ae0be097c6cd5b65edeedf19f78c57158b2c43ffa412323d
    SOURCE_SUBDIR iild-header-only
    SYSTEM
)
FetchContent_MakeAvailable(iild_safetensors_cpp)
install(FILES "${iild_safetensors_cpp_SOURCE_DIR}/LICENSE"
    "${iild_safetensors_cpp_SOURCE_DIR}/safetensors.hh"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/iiLocalDiffusion/safetensors-cpp")
message(STATUS "iiLocalDiffusion optional LibTorch: ${Torch_VERSION}; ROCm requires an AMD HIP build at runtime")
