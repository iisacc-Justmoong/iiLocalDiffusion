cmake_minimum_required(VERSION 3.31)

project(
    iiLocalDiffusion
    VERSION 0.3.0
    DESCRIPTION "A local generative-model pipeline runtime"
    LANGUAGES CXX
)

if(PROJECT_IS_TOP_LEVEL AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(_sdk_home "$ENV{HOME}")
    if(_sdk_home STREQUAL "")
        set(_sdk_home "$ENV{USERPROFILE}")
    endif()
    set(CMAKE_INSTALL_PREFIX "${_sdk_home}/.local/SDK/iiLocalDiffusion"
        CACHE PATH "SDK installation prefix" FORCE)
    unset(_sdk_home)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
find_package(Threads REQUIRED)

find_package(json-c CONFIG QUIET)
if(TARGET json-c::json-c)
    set(IILD_JSON_C_TARGET json-c::json-c)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(JSON_C REQUIRED IMPORTED_TARGET json-c>=0.18)
    set(IILD_JSON_C_TARGET PkgConfig::JSON_C)
endif()

cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES ${IILD_JSON_C_TARGET})
check_cxx_source_compiles([[
#include <json-c/json_c_version.h>
#if JSON_C_VERSION_NUM < ((0 << 16) | (18 << 8))
#error json-c 0.18 or newer is required
#endif
int main() { return 0; }
]] IILD_JSON_C_VERSION_SUPPORTED)
cmake_pop_check_state()
if(NOT IILD_JSON_C_VERSION_SUPPORTED)
    message(FATAL_ERROR "iiLocalDiffusion requires json-c 0.18 or newer")
endif()

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif()

option(IILD_BUILD_TOOLS "Build iiLocalDiffusion command-line tools" ${PROJECT_IS_TOP_LEVEL})
option(IILD_INSTALL_PYTHON_REFERENCE "Install the Python generation entry point and reference sources" ON)
option(IILD_ENABLE_COREML "Build the Apple Core ML component runtime" ${APPLE})
if(IILD_ENABLE_COREML AND NOT APPLE)
    message(FATAL_ERROR "Core ML requires an Apple platform; set IILD_ENABLE_COREML=OFF")
endif()
include(cmake/IildMlx.cmake)
include(cmake/IildTorch.cmake)

add_library(iiLocalDiffusion SHARED
    src/Compute/ComputeRuntime.cpp
    src/Compute/CoreMLModel.cpp
    src/Compute/LinearLayer.cpp
    src/Compute/TorchLinearBackend.cpp
    src/Flux/FluxModelManifest.cpp
    src/ModelManifest/DiffusionModelManifest.cpp
    src/ModelManifest/ModelManifest.cpp
    src/StableDiffusion/StableDiffusionModelManifest.cpp
)
add_library(iiLocalDiffusion::iiLocalDiffusion ALIAS iiLocalDiffusion)

target_compile_features(iiLocalDiffusion PUBLIC cxx_std_20)
set_target_properties(iiLocalDiffusion PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    VERSION ${PROJECT_VERSION}
    VISIBILITY_INLINES_HIDDEN YES
)

target_include_directories(iiLocalDiffusion
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(iiLocalDiffusion
    PRIVATE
        ${IILD_JSON_C_TARGET}
        Threads::Threads
)

if(IILD_ENABLE_MLX)
    target_link_libraries(iiLocalDiffusion PRIVATE mlx)
    target_compile_definitions(iiLocalDiffusion PRIVATE IILD_HAS_MLX=1)
else()
    target_compile_definitions(iiLocalDiffusion PRIVATE IILD_HAS_MLX=0)
endif()
if(IILD_ENABLE_LIBTORCH)
    target_link_libraries(iiLocalDiffusion PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(iiLocalDiffusion SYSTEM PRIVATE
        ${TORCH_INCLUDE_DIRS} "${iild_safetensors_cpp_SOURCE_DIR}")
    target_compile_options(iiLocalDiffusion PRIVATE ${iild_torch_compile_options})
    target_sources(iiLocalDiffusion PRIVATE src/Compute/TorchSafetensors.cpp)
    target_compile_definitions(iiLocalDiffusion PRIVATE IILD_HAS_LIBTORCH=1)
else()
    target_compile_definitions(iiLocalDiffusion PRIVATE IILD_HAS_LIBTORCH=0)
endif()
if(IILD_ENABLE_COREML)
    enable_language(OBJCXX)
    find_library(IILD_COREML_FRAMEWORK CoreML REQUIRED)
    find_library(IILD_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    target_sources(iiLocalDiffusion PRIVATE src/Compute/CoreMLModel.mm)
    set_source_files_properties(src/Compute/CoreMLModel.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(iiLocalDiffusion PROPERTIES OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED YES
        OBJCXX_EXTENSIONS OFF OBJCXX_VISIBILITY_PRESET hidden)
    target_link_libraries(iiLocalDiffusion PRIVATE ${IILD_COREML_FRAMEWORK} ${IILD_FOUNDATION_FRAMEWORK})
    target_compile_definitions(iiLocalDiffusion PRIVATE IILD_HAS_COREML=1)
else()
    target_compile_definitions(iiLocalDiffusion PRIVATE IILD_HAS_COREML=0)
endif()
if(APPLE)
    set_property(TARGET iiLocalDiffusion PROPERTY INSTALL_RPATH "@loader_path")
else()
    set_property(TARGET iiLocalDiffusion PROPERTY INSTALL_RPATH "$ORIGIN")
endif()
if(IILD_ENABLE_LIBTORCH)
    # LibTorch/ROCm remains an externally installed SDK, not a bundled copy.
    set_property(TARGET iiLocalDiffusion APPEND PROPERTY INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
endif()

if(MSVC)
    set(IILD_WARNING_OPTIONS /W4 /permissive-)
else()
    set(IILD_WARNING_OPTIONS
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
    )
endif()
target_compile_options(iiLocalDiffusion PRIVATE ${IILD_WARNING_OPTIONS})

if(IILD_BUILD_TOOLS)
    add_executable(iild-run
        tools/iild-run/ComputeCommands.cpp
        tools/iild-run/NeuralCommands.cpp
        tools/iild-run/main.cpp
    )

    target_compile_features(iild-run PRIVATE cxx_std_20)
    target_compile_options(iild-run PRIVATE ${IILD_WARNING_OPTIONS})
    target_link_libraries(iild-run PRIVATE iiLocalDiffusion::iiLocalDiffusion)
    file(RELATIVE_PATH iild_lib_from_bin
        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    if(APPLE)
        set_property(TARGET iild-run PROPERTY INSTALL_RPATH "@loader_path/${iild_lib_from_bin}")
    elseif(UNIX)
        set_property(TARGET iild-run PROPERTY INSTALL_RPATH "$ORIGIN/${iild_lib_from_bin}")
    endif()
endif()

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
    if(UNIX)
        add_test(NAME iiLocalDiffusion.InstallScript
            COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/InstallScriptTest.sh")
    endif()
    find_package(Python3 3.10 COMPONENTS Interpreter QUIET)

    add_executable(ModelManifestTests
        tests/ModelManifestTests.cpp
    )

    target_compile_features(ModelManifestTests PRIVATE cxx_std_20)
    target_compile_definitions(ModelManifestTests PRIVATE
        IILD_TEST_WORK_DIRECTORY="${CMAKE_CURRENT_BINARY_DIR}/test-work"
    )
    target_compile_options(ModelManifestTests PRIVATE ${IILD_WARNING_OPTIONS})
    target_link_libraries(ModelManifestTests PRIVATE iiLocalDiffusion::iiLocalDiffusion)

    add_test(NAME ModelManifestTests COMMAND ModelManifestTests)
    add_executable(ComputeTests tests/ComputeTests.cpp)
    target_compile_features(ComputeTests PRIVATE cxx_std_20)
    target_compile_options(ComputeTests PRIVATE ${IILD_WARNING_OPTIONS})
    target_compile_definitions(ComputeTests PRIVATE
        IILD_TEST_MLX_ENABLED=$<BOOL:${IILD_ENABLE_MLX}>
        IILD_COMPUTE_TEST_DIRECTORY="${CMAKE_CURRENT_BINARY_DIR}/compute-test-work")
    target_link_libraries(ComputeTests PRIVATE iiLocalDiffusion::iiLocalDiffusion Threads::Threads)
    if(IILD_ENABLE_MLX)
        target_link_libraries(ComputeTests PRIVATE mlx)
    endif()
    add_test(NAME ComputeTests COMMAND ComputeTests)
    add_executable(RocmTests tests/RocmTests.cpp)
    target_compile_features(RocmTests PRIVATE cxx_std_20)
    target_compile_options(RocmTests PRIVATE ${IILD_WARNING_OPTIONS})
    target_compile_definitions(RocmTests PRIVATE
        IILD_TEST_LIBTORCH_ENABLED=$<BOOL:${IILD_ENABLE_LIBTORCH}>
        IILD_ROCM_TEST_DIRECTORY="${CMAKE_CURRENT_BINARY_DIR}/rocm-test-work")
    target_link_libraries(RocmTests PRIVATE iiLocalDiffusion::iiLocalDiffusion Threads::Threads)
    if(IILD_ENABLE_LIBTORCH)
        target_link_libraries(RocmTests PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(RocmTests SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
        target_compile_options(RocmTests PRIVATE ${iild_torch_compile_options})
    endif()
    add_test(NAME RocmTests COMMAND RocmTests)
    add_executable(CoreMLTests tests/CoreMLTests.cpp)
    target_compile_features(CoreMLTests PRIVATE cxx_std_20)
    target_compile_options(CoreMLTests PRIVATE ${IILD_WARNING_OPTIONS})
    target_compile_definitions(CoreMLTests PRIVATE IILD_TEST_COREML_ENABLED=$<BOOL:${IILD_ENABLE_COREML}>)
    target_link_libraries(CoreMLTests PRIVATE iiLocalDiffusion::iiLocalDiffusion ${IILD_JSON_C_TARGET} Threads::Threads)
    add_test(NAME CoreMLTests COMMAND CoreMLTests)
    set(IILD_COREML_TEST_FIXTURE "" CACHE PATH "Optional converted Core ML model/oracle fixture directory")
    if(IILD_ENABLE_COREML AND IILD_COREML_TEST_FIXTURE)
        add_test(NAME CoreMLFixtureTests COMMAND CoreMLTests --fixture "${IILD_COREML_TEST_FIXTURE}")
    endif()
    if(IILD_BUILD_TOOLS)
        add_test(
            NAME ComputeCliTests
            COMMAND "${CMAKE_COMMAND}" "-DCLI=$<TARGET_FILE:iild-run>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/ComputeCliTests.cmake"
        )
        add_test(
            NAME CliContractTests
            COMMAND
                "${CMAKE_COMMAND}"
                "-DCLI=$<TARGET_FILE:iild-run>"
                "-DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sd-v1-manifest"
                "-DSDXL_FIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sdxl-base-manifest"
                "-DFLUX_FIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/flux1-schnell-manifest"
                "-DWORK_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/cli-test-work"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/CliContractTests.cmake"
        )
    endif()

    add_test(
        NAME InstallConsumerTests
        COMMAND
            "${CMAKE_COMMAND}"
            "-DBUILD_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}"
            "-DWORK_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/install-consumer-test"
            "-DGENERATOR=${CMAKE_GENERATOR}"
            "-DCONFIG=$<CONFIG>"
            "-DEXECUTABLE_SUFFIX=${CMAKE_EXECUTABLE_SUFFIX}"
            "-DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sd-v1-manifest"
            "-DSDXL_FIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sdxl-base-manifest"
            "-DFLUX_FIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/flux1-schnell-manifest"
            "-DMLX_BACKEND=${iild_gpu_backend}"
            "-DLIBTORCH_ENABLED=${IILD_ENABLE_LIBTORCH}"
            "-DCOREML_FIXTURE=${IILD_COREML_TEST_FIXTURE}"
            "-DBUILD_TOOLS=${IILD_BUILD_TOOLS}"
            "-DPYTHON_REFERENCE_ENABLED=${IILD_INSTALL_PYTHON_REFERENCE}"
            "-DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
            "-DBIN_DIRECTORY=${CMAKE_INSTALL_BINDIR}"
            "-DDOC_DIRECTORY=${CMAKE_INSTALL_DOCDIR}"
            "-DREFERENCE_DIRECTORY=${CMAKE_INSTALL_DATADIR}/iiLocalDiffusion/reference"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/InstallConsumerTests.cmake"
    )

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME ReferenceScriptsTests
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/ReferenceScriptsTests.py"
        )
        set_tests_properties(ReferenceScriptsTests PROPERTIES
            ENVIRONMENT "PYTHONPYCACHEPREFIX=${CMAKE_CURRENT_BINARY_DIR}/pycache"
        )

        add_test(
            NAME ModelLoadingTests
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/ModelLoadingTests.py"
        )
        set_tests_properties(ModelLoadingTests PROPERTIES
            ENVIRONMENT "PYTHONPYCACHEPREFIX=${CMAKE_CURRENT_BINARY_DIR}/pycache"
        )

        add_test(
            NAME HardwarePolicyTests
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/HardwarePolicyTests.py"
        )
        set_tests_properties(HardwarePolicyTests PROPERTIES
            ENVIRONMENT "PYTHONPYCACHEPREFIX=${CMAKE_CURRENT_BINARY_DIR}/pycache"
        )

        add_test(
            NAME TensorCorePolicyTests
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/TensorCorePolicyTests.py"
        )
        set_tests_properties(TensorCorePolicyTests PROPERTIES
            ENVIRONMENT "PYTHONPYCACHEPREFIX=${CMAKE_CURRENT_BINARY_DIR}/pycache"
        )
        add_test(NAME RocmPolicyTests
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/RocmPolicyTests.py")
        set_tests_properties(RocmPolicyTests PROPERTIES
            ENVIRONMENT "PYTHONPYCACHEPREFIX=${CMAKE_CURRENT_BINARY_DIR}/pycache")
        add_test(NAME ReferenceHardwareHelp
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/reference/diffusers/hardware.py" --help)

        add_test(
            NAME ResourcePolicyTests
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/ResourcePolicyTests.py"
        )
        set_tests_properties(ResourcePolicyTests PROPERTIES
            ENVIRONMENT "PYTHONPYCACHEPREFIX=${CMAKE_CURRENT_BINARY_DIR}/pycache"
        )

        foreach(generation_test IN ITEMS
                GenerationOptionsTests GenerationOutputTests
                GenerationSchedulerTests GenerationTensorInputsTests EncoderCompatibilityTests
                ControlNetTests HiresOptionsTests HiresRuntimeTests HiresRepeatRuntimeTests HiresFluxControlNetTests
                TextEmbeddingOptionsTests TextEmbeddingRuntimeTests)
            add_test(NAME ${generation_test}
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/${generation_test}.py")
            set_tests_properties(${generation_test} PROPERTIES
                ENVIRONMENT "PYTHONPYCACHEPREFIX=${CMAKE_CURRENT_BINARY_DIR}/pycache")
        endforeach()
        foreach(model_compatibility_test IN ITEMS
                CivitaiCatalogTests ModelFamilyTests GenericDiffusersTests
                ComfyUIRuntimeTests GenerationRouterTests RuntimeCompatibilityTests
                DownloadedModelTests CheckpointConversionTests ComfyUIImageWorkflowTests
                LocalImageTests SetupComfyUITests)
            add_test(NAME ${model_compatibility_test}
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/${model_compatibility_test}.py")
            set_tests_properties(${model_compatibility_test} PROPERTIES
                ENVIRONMENT "PYTHONPYCACHEPREFIX=${CMAKE_CURRENT_BINARY_DIR}/pycache")
        endforeach()
        foreach(generation_preset IN ITEMS sd15 sdxl-base flux1-schnell sdxl illustrious
                noobai noobai-v-pred flux1-dev flux1-krea-dev sd15-compatible flux1-schnell-compatible)
            add_test(NAME ReferenceGenerateConfig-${generation_preset}
                COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/reference/diffusers/generate.py"
                    --preset ${generation_preset} --print-config)
            set_tests_properties(ReferenceGenerateConfig-${generation_preset} PROPERTIES
                ENVIRONMENT "PYTHONPYCACHEPREFIX=${CMAKE_CURRENT_BINARY_DIR}/pycache")
        endforeach()

        add_test(
            NAME MlxValidationTests
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/MlxValidationTests.py"
        )
        set_tests_properties(MlxValidationTests PROPERTIES
            ENVIRONMENT "PYTHONPYCACHEPREFIX=${CMAKE_CURRENT_BINARY_DIR}/pycache"
        )

        add_test(NAME CoreMLConversionTests
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/CoreMLConversionTests.py")
        set_tests_properties(CoreMLConversionTests PROPERTIES
            ENVIRONMENT "PYTHONPYCACHEPREFIX=${CMAKE_CURRENT_BINARY_DIR}/pycache")

        add_test(
            NAME ReferenceGenerateHelp
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/reference/diffusers/generate.py" --help
        )
        add_test(
            NAME ReferenceInspectHelp
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/reference/diffusers/inspect_pipeline.py" --help
        )
    endif()
endif()

install(
    TARGETS iiLocalDiffusion
    EXPORT iiLocalDiffusionTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(IILD_BUILD_TOOLS)
    install(TARGETS iild-run RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(FILES README.md DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(DIRECTORY docs DESTINATION ${CMAKE_INSTALL_DOCDIR} FILES_MATCHING PATTERN "*.md")

if(IILD_INSTALL_PYTHON_REFERENCE)
    set(iild_reference_directory "${CMAKE_INSTALL_DATADIR}/iiLocalDiffusion/reference")
    file(RELATIVE_PATH IILD_REFERENCE_FROM_BINDIR
        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
        "${CMAKE_INSTALL_PREFIX}/${iild_reference_directory}")
    configure_file(cmake/iild-generate.py.in "${CMAKE_CURRENT_BINARY_DIR}/iild-generate" @ONLY)
    install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/iild-generate" DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES reference/generate.py reference/setup_comfyui.py DESTINATION ${iild_reference_directory})
    install(DIRECTORY reference/diffusers DESTINATION ${iild_reference_directory}
        FILES_MATCHING PATTERN "*.py" PATTERN "*.json" PATTERN "requirements*.txt" PATTERN "README.md"
        PATTERN ".venv" EXCLUDE PATTERN "__pycache__" EXCLUDE)
endif()

install(
    FILES
        src/Export.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    FILES
        src/Compute/ComputeRuntime.hpp
        src/Compute/CoreMLModel.hpp
        src/Compute/LinearLayer.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Compute
)

install(
    FILES
        src/Flux/FluxModelManifest.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Flux
)

install(
    FILES
        src/ModelManifest/DiffusionModelManifest.hpp
        src/ModelManifest/ModelManifest.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ModelManifest
)

install(
    FILES
        src/StableDiffusion/StableDiffusionModelManifest.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/StableDiffusion
)

configure_package_config_file(
    cmake/iiLocalDiffusionConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/iiLocalDiffusionConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/iiLocalDiffusion
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/iiLocalDiffusionConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(
    EXPORT iiLocalDiffusionTargets
    FILE iiLocalDiffusionTargets.cmake
    NAMESPACE iiLocalDiffusion::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/iiLocalDiffusion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/iiLocalDiffusionConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/iiLocalDiffusionConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/iiLocalDiffusion
)
