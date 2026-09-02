cmake_minimum_required(VERSION 3.31)

project(
    iiLocalDiffusion
    VERSION 0.3.0
    DESCRIPTION "A local generative-model pipeline runtime"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

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

add_library(iiLocalDiffusion SHARED
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
)

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
        tools/iild-run/main.cpp
    )

    target_compile_features(iild-run PRIVATE cxx_std_20)
    target_compile_options(iild-run PRIVATE ${IILD_WARNING_OPTIONS})
    target_link_libraries(iild-run PRIVATE iiLocalDiffusion::iiLocalDiffusion)
endif()

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
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
    if(IILD_BUILD_TOOLS)
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
