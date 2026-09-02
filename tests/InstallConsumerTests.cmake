if(NOT DEFINED BUILD_DIRECTORY OR NOT DEFINED WORK_DIRECTORY OR NOT DEFINED GENERATOR
   OR NOT DEFINED CONFIG OR NOT DEFINED EXECUTABLE_SUFFIX OR NOT DEFINED FIXTURE
   OR NOT DEFINED SDXL_FIXTURE OR NOT DEFINED FLUX_FIXTURE)
    message(FATAL_ERROR
        "BUILD_DIRECTORY, WORK_DIRECTORY, GENERATOR, CONFIG, EXECUTABLE_SUFFIX, FIXTURE, SDXL_FIXTURE, and FLUX_FIXTURE are required")
endif()

set(stage_directory "${WORK_DIRECTORY}/stage")
set(source_directory "${WORK_DIRECTORY}/consumer")
set(binary_directory "${WORK_DIRECTORY}/consumer-build")

file(REMOVE_RECURSE "${WORK_DIRECTORY}")
file(MAKE_DIRECTORY "${source_directory}")

set(configuration_arguments)
if(NOT CONFIG STREQUAL "")
    list(APPEND configuration_arguments --config "${CONFIG}")
endif()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" --install "${BUILD_DIRECTORY}"
        --prefix "${stage_directory}" ${configuration_arguments}
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_output
    ERROR_VARIABLE install_error
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR
        "install failed\nstdout=[${install_output}]\nstderr=[${install_error}]")
endif()

file(WRITE "${source_directory}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.31)
project(iiLocalDiffusionConsumer LANGUAGES CXX)
find_package(iiLocalDiffusion 0.3 REQUIRED CONFIG)
add_executable(consumer main.cpp)
target_compile_features(consumer PRIVATE cxx_std_20)
target_link_libraries(consumer PRIVATE iiLocalDiffusion::iiLocalDiffusion)
set_target_properties(consumer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin"
)
]=])

file(WRITE "${source_directory}/main.cpp" [=[
#include <Flux/FluxModelManifest.hpp>
#include <ModelManifest/DiffusionModelManifest.hpp>
#include <StableDiffusion/StableDiffusionModelManifest.hpp>

#include <iostream>
#include <type_traits>
#include <variant>

int main(const int argc, const char *const argv[])
{
    if (argc != 2)
    {
        return 2;
    }
    const auto manifest = iild::loadModelManifest(argv[1]);
    const bool specializedApiIsUsable = std::visit(
        [](const auto &value) {
            using Manifest = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Manifest, iild::StableDiffusionModelManifest>)
            {
                const auto compatibility = value.compatibilityKind();
                return (compatibility == iild::StableDiffusionCompatibility::v1 ||
                        compatibility == iild::StableDiffusionCompatibility::xlBase) &&
                    !value.unet().artifacts.empty();
            }
            else
            {
                return value.compatibilityKind() == iild::FluxCompatibility::v1Schnell &&
                    !value.tokenizer2().artifacts.empty() &&
                    !value.textEncoder2().artifacts.empty() &&
                    !value.transformer().artifacts.empty();
            }
        },
        manifest);
    if (!specializedApiIsUsable)
    {
        return 3;
    }
    std::visit(
        [](const auto &value) {
            std::cout << value.compatibility() << ':' << value.pipelineClass() << '\n';
        },
        manifest);
    return 0;
}
]=])

execute_process(
    COMMAND
        "${CMAKE_COMMAND}"
        -S "${source_directory}"
        -B "${binary_directory}"
        -G "${GENERATOR}"
        "-DCMAKE_PREFIX_PATH=${stage_directory}"
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_error
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR
        "consumer configure failed\nstdout=[${configure_output}]\nstderr=[${configure_error}]")
endif()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" --build "${binary_directory}"
        --parallel ${configuration_arguments}
    RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_output
    ERROR_VARIABLE build_error
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR
        "consumer build failed\nstdout=[${build_output}]\nstderr=[${build_error}]")
endif()

execute_process(
    COMMAND "${binary_directory}/bin/consumer${EXECUTABLE_SUFFIX}" "${FIXTURE}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
)
if(NOT run_result EQUAL 0
   OR NOT run_output STREQUAL "StableDiffusionV1:StableDiffusionPipeline\n"
   OR NOT run_error STREQUAL "")
    message(FATAL_ERROR
        "consumer run failed\nexit=[${run_result}]\nstdout=[${run_output}]\nstderr=[${run_error}]")
endif()

execute_process(
    COMMAND "${binary_directory}/bin/consumer${EXECUTABLE_SUFFIX}" "${SDXL_FIXTURE}"
    RESULT_VARIABLE sdxl_run_result
    OUTPUT_VARIABLE sdxl_run_output
    ERROR_VARIABLE sdxl_run_error
)
if(NOT sdxl_run_result EQUAL 0
   OR NOT sdxl_run_output STREQUAL "StableDiffusionXLBase:StableDiffusionXLPipeline\n"
   OR NOT sdxl_run_error STREQUAL "")
    message(FATAL_ERROR
        "SDXL consumer run failed\nexit=[${sdxl_run_result}]\nstdout=[${sdxl_run_output}]\nstderr=[${sdxl_run_error}]")
endif()

execute_process(
    COMMAND "${binary_directory}/bin/consumer${EXECUTABLE_SUFFIX}" "${FLUX_FIXTURE}"
    RESULT_VARIABLE flux_run_result
    OUTPUT_VARIABLE flux_run_output
    ERROR_VARIABLE flux_run_error
)
if(NOT flux_run_result EQUAL 0
   OR NOT flux_run_output STREQUAL "Flux1Schnell:FluxPipeline\n"
   OR NOT flux_run_error STREQUAL "")
    message(FATAL_ERROR
        "FLUX consumer run failed\nexit=[${flux_run_result}]\nstdout=[${flux_run_output}]\nstderr=[${flux_run_error}]")
endif()

file(REMOVE_RECURSE "${WORK_DIRECTORY}")
