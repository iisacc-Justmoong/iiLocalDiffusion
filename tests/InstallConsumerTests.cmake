if(NOT DEFINED BUILD_DIRECTORY OR NOT DEFINED WORK_DIRECTORY OR NOT DEFINED GENERATOR
   OR NOT DEFINED CONFIG OR NOT DEFINED EXECUTABLE_SUFFIX OR NOT DEFINED FIXTURE
   OR NOT DEFINED SDXL_FIXTURE OR NOT DEFINED FLUX_FIXTURE OR NOT DEFINED BUILD_TOOLS
   OR NOT DEFINED PYTHON_REFERENCE_ENABLED OR NOT DEFINED BIN_DIRECTORY
   OR NOT DEFINED DOC_DIRECTORY OR NOT DEFINED REFERENCE_DIRECTORY)
    message(FATAL_ERROR
        "Build, fixture, installation directory, and enabled component arguments are required")
endif()

set(stage_directory "${WORK_DIRECTORY}/stage")
set(source_directory "${WORK_DIRECTORY}/consumer")
set(binary_directory "${source_directory}/build")

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

# Run against a relocated prefix, including private runtime resources.
file(RENAME "${stage_directory}" "${WORK_DIRECTORY}/relocated stage")
set(stage_directory "${WORK_DIRECTORY}/relocated stage")
if(LIBTORCH_ENABLED AND
   (NOT EXISTS "${stage_directory}/share/licenses/iiLocalDiffusion/libtorch/LICENSE" OR
    NOT EXISTS "${stage_directory}/share/licenses/iiLocalDiffusion/safetensors-cpp/LICENSE"))
    message(FATAL_ERROR "The installed optional LibTorch backend is missing license notices")
endif()
if(MLX_BACKEND STREQUAL "metal" AND NOT EXISTS "${stage_directory}/lib/mlx.metallib")
    message(FATAL_ERROR "The installed Metal runtime is missing mlx.metallib")
endif()

foreach(document IN ITEMS README.md docs/installation.md docs/hires-fix.md docs/generation-parameters.md)
    if(NOT EXISTS "${stage_directory}/${DOC_DIRECTORY}/${document}")
        message(FATAL_ERROR "The installed package is missing documentation: ${document}")
    endif()
endforeach()

if(BUILD_TOOLS)
    execute_process(
        COMMAND "${stage_directory}/${BIN_DIRECTORY}/iild-run${EXECUTABLE_SUFFIX}" inspect "${FIXTURE}"
        WORKING_DIRECTORY "${source_directory}"
        RESULT_VARIABLE installed_cli_result
        OUTPUT_VARIABLE installed_cli_output
        ERROR_VARIABLE installed_cli_error
    )
    if(NOT installed_cli_result EQUAL 0)
        message(FATAL_ERROR
            "relocated CLI failed\nexit=[${installed_cli_result}]\nstdout=[${installed_cli_output}]\nstderr=[${installed_cli_error}]")
    endif()
endif()

if(PYTHON_REFERENCE_ENABLED)
    set(installed_launcher "${stage_directory}/${BIN_DIRECTORY}/iild-generate")
    set(installed_reference "${stage_directory}/${REFERENCE_DIRECTORY}")
    if(NOT EXISTS "${installed_launcher}")
        message(FATAL_ERROR "The installed generation launcher is missing")
    endif()
    foreach(resource IN ITEMS generate.py setup_comfyui.py diffusers/generate.py
            diffusers/civitai_catalog.json diffusers/requirements.txt diffusers/requirements-common.txt)
        if(NOT EXISTS "${installed_reference}/${resource}")
            message(FATAL_ERROR "The installed generation runtime is missing ${resource}")
        endif()
    endforeach()
    if(PYTHON_EXECUTABLE)
        foreach(expected_passes IN ITEMS 1 3)
            set(hires_arguments --hires-fix)
            if(expected_passes EQUAL 3)
                list(APPEND hires_arguments --hires-passes 3)
            endif()
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E env "IILD_PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
                    "${PYTHON_EXECUTABLE}" "${installed_launcher}"
                    --backend preset ${hires_arguments} --print-config
                WORKING_DIRECTORY "${source_directory}"
                RESULT_VARIABLE generation_result
                OUTPUT_VARIABLE generation_output
                ERROR_VARIABLE generation_error
            )
            if(NOT generation_result EQUAL 0)
                message(FATAL_ERROR
                    "relocated generator failed for ${expected_passes} HiRes passes\nexit=[${generation_result}]\nstdout=[${generation_output}]\nstderr=[${generation_error}]")
            endif()
            string(JSON pass_type TYPE "${generation_output}" hires_passes)
            string(JSON actual_passes GET "${generation_output}" hires_passes)
            if(NOT pass_type STREQUAL "NUMBER" OR NOT actual_passes STREQUAL "${expected_passes}")
                message(FATAL_ERROR
                    "The installed generator resolved ${actual_passes} (${pass_type}) instead of ${expected_passes} HiRes passes")
            endif()
        endforeach()

        execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" "${installed_launcher}" --list-base-models
            WORKING_DIRECTORY "${source_directory}"
            RESULT_VARIABLE catalog_result
            OUTPUT_VARIABLE catalog_output
            ERROR_VARIABLE catalog_error
        )
        if(NOT catalog_result EQUAL 0)
            message(FATAL_ERROR
                "relocated catalog command failed\nexit=[${catalog_result}]\nstdout=[${catalog_output}]\nstderr=[${catalog_error}]")
        endif()
        file(READ "${installed_reference}/diffusers/civitai_catalog.json" installed_catalog)
        string(JSON expected_models GET "${installed_catalog}" base_models)
        string(JSON actual_models GET "${catalog_output}" base_models)
        string(JSON model_count LENGTH "${actual_models}")
        if(model_count LESS 1 OR NOT actual_models STREQUAL expected_models)
            message(FATAL_ERROR "The installed generator did not return the installed Civitai catalog")
        endif()
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env
                "IILD_PYTHON_EXECUTABLE=${WORK_DIRECTORY}/missing python"
                "${PYTHON_EXECUTABLE}" "${installed_launcher}" --list-base-models
            WORKING_DIRECTORY "${source_directory}"
            RESULT_VARIABLE invalid_python_result
            OUTPUT_VARIABLE invalid_python_output
            ERROR_VARIABLE invalid_python_error
        )
        if(invalid_python_result EQUAL 0 OR
           NOT invalid_python_error MATCHES "IILD_PYTHON_EXECUTABLE is not an executable")
            message(FATAL_ERROR
                "The installed launcher did not reject an invalid interpreter\nexit=[${invalid_python_result}]\nstdout=[${invalid_python_output}]\nstderr=[${invalid_python_error}]")
        endif()
    else()
        message(STATUS "Python is unavailable; installed generation resources were checked, but launcher execution was not verified")
    endif()
endif()

file(WRITE "${source_directory}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.31)
project(iiLocalDiffusionConsumer LANGUAGES CXX)
find_package(iiLocalDiffusion 0.3 REQUIRED CONFIG)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    list(FIND CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES
        "$ENV{LIBRARY_PATH}" implicit_package_directory_index)
    if(implicit_package_directory_index EQUAL -1)
        message(FATAL_ERROR "The regression requires the package in an implicit linker directory")
    endif()
endif()
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
#include <Compute/LinearLayer.hpp>
#include <Compute/CoreMLModel.hpp>
#include <ModelManifest/DiffusionModelManifest.hpp>
#include <StableDiffusion/StableDiffusionModelManifest.hpp>

#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

int main(const int argc, const char *const argv[])
{
    if (argc != 2 && argc != 3)
    {
        return 2;
    }
    const auto capabilities = iild::computeCapabilities();
    const auto amd = iild::rocmCapabilities();
    if (amd.available != (amd.hip && amd.deviceCount > 0)) return 9;
    if (iild::computeDeviceName(iild::ComputeDevice::rocm) != "rocm") return 10;
    const auto neural = iild::coreMLCapabilities();
    if (neural.neuralEngine != (neural.neuralEngineCores > 0)) return 6;
    if (iild::classifyTensorCoreSupport("GeForce GTX 1650", 7, 5) != iild::AcceleratorSupport::unsupported)
        return 7;
    if (argc == 3 && neural.runtime)
    {
        auto component = iild::CoreMLModel::load(argv[2]);
        iild::CoreMLModel::Features values;
        for (const auto &feature : component.info().inputs)
            values.emplace(feature.name, std::vector<float>(feature.elementCount, 1));
        if (component.predict(values).empty() || component.info().neuralEnginePreferredOperations == 0)
            return 8;
    }
    if (capabilities.mlx || amd.libtorch)
    {
        iild::ComputeOptions options;
        if (!capabilities.metal && !capabilities.cuda && !amd.available)
        {
            options.device = iild::ComputeDevice::cpu;
        }
        auto layer = iild::LinearLayer::fromWeights(std::vector<float>{2.0F}, 1, 1, {}, options);
        if (layer.forward(std::vector<float>{3.0F}, 1) != std::vector<float>{6.0F})
        {
            return 4;
        }
        if (capabilities.metal || capabilities.cuda || amd.available)
        {
            auto hybrid = iild::LinearLayer::fromWeights(std::vector<float>{2, 4}, 1, 2, {},
                options, {0.5, iild::WeightStorage::ram, 4}, {iild::LinearPrecision::float16});
            if (hybrid.forward(std::vector<float>{3}, 1) != std::vector<float>{6, 12} ||
                hybrid.resourceInfo().cpuOutputFeatures != 1 ||
                hybrid.resourceInfo().stagedAcceleratorWeightBytes != 2 ||
                hybrid.precision() != iild::LinearPrecision::float16)
            {
                return 5;
            }
        }
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

# Exercise an ordinary imported target while its relocated library directory is
# also an implicit compiler search path. Consumers provide no RPATH workaround.
set(consumer_environment)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(consumer_environment "${CMAKE_COMMAND}" -E env
        --unset=DYLD_LIBRARY_PATH --unset=DYLD_FALLBACK_LIBRARY_PATH
        --unset=DYLD_FRAMEWORK_PATH --unset=DYLD_FALLBACK_FRAMEWORK_PATH
        --unset=DYLD_VERSIONED_LIBRARY_PATH --unset=DYLD_VERSIONED_FRAMEWORK_PATH
        "LIBRARY_PATH=${stage_directory}/lib")
endif()

execute_process(
    COMMAND ${consumer_environment}
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
    COMMAND ${consumer_environment}
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

set(coreml_arguments)
if(COREML_FIXTURE AND EXISTS "${COREML_FIXTURE}/model.mlmodelc")
    list(APPEND coreml_arguments "${COREML_FIXTURE}/model.mlmodelc")
endif()

execute_process(
    COMMAND ${consumer_environment} "${binary_directory}/bin/consumer${EXECUTABLE_SUFFIX}" "${FIXTURE}" ${coreml_arguments}
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
    COMMAND ${consumer_environment} "${binary_directory}/bin/consumer${EXECUTABLE_SUFFIX}" "${SDXL_FIXTURE}"
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
    COMMAND ${consumer_environment} "${binary_directory}/bin/consumer${EXECUTABLE_SUFFIX}" "${FLUX_FIXTURE}"
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
