if(NOT DEFINED CLI OR NOT DEFINED FIXTURE OR NOT DEFINED SDXL_FIXTURE
   OR NOT DEFINED FLUX_FIXTURE OR NOT DEFINED WORK_DIRECTORY)
    message(FATAL_ERROR "CLI, FIXTURE, SDXL_FIXTURE, FLUX_FIXTURE, and WORK_DIRECTORY are required")
endif()

function(assert_process name expected_result expected_output expected_error)
    execute_process(
        COMMAND "${CLI}" ${ARGN}
        RESULT_VARIABLE actual_result
        OUTPUT_VARIABLE actual_output
        ERROR_VARIABLE actual_error
    )

    if(NOT "${actual_result}" STREQUAL "${expected_result}")
        message(FATAL_ERROR
            "${name}: expected exit ${expected_result}, got ${actual_result}\n"
            "stdout=[${actual_output}]\nstderr=[${actual_error}]")
    endif()
    if(NOT "${actual_output}" STREQUAL "${expected_output}")
        message(FATAL_ERROR
            "${name}: stdout mismatch\nexpected=[${expected_output}]\nactual=[${actual_output}]")
    endif()
    if(NOT "${actual_error}" STREQUAL "${expected_error}")
        message(FATAL_ERROR
            "${name}: stderr mismatch\nexpected=[${expected_error}]\nactual=[${actual_error}]")
    endif()
endfunction()

file(REAL_PATH "${FIXTURE}" canonical_fixture)
string(CONCAT success_output
    "iiLocalDiffusion Model Inspector\n\n"
    "Model root: ${canonical_fixture}\n"
    "Format: Diffusers\n"
    "Pipeline: StableDiffusionPipeline\n"
    "Compatibility: StableDiffusionV1\n\n"
    "Tokenizer metadata: valid\n"
    "Text encoder metadata: valid\n"
    "UNet metadata: valid\n"
    "VAE metadata: valid\n"
    "Scheduler metadata: valid\n\n"
    "Weight files: present; contents not inspected\n"
    "Result: valid-metadata\n")

assert_process("valid inspect" 0 "${success_output}" "" inspect "${FIXTURE}")

file(REAL_PATH "${SDXL_FIXTURE}" canonical_sdxl_fixture)
string(CONCAT sdxl_success_output
    "iiLocalDiffusion Model Inspector\n\n"
    "Model root: ${canonical_sdxl_fixture}\n"
    "Format: Diffusers\n"
    "Pipeline: StableDiffusionXLPipeline\n"
    "Compatibility: StableDiffusionXLBase\n\n"
    "Tokenizer metadata: valid\n"
    "Text encoder metadata: valid\n"
    "Tokenizer 2 metadata: valid\n"
    "Text encoder 2 metadata: valid\n"
    "UNet metadata: valid\n"
    "VAE metadata: valid\n"
    "Scheduler metadata: valid\n\n"
    "Weight files: present; contents not inspected\n"
    "Result: valid-metadata\n")
assert_process("valid SDXL inspect" 0 "${sdxl_success_output}" "" inspect "${SDXL_FIXTURE}")

file(REAL_PATH "${FLUX_FIXTURE}" canonical_flux_fixture)
string(CONCAT flux_success_output
    "iiLocalDiffusion Model Inspector\n\n"
    "Model root: ${canonical_flux_fixture}\n"
    "Format: Diffusers\n"
    "Pipeline: FluxPipeline\n"
    "Compatibility: Flux1Schnell\n\n"
    "Tokenizer metadata: valid\n"
    "Text encoder metadata: valid\n"
    "Tokenizer 2 metadata: valid\n"
    "Text encoder 2 metadata: valid\n"
    "Transformer metadata: valid\n"
    "VAE metadata: valid\n"
    "Scheduler metadata: valid\n\n"
    "Weight files: present; contents not inspected\n"
    "Result: valid-metadata\n")
assert_process("valid FLUX inspect" 0 "${flux_success_output}" "" inspect "${FLUX_FIXTURE}")
string(CONCAT usage_output
    "Usage: iild-run inspect <model-root>\n"
    "       iild-run devices\n"
    "       iild-run compute [--device auto|metal|cuda|rocm|cpu] [--device-index N]\n"
    "                        [--cpu-share FRACTION] [--weight-storage device|ram]\n"
    "                        [--gpu-weight-mib N] [--precision fp32|fp16|bf16]\n"
    "       iild-run neural-compute --model PATH.mlmodelc [--compute-units cpu-ne|all|cpu]\n"
    "                               [--allow-cpu-plan] [--iterations N]\n")
assert_process("help" 0 "${usage_output}" "" --help)
assert_process("missing arguments" 2 "" "${usage_output}")

file(REMOVE_RECURSE "${WORK_DIRECTORY}")
file(MAKE_DIRECTORY "${WORK_DIRECTORY}/invalid")
set(missing_root "${WORK_DIRECTORY}/missing")
assert_process(
    "missing root"
    3
    ""
    "error[input-unavailable]: model root is unavailable or is not a directory: ${missing_root}\n"
    inspect "${missing_root}")

assert_process(
    "invalid package"
    4
    ""
    "error[invalid-package]: model_index.json is missing or is not a regular file: ${WORK_DIRECTORY}/invalid/model_index.json\n"
    inspect "${WORK_DIRECTORY}/invalid")

file(MAKE_DIRECTORY "${WORK_DIRECTORY}/unsupported")
file(WRITE "${WORK_DIRECTORY}/unsupported/model_index.json"
    "{\"_class_name\":\"UnknownPipeline\"}\n")
assert_process(
    "unsupported model"
    5
    ""
    "error[unsupported-model]: unsupported model_index.json _class_name: 'UnknownPipeline'\n"
    inspect "${WORK_DIRECTORY}/unsupported")

file(REMOVE_RECURSE "${WORK_DIRECTORY}")
