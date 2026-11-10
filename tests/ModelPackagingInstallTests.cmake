if(NOT DEFINED BUILD_DIRECTORY OR NOT DEFINED STAGE_DIRECTORY OR NOT DEFINED BIN_DIRECTORY)
    message(FATAL_ERROR "Build, staging, and binary directories are required")
endif()

if(EXISTS "${STAGE_DIRECTORY}" OR IS_SYMLINK "${STAGE_DIRECTORY}")
    message(FATAL_ERROR "Packaging stage already exists; refusing to replace it")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIRECTORY}"
        --prefix "${STAGE_DIRECTORY}" --component ModelPackaging
    RESULT_VARIABLE install_result OUTPUT_VARIABLE install_output ERROR_VARIABLE install_error)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "ModelPackaging installation failed: ${install_output}\n${install_error}")
endif()
if(NOT EXISTS "${STAGE_DIRECTORY}/share/iiLocalDiffusion/python/iild_model_packaging.py")
    message(FATAL_ERROR "The ModelPackaging Python API was not installed")
endif()

# An argument error must reach the installed CLI after the loader resolves all
# dependencies; do not supply a developer-machine dynamic-library search path.
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        --unset=DYLD_LIBRARY_PATH --unset=DYLD_FALLBACK_LIBRARY_PATH
        --unset=DYLD_FRAMEWORK_PATH --unset=DYLD_FALLBACK_FRAMEWORK_PATH
        --unset=DYLD_VERSIONED_LIBRARY_PATH --unset=DYLD_VERSIONED_FRAMEWORK_PATH
        --unset=DYLD_INSERT_LIBRARIES --unset=DYLD_ROOT_PATH
        "${STAGE_DIRECTORY}/${BIN_DIRECTORY}/iild-model-package"
    RESULT_VARIABLE run_result OUTPUT_VARIABLE run_output ERROR_VARIABLE run_error)
if(NOT run_result EQUAL 2 OR NOT run_output MATCHES "iild-model-package-report-v1")
    message(FATAL_ERROR "Installed ModelPackaging CLI failed: ${run_result}\n${run_output}\n${run_error}")
endif()
