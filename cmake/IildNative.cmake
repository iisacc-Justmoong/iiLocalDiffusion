option(IILD_ENABLE_NATIVE_DIFFUSION "Enable in-process checkpoint image generation" ${IOS})
if(IILD_ENABLE_NATIVE_DIFFUSION)
    enable_language(C)
    if(APPLE)
        enable_language(OBJC OBJCXX)
    endif()
    include(FetchContent)
    # Keep the C API and ggml version reproducible. Do not fetch the web UI or codecs.
    FetchContent_Declare(iild_sdcpp
        GIT_REPOSITORY https://github.com/leejet/stable-diffusion.cpp.git
        GIT_TAG d04e8950c1ec8d30248cbe996682b3182fb1adf6
        GIT_SUBMODULES ggml
        GIT_SUBMODULES_RECURSE FALSE)
    function(iild_add_native_backend)
        set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(SD_BUILD_SHARED_GGML_LIB OFF CACHE BOOL "" FORCE)
        set(SD_WEBP OFF CACHE BOOL "" FORCE)
        set(SD_WEBM OFF CACHE BOOL "" FORCE)
        set(SD_METAL ${APPLE} CACHE BOOL "" FORCE)
        set(SD_RPC OFF CACHE BOOL "" FORCE)
        set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
        set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
        set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
        set(GGML_METAL_USE_BF16 OFF CACHE BOOL "" FORCE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        FetchContent_MakeAvailable(iild_sdcpp)
        # Match the pinned backend's public ggml_tensor layout in its consumers.
        target_compile_definitions(stable-diffusion INTERFACE GGML_MAX_NAME=160)
        # Also apply with FETCHCONTENT_SOURCE_DIR_IILD_SDCPP (PATCH_COMMAND is
        # skipped for that override). Refuse drift instead of guessing a patch.
        find_package(Git REQUIRED)
        foreach(patch_name native-progress-cancellation native-mapped-upload native-tensor-wakeup native-metal-shared-upload native-conversion-cancellation native-thread-safe-logging native-model-family native-mapped-buffer-compatibility native-inplace-backend-compatibility native-metal-storage-ops native-vae-fallback native-cpu-flash-attention native-vae-decode-safety native-krea2-embedded-encoder native-metal-fp8-capability native-backend-telemetry native-bulk-weight-read native-resident-model-memory native-resident-progress native-runtime-residency native-reference-capacity native-pixel-upscalers native-text-conditioning native-controlnet-fail-closed native-controlnet-hires-hint native-canny-preprocessing native-resident-alignment native-freeu native-learned-upscaler native-resident-detailer native-sdxl-refiner native-refiner-sampling native-controlnet-region-mask native-multi-controlnet native-ip-adapter-execution native-multi-ip-attention native-ip-adapter-resources)
            set(native_patch "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/${patch_name}.patch")
            set(native_source "${iild_sdcpp_SOURCE_DIR}")
            if(patch_name MATCHES "^(native-progress-cancellation|native-learned-upscaler|native-ip-adapter-execution|native-controlnet-region-mask|native-multi-ip-attention|native-multi-controlnet|native-refiner-sampling|native-model-family|native-vae-fallback|native-controlnet-hires-hint)$")
                # The resource follow-up refactors legacy IP preparation and overlaps
                # older sampler/loader hunks. Require its exact applied state.
                execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check
                    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/native-ip-adapter-resources.patch"
                    WORKING_DIRECTORY "${native_source}" RESULT_VARIABLE native_ip_resources_applied
                    OUTPUT_QUIET ERROR_QUIET)
                if(native_ip_resources_applied EQUAL 0)
                    continue()
                endif()
            endif()
            if(patch_name MATCHES "^(native-controlnet-region-mask|native-ip-adapter-execution)$")
                # Multi-IP attention reuses mask coverage and extends IP binding validation.
                # Accept superseded hunks only when the exact follow-up reverses cleanly.
                execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check
                    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/native-multi-ip-attention.patch"
                    WORKING_DIRECTORY "${native_source}" RESULT_VARIABLE native_multi_ip_applied
                    OUTPUT_QUIET ERROR_QUIET)
                if(native_multi_ip_applied EQUAL 0)
                    continue()
                endif()
            endif()
            if(patch_name MATCHES "^(native-model-family|native-vae-fallback|native-refiner-sampling|native-controlnet-region-mask|native-controlnet-hires-hint)$")
                # The exact multi-control follow-up supersedes context in these
                # owned patches. Require that follow-up to reverse cleanly.
                execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check
                    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/native-multi-controlnet.patch"
                    WORKING_DIRECTORY "${native_source}" RESULT_VARIABLE native_multi_applied
                    OUTPUT_QUIET ERROR_QUIET)
                if(native_multi_applied EQUAL 0)
                    continue()
                endif()
            endif()
            if(patch_name STREQUAL "native-model-family" OR patch_name STREQUAL "native-vae-fallback")
                # Refiner adds a distinct family while sharing the SDXL VAE.
                # Validate the exact later patch before accepting these older,
                # overlapping reverse checks on an already patched build tree.
                execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check
                    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/native-refiner-sampling.patch"
                    WORKING_DIRECTORY "${native_source}" RESULT_VARIABLE native_refiner_applied
                    OUTPUT_QUIET ERROR_QUIET)
                if(native_refiner_applied EQUAL 0)
                    continue()
                endif()
            endif()
            if(patch_name STREQUAL "native-runtime-residency")
                # The alignment follow-up replaces two allocation hunks in this
                # patch. Validate that exact follow-up before accepting the
                # superseded reverse-check, including on repeat configuration.
                execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check
                    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/native-resident-alignment.patch"
                    WORKING_DIRECTORY "${native_source}" RESULT_VARIABLE native_alignment_applied
                    OUTPUT_QUIET ERROR_QUIET)
                if(native_alignment_applied EQUAL 0)
                    continue()
                endif()
            endif()
            if(patch_name STREQUAL "native-resident-progress")
                file(READ "${native_source}/src/core/util.cpp" native_runtime_loader)
                string(FIND "${native_runtime_loader}" "resident_source_runtime()" native_runtime_applied)
                if(NOT native_runtime_applied EQUAL -1)
                    continue()
                endif()
            endif()
            if(patch_name STREQUAL "native-resident-model-memory")
                file(READ "${native_source}/include/stable-diffusion.h" native_resident_api)
                file(READ "${native_source}/src/core/util.cpp" native_resident_loader)
                file(READ "${native_source}/src/model_loader.cpp" native_resident_files)
                string(FIND "${native_resident_api}" "memory_resident_model" native_resident_api_applied)
                string(FIND "${native_resident_loader}" "ResidentMemoryMmapWrapper" native_resident_loader_applied)
                string(FIND "${native_resident_files}" "Failed to create an in-memory model tensor buffer." native_resident_files_applied)
                if(NOT native_resident_api_applied EQUAL -1 AND NOT native_resident_loader_applied EQUAL -1
                    AND NOT native_resident_files_applied EQUAL -1)
                    continue()
                endif()
            endif()
            # Bulk reads supersede the shared-upload memcpy body. Validate the
            # final patch before checking that older overlapping patch, so both
            # existing build trees and fresh checkouts remain reconfigurable.
            if(patch_name STREQUAL "native-mapped-upload")
                file(READ "${native_source}/src/core/util.cpp" native_bulk_util)
                string(FIND "${native_bulk_util}" "Large bounded reads avoid one filesystem operation" native_bulk_source_applied)
                if(NOT native_bulk_source_applied EQUAL -1)
                    continue()
                endif()
                execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check
                    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/native-bulk-weight-read.patch"
                    WORKING_DIRECTORY "${native_source}" RESULT_VARIABLE native_bulk_applied
                    OUTPUT_QUIET ERROR_QUIET)
                if(native_bulk_applied EQUAL 0)
                    continue()
                endif()
            endif()
            if(patch_name STREQUAL "native-bulk-weight-read")
                file(READ "${native_source}/include/stable-diffusion.h" native_resident_api)
                file(READ "${native_source}/src/core/util.cpp" native_bulk_util)
                string(FIND "${native_resident_api}" "memory_resident_model" native_resident_api_applied)
                string(FIND "${native_bulk_util}" "Large bounded reads avoid one filesystem operation" native_bulk_source_applied)
                if(NOT native_resident_api_applied EQUAL -1 AND NOT native_bulk_source_applied EQUAL -1)
                    continue()
                endif()
            endif()
            execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${native_patch}"
                WORKING_DIRECTORY "${native_source}" RESULT_VARIABLE native_patched
                OUTPUT_QUIET ERROR_QUIET)
            if(NOT native_patched EQUAL 0)
                execute_process(COMMAND "${GIT_EXECUTABLE}" apply --check "${native_patch}"
                    WORKING_DIRECTORY "${native_source}" COMMAND_ERROR_IS_FATAL ANY)
                execute_process(COMMAND "${GIT_EXECUTABLE}" apply "${native_patch}"
                    WORKING_DIRECTORY "${native_source}" COMMAND_ERROR_IS_FATAL ANY)
            endif()
        endforeach()
        if(APPLE)
            set_source_files_properties("${native_source}/src/model_loader.cpp" "${native_source}/src/model_manager.cpp"
                TARGET_DIRECTORY stable-diffusion PROPERTIES COMPILE_DEFINITIONS IILD_NATIVE_METAL_SHARED_UPLOAD)
        endif()
        install(FILES "${iild_sdcpp_SOURCE_DIR}/LICENSE"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/iiLocalDiffusion/licenses" RENAME stable-diffusion.cpp-LICENSE)
        install(FILES "${iild_sdcpp_SOURCE_DIR}/ggml/LICENSE"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/iiLocalDiffusion/licenses" RENAME ggml-LICENSE)
    endfunction()
    iild_add_native_backend()
    # The pinned backend compiles ggml_tensor with a 160-byte name field.
    # Consumers of its internal headers must use the same struct layout.
    target_compile_definitions(stable-diffusion INTERFACE GGML_MAX_NAME=160)
    install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../ThirdParty/NativeDiffusion/"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/iiLocalDiffusion/licenses")
endif()
