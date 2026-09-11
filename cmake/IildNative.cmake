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
        # Also apply with FETCHCONTENT_SOURCE_DIR_IILD_SDCPP (PATCH_COMMAND is
        # skipped for that override). Refuse drift instead of guessing a patch.
        find_package(Git REQUIRED)
        foreach(patch_name native-progress-cancellation native-mapped-upload native-tensor-wakeup native-metal-shared-upload native-conversion-cancellation native-thread-safe-logging)
            set(native_patch "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/${patch_name}.patch")
            set(native_source "${iild_sdcpp_SOURCE_DIR}")
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
            set_source_files_properties("${native_source}/src/model_loader.cpp"
                TARGET_DIRECTORY stable-diffusion PROPERTIES COMPILE_DEFINITIONS IILD_NATIVE_METAL_SHARED_UPLOAD)
        endif()
        install(FILES "${iild_sdcpp_SOURCE_DIR}/LICENSE"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/iiLocalDiffusion/licenses" RENAME stable-diffusion.cpp-LICENSE)
        install(FILES "${iild_sdcpp_SOURCE_DIR}/ggml/LICENSE"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/iiLocalDiffusion/licenses" RENAME ggml-LICENSE)
    endfunction()
    iild_add_native_backend()
    install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../ThirdParty/NativeDiffusion/"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/iiLocalDiffusion/licenses")
endif()
