# Copy the installed runtime assets beside desktop executables or inside Apple
# bundles. Android consumers copy these files from assets into private storage
# and pass that directory in NativeGenerationOptions::resourceDirectory.
function(iiLocalDiffusion_deploy_generation_resources target)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Unknown iiLocalDiffusion consumer target: ${target}")
    endif()
    if(NOT EXISTS "${iiLocalDiffusion_GENERATION_RESOURCES}/generation-defaults.json")
        message(FATAL_ERROR "The iiLocalDiffusion generation resources are not installed")
    endif()
    if(ANDROID)
        message(FATAL_ERROR "Copy iiLocalDiffusion_GENERATION_RESOURCES into Android private storage and set NativeGenerationOptions::resourceDirectory")
    endif()
    get_target_property(is_bundle "${target}" MACOSX_BUNDLE)
    if(APPLE AND is_bundle)
        if(IOS)
            set(destination "$<TARGET_BUNDLE_DIR:${target}>/iiLocalDiffusion/resources")
        else()
            set(destination "$<TARGET_BUNDLE_DIR:${target}>/Contents/Resources/iiLocalDiffusion/resources")
        endif()
    else()
        set(destination "$<TARGET_FILE_DIR:${target}>/iiLocalDiffusion/resources")
    endif()
    add_custom_command(TARGET "${target}" POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${iiLocalDiffusion_GENERATION_RESOURCES}" "${destination}"
        VERBATIM)
endfunction()
