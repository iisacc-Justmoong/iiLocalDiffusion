include_guard(GLOBAL)

# Package only public model resources; local data and build-time dependencies
# must never be included just because they contain a matching file extension.
function(iild_install_resources source destination)
    install(DIRECTORY "${source}/" DESTINATION "${destination}"
        COMPONENT Runtime FILES_MATCHING
        PATTERN "*.safetensors" PATTERN "config.json" PATTERN "LICENSE*"
        PATTERN "README.md" PATTERN "NOTICE.md"
        PATTERN ".cache" EXCLUDE PATTERN ".state" EXCLUDE PATTERN ".recovery" EXCLUDE
        PATTERN "dependencies" EXCLUDE PATTERN "outputs" EXCLUDE
        PATTERN "test-workspaces" EXCLUDE PATTERN "__pycache__" EXCLUDE)
    # The existing Qwen tokenizer is public model data, not the excluded ComfyUI dependency.
    foreach(tokenizer IN ITEMS qwen3vl-4b qwen35-9b)
      foreach(name IN ITEMS vocab.json merges.txt tokenizer_config.json)
        if(EXISTS "${source}/text_encoders/${tokenizer}/${name}")
            install(FILES "${source}/text_encoders/${tokenizer}/${name}"
                DESTINATION "${destination}/text_encoders/${tokenizer}" COMPONENT Runtime)
        endif()
      endforeach()
    endforeach()
    foreach(name IN ITEMS vocab.json merges.txt tokenizer_config.json special_tokens_map.json)
        if(EXISTS "${source}/text_encoders/flux1-clip/${name}")
            install(FILES "${source}/text_encoders/flux1-clip/${name}"
                DESTINATION "${destination}/text_encoders/flux1-clip" COMPONENT Runtime)
        endif()
    endforeach()
    foreach(name IN ITEMS tokenizer.json tokenizer_config.json special_tokens_map.json)
        if(EXISTS "${source}/text_encoders/flux1-t5/${name}")
            install(FILES "${source}/text_encoders/flux1-t5/${name}"
                DESTINATION "${destination}/text_encoders/flux1-t5" COMPONENT Runtime)
        endif()
    endforeach()
endfunction()
