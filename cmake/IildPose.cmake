# Native DWPose is separately selectable while product resource/UI wiring is
# staged. No network/model download occurs at application runtime.
option(IILD_ENABLE_POSE "Build in-memory ONNX DWPose preprocessing" OFF)
set(IILD_ONNXRUNTIME_ROOT "" CACHE PATH "Existing native ONNX Runtime distribution")
if(NOT IILD_ENABLE_POSE)
    return()
endif()
if(NOT IILD_ONNXRUNTIME_ROOT)
    if(APPLE AND NOT IOS AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        set(_pose_archive "onnxruntime-osx-arm64-1.30.0.tgz")
        set(_pose_hash "6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012")
    else()
        message(FATAL_ERROR "Set IILD_ONNXRUNTIME_ROOT to a target-native ONNX Runtime 1.30 distribution")
    endif()
    include(FetchContent)
    FetchContent_Declare(iild_onnxruntime
        URL "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/${_pose_archive}"
        URL_HASH "SHA256=${_pose_hash}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(iild_onnxruntime)
    set(_pose_root "${iild_onnxruntime_SOURCE_DIR}")
else()
    set(_pose_root "${IILD_ONNXRUNTIME_ROOT}")
endif()
find_path(IILD_POSE_ORT_INCLUDE onnxruntime_cxx_api.h PATHS "${_pose_root}/include" NO_DEFAULT_PATH REQUIRED)
find_library(IILD_POSE_ORT_LIBRARY NAMES onnxruntime PATHS "${_pose_root}/lib" NO_DEFAULT_PATH REQUIRED)
add_library(iild_pose_ort SHARED IMPORTED)
if(WIN32)
    set_target_properties(iild_pose_ort PROPERTIES IMPORTED_IMPLIB "${IILD_POSE_ORT_LIBRARY}"
        IMPORTED_LOCATION "${_pose_root}/lib/onnxruntime.dll")
else()
    set_target_properties(iild_pose_ort PROPERTIES IMPORTED_LOCATION "${IILD_POSE_ORT_LIBRARY}")
endif()
set_target_properties(iild_pose_ort PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${IILD_POSE_ORT_INCLUDE}")
install(DIRECTORY "${_pose_root}/lib/" DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT Runtime
    FILES_MATCHING PATTERN "*.dylib" PATTERN "*.so*" PATTERN "*.dll")
install(FILES "${_pose_root}/LICENSE" "${_pose_root}/ThirdPartyNotices.txt"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/iiLocalDiffusion/licenses/onnxruntime" COMPONENT Runtime)
