# Keep fetched source trees in project resources and all compilation in build/.
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}/dependencies" CACHE PATH
    "Fetched dependency build metadata")
set(IILD_MLX_USER_SOURCE OFF)
if(FETCHCONTENT_SOURCE_DIR_MLX AND NOT FETCHCONTENT_SOURCE_DIR_MLX STREQUAL
        "${CMAKE_CURRENT_SOURCE_DIR}/resources/dependencies/mlx-src")
    set(IILD_MLX_USER_SOURCE ON)
endif()
foreach(_iild_dependency IN ITEMS mlx metal_cpp json fmt cccl nvtx3 cudnn cutlass iild_sdcpp iild_safetensors_cpp)
    string(TOUPPER "${_iild_dependency}" _iild_key)
    set(_iild_source "${CMAKE_CURRENT_SOURCE_DIR}/resources/dependencies/${_iild_dependency}-src")
    if(NOT FETCHCONTENT_SOURCE_DIR_${_iild_key} AND IS_DIRECTORY "${_iild_source}")
        set(FETCHCONTENT_SOURCE_DIR_${_iild_key} "${_iild_source}")
    endif()
endforeach()
