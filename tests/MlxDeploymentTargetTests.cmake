include("${CMAKE_CURRENT_LIST_DIR}/../cmake/IildMlxDeployment.cmake")

function(expect_version predefines expected)
    iild_macos_target_from_predefines(actual "${predefines}")
    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR "Expected '${expected}', got '${actual}' from '${predefines}'")
    endif()
endfunction()

expect_version("#define __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ 270000\n" "27.0.0")
expect_version("#define __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ 140401\n" "14.4.1")
expect_version("#define __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ 101507\n" "10.15.7")
expect_version("#define SOMETHING_ELSE 270001\n" "")

# Explicit deployment targets must be honored without invoking the compiler.
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.4")
set(CMAKE_CXX_COMPILER "/nonexistent/compiler")
iild_mlx_deployment_target(actual)
if(NOT actual STREQUAL "14.4")
    message(FATAL_ERROR "Explicit macOS deployment target changed: ${actual}")
endif()

# On macOS, exercise the host compiler's preprocessor without launching an app.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND IILD_TEST_COMPILER)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "")
    set(CMAKE_CXX_COMPILER "${IILD_TEST_COMPILER}")
    set(CMAKE_OSX_SYSROOT "${IILD_TEST_SYSROOT}")
    iild_mlx_deployment_target(actual)
    if(NOT actual MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
        message(FATAL_ERROR "Invalid compiler deployment target: ${actual}")
    endif()
    message(STATUS "Compiler's macOS deployment target: ${actual}")
endif()
