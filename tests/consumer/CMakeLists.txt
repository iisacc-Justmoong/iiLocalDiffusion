cmake_minimum_required(VERSION 3.31)
project(iiLocalDiffusionInstalledPoseConsumer LANGUAGES CXX)
include(CTest)
find_package(iiLocalDiffusion 0.6 CONFIG REQUIRED)
find_package(Threads REQUIRED)
add_executable(iiLocalDiffusionInstalledPose ../NativePoseTests.cpp)
target_link_libraries(iiLocalDiffusionInstalledPose PRIVATE iiLocalDiffusion::iiLocalDiffusion Threads::Threads)
target_compile_features(iiLocalDiffusionInstalledPose PRIVATE cxx_std_23)
add_test(NAME iiLocalDiffusion.InstalledPose COMMAND iiLocalDiffusionInstalledPose
    "${CMAKE_CURRENT_BINARY_DIR}/pose-fixture")
if(APPLE)
    set_tests_properties(iiLocalDiffusion.InstalledPose PROPERTIES ENVIRONMENT_MODIFICATION
        "DYLD_LIBRARY_PATH=unset:;DYLD_FALLBACK_LIBRARY_PATH=unset:;DYLD_FRAMEWORK_PATH=unset:;DYLD_FALLBACK_FRAMEWORK_PATH=unset:")
endif()
