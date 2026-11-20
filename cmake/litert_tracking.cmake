option(SIGHT_ENABLE_RTMW "Register the verified optional native RTMW GPU filter" OFF)
set(SIGHT_LITERT_LICENSE "" CACHE FILEPATH "License from matching local LiteRT wheel")
set(SIGHT_LITERT_SDK_DIR "" CACHE PATH "LiteRT 2.3.0 C API SDK headers (optional)")
set(SIGHT_LITERT_RUNTIME_DIR "" CACHE PATH "Local matching LiteRT runtime and WebGPU library")
set(SIGHT_RTMW_MODEL "" CACHE FILEPATH "Local RTMW-m model for native regression")
set(SIGHT_RTMW_TEST_INPUT "" CACHE FILEPATH "Float32 NCHW 1x3x256x192 real test crop")
if(SIGHT_LITERT_SDK_DIR AND SIGHT_LITERT_RUNTIME_DIR)
    set(_sdk "${SIGHT_LITERT_SDK_DIR}")
    set(_runtime "${SIGHT_LITERT_RUNTIME_DIR}")
    if(NOT EXISTS "${_sdk}/litert/c/litert_compiled_model.h" OR
       NOT EXISTS "${_runtime}/libLiteRt.so" OR
       NOT EXISTS "${_runtime}/libLiteRtWebGpuAccelerator.so")
        message(FATAL_ERROR "Incomplete LiteRT tracking SDK/runtime")
    endif()
    configure_file("${_sdk}/litert/build_common/build_config.h.in"
        "${CMAKE_BINARY_DIR}/sight-litert-include/litert/build_common/build_config.h")
    add_library(sight_litert_runtime STATIC src/core/litert_runtime.cpp)
    set_target_properties(sight_litert_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(sight_litert_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src"
        PRIVATE "${_sdk}" "${CMAKE_BINARY_DIR}/sight-litert-include")
    target_link_libraries(sight_litert_runtime PUBLIC "${_runtime}/libLiteRt.so" dl)
    if(SIGHT_ENABLE_RTMW)
        file(SHA256 "${_runtime}/libLiteRt.so" _lite_hash)
        file(SHA256 "${_runtime}/libLiteRtWebGpuAccelerator.so" _gpu_hash)
        if(NOT _lite_hash STREQUAL "eec47eb35e2924b3166015887e652a58edc9af823ae38e02bf83bef2548300bb"
           OR NOT _gpu_hash STREQUAL "b67f193ae6eb8ad3a15349f54aba696c97d6f981c92bc2eb1096cbe9e2835b25"
           OR NOT EXISTS "${SIGHT_LITERT_LICENSE}")
            message(FATAL_ERROR "RTMW opt-in requires the qualified LiteRT 2.3.0 Linux wheel runtime and license")
        endif()
        target_sources(live-latent-lab-sight PRIVATE src/core/rtmw_tracker.cpp)
        target_link_libraries(live-latent-lab-sight PRIVATE sight_litert_runtime)
        target_compile_definitions(live-latent-lab-sight PRIVATE SIGHT_HAVE_RTMW)
        install(FILES "${_runtime}/libLiteRt.so" "${_runtime}/libLiteRtWebGpuAccelerator.so" DESTINATION bin/64bit)
        install(FILES "${SIGHT_LITERT_LICENSE}" DESTINATION data/licenses/litert RENAME LICENSE)
    endif()
    add_executable(sight_litert_persistent_test tests/litert_persistent_session_test.cpp)
    target_link_libraries(sight_litert_persistent_test PRIVATE sight_litert_runtime)
    add_executable(sight_rtmw_real_test tests/rtmw_tracker_real_test.cpp
        src/core/rtmw_tracker.cpp src/core/object_detector.cpp)
    target_include_directories(sight_rtmw_real_test PRIVATE src
        external/mediapipe-tasks-legacy/include external/mediapipe-tasks-123/include)
    target_link_libraries(sight_rtmw_real_test PRIVATE sight_litert_runtime dl)
    set(SIGHT_RTMW_PERSON_DETECTOR "" CACHE FILEPATH "Local person detection TFLite model")
    if(SIGHT_RTMW_MODEL AND SIGHT_RTMW_PERSON_DETECTOR AND SIGHT_MEDIAPIPE_123_LIB AND SIGHT_MEDIAPIPE_TEST_RGBA)
        add_test(NAME sight_rtmw_real_gpu COMMAND sight_rtmw_real_test
            "${SIGHT_RTMW_MODEL}" "${SIGHT_RTMW_PERSON_DETECTOR}"
            "${SIGHT_MEDIAPIPE_123_LIB}" "${SIGHT_MEDIAPIPE_TEST_RGBA}")
        set_tests_properties(sight_rtmw_real_gpu PROPERTIES TIMEOUT 45 RUN_SERIAL TRUE
            ENVIRONMENT "LD_LIBRARY_PATH=${_runtime}:$ENV{LD_LIBRARY_PATH}")
    endif()
    set(SIGHT_YOLOX_MODEL "" CACHE FILEPATH "Local YOLOX-Nano LiteRT model")
    if(SIGHT_RTMW_MODEL AND SIGHT_YOLOX_MODEL AND SIGHT_MEDIAPIPE_TEST_RGBA)
        add_test(NAME sight_rtmw_yolox_gpu COMMAND sight_rtmw_real_test
            "${SIGHT_RTMW_MODEL}" "${SIGHT_YOLOX_MODEL}" ""
            "${SIGHT_MEDIAPIPE_TEST_RGBA}" yolox)
        set_tests_properties(sight_rtmw_yolox_gpu PROPERTIES TIMEOUT 45 RUN_SERIAL TRUE
            ENVIRONMENT "LD_LIBRARY_PATH=${_runtime}:$ENV{LD_LIBRARY_PATH}")
    endif()
    if(SIGHT_RTMW_MODEL AND SIGHT_RTMW_TEST_INPUT)
        add_test(NAME sight_litert_persistent_gpu
            COMMAND sight_litert_persistent_test "${SIGHT_RTMW_MODEL}" "${SIGHT_RTMW_TEST_INPUT}")
        set_tests_properties(sight_litert_persistent_gpu PROPERTIES TIMEOUT 45 RUN_SERIAL TRUE
            ENVIRONMENT "LD_LIBRARY_PATH=${_runtime}:$ENV{LD_LIBRARY_PATH}")
    endif()
endif()

if(SIGHT_ENABLE_RTMW AND NOT TARGET sight_litert_runtime)
    message(FATAL_ERROR "RTMW requires explicit matching LiteRT SDK and runtime directories")
endif()
