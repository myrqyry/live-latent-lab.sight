# Opt-in native prompt/image generation inside the existing Sight module.
# Build a single current stable-diffusion.cpp engine without enabling the
# unrelated, unverified legacy effects or requiring the LiteRT SDK.
option(SIGHT_ENABLE_NATIVE_SD_GENERATOR
    "Expose an experimental local stable-diffusion.cpp image source" OFF)
if(SIGHT_ENABLE_NATIVE_SD_GENERATOR)
    if(NOT SIGHT_HAVE_DENSE_FLOW)
        message(FATAL_ERROR "Native diffusion requires OpenCV core/imgproc/video (SIGHT_ENABLE_DENSE_FLOW=ON)")
    endif()
    set(SIGHT_NATIVE_SD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libs/stable-diffusion.cpp" CACHE PATH
        "Local checkout of the pinned stable-diffusion.cpp source")
    if(NOT EXISTS "${SIGHT_NATIVE_SD_ROOT}/include/stable-diffusion.h" OR
       NOT EXISTS "${SIGHT_NATIVE_SD_ROOT}/CMakeLists.txt" OR
       NOT EXISTS "${SIGHT_NATIVE_SD_ROOT}/ggml/CMakeLists.txt")
        message(FATAL_ERROR "Current stable-diffusion.cpp submodules are required for native Sight image generation")
    endif()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    option(SD_CUDA "Enable stable-diffusion.cpp CUDA backend" OFF)
    set(SIGHT_NATIVE_SD_PREBUILT_LIBRARY "" CACHE FILEPATH
        "Local experimental SD.cpp shared build from the pinned checkout (optional)")
    if(SIGHT_NATIVE_SD_PREBUILT_LIBRARY)
        if(NOT IS_ABSOLUTE "${SIGHT_NATIVE_SD_PREBUILT_LIBRARY}" OR
           NOT EXISTS "${SIGHT_NATIVE_SD_PREBUILT_LIBRARY}")
            message(FATAL_ERROR "SIGHT_NATIVE_SD_PREBUILT_LIBRARY must be an existing absolute library path")
        endif()
        # Developer-local reuse of the exact pinned API; normal builds
        # still compile the vendored source. Keep the runtime beside OBS.
        add_library(stable-diffusion SHARED IMPORTED)
        set_target_properties(stable-diffusion PROPERTIES
            IMPORTED_LOCATION "${SIGHT_NATIVE_SD_PREBUILT_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${SIGHT_NATIVE_SD_ROOT}/include")
        install(FILES "${SIGHT_NATIVE_SD_PREBUILT_LIBRARY}" DESTINATION bin/64bit RENAME libstable-diffusion.so)
    else()
        add_subdirectory("${SIGHT_NATIVE_SD_ROOT}"
            "${CMAKE_CURRENT_BINARY_DIR}/sight-sd-native" EXCLUDE_FROM_ALL)
    endif()
    target_sources(live-latent-lab-sight PRIVATE src/image-generator-source.cpp)
    if(NOT SIGHT_HAVE_PREPROCESSORS)
        target_sources(live-latent-lab-sight PRIVATE src/core/preprocessors_processor.cpp)
    endif()
    install(FILES data/sight-generator-preview.effect DESTINATION data)
    target_include_directories(live-latent-lab-sight PRIVATE "${SIGHT_NATIVE_SD_ROOT}/include")
    target_link_libraries(live-latent-lab-sight PRIVATE stable-diffusion)
    target_compile_definitions(live-latent-lab-sight PRIVATE SIGHT_HAVE_NATIVE_SD_IMAGE_SOURCE)
    if(SD_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(live-latent-lab-sight PRIVATE CUDA::cudart)
        set_source_files_properties(src/image-generator-source.cpp PROPERTIES COMPILE_DEFINITIONS SIGHT_SD_GENERATOR_CUDA_GUARD)
    endif()
    add_executable(sight_native_sd_feedback_test tests/native_sd_feedback_test.cpp src/core/preprocessors_processor.cpp)
    target_include_directories(sight_native_sd_feedback_test PRIVATE src)
    target_include_directories(sight_native_sd_feedback_test SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(sight_native_sd_feedback_test PRIVATE ${OpenCV_LIBS})
    add_test(NAME sight_native_sd_feedback_unit COMMAND sight_native_sd_feedback_test)
    add_executable(sight_native_sd_exposure_test tests/native_sd_exposure_test.cpp)
    target_include_directories(sight_native_sd_exposure_test PRIVATE src)
    add_test(NAME sight_native_sd_exposure_unit COMMAND sight_native_sd_exposure_test)
    add_executable(sight_native_sd_schedule_test tests/native_sd_schedule_test.cpp)
    target_include_directories(sight_native_sd_schedule_test PRIVATE src)
    add_test(NAME sight_native_sd_schedule_unit COMMAND sight_native_sd_schedule_test)
    add_executable(sight_native_sd_preview_test tests/native_sd_preview_test.cpp)
    target_include_directories(sight_native_sd_preview_test PRIVATE src)
    add_test(NAME sight_native_sd_preview_unit COMMAND sight_native_sd_preview_test)
    if(SIGHT_HAVE_DENSE_FLOW)
        add_executable(sight_native_sd_preview_flow_test tests/native_sd_preview_flow_test.cpp src/core/dense_motion.cpp)
        target_include_directories(sight_native_sd_preview_flow_test PRIVATE src)
        target_include_directories(sight_native_sd_preview_flow_test SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_compile_definitions(sight_native_sd_preview_flow_test PRIVATE SIGHT_HAVE_DENSE_FLOW)
        target_link_libraries(sight_native_sd_preview_flow_test PRIVATE opencv_core opencv_imgproc opencv_video)
        add_test(NAME sight_native_sd_preview_flow_unit COMMAND sight_native_sd_preview_flow_test)
    endif()

    # The opt-in static image engine has independent upstream licenses. Keep
    # notices with a staged package, not merely in source control.
    install(FILES "${SIGHT_NATIVE_SD_ROOT}/LICENSE"
        DESTINATION data/licenses/stable-diffusion.cpp)
    install(FILES "${SIGHT_NATIVE_SD_ROOT}/ggml/LICENSE"
        DESTINATION data/licenses/ggml)
    install(FILES "${SIGHT_NATIVE_SD_ROOT}/thirdparty/utf8proc/LICENSE.md"
        DESTINATION data/licenses/utf8proc)
    install(FILES "${SIGHT_NATIVE_SD_ROOT}/thirdparty/oniguruma/COPYING"
        DESTINATION data/licenses/oniguruma)

    add_executable(sight_native_sd_adapter_test tests/test_stable_diffusion_backend.cpp)
    target_include_directories(sight_native_sd_adapter_test PRIVATE src "${SIGHT_NATIVE_SD_ROOT}/include")
    target_compile_options(sight_native_sd_adapter_test PRIVATE -UNDEBUG)
    target_link_libraries(sight_native_sd_adapter_test PRIVATE stable-diffusion pthread)
    add_test(NAME sight_native_sd_adapter COMMAND sight_native_sd_adapter_test)

    add_executable(sight_native_sd_options_test tests/test_sd_modern_options.cpp)
    target_include_directories(sight_native_sd_options_test PRIVATE "${SIGHT_NATIVE_SD_ROOT}/include")
    target_compile_options(sight_native_sd_options_test PRIVATE -UNDEBUG)
    target_link_libraries(sight_native_sd_options_test PRIVATE stable-diffusion)
    add_test(NAME sight_native_sd_options COMMAND sight_native_sd_options_test)

    find_program(SIGHT_SD_XVFB_RUN NAMES xvfb-run)
    if(SIGHT_SD_XVFB_RUN)
        add_executable(sight_native_sd_preview_shader_test tests/native_sd_preview_shader_test.cpp)
        target_include_directories(sight_native_sd_preview_shader_test PRIVATE src "${SIGHT_OBS_INCLUDE}")
        target_link_libraries(sight_native_sd_preview_shader_test PRIVATE "${SIGHT_OBS_LIBRARY}" pthread)
        add_test(NAME sight_native_sd_preview_shader COMMAND "${SIGHT_SD_XVFB_RUN}" -a
            "$<TARGET_FILE:sight_native_sd_preview_shader_test>" "${CMAKE_CURRENT_SOURCE_DIR}/data/sight-generator-preview.effect")
        set_tests_properties(sight_native_sd_preview_shader PROPERTIES TIMEOUT 30)
        add_executable(sight_native_sd_source_obs_test tests/sight_native_sd_source_obs.cpp)
        target_include_directories(sight_native_sd_source_obs_test PRIVATE "${SIGHT_OBS_INCLUDE}")
        target_link_libraries(sight_native_sd_source_obs_test PRIVATE "${SIGHT_OBS_LIBRARY}" pthread)
        add_test(NAME sight_native_sd_source_obs
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                "$<TARGET_FILE:sight_native_sd_source_obs_test>"
                "$<TARGET_FILE:live-latent-lab-sight>"
                "${CMAKE_CURRENT_SOURCE_DIR}/data")
        set_tests_properties(sight_native_sd_source_obs PROPERTIES TIMEOUT 40 RUN_SERIAL TRUE)
    endif()
endif()
