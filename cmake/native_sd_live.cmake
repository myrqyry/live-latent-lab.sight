option(SIGHT_ENABLE_NATIVE_SD_LIVE_FILTER "Expose experimental native SD1.x live diffusion filter" OFF)
if(SIGHT_ENABLE_NATIVE_SD_LIVE_FILTER)
    if(NOT SIGHT_ENABLE_NATIVE_SD_GENERATOR OR NOT TARGET stable-diffusion)
        message(FATAL_ERROR "Native live diffusion requires SIGHT_ENABLE_NATIVE_SD_GENERATOR")
    endif()
    target_sources(live-latent-lab-sight PRIVATE src/native-sd-live-filter.cpp)
    install(FILES data/sight-live-diffusion-motion.effect DESTINATION data)
    target_compile_definitions(live-latent-lab-sight PRIVATE SIGHT_HAVE_NATIVE_SD_LIVE_FILTER)
    foreach(_smoke_target IN ITEMS sight_obs_effect_render_smoke sight_privacy_text_obs_smoke)
        if(TARGET ${_smoke_target})
            target_compile_definitions(${_smoke_target} PRIVATE SIGHT_SMOKE_EXPECT_NATIVE_SD_LIVE)
        endif()
    endforeach()
    if(SD_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(live-latent-lab-sight PRIVATE CUDA::cudart)
        set_source_files_properties(src/native-sd-live-filter.cpp PROPERTIES COMPILE_DEFINITIONS SIGHT_SD_LIVE_CUDA_GUARD)
    endif()
    add_executable(sight_native_sd_live_obs_test tests/sight_native_sd_live_obs.cpp)
    target_include_directories(sight_native_sd_live_obs_test PRIVATE "${SIGHT_OBS_INCLUDE}")
    target_link_libraries(sight_native_sd_live_obs_test PRIVATE "${SIGHT_OBS_LIBRARY}" pthread)
    if(SIGHT_HAVE_DENSE_FLOW)
        target_compile_definitions(sight_native_sd_live_obs_test PRIVATE SIGHT_SD_TEST_DENSE_FLOW)
        add_executable(sight_native_sd_temporal_test tests/native_sd_temporal_test.cpp src/core/dense_motion.cpp)
        target_include_directories(sight_native_sd_temporal_test PRIVATE src)
        target_include_directories(sight_native_sd_temporal_test SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_compile_definitions(sight_native_sd_temporal_test PRIVATE SIGHT_HAVE_DENSE_FLOW)
        target_link_libraries(sight_native_sd_temporal_test PRIVATE opencv_core opencv_imgproc opencv_video)
        add_test(NAME sight_native_sd_temporal_unit COMMAND sight_native_sd_temporal_test)
    endif()
    if(SIGHT_SD_XVFB_RUN)
        add_test(NAME sight_native_sd_live_obs COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_native_sd_live_obs_test>" "$<TARGET_FILE:live-latent-lab-sight>" "${CMAKE_CURRENT_SOURCE_DIR}/data")
        set_tests_properties(sight_native_sd_live_obs PROPERTIES TIMEOUT 240 RUN_SERIAL TRUE)
        # Registration-only smoke must not be confused with actual inference.
        set(SIGHT_SD_TEST_MODEL "" CACHE FILEPATH "Optional local SD1.x LCM checkpoint for real CUDA acceptance")
        set(SIGHT_SD_TEST_TAE "" CACHE FILEPATH "Optional matching local TAESD for real CUDA acceptance")
        if(SIGHT_SD_TEST_MODEL OR SIGHT_SD_TEST_TAE)
            if(NOT SD_CUDA OR NOT EXISTS "${SIGHT_SD_TEST_MODEL}" OR NOT EXISTS "${SIGHT_SD_TEST_TAE}")
                message(FATAL_ERROR "Real native CUDA qualification requires SD_CUDA and both local model paths")
            endif()
            add_test(NAME sight_native_sd_live_cuda_real_obs
                COMMAND "${CMAKE_COMMAND}" -E env
                    "SIGHT_SD_LIVE_TEST_MODEL=${SIGHT_SD_TEST_MODEL}"
                    "SIGHT_SD_LIVE_TEST_TAE=${SIGHT_SD_TEST_TAE}"
                    "SIGHT_SD_LIVE_TEST_DEVICE=CUDA0" "SIGHT_SD_LIVE_TEST_MOTION=1" "SIGHT_SD_LIVE_TEST_TEMPORAL=1"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                    "$<TARGET_FILE:sight_native_sd_live_obs_test>" "$<TARGET_FILE:live-latent-lab-sight>" "${CMAKE_CURRENT_SOURCE_DIR}/data")
            set_tests_properties(sight_native_sd_live_cuda_real_obs PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE)
            add_test(NAME sight_native_sd_live_cuda_weight_switch_obs
                COMMAND "${CMAKE_COMMAND}" -E env
                    "SIGHT_SD_LIVE_TEST_MODEL=${SIGHT_SD_TEST_MODEL}"
                    "SIGHT_SD_LIVE_TEST_TAE=${SIGHT_SD_TEST_TAE}"
                    "SIGHT_SD_LIVE_TEST_DEVICE=CUDA0" "SIGHT_SD_LIVE_TEST_WEIGHT_SWITCH=1"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                    "$<TARGET_FILE:sight_native_sd_live_obs_test>" "$<TARGET_FILE:live-latent-lab-sight>" "${CMAKE_CURRENT_SOURCE_DIR}/data")
            set_tests_properties(sight_native_sd_live_cuda_weight_switch_obs PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE)

            add_test(NAME sight_native_sd_recursive_cuda_real_obs
                COMMAND "${CMAKE_COMMAND}" -E env
                    "SIGHT_SD_FEEDBACK_MODEL=${SIGHT_SD_TEST_MODEL}"
                    "SIGHT_SD_FEEDBACK_TAE=${SIGHT_SD_TEST_TAE}"
                    "SIGHT_SD_FEEDBACK_DEVICE=CUDA0" "SIGHT_SD_FEEDBACK_FRAMES=8"
                    "SIGHT_SD_FEEDBACK_FINITE=1" "SIGHT_SD_FEEDBACK_EXPOSURE=1" "SIGHT_SD_FEEDBACK_PREVIEW_FLOW=1"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                    "$<TARGET_FILE:sight_native_sd_source_obs_test>" "$<TARGET_FILE:live-latent-lab-sight>" "${CMAKE_CURRENT_SOURCE_DIR}/data")
            set_tests_properties(sight_native_sd_recursive_cuda_real_obs PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE)
        endif()

    endif()
endif()
