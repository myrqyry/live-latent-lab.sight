# Experimental local CUDA-worker integration, deliberately excluded by default.
option(SIGHT_ENABLE_NEURAL_RENDERING "Enable experimental Small31 neural rendering filter" OFF)
if(SIGHT_ENABLE_NEURAL_RENDERING)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "Sight neural rendering worker transport requires Linux")
    endif()
    target_sources(live-latent-lab-sight PRIVATE src/neural-rendering-filter.cpp)
    install(FILES data/sight-neural-rendering.effect DESTINATION data)
    install(FILES
        data/neural-rendering/LICENSE-OpenDLSS-NR.txt
        data/neural-rendering/UPSTREAM-NOTICE.txt
        data/neural-rendering/fused.cu
        data/neural-rendering/fused.py
        data/neural-rendering/input_conditioning.py
        data/neural-rendering/primitives.cuh
        data/neural-rendering/quantized.py
        data/neural-rendering/replay.py
        data/neural-rendering/still.py
        data/neural-rendering/worker.py
        DESTINATION data/neural-rendering)
    target_compile_definitions(live-latent-lab-sight PRIVATE SIGHT_HAVE_NEURAL_RENDERING)
    foreach(_smoke_target IN ITEMS sight_obs_effect_render_smoke sight_privacy_text_obs_smoke)
        if(TARGET ${_smoke_target})
            target_compile_definitions(${_smoke_target} PRIVATE SIGHT_SMOKE_EXPECT_NEURAL_RENDERING)
        endif()
    endforeach()
    if(SIGHT_HAVE_DENSE_FLOW)
        add_executable(sight_neural_render_motion_test tests/neural_render_motion_test.cpp src/core/dense_motion.cpp)
        target_include_directories(sight_neural_render_motion_test PRIVATE src)
        target_include_directories(sight_neural_render_motion_test SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_compile_definitions(sight_neural_render_motion_test PRIVATE SIGHT_HAVE_DENSE_FLOW)
        target_link_libraries(sight_neural_render_motion_test PRIVATE opencv_core opencv_imgproc opencv_video pthread)
        add_test(NAME sight_neural_render_motion_unit COMMAND sight_neural_render_motion_test)
    endif()
    add_executable(sight_neural_render_residual_test tests/neural_render_residual_test.cpp)
    target_include_directories(sight_neural_render_residual_test PRIVATE src)
    target_link_libraries(sight_neural_render_residual_test PRIVATE pthread)
    add_test(NAME sight_neural_render_residual_unit COMMAND sight_neural_render_residual_test)
    add_executable(sight_neural_render_depth_test tests/neural_render_depth_test.cpp)
    target_include_directories(sight_neural_render_depth_test PRIVATE src "${SIGHT_OBS_INCLUDE}")
    target_link_libraries(sight_neural_render_depth_test PRIVATE "${SIGHT_OBS_LIBRARY}")
    add_test(NAME sight_neural_render_depth_unit COMMAND sight_neural_render_depth_test)
    find_program(SIGHT_NR_XVFB_RUN NAMES xvfb-run)
    if(SIGHT_NR_XVFB_RUN)
        add_executable(sight_neural_upsample_gpu_test tests/sight_neural_upsample_gpu.cpp)
        target_include_directories(sight_neural_upsample_gpu_test PRIVATE "${SIGHT_OBS_INCLUDE}")
        target_link_libraries(sight_neural_upsample_gpu_test PRIVATE "${SIGHT_OBS_LIBRARY}" pthread)
        add_test(NAME sight_neural_upsample_gpu COMMAND "${SIGHT_NR_XVFB_RUN}" -a
            "$<TARGET_FILE:sight_neural_upsample_gpu_test>" "${CMAKE_CURRENT_SOURCE_DIR}/data/sight-neural-rendering.effect")
        set_tests_properties(sight_neural_upsample_gpu PROPERTIES TIMEOUT 30)

        add_executable(sight_neural_rendering_obs_test tests/sight_neural_rendering_obs.cpp)
        target_include_directories(sight_neural_rendering_obs_test PRIVATE "${SIGHT_OBS_INCLUDE}")
        target_link_libraries(sight_neural_rendering_obs_test PRIVATE "${SIGHT_OBS_LIBRARY}" pthread)
        add_test(NAME sight_neural_rendering_registration
            COMMAND "${CMAKE_COMMAND}" -E env "SIGHT_NR_REGISTRATION_ONLY=1"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_neural_rendering_obs_test>"
            "$<TARGET_FILE:live-latent-lab-sight>" "${CMAKE_CURRENT_SOURCE_DIR}/data")
        set_tests_properties(sight_neural_rendering_registration PROPERTIES TIMEOUT 30)
        set(SIGHT_NR_TEST_CONFIG "" CACHE FILEPATH "Trusted local Small31 worker test configuration")
        if(SIGHT_NR_TEST_CONFIG)
            add_test(NAME sight_neural_rendering_obs
                COMMAND "${CMAKE_COMMAND}" -E env "SIGHT_NR_CONFIG=${SIGHT_NR_TEST_CONFIG}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                "$<TARGET_FILE:sight_neural_rendering_obs_test>"
                "$<TARGET_FILE:live-latent-lab-sight>" "${CMAKE_CURRENT_SOURCE_DIR}/data")
            set_tests_properties(sight_neural_rendering_obs PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)
            add_test(NAME sight_neural_rendering_custom_draw_obs
                COMMAND "${CMAKE_COMMAND}" -E env "SIGHT_NR_CONFIG=${SIGHT_NR_TEST_CONFIG}"
                "SIGHT_NR_CUSTOM_DRAW=1"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                "$<TARGET_FILE:sight_neural_rendering_obs_test>"
                "$<TARGET_FILE:live-latent-lab-sight>" "${CMAKE_CURRENT_SOURCE_DIR}/data")
            set_tests_properties(sight_neural_rendering_custom_draw_obs PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)
            # Exercise the packaged worker with the same relative data path
            # used by local staging tools and direct smoke invocations.
            if(SIGHT_DEPTH_MODEL)
                add_test(NAME sight_neural_rendering_combined_obs
                    COMMAND "${CMAKE_COMMAND}" -E env "SIGHT_NR_CONFIG=${SIGHT_NR_TEST_CONFIG}"
                    "SIGHT_NR_CUSTOM_DRAW=1" "SIGHT_NR_TEST_COMBINED=1"
                    "SIGHT_NR_TEST_DEPTH_MODEL=${SIGHT_DEPTH_MODEL}" "SIGHT_ONNX_DEVICE=cpu"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                    "$<TARGET_FILE:sight_neural_rendering_obs_test>"
                    "$<TARGET_FILE:live-latent-lab-sight>" data)
                set_tests_properties(sight_neural_rendering_combined_obs PROPERTIES
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 90 RUN_SERIAL TRUE)
            endif()


        endif()
    endif()
endif()
