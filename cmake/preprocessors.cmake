# Native preprocessor suite, independent of experimental diffusion and private SDKs.
option(SIGHT_ENABLE_PREPROCESSORS "Enable native image and neural preprocessors when OpenCV is available" ON)
if(NOT SIGHT_ENABLE_PREPROCESSORS)
    return()
endif()
if(NOT SIGHT_HAVE_DENSE_FLOW)
    find_package(OpenCV QUIET COMPONENTS core imgproc)
endif()
if(NOT OpenCV_FOUND)
    message(STATUS "Sight Preprocessors omitted: OpenCV core/imgproc unavailable")
    return()
endif()
set(SIGHT_HAVE_PREPROCESSORS TRUE)
target_compile_definitions(live-latent-lab-sight PRIVATE SIGHT_HAVE_PREPROCESSORS)
target_sources(live-latent-lab-sight PRIVATE
    src/preprocessors-filter.cpp src/core/preprocessors_processor.cpp
    src/core/onnx_teed_processor.cpp
    src/core/onnx_pidinet_processor.cpp
    src/core/pidinet_scribble.cpp
    src/core/onnx_hed_processor.cpp)
target_include_directories(live-latent-lab-sight SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(live-latent-lab-sight PRIVATE ${OpenCV_LIBS})
add_executable(sight_preprocessors_unit_test
    tests/sight_preprocessors_unit_test.cpp
    src/core/preprocessors_processor.cpp)
target_include_directories(sight_preprocessors_unit_test PRIVATE src/core)
target_include_directories(sight_preprocessors_unit_test SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sight_preprocessors_unit_test PRIVATE ${OpenCV_LIBS})
target_compile_options(sight_preprocessors_unit_test PRIVATE -UNDEBUG)
add_test(NAME sight_preprocessors_unit COMMAND sight_preprocessors_unit_test)
# Opt-in Rust tensor preparation. OBS graphics and ONNX execution remain
# in C++. Cargo builds one dependency-free static archive and cannot reach
# the network. No runtime loader, Python, or second OBS plugin.
option(SIGHT_ENABLE_RUST_PREPROCESSOR_CORE
    "Use the Rust frame/tensor normalization core for native ONNX Depth" OFF)
if(SIGHT_ENABLE_RUST_PREPROCESSOR_CORE)
    find_program(SIGHT_CARGO NAMES cargo HINTS "$ENV{HOME}/.cargo/bin")
    if(NOT SIGHT_CARGO)
        message(FATAL_ERROR
            "Rust tensor core was requested but cargo is unavailable")
    endif()
    set(SIGHT_RUST_MANIFEST
        "${CMAKE_CURRENT_SOURCE_DIR}/rust/sight-frame-core/Cargo.toml")
    set(SIGHT_RUST_TARGET
        "${CMAKE_CURRENT_BINARY_DIR}/rust-frame-core-target")
    set(SIGHT_RUST_ARCHIVE
        "${SIGHT_RUST_TARGET}/release/libsight_frame_core.a")
    add_custom_command(OUTPUT "${SIGHT_RUST_ARCHIVE}"
        COMMAND "${CMAKE_COMMAND}" -E env
            "CARGO_TARGET_DIR=${SIGHT_RUST_TARGET}"
            "${SIGHT_CARGO}" build --offline --release
                --manifest-path "${SIGHT_RUST_MANIFEST}"
        DEPENDS "${SIGHT_RUST_MANIFEST}"
            "${CMAKE_CURRENT_SOURCE_DIR}/rust/sight-frame-core/src/lib.rs"
            "${CMAKE_CURRENT_SOURCE_DIR}/rust/sight-frame-core/Cargo.lock"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Building Sight Rust CPU tensor core (offline)"
        VERBATIM)
    add_custom_target(sight_rust_frame_core_build
        DEPENDS "${SIGHT_RUST_ARCHIVE}")
    add_library(sight_rust_frame_core STATIC IMPORTED GLOBAL)
    set_target_properties(sight_rust_frame_core PROPERTIES
        IMPORTED_LOCATION "${SIGHT_RUST_ARCHIVE}")
    add_dependencies(sight_rust_frame_core sight_rust_frame_core_build)
    target_link_libraries(live-latent-lab-sight PRIVATE
        sight_rust_frame_core dl pthread m)
    target_compile_definitions(live-latent-lab-sight PRIVATE
        SIGHT_HAVE_RUST_FRAME_CORE=1)

    add_executable(sight_rust_frame_core_abi_test
        tests/sight_rust_frame_core_abi_test.cpp)
    target_include_directories(sight_rust_frame_core_abi_test PRIVATE src)
    target_link_libraries(sight_rust_frame_core_abi_test PRIVATE
        sight_rust_frame_core dl pthread m)
    target_compile_options(sight_rust_frame_core_abi_test PRIVATE -UNDEBUG)
    add_test(NAME sight_rust_frame_core_abi
        COMMAND sight_rust_frame_core_abi_test)
endif()
if(SIGHT_PYTHON3_EXECUTABLE)
    add_test(NAME sight_native_interface_contract
        COMMAND "${SIGHT_PYTHON3_EXECUTABLE}" -m unittest discover -s tests -p test_native_interface_contract.py)
    set_tests_properties(sight_native_interface_contract PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 15)
endif()
foreach(smoke_target IN ITEMS sight_obs_effect_render_smoke sight_privacy_text_obs_smoke)
    if(TARGET ${smoke_target})
        target_compile_definitions(${smoke_target} PRIVATE SIGHT_SMOKE_EXPECT_PREPROCESSORS)
    endif()
endforeach()
if(SIGHT_XVFB_RUN)
    add_executable(sight_preprocessors_obs_smoke
        tests/sight_preprocessors_obs_smoke.cpp)
    target_include_directories(sight_preprocessors_obs_smoke PRIVATE
        "${SIGHT_OBS_INCLUDE}")
    target_link_libraries(sight_preprocessors_obs_smoke PRIVATE
        "${SIGHT_OBS_LIBRARY}" pthread)
    add_test(NAME sight_preprocessors_real_obs
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_preprocessors_obs_smoke>"
            "$<TARGET_FILE:live-latent-lab-sight>"
            "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_preprocessors_real_obs PROPERTIES TIMEOUT 50)
    add_test(NAME sight_preprocessors_missing_model_obs
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_preprocessors_obs_smoke>"
            "$<TARGET_FILE:live-latent-lab-sight>"
            "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_preprocessors_missing_model_obs PROPERTIES
        TIMEOUT 40 ENVIRONMENT "SIGHT_PREPROCESSORS_BAD_MODEL_CASE=1")
    add_test(NAME sight_preprocessors_teed_missing_model_obs
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_preprocessors_obs_smoke>"
            "$<TARGET_FILE:live-latent-lab-sight>"
            "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_preprocessors_teed_missing_model_obs PROPERTIES
        TIMEOUT 45
        ENVIRONMENT "SIGHT_PREPROCESSORS_TEED_BAD_MODEL_CASE=1")
    add_test(NAME sight_preprocessors_hed_missing_model_obs
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_preprocessors_obs_smoke>"
            "$<TARGET_FILE:live-latent-lab-sight>"
            "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_preprocessors_hed_missing_model_obs PROPERTIES
        TIMEOUT 50
        ENVIRONMENT "SIGHT_PREPROCESSORS_HED_BAD_MODEL_CASE=1")
    add_test(NAME sight_preprocessors_pidinet_missing_model_obs
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_preprocessors_obs_smoke>"
            "$<TARGET_FILE:live-latent-lab-sight>"
            "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_preprocessors_pidinet_missing_model_obs PROPERTIES
        TIMEOUT 40
        ENVIRONMENT "SIGHT_PREPROCESSORS_PIDINET_BAD_MODEL_CASE=1")
endif()
# Five-output HED actual native ONNX reference-hash model test.
add_executable(sight_hed_model_test
    tests/sight_hed_model_test.cpp src/core/onnx_hed_processor.cpp
    src/core/pidinet_scribble.cpp)
target_include_directories(sight_hed_model_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_ORT_INCLUDE}")
target_include_directories(sight_hed_model_test SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sight_hed_model_test PRIVATE
    "${SIGHT_ORT_LIBRARY}" ${OpenCV_LIBS})
set_target_properties(sight_hed_model_test PROPERTIES
    BUILD_RPATH "${SIGHT_ORT_LIB_DIR}")
target_compile_options(sight_hed_model_test PRIVATE -UNDEBUG)
if(SIGHT_ENABLE_RUST_PREPROCESSOR_CORE)
    target_link_libraries(sight_hed_model_test PRIVATE
        sight_rust_frame_core dl pthread m)
    target_compile_definitions(sight_hed_model_test PRIVATE
        SIGHT_HAVE_RUST_FRAME_CORE=1)
endif()
set(SIGHT_HED_MODEL "" CACHE FILEPATH
    "Optional local HED five-scale ONNX for actual model/OBS tests")
if(SIGHT_HED_MODEL)
    if(NOT EXISTS "${SIGHT_HED_MODEL}")
        message(FATAL_ERROR "HED local ONNX not found: ${SIGHT_HED_MODEL}")
    endif()
    add_test(NAME sight_hed_real_inference
        COMMAND sight_hed_model_test "${SIGHT_HED_MODEL}")
    set_tests_properties(sight_hed_real_inference PROPERTIES TIMEOUT 75)
    if(SIGHT_XVFB_RUN)
        add_test(NAME sight_preprocessors_hed_real_obs
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                "$<TARGET_FILE:sight_preprocessors_obs_smoke>"
                "$<TARGET_FILE:live-latent-lab-sight>"
                "${CMAKE_CURRENT_SOURCE_DIR}/data")
        set_tests_properties(sight_preprocessors_hed_real_obs PROPERTIES
            TIMEOUT 95
            ENVIRONMENT "SIGHT_PREPROCESSORS_HED_MODEL=${SIGHT_HED_MODEL}")
    endif()
endif()

add_executable(sight_pidinet_model_test
    tests/sight_pidinet_model_test.cpp src/core/onnx_pidinet_processor.cpp
    src/core/pidinet_scribble.cpp)
target_include_directories(sight_pidinet_model_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_ORT_INCLUDE}")
target_link_libraries(sight_pidinet_model_test PRIVATE "${SIGHT_ORT_LIBRARY}")
target_include_directories(sight_pidinet_model_test SYSTEM PRIVATE
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sight_pidinet_model_test PRIVATE ${OpenCV_LIBS})
set_target_properties(sight_pidinet_model_test PROPERTIES
    BUILD_RPATH "${SIGHT_ORT_LIB_DIR}")
if(SIGHT_ENABLE_RUST_PREPROCESSOR_CORE)
    target_link_libraries(sight_pidinet_model_test PRIVATE
        sight_rust_frame_core dl pthread m)
    target_compile_definitions(sight_pidinet_model_test PRIVATE
        SIGHT_HAVE_RUST_FRAME_CORE=1)
endif()

set(SIGHT_PIDINET_MODEL "" CACHE FILEPATH
    "Optional local PiDiNet native ONNX checkpoint for real model/OBS tests")
if(SIGHT_PIDINET_MODEL)
    if(NOT EXISTS "${SIGHT_PIDINET_MODEL}")
        message(FATAL_ERROR "PiDiNet ONNX not found: ${SIGHT_PIDINET_MODEL}")
    endif()
    if(SIGHT_XVFB_RUN)
        add_test(NAME sight_preprocessors_pidinet_real_obs
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                "$<TARGET_FILE:sight_preprocessors_obs_smoke>"
                "$<TARGET_FILE:live-latent-lab-sight>"
                "${CMAKE_CURRENT_SOURCE_DIR}/data")
        set_tests_properties(sight_preprocessors_pidinet_real_obs PROPERTIES
            TIMEOUT 90
            ENVIRONMENT "SIGHT_PREPROCESSORS_PIDINET_MODEL=${SIGHT_PIDINET_MODEL}")
        add_test(NAME sight_preprocessors_pidinet_model_switch_obs
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                "$<TARGET_FILE:sight_preprocessors_obs_smoke>"
                "$<TARGET_FILE:live-latent-lab-sight>"
                "${CMAKE_CURRENT_SOURCE_DIR}/data")
        set_tests_properties(sight_preprocessors_pidinet_model_switch_obs PROPERTIES
            TIMEOUT 90
            ENVIRONMENT
                "SIGHT_PREPROCESSORS_PIDINET_MODEL=${SIGHT_PIDINET_MODEL};SIGHT_PREPROCESSORS_PIDINET_BAD_MODEL_CASE=1")
    endif()
    add_test(NAME sight_pidinet_real_inference
        COMMAND sight_pidinet_model_test "${SIGHT_PIDINET_MODEL}")
    set_tests_properties(sight_pidinet_real_inference PROPERTIES TIMEOUT 55)
endif()

add_executable(sight_teed_model_test
    tests/sight_teed_model_test.cpp src/core/onnx_teed_processor.cpp)
target_include_directories(sight_teed_model_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_ORT_INCLUDE}")
target_link_libraries(sight_teed_model_test PRIVATE "${SIGHT_ORT_LIBRARY}")
set_target_properties(sight_teed_model_test PROPERTIES
    BUILD_RPATH "${SIGHT_ORT_LIB_DIR}")
if(SIGHT_ENABLE_RUST_PREPROCESSOR_CORE)
    target_link_libraries(sight_teed_model_test PRIVATE
        sight_rust_frame_core dl pthread m)
    target_compile_definitions(sight_teed_model_test PRIVATE
        SIGHT_HAVE_RUST_FRAME_CORE=1)
endif()

set(SIGHT_TEED_MODEL "" CACHE FILEPATH
    "Optional locally exported TEED ONNX checkpoint for native model/OBS tests")
if(SIGHT_TEED_MODEL)
    if(NOT EXISTS "${SIGHT_TEED_MODEL}")
        message(FATAL_ERROR "TEED model not found: ${SIGHT_TEED_MODEL}")
    endif()
    add_test(NAME sight_teed_real_inference
        COMMAND sight_teed_model_test "${SIGHT_TEED_MODEL}")
    set_tests_properties(sight_teed_real_inference PROPERTIES TIMEOUT 60)
    if(SIGHT_XVFB_RUN)
        add_test(NAME sight_preprocessors_teed_real_obs
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                "$<TARGET_FILE:sight_preprocessors_obs_smoke>"
                "$<TARGET_FILE:live-latent-lab-sight>"
                "${CMAKE_CURRENT_SOURCE_DIR}/data")
        set_tests_properties(sight_preprocessors_teed_real_obs PROPERTIES
            TIMEOUT 95
            ENVIRONMENT "SIGHT_PREPROCESSORS_TEED_MODEL=${SIGHT_TEED_MODEL}")
    endif()
endif()

if(SIGHT_DEPTH_MODEL AND SIGHT_XVFB_RUN)
        add_test(NAME sight_preprocessors_depth_real_obs
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                "$<TARGET_FILE:sight_preprocessors_obs_smoke>"
                "$<TARGET_FILE:live-latent-lab-sight>"
                "${CMAKE_CURRENT_SOURCE_DIR}/data")
        set_tests_properties(sight_preprocessors_depth_real_obs PROPERTIES
            TIMEOUT 95
            ENVIRONMENT "SIGHT_PREPROCESSORS_DEPTH_MODEL=${SIGHT_DEPTH_MODEL}")
endif()
