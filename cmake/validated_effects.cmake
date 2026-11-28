# Progressive activation: build one Sight plugin with its verified effects.
# Other effects remain in source control, unregistered and not compiled.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Validated progressive-effects build currently targets Linux")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Do not use libobs' pkg-config compile flags for C++: some Linux OBS
# packages export -std=gnu17 and -Werror, which break this C++20 plugin.
if(DEFINED ENV{SIGHT_OBS_SDK_ROOT} AND NOT "$ENV{SIGHT_OBS_SDK_ROOT}" STREQUAL "")
    set(_sight_sdk_include "$ENV{SIGHT_OBS_SDK_ROOT}/include/obs")
    set(_sight_sdk_lib "$ENV{SIGHT_OBS_SDK_ROOT}/lib" "$ENV{SIGHT_OBS_SDK_ROOT}/lib/x86_64-linux-gnu")
else()
    set(_sight_sdk_include /usr/local/include/obs /usr/include/obs)
    set(_sight_sdk_lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
endif()
find_path(SIGHT_OBS_INCLUDE obs-module.h REQUIRED PATHS ${_sight_sdk_include} NO_DEFAULT_PATH)
include("${CMAKE_CURRENT_LIST_DIR}/obs-api-version.cmake")
sight_read_obs_api_version("${SIGHT_OBS_INCLUDE}/obs-config.h" SIGHT_OBS_API_MAJOR SIGHT_OBS_API_MINOR)
# OBS refuses to load modules compiled against a newer libobs major/minor.
# Validated effects are re-verified against this exact API series; changing it
# requires a fresh full validation pass, not just a successful compile.
set(SIGHT_EXPECTED_LIBOBS_MAJOR 32 CACHE STRING
    "libobs API major version this validated build is certified against")
set(SIGHT_EXPECTED_LIBOBS_MINOR 0 CACHE STRING
    "libobs API minor version this validated build is certified against")
if(NOT SIGHT_OBS_API_MAJOR EQUAL SIGHT_EXPECTED_LIBOBS_MAJOR OR
   NOT SIGHT_OBS_API_MINOR GREATER_EQUAL SIGHT_EXPECTED_LIBOBS_MINOR)
    message(FATAL_ERROR
        "Header libobs API is ${SIGHT_OBS_API_MAJOR}.${SIGHT_OBS_API_MINOR}, expected "
        "${SIGHT_EXPECTED_LIBOBS_MAJOR}.${SIGHT_EXPECTED_LIBOBS_MINOR}+; "
        "re-validate effects before changing this target")
endif()
message(STATUS "Sight validated build certified against libobs API "
        "${SIGHT_OBS_API_MAJOR}.${SIGHT_OBS_API_MINOR} (patch differences load-safe)")
find_library(SIGHT_OBS_LIBRARY NAMES obs REQUIRED
    PATHS ${_sight_sdk_lib} NO_DEFAULT_PATH)

set(SIGHT_ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime" CACHE PATH
    "ONNX Runtime installation with include/ and lib/")
find_path(SIGHT_ORT_INCLUDE onnxruntime_cxx_api.h REQUIRED
    PATHS "${SIGHT_ORT_ROOT}/include" NO_DEFAULT_PATH)
find_library(SIGHT_ORT_LIBRARY
    NAMES onnxruntime libonnxruntime.so.1.17.1
    REQUIRED PATHS "${SIGHT_ORT_ROOT}/lib" NO_DEFAULT_PATH)
get_filename_component(SIGHT_ORT_LIB_DIR "${SIGHT_ORT_LIBRARY}" DIRECTORY)
# Explicit split SDKs can be used to test ABI compatibility locally, but
# a package must use headers, library and notices from one distribution.
get_filename_component(_sight_ort_root_real "${SIGHT_ORT_ROOT}" REALPATH)
get_filename_component(_sight_ort_library_real "${SIGHT_ORT_LIBRARY}" REALPATH)
get_filename_component(_sight_ort_include_real "${SIGHT_ORT_INCLUDE}" REALPATH)
string(FIND "${_sight_ort_library_real}" "${_sight_ort_root_real}/lib/" _sight_ort_library_in_root)
string(FIND "${_sight_ort_include_real}/" "${_sight_ort_root_real}/include/" _sight_ort_include_in_root)
set(SIGHT_ORT_SPLIT_SDK FALSE)
if(NOT _sight_ort_library_in_root EQUAL 0 OR NOT _sight_ort_include_in_root EQUAL 0)
    set(SIGHT_ORT_SPLIT_SDK TRUE)
    message(WARNING "Split ONNX Runtime SDK: local test builds only; staging disabled")
endif()


# VRM and Live2D parse JSON in both plugin and native tests.
find_package(nlohmann_json 3.2 REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/baseline-sources.cmake")
add_library(live-latent-lab-sight MODULE ${SIGHT_BASELINE_SOURCES})
target_link_libraries(live-latent-lab-sight PRIVATE nlohmann_json::nlohmann_json)
target_compile_definitions(live-latent-lab-sight PRIVATE
    SIGHT_ANIMEGAN_ONLY USE_ONNX_MEDIAPIPE)
option(SIGHT_SETTINGS_PAGE "Include native Qt Plugins settings page" ON)
if(SIGHT_SETTINGS_PAGE)
    find_package(Qt6 COMPONENTS Widgets QUIET)
    if(Qt6_FOUND)
        target_sources(live-latent-lab-sight PRIVATE src/ui/sight_settings.cpp)
        target_compile_definitions(live-latent-lab-sight PRIVATE SIGHT_WITH_SETTINGS_PAGE)
        target_link_libraries(live-latent-lab-sight PRIVATE Qt6::Widgets)
        add_executable(sight_plugin_settings_test tests/verify-plugin-settings.cpp)
        target_link_libraries(sight_plugin_settings_test PRIVATE Qt6::Widgets)
        add_test(NAME sight_plugin_settings_page COMMAND sight_plugin_settings_test)
        set_tests_properties(sight_plugin_settings_page PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
    else()
        message(STATUS "Qt6 unavailable: native Sight settings page disabled")
    endif()
endif()

# Optional local OCR: no mandatory dependency for existing builds.
option(SIGHT_ENABLE_OCR "Enable native Tesseract OCR when development files are available" ON)
if(SIGHT_ENABLE_OCR)
    find_path(SIGHT_TESSERACT_INCLUDE tesseract/baseapi.h)
    find_library(SIGHT_TESSERACT_LIBRARY NAMES tesseract libtesseract.so.5)
    if(SIGHT_TESSERACT_INCLUDE AND SIGHT_TESSERACT_LIBRARY)
        target_sources(live-latent-lab-sight PRIVATE src/text-obfuscation-filter.cpp src/core/text_ocr.cpp)
        target_include_directories(live-latent-lab-sight PRIVATE "${SIGHT_TESSERACT_INCLUDE}")
        target_link_libraries(live-latent-lab-sight PRIVATE "${SIGHT_TESSERACT_LIBRARY}")
        target_compile_definitions(live-latent-lab-sight PRIVATE SIGHT_HAVE_OCR)
        set(SIGHT_HAVE_OCR TRUE)
        message(STATUS "Sight native Tesseract OCR enabled")
        option(SIGHT_ENABLE_OCR_EDITOR "Enable Qt6 visual OCR region editor when available" ON)
        if(SIGHT_ENABLE_OCR_EDITOR)
            find_package(Qt6 QUIET COMPONENTS Widgets)
            if(TARGET Qt6::Widgets)
                target_sources(live-latent-lab-sight PRIVATE src/ui/ocr-region-editor.cpp)
                target_link_libraries(live-latent-lab-sight PRIVATE Qt6::Widgets)
                target_compile_definitions(live-latent-lab-sight PRIVATE SIGHT_HAVE_OCR_EDITOR)
                set(SIGHT_HAVE_OCR_EDITOR TRUE)
                message(STATUS "Sight Qt6 visual OCR editor enabled")
            else()
                message(STATUS "Sight OCR visual editor omitted: Qt6 Widgets unavailable")
            endif()
        endif()
    else()
        message(STATUS "Sight native OCR omitted: Tesseract development files unavailable")
    endif()
endif()

option(SIGHT_ENABLE_DENSE_FLOW "Enable OpenCV DIS spatial motion when available" ON)
if(SIGHT_ENABLE_DENSE_FLOW)
    find_package(OpenCV QUIET COMPONENTS core imgproc video)
    if(OpenCV_FOUND)
        target_sources(live-latent-lab-sight PRIVATE src/spatial-motion-filter.cpp src/core/dense_motion.cpp)
        target_include_directories(live-latent-lab-sight SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(live-latent-lab-sight PRIVATE ${OpenCV_LIBS})
        target_compile_definitions(live-latent-lab-sight PRIVATE SIGHT_HAVE_DENSE_FLOW)
        set(SIGHT_HAVE_DENSE_FLOW TRUE)
    endif()
endif()
# Private, explicitly opted-in official SDK. Core is loaded by absolute
# local path at model load time; never bundled or linked as a dependency.
set(SIGHT_CUBISM_SDK_ROOT "" CACHE PATH "Local official Cubism Native SDK 5 R5 (private development)")
if(SIGHT_CUBISM_SDK_ROOT)
    set(SIGHT_CUBISM_CORE "${SIGHT_CUBISM_SDK_ROOT}/Core/dll/linux/x86_64/libLive2DCubismCore.so")
    if(NOT EXISTS "${SIGHT_CUBISM_CORE}" OR NOT EXISTS "${SIGHT_CUBISM_SDK_ROOT}/Core/include/Live2DCubismCore.h")
        message(FATAL_ERROR "Invalid Cubism Native SDK root")
    endif()
    target_sources(live-latent-lab-sight PRIVATE src/native-live2d-source.cpp
        src/live2d/native_model.cpp src/live2d/model_package.cpp src/live2d/face_parameters.cpp src/live2d/animation.cpp)
    target_include_directories(live-latent-lab-sight PRIVATE "${SIGHT_CUBISM_SDK_ROOT}/Core/include")
    target_compile_definitions(live-latent-lab-sight PRIVATE SIGHT_HAVE_LIVE2D SIGHT_CUBISM_CORE_PATH="${SIGHT_CUBISM_CORE}")
endif()
# AVX2 is enabled only for the self-contained native face CNN and its header
# users, rather than changing flags for the rest of Sight.
set_source_files_properties(
    src/fast-face-detector.cpp
    src/3rdparty/libfacedetection/facedetectcnn.cpp
    src/3rdparty/libfacedetection/facedetectcnn-data.cpp
    src/3rdparty/libfacedetection/facedetectcnn-model.cpp
    PROPERTIES COMPILE_DEFINITIONS _ENABLE_AVX2
               COMPILE_OPTIONS "-mavx2;-mfma")
target_include_directories(live-latent-lab-sight PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_ORT_INCLUDE}" "${SIGHT_OBS_INCLUDE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-legacy/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-123/include")
target_link_libraries(live-latent-lab-sight PRIVATE
    "${SIGHT_OBS_LIBRARY}" "${SIGHT_ORT_LIBRARY}" dl)
set_target_properties(live-latent-lab-sight PROPERTIES
    PREFIX ""
    BUILD_RPATH "${SIGHT_ORT_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN")
# Refuse accidental system-wide installation of this user-owned OBS plugin.
# The guard must execute before the first file(INSTALL) directive.
# An explicit safe --prefix staging directory remains supported.
install(CODE [[
    if(CMAKE_INSTALL_PREFIX MATCHES "^/usr(/|$)" OR
       CMAKE_INSTALL_PREFIX STREQUAL "/")
        message(FATAL_ERROR
            "Sight refuses system-wide installation. Configure "
            "-DCMAKE_INSTALL_PREFIX=$ENV{HOME}/.config/obs-studio/plugins/live-latent-lab-sight "
            "or use cmake --install <build> --prefix <user-owned-stage-dir>.")
    endif()
]])

# Original Sight license and corresponding-source notice.
if(SIGHT_ORT_SPLIT_SDK)
    install(CODE [[message(FATAL_ERROR
        "Sight refuses staging a split ONNX Runtime SDK. Supply a single matching SIGHT_ORT_ROOT distribution and its notices in a fresh build directory.")]])
endif()
install(FILES LICENSE LICENSING.md DESTINATION data/licenses/sight)
install(TARGETS live-latent-lab-sight
    LIBRARY DESTINATION bin/64bit)
# Test fixtures are not redistribution inputs by default. Explicit local
# staging of an ABI-verified runtime is a separate user choice.
option(SIGHT_STAGE_LOCAL_MEDIAPIPE "Stage a separately acquired local MediaPipe runtime (not redistribution approval)" OFF)
if(SIGHT_STAGE_LOCAL_MEDIAPIPE AND NOT SIGHT_MEDIAPIPE_LIB)
    message(FATAL_ERROR "SIGHT_STAGE_LOCAL_MEDIAPIPE requires SIGHT_MEDIAPIPE_LIB")
endif()
if(SIGHT_MEDIAPIPE_LIB)
    if(NOT EXISTS "${SIGHT_MEDIAPIPE_LIB}")
        message(FATAL_ERROR "Missing configured MediaPipe runtime")
    endif()
    file(SHA256 "${SIGHT_MEDIAPIPE_LIB}" SIGHT_TASKS_LIBRARY_SHA256)
    # The legacy runtime and official MediaPipe 1.2.3 manylinux x86_64
    # runtime have different C options layouts. Both have real model tests.
    # Never package an untested ABI; enforce SHA256, not just file length.
    if(NOT SIGHT_TASKS_LIBRARY_SHA256 STREQUAL "7ba887d5e09d87440b3713816ec02897d6a849eadf2325c9aabe3707cbdac86b" AND
       NOT SIGHT_TASKS_LIBRARY_SHA256 STREQUAL "7ce8b23159d3fd4e8b727e35afbfb35f3d45d6bde1ec11cc01c316b3bf469f63")
        message(FATAL_ERROR "Unsupported MediaPipe binary ABI: ${SIGHT_TASKS_LIBRARY_SHA256}")
    endif()
    if(SIGHT_STAGE_LOCAL_MEDIAPIPE)
        install(FILES "${SIGHT_MEDIAPIPE_LIB}" DESTINATION bin/64bit)
        # Header package license does not establish binary redistribution rights.
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-legacy/LICENSE"
            DESTINATION data/licenses/mediapipe)
    endif()
endif()

# The bundled ONNX Runtime is a separate third-party project. Keep its
# license and notices with any staged/installed binary, not just in git.
foreach(SIGHT_ORT_NOTICE IN ITEMS LICENSE ThirdPartyNotices.txt)
    if(NOT EXISTS "${SIGHT_ORT_ROOT}/${SIGHT_ORT_NOTICE}")
        message(FATAL_ERROR "ONNX Runtime notice file missing from ${SIGHT_ORT_ROOT}: ${SIGHT_ORT_NOTICE}")
    endif()
endforeach()
install(FILES "${SIGHT_ORT_ROOT}/LICENSE" "${SIGHT_ORT_ROOT}/ThirdPartyNotices.txt"
    DESTINATION data/licenses/onnxruntime)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/libfacedetection/LICENSE"
    DESTINATION data/licenses/libfacedetection)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/vrm/vendor/tinygltf-LICENSE"
    DESTINATION data/licenses/tinygltf RENAME LICENSE)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/vrm/vendor/stb-LICENSE"
    DESTINATION data/licenses/stb RENAME LICENSE)

# Install the runtime under the ELF name recorded by the linker, even
# when a GPU distribution provides a differently named versioned file.
get_filename_component(_sight_ort_real "${SIGHT_ORT_LIBRARY}" REALPATH)
execute_process(COMMAND "${CMAKE_READELF}" -d "${_sight_ort_real}"
    OUTPUT_VARIABLE _sight_ort_dynamic RESULT_VARIABLE _sight_ort_readelf)
if(NOT _sight_ort_readelf EQUAL 0 OR
   NOT _sight_ort_dynamic MATCHES "SONAME[^\n]*\\[([A-Za-z0-9_.-]+)\\]")
    message(FATAL_ERROR "Cannot determine ONNX Runtime SONAME: ${_sight_ort_real}")
endif()
set(SIGHT_ORT_SONAME "${CMAKE_MATCH_1}")
install(FILES "${_sight_ort_real}" DESTINATION bin/64bit RENAME "${SIGHT_ORT_SONAME}")
foreach(_sight_ort_provider IN ITEMS shared cuda)
    set(_sight_ort_provider_file "${SIGHT_ORT_LIB_DIR}/libonnxruntime_providers_${_sight_ort_provider}.so")
    if(EXISTS "${_sight_ort_provider_file}")
        install(FILES "${_sight_ort_provider_file}" DESTINATION bin/64bit)
    endif()
endforeach()
# Explicit payload: never stage local assets just because they are under data/.
install(FILES data/locale/en-US.ini DESTINATION data/locale)
install(FILES data/presets/anime.json DESTINATION data/presets)
install(FILES
    data/sight-depth-camera.effect
    data/sight-privacy-conceal.effect
    data/sight-spatial-motion.effect
    data/sight-subject-matte.effect
    data/sight-text-obfuscation.effect
    data/sight-tracking-overlay.effect
    DESTINATION data)

enable_testing()
add_executable(sight_native_ort_smoke tests/native_ort_smoke.cpp)
target_include_directories(sight_native_ort_smoke PRIVATE src "${SIGHT_ORT_INCLUDE}")
target_link_libraries(sight_native_ort_smoke PRIVATE "${SIGHT_ORT_LIBRARY}")
add_test(NAME sight_native_ort_utf8 COMMAND sight_native_ort_smoke)
add_executable(sight_tasks_platform_policy tests/tasks_platform_policy_test.cpp)
target_include_directories(sight_tasks_platform_policy PRIVATE src)
add_test(NAME sight_tasks_platform_policy COMMAND sight_tasks_platform_policy)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/litert_tracking.cmake")
# Only uses Python's stdlib; tests installation safety without touching OBS.
find_program(SIGHT_PYTHON3_EXECUTABLE NAMES python3)
if(SIGHT_PYTHON3_EXECUTABLE)
    add_test(NAME sight_release_safety
        COMMAND "${SIGHT_PYTHON3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_release_safety.py")
    set_tests_properties(sight_release_safety PROPERTIES TIMEOUT 60)
    add_test(NAME sight_acquire_mediapipe_unit COMMAND "${SIGHT_PYTHON3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_acquire_mediapipe.py")
endif()
add_executable(sight_subject_motion_test tests/subject_motion_test.cpp)
target_include_directories(sight_subject_motion_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME sight_subject_motion_unit COMMAND sight_subject_motion_test)
find_program(SIGHT_XVFB_RUN NAMES xvfb-run)
if(TARGET sight_plugin_settings_test AND SIGHT_XVFB_RUN)
    add_executable(sight_settings_picker_test tests/verify-settings-picker.cpp)
    target_include_directories(sight_settings_picker_test PRIVATE "${SIGHT_OBS_INCLUDE}")
    target_link_libraries(sight_settings_picker_test PRIVATE Qt6::Core "${SIGHT_OBS_LIBRARY}")
    add_test(NAME sight_settings_native_picker COMMAND "${SIGHT_XVFB_RUN}" -a
        $<TARGET_FILE:sight_settings_picker_test> $<TARGET_FILE:live-latent-lab-sight> "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_settings_native_picker PROPERTIES TIMEOUT 30)
endif()

if(SIGHT_XVFB_RUN)
    # Real OBS video-render regression for optional manual text zones.
    # Independent of any external OCR/model installation or real webcam.
    add_executable(sight_privacy_text_obs_smoke tests/obs_sight_render_smoke.cpp)
    target_include_directories(sight_privacy_text_obs_smoke PRIVATE "${SIGHT_OBS_INCLUDE}")
    target_link_libraries(sight_privacy_text_obs_smoke PRIVATE "${SIGHT_OBS_LIBRARY}" pthread)
    add_test(NAME sight_privacy_manual_text_obs
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_privacy_text_obs_smoke>"
            "$<TARGET_FILE:live-latent-lab-sight>"
            "${CMAKE_CURRENT_SOURCE_DIR}/data/placeholder-model-not-used.onnx"
            "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_privacy_manual_text_obs PROPERTIES
        TIMEOUT 45
        ENVIRONMENT "SIGHT_SMOKE_EFFECT=sight_privacy_v2_filter;SIGHT_SMOKE_PRIVACY_TEXT_ZONES=0.25,0.25,0.75,0.75;SIGHT_SMOKE_PRIVACY_EXPECT_MANUAL=1")
    add_executable(sight_privacy_gpu_test tests/obs_privacy_gpu_test.cpp)
    target_include_directories(sight_privacy_gpu_test PRIVATE src "${SIGHT_OBS_INCLUDE}")
    target_link_libraries(sight_privacy_gpu_test PRIVATE "${SIGHT_OBS_LIBRARY}")
    add_test(NAME sight_privacy_gpu_concealment
        COMMAND "${SIGHT_XVFB_RUN}" -a -s "-screen 0 1280x720x24"
            "$<TARGET_FILE:sight_privacy_gpu_test>"
            "${CMAKE_CURRENT_SOURCE_DIR}/data/sight-privacy-conceal.effect")
    # Render the production Subject Cutout shader with an explicit test matte.
    # This verifies mask-selective GPU composition, not ONNX inference.
    add_executable(sight_subject_treatments_gpu_test
        tests/obs_subject_treatments_gpu_test.cpp)
    target_include_directories(sight_subject_treatments_gpu_test PRIVATE
        "${SIGHT_OBS_INCLUDE}")
    target_link_libraries(sight_subject_treatments_gpu_test PRIVATE
        "${SIGHT_OBS_LIBRARY}")
    add_test(NAME sight_subject_treatments_gpu
        COMMAND "${SIGHT_XVFB_RUN}" -a -s "-screen 0 1280x720x24"
            "$<TARGET_FILE:sight_subject_treatments_gpu_test>"
            "${CMAKE_CURRENT_SOURCE_DIR}/data/sight-subject-matte.effect")
    set_tests_properties(sight_subject_treatments_gpu PROPERTIES TIMEOUT 30)
endif()


# Respect an explicitly configured OBS image-source path and discover common
# distro layouts; privacy checks must not assume /usr/local only.
find_file(SIGHT_IMAGE_SOURCE_MODULE NAMES image-source.so
    PATHS /usr/local/lib/obs-plugins
          /usr/lib/x86_64-linux-gnu/obs-plugins
          /usr/lib/obs-plugins
          /usr/lib64/obs-plugins
    NO_DEFAULT_PATH)

add_executable(sight_semantic_detector_real_test tests/semantic_detector_real_test.cpp)
target_include_directories(sight_semantic_detector_real_test PRIVATE src "${SIGHT_ORT_INCLUDE}")
target_link_libraries(sight_semantic_detector_real_test PRIVATE "${SIGHT_ORT_LIBRARY}")
set(_semantic_ort_dir "${CMAKE_CURRENT_BINARY_DIR}/semantic-test-runtime")
file(MAKE_DIRECTORY "${_semantic_ort_dir}")
file(CREATE_LINK "${_sight_ort_real}" "${_semantic_ort_dir}/${SIGHT_ORT_SONAME}" SYMBOLIC)
foreach(_provider IN ITEMS shared cuda)
    if(EXISTS "${SIGHT_ORT_LIB_DIR}/libonnxruntime_providers_${_provider}.so")
        file(CREATE_LINK "${SIGHT_ORT_LIB_DIR}/libonnxruntime_providers_${_provider}.so"
            "${_semantic_ort_dir}/libonnxruntime_providers_${_provider}.so" SYMBOLIC)
    endif()
endforeach()
set_target_properties(sight_semantic_detector_real_test PROPERTIES BUILD_RPATH "${_semantic_ort_dir};${SIGHT_ORT_LIB_DIR}")
set(SIGHT_SEMANTIC_TEST_MODEL "" CACHE FILEPATH "Optional local 14-class Nevermore small ONNX for real detector validation")
set(SIGHT_SEMANTIC_TEST_RGBA "" CACHE FILEPATH "Optional local 512x512 RGBA portrait for semantic detector validation")
if(EXISTS "${SIGHT_SEMANTIC_TEST_MODEL}" AND EXISTS "${SIGHT_SEMANTIC_TEST_RGBA}")
    add_test(NAME sight_semantic_real_inference COMMAND sight_semantic_detector_real_test
        "${SIGHT_SEMANTIC_TEST_MODEL}" "${SIGHT_SEMANTIC_TEST_RGBA}")
endif()

set(SIGHT_SEMANTIC_TEST_SMALL_FP16 "" CACHE FILEPATH "Optional local Nevermore small FP16 ONNX")
set(SIGHT_SEMANTIC_TEST_TINY_FP16 "" CACHE FILEPATH "Optional local Nevermore tiny FP16 ONNX")
if(EXISTS "${SIGHT_SEMANTIC_TEST_RGBA}")
    foreach(_variant IN ITEMS SMALL TINY)
        if(EXISTS "${SIGHT_SEMANTIC_TEST_${_variant}_FP16}")
            if(_variant STREQUAL "SMALL")
                set(_classes 14)
            else()
                set(_classes 10)
            endif()
            string(TOLOWER "${_variant}" _suffix)
            add_test(NAME sight_semantic_${_suffix}_fp16_inference
                COMMAND sight_semantic_detector_real_test
                    "${SIGHT_SEMANTIC_TEST_${_variant}_FP16}" "${SIGHT_SEMANTIC_TEST_RGBA}" "${_classes}")
        endif()
    endforeach()
endif()

add_executable(sight_semantic_temporal_test tests/semantic_temporal_test.cpp)
target_include_directories(sight_semantic_temporal_test PRIVATE src)
add_test(NAME sight_semantic_temporal_unit COMMAND sight_semantic_temporal_test)

add_executable(sight_semantic_regions_test tests/semantic_regions_test.cpp)
target_include_directories(sight_semantic_regions_test PRIVATE src)
add_test(NAME sight_semantic_regions_unit COMMAND sight_semantic_regions_test)

add_executable(sight_privacy_coverage_test tests/privacy_coverage_test.cpp)
target_include_directories(sight_privacy_coverage_test PRIVATE src)
add_test(NAME sight_privacy_coverage_unit COMMAND sight_privacy_coverage_test)

# Respect an explicitly configured OBS image-source path and discover common
# distro layouts; privacy checks must not assume /usr/local only.
find_file(SIGHT_IMAGE_SOURCE_MODULE NAMES image-source.so
    PATHS /usr/local/lib/obs-plugins
          /usr/lib/x86_64-linux-gnu/obs-plugins
          /usr/lib/obs-plugins
          /usr/lib64/obs-plugins
    NO_DEFAULT_PATH)

set(SIGHT_SEMANTIC_TEST_IMAGE "" CACHE FILEPATH "Optional local portrait PNG for semantic OBS validation")
if(TARGET sight_privacy_text_obs_smoke AND SIGHT_IMAGE_SOURCE_MODULE AND
   EXISTS "${SIGHT_SEMANTIC_TEST_MODEL}" AND EXISTS "${SIGHT_SEMANTIC_TEST_IMAGE}")
    add_test(NAME sight_semantic_obs_integration
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_privacy_text_obs_smoke>"
            "$<TARGET_FILE:live-latent-lab-sight>"
            "${SIGHT_SEMANTIC_TEST_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_semantic_obs_integration PROPERTIES TIMEOUT 45
        ENVIRONMENT "SIGHT_SMOKE_SEMANTIC_MODEL=${SIGHT_SEMANTIC_TEST_MODEL};SIGHT_SMOKE_EFFECT=sight_privacy_v2_filter;SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_SEMANTIC_TEST_IMAGE};SIGHT_SMOKE_HOLD_MS=2500;SIGHT_SMOKE_PRIVACY_EXPECT_MASKED=1"
        ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:${_semantic_ort_dir}")
    foreach(_mode IN ITEMS buffered missing_model)
        add_test(NAME sight_semantic_obs_${_mode}
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                "$<TARGET_FILE:sight_privacy_text_obs_smoke>"
                "$<TARGET_FILE:live-latent-lab-sight>"
                "${SIGHT_SEMANTIC_TEST_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
        if(_mode STREQUAL "buffered")
            set(_semantic_env "SIGHT_SMOKE_SEMANTIC_MODEL=${SIGHT_SEMANTIC_TEST_MODEL};SIGHT_SMOKE_PRIVACY_EXPECT_MASKED=1")
        else()
            set(_semantic_env "SIGHT_SMOKE_SEMANTIC_MODEL=${CMAKE_CURRENT_BINARY_DIR}/semantic-intentionally-absent/model.onnx;SIGHT_SMOKE_SEMANTIC_EXPECT_SLATE=1")
        endif()
        set_tests_properties(sight_semantic_obs_${_mode} PROPERTIES TIMEOUT 45
            ENVIRONMENT "${_semantic_env};SIGHT_SMOKE_SEMANTIC_BUFFERED=1;SIGHT_SMOKE_EFFECT=sight_privacy_v2_filter;SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_SEMANTIC_TEST_IMAGE};SIGHT_SMOKE_HOLD_MS=2500"
            ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:${_semantic_ort_dir}")
    endforeach()

endif()

# Real face CNN + GPU-filter smoke tests using a pinned local 512x512 reference.
if(SIGHT_MEDIAPIPE_TEST_RGBA AND SIGHT_MEDIAPIPE_FACE_MODEL AND SIGHT_XVFB_RUN)
    if(SIGHT_IMAGE_SOURCE_MODULE)
        add_test(NAME sight_privacy_v2_obs_integration
            COMMAND bash
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify-privacy-v2.sh"
                "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                "$<TARGET_FILE:live-latent-lab-sight>"
                "${CMAKE_CURRENT_SOURCE_DIR}/data"
                "${SIGHT_MEDIAPIPE_TEST_RGBA}"
                "${SIGHT_MEDIAPIPE_FACE_MODEL}")
        set_tests_properties(sight_privacy_v2_obs_integration PROPERTIES
            TIMEOUT 60
            ENVIRONMENT "SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE}")
    endif()
endif()

add_executable(sight_tracking_overlay_test tests/tracking_overlay_test.cpp
    src/core/tracking_overlay.cpp)
target_include_directories(sight_tracking_overlay_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME sight_tracking_overlay_unit COMMAND sight_tracking_overlay_test)
add_executable(sight_tracking_signal_bus_test
    tests/tracking_signal_bus_test.cpp src/core/tracking_signal_bus.cpp)
target_include_directories(sight_tracking_signal_bus_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME sight_tracking_signal_bus_unit
    COMMAND sight_tracking_signal_bus_test)
add_executable(sight_tasks_abi_123_layout_test
    tests/tasks_abi_123_layout_test.cpp)
target_include_directories(sight_tasks_abi_123_layout_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-legacy/include")
add_test(NAME sight_tasks_abi_123_layout_unit
    COMMAND sight_tasks_abi_123_layout_test)

# Pure conversion of a single Holistic result into independent Sight
# Face/Body/Hand tracking payloads, without loading a graph or opening OBS.
add_executable(sight_holistic_signal_adapter_test
    tests/holistic_signal_adapter_test.cpp
    src/core/hand_signals.cpp
    src/core/tracking_signal_bus.cpp)
target_include_directories(sight_holistic_signal_adapter_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-legacy/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-123/include")
target_compile_options(sight_holistic_signal_adapter_test PRIVATE -UNDEBUG)
add_test(NAME sight_holistic_signal_adapter_unit
    COMMAND sight_holistic_signal_adapter_test)

# Deterministic pose/hand stability edge cases without any camera/model
# dependency. Production Holistic raw signal publication remains untouched.
add_executable(sight_holistic_overlay_stability_test
    tests/holistic_overlay_stability_test.cpp)
target_include_directories(sight_holistic_overlay_stability_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-legacy/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-123/include")
target_compile_options(sight_holistic_overlay_stability_test PRIVATE -UNDEBUG)
add_test(NAME sight_holistic_overlay_stability_unit
    COMMAND sight_holistic_overlay_stability_test)
# Pure mask credibility and joint-gating checks; no MediaPipe/OBS required.
add_executable(sight_holistic_person_mask_test
    tests/holistic_person_mask_test.cpp)
target_include_directories(sight_holistic_person_mask_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-legacy/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-123/include")
target_compile_options(sight_holistic_person_mask_test PRIVATE -UNDEBUG)
add_test(NAME sight_holistic_person_mask_unit
    COMMAND sight_holistic_person_mask_test)



# Object detection has its own stable-ID layer. Its deterministic unit
# tests are separate from actual MediaPipe inference acceptance.
add_executable(sight_object_tracks_test tests/object_tracks_test.cpp)
target_include_directories(sight_object_tracks_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(sight_object_tracks_test PRIVATE -UNDEBUG)
add_test(NAME sight_object_tracks_unit COMMAND sight_object_tracks_test)

add_executable(sight_object_overlay_test
    tests/object_overlay_test.cpp src/core/object_overlay.cpp)
target_include_directories(sight_object_overlay_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(sight_object_overlay_test PRIVATE -UNDEBUG)
add_test(NAME sight_object_overlay_unit COMMAND sight_object_overlay_test)

add_executable(sight_object_real_inference_test
    tests/object_detector_real_test.cpp src/core/object_detector.cpp)
target_include_directories(sight_object_real_inference_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-legacy/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-123/include")
target_compile_options(sight_object_real_inference_test PRIVATE -UNDEBUG)
target_link_libraries(sight_object_real_inference_test PRIVATE dl)

# The three payloads from a single inference must appear together, and a
# channel collision cannot update only some of them.
add_executable(sight_holistic_signal_batch_test
    tests/holistic_signal_batch_test.cpp src/core/tracking_signal_bus.cpp)
target_include_directories(sight_holistic_signal_batch_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(sight_holistic_signal_batch_test PRIVATE -UNDEBUG)
add_test(NAME sight_holistic_signal_batch_unit
    COMMAND sight_holistic_signal_batch_test)
# Runtime discovery must prefer the separately installed official ABI
# without replacing the legacy library or using arbitrary filesystem paths.
add_executable(sight_holistic_runtime_discovery_test
    tests/holistic_runtime_discovery_test.cpp)
target_include_directories(sight_holistic_runtime_discovery_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME sight_holistic_runtime_discovery_unit
    COMMAND sight_holistic_runtime_discovery_test)


# C API is dynamically loaded, so compilation needs no model or .so. Only
# the optional CTest invocation needs a user-supplied .task and fixture.
add_executable(sight_holistic_real_inference_test
    tests/holistic_tracker_real_test.cpp
    src/core/holistic_tracker.cpp src/core/hand_signals.cpp)
target_include_directories(sight_holistic_real_inference_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-legacy/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-123/include")
target_compile_options(sight_holistic_real_inference_test PRIVATE -UNDEBUG)
target_link_libraries(sight_holistic_real_inference_test PRIVATE dl)


add_executable(sight_tracking_snapshot_export_test
    tests/sight_snapshot_export_test.cpp src/core/tracking_signal_bus.cpp)
target_include_directories(sight_tracking_snapshot_export_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME sight_tracking_snapshot_export_unit
    COMMAND sight_tracking_snapshot_export_test)

if(SIGHT_XVFB_RUN)
    add_executable(sight_tracking_snapshot_proc_obs_test
        tests/sight_snapshot_proc_obs_test.cpp)
    target_include_directories(sight_tracking_snapshot_proc_obs_test PRIVATE
        "${SIGHT_OBS_INCLUDE}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(sight_tracking_snapshot_proc_obs_test PRIVATE
        "${SIGHT_OBS_LIBRARY}")
    add_test(NAME sight_tracking_snapshot_proc_obs
        COMMAND "${SIGHT_XVFB_RUN}" -a -s "-screen 0 1280x720x24"
            "$<TARGET_FILE:sight_tracking_snapshot_proc_obs_test>"
            "$<TARGET_FILE:live-latent-lab-sight>"
            "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_tracking_snapshot_proc_obs PROPERTIES TIMEOUT 30)
endif()
add_executable(sight_live2d_animation_test tests/live2d_animation_test.cpp src/live2d/animation.cpp)
target_include_directories(sight_live2d_animation_test PRIVATE src)
target_link_libraries(sight_live2d_animation_test PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(sight_live2d_animation_test PRIVATE -UNDEBUG)
add_test(NAME sight_live2d_animation_unit COMMAND sight_live2d_animation_test)
add_executable(sight_live2d_intake_test tests/live2d_intake_test.cpp
    src/live2d/model_package.cpp src/live2d/face_parameters.cpp)
target_include_directories(sight_live2d_intake_test PRIVATE src)
target_link_libraries(sight_live2d_intake_test PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(sight_live2d_intake_test PRIVATE -UNDEBUG)
add_test(NAME sight_live2d_intake_unit COMMAND sight_live2d_intake_test)

add_executable(sight_vrm_asset_test tests/vrm_asset_test.cpp src/vrm/vrm_asset.cpp)
target_include_directories(sight_vrm_asset_test PRIVATE src)
target_link_libraries(sight_vrm_asset_test PRIVATE nlohmann_json::nlohmann_json)
# Keep test assertions active in Release / RelWithDebInfo builds.
target_compile_options(sight_vrm_asset_test PRIVATE -UNDEBUG)
add_test(NAME sight_vrm_asset_unit COMMAND sight_vrm_asset_test)
add_executable(sight_hand_frame_dedup_test tests/hand_frame_dedup_test.cpp)
target_include_directories(sight_hand_frame_dedup_test PRIVATE src)
target_compile_options(sight_hand_frame_dedup_test PRIVATE -UNDEBUG)
add_test(NAME sight_hand_frame_dedup_unit COMMAND sight_hand_frame_dedup_test)

add_executable(sight_face_quality_test tests/face_quality_test.cpp)
target_include_directories(sight_face_quality_test PRIVATE src)
target_compile_options(sight_face_quality_test PRIVATE -UNDEBUG)
add_test(NAME sight_face_quality_unit COMMAND sight_face_quality_test)

add_executable(sight_hand_temporal_smoothing_test
    tests/hand_temporal_smoothing_test.cpp src/core/hand_signals.cpp)
target_include_directories(sight_hand_temporal_smoothing_test PRIVATE src)
target_compile_options(sight_hand_temporal_smoothing_test PRIVATE -UNDEBUG)
add_test(NAME sight_hand_temporal_smoothing_unit COMMAND sight_hand_temporal_smoothing_test)

add_executable(sight_tasks_world_landmarks_test
    tests/tasks_world_landmarks_test.cpp src/core/tracking_signal_bus.cpp)
target_include_directories(sight_tasks_world_landmarks_test PRIVATE
    src external/mediapipe-tasks-legacy/include)
target_compile_options(sight_tasks_world_landmarks_test PRIVATE -UNDEBUG)
add_test(NAME sight_tasks_world_landmarks_unit COMMAND sight_tasks_world_landmarks_test)

add_executable(sight_vrm_tracking_test tests/vrm_tracking_test.cpp
    src/vrm/vrm_asset.cpp src/vrm/vrm_tracking.cpp)
target_include_directories(sight_vrm_tracking_test PRIVATE src)
target_link_libraries(sight_vrm_tracking_test PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(sight_vrm_tracking_test PRIVATE -UNDEBUG)
add_test(NAME sight_vrm_tracking_unit COMMAND sight_vrm_tracking_test)
add_executable(sight_vrm_coverage_test
    tests/vrm_coverage_test.cpp src/vrm/vrm_coverage.cpp
    src/vrm/vrm_asset.cpp src/vrm/vrm_tracking.cpp)
target_include_directories(sight_vrm_coverage_test PRIVATE src)
target_link_libraries(sight_vrm_coverage_test PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(sight_vrm_coverage_test PRIVATE -UNDEBUG)
add_test(NAME sight_vrm_coverage_unit COMMAND sight_vrm_coverage_test)

add_executable(sight_vrm_transform_interop_test
    tests/vrm_transform_interop_test.cpp src/vrm/vrm_asset.cpp
    src/vrm/vrm_tracking.cpp)
target_include_directories(sight_vrm_transform_interop_test PRIVATE src)
target_link_libraries(sight_vrm_transform_interop_test PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(sight_vrm_transform_interop_test PRIVATE -UNDEBUG)
add_test(NAME sight_vrm_transform_interop_unit
    COMMAND sight_vrm_transform_interop_test)

add_executable(sight_vrm_body_tracking_test tests/vrm_body_tracking_test.cpp
    src/vrm/vrm_asset.cpp src/vrm/vrm_tracking.cpp)
target_include_directories(sight_vrm_body_tracking_test PRIVATE src)
target_link_libraries(sight_vrm_body_tracking_test PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(sight_vrm_body_tracking_test PRIVATE -UNDEBUG)
add_test(NAME sight_vrm_body_tracking_unit COMMAND sight_vrm_body_tracking_test)
add_executable(sight_vrm_hand_tracking_test tests/vrm_hand_tracking_test.cpp
    src/vrm/vrm_asset.cpp src/vrm/vrm_tracking.cpp)
target_include_directories(sight_vrm_hand_tracking_test PRIVATE src)
target_link_libraries(sight_vrm_hand_tracking_test PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(sight_vrm_hand_tracking_test PRIVATE -UNDEBUG)
add_test(NAME sight_vrm_hand_tracking_unit COMMAND sight_vrm_hand_tracking_test)
if(SIGHT_XVFB_RUN)
    add_executable(sight_obs_native_vrm_smoke tests/obs_native_vrm_smoke.cpp)
    target_include_directories(sight_obs_native_vrm_smoke PRIVATE "${SIGHT_OBS_INCLUDE}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
    # Link to the real plugin module so that test signals reach its actual
    # process-local tracking bus, rather than a duplicate test-side bus.
    # The Sight OBS module is named live-latent-lab-sight.so (not lib*.so).
    # Exact -l:filename selects it, rather than CMake rewriting the basename
    # as -llive-latent-lab-sight which ld cannot resolve.
    target_link_directories(sight_obs_native_vrm_smoke PRIVATE "${CMAKE_BINARY_DIR}")
    target_link_libraries(sight_obs_native_vrm_smoke PRIVATE "${SIGHT_OBS_LIBRARY}" pthread "-l:live-latent-lab-sight.so")
    add_dependencies(sight_obs_native_vrm_smoke live-latent-lab-sight)
    add_test(NAME sight_native_vrm_obs_render
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_native_vrm_smoke.sh"
            "$<TARGET_FILE:sight_obs_native_vrm_smoke>"
            "$<TARGET_FILE:live-latent-lab-sight>"
            "$<TARGET_FILE:sight_vrm_asset_test>"
            "${SIGHT_XVFB_RUN}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_native_vrm_obs_render PROPERTIES TIMEOUT 30)
endif()
if(SIGHT_CUBISM_SDK_ROOT AND SIGHT_XVFB_RUN)
    add_executable(sight_obs_native_live2d_smoke tests/obs_native_live2d_smoke.cpp)
    target_include_directories(sight_obs_native_live2d_smoke PRIVATE "${SIGHT_OBS_INCLUDE}" src)
    target_link_directories(sight_obs_native_live2d_smoke PRIVATE "${CMAKE_BINARY_DIR}")
    target_link_libraries(sight_obs_native_live2d_smoke PRIVATE "${SIGHT_OBS_LIBRARY}" pthread "-l:live-latent-lab-sight.so")
    add_dependencies(sight_obs_native_live2d_smoke live-latent-lab-sight)
    add_test(NAME sight_native_live2d_obs_render COMMAND "${SIGHT_XVFB_RUN}" -a
        "$<TARGET_FILE:sight_obs_native_live2d_smoke>" "$<TARGET_FILE:live-latent-lab-sight>"
        "${SIGHT_CUBISM_SDK_ROOT}/Samples/Resources/Haru/Haru.model3.json" "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_native_live2d_obs_render PROPERTIES TIMEOUT 60)
endif()
add_executable(sight_value_modulation_test
    tests/value_modulation_test.cpp src/core/value_modulation.cpp
    src/core/hand_signals.cpp)
target_include_directories(sight_value_modulation_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME sight_value_modulation_unit COMMAND sight_value_modulation_test)



set(SIGHT_MEDIAPIPE_LIB "" CACHE FILEPATH "Pinned verified MediaPipe Tasks C API runtime")
set(SIGHT_MEDIAPIPE_FACE_MODEL "" CACHE FILEPATH "Face task model for real smoke")
set(SIGHT_MEDIAPIPE_POSE_MODEL "" CACHE FILEPATH "Pose task model for real smoke")
set(SIGHT_MEDIAPIPE_TEST_RGBA "" CACHE FILEPATH "Real RGBA 512x512 face/pose fixture")
set(SIGHT_MEDIAPIPE_TEST_IMAGE "" CACHE FILEPATH "Real 512x512 test face image")
set(SIGHT_GESTURE_MODEL "" CACHE FILEPATH "Verified gesture model for real hand tracking")
set(SIGHT_GESTURE_TEST_RGBA "" CACHE FILEPATH "RGBA 512x512 with real hand and gesture")
set(SIGHT_GESTURE_TEST_IMAGE "" CACHE FILEPATH "Real 512x512 hand picture for OBS")
if(SIGHT_GESTURE_MODEL)
    if(NOT EXISTS "${SIGHT_GESTURE_MODEL}")
        message(FATAL_ERROR "Gesture model missing: ${SIGHT_GESTURE_MODEL}")
    endif()
    file(SHA256 "${SIGHT_GESTURE_MODEL}" SIGHT_GESTURE_SHA256)
    if(NOT SIGHT_GESTURE_SHA256 STREQUAL
       "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482")
        message(FATAL_ERROR "Unverified gesture model: ${SIGHT_GESTURE_SHA256}")
    endif()
endif()

if(SIGHT_MEDIAPIPE_LIB AND SIGHT_MEDIAPIPE_FACE_MODEL AND SIGHT_MEDIAPIPE_POSE_MODEL AND SIGHT_MEDIAPIPE_TEST_RGBA)
    add_executable(sight_face_quality_real_test
        tests/face_quality_real_test.cpp src/core/tasks_tracker.cpp src/core/hand_signals.cpp)
    target_include_directories(sight_face_quality_real_test PRIVATE
        src external/mediapipe-tasks-legacy/include external/mediapipe-tasks-123/include)
    target_link_libraries(sight_face_quality_real_test PRIVATE dl)
    if(SIGHT_MEDIAPIPE_FACE_MODEL AND SIGHT_MEDIAPIPE_TEST_RGBA)
        add_test(NAME sight_face_quality_real
            COMMAND sight_face_quality_real_test "${SIGHT_MEDIAPIPE_LIB}"
                "${SIGHT_MEDIAPIPE_FACE_MODEL}" "${SIGHT_MEDIAPIPE_TEST_RGBA}")
        set_tests_properties(sight_face_quality_real PROPERTIES TIMEOUT 35)
    endif()
    add_executable(sight_tasks_real_inference
        tests/tasks_tracking_inference_test.cpp src/core/tasks_tracker.cpp
        src/core/hand_signals.cpp)
    target_include_directories(sight_tasks_real_inference PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-legacy/include")
    target_link_libraries(sight_tasks_real_inference PRIVATE dl)
    if(SIGHT_GESTURE_MODEL AND SIGHT_GESTURE_TEST_RGBA)
        add_test(NAME sight_tasks_hand_real_inference
            COMMAND sight_tasks_real_inference "${SIGHT_MEDIAPIPE_LIB}"
                "${SIGHT_GESTURE_MODEL}" "${SIGHT_GESTURE_TEST_RGBA}" hand)
        set_tests_properties(sight_tasks_hand_real_inference PROPERTIES TIMEOUT 35)
    endif()
    foreach(kind IN ITEMS face pose)
        if(kind STREQUAL "face")
            set(taskmodel "${SIGHT_MEDIAPIPE_FACE_MODEL}")
        else()
            set(taskmodel "${SIGHT_MEDIAPIPE_POSE_MODEL}")
        endif()
        add_test(NAME sight_tasks_${kind}_real_inference
            COMMAND sight_tasks_real_inference "${SIGHT_MEDIAPIPE_LIB}" "${taskmodel}"
                "${SIGHT_MEDIAPIPE_TEST_RGBA}" "${kind}")
        set_tests_properties(sight_tasks_${kind}_real_inference PROPERTIES TIMEOUT 35)
    endforeach()
endif()

# Optional 1.2.3 compatibility tests. Explicit and isolated: no model or
# wheel download, no auto-upgrade, no changes to OBS user configuration.
set(SIGHT_MEDIAPIPE_123_LIB "" CACHE FILEPATH
    "Official MediaPipe 1.2.3 Linux x86_64 wheel libmediapipe.so for extra CTests")
if(SIGHT_MEDIAPIPE_123_LIB)
    if(NOT EXISTS "${SIGHT_MEDIAPIPE_123_LIB}")
        message(FATAL_ERROR "MediaPipe 1.2.3 test runtime missing: ${SIGHT_MEDIAPIPE_123_LIB}")
    endif()
    file(SHA256 "${SIGHT_MEDIAPIPE_123_LIB}" SIGHT_MEDIAPIPE_123_SHA256)
    if(NOT SIGHT_MEDIAPIPE_123_SHA256 STREQUAL
       "7ce8b23159d3fd4e8b727e35afbfb35f3d45d6bde1ec11cc01c316b3bf469f63")
        message(FATAL_ERROR "Unverified official MediaPipe 1.2.3 binary: ${SIGHT_MEDIAPIPE_123_SHA256}")
    endif()
    if(TARGET sight_tasks_real_inference)
        foreach(kind IN ITEMS face pose)
            if(kind STREQUAL "face")
                set(taskmodel "${SIGHT_MEDIAPIPE_FACE_MODEL}")
            else()
                set(taskmodel "${SIGHT_MEDIAPIPE_POSE_MODEL}")
            endif()
            add_test(NAME sight_tasks_123_${kind}_real_inference
                COMMAND sight_tasks_real_inference "${SIGHT_MEDIAPIPE_123_LIB}" "${taskmodel}"
                    "${SIGHT_MEDIAPIPE_TEST_RGBA}" "${kind}")
            set_tests_properties(sight_tasks_123_${kind}_real_inference
                PROPERTIES TIMEOUT 45)
        endforeach()
        if(SIGHT_GESTURE_MODEL AND SIGHT_GESTURE_TEST_RGBA)
            add_test(NAME sight_tasks_123_hand_real_inference
                COMMAND sight_tasks_real_inference "${SIGHT_MEDIAPIPE_123_LIB}"
                    "${SIGHT_GESTURE_MODEL}" "${SIGHT_GESTURE_TEST_RGBA}" hand)
            set_tests_properties(sight_tasks_123_hand_real_inference
                PROPERTIES TIMEOUT 45)
        endif()
    endif()
endif()

# Real object detection is opt-in; weights/fixture are never installed
# or fetched by CMake. They belong to the user's local test environment.
set(SIGHT_OBJECT_MODEL "" CACHE FILEPATH "Local MediaPipe-compatible object detection .tflite")
set(SIGHT_OBJECT_TEST_RGBA "" CACHE FILEPATH "Matching real RGBA fixture")
set(SIGHT_OBJECT_TEST_IMAGE "" CACHE FILEPATH "Matching image for isolated OBS")
set(SIGHT_OBJECT_TEST_WIDTH 512 CACHE STRING "Real object fixture width")
set(SIGHT_OBJECT_TEST_HEIGHT 333 CACHE STRING "Real object fixture height")
if(SIGHT_OBJECT_MODEL AND SIGHT_MEDIAPIPE_123_LIB AND SIGHT_OBJECT_TEST_RGBA)
    add_test(NAME sight_object_123_real_inference
        COMMAND sight_object_real_inference_test
            "${SIGHT_MEDIAPIPE_123_LIB}"
            "${SIGHT_OBJECT_MODEL}" "${SIGHT_OBJECT_TEST_RGBA}"
            "${SIGHT_OBJECT_TEST_WIDTH}" "${SIGHT_OBJECT_TEST_HEIGHT}")
    set_tests_properties(sight_object_123_real_inference PROPERTIES TIMEOUT 50)
endif()

# Holistic samples are explicitly user-supplied; NEVER download them in
# configure/test or bundle model weights with the plugin.
set(SIGHT_HOLISTIC_MODEL "" CACHE FILEPATH "MediaPipe Holistic float16 .task for optional tests")
set(SIGHT_HOLISTIC_TEST_RGBA "" CACHE FILEPATH "512x512 RGBA image fixture")
set(SIGHT_HOLISTIC_TEST_IMAGE "" CACHE FILEPATH "matching PNG used by OBS image source")
if(SIGHT_HOLISTIC_MODEL AND SIGHT_MEDIAPIPE_123_LIB AND SIGHT_HOLISTIC_TEST_RGBA)
    if(NOT EXISTS "${SIGHT_HOLISTIC_MODEL}")
        message(FATAL_ERROR "Holistic model test file not found")
    endif()
    add_test(NAME sight_holistic_123_real_inference
        COMMAND sight_holistic_real_inference_test
            "${SIGHT_MEDIAPIPE_123_LIB}" "${SIGHT_HOLISTIC_MODEL}" "${SIGHT_HOLISTIC_TEST_RGBA}")
    set_tests_properties(sight_holistic_123_real_inference PROPERTIES TIMEOUT 45)
endif()

add_executable(sight_animegan_model_test
    tests/animegan_inference_test.cpp src/core/enhancement_processor.cpp
    src/common/sight-log.cpp)
target_include_directories(sight_animegan_model_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_ORT_INCLUDE}" "${SIGHT_OBS_INCLUDE}")
target_compile_definitions(sight_animegan_model_test PRIVATE USE_ONNX_MEDIAPIPE)
target_link_libraries(sight_animegan_model_test PRIVATE "${SIGHT_OBS_LIBRARY}")
target_link_libraries(sight_animegan_model_test PRIVATE "${SIGHT_ORT_LIBRARY}")
set_target_properties(sight_animegan_model_test PROPERTIES
    BUILD_RPATH "${SIGHT_ORT_LIB_DIR}")

add_executable(sight_sketch_handoff_test tests/sketch_handoff_test.cpp
    src/core/enhancement_processor.cpp src/common/sight-log.cpp)
target_include_directories(sight_sketch_handoff_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_ORT_INCLUDE}" "${SIGHT_OBS_INCLUDE}")
target_compile_definitions(sight_sketch_handoff_test PRIVATE USE_ONNX_MEDIAPIPE)
target_link_libraries(sight_sketch_handoff_test PRIVATE "${SIGHT_OBS_LIBRARY}" "${SIGHT_ORT_LIBRARY}")
set_target_properties(sight_sketch_handoff_test PROPERTIES BUILD_RPATH "${SIGHT_ORT_LIB_DIR}")
add_test(NAME sight_sketch_handoff_unit COMMAND sight_sketch_handoff_test)
add_executable(sight_cutout_handoff_test tests/cutout_handoff_test.cpp
    src/core/segmentation_processor.cpp src/common/sight-log.cpp)
target_include_directories(sight_cutout_handoff_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_ORT_INCLUDE}" "${SIGHT_OBS_INCLUDE}")
target_compile_definitions(sight_cutout_handoff_test PRIVATE USE_ONNX_MEDIAPIPE SIGHT_ANIMEGAN_ONLY)
target_link_libraries(sight_cutout_handoff_test PRIVATE "${SIGHT_OBS_LIBRARY}" "${SIGHT_ORT_LIBRARY}")
set_target_properties(sight_cutout_handoff_test PROPERTIES BUILD_RPATH "${SIGHT_ORT_LIB_DIR}")
add_test(NAME sight_cutout_handoff_unit COMMAND sight_cutout_handoff_test)
add_executable(sight_depth_handoff_test tests/depth_handoff_test.cpp
    src/core/onnx_depth_processor.cpp src/common/sight-log.cpp)
target_include_directories(sight_depth_handoff_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_ORT_INCLUDE}" "${SIGHT_OBS_INCLUDE}")
target_link_libraries(sight_depth_handoff_test PRIVATE "${SIGHT_OBS_LIBRARY}" "${SIGHT_ORT_LIBRARY}")
set_target_properties(sight_depth_handoff_test PROPERTIES BUILD_RPATH "${SIGHT_ORT_LIB_DIR}")
add_test(NAME sight_depth_handoff_unit COMMAND sight_depth_handoff_test)



# Explicitly exercise the no-ONNX build's unavailable-backend contract.
add_executable(sight_segmentation_unavailable_test
    tests/segmentation_unavailable_test.cpp src/core/segmentation_processor.cpp)
target_include_directories(sight_segmentation_unavailable_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_OBS_INCLUDE}")
add_test(NAME sight_segmentation_unavailable COMMAND sight_segmentation_unavailable_test)

set(SIGHT_U2NET_MODEL "" CACHE FILEPATH
    "Optional local U2Net ONNX model for real-model and OBS matte tests")
set(SIGHT_U2NET_TEST_IMAGE "" CACHE FILEPATH
    "Optional local subject image for verifying nonconstant U2Net OBS alpha")
if(SIGHT_U2NET_MODEL)
    if(NOT EXISTS "${SIGHT_U2NET_MODEL}")
        message(FATAL_ERROR "U2Net model does not exist: ${SIGHT_U2NET_MODEL}")
    endif()
    add_executable(sight_u2net_model_test
        tests/u2net_inference_test.cpp src/core/segmentation_processor.cpp
        src/common/sight-log.cpp)
    target_compile_definitions(sight_u2net_model_test PRIVATE USE_ONNX_MEDIAPIPE)
    target_include_directories(sight_u2net_model_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_ORT_INCLUDE}" "${SIGHT_OBS_INCLUDE}")
    target_link_libraries(sight_u2net_model_test PRIVATE "${SIGHT_ORT_LIBRARY}"
        "${SIGHT_OBS_LIBRARY}")
    set_target_properties(sight_u2net_model_test PROPERTIES
        BUILD_RPATH "${SIGHT_ORT_LIB_DIR}")
    add_test(NAME sight_u2net_real_inference
        COMMAND sight_u2net_model_test "${SIGHT_U2NET_MODEL}")

    # Manual CPU performance comparison; intentionally not a timed CTest
    # assertion because shared CI machines have variable CPU availability.
    add_executable(sight_u2net_benchmark tests/benchmark_u2net_threads.cpp)
    target_include_directories(sight_u2net_benchmark PRIVATE "${SIGHT_ORT_INCLUDE}")
    target_link_libraries(sight_u2net_benchmark PRIVATE "${SIGHT_ORT_LIBRARY}")
    set_target_properties(sight_u2net_benchmark PROPERTIES
        BUILD_RPATH "${SIGHT_ORT_LIB_DIR}")
endif()

add_executable(sight_inference_health_test tests/inference-health-test.cpp)
add_test(NAME sight_inference_health_unit COMMAND sight_inference_health_test)
add_executable(sight_flow_matte_test tests/flow-matte-test.cpp)
target_include_directories(sight_flow_matte_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME sight_flow_matte_unit COMMAND sight_flow_matte_test)
add_executable(sight_depth_temporal_test tests/depth_temporal_policy_test.cpp)
target_include_directories(sight_depth_temporal_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME sight_depth_temporal_policy COMMAND sight_depth_temporal_test)

set(SIGHT_DEPTH_MODEL "" CACHE FILEPATH
    "Pinned Depth Anything V2 small ONNX float32 model for real model/OBS tests")
if(SIGHT_DEPTH_MODEL)
    if(NOT EXISTS "${SIGHT_DEPTH_MODEL}")
        message(FATAL_ERROR "Depth model missing: ${SIGHT_DEPTH_MODEL}")
    endif()
    if(SIGHT_XVFB_RUN)
        add_executable(sight_backend_status_test tests/verify-backend-status.cpp)
        target_include_directories(sight_backend_status_test PRIVATE "${SIGHT_OBS_INCLUDE}")
        target_link_libraries(sight_backend_status_test PRIVATE "${SIGHT_OBS_LIBRARY}" pthread)
        add_test(NAME sight_backend_status_real_obs
            COMMAND "${SIGHT_XVFB_RUN}" -a $<TARGET_FILE:sight_backend_status_test>
                $<TARGET_FILE:live-latent-lab-sight> "${CMAKE_CURRENT_SOURCE_DIR}/data" "${SIGHT_DEPTH_MODEL}")
        set_tests_properties(sight_backend_status_real_obs PROPERTIES TIMEOUT 35
            ENVIRONMENT "SIGHT_ONNX_DEVICE=cpu"
            ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:${CMAKE_CURRENT_BINARY_DIR}/semantic-test-runtime")
    endif()
    add_executable(sight_depth_model_test
        tests/onnx_depth_inference_test.cpp src/core/onnx_depth_processor.cpp)
    target_include_directories(sight_depth_model_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_ORT_INCLUDE}")
    target_link_libraries(sight_depth_model_test PRIVATE "${SIGHT_ORT_LIBRARY}")
    set_target_properties(sight_depth_model_test PROPERTIES
        BUILD_RPATH "${SIGHT_ORT_LIB_DIR}")
    add_test(NAME sight_depth_real_inference
        COMMAND sight_depth_model_test "${SIGHT_DEPTH_MODEL}")
endif()

set(SIGHT_ANIMEGAN_MODEL "" CACHE FILEPATH
    "Optional local AnimeGAN ONNX model for real-model CTest")
if(SIGHT_ANIMEGAN_MODEL)
    if(NOT EXISTS "${SIGHT_ANIMEGAN_MODEL}")
        message(FATAL_ERROR "AnimeGAN model does not exist: ${SIGHT_ANIMEGAN_MODEL}")
    endif()
    add_test(NAME sight_animegan_real_inference
        COMMAND sight_animegan_model_test "${SIGHT_ANIMEGAN_MODEL}")

    if(SIGHT_OBS_LIBRARY)
        find_program(SIGHT_XVFB_RUN xvfb-run)
        if(SIGHT_XVFB_RUN)
            add_executable(sight_obs_effect_render_smoke
                tests/obs_sight_render_smoke.cpp)
            target_include_directories(sight_obs_effect_render_smoke
                PRIVATE "${SIGHT_OBS_INCLUDE}")
            target_link_libraries(sight_obs_effect_render_smoke
                PRIVATE "${SIGHT_OBS_LIBRARY}" pthread)
            if(SIGHT_HAVE_OCR_EDITOR)
                target_link_libraries(sight_obs_effect_render_smoke PRIVATE Qt6::Widgets)
                target_compile_definitions(sight_obs_effect_render_smoke PRIVATE SIGHT_HAVE_OCR_EDITOR)
            endif()
            foreach(case IN ITEMS square landscape portrait misaligned)
                add_test(NAME sight_animegan_obs_${case}
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_ANIMEGAN_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_animegan_obs_${case}
                    PROPERTIES TIMEOUT 40)
            endforeach()
            set_tests_properties(sight_animegan_obs_landscape PROPERTIES
                ENVIRONMENT "SIGHT_SMOKE_WIDTH=640;SIGHT_SMOKE_HEIGHT=360")
            set_tests_properties(sight_animegan_obs_portrait PROPERTIES
                ENVIRONMENT "SIGHT_SMOKE_WIDTH=750;SIGHT_SMOKE_HEIGHT=1000")
            set_tests_properties(sight_animegan_obs_misaligned PROPERTIES
                ENVIRONMENT "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=342")
            add_test(NAME sight_animegan_obs_high_detail
                COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                    "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                    "$<TARGET_FILE:live-latent-lab-sight>"
                    "${SIGHT_ANIMEGAN_MODEL}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/data")
            set_tests_properties(sight_animegan_obs_high_detail PROPERTIES
                TIMEOUT 40
                ENVIRONMENT "SIGHT_SMOKE_WIDTH=750;SIGHT_SMOKE_HEIGHT=1000;SIGHT_SMOKE_SKETCH_SIZE=768")
            add_test(NAME sight_animegan_obs_1024_detail
                COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                    "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                    "$<TARGET_FILE:live-latent-lab-sight>"
                    "${SIGHT_ANIMEGAN_MODEL}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/data")
            set_tests_properties(sight_animegan_obs_1024_detail PROPERTIES
                TIMEOUT 70
                ENVIRONMENT "SIGHT_SMOKE_WIDTH=750;SIGHT_SMOKE_HEIGHT=1000;SIGHT_SMOKE_SKETCH_SIZE=1024")
            add_test(NAME sight_animegan_obs_capture_gate
                COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                    "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                    "$<TARGET_FILE:live-latent-lab-sight>"
                    "${SIGHT_ANIMEGAN_MODEL}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/data")
            set_tests_properties(sight_animegan_obs_capture_gate PROPERTIES
                TIMEOUT 40
                ENVIRONMENT "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_CAPTURE_GATE=1")

            add_test(NAME sight_animegan_obs_buffer_reuse
                COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                    "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                    "$<TARGET_FILE:live-latent-lab-sight>"
                    "${SIGHT_ANIMEGAN_MODEL}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/data")
            set_tests_properties(sight_animegan_obs_buffer_reuse PROPERTIES
                TIMEOUT 50 ENVIRONMENT "SIGHT_SMOKE_WIDTH=320;SIGHT_SMOKE_HEIGHT=180;SIGHT_SMOKE_SKETCH_SIZE=256;SIGHT_SMOKE_SKETCH_REUSE=1")

            # Explicit local weights and portrait fixture: no downloads in CTest.
            set(SIGHT_CUTOUT_PORTRAIT "" CACHE FILEPATH "512x512 portrait fixture for cutout model qualification")
            set(SIGHT_U2NETP_MODEL "" CACHE FILEPATH "Local U2NetP ONNX")
            set(SIGHT_MODNET_MODEL "" CACHE FILEPATH "Local Xenova MODNet float32 ONNX")
            set(SIGHT_LONGPIPE_MODEL "" CACHE FILEPATH "Local Longpipe refined ONNX")
            set(SIGHT_LONGPIPE_FLOW_MODEL "" CACHE FILEPATH "Matching local Longpipe optical-flow ONNX")
            set(SIGHT_LONGPIPE_FLOW_MOVED_IMAGE "" CACHE FILEPATH "Optional translated 512x512 portrait fixture")
            if(SIGHT_U2NETP_MODEL)
                add_executable(sight_cutout_device_test tests/verify-cutout-device.cpp)
                target_include_directories(sight_cutout_device_test PRIVATE "${SIGHT_OBS_INCLUDE}")
                target_link_libraries(sight_cutout_device_test PRIVATE "${SIGHT_OBS_LIBRARY}" pthread)
                add_test(NAME sight_cutout_device_real_obs
                    COMMAND "${SIGHT_XVFB_RUN}" -a $<TARGET_FILE:sight_cutout_device_test>
                        $<TARGET_FILE:live-latent-lab-sight> "${CMAKE_CURRENT_SOURCE_DIR}/data" "${SIGHT_U2NETP_MODEL}" cpu)
                set_tests_properties(sight_cutout_device_real_obs PROPERTIES TIMEOUT 60
                    ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:${CMAKE_CURRENT_BINARY_DIR}/semantic-test-runtime")
            endif()
            if(SIGHT_CUTOUT_PORTRAIT)
                find_file(SIGHT_CUTOUT_IMAGE_MODULE NAMES image-source.so
                    PATHS /usr/local/lib/obs-plugins /usr/lib/x86_64-linux-gnu/obs-plugins NO_DEFAULT_PATH)
                if(NOT EXISTS "${SIGHT_CUTOUT_PORTRAIT}" OR NOT SIGHT_CUTOUT_IMAGE_MODULE)
                    message(FATAL_ERROR "Cutout portrait qualification requires the fixture and OBS image-source")
                endif()
                foreach(family IN ITEMS U2NETP MODNET LONGPIPE)
                    if(SIGHT_${family}_MODEL)
                        if(NOT EXISTS "${SIGHT_${family}_MODEL}")
                            message(FATAL_ERROR "Missing cutout model: ${SIGHT_${family}_MODEL}")
                        endif()
                        if(family STREQUAL "U2NETP")
                            set(family_id 0)
                        elseif(family STREQUAL "MODNET")
                            set(family_id 1)
                        else()
                            set(family_id 2)
                        endif()
                        string(TOLOWER "${family}" family_name)
                        add_test(NAME sight_cutout_${family_name}_real_obs
                            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                                "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                                "$<TARGET_FILE:live-latent-lab-sight>"
                                "${SIGHT_${family}_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
                        set_tests_properties(sight_cutout_${family_name}_real_obs PROPERTIES TIMEOUT 60
                            ENVIRONMENT "SIGHT_ONNX_DEVICE=cpu;SIGHT_SMOKE_EXPECT_ONNX_DEVICE=cpu;SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_u2net_filter;SIGHT_SMOKE_CUTOUT_MODEL=${family_id};SIGHT_SMOKE_CUTOUT_SHARED=1;SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_CUTOUT_IMAGE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_CUTOUT_PORTRAIT};SIGHT_SMOKE_EXPECT_FOREGROUND=1")
                    endif()
                endforeach()
                if(SIGHT_LONGPIPE_MODEL AND SIGHT_LONGPIPE_FLOW_MODEL)
                    if(NOT EXISTS "${SIGHT_LONGPIPE_FLOW_MODEL}")
                        message(FATAL_ERROR "Missing Longpipe flow model")
                    endif()
                    foreach(kind IN ITEMS flow flow_mismatch)
                        if(kind STREQUAL "flow")
                            set(flow_file "${SIGHT_LONGPIPE_FLOW_MODEL}")
                            set(expected_flow "stationary stabilization")
                        else()
                            set(flow_file "${SIGHT_LONGPIPE_MODEL}")
                            set(expected_flow "Flow unavailable")
                        endif()
                        add_test(NAME sight_cutout_longpipe_${kind}_real_obs
                            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                                "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                                "$<TARGET_FILE:live-latent-lab-sight>"
                                "${SIGHT_LONGPIPE_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
                        set_tests_properties(sight_cutout_longpipe_${kind}_real_obs PROPERTIES TIMEOUT 60
                            ENVIRONMENT "SIGHT_ONNX_DEVICE=cpu;SIGHT_SMOKE_EXPECT_ONNX_DEVICE=cpu;SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_u2net_filter;SIGHT_SMOKE_CUTOUT_MODEL=2;SIGHT_SMOKE_CUTOUT_FLOW=${flow_file};SIGHT_SMOKE_CUTOUT_FLOW_EXPECT=${expected_flow};SIGHT_SMOKE_CUTOUT_SHARED=1;SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_CUTOUT_IMAGE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_CUTOUT_PORTRAIT};SIGHT_SMOKE_EXPECT_FOREGROUND=1;SIGHT_SMOKE_HOLD_MS=1800")
                    endforeach()
                    if(SIGHT_LONGPIPE_FLOW_MOVED_IMAGE)
                        if(NOT EXISTS "${SIGHT_LONGPIPE_FLOW_MOVED_IMAGE}")
                            message(FATAL_ERROR "Missing translated portrait fixture")
                        endif()
                        set_property(TEST sight_cutout_longpipe_flow_real_obs APPEND PROPERTY
                            ENVIRONMENT "SIGHT_SMOKE_CUTOUT_FLOW_MOVE=${SIGHT_LONGPIPE_FLOW_MOVED_IMAGE}")
                    endif()
                endif()
            endif()

            if(SIGHT_U2NET_MODEL)
                add_test(NAME sight_u2net_obs_cutout
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_U2NET_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_u2net_obs_cutout PROPERTIES TIMEOUT 60
                    ENVIRONMENT "SIGHT_SMOKE_WIDTH=750;SIGHT_SMOKE_HEIGHT=1000;SIGHT_SMOKE_EFFECT=sight_u2net_filter")
                add_test(NAME sight_u2net_obs_buffer_reuse
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_U2NET_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_u2net_obs_buffer_reuse PROPERTIES TIMEOUT 60
                    ENVIRONMENT "SIGHT_SMOKE_WIDTH=320;SIGHT_SMOKE_HEIGHT=180;SIGHT_SMOKE_EFFECT=sight_u2net_filter;SIGHT_SMOKE_CUTOUT_REUSE=1")
                add_test(NAME sight_u2net_obs_refinement
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_U2NET_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_u2net_obs_refinement PROPERTIES TIMEOUT 60
                    ENVIRONMENT "SIGHT_SMOKE_WIDTH=750;SIGHT_SMOKE_HEIGHT=1000;SIGHT_SMOKE_EFFECT=sight_u2net_filter;SIGHT_SMOKE_REFINE_EDGES=1")
                add_test(NAME sight_u2net_obs_matte_controls
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_U2NET_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_u2net_obs_matte_controls PROPERTIES TIMEOUT 60
                    ENVIRONMENT "SIGHT_SMOKE_WIDTH=750;SIGHT_SMOKE_HEIGHT=1000;SIGHT_SMOKE_EFFECT=sight_u2net_filter;SIGHT_SMOKE_REFINE_EDGES=1;SIGHT_SMOKE_MATTE_CONTROLS=1;SIGHT_SMOKE_HOLD_MS=2500")
                if(SIGHT_U2NET_TEST_IMAGE)
                    if(NOT EXISTS "${SIGHT_U2NET_TEST_IMAGE}")
                        message(FATAL_ERROR "Missing U2Net subject test image: ${SIGHT_U2NET_TEST_IMAGE}")
                    endif()
                    find_file(SIGHT_U2NET_IMAGE_MODULE NAMES image-source.so
                        PATHS /usr/local/lib/obs-plugins
                              /usr/lib/x86_64-linux-gnu/obs-plugins
                        NO_DEFAULT_PATH)
                    if(SIGHT_U2NET_IMAGE_MODULE)
                        add_test(NAME sight_u2net_obs_subject
                            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                                "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                                "$<TARGET_FILE:live-latent-lab-sight>"
                                "${SIGHT_U2NET_MODEL}"
                                "${CMAKE_CURRENT_SOURCE_DIR}/data")
                        set_tests_properties(sight_u2net_obs_subject PROPERTIES
                            TIMEOUT 60
                            ENVIRONMENT
                                "SIGHT_SMOKE_WIDTH=750;SIGHT_SMOKE_HEIGHT=1000;SIGHT_SMOKE_EFFECT=sight_u2net_filter;SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_U2NET_IMAGE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_U2NET_TEST_IMAGE};SIGHT_SMOKE_EXPECT_FOREGROUND=1")
                    endif()
                endif()
            endif()

            if(SIGHT_DEPTH_MODEL)
                add_test(NAME sight_depth_obs_camera
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_DEPTH_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_depth_obs_camera PROPERTIES
                    TIMEOUT 90
                    ENVIRONMENT
                      "SIGHT_SMOKE_WIDTH=750;SIGHT_SMOKE_HEIGHT=1000;SIGHT_SMOKE_EFFECT=sight_depth_filter;SIGHT_SMOKE_DEPTH_EDGE=252;SIGHT_SMOKE_DEPTH_CADENCE=4;SIGHT_SMOKE_DEPTH_SMOOTH=65")
                # Verify that an old map cannot remain on screen indefinitely
                # after inference stops. Native output must fail open.
                add_test(NAME sight_depth_obs_expired_passthrough
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_DEPTH_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_depth_obs_expired_passthrough PROPERTIES
                    TIMEOUT 90
                    ENVIRONMENT
                      "SIGHT_SMOKE_WIDTH=128;SIGHT_SMOKE_HEIGHT=128;SIGHT_SMOKE_EFFECT=sight_depth_filter;SIGHT_SMOKE_DEPTH_EDGE=252;SIGHT_SMOKE_DEPTH_MAX_AGE=2;SIGHT_SMOKE_DEPTH_VERIFY_EXPIRY=6500")
                # If the OBS source geometry changes, old depth must not
                # stretch across the resized source before new inference.
                add_test(NAME sight_depth_obs_buffer_reuse
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_DEPTH_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_depth_obs_buffer_reuse PROPERTIES TIMEOUT 90
                    ENVIRONMENT "SIGHT_SMOKE_WIDTH=320;SIGHT_SMOKE_HEIGHT=180;SIGHT_SMOKE_EFFECT=sight_depth_filter;SIGHT_SMOKE_DEPTH_REUSE=1;SIGHT_SMOKE_DEPTH_REUSE_EDGE=1;SIGHT_SMOKE_DEPTH_REJECT_STALE=1")
                add_test(NAME sight_depth_obs_resize_passthrough
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_DEPTH_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_depth_obs_resize_passthrough PROPERTIES
                    TIMEOUT 90
                    ENVIRONMENT
                      "SIGHT_SMOKE_WIDTH=128;SIGHT_SMOKE_HEIGHT=128;SIGHT_SMOKE_EFFECT=sight_depth_filter;SIGHT_SMOKE_DEPTH_EDGE=252;SIGHT_SMOKE_DEPTH_VERIFY_RESIZE=1")
                # Exercise genuine Depth Anything and OBS rendering with
                # the optional appearance-change-aware blend enabled.
                add_test(NAME sight_depth_obs_change_aware
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_DEPTH_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_depth_obs_change_aware PROPERTIES
                    TIMEOUT 90
                    ENVIRONMENT
                      "SIGHT_SMOKE_WIDTH=640;SIGHT_SMOKE_HEIGHT=360;SIGHT_SMOKE_EFFECT=sight_depth_filter;SIGHT_SMOKE_DEPTH_EDGE=252;SIGHT_SMOKE_DEPTH_CADENCE=1;SIGHT_SMOKE_DEPTH_SMOOTH=80;SIGHT_SMOKE_DEPTH_REJECT_STALE=1;SIGHT_SMOKE_HOLD_MS=1800")
            endif()
            if(SIGHT_DEPTH_MODEL)
                add_test(NAME sight_depth_obs_fog
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_DEPTH_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_depth_obs_fog PROPERTIES
                    TIMEOUT 90
                    ENVIRONMENT
                      "SIGHT_SMOKE_WIDTH=640;SIGHT_SMOKE_HEIGHT=360;SIGHT_SMOKE_EFFECT=sight_depth_filter;SIGHT_SMOKE_DEPTH_FOG=50;SIGHT_SMOKE_DEPTH_CADENCE=8;SIGHT_SMOKE_DEPTH_SMOOTH=80")
            endif()
            if(SIGHT_IMAGE_SOURCE_MODULE)
                add_test(NAME sight_animegan_obs_real_image
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_ANIMEGAN_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_animegan_obs_real_image PROPERTIES
                    TIMEOUT 40
                    ENVIRONMENT "SIGHT_SMOKE_WIDTH=750;SIGHT_SMOKE_HEIGHT=1000;SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE}")
                if(SIGHT_DEPTH_MODEL)
                    add_test(NAME sight_depth_obs_image
                        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                            "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                            "$<TARGET_FILE:live-latent-lab-sight>"
                            "${SIGHT_DEPTH_MODEL}"
                            "${CMAKE_CURRENT_SOURCE_DIR}/data")
                    set_tests_properties(sight_depth_obs_image PROPERTIES
                        TIMEOUT 90
                        ENVIRONMENT
                          "SIGHT_SMOKE_WIDTH=750;SIGHT_SMOKE_HEIGHT=1000;SIGHT_SMOKE_EFFECT=sight_depth_filter;SIGHT_SMOKE_DEPTH_EDGE=336;SIGHT_SMOKE_DEPTH_CADENCE=2;SIGHT_SMOKE_DEPTH_SMOOTH=45;SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE}")
                endif()
                if(SIGHT_MEDIAPIPE_LIB AND SIGHT_MEDIAPIPE_TEST_IMAGE)
                    add_test(NAME sight_value_router_obs_face_gate
                        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                            "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                            "$<TARGET_FILE:live-latent-lab-sight>"
                            "${SIGHT_MEDIAPIPE_FACE_MODEL}"
                            "${CMAKE_CURRENT_SOURCE_DIR}/data")
                    set_tests_properties(sight_value_router_obs_face_gate PROPERTIES
                        TIMEOUT 90
                        ENVIRONMENT
                            "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_value_router_filter;SIGHT_SMOKE_ROUTER_GATE=1;SIGHT_SMOKE_ROUTER_SIGNAL=1;SIGHT_SMOKE_EXPECT_TASK=face;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_MEDIAPIPE_TEST_IMAGE}")
                endif()
                if(SIGHT_MEDIAPIPE_LIB AND SIGHT_MEDIAPIPE_TEST_IMAGE)
                    foreach(kind IN ITEMS face body)
                        if(kind STREQUAL "face")
                            set(model "${SIGHT_MEDIAPIPE_FACE_MODEL}")
                        else()
                            set(model "${SIGHT_MEDIAPIPE_POSE_MODEL}")
                        endif()
                        add_test(NAME sight_tasks_obs_${kind}
                            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                                "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                                "$<TARGET_FILE:live-latent-lab-sight>"
                                "${model}"
                                "${CMAKE_CURRENT_SOURCE_DIR}/data")
                        set_tests_properties(sight_tasks_obs_${kind} PROPERTIES
                            TIMEOUT 75
                            ENVIRONMENT
                                "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_unified_tracking_filter;SIGHT_SMOKE_TRACK_MODE=${kind};SIGHT_SMOKE_TRACK_EDGE=512;SIGHT_SMOKE_EXPECT_TASK=${kind};SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_MEDIAPIPE_TEST_IMAGE}")
                    endforeach()
                if(SIGHT_MEDIAPIPE_LIB AND SIGHT_GESTURE_MODEL AND SIGHT_GESTURE_TEST_IMAGE)
                    add_test(NAME sight_tasks_obs_hand
                        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                            "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                            "$<TARGET_FILE:live-latent-lab-sight>"
                            "${SIGHT_GESTURE_MODEL}"
                            "${CMAKE_CURRENT_SOURCE_DIR}/data")
                    set_tests_properties(sight_tasks_obs_hand PROPERTIES
                        TIMEOUT 90
                        ENVIRONMENT
                            "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_unified_tracking_filter;SIGHT_SMOKE_TRACK_MODE=hand;SIGHT_SMOKE_EXPECT_TASK=hand;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_GESTURE_TEST_IMAGE}")
                    add_test(NAME sight_signals_obs_hand_receiver
                        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                            "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                            "$<TARGET_FILE:live-latent-lab-sight>"
                            "${SIGHT_GESTURE_MODEL}"
                            "${CMAKE_CURRENT_SOURCE_DIR}/data")
                    set_tests_properties(sight_signals_obs_hand_receiver PROPERTIES
                        TIMEOUT 90
                        ENVIRONMENT
                            "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_tracking_receiver_filter;SIGHT_SMOKE_SIGNAL_PRODUCER=hand;SIGHT_SMOKE_SIGNAL_CHANNEL=hand;SIGHT_SMOKE_EXPECT_TASK=hand;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_GESTURE_TEST_IMAGE}")
                    # Reproduces the live webcam bug: the native source is
                    # 640x360, but the model observes a 512x288 inference
                    # frame. Normalized landmarks must retain 640x360 source
                    # metadata so downstream shader consumers can validate geometry.
                    # Test-only image is generated from the provided real
                    # hand fixture; no test image committed to source control.
                    find_program(SIGHT_TEST_CONVERT_BIN NAMES magick convert)
                    if(SIGHT_TEST_CONVERT_BIN)
                        set(SIGHT_HAND_WIDE_IMAGE
                            "${CMAKE_CURRENT_BINARY_DIR}/sight-hand-640x360.png")
                        execute_process(
                            COMMAND "${SIGHT_TEST_CONVERT_BIN}"
                                "${SIGHT_GESTURE_TEST_IMAGE}"
                                -resize 360x360 -background white
                                -gravity center -extent 640x360
                                "${SIGHT_HAND_WIDE_IMAGE}"
                            RESULT_VARIABLE SIGHT_HAND_WIDE_RESULT)
                        if(NOT SIGHT_HAND_WIDE_RESULT EQUAL 0)
                            message(FATAL_ERROR
                                "Could not prepare real-hand 16:9 test image")
                        endif()
                        add_test(NAME sight_signals_obs_hand_native_wide
                            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                                "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                                "$<TARGET_FILE:live-latent-lab-sight>"
                                "${SIGHT_GESTURE_MODEL}"
                                "${CMAKE_CURRENT_SOURCE_DIR}/data")
                        set_tests_properties(sight_signals_obs_hand_native_wide PROPERTIES
                            TIMEOUT 90
                            ENVIRONMENT
                                "SIGHT_SMOKE_WIDTH=640;SIGHT_SMOKE_HEIGHT=360;SIGHT_SMOKE_EFFECT=sight_tracking_receiver_filter;SIGHT_SMOKE_SIGNAL_PRODUCER=hand;SIGHT_SMOKE_SIGNAL_CHANNEL=hand;SIGHT_SMOKE_EXPECT_TASK=hand;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_HAND_WIDE_IMAGE}")
                    endif()
                    add_test(NAME sight_value_router_obs_hand_pinch
                        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                            "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                            "$<TARGET_FILE:live-latent-lab-sight>"
                            "${SIGHT_GESTURE_MODEL}"
                            "${CMAKE_CURRENT_SOURCE_DIR}/data")
                    set_tests_properties(sight_value_router_obs_hand_pinch PROPERTIES
                        TIMEOUT 90
                        ENVIRONMENT
                            "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_value_router_filter;SIGHT_SMOKE_ROUTER_SIGNAL=hand;SIGHT_SMOKE_SIGNAL_PRODUCER=hand;SIGHT_SMOKE_EXPECT_TASK=hand;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_GESTURE_TEST_IMAGE}")
                endif()
                add_test(NAME sight_value_router_obs_wave
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_ANIMEGAN_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_value_router_obs_wave PROPERTIES
                    TIMEOUT 90
                    ENVIRONMENT
                        "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_value_router_filter;SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE}")
                add_test(NAME sight_value_router_obs_restore
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_ANIMEGAN_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_value_router_obs_restore PROPERTIES
                    TIMEOUT 90
                    ENVIRONMENT
                        "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_value_router_filter;SIGHT_SMOKE_ROUTER_RESTORE=1;SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE}")
                add_test(NAME sight_value_router_obs_gate
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_ANIMEGAN_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_value_router_obs_gate PROPERTIES
                    TIMEOUT 90
                    ENVIRONMENT
                        "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_value_router_filter;SIGHT_SMOKE_ROUTER_GATE=1;SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE}")
                if(SIGHT_MEDIAPIPE_LIB AND SIGHT_MEDIAPIPE_TEST_IMAGE)
                    add_test(NAME sight_value_router_obs_discovery
                        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                            "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                            "$<TARGET_FILE:live-latent-lab-sight>"
                            "${SIGHT_MEDIAPIPE_FACE_MODEL}"
                            "${CMAKE_CURRENT_SOURCE_DIR}/data")
                    set_tests_properties(sight_value_router_obs_discovery PROPERTIES
                        TIMEOUT 90
                        ENVIRONMENT "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_value_router_filter;SIGHT_SMOKE_ROUTER_SIGNAL=1;SIGHT_SMOKE_ROUTER_DISCOVERY=1;SIGHT_SMOKE_EXPECT_TASK=face;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_MEDIAPIPE_TEST_IMAGE}")
                endif()
                if(SIGHT_MEDIAPIPE_LIB AND SIGHT_MEDIAPIPE_TEST_IMAGE)
                    add_test(NAME sight_value_router_obs_face
                        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                            "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                            "$<TARGET_FILE:live-latent-lab-sight>"
                            "${SIGHT_MEDIAPIPE_FACE_MODEL}"
                            "${CMAKE_CURRENT_SOURCE_DIR}/data")
                    set_tests_properties(sight_value_router_obs_face PROPERTIES
                        TIMEOUT 90
                        ENVIRONMENT
                            "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_value_router_filter;SIGHT_SMOKE_ROUTER_SIGNAL=1;SIGHT_SMOKE_EXPECT_TASK=face;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_MEDIAPIPE_TEST_IMAGE}")
                endif()
                if(SIGHT_MEDIAPIPE_LIB AND SIGHT_MEDIAPIPE_TEST_IMAGE)
                    foreach(kind IN ITEMS face body)
                        if(kind STREQUAL "face")
                            set(model "${SIGHT_MEDIAPIPE_FACE_MODEL}")
                        else()
                            set(model "${SIGHT_MEDIAPIPE_POSE_MODEL}")
                        endif()
                        add_test(NAME sight_signals_obs_${kind}_receiver
                            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                                "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                                "$<TARGET_FILE:live-latent-lab-sight>"
                                "${model}"
                                "${CMAKE_CURRENT_SOURCE_DIR}/data")
                        set_tests_properties(sight_signals_obs_${kind}_receiver PROPERTIES
                            TIMEOUT 90
                            ENVIRONMENT
                                "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_tracking_receiver_filter;SIGHT_SMOKE_SIGNAL_PRODUCER=${kind};SIGHT_SMOKE_SIGNAL_CHANNEL=${kind};SIGHT_SMOKE_EXPECT_TASK=${kind};SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_MEDIAPIPE_TEST_IMAGE}")
                    endforeach()
                endif()
                endif()
            else()
                message(STATUS "OBS image-source module unavailable; image-source regression skipped")
            endif()
        else()
            message(STATUS "xvfb-run unavailable; OBS render smoke not registered")
        endif()
    endif()
endif()
# Real receiver publication cadence on static image sources. Model/fixture
# paths stay explicit and every registered case runs the current module.
if(SIGHT_MEDIAPIPE_123_LIB AND SIGHT_IMAGE_SOURCE_MODULE AND
   SIGHT_XVFB_RUN AND TARGET sight_obs_effect_render_smoke)
    foreach(kind IN ITEMS face body hand holistic)
        if(kind STREQUAL "face")
            set(static_model "${SIGHT_MEDIAPIPE_FACE_MODEL}")
            set(static_image "${SIGHT_MEDIAPIPE_TEST_IMAGE}")
            set(static_channel "face")
        elseif(kind STREQUAL "body")
            set(static_model "${SIGHT_MEDIAPIPE_POSE_MODEL}")
            set(static_image "${SIGHT_MEDIAPIPE_TEST_IMAGE}")
            set(static_channel "body")
        elseif(kind STREQUAL "hand")
            set(static_model "${SIGHT_GESTURE_MODEL}")
            set(static_image "${SIGHT_GESTURE_TEST_IMAGE}")
            set(static_channel "hand")
        else()
            set(static_model "${SIGHT_HOLISTIC_MODEL}")
            # A face fixture suffices for observing Holistic face
            # publications; it need not contain a visible hand.
            set(static_image "${SIGHT_MEDIAPIPE_TEST_IMAGE}")
            set(static_channel "holistic/face")
        endif()
        if(static_model AND static_image)
            add_test(NAME sight_tasks_123_obs_static_${kind}
                COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                    "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                    "$<TARGET_FILE:live-latent-lab-sight>"
                    "${static_model}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
            set_tests_properties(sight_tasks_123_obs_static_${kind} PROPERTIES
                TIMEOUT 90
                ENVIRONMENT
                    "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_tracking_receiver_filter;SIGHT_SMOKE_SIGNAL_PRODUCER=${kind};SIGHT_SMOKE_SIGNAL_CHANNEL=${static_channel};SIGHT_SMOKE_STATIC_TRACKING=1;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_123_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${static_image}")
        endif()
    endforeach()
endif()

# Full OBS -> MediaPipe -> synchronized 3-channel bus -> read-only OBS
# procedure acceptance in an isolated Xvfb process. Never affect live OBS.
if(SIGHT_MEDIAPIPE_123_LIB AND SIGHT_HOLISTIC_MODEL AND
   SIGHT_HOLISTIC_TEST_IMAGE AND SIGHT_IMAGE_SOURCE_MODULE AND
   SIGHT_XVFB_RUN AND TARGET sight_obs_effect_render_smoke)
    add_test(NAME sight_holistic_123_obs_three_snapshots
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_obs_effect_render_smoke>"
            "$<TARGET_FILE:live-latent-lab-sight>"
            "${SIGHT_HOLISTIC_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_holistic_123_obs_three_snapshots PROPERTIES
        TIMEOUT 90
        ENVIRONMENT
            "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_tracking_receiver_filter;SIGHT_SMOKE_SIGNAL_PRODUCER=holistic;SIGHT_SMOKE_SIGNAL_CHANNEL=holistic/face;SIGHT_SMOKE_HOLISTIC_TRIPLE=1;SIGHT_SMOKE_EXPECT_TASK=holistic;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_123_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_HOLISTIC_TEST_IMAGE}")
endif()

# OBS model -> one Holistic graph -> actual same-source overlay pixels.
# Off must pass the camera through; all/on and per-kind use the same
# user-supplied model and native library with actual inferred landmarks.
if(SIGHT_MEDIAPIPE_123_LIB AND SIGHT_HOLISTIC_MODEL AND
   SIGHT_HOLISTIC_TEST_IMAGE AND SIGHT_IMAGE_SOURCE_MODULE AND
   SIGHT_XVFB_RUN AND TARGET sight_obs_effect_render_smoke)
    foreach(overlay_variant IN ITEMS off all face body hands toggle
                                     stability_raw stability_stable stability_strict
                                     person_mask_on person_mask_toggle)
        if(overlay_variant STREQUAL "person_mask_on")
            set(overlay_args "SIGHT_SMOKE_HOLISTIC_OVERLAY=1"
                "SIGHT_SMOKE_HOLISTIC_PERSON_MASK=1")
        elseif(overlay_variant STREQUAL "person_mask_toggle")
            set(overlay_args "SIGHT_SMOKE_HOLISTIC_OVERLAY=1"
                "SIGHT_SMOKE_HOLISTIC_MASK_TOGGLE=1")
        elseif(overlay_variant STREQUAL "stability_raw")
            set(overlay_args "SIGHT_SMOKE_HOLISTIC_OVERLAY=1"
                "SIGHT_SMOKE_STABILITY_MODE=0")
        elseif(overlay_variant STREQUAL "stability_stable")
            set(overlay_args "SIGHT_SMOKE_HOLISTIC_OVERLAY=1"
                "SIGHT_SMOKE_STABILITY_MODE=3")
        elseif(overlay_variant STREQUAL "stability_strict")
            set(overlay_args "SIGHT_SMOKE_HOLISTIC_OVERLAY=1"
                "SIGHT_SMOKE_CONFIDENCE=75" "SIGHT_SMOKE_DISABLE_FADE=1")
        elseif(overlay_variant STREQUAL "toggle")
            set(overlay_args "SIGHT_SMOKE_HOLISTIC_LIVE_TOGGLE=1")
        elseif(overlay_variant STREQUAL "off")
            set(overlay_args "SIGHT_SMOKE_EXPECT_PASSTHROUGH=1")
        else()
            set(overlay_args "SIGHT_SMOKE_HOLISTIC_OVERLAY=1")
            if(overlay_variant STREQUAL "all")
                list(APPEND overlay_args "SIGHT_SMOKE_HOLISTIC_OVERLAY_MODE=0")
            elseif(overlay_variant STREQUAL "face")
                list(APPEND overlay_args "SIGHT_SMOKE_HOLISTIC_OVERLAY_MODE=1")
            elseif(overlay_variant STREQUAL "body")
                list(APPEND overlay_args "SIGHT_SMOKE_HOLISTIC_OVERLAY_MODE=2")
            else()
                list(APPEND overlay_args "SIGHT_SMOKE_HOLISTIC_OVERLAY_MODE=3")
            endif()
        endif()
        add_test(NAME sight_holistic_123_obs_overlay_${overlay_variant}
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                "$<TARGET_FILE:live-latent-lab-sight>"
                "${SIGHT_HOLISTIC_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
        set_tests_properties(sight_holistic_123_obs_overlay_${overlay_variant} PROPERTIES
            TIMEOUT 90
            ENVIRONMENT
               "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_unified_tracking_filter;SIGHT_SMOKE_TRACK_MODE=holistic;SIGHT_SMOKE_TRACK_EDGE=512;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_123_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_HOLISTIC_TEST_IMAGE};${overlay_args}")
    endforeach()
endif()

if(SIGHT_OBJECT_MODEL AND SIGHT_MEDIAPIPE_123_LIB AND
   SIGHT_OBJECT_TEST_IMAGE AND SIGHT_IMAGE_SOURCE_MODULE AND
   TARGET sight_obs_effect_render_smoke)
    foreach(mode IN ITEMS overlay on_source_only)
        add_test(NAME sight_object_123_obs_${mode}
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                "$<TARGET_FILE:live-latent-lab-sight>"
                "${SIGHT_OBJECT_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
        if(mode STREQUAL "overlay")
            set(object_extra "SIGHT_SMOKE_EXPECT_OBJECT=1")
        else()
            set(object_extra "SIGHT_SMOKE_OBJECT_OVERLAY_OFF=1;SIGHT_SMOKE_EXPECT_PASSTHROUGH=1")
        endif()
        set_tests_properties(sight_object_123_obs_${mode} PROPERTIES
            TIMEOUT 90
            ENVIRONMENT
            "SIGHT_SMOKE_WIDTH=${SIGHT_OBJECT_TEST_WIDTH};SIGHT_SMOKE_HEIGHT=${SIGHT_OBJECT_TEST_HEIGHT};SIGHT_SMOKE_EFFECT=sight_unified_tracking_filter;SIGHT_SMOKE_TRACK_MODE=object;SIGHT_SMOKE_TRACK_EDGE=512;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_123_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_OBJECT_TEST_IMAGE};${object_extra}")
    endforeach()
endif()

message(STATUS "Sight validated-effects build: PortraitSketch, U2Net, Depth, Unified face/body/hand/holistic/object tracking; no diffusion/CUDA dependency")
message(STATUS "ONNX Runtime: ${SIGHT_ORT_LIBRARY}")

# The language data is deliberately external to the plugin package. A
# configured real model enables reproducible OCR inference/render tests.
# Geometry/output contracts are model-free and do not need Tesseract.
add_executable(sight_ocr_output_test tests/ocr_output_test.cpp)
target_include_directories(sight_ocr_output_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME sight_ocr_output_unit COMMAND sight_ocr_output_test)

set(SIGHT_OCR_TESSDATA_DIR "" CACHE PATH "Directory with eng.traineddata for OCR verification")
if(SIGHT_HAVE_OCR AND SIGHT_OCR_TESSDATA_DIR AND
   EXISTS "${SIGHT_OCR_TESSDATA_DIR}/eng.traineddata" AND SIGHT_PYTHON3_EXECUTABLE)
    add_executable(sight_text_ocr_test tests/text_ocr_test.cpp src/core/text_ocr.cpp)
    target_include_directories(sight_text_ocr_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_TESSERACT_INCLUDE}")
    target_link_libraries(sight_text_ocr_test PRIVATE "${SIGHT_TESSERACT_LIBRARY}")
    add_test(NAME sight_text_ocr_real COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_text_ocr.sh"
        core "$<TARGET_FILE:sight_text_ocr_test>" "${SIGHT_OCR_TESSDATA_DIR}")
    set_tests_properties(sight_text_ocr_real PROPERTIES TIMEOUT 45)
    if(SIGHT_XVFB_RUN AND SIGHT_IMAGE_SOURCE_MODULE AND TARGET sight_obs_effect_render_smoke)
        add_test(NAME sight_text_obfuscation_obs COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_text_ocr.sh"
            obs "$<TARGET_FILE:sight_obs_effect_render_smoke>" "$<TARGET_FILE:live-latent-lab-sight>"
            "${SIGHT_OCR_TESSDATA_DIR}" "${SIGHT_IMAGE_SOURCE_MODULE}")
        set_tests_properties(sight_text_obfuscation_obs PROPERTIES TIMEOUT 60)
        if(SIGHT_HAVE_OCR_EDITOR)
            add_test(NAME sight_ocr_editor_obs COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_text_ocr.sh"
                obs "$<TARGET_FILE:sight_obs_effect_render_smoke>" "$<TARGET_FILE:live-latent-lab-sight>"
                "${SIGHT_OCR_TESSDATA_DIR}" "${SIGHT_IMAGE_SOURCE_MODULE}")
            set_tests_properties(sight_ocr_editor_obs PROPERTIES TIMEOUT 90
                ENVIRONMENT "SIGHT_SMOKE_OCR_EDITOR=1")
        endif()
    endif()
endif()

if(SIGHT_HAVE_DENSE_FLOW)
    add_executable(sight_flow_capture_buffer_test tests/flow_capture_buffer_test.cpp)
    target_include_directories(sight_flow_capture_buffer_test PRIVATE src)
    add_test(NAME sight_flow_capture_buffer_unit COMMAND sight_flow_capture_buffer_test)
    add_executable(sight_dense_motion_test tests/dense_motion_test.cpp src/core/dense_motion.cpp)
    target_include_directories(sight_dense_motion_test PRIVATE src)
    target_include_directories(sight_dense_motion_test SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(sight_dense_motion_test PRIVATE ${OpenCV_LIBS})
    add_test(NAME sight_dense_motion_unit COMMAND sight_dense_motion_test)
    add_executable(sight_camera_stabilizer_test tests/camera_stabilizer_test.cpp)
    target_include_directories(sight_camera_stabilizer_test PRIVATE src)
    add_test(NAME sight_camera_stabilizer_unit COMMAND sight_camera_stabilizer_test)
    if(SIGHT_XVFB_RUN AND SIGHT_IMAGE_SOURCE_MODULE AND SIGHT_PYTHON3_EXECUTABLE AND TARGET sight_obs_effect_render_smoke)
        add_test(NAME sight_spatial_motion_obs COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_spatial_motion_obs.sh"
            "$<TARGET_FILE:sight_obs_effect_render_smoke>" "$<TARGET_FILE:live-latent-lab-sight>" "${SIGHT_IMAGE_SOURCE_MODULE}")
        set_tests_properties(sight_spatial_motion_obs PROPERTIES TIMEOUT 60)
    endif()
endif()

if(SIGHT_ENABLE_RTMW AND SIGHT_RTMW_MODEL AND SIGHT_RTMW_PERSON_DETECTOR
   AND SIGHT_MEDIAPIPE_123_LIB AND SIGHT_MEDIAPIPE_TEST_IMAGE
   AND SIGHT_XVFB_RUN AND SIGHT_IMAGE_SOURCE_MODULE AND TARGET sight_obs_effect_render_smoke)
    add_test(NAME sight_rtmw_obs_gpu
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_obs_effect_render_smoke>" "$<TARGET_FILE:live-latent-lab-sight>"
            "${SIGHT_RTMW_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_rtmw_obs_gpu PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE
        ENVIRONMENT "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_tracking_receiver_filter;SIGHT_SMOKE_SIGNAL_PRODUCER=rtmw;SIGHT_SMOKE_SIGNAL_CHANNEL=wholebody;SIGHT_SMOKE_EXPECT_TASK=rtmw;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_123_LIB};SIGHT_SMOKE_RTMW_DETECTOR=${SIGHT_RTMW_PERSON_DETECTOR};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_MEDIAPIPE_TEST_IMAGE}")
    add_test(NAME sight_rtmw_obs_router_gpu
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_obs_effect_render_smoke>" "$<TARGET_FILE:live-latent-lab-sight>"
            "${SIGHT_RTMW_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_rtmw_obs_router_gpu PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE
        ENVIRONMENT "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_value_router_filter;SIGHT_SMOKE_ROUTER_SIGNAL=rtmw;SIGHT_SMOKE_EXPECT_TASK=rtmw;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_123_LIB};SIGHT_SMOKE_RTMW_DETECTOR=${SIGHT_RTMW_PERSON_DETECTOR};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_MEDIAPIPE_TEST_IMAGE}")

endif()

if(SIGHT_ENABLE_RTMW AND SIGHT_RTMW_MODEL AND SIGHT_YOLOX_MODEL
   AND SIGHT_MEDIAPIPE_TEST_IMAGE AND SIGHT_XVFB_RUN
   AND SIGHT_IMAGE_SOURCE_MODULE AND TARGET sight_obs_effect_render_smoke)
    foreach(kind IN ITEMS receiver router)
        if(kind STREQUAL "receiver")
            set(_yolo_effect "sight_tracking_receiver_filter")
            set(_yolo_receiver "SIGHT_SMOKE_SIGNAL_PRODUCER=rtmw;SIGHT_SMOKE_SIGNAL_CHANNEL=wholebody")
        else()
            set(_yolo_effect "sight_value_router_filter")
            set(_yolo_receiver "SIGHT_SMOKE_ROUTER_SIGNAL=rtmw")
        endif()
        add_test(NAME sight_rtmw_yolox_obs_${kind}
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                "$<TARGET_FILE:sight_obs_effect_render_smoke>" "$<TARGET_FILE:live-latent-lab-sight>"
                "${SIGHT_RTMW_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
        set_tests_properties(sight_rtmw_yolox_obs_${kind} PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE
            ENVIRONMENT "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=${_yolo_effect};${_yolo_receiver};SIGHT_SMOKE_EXPECT_TASK=rtmw;SIGHT_SMOKE_RTMW_DETECTOR=${SIGHT_YOLOX_MODEL};SIGHT_SMOKE_EXPECT_RTMW_BACKEND=yolox-gpu;SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_MEDIAPIPE_TEST_IMAGE}")
    endforeach()
endif()

if(SIGHT_MEDIAPIPE_123_LIB AND SIGHT_MEDIAPIPE_FACE_MODEL AND
   SIGHT_MEDIAPIPE_TEST_IMAGE AND SIGHT_XVFB_RUN AND SIGHT_IMAGE_SOURCE_MODULE
   AND TARGET sight_obs_effect_render_smoke)
    add_test(NAME sight_consumers_obs_discovery
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_obs_effect_render_smoke>" "$<TARGET_FILE:live-latent-lab-sight>"
            "${SIGHT_MEDIAPIPE_FACE_MODEL}" "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(sight_consumers_obs_discovery PROPERTIES TIMEOUT 90
        ENVIRONMENT "SIGHT_SMOKE_WIDTH=512;SIGHT_SMOKE_HEIGHT=512;SIGHT_SMOKE_EFFECT=sight_tracking_receiver_filter;SIGHT_SMOKE_SIGNAL_PRODUCER=face;SIGHT_SMOKE_SIGNAL_CHANNEL=face;SIGHT_SMOKE_CONSUMER_DISCOVERY=1;SIGHT_SMOKE_EXPECT_TASK=face;SIGHT_SMOKE_TASK_RUNTIME=${SIGHT_MEDIAPIPE_123_LIB};SIGHT_SMOKE_IMAGE_PLUGIN=${SIGHT_IMAGE_SOURCE_MODULE};SIGHT_SMOKE_IMAGE_FILE=${SIGHT_MEDIAPIPE_TEST_IMAGE}")
endif()

# Real six-mode switch qualification; model assets stay external.
if(SIGHT_ENABLE_RTMW AND SIGHT_RTMW_MODEL AND SIGHT_YOLOX_MODEL AND
   SIGHT_MEDIAPIPE_123_LIB AND SIGHT_MEDIAPIPE_FACE_MODEL AND
   SIGHT_MEDIAPIPE_POSE_MODEL AND SIGHT_GESTURE_MODEL AND SIGHT_HOLISTIC_MODEL AND
   SIGHT_MEDIAPIPE_TEST_IMAGE AND SIGHT_GESTURE_TEST_IMAGE AND SIGHT_XVFB_RUN AND
   SIGHT_IMAGE_SOURCE_MODULE)
    get_filename_component(_unified_model_dir "${SIGHT_MEDIAPIPE_FACE_MODEL}" DIRECTORY)
    set(_unified_object_model "${SIGHT_OBJECT_MODEL}")
    if(NOT _unified_object_model)
        set(_unified_object_model "${_unified_model_dir}/efficientdet_lite0.tflite")
    endif()
    if(EXISTS "${_unified_object_model}")
        add_executable(sight_unified_tracking_obs tests/verify-unified-tracking-obs.cpp)
        target_include_directories(sight_unified_tracking_obs PRIVATE "${SIGHT_OBS_INCLUDE}" src)
        target_link_libraries(sight_unified_tracking_obs PRIVATE "${SIGHT_OBS_LIBRARY}" pthread)
        add_test(NAME sight_unified_tracking_real_obs COMMAND
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
            "$<TARGET_FILE:sight_unified_tracking_obs>"
            "$<TARGET_FILE:live-latent-lab-sight>" "${CMAKE_CURRENT_SOURCE_DIR}/data"
            "${SIGHT_MEDIAPIPE_123_LIB}" "${SIGHT_MEDIAPIPE_TEST_IMAGE}" "${SIGHT_GESTURE_TEST_IMAGE}"
            "${SIGHT_MEDIAPIPE_FACE_MODEL}" "${SIGHT_MEDIAPIPE_POSE_MODEL}" "${SIGHT_GESTURE_MODEL}"
            "${SIGHT_HOLISTIC_MODEL}" "${_unified_object_model}" "${SIGHT_RTMW_MODEL}"
            "${SIGHT_YOLOX_MODEL}" "${SIGHT_IMAGE_SOURCE_MODULE}")
        set_tests_properties(sight_unified_tracking_real_obs PROPERTIES TIMEOUT 150 RUN_SERIAL TRUE)
    endif()
endif()

include(cmake/preprocessors.cmake)

# Optional NVIDIA/OpenGL texture path; baseline platforms have no CUDA
# build/runtime dependency. PTX is built locally, not committed.
option(SIGHT_ENABLE_CUDA_TEXTURE_INTEROP "Enable native CUDA/OpenGL Portrait Sketch texture path" OFF)
if(SIGHT_ENABLE_CUDA_TEXTURE_INTEROP)
    if(NOT EXISTS "${SIGHT_ORT_LIB_DIR}/libonnxruntime_providers_cuda.so")
        message(FATAL_ERROR "CUDA texture interop requires a matching ONNX Runtime GPU SDK")
    endif()
    find_package(CUDAToolkit REQUIRED)
    find_package(OpenGL REQUIRED)
    set(SIGHT_CUDA_TEXTURE_ARCH "75" CACHE STRING "Minimum CUDA virtual architecture for texture kernels")
    set(_sketch_ptx "${CMAKE_CURRENT_BINARY_DIR}/sketch-cuda.ptx")
    add_custom_command(OUTPUT "${_sketch_ptx}"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx "-arch=compute_${SIGHT_CUDA_TEXTURE_ARCH}" -O3
                "${CMAKE_CURRENT_SOURCE_DIR}/src/core/sketch-cuda-kernels.cu" -o "${_sketch_ptx}"
        DEPENDS src/core/sketch-cuda-kernels.cu VERBATIM)
    add_custom_target(sight_sketch_cuda_ptx DEPENDS "${_sketch_ptx}")
    add_dependencies(live-latent-lab-sight sight_sketch_cuda_ptx)
    target_compile_definitions(live-latent-lab-sight PRIVATE SIGHT_WITH_CUDA_TEXTURE)
    target_link_libraries(live-latent-lab-sight PRIVATE CUDA::cuda_driver OpenGL::GL)
    install(FILES "${_sketch_ptx}" DESTINATION data)
endif()

# Worker-only device tensor I/O, independent of graphics interop.
option(SIGHT_ENABLE_CUDA_DEPTH_IO "Bind Depth Anything tensors to CUDA device memory" ${SIGHT_ENABLE_CUDA_TEXTURE_INTEROP})
if(SIGHT_ENABLE_CUDA_DEPTH_IO)
    if(NOT EXISTS "${SIGHT_ORT_LIB_DIR}/libonnxruntime_providers_cuda.so")
        message(FATAL_ERROR "Depth device I/O requires a matching CUDA ONNX Runtime SDK")
    endif()
    find_package(CUDAToolkit REQUIRED)
    set(SIGHT_CUDA_TEXTURE_ARCH "75" CACHE STRING "Minimum CUDA virtual architecture for texture kernels")
    set(_depth_ptx "${CMAKE_CURRENT_BINARY_DIR}/depth-cuda.ptx")
    set(_depth_header "${CMAKE_CURRENT_BINARY_DIR}/sight-depth-cuda-ptx.hpp")
    add_custom_command(OUTPUT "${_depth_header}"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx "-arch=compute_${SIGHT_CUDA_TEXTURE_ARCH}" -O3 --fmad=false --prec-div=true
                "${CMAKE_CURRENT_SOURCE_DIR}/src/core/depth-cuda-kernels.cu" -o "${_depth_ptx}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${_depth_ptx}" "-DOUTPUT=${_depth_header}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed-depth-ptx.cmake"
        DEPENDS src/core/depth-cuda-kernels.cu cmake/embed-depth-ptx.cmake
        BYPRODUCTS "${_depth_ptx}" VERBATIM)
    add_custom_target(sight_depth_cuda_ptx DEPENDS "${_depth_header}")
    add_executable(sight_depth_cuda_io_test tests/verify-depth-cuda-io.cpp src/core/onnx_depth_processor.cpp)
    target_include_directories(sight_depth_cuda_io_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SIGHT_ORT_INCLUDE}")
    target_link_libraries(sight_depth_cuda_io_test PRIVATE "${SIGHT_ORT_LIBRARY}")
    set_target_properties(sight_depth_cuda_io_test PROPERTIES BUILD_RPATH "${SIGHT_ORT_LIB_DIR}")
    foreach(target IN ITEMS live-latent-lab-sight sight_depth_cuda_io_test)
        add_dependencies(${target} sight_depth_cuda_ptx)
        target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_compile_definitions(${target} PRIVATE SIGHT_WITH_CUDA_DEPTH_IO)
        target_link_libraries(${target} PRIVATE CUDA::cuda_driver)
    endforeach()
    if(SIGHT_DEPTH_MODEL)
        add_test(NAME sight_depth_cuda_device_io COMMAND sight_depth_cuda_io_test "${SIGHT_DEPTH_MODEL}")
        set_tests_properties(sight_depth_cuda_device_io PROPERTIES TIMEOUT 90 RUN_SERIAL TRUE)
    endif()
endif()
