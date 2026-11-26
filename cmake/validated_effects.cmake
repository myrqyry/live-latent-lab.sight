# Progressive activation: build one Sight plugin with its verified effects.
# Other effects remain in source control, unregistered and not compiled.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Validated progressive-effects build currently targets Linux")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Do not use libobs' pkg-config compile flags for C++: some Linux OBS
# packages export -std=gnu17 and -Werror, which break this C++20 plugin.
find_path(SIGHT_OBS_INCLUDE obs-module.h REQUIRED
    PATHS /usr/local/include/obs /usr/include/obs NO_DEFAULT_PATH)
file(READ "${SIGHT_OBS_INCLUDE}/obs-config.h" SIGHT_OBS_CONFIG_H)
string(REGEX MATCH "LIBOBS_API_MAJOR_VER +([0-9]+)" _sight_obs_major "${SIGHT_OBS_CONFIG_H}")
set(SIGHT_OBS_API_MAJOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "LIBOBS_API_MINOR_VER +([0-9]+)" _sight_obs_minor "${SIGHT_OBS_CONFIG_H}")
set(SIGHT_OBS_API_MINOR "${CMAKE_MATCH_1}")
if(NOT SIGHT_OBS_API_MAJOR OR NOT SIGHT_OBS_API_MINOR)
    message(FATAL_ERROR "Could not read LIBOBS_API version from ${SIGHT_OBS_INCLUDE}/obs-config.h")
endif()
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
    PATHS /usr/local/lib /usr/lib/x86_64-linux-gnu NO_DEFAULT_PATH)

set(SIGHT_ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime" CACHE PATH
    "ONNX Runtime installation with include/ and lib/")
find_path(SIGHT_ORT_INCLUDE onnxruntime_cxx_api.h REQUIRED
    PATHS "${SIGHT_ORT_ROOT}/include" NO_DEFAULT_PATH)
find_library(SIGHT_ORT_LIBRARY
    NAMES onnxruntime libonnxruntime.so.1.17.1
    REQUIRED PATHS "${SIGHT_ORT_ROOT}/lib" NO_DEFAULT_PATH)
get_filename_component(SIGHT_ORT_LIB_DIR "${SIGHT_ORT_LIBRARY}" DIRECTORY)

add_library(live-latent-lab-sight MODULE
    src/plugin-main.cpp
    src/animegan-filter.cpp
    src/core/enhancement_processor.cpp
    src/common/sight-log.cpp
    src/u2net-filter.cpp
    src/core/segmentation_processor.cpp
    src/depth-filter.cpp
    src/core/onnx_depth_processor.cpp
    src/tasks-tracking-filter.cpp
    src/core/tasks_tracker.cpp
    src/core/holistic_tracker.cpp
    src/core/object_detector.cpp
    src/core/object_overlay.cpp
    src/core/hand_signals.cpp
    src/core/tracking_overlay.cpp
    src/core/tracking_signal_bus.cpp
    src/tracking-signal-receiver.cpp
    src/privacy-v2-filter.cpp
    src/fast-face-detector.cpp
    src/3rdparty/libfacedetection/facedetectcnn.cpp
    src/3rdparty/libfacedetection/facedetectcnn-data.cpp
    src/3rdparty/libfacedetection/facedetectcnn-model.cpp
    src/value-router-filter.cpp
    src/core/value_modulation.cpp
    src/native-vrm-source.cpp
    src/vrm/vrm_asset.cpp
    src/vrm/vrm_tracking.cpp
    src/vrm/vrm_coverage.cpp
)
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

install(TARGETS live-latent-lab-sight
    LIBRARY DESTINATION bin/64bit)
# Bundle the exact ABI-tested legacy MediaPipe binary when supplied.
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
    install(FILES "${SIGHT_MEDIAPIPE_LIB}" DESTINATION bin/64bit)
    # The header package supplies this license text; confirm rights to
    # redistribute the separately obtained native library before publishing.
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/external/mediapipe-tasks-legacy/LICENSE"
        DESTINATION data/licenses/mediapipe)
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

# The versioned ONNX Runtime binary is installed beside the plugin for
# $ORIGIN resolution. Model weights never enter this package.
install(FILES "${SIGHT_ORT_LIBRARY}" DESTINATION bin/64bit)
install(DIRECTORY data/ DESTINATION data)

enable_testing()
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/litert_tracking.cmake")
# Only uses Python's stdlib; tests installation safety without touching OBS.
find_program(SIGHT_PYTHON3_EXECUTABLE NAMES python3)
if(SIGHT_PYTHON3_EXECUTABLE)
    add_test(NAME sight_release_safety
        COMMAND "${SIGHT_PYTHON3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_release_safety.py")
    set_tests_properties(sight_release_safety PROPERTIES TIMEOUT 60)
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



# Object detection has its own stable-ID layer. None of these tests
# substitutes mock boxes for MediaPipe native inference acceptance.
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
target_compile_options(sight_live2d_animation_test PRIVATE -UNDEBUG)
add_test(NAME sight_live2d_animation_unit COMMAND sight_live2d_animation_test)
add_executable(sight_live2d_intake_test tests/live2d_intake_test.cpp
    src/live2d/model_package.cpp src/live2d/face_parameters.cpp)
target_include_directories(sight_live2d_intake_test PRIVATE src)
target_compile_options(sight_live2d_intake_test PRIVATE -UNDEBUG)
add_test(NAME sight_live2d_intake_unit COMMAND sight_live2d_intake_test)

add_executable(sight_vrm_asset_test tests/vrm_asset_test.cpp src/vrm/vrm_asset.cpp)
target_include_directories(sight_vrm_asset_test PRIVATE src)
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
target_compile_options(sight_vrm_tracking_test PRIVATE -UNDEBUG)
add_test(NAME sight_vrm_tracking_unit COMMAND sight_vrm_tracking_test)
add_executable(sight_vrm_coverage_test
    tests/vrm_coverage_test.cpp src/vrm/vrm_coverage.cpp
    src/vrm/vrm_asset.cpp src/vrm/vrm_tracking.cpp)
target_include_directories(sight_vrm_coverage_test PRIVATE src)
target_compile_options(sight_vrm_coverage_test PRIVATE -UNDEBUG)
add_test(NAME sight_vrm_coverage_unit COMMAND sight_vrm_coverage_test)

add_executable(sight_vrm_transform_interop_test
    tests/vrm_transform_interop_test.cpp src/vrm/vrm_asset.cpp
    src/vrm/vrm_tracking.cpp)
target_include_directories(sight_vrm_transform_interop_test PRIVATE src)
target_compile_options(sight_vrm_transform_interop_test PRIVATE -UNDEBUG)
add_test(NAME sight_vrm_transform_interop_unit
    COMMAND sight_vrm_transform_interop_test)

add_executable(sight_vrm_body_tracking_test tests/vrm_body_tracking_test.cpp
    src/vrm/vrm_asset.cpp src/vrm/vrm_tracking.cpp)
target_include_directories(sight_vrm_body_tracking_test PRIVATE src)
target_compile_options(sight_vrm_body_tracking_test PRIVATE -UNDEBUG)
add_test(NAME sight_vrm_body_tracking_unit COMMAND sight_vrm_body_tracking_test)
add_executable(sight_vrm_hand_tracking_test tests/vrm_hand_tracking_test.cpp
    src/vrm/vrm_asset.cpp src/vrm/vrm_tracking.cpp)
target_include_directories(sight_vrm_hand_tracking_test PRIVATE src)
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

            if(SIGHT_U2NET_MODEL)
                add_test(NAME sight_u2net_obs_cutout
                    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_obs_sight_render.sh"
                        "$<TARGET_FILE:sight_obs_effect_render_smoke>"
                        "$<TARGET_FILE:live-latent-lab-sight>"
                        "${SIGHT_U2NET_MODEL}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/data")
                set_tests_properties(sight_u2net_obs_cutout PROPERTIES TIMEOUT 60
                    ENVIRONMENT "SIGHT_SMOKE_WIDTH=750;SIGHT_SMOKE_HEIGHT=1000;SIGHT_SMOKE_EFFECT=sight_u2net_filter")
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
                      "SIGHT_SMOKE_WIDTH=128;SIGHT_SMOKE_HEIGHT=128;SIGHT_SMOKE_EFFECT=sight_depth_filter;SIGHT_SMOKE_DEPTH_EDGE=252;SIGHT_SMOKE_DEPTH_MAX_AGE=2;SIGHT_SMOKE_DEPTH_VERIFY_EXPIRY=2900")
                # If the OBS source geometry changes, old depth must not
                # stretch across the resized source before new inference.
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
# user-supplied model and native library, never test-only fake landmarks.
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
    endif()
endif()

if(SIGHT_HAVE_DENSE_FLOW)
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
