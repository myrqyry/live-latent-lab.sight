cmake_minimum_required(VERSION 3.16)
project(live-latent-lab-sight VERSION 0.3.0 LANGUAGES C CXX)

# Only finished effects are exposed by default. The original comprehensive
# build remains available for development of the other effects.
option(SIGHT_ENABLE_UNVERIFIED_EFFECTS
    "Build and register the legacy experimental effects (requires LiteRT and SD)" OFF)
if(NOT SIGHT_ENABLE_UNVERIFIED_EFFECTS)
    include(cmake/validated_effects.cmake)
    return()
endif()

enable_testing()

# ===== PLATFORM & LANGUAGE =====
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenSSL REQUIRED COMPONENTS Crypto)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(litert)

# Enable optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
endif()

# Detect platform
if(WIN32)
    set(OS_PLATFORM "windows")
elseif(APPLE)
    set(OS_PLATFORM "macos")
elseif(UNIX)
    set(OS_PLATFORM "linux")
endif()

# ===== OBS LIBRARY DETECTION =====
if(OBS_DIR)
    message(STATUS "Using Manual OBS_DIR: ${OBS_DIR}")
    set(LibOBS_INCLUDE_DIR 
        "${OBS_DIR}/libobs"
        "${OBS_DIR}/build/config")
    set(LibOBS_LIBRARY "${OBS_DIR}/build/libobs/libobs.so")
else()
    find_package(PkgConfig REQUIRED)
    # Check if libobs is installed via pkg-config (common on Linux)
    pkg_check_modules(LIBOBS libobs)
    if(LIBOBS_FOUND)
        # pkg-config may return the wrong include path, so verify obs-module.h exists
        set(LibOBS_LIBRARY ${LIBOBS_LIBRARIES})
        
        # Try to find obs-module.h in various locations
        find_path(OBS_HEADER_PATH obs-module.h
            PATHS 
                ${LIBOBS_INCLUDE_DIRS}
                /usr/include/obs
                /usr/local/include/obs
                /usr/include
                /usr/local/include
            PATH_SUFFIXES libobs obs
        )
        
        if(OBS_HEADER_PATH)
            set(LibOBS_INCLUDE_DIR ${OBS_HEADER_PATH})
            message(STATUS "Found obs-module.h at: ${OBS_HEADER_PATH}")
        else()
            # Fallback: use pkg-config paths and hope for the best
            set(LibOBS_INCLUDE_DIR ${LIBOBS_INCLUDE_DIRS})
            message(WARNING "Could not find obs-module.h, using pkg-config paths: ${LIBOBS_INCLUDE_DIRS}")
        endif()
    else()
        # If not found, try to find it in standard paths or fail gracefully
        message(STATUS "LibOBS not found via pkg-config, searching standard paths...")
        find_library(LIBOBS_LIBRARY obs)
        find_path(LibOBS_INCLUDE_DIR obs-module.h 
            PATHS /usr/include /usr/local/include
            PATH_SUFFIXES libobs obs)
        if (NOT LIBOBS_LIBRARY OR NOT LibOBS_INCLUDE_DIR)
             message(WARNING "LibOBS not found. Plugin target will be disabled, but standalone tests can be built.")
        endif()
    endif()
endif()

message(STATUS "LibOBS Include: ${LibOBS_INCLUDE_DIR}")
message(STATUS "LibOBS Library: ${LibOBS_LIBRARY}")

# ===== QT FOR DOCK PANEL =====
find_package(Qt6 COMPONENTS Widgets Network QUIET)
if(Qt6_FOUND)
    message(STATUS "Qt6 Widgets and Network found - enabling custom dock panel")
    set(ENABLE_DOCK_PANEL ON)
    set(QT_LIBRARIES Qt6::Widgets Qt6::Network)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)
else()
    find_package(Qt5 COMPONENTS Widgets Network QUIET)
    if(Qt5_FOUND)
        message(STATUS "Qt5 Widgets and Network found - enabling custom dock panel")
        set(ENABLE_DOCK_PANEL ON)
        set(QT_LIBRARIES Qt5::Widgets Qt5::Network)
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_AUTOUIC ON)
        set(CMAKE_AUTORCC ON)
    else()
        message(STATUS "Qt6/Qt5 not found - dock panel will be disabled")
        set(ENABLE_DOCK_PANEL OFF)
    endif()
endif()

# ===== OBS FRONTEND API =====
if(ENABLE_DOCK_PANEL)
    # Find obs-frontend-api
    find_library(OBS_FRONTEND_LIBRARY obs-frontend-api
        PATHS 
            /usr/lib
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
    )
    
    find_path(OBS_FRONTEND_INCLUDE obs-frontend-api.h
        PATHS
            /usr/include/obs
            /usr/local/include/obs
            ${LibOBS_INCLUDE_DIR}
        PATH_SUFFIXES frontend-api
    )
    
    if(OBS_FRONTEND_LIBRARY AND OBS_FRONTEND_INCLUDE)
        message(STATUS "OBS Frontend API found: ${OBS_FRONTEND_LIBRARY}")
    else()
        message(WARNING "OBS Frontend API not found - dock panel disabled")
        set(ENABLE_DOCK_PANEL OFF)
    endif()
endif()

# ===== STABLE DIFFUSION C++ =====
set(SD_CPP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/stable-diffusion.cpp")

if(NOT EXISTS "${SD_CPP_PATH}/CMakeLists.txt")
    message(WARNING "stable-diffusion.cpp not found! Run: git submodule update --init --recursive")
endif()

# Configure SD build options
set(SD_CUDA ON CACHE BOOL "Enable CUDA acceleration" FORCE)
option(SD_VULKAN "Enable Vulkan acceleration" OFF)
option(SD_METAL "Enable Metal acceleration (macOS)" OFF)

set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# ===== MEDIAPIPE BACKEND =====
option(USE_ONNX_MEDIAPIPE "Enable ONNX Runtime for real MediaPipe face detection/landmarks" ON)
if(USE_ONNX_MEDIAPIPE)
    find_package(PkgConfig QUIET)
    set(ONNXRUNTIME_FOUND FALSE)
    
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(ONNXRUNTIME_PKG QUIET libonnxruntime)
        if(ONNXRUNTIME_PKG_FOUND)
            set(ONNXRUNTIME_FOUND TRUE)
            set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_PKG_INCLUDE_DIRS})
            set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_PKG_LIBRARIES})
            set(ONNXRUNTIME_LIBRARY_DIRS ${ONNXRUNTIME_PKG_LIBRARY_DIRS})
            message(STATUS "Found ONNX Runtime via pkg-config")
        endif()
    endif()
    
    if(NOT ONNXRUNTIME_FOUND)
        set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime")
        
        # The checked-in ONNX Runtime package has a versioned .so but no
        # unversioned linker symlink. Search both names to avoid silently
        # compiling the non-inference fallback when the runtime is present.
        find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime libonnxruntime.so.1.17.1
            PATHS
                ${ORT_ROOT}/lib
                /usr/lib
                /usr/local/lib
                /opt/onnxruntime/lib
        )
        find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
            PATHS
                ${ORT_ROOT}/include
                /usr/include/onnxruntime
                /usr/local/include/onnxruntime
                /opt/onnxruntime/include/onnxruntime
        )
        if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
            set(ONNXRUNTIME_FOUND TRUE)
            set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIB})
            set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE})
            message(STATUS "Found ONNX Runtime at: ${ONNXRUNTIME_LIB}")
        endif()
    endif()
    
    if(ONNXRUNTIME_FOUND)
        message(STATUS "ONNX Runtime found — enabling real MediaPipe inference")
        add_compile_definitions(USE_ONNX_MEDIAPIPE)
        include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
    else()
        message(WARNING "ONNX Runtime not found — MediaPipe face pipeline unavailable (no simulated fallback)")
        set(USE_ONNX_MEDIAPIPE OFF)
    endif()
endif()

# ===== OPENVINO EXECUTION PROVIDER =====
option(USE_OPENVINO_EP "Enable OpenVINO Execution Provider for ONNX Runtime (Intel acceleration)" OFF)
if(USE_OPENVINO_EP AND USE_ONNX_MEDIAPIPE)
    find_package(OpenVINO QUIET COMPONENTS Runtime)
    if(OpenVINO_FOUND)
        message(STATUS "OpenVINO found — enabling OpenVINO EP for ONNX Runtime")
        add_compile_definitions(USE_OPENVINO_EP)
    else()
        message(WARNING "OpenVINO not found — OpenVINO EP disabled")
        set(USE_OPENVINO_EP OFF CACHE BOOL "" FORCE)
    endif()
endif()

# Enable appropriate backend
if(SD_CUDA AND UNIX AND NOT APPLE)
    message(STATUS "Building with CUDA support")
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
         set(SD_CUDA ON CACHE BOOL "" FORCE)
    else()
         message(WARNING "CUDA Toolkit not found, falling back to Vulkan or CPU if possible.")
         set(SD_CUDA OFF CACHE BOOL "" FORCE)
         # Try Vulkan?
         find_package(Vulkan QUIET)
         if(Vulkan_FOUND)
             message(STATUS "Vulkan found, enabling SD_VULKAN")
             set(SD_VULKAN ON CACHE BOOL "" FORCE)
         endif()
    endif()
elseif(SD_VULKAN)
    message(STATUS "Building with Vulkan support")
    find_package(Vulkan REQUIRED)
    set(SD_VULKAN ON CACHE BOOL "" FORCE)
elseif(APPLE AND SD_METAL)
    message(STATUS "Building with Metal support")
    set(SD_METAL ON CACHE BOOL "" FORCE)
endif()

if(EXISTS "${SD_CPP_PATH}/CMakeLists.txt")
    add_subdirectory(${SD_CPP_PATH} sd_cpp_build EXCLUDE_FROM_ALL)
endif()

# ===== LIBFACEDETECTION =====
# Define sources for the library
set(LIBFACEDETECTION_SOURCES
    src/3rdparty/libfacedetection/facedetectcnn.cpp
    src/3rdparty/libfacedetection/facedetectcnn-data.cpp
    src/3rdparty/libfacedetection/facedetectcnn-model.cpp
    # facedetectcnn.h is a header, not strictly needed in sources list but good for IDEs
    src/3rdparty/libfacedetection/facedetectcnn.h
)

# SIMD Optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    add_compile_definitions(_ENABLE_NEON)
    message(STATUS "Enabling NEON for libfacedetection")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    # Assume AVX2 is available on modern x86_64, or at least safe to enable if -march=native is on.
    # Ideally we'd check, but for this plugin context we'll align with the -march=native or just enable it.
    add_compile_definitions(_ENABLE_AVX2)
    # GCC/Clang require explicit flags to use intrinsics even if macros are defined
    add_compile_options(-mavx2 -mfma) 
    message(STATUS "Enabling AVX2 for libfacedetection")
endif()


# ===== DEFINE PLUGIN =====
set(PLUGIN_SOURCES
    src/plugin-main.cpp
    src/core/latent-engine.cpp
    src/core/litert_runtime.cpp
    src/core/litert_model_manager.cpp
    src/core/litert_model_catalog.cpp
    src/core/depth_estimator.cpp
    src/core/head_pose_estimator.cpp
    src/core/litert_keypoint_estimator.cpp
    src/core/segmentation_processor.cpp
    src/core/sam_predictor.cpp
    src/common/sight-log.cpp
    src/core/optical_flow.cpp
    src/core/point_tracker.cpp
    src/common/mesh_renderer.cpp
    src/style-transfer-filter.cpp
    src/image-generator-source.cpp
    src/neural-enhance-filter.cpp
    src/latent-codec-test-filter.cpp
    src/privacy-filter.cpp
    src/mediapipe-filter.cpp
    src/parallax-filter.cpp
    src/mediapipe/mp_wrapper.cpp
    src/core/nsfw_detector.cpp
    src/core/fast_style_transfer.cpp
    src/core/enhancement_processor.cpp
    src/fast-face-detector.cpp
    ${LIBFACEDETECTION_SOURCES}
)

# Add dock panel sources if Qt is available
if(ENABLE_DOCK_PANEL)
    list(APPEND PLUGIN_SOURCES
        src/ai-style-dock.cpp
        src/ai-style-dock.hpp
    )
endif()

if(LibOBS_LIBRARY AND LibOBS_INCLUDE_DIR)
add_library(live-latent-lab-sight MODULE ${PLUGIN_SOURCES})

add_executable(litert_build_smoke tests/litert_build_smoke.cpp)
target_link_libraries(litert_build_smoke PRIVATE litert_cc_api)

add_executable(litert_runtime_test
    src/core/litert_runtime.cpp
    tests/litert_runtime_test.cpp
)
target_include_directories(litert_runtime_test PRIVATE src)
target_link_libraries(litert_runtime_test PRIVATE litert_cc_api)

add_executable(litert_model_manager_test
    tests/litert_model_manager_test.cpp
    src/core/litert_model_manager.cpp
    src/core/litert_model_catalog.cpp
)
target_include_directories(litert_model_manager_test PRIVATE src)
target_link_libraries(litert_model_manager_test PRIVATE OpenSSL::Crypto)

add_executable(litert_geometry_processors_test
    tests/litert_geometry_processors_test.cpp
    src/core/depth_estimator.cpp
    src/core/head_pose_estimator.cpp
    src/core/litert_keypoint_estimator.cpp
    src/core/litert_runtime.cpp
    src/fast-face-detector.cpp
    ${LIBFACEDETECTION_SOURCES}
)
target_include_directories(litert_geometry_processors_test PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/libfacedetection
    ${LibOBS_INCLUDE_DIR}
)
target_link_libraries(litert_geometry_processors_test PRIVATE
    litert_cc_api
    ${LibOBS_LIBRARY}
)

add_executable(litert_keypoint_estimator_test
    tests/litert_keypoint_estimator_test.cpp
    src/core/litert_keypoint_estimator.cpp
    src/core/litert_runtime.cpp
)
target_include_directories(litert_keypoint_estimator_test PRIVATE src)
target_link_libraries(litert_keypoint_estimator_test PRIVATE litert_cc_api)

add_test(NAME litert_build_smoke COMMAND litert_build_smoke)
add_test(NAME litert_runtime_test COMMAND litert_runtime_test)
add_test(NAME litert_model_manager_test COMMAND litert_model_manager_test)
add_test(NAME litert_geometry_processors_test COMMAND litert_geometry_processors_test)
add_test(NAME litert_keypoint_estimator_test COMMAND litert_keypoint_estimator_test)

    target_include_directories(live-latent-lab-sight PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/entt/single_include
        ${LibOBS_INCLUDE_DIR}
        ${SD_CPP_PATH}
        ${SD_CPP_PATH}/thirdparty
    )
    
    # Add Qt and frontend includes if dock panel is enabled
    if(ENABLE_DOCK_PANEL)
        target_include_directories(live-latent-lab-sight PRIVATE
            ${OBS_FRONTEND_INCLUDE}
        )
        target_compile_definitions(live-latent-lab-sight PRIVATE ENABLE_DOCK_PANEL)
    endif()
    
    # Find OpenMP for parallel image processing
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found - enabling parallel image scaling")
        target_link_libraries(live-latent-lab-sight PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    target_link_libraries(live-latent-lab-sight PRIVATE
        ${LibOBS_LIBRARY}
        stable-diffusion
        pthread
    )
    
    # Link Qt and frontend API if dock panel is enabled
    if(ENABLE_DOCK_PANEL)
        target_link_libraries(live-latent-lab-sight PRIVATE
            ${QT_LIBRARIES}
            ${OBS_FRONTEND_LIBRARY}
        )
    endif()
    
    # Link CUDA libraries if enabled (required since stable-diffusion is static)
    if(SD_CUDA)
        target_link_libraries(live-latent-lab-sight PRIVATE
            CUDA::cublas
            CUDA::cudart
        )
    endif()

    target_link_libraries(live-latent-lab-sight PRIVATE litert_cc_api)
    
    if(USE_ONNX_MEDIAPIPE)
        target_link_libraries(live-latent-lab-sight PRIVATE ${ONNXRUNTIME_LIBRARIES})
        if(ONNXRUNTIME_LIBRARY_DIRS)
            target_link_directories(live-latent-lab-sight PRIVATE ${ONNXRUNTIME_LIBRARY_DIRS})
        endif()
    endif()
    
    if(USE_OPENVINO_EP)
        target_link_libraries(live-latent-lab-sight PRIVATE openvino::runtime)
    endif()
    
    # Cross-platform library properties
    set_target_properties(live-latent-lab-sight PROPERTIES
        PREFIX ""
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        INSTALL_RPATH
            "$ORIGIN;${LITERT_RUNTIME_DIR};${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime/lib"
        BUILD_WITH_INSTALL_RPATH ON
    )
    
    # Platform-specific suffix and settings
    if(WIN32)
        set_target_properties(live-latent-lab-sight PROPERTIES SUFFIX ".dll")
    elseif(APPLE)
        set_target_properties(live-latent-lab-sight PROPERTIES 
            SUFFIX ".so"
            BUNDLE FALSE
            MACOSX_RPATH TRUE
        )
    else()  # Linux/Unix
        set_target_properties(live-latent-lab-sight PROPERTIES SUFFIX ".so")
    endif()

# ===== INSTALLATION =====
# Standard OBS Install Logic
include(GNUInstallDirs)

# Allow overriding the install destinations
if(NOT DEFINED OBS_PLUGIN_DESTINATION)
    set(OBS_PLUGIN_DESTINATION "${CMAKE_INSTALL_LIBDIR}/obs-plugins")
endif()

if(NOT DEFINED OBS_DATA_DESTINATION)
    set(OBS_DATA_DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/obs/obs-plugins/${CMAKE_PROJECT_NAME}")
endif()

# Use the variables in your install commands
install(TARGETS live-latent-lab-sight LIBRARY DESTINATION ${OBS_PLUGIN_DESTINATION})
install(FILES
    "${LITERT_LIBRARY}"
    "${LITERT_GPU_LIBRARY}"
    DESTINATION ${OBS_PLUGIN_DESTINATION}
)
install(DIRECTORY data/ DESTINATION ${OBS_DATA_DESTINATION})
endif()

# ===== TESTING =====
enable_testing()

# 1. Logic Test (Standalone)
add_executable(test_transform tests/test_transform.cpp src/common/image-utils.hpp)
target_include_directories(test_transform PRIVATE src src/common)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_transform PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME test_transform COMMAND test_transform)

# 2. Stable Diffusion Backend Validation (does not load a model)
add_executable(test_stable_diffusion_backend tests/test_stable_diffusion_backend.cpp)
target_include_directories(test_stable_diffusion_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${SD_CPP_PATH}
    ${SD_CPP_PATH}/thirdparty
)
target_link_libraries(test_stable_diffusion_backend PRIVATE stable-diffusion pthread)
if(SD_CUDA)
    target_link_libraries(test_stable_diffusion_backend PRIVATE CUDA::cublas CUDA::cudart)
endif()
add_test(NAME test_stable_diffusion_backend COMMAND test_stable_diffusion_backend)

# 3. Memory Leak Test (Requires LibOBS)
if(LibOBS_LIBRARY AND LibOBS_INCLUDE_DIR)
    # Create a list of sources for the test, excluding entry points and UI-only files
    set(PLUGIN_TEST_SOURCES ${PLUGIN_SOURCES})
    list(FILTER PLUGIN_TEST_SOURCES EXCLUDE REGEX "ai-style-dock\\.(cpp|hpp)$")

    add_executable(test_memory_leak tests/test_memory_leak.cpp ${PLUGIN_TEST_SOURCES})

    target_include_directories(test_memory_leak PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/entt/single_include
        ${LibOBS_INCLUDE_DIR}
        ${SD_CPP_PATH}
        ${SD_CPP_PATH}/thirdparty
    )

    target_link_libraries(test_memory_leak PRIVATE
        ${LibOBS_LIBRARY}
        stable-diffusion
        pthread
        OpenSSL::Crypto
    )

    if(SD_CUDA)
        target_link_libraries(test_memory_leak PRIVATE CUDA::cublas CUDA::cudart)
    endif()

    target_link_libraries(test_memory_leak PRIVATE litert_cc_api)

    if(ENABLE_DOCK_PANEL)
        target_link_libraries(test_memory_leak PRIVATE ${QT_LIBRARIES})
    endif()

    if(USE_ONNX_MEDIAPIPE)
        target_link_libraries(test_memory_leak PRIVATE ${ONNXRUNTIME_LIBRARIES})
        if(ONNXRUNTIME_LIBRARY_DIRS)
            target_link_directories(test_memory_leak PRIVATE ${ONNXRUNTIME_LIBRARY_DIRS})
        endif()
    endif()

    if(USE_OPENVINO_EP)
        target_link_libraries(test_memory_leak PRIVATE openvino::runtime)
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_memory_leak PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_test(NAME test_memory_leak COMMAND test_memory_leak)
endif()
