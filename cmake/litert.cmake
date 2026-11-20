set(LITERT_SDK_ROOT "" CACHE PATH
    "Root directory of the LiteRT C++ SDK with runtime libraries")

if(NOT LITERT_SDK_ROOT)
    message(FATAL_ERROR
        "LITERT_SDK_ROOT is required. Set it to the extracted LiteRT C++ SDK root.")
endif()

if(NOT IS_DIRECTORY "${LITERT_SDK_ROOT}")
    message(FATAL_ERROR
        "LiteRT SDK root does not exist or is not a directory: '${LITERT_SDK_ROOT}'.")
endif()

if(NOT EXISTS "${LITERT_SDK_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
        "LiteRT SDK CMakeLists.txt was not found: '${LITERT_SDK_ROOT}/CMakeLists.txt'.")
endif()

set(LITERT_RUNTIME_DIR "${LITERT_SDK_ROOT}" CACHE PATH
    "Directory containing libLiteRt.so and libLiteRtWebGpuAccelerator.so")
get_filename_component(LITERT_RUNTIME_DIR "${LITERT_RUNTIME_DIR}" ABSOLUTE
    BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LITERT_RUNTIME_DIR "${LITERT_RUNTIME_DIR}" CACHE PATH
    "Directory containing libLiteRt.so and libLiteRtWebGpuAccelerator.so" FORCE)

set(LITERT_COMPILED_MODEL_HEADER
    "${LITERT_SDK_ROOT}/litert/cc/litert_compiled_model.h")
if(NOT EXISTS "${LITERT_COMPILED_MODEL_HEADER}")
    message(FATAL_ERROR
        "LiteRT CompiledModel header was not found: '${LITERT_COMPILED_MODEL_HEADER}'.")
endif()

set(LITERT_LIBRARY "${LITERT_RUNTIME_DIR}/libLiteRt.so" CACHE FILEPATH
    "LiteRT runtime library" FORCE)
if(NOT EXISTS "${LITERT_LIBRARY}")
    message(FATAL_ERROR "LiteRT runtime library was not found: '${LITERT_LIBRARY}'.")
endif()

set(LITERT_GPU_LIBRARY "${LITERT_RUNTIME_DIR}/libLiteRtWebGpuAccelerator.so" CACHE FILEPATH
    "LiteRT WebGPU accelerator library" FORCE)
if(NOT EXISTS "${LITERT_GPU_LIBRARY}")
    message(FATAL_ERROR
        "LiteRT WebGPU accelerator library was not found: '${LITERT_GPU_LIBRARY}'.")
endif()

# Google ships the C++ API as source and imports the runtime library from the
# SDK root. The WebGPU accelerator is discovered at runtime beside it.
add_subdirectory("${LITERT_SDK_ROOT}" "${CMAKE_BINARY_DIR}/litert_cc_sdk" EXCLUDE_FROM_ALL)

set(LITERT_INCLUDE_DIR "${LITERT_SDK_ROOT}" CACHE PATH
    "LiteRT C++ API include root" FORCE)

message(STATUS "LiteRT SDK: ${LITERT_SDK_ROOT}")
message(STATUS "LiteRT runtime: ${LITERT_LIBRARY}")
message(STATUS "LiteRT WebGPU accelerator: ${LITERT_GPU_LIBRARY}")
