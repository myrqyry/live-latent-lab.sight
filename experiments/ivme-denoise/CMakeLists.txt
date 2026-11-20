cmake_minimum_required(VERSION 3.20)
project(sight-ivme-backend-smoke LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
get_filename_component(SIGHT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(ORT_DIR "${SIGHT_ROOT}/external/onnxruntime")
set(ORT_SO "${ORT_DIR}/lib/libonnxruntime.so.1.17.1")
add_executable(ivme_onnx_test
    "${SIGHT_ROOT}/tests/ivme_onnx_test.cpp"
    "${SIGHT_ROOT}/src/core/onnx_ivme_denoiser.cpp")
target_include_directories(ivme_onnx_test PRIVATE
    "${SIGHT_ROOT}/src" "${ORT_DIR}/include")
target_link_libraries(ivme_onnx_test PRIVATE "${ORT_SO}")
set_target_properties(ivme_onnx_test PROPERTIES BUILD_RPATH "${ORT_DIR}/lib")
add_executable(ivme_obs_native_720p_smoke tests_obs_filter.cpp)
target_include_directories(ivme_obs_native_720p_smoke PRIVATE /usr/include/obs)
target_compile_definitions(ivme_obs_native_720p_smoke PRIVATE
 SIGHT_TEST_SOURCE_W=1280 SIGHT_TEST_SOURCE_H=720 SIGHT_TEST_CAPTURE_EDGE=0)
target_link_libraries(ivme_obs_native_720p_smoke PRIVATE /usr/lib/x86_64-linux-gnu/libobs.so)
enable_testing()
if(IVME_TEST_MODEL)
 add_test(NAME ivme_real_weight_denoise COMMAND ivme_onnx_test "${IVME_TEST_MODEL}")
 set_tests_properties(ivme_real_weight_denoise PROPERTIES TIMEOUT 60)
 add_test(NAME ivme_real_native_720p COMMAND ivme_onnx_test "${IVME_TEST_MODEL}" --native-720p)
 set_tests_properties(ivme_real_native_720p PROPERTIES TIMEOUT 120)
endif()

# Isolated native OBS module: never overwrites the existing Sight plugin.
add_library(sight-ivme-denoise MODULE
    src/ivme_plugin.cpp
    src/ivme_filter.cpp
    "${SIGHT_ROOT}/src/core/onnx_ivme_denoiser.cpp")
target_include_directories(sight-ivme-denoise PRIVATE
    /usr/include/obs "${SIGHT_ROOT}/src" "${ORT_DIR}/include")
target_link_libraries(sight-ivme-denoise PRIVATE
    /usr/lib/x86_64-linux-gnu/libobs.so "${ORT_SO}")
set_target_properties(sight-ivme-denoise PROPERTIES
    PREFIX "" BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "$ORIGIN")
install(TARGETS sight-ivme-denoise LIBRARY DESTINATION bin/64bit)
install(FILES data/sight-ivme-denoise.effect DESTINATION data)
install(FILES data/locale/en-US.ini DESTINATION data/locale)
install(FILES "${ORT_SO}" DESTINATION bin/64bit)

add_executable(ivme_obs_smoke tests_obs_filter.cpp)
target_include_directories(ivme_obs_smoke PRIVATE /usr/include/obs)
target_link_libraries(ivme_obs_smoke PRIVATE /usr/lib/x86_64-linux-gnu/libobs.so)
if(IVME_TEST_MODEL)
  add_test(NAME ivme_real_onnx_obs_filter
      COMMAND xvfb-run -a -s "-screen 0 1280x720x24"
      $<TARGET_FILE:ivme_obs_smoke>
      $<TARGET_FILE:sight-ivme-denoise>
      "${CMAKE_CURRENT_LIST_DIR}/data" "${IVME_TEST_MODEL}")
  set_tests_properties(ivme_real_onnx_obs_filter PROPERTIES TIMEOUT 90)
  add_test(NAME ivme_real_onnx_obs_native_720p
      COMMAND xvfb-run -a -s "-screen 0 1280x720x24"
      $<TARGET_FILE:ivme_obs_native_720p_smoke>
      $<TARGET_FILE:sight-ivme-denoise>
      "${CMAKE_CURRENT_LIST_DIR}/data" "${IVME_TEST_MODEL}")
  set_tests_properties(ivme_real_onnx_obs_native_720p PROPERTIES TIMEOUT 90
    PASS_REGULAR_EXPRESSION "First REAL native ONNX denoised frame 1280x720")
endif()
