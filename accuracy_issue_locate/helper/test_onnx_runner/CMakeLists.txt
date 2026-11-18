cmake_minimum_required(VERSION 3.16)
project(test_onnx_runner VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime install root, laid out as <root>/include and <root>/lib.
set(ONNXRUNTIME_ROOT "${CMAKE_INSTALL_PREFIX}" CACHE PATH
    "ONNX Runtime install root")

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}/include"
  PATH_SUFFIXES onnxruntime onnxruntime/core/session
  REQUIRED)
find_library(ORT_LIBRARY onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}/lib"
  REQUIRED)

add_executable(test_onnx_runner test_onnx_runner.cpp)
target_include_directories(test_onnx_runner PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(test_onnx_runner PRIVATE "${ORT_LIBRARY}")

if(MSVC)
  # Keep a PDB in Release too so crashes can be symbolized under a debugger.
  target_compile_options(test_onnx_runner PRIVATE /W3 /Zc:__cplusplus /Zi)
  target_link_options(test_onnx_runner PRIVATE /DEBUG)
else()
  target_compile_options(test_onnx_runner PRIVATE -Wall)
endif()

install(TARGETS test_onnx_runner RUNTIME DESTINATION bin)
