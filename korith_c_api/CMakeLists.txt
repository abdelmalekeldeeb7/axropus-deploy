cmake_minimum_required(VERSION 3.22)

project(KorithCAPI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED ENV{LLAMA_ROOT} OR "$ENV{LLAMA_ROOT}" STREQUAL "")
  message(FATAL_ERROR
    "LLAMA_ROOT environment variable is not set.\n"
    "It must point to a llama.cpp checkout/build containing:\n"
    "  - $LLAMA_ROOT/include/llama.h\n"
    "  - $LLAMA_ROOT/build/bin/libllama.so")
endif()

set(LLAMA_ROOT "$ENV{LLAMA_ROOT}")
set(LLAMA_INCLUDE_DIR "${LLAMA_ROOT}/include")
set(GGML_INCLUDE_DIR  "${LLAMA_ROOT}/ggml/include")
set(LLAMA_HEADER      "${LLAMA_INCLUDE_DIR}/llama.h")
set(LLAMA_LIBRARY     "${LLAMA_ROOT}/build/bin/libllama.so")

message(STATUS "LLAMA_ROOT=${LLAMA_ROOT}")
message(STATUS "LLAMA_INCLUDE_DIR=${LLAMA_INCLUDE_DIR}")
message(STATUS "GGML_INCLUDE_DIR=${GGML_INCLUDE_DIR}")
message(STATUS "LLAMA_LIBRARY=${LLAMA_LIBRARY}")

if(NOT EXISTS "${LLAMA_HEADER}")
  message(FATAL_ERROR "Missing llama header: ${LLAMA_HEADER}")
endif()

if(NOT EXISTS "${LLAMA_LIBRARY}")
  message(FATAL_ERROR "Missing llama shared library: ${LLAMA_LIBRARY}")
endif()

add_library(korith SHARED
  korith.cpp
  ../core/amf_store.cpp
)

set_target_properties(korith PROPERTIES
  OUTPUT_NAME korith
  CXX_EXTENSIONS OFF
  BUILD_RPATH "${LLAMA_ROOT}/build/bin"
  INSTALL_RPATH "${LLAMA_ROOT}/build/bin"
)

target_compile_options(korith PRIVATE
  -O3
  -Wall
  -Wextra
)

target_include_directories(korith PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../core"
  "${LLAMA_INCLUDE_DIR}"
  "${GGML_INCLUDE_DIR}"
)

target_link_libraries(korith PRIVATE
  "${LLAMA_LIBRARY}"
)

message(STATUS "Configured libkorith.so (C ABI) successfully")
