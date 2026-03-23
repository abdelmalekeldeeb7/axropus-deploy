cmake_minimum_required(VERSION 3.22)

project(korith_dynamic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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

find_package(Threads REQUIRED)

option(KORITH_ENABLE_CUDA_ACCEPT_SCAN "Enable CUDA-backed speculative accept scan" ON)
set(KORITH_HAS_CUDA_ACCEPT_SCAN OFF)
if(KORITH_ENABLE_CUDA_ACCEPT_SCAN)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(KORITH_HAS_CUDA_ACCEPT_SCAN ON)
    message(STATUS "CUDA accept-scan enabled (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "CUDA compiler not found; using host accept-scan fallback")
  endif()
endif()

set(SCHEDULER_CRATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/controller")
set(SCHEDULER_BUILD_DIR "${SCHEDULER_CRATE_DIR}/target/release")
set(SCHEDULER_LIB_NAME "libkorith_scheduler.so")
set(SCHEDULER_LIB_SRC "${SCHEDULER_BUILD_DIR}/${SCHEDULER_LIB_NAME}")
set(SCHEDULER_LIB_DST_DIR "${CMAKE_BINARY_DIR}/lib")
set(SCHEDULER_LIB_DST "${SCHEDULER_LIB_DST_DIR}/${SCHEDULER_LIB_NAME}")
file(GLOB SCHEDULER_RUST_SOURCES
  "${SCHEDULER_CRATE_DIR}/*.rs"
  "${SCHEDULER_CRATE_DIR}/src/*.rs"
  "${SCHEDULER_CRATE_DIR}/Cargo.toml"
  "${SCHEDULER_CRATE_DIR}/Cargo.lock")

add_custom_command(
  OUTPUT "${SCHEDULER_LIB_SRC}"
  COMMAND cargo build --release --manifest-path "${SCHEDULER_CRATE_DIR}/Cargo.toml"
  DEPENDS ${SCHEDULER_RUST_SOURCES}
  WORKING_DIRECTORY "${SCHEDULER_CRATE_DIR}"
  COMMENT "Building Rust scheduler"
)

add_custom_command(
  OUTPUT "${SCHEDULER_LIB_DST}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${SCHEDULER_LIB_DST_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${SCHEDULER_LIB_SRC}" "${SCHEDULER_LIB_DST}"
  DEPENDS "${SCHEDULER_LIB_SRC}"
  COMMENT "Copying scheduler shared library to build/lib"
)

add_custom_target(korith_scheduler ALL DEPENDS "${SCHEDULER_LIB_DST}")

set(KORITH_DYNAMIC_SOURCES
  core/main.cpp
  core/engine.cpp
  core/amf_store.cpp
  core/memory_field.cpp
  core/batch_executor.cpp
  core/collapse_controller.cpp
  core/collapse_gate.cpp
  core/speculative_engine.cpp
  core/engine_pool.cpp
  core/thermo_controller.cpp
  core/metrics.cpp
  core/spec_v2.cpp
  holographic/memory.cpp
)

if(KORITH_HAS_CUDA_ACCEPT_SCAN)
  list(APPEND KORITH_DYNAMIC_SOURCES
    core/kernels/accept_scan.cu
    core/kernels/spec_verify.cu
    core/cuda_graph_decode.cu
    core/amf_direct_kv.cu
  )
endif()

add_executable(korith_dynamic ${KORITH_DYNAMIC_SOURCES})

option(KORITH_BUILD_TESTS "Build Korith tests" OFF)
if(KORITH_BUILD_TESTS)
  add_executable(amf_prefix_test
    tests/amf_prefix_test.cpp
    core/amf_store.cpp
  )
  target_include_directories(amf_prefix_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/core"
    "${LLAMA_INCLUDE_DIR}"
    "${GGML_INCLUDE_DIR}"
  )

  add_executable(memory_field_test
    tests/memory_field_test.cpp
    core/memory_field.cpp
  )
  target_include_directories(memory_field_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/core"
  )
endif()

target_compile_options(korith_dynamic PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-DNDEBUG;-Wall;-Wextra>
)

if(KORITH_HAS_CUDA_ACCEPT_SCAN)
  target_compile_definitions(korith_dynamic PRIVATE KORITH_USE_CUDA_ACCEPT_SCAN=1)
  target_compile_definitions(korith_dynamic PRIVATE KORITH_USE_CUDA_GRAPHS=1)
  set(KORITH_CUDA_GENCODE_FLAGS
    -gencode=arch=compute_80,code=sm_80
    -gencode=arch=compute_89,code=sm_89
    -gencode=arch=compute_90,code=sm_90
  )
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    list(APPEND KORITH_CUDA_GENCODE_FLAGS
      -gencode=arch=compute_120,code=sm_120
    )
  endif()
  target_compile_options(korith_dynamic PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -O3
      -DNDEBUG
      ${KORITH_CUDA_GENCODE_FLAGS}
    >
  )
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    target_link_libraries(korith_dynamic PRIVATE CUDA::cudart)
  endif()
endif()

target_include_directories(korith_dynamic PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/core"
  "${LLAMA_INCLUDE_DIR}"
  "${GGML_INCLUDE_DIR}"
  # Internal llama.cpp headers needed by amf_direct_kv.cu for KV cache struct access.
  # These are optional — amf_direct_kv.cu guards with __has_include and degrades gracefully.
  "${LLAMA_ROOT}/src"
)

set_target_properties(korith_dynamic PROPERTIES
  CXX_EXTENSIONS OFF
  BUILD_RPATH "$ORIGIN/lib;${LLAMA_ROOT}/build/bin"
  INSTALL_RPATH "$ORIGIN/lib;${LLAMA_ROOT}/build/bin"
)

add_dependencies(korith_dynamic korith_scheduler)

target_link_libraries(korith_dynamic PRIVATE
  "${LLAMA_LIBRARY}"
  Threads::Threads
)
