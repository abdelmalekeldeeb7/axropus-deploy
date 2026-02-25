cmake_minimum_required(VERSION 3.20)
project(korith_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(KORITH_ENGINE_USE_CUDA "Build with CUDA kernels" OFF)

set(KORITH_ENGINE_SOURCES
  korith_engine.cpp
)

if(KORITH_ENGINE_USE_CUDA)
  enable_language(CUDA)
  list(APPEND KORITH_ENGINE_SOURCES korith_engine_cuda.cu)
endif()

add_library(korith_engine SHARED ${KORITH_ENGINE_SOURCES})

target_include_directories(korith_engine PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(KORITH_ENGINE_USE_CUDA)
  target_compile_definitions(korith_engine PRIVATE KORITH_ENGINE_USE_CUDA=1)
else()
  target_compile_definitions(korith_engine PRIVATE KORITH_ENGINE_STUB=1)
endif()
