cmake_minimum_required(VERSION 3.18)
project(mp_bench LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "mp_bench_worker requires Linux, libnuma and libibverbs")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
get_filename_component(MPCOMM_REPOSITORY "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

option(MP_BENCH_WITH_CUDA "Enable GPU buffers and CUDA device probing" ON)
set(USE_CUDA ${MP_BENCH_WITH_CUDA} CACHE BOOL "Enable CUDA Driver API support" FORCE)
set(USE_CUDA_KERNELS OFF CACHE BOOL "Do not require nvcc for mp_bench" FORCE)
set(BUILD_MPCOMM_PYTHON OFF CACHE BOOL "Do not build Python bindings" FORCE)
set(BUILD_MPCOMM_EXAMPLES OFF CACHE BOOL "Do not build unrelated examples" FORCE)

find_package(Threads REQUIRED)
find_path(NUMA_INCLUDE_DIR numa.h REQUIRED)
find_library(NUMA_LIBRARY numa REQUIRED)
if(MP_BENCH_WITH_CUDA)
    find_path(CUDA_INCLUDE_DIR cuda.h
        HINTS "$ENV{CUDA_HOME}/include" "$ENV{CUDA_PATH}/include"
        PATHS /usr/local/cuda/include REQUIRED)
    find_library(CUDA_DRIVER_LIBRARY cuda
        HINTS "$ENV{CUDA_HOME}/lib64" "$ENV{CUDA_PATH}/lib64"
        PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
              /usr/lib/aarch64-linux-gnu /usr/lib64 REQUIRED)
endif()

if(NOT TARGET mpcomm_shared)
    add_subdirectory("${MPCOMM_REPOSITORY}" "${CMAKE_CURRENT_BINARY_DIR}/mpcomm" EXCLUDE_FROM_ALL)
endif()
get_target_property(_mpcomm_definitions mpcomm_shared INTERFACE_COMPILE_DEFINITIONS)
if(MP_BENCH_WITH_CUDA AND NOT "USE_CUDA" IN_LIST _mpcomm_definitions)
    message(FATAL_ERROR "The existing mpcomm_shared target must be configured with USE_CUDA=ON")
elseif(NOT MP_BENCH_WITH_CUDA AND "USE_CUDA" IN_LIST _mpcomm_definitions)
    message(FATAL_ERROR "CPU-only mp_bench requires mpcomm_shared configured with USE_CUDA=OFF")
endif()

add_executable(mp_bench_worker mp_bench_worker.cpp)
target_compile_features(mp_bench_worker PRIVATE cxx_std_17)
target_include_directories(mp_bench_worker PRIVATE "${NUMA_INCLUDE_DIR}")
target_compile_options(mp_bench_worker PRIVATE -Wall -Wextra -Wpedantic -O2)
target_link_libraries(mp_bench_worker PRIVATE mpcomm_shared "${NUMA_LIBRARY}" Threads::Threads)
if(MP_BENCH_WITH_CUDA)
    target_compile_definitions(mp_bench_worker PRIVATE MP_BENCH_WITH_CUDA=1)
    target_include_directories(mp_bench_worker PRIVATE "${CUDA_INCLUDE_DIR}")
    target_link_libraries(mp_bench_worker PRIVATE "${CUDA_DRIVER_LIBRARY}")
else()
    target_compile_definitions(mp_bench_worker PRIVATE MP_BENCH_WITH_CUDA=0)
endif()
