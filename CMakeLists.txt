cmake_minimum_required(VERSION 3.16)
project(TGTplus LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
option(TGT_BUILD_AESC "Build spanning-centrality programs (requires Intel oneAPI)" ON)
find_package(OpenMP COMPONENTS CXX)

# Header-only dependencies for the edgeutils dataset preparation utilities.
# Populated without running their CMake scripts; sources are used directly.
include(FetchContent)
FetchContent_Declare(eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
FetchContent_Declare(spectra
  URL https://github.com/yixuan/spectra/archive/refs/tags/v1.0.1.tar.gz)
FetchContent_Declare(httplib
  URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.15.3.tar.gz)
FetchContent_Declare(miniz
  URL https://github.com/richgel999/miniz/archive/refs/tags/3.0.2.tar.gz)
FetchContent_Populate(eigen)
FetchContent_Populate(spectra)
FetchContent_Populate(httplib)
FetchContent_Populate(miniz)

function(tgt_executable name source simd)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE src)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(simd)
    target_compile_options(${name} PRIVATE -mavx2)
  endif()
endfunction()

# httplib pulls in glibc's resolv.h... (see src/dataset_download.h); TLS is
# needed to follow http->https redirects (e.g. nrvis.com).
find_package(OpenSSL)
tgt_executable(edgeutils src/edgeutils.cpp FALSE)
if(OPENSSL_FOUND)
  target_compile_definitions(edgeutils PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
  target_link_libraries(edgeutils PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()
target_sources(edgeutils PRIVATE
  ${miniz_SOURCE_DIR}/miniz.c
  ${miniz_SOURCE_DIR}/miniz_tinfl.c
  ${miniz_SOURCE_DIR}/miniz_tdef.c
  ${miniz_SOURCE_DIR}/miniz_zip.c)
# miniz expects a generated miniz_export.h; a static build needs it empty.
file(WRITE "${CMAKE_BINARY_DIR}/miniz_shim/miniz_export.h"
"#ifndef MINIZ_EXPORT_H\n#define MINIZ_EXPORT_H\n#define MINIZ_EXPORT\n#define MINIZ_NO_EXPORT\n#endif\n")
target_include_directories(edgeutils PRIVATE
  ${CMAKE_BINARY_DIR}/miniz_shim
  ${eigen_SOURCE_DIR}
  ${spectra_SOURCE_DIR}/include
  ${httplib_SOURCE_DIR}
  ${miniz_SOURCE_DIR})
if(OpenMP_CXX_FOUND)
  target_link_libraries(edgeutils PRIVATE OpenMP::OpenMP_CXX)
endif()

# Exact spanning centrality reference values (compiler-agnostic).
tgt_executable(exact_sc src/exact_sc.cpp FALSE)
target_include_directories(exact_sc PRIVATE ${eigen_SOURCE_DIR})

if(TGT_BUILD_AESC)
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    message(FATAL_ERROR "AESC builds on x86-64; on Apple Silicon use an amd64 oneAPI container.")
  endif()
  # CMake < 3.20 reports the icpx front end as "Clang"; accept its path too.
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT CMAKE_CXX_COMPILER MATCHES "icpx")
    message(FATAL_ERROR "AESC uses Intel SVML intrinsics; configure with CXX=icpx.")
  endif()
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  tgt_executable(tgtp src/main.cpp TRUE)
  tgt_executable(benchmark src/benchmark.cpp TRUE)
  # Single-threaded MC baseline (scalar two-way randomwalk_mc kernel).
  tgt_executable(benchmark_baseline src/benchmark_baseline.cpp FALSE)
  tgt_executable(rw_experiment src/rw_experiment.cpp FALSE)
  tgt_executable(rw_experiment_simd src/rw_experiment_simd.cpp TRUE)
  tgt_executable(rw_experiment_saba src/rw_experiment_saba.cpp TRUE)
  # All timed implementations use the publication's optimization flags.
  foreach(name tgtp benchmark benchmark_baseline rw_experiment rw_experiment_simd rw_experiment_saba)
    target_compile_options(${name} PRIVATE -Ofast -march=native)
  endforeach()
endif()
