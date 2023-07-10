message(STATUS "--------------------------------------")
message(STATUS "-> noa::noa: configuring public target...")

# ---------------------------------------------------------------------------------------
# Linking options and libraries
# ---------------------------------------------------------------------------------------
# Common:
include(${PROJECT_SOURCE_DIR}/cmake/ext/fmt.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/spdlog.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/half.cmake)

# Interface gathering all of the dependencies of the library.
add_library(noa_public_libraries INTERFACE)
add_library(noa_private_libraries INTERFACE)

target_link_libraries(noa_public_libraries
    INTERFACE
    fmt::fmt
    spdlog::spdlog
    half::half
    )

if (NOA_ENABLE_TIFF)
    include(${PROJECT_SOURCE_DIR}/cmake/ext/tiff.cmake)
    target_link_libraries(noa_private_libraries INTERFACE TIFF::TIFF)
endif ()

# CPU backend:
if (NOA_ENABLE_CPU)
    find_package(Threads REQUIRED)

    if (NOA_CPU_OPENMP)
        find_package(OpenMP 4.5 REQUIRED)
        # OpenMP pragmas are included in the source files of the user, so use public visibility.
        target_link_libraries(noa_public_libraries INTERFACE OpenMP::OpenMP_CXX)
    endif ()

    include(${PROJECT_SOURCE_DIR}/cmake/ext/fftw.cmake)
    target_link_libraries(noa_public_libraries
        INTERFACE
        Threads::Threads
        ${FFTW3_TARGETS}
        )

    include(${PROJECT_SOURCE_DIR}/cmake/ext/cblas.cmake)
    include(${PROJECT_SOURCE_DIR}/cmake/ext/lapacke.cmake)
    target_link_libraries(noa_private_libraries
        INTERFACE
        CBLAS::CBLAS
        LAPACKE::LAPACKE
        )
endif ()

# CUDA backend:
if (NOA_ENABLE_CUDA)
    include(${PROJECT_SOURCE_DIR}/cmake/ext/cuda-toolkit.cmake)
    target_link_libraries(noa_public_libraries
        INTERFACE
        CUDA::cuda_driver
        $<IF:$<BOOL:${NOA_CUDA_CUDART_STATIC}>, CUDA::cudart_static, CUDA::cudart>
        $<IF:$<BOOL:${NOA_CUDA_CUFFT_STATIC}>, CUDA::cufft_static, CUDA::cufft>
        )
    target_link_libraries(noa_private_libraries
        INTERFACE
        $<IF:$<BOOL:${NOA_CUDA_CURAND_STATIC}>, CUDA::curand_static, CUDA::curand>
        $<IF:$<BOOL:${NOA_CUDA_CUBLAS_STATIC}>, CUDA::cublas_static, CUDA::cublas>
        )
endif ()

# ---------------------------------------------------------------------------------------
# Creating the target
# ---------------------------------------------------------------------------------------
add_library(noa STATIC ${NOA_SOURCES} ${NOA_HEADERS})
add_library(noa::noa ALIAS noa)

target_link_libraries(noa
    PRIVATE prj_common_option prj_compiler_warnings
    PRIVATE noa_private_libraries
    PUBLIC noa_public_libraries
    )

set_target_properties(noa
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
if (NOA_ENABLE_CUDA)
    set_target_properties(noa
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
#        CUDA_RESOLVE_DEVICE_SYMBOLS ON # FIXME
        # CUDA_ARCHITECTURES defaults to CMAKE_CUDA_ARCHITECTURES, which is already set.
        )
endif ()

if (NOA_ENABLE_PCH)
    target_precompile_headers(noa
        PRIVATE
        # Streams:
        <iostream>
        <fstream>
        <string>
        <string_view>

        # Containers:
        <map>
        <unordered_map>
        <vector>
        <array>
        <tuple>

        # Others:
        <cstdint>
        <cctype>
        <cstring>
        <cerrno>
        <cmath>
        <exception>
        <filesystem>
        <thread>
        <utility>
        <algorithm>
        <memory>
        <type_traits>
        <complex>
        <bitset>

        # spdlog
        <spdlog/spdlog.h>
        <spdlog/sinks/stdout_color_sinks.h>
        <spdlog/sinks/basic_file_sink.h>
        <spdlog/fmt/bundled/compile.h>
        <spdlog/fmt/bundled/ranges.h>
        <spdlog/fmt/bundled/os.h>
        <spdlog/fmt/bundled/chrono.h>
        <spdlog/fmt/bundled/color.h>
        )
endif ()

# Set definitions:
target_compile_definitions(noa
    PUBLIC
    "$<$<CONFIG:DEBUG>:NOA_DEBUG>"
    "$<$<BOOL:${NOA_ENABLE_CPU}>:NOA_ENABLE_CPU>"
    "$<$<BOOL:${NOA_ENABLE_CUDA}>:NOA_ENABLE_CUDA>"
    "$<$<BOOL:${NOA_ENABLE_TIFF}>:NOA_ENABLE_TIFF>"
    "$<$<BOOL:${NOA_CPU_OPENMP}>:NOA_ENABLE_OPENMP>"
    "$<$<BOOL:${NOA_ENABLE_CHECKS_AT_RELEASE}>:NOA_ENABLE_CHECKS_AT_RELEASE>"
    "$<$<BOOL:${NOA_CUDA_ENABLE_ASSERT}>:NOA_CUDA_ENABLE_ASSERT>"
    "$<$<BOOL:${FFTW3_THREADS}>:NOA_FFTW_THREADS>"
    "$<$<BOOL:${FFTW3_OPENMP}>:NOA_FFTW_THREADS>"
    )

# Set included directories:
target_include_directories(noa
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
    "$<BUILD_INTERFACE:${NOA_GENERATED_HEADERS_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/utils/Version.hpp.in"
    "${NOA_GENERATED_HEADERS_DIR}/noa/Version.hpp"
    @ONLY)

# Since it is static library only, the SOVERSION shouldn't matter.
set_target_properties(noa PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION})

# ---------------------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------------------
# TODO We only have static linking so: LIBRARY could be removed, same for RUNTIME although in Windows I'm not sure...
# TODO We don't use components, but maybe we could use them to specify what's inside the library? e.g. cuda/cpu backend.
install(TARGETS prj_common_option prj_compiler_warnings noa_private_libraries noa_public_libraries noa
    EXPORT noa

    INCLUDES
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"

    LIBRARY
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT noa_runtime
    NAMELINK_COMPONENT noa_development

    ARCHIVE
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT noa_development

    RUNTIME
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT noa_runtime
    )

# Headers:
foreach (FILE ${NOA_HEADERS})
    get_filename_component(DIR ${FILE} DIRECTORY)
    install(FILES ${FILE} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/noa/${DIR})
endforeach ()

# Generated headers:
install(FILES
    "${NOA_GENERATED_HEADERS_DIR}/noa/Version.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/noa")

message(STATUS "-> noa::noa: configuring public target... done")
message(STATUS "--------------------------------------\n")
