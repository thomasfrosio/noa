message(STATUS "--------------------------------------")
message(STATUS "noa::noa: configuring public target...")

# ---------------------------------------------------------------------------------------
# Linking options and libraries
# ---------------------------------------------------------------------------------------
# Common:
include(${PROJECT_SOURCE_DIR}/ext/spdlog/spdlog.cmake)
include(${PROJECT_SOURCE_DIR}/ext/half/half.cmake)

add_library(noa_libraries INTERFACE)
target_link_libraries(noa_libraries
        INTERFACE
        spdlog::spdlog
        half-ieee754
        )

if (NOA_ENABLE_TIFF)
    include(${PROJECT_SOURCE_DIR}/ext/tiff/tiff.cmake)
    target_link_libraries(noa_libraries INTERFACE TIFF::TIFF)
endif ()

# CPU backend:
if (NOA_ENABLE_CPU OR NOA_ENABLE_UNIFIED)
    find_package(Threads REQUIRED)

    if (NOA_ENABLE_OPENMP)
        find_package(OpenMP 4.5)
        if (OpenMP_FOUND)
            target_link_libraries(noa_libraries INTERFACE OpenMP::OpenMP_CXX)
        else ()
            message(WARN "NOA_ENABLE_OPENMP is ON, but could not find OpenMP")
            set(NOA_ENABLE_OPENMP OFF)
        endif()
    endif ()

    include(${PROJECT_SOURCE_DIR}/ext/openblas/openblas.cmake)
    include(${PROJECT_SOURCE_DIR}/ext/fftw/fftw.cmake)
    target_link_libraries(noa_libraries
            INTERFACE
            Threads::Threads
            OpenBLAS::OpenBLAS
            fftw3::float
            fftw3::double
            )
endif ()

# CUDA backend:
if (NOA_ENABLE_CUDA)
    include(${PROJECT_SOURCE_DIR}/ext/cuda-toolkit/cuda-toolkit.cmake)
    if (NOA_CUDA_CUDART_STATIC)
        target_link_libraries(noa_libraries INTERFACE CUDA::cudart_static)
    else ()
        target_link_libraries(noa_libraries INTERFACE CUDA::cudart)
    endif ()

    if (NOA_CUDA_CUFFT_STATIC)
        target_link_libraries(noa_libraries INTERFACE CUDA::cufft_static)
    else ()
        target_link_libraries(noa_libraries INTERFACE CUDA::cufft)
    endif ()

    if (NOA_CUDA_CURAND_STATIC)
        target_link_libraries(noa_libraries INTERFACE CUDA::curand_static)
    else ()
        target_link_libraries(noa_libraries INTERFACE CUDA::curand)
    endif ()

    if (NOA_CUDA_CUBLAS_STATIC)
        target_link_libraries(noa_libraries INTERFACE CUDA::cublas_static)
    else ()
        target_link_libraries(noa_libraries INTERFACE CUDA::cublas)
    endif ()

    if (NOA_CUDA_CUSOLVER_STATIC)
        target_link_libraries(noa_libraries INTERFACE CUDA::cusolver_static)
    else ()
        target_link_libraries(noa_libraries INTERFACE CUDA::cusolver)
    endif ()

    target_compile_options(noa_libraries INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda>)
endif ()

# ---------------------------------------------------------------------------------------
# Creating the target
# ---------------------------------------------------------------------------------------
add_library(noa ${NOA_SOURCES} ${NOA_HEADERS})
add_library(noa::noa ALIAS noa)

target_link_libraries(noa
        PRIVATE
        prj_common_option
        prj_compiler_warnings
        PUBLIC
        noa_libraries
        )

# Not sure why, but these need to be set on the target directly
if (NOA_ENABLE_CUDA)
    set_target_properties(noa
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            # CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_ARCHITECTURES ${NOA_CUDA_ARCH}
            )
endif ()

if (NOA_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(
            RESULT
            result
            OUTPUT
            output)
    if (result)
        set_target_properties(noa PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    else ()
        message(SEND_ERROR "IPO is not supported: ${output}")
    endif ()
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
        "$<$<BOOL:${NOA_ENABLE_UNIFIED}>:NOA_ENABLE_UNIFIED>"
        "$<$<BOOL:${NOA_ENABLE_TIFF}>:NOA_ENABLE_TIFF>"
        "$<$<BOOL:${NOA_ENABLE_OPENMP}>:NOA_ENABLE_OPENMP>"
        "$<$<BOOL:${NOA_FFTW_THREADS}>:NOA_FFTW_THREADS>"
        "$<$<BOOL:${NOA_ENABLE_CHECKS_RELEASE}>:NOA_ENABLE_CHECKS_RELEASE>"
        )

# Set included directories:
target_include_directories(noa
        PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
        "$<BUILD_INTERFACE:${NOA_GENERATED_HEADERS_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )

# ---------------------------------------------------------------------------------------
# API Compatibility - Versioning
# ---------------------------------------------------------------------------------------
# The library uses the semantic versioning: MAJOR.MINOR.PATCH
# PATCH: Bug fix only. No API changes.
# MINOR: Non-breaking additions to the API (i.e. something was added).
#        W ensure that if someone had built against the previous version, they could simply
#        rebuild their application with the new version without having to change the way they
#        interact with the library. When updated, PATCH should be reset to 0.
# MAJOR: Breaking change. Reset MINOR and PATCH to 0.
configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/settings/Version.h.in"
        "${NOA_GENERATED_HEADERS_DIR}/noa/Version.h"
        @ONLY)

# NOTE: Since it is static library only, the SOVERSION doesn't matter much.
set_target_properties(noa PROPERTIES
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION})

# ---------------------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------------------
# Targets:
#   - <install_path>/lib/libnoa(b).(a|so) (on Linux)
#   - header location after install: <prefix>/noa/*.h
#   - headers can be included by C++ code `#include <noa/*.h>`
install(TARGETS spdlog half-ieee754 prj_common_option prj_compiler_warnings noa_libraries noa
        EXPORT "${NOA_TARGETS_EXPORT_NAME}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"

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
#   - *.h -> <install_path>/include/noa/*.h
foreach (FILE ${NOA_HEADERS})
    get_filename_component(DIR ${FILE} DIRECTORY)
    install(FILES ${FILE} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/noa/${DIR})
endforeach ()

# Headers:
#   - <build_path>/noa_generated_headers/noa/Version.h -> <install_path>/include/noa/Version.h
install(FILES
        "${NOA_GENERATED_HEADERS_DIR}/noa/Version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/noa")

# Package config:
#   - <install_path>/lib/cmake/noa/NoaConfig.cmake
#   - <install_path>/lib/cmake/noa/NoaConfigVersion.cmake
install(FILES
        "${NOA_CONFIG_FILE}"
        "${NOA_CONFIG_VERSION_FILE}"
        DESTINATION "${NOA_INSTALL_LIBDIR}")

# Package config:
#   - <install_path>/lib/cmake/Noa/NoaTargets.cmake
install(EXPORT "${NOA_TARGETS_EXPORT_NAME}"
        FILE "${NOA_TARGETS_EXPORT_NAME}.cmake"
        DESTINATION "${NOA_INSTALL_LIBDIR}"
        NAMESPACE "noa::")

message(STATUS "noa::noa: configuring public target... done")
message(STATUS "--------------------------------------\n")
