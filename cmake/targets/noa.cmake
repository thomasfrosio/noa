message(STATUS "Configuring target: noa::noa_static")

# ---------------------------------------------------------------------------------------
# Options and libraries
# ---------------------------------------------------------------------------------------

add_library(noa_libraries INTERFACE)
add_library(noa_options INTERFACE)

target_link_libraries(noa_libraries
        INTERFACE
        Threads::Threads
        spdlog::spdlog
        TIFF::TIFF
        half-ieee754
        )

# ---------------------------------------------------------------------------------------
# CPU backend
# ---------------------------------------------------------------------------------------
if (NOA_ENABLE_CPU)
    set(NOA_HEADERS ${NOA_HEADERS} ${NOA_CPU_HEADERS})
    set(NOA_SOURCES ${NOA_SOURCES} ${NOA_CPU_SOURCES})

    target_link_libraries(noa_libraries
            INTERFACE
            fftw3::float
            fftw3::double
            )

    find_package(OpenMP 4.5 REQUIRED)
    if (NOA_ENABLE_OPENMP)
        target_link_libraries(noa_libraries
                INTERFACE
                OpenMP::OpenMP_CXX
                )
    endif ()

    # FFTW3 multithreading: requires either the library using the system threads or the OpenMP threads.
    if (NOA_FFTW_USE_THREADS)
        if (NOA_ENABLE_OPENMP)
            if (NOA_FFTW_FLOAT_OPENMP_LIB_FOUND AND NOA_FFTW_DOUBLE_OPENMP_LIB_FOUND)
                target_link_libraries(noa_libraries
                        INTERFACE
                        fftw3::float_threads # fftw3::float_omp
                        fftw3::double_threads # fftw3::double_omp
                        # The OMP version is taking a LOT of time for some transforms... it doesn't seem right...
                        )
            else ()
                message(FATAL_ERROR "With NOA_ENABLE_OPENMP and NOA_FFTW_USE_THREADS, the OpenMP versions of fftw3 are required but could not be found")
            endif ()
        else ()
            if (NOA_FFTW_FLOAT_THREADS_LIB_FOUND AND NOA_FFTW_DOUBLE_THREADS_LIB_FOUND)
                target_link_libraries(noa_libraries
                        INTERFACE
                        fftw3::float_threads
                        fftw3::double_threads
                        )
            else ()
                message(FATAL_ERROR "With NOA_FFTW_USE_THREADS, the multi-threaded versions of fftw3 are required but could not be found")
            endif ()
        endif ()
    endif ()
endif ()

# ---------------------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------------------
if (NOA_ENABLE_CUDA)
    set(NOA_HEADERS ${NOA_HEADERS} ${NOA_CUDA_HEADERS})
    set(NOA_SOURCES ${NOA_SOURCES} ${NOA_CUDA_SOURCES})

    # TODO compilation fails with noa_tests when using cufft_static...?
    #      Maybe look here: https://github.com/arrayfire/arrayfire/blob/master/src/backend/cuda/CMakeLists.txt
    target_link_libraries(noa_libraries
            INTERFACE
            CUDA::cudart
            CUDA::cufft
            )
endif ()

# ---------------------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------------------
add_library(noa_static ${NOA_SOURCES} ${NOA_HEADERS})
add_library(noa::noa_static ALIAS noa_static)

target_link_libraries(noa_static
        PRIVATE
        prj_common_option
        prj_compiler_warnings
        noa_options
        PUBLIC
        noa_libraries
        )

# Not sure why, but these need to be set on the target directly
if (NOA_ENABLE_CUDA)
    set_target_properties(noa_static
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            # CUDA_RESOLVE_DEVICE_SYMBOLS ON
            `CUDA_ARCHITECTURES` ${NOA_CUDA_ARCH}
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
        set_target_properties(noa_static PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    else ()
        message(SEND_ERROR "IPO is not supported: ${output}")
    endif ()
endif ()

if (NOA_ENABLE_PCH)
    target_precompile_headers(noa_static
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
target_compile_definitions(noa_static
        PUBLIC
        "$<$<CONFIG:DEBUG>:NOA_DEBUG>"
        "$<$<BOOL:${NOA_ENABLE_PROFILER}>:NOA_PROFILE>"
        "$<$<BOOL:${NOA_ENABLE_CUDA}>:NOA_ENABLE_CUDA>"
        "$<$<BOOL:${NOA_ENABLE_TIFF}>:NOA_ENABLE_TIFF>"
        "$<$<BOOL:${NOA_ENABLE_OPENMP}>:NOA_ENABLE_OPENMP>"
        "$<$<BOOL:${NOA_FFTW_USE_THREADS}>:NOA_FFTW_USE_THREADS>"
        )

# Set included directories:
target_include_directories(noa_static
        PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
        "$<BUILD_INTERFACE:${NOA_GENERATED_HEADERS_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )

# ---------------------------------------------------------------------------------------
# API Compatibility - Versioning
# ---------------------------------------------------------------------------------------
configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/settings/Version.h.in"
        "${NOA_GENERATED_HEADERS_DIR}/noa/Version.h"
        @ONLY)

# NOTE: Since it is static library only, the SOVERSION doesn't matter much for now.
#
# The library uses the semantic versioning: MAJOR.MINOR.PATCH
# PATCH: Bug fix only. No API changes.
# MINOR: Non-breaking additions to the API (i.e. something was added).
#        This ensure that if someone had built against the previous version, they could simply
#        replace with the new version without having to rebuild their application.
#        When updated, PATCH should be reset to 0.
# MAJOR: Breaking change. Reset MINOR and PATCH to 0.

# In the case of shared library:
# UNIX-based:
#   libnoa.so                       : NAME LINK (for build-time linker)
#   libnoa.so.SOVERSION             : SONAME (for runtime loader)
#   libnoa.so.MAJOR.MINOR.PATCH     : REAL LIBRARY (for human and packages)
#
#   The SONAME is used to specify the build version and API version respectively. Specifying the
#   SOVERSION is important, since CMake defaults it to the VERSION, effectively saying that any
#   MINOR or PATCH should break the API (which can break the runtime loader). The SOVERSION will
#   be set on the noa target and in our case will be equal to the PROJECT_VERSION_MAJOR.
#
# Windows:
#   noa.dll     : Acts kind of the SONAME
#   noa.lib     : Acts kind of the NAME LINK
#   Some version details may be encoded into the binaries (if Makefile or Ninja),
#   but this is usually not used.

set_target_properties(noa_static PROPERTIES
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION})

# ---------------------------------------------------------------------------------------
# Symbol visibility
# ---------------------------------------------------------------------------------------
# NOTE: Since it is static library only, ignore the export/import for now.
#
# Visual Studio:
#   Visual Studio hides symbols by default.
#   The attribute __declspec(dllexport) is used when building the library.
#   The attribute __declspec(dllimport) is used when using the library.
# GCC and Clang:
#   GCC and Clang do NOT hide symbols by default.
#   Compiler option -fvisibility=hidden to change default visibility to hidden.
#   Compiler option -fvisibility-inlines-hidden to change visibility of inlined code (including templates).
#   Then, use __attribute__((visibility("default"))) to make something visible visible.

# Hides everything by default - export manually (for Visual Studio, do nothing).
#set_target_properties(noa PROPERTIES
#        CMAKE_CXX_VISIBILITY_PRESET hidden
#        CMAKE_VISIBILITY_INLINES_HIDDEN YES)

# Generate the attribute given the compiler and usage (build vs using). CMake does that nicely so
# we don't have to distinguish between the different scenarios.
#   - generates noa/API.h
#   - ensure the macro NOA_API is defined. This can be added to classes, free functions and global variables;
#include(GenerateExportHeader)
#generate_export_header(noa
#        EXPORT_MACRO_NAME NOA_API
#        EXPORT_FILE_NAME ${NOA_GENERATED_HEADERS_DIR}/noa/API.h)

# ---------------------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------------------
# Targets:
#   - <install_path>/lib/libnoa(b).(a|so)
#   - header location after install: <prefix>/noa/*.h
#   - headers can be included by C++ code `#include <noa/*.h>`
install(TARGETS spdlog half-ieee754 prj_common_option prj_compiler_warnings noa_options noa_libraries noa_static
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
#   - <build_path>/noa_generated_headers/noa/API.h     -> <install_path>/include/noa/API.h
install(FILES
        # "${NOA_GENERATED_HEADERS_DIR}/noa/API.h"
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

message("")
