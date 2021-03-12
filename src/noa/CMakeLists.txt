# ---------------------------------------------------------------------------------------
# Set up the source and header files
# ---------------------------------------------------------------------------------------
set(NOA_SOURCES
        Log.cpp

        # Utilities
        util/Inputs.cpp
        util/IO.cpp
        util/files/MRCFile.cpp
        util/files/ImageFile.cpp

        # CPU backend
        cpu/fourier/Plan.cpp

        )

set(NOA_HEADERS
        Definitions.h
        Errno.h
        Environment.h
        Exception.h
        Log.h
        Types.h

        # CPU backend
        cpu/PtrHost.h
        cpu/Threadpool.h
        cpu/Fourier.h
        cpu/fourier/Transforms.h
        cpu/fourier/Plan.h
        cpu/fourier/Resize.h
        cpu/fourier/Remap.h

        # Utilities
        util/Complex.h
        util/complex/Complex-inl.h
        util/complex/carith.h
        util/complex/cproj.h
        util/complex/cexp.h
        util/complex/clog.h
        util/complex/cpow.h
        util/complex/csqrt.h
        util/complex/math_private.h
        util/Flag.h
        util/FloatX.h
        util/Float2.h
        util/Float3.h
        util/Float4.h
        util/Inputs.h
        util/IntX.h
        util/Int2.h
        util/Int3.h
        util/Int4.h
        util/IO.h
        util/Math.h
        util/OS.h
        util/Traits.h
        util/traits/BaseTypes.h
        util/traits/Containers.h
        util/Sizes.h
        util/String.h
        util/string/Convert.h
        util/string/Format.h
        util/string/Parse.h
        util/files/ImageFile.h
        util/files/MRCFile.h
        util/files/TextFile.h


        )

# ---------------------------------------------------------------------------------------
# Options and libraries
# ---------------------------------------------------------------------------------------
add_library(noa_libraries INTERFACE)
add_library(noa_common_option INTERFACE)

target_link_libraries(noa_libraries
        INTERFACE
        Threads::Threads
        spdlog::spdlog
        # TIFF::TIFF
        fftw::float
        fftw::double
        fftw::float_threads
        fftw::double_threads
        )

# ---------------------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------------------
set(NOA_CUDA_SOURCES
        gpu/cuda/fourier/Remap.cu
        gpu/cuda/fourier/Resize.cu

        )

set(NOA_CUDA_HEADERS
        gpu/cuda/Exception.h
        gpu/cuda/Types.h

        #gpu/cuda/Kernel.h
        gpu/cuda/Memory.h
        gpu/cuda/PtrDevice.h
        gpu/cuda/PtrPinned.h
        gpu/cuda/PtrDevicePadded.h
        gpu/cuda/PtrArray.h
        gpu/cuda/PtrTexture.h

        # Fourier
        gpu/cuda/Fourier.h
        gpu/cuda/fourier/Exception.h
        gpu/cuda/fourier/Plan.h
        gpu/cuda/fourier/Transforms.h
        gpu/cuda/fourier/Resize.h
        gpu/cuda/fourier/Remap.h

        # Utilities
        gpu/cuda/util/Device.h
        gpu/cuda/util/Event.h
        gpu/cuda/util/Stream.h
        gpu/cuda/util/Sizes.h


        )

if (NOA_BUILD_CUDA)
    set(NOA_HEADERS ${NOA_HEADERS} ${NOA_CUDA_HEADERS})
    set(NOA_SOURCES ${NOA_SOURCES} ${NOA_CUDA_SOURCES})

    if (NOA_CUDA_USE_STATIC_LIBS)
        target_link_libraries(noa_libraries
                INTERFACE
                CUDA::cudart_static
                CUDA::cufft # TODO: compilation fails with noa_tests when using cufft_static...?
                CUDA::cublas
                )
    else ()
        target_link_libraries(noa_libraries
                INTERFACE
                CUDA::cudart
                CUDA::cublas
                CUDA::cufft
                )
    endif ()

    set_target_properties(noa_common_option
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${NOA_CUDA_ARCH})
endif ()

# ---------------------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------------------
add_library(noa ${NOA_SOURCES} ${NOA_HEADERS})
add_library(Noa::noa ALIAS noa)

target_link_libraries(noa
        PRIVATE
        prj_common_option
        prj_cxx_warnings
        noa_common_option
        PUBLIC
        noa_libraries
        )

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

# Some definitions
target_compile_definitions(noa
        PUBLIC
        "$<$<CONFIG:DEBUG>:NOA_DEBUG>"
        "$<$<BOOL:${NOA_BUILD_CUDA}>:NOA_BUILD_CUDA>"
        )

# Global includes. Used by all targets linking to the library (PUBLIC)
# Headers in source tree can be included as well as the noa_generated_headers
target_include_directories(noa
        PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
        "$<BUILD_INTERFACE:${NOA_GENERATED_HEADERS_DIR}>"
        "$<INSTALL_INTERFACE:.>"
        )

# ---------------------------------------------------------------------------------------
# Symbol visibility - NOTE: Currently not used since building shared lib is not supported
# ---------------------------------------------------------------------------------------
# Hide everything by default - export manually
set_target_properties(noa PROPERTIES
        CMAKE_CXX_VISIBILITY_PRESET hidden
        CMAKE_VISIBILITY_INLINES_HIDDEN YES)

# Generate the export header for noa.
#   - generates API.h
#   - ensure NOA_API is defined
include(GenerateExportHeader)
generate_export_header(noa
        EXPORT_MACRO_NAME NOA_API
        EXPORT_FILE_NAME ${NOA_GENERATED_HEADERS_DIR}/noa/API.h)

# ---------------------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------------------
# Targets:
#   - <prefix>/lib/libnoa.a
#   - header location after install: <prefix>/noa/*.h
#   - headers can be included by C++ code `#include <noa/*.h>`
install(TARGETS spdlog prj_common_option prj_cxx_warnings noa_common_option noa_libraries noa
        EXPORT "${NOA_TARGETS_EXPORT_NAME}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Headers:
#   - core/*.h -> <prefix>/include/noa/*.h
install(FILES ${NOA_HEADERS}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/noa")

# Headers:
#   - generated_headers/noa/version.h -> <prefix>/include/noa/version.h
install(FILES
        "${NOA_GENERATED_HEADERS_DIR}/noa/API.h"
        "${NOA_GENERATED_HEADERS_DIR}/noa/Version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/noa")

# Config
#   - <prefix>/lib/cmake/Foo/FooConfig.cmake
#   - <prefix>/lib/cmake/Foo/FooConfigVersion.cmake
install(FILES
        "${NOA_CONFIG_FILE}"
        "${NOA_CONFIG_VERSION_FILE}"
        DESTINATION "${NOA_INSTALL_LIBDIR}")

# Config
#   - <prefix>/lib/cmake/Foo/FooTargets.cmake
install(EXPORT "${NOA_TARGETS_EXPORT_NAME}"
        FILE "NoaTargets.cmake"
        DESTINATION "${NOA_INSTALL_LIBDIR}"
        NAMESPACE "Noa::")
