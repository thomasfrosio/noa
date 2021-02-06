# ---------------------------------------------------------------------------------------
# Set up the source and header files
# ---------------------------------------------------------------------------------------
set(NOA_SOURCES
        # Utilities
        Log.cpp
        util/IO.cpp
        util/files/MRCFile.cpp
        util/files/ImageFile.cpp

        # Managers
        util/Inputs.cpp

        )

set(NOA_HEADERS
        Base.h
        Backend.h   # Included in Base.h
        Define.h    # Included in Base.h
        Errno.h     # Included in Base.h
        Exception.h # Included in Base.h
        Log.h       # Included in Base.h
        Types.h     # Included in Base.h

        # Utilities
        util/Complex.h              # Included in Base.h
        util/complex/Complex-inl.h  # Included in Base.h
        util/complex/carith.h       # Included in Base.h
        util/complex/cproj.h        # Included in Base.h
        util/complex/cexp.h         # Included in Base.h
        util/complex/clog.h         # Included in Base.h
        util/complex/cpow.h         # Included in Base.h
        util/complex/csqrt.h        # Included in Base.h
        util/complex/math_private.h # Included in Base.h
        util/Flag.h
        util/FloatX.h               # Included in Base.h
        util/Float2.h               # Included in Base.h
        util/Float3.h               # Included in Base.h
        util/Float4.h               # Included in Base.h
        util/Inputs.h
        util/IntX.h                 # Included in Base.h
        util/Int2.h                 # Included in Base.h
        util/Int3.h                 # Included in Base.h
        util/Int4.h                 # Included in Base.h
        util/IO.h
        util/Math.h                 # Included in Base.h
        util/OS.h
        util/Traits.h
        util/traits/BaseTypes.h     # Included in Base.h
        util/traits/Containers.h
        util/String.h
        util/string/Convert.h
        util/string/Format.h        # Included in Base.h
        util/string/Parse.h
        util/files/ImageFile.h
        util/files/MRCFile.h
        util/files/TextFile.h

        # CPU backend
        cpu/PtrHost.h
        cpu/Threadpool.h

        # GPU backend
        gpu/Base.h

        )

# ---------------------------------------------------------------------------------------
# Options and libraries
# ---------------------------------------------------------------------------------------
add_library(noa_libraries INTERFACE)
add_library(noa_compiler_option INTERFACE)

target_link_libraries(noa_libraries
        INTERFACE
        spdlog::spdlog
        # TIFF::TIFF
        )

# ---------------------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------------------
set(NOA_CUDA_SOURCES

        )

set(NOA_CUDA_HEADERS
        gpu/cuda/Memory.h

        gpu/cuda/PtrDevice.h
        gpu/cuda/PtrPinned.h
        gpu/cuda/PtrDevicePitch.h

        gpu/cuda/Device.h
        gpu/cuda/Exception.h
        gpu/cuda/Base.h
        gpu/cuda/Stream.h

        gpu/cuda/FFT.h
        gpu/cuda/fft/Exception.h



        )

if (NOA_BUILD_CUDA)
    set(NOA_HEADERS ${NOA_HEADERS} ${NOA_CUDA_HEADERS})
    set(NOA_SOURCES ${NOA_SOURCES} ${NOA_CUDA_SOURCES} )

    target_link_libraries(noa_libraries
            INTERFACE
            CUDA::cudart_static
            CUDA::cublas_static
            CUDA::cufft_static)

    set_target_properties(noa_compiler_option
            PROPERTIES
            # CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${NOA_CUDA_ARCH})
endif ()

# ---------------------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------------------
add_library(noa ${NOA_SOURCES} ${NOA_HEADERS})
add_library(Noa::noa ALIAS noa)

target_link_libraries(noa
        PRIVATE
        prj_compiler_option
        noa_compiler_option
        PUBLIC
        noa_libraries
        )

if (NOA_ENABLE_PCH)
    target_precompile_headers(noa
            PRIVATE
            ${PROJECT_SOURCE_DIR}/src/noa/Base.h
            )
endif ()

# Some definitions
target_compile_definitions(noa
        PUBLIC
        "$<$<CONFIG:DEBUG>:NOA_DEBUG>"
        "$<$<BOOL:${NOA_BUILD_CUDA}>:NOA_BUILD_CUDA>"
        "$<$<BOOL:${NOA_BUILD_OPENCL}>:NOA_BUILD_OPENCL>"
        "$<$<BOOL:${NOA_ENABLE_SINGLE_PRECISION}>:NOA_SINGLE_PRECISION>"
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
install(TARGETS spdlog prj_compiler_option noa_compiler_option noa_libraries noa
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
