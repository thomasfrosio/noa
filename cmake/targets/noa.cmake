message(STATUS "--------------------------------------")
message(STATUS "-> noa::noa: configuring public target...")

# Directories to store generated source files (that include headers, hence the public/private).
set(NOA_GENERATED_SOURCES_PUBLIC_DIR "${NOA_BINARY_DIR}/noa_generated_sources_public")
set(NOA_GENERATED_SOURCES_PRIVATE_DIR "${NOA_BINARY_DIR}/noa_generated_sources_private")

# ---------------------------------------------------------------------------------------
# Linking options and libraries
# ---------------------------------------------------------------------------------------
# Common:
include(${PROJECT_SOURCE_DIR}/cmake/ext/fmt.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/half.cmake)

# Interface gathering all of the dependencies of the library.
add_library(noa_public_libraries INTERFACE)
add_library(noa_private_libraries INTERFACE)

# Core:
target_link_libraries(noa_public_libraries
    INTERFACE
    fmt::fmt
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
        # The OpenMP::OpenMP_CXX target sets INTERFACE_COMPILE_OPTIONS to CXX only, so CUDA source
        # files will not have OpenMP. Therefore, we need to pass the flags/include/lib directly
        # so that CMake adds OpenMP to every of our source files (as well as the application source
        # files)
        target_compile_options(noa_public_libraries INTERFACE ${OpenMP_CXX_FLAGS})
        target_include_directories(noa_public_libraries INTERFACE ${OpenMP_CXX_INCLUDE_DIRS})
        target_link_libraries(noa_public_libraries INTERFACE ${OpenMP_CXX_LIBRARIES})
    endif ()

    target_link_libraries(noa_public_libraries
        INTERFACE
        Threads::Threads
        )

    include(${PROJECT_SOURCE_DIR}/cmake/ext/eigen.cmake)
    include(${PROJECT_SOURCE_DIR}/cmake/ext/fftw.cmake)
    target_link_libraries(noa_private_libraries
        INTERFACE
        Eigen3::Eigen
        ${FFTW3_TARGETS}
        )
endif ()

# CUDA backend:
if (NOA_ENABLE_CUDA)
    include(${PROJECT_SOURCE_DIR}/cmake/ext/cuda-toolkit.cmake)
    include(${PROJECT_SOURCE_DIR}/cmake/ext/cuda-rtc.cmake)
    target_link_libraries(noa_public_libraries
        INTERFACE
        CUDA::cuda_driver
        $<IF:$<BOOL:${NOA_CUDA_STATIC}>, CUDA::cudart_static, CUDA::cudart>
        $<IF:$<BOOL:${NOA_CUDA_STATIC}>, CUDA::cufft_static, CUDA::cufft>
        )
    target_link_libraries(noa_private_libraries
        INTERFACE
        $<IF:$<BOOL:${NOA_CUDA_STATIC}>, CUDA::cublas_static, CUDA::cublas>
        cuda_rtc::jitify2
        )

    # Preprocess CUDA kernels.
    if (NOA_CUDA_JIT)
        set(noa_jit_include_directories ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src)
        list(TRANSFORM noa_jit_include_directories PREPEND -I)

        set(noa_jit_generated_dir ${NOA_GENERATED_SOURCES_PRIVATE_DIR})
        set(noa_jit_generated_sources ${NOA_CUDA_PREPROCESS_SOURCES})
        list(TRANSFORM noa_jit_generated_sources PREPEND ${noa_jit_generated_dir}/)
        list(TRANSFORM noa_jit_generated_sources APPEND .jit.cpp)
        list(APPEND noa_jit_generated_sources ${NOA_GENERATED_SOURCES_PRIVATE_DIR}/noa_shared_headers.jit.cpp)

        # TODO If cuda is linked statically, let jitify parse and include the cuda headers (current behavior).
        #      Otherwise, pass the include dir to nvrtc and check for the path at runtime e.g. using an env variable.
        # FIXME cuda_f16.h, cuda.h, cuda_runtime.h should not be patched, or anything in the cuda toolkit include dir.
        add_custom_command(
            OUTPUT
            ${noa_jit_generated_sources}

            COMMAND cuda_rtc::preprocess ${NOA_CUDA_PREPROCESS_SOURCES}
            # preprocess
            --output-directory ${noa_jit_generated_dir}
            --shared-headers shared_headers
            --namespace noa

            # jitify2
            ${noa_jit_include_directories}
            --std=c++${CMAKE_CXX_STANDARD}
            --cuda-std
            --minify
            --no-replace-pragma-once
            --no-builtin-headers

            # nvrtc options
            --include-path=${CUDAToolkit_INCLUDE_DIRS}

            # Make it depends on the core headers and the cuda headers.
            # Some of these headers are not actually passed to nvrtc, so we might
            # rerun this command for nothing, but it's better than missing a dependency.
            # Note: we don't have private headers atm, so while named "public",
                    these are all the headers we have.
            DEPENDS cuda_rtc::preprocess ${NOA_CORE_HEADERS} ${NOA_CUDA_HEADERS}

            # The working directory is quite important because nvrtc includes it to resolve quoted
            # #includes. This is a problem because we want jitify to intercept every #include statement
            # so an implicitly included directory is not great. We could use the --no-source-include
            # options from nvrtc, but this break transitive includes with quotes (cuda/std/type_traits
            # doesn't compile with that option for example). The solution is to make sure the cwd cannot
            # be used by nvrtc to resolve quoted #include statements. In noa, all of our quoted includes
            # are to include our headers and all of them are relative to src/, eg #include "noa/Array.hpp",
            # so we need to make sure that the working directory is not src/. Using src/noa/ is actually
            # perfect as long as we don't do eg #include "Array.hpp", which we don't. src/noa/ also
            # preserves the include structure in the output directory, so all good!
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/src/noa"
            VERBATIM
        )

        # We need to add the generated sources to the list of sources for the library.
        # These are absolute paths, so we can just add them to the list.
        list(APPEND NOA_SOURCES ${noa_jit_generated_shared_headers_source})
    endif ()
endif ()

# Version file.
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/utils/Version.hpp.in"
    "${NOA_GENERATED_SOURCES_PUBLIC_DIR}/noa/Version.hpp"
    @ONLY)

# ---------------------------------------------------------------------------------------
# Creating the target and set up the build
# ---------------------------------------------------------------------------------------
add_library(noa STATIC ${NOA_SOURCES} ${NOA_HEADERS})
add_library(noa::noa ALIAS noa)

target_include_directories(noa
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
    "$<BUILD_INTERFACE:${NOA_GENERATED_SOURCES_PUBLIC_DIR}>"
    "$<BUILD_INTERFACE:${NOA_GENERATED_SOURCES_PRIVATE_DIR}>"
)

target_link_libraries(noa
    PRIVATE prj_common_option prj_compiler_warnings
    PRIVATE noa_private_libraries
    PUBLIC noa_public_libraries
    )

set_target_properties(noa
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
if (NOA_ENABLE_CUDA AND NOT NOA_CUDA_JIT)
    set_target_properties(noa PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
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
    "$<$<BOOL:${NOA_CUDA_JIT}>:NOA_CUDA_JIT>"
    "$<$<BOOL:${NOA_CUDA_ENABLE_ASSERT}>:NOA_CUDA_ENABLE_ASSERT>"
    "$<$<BOOL:${FFTW3_THREADS}>:NOA_FFTW_THREADS>"
    "$<$<BOOL:${FFTW3_OPENMP}>:NOA_FFTW_THREADS>"
    )

# Since it is static library only, the SOVERSION shouldn't matter.
set_target_properties(noa PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION})

# ---------------------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------------------
# TODO We only have static linking so: LIBRARY could be removed, same for RUNTIME although in Windows I'm not sure...
# TODO We don't use components, but maybe we could use them to specify what's inside the library? e.g. cuda/cpu backend.
install(
    TARGETS prj_common_option prj_compiler_warnings noa_private_libraries noa_public_libraries noa
    EXPORT noa

    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}

    LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT noa_runtime
    NAMELINK_COMPONENT noa_development

    ARCHIVE
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT noa_development

    RUNTIME
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT noa_runtime
    )

# Install public headers:
foreach (FILE ${NOA_HEADERS})
    get_filename_component(DIR ${FILE} DIRECTORY)
    install(FILES ${FILE} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/noa/${DIR}")
endforeach ()

# Generated headers:
# Note: For now, do it manually since we only have one public header.
install(FILES
    "${NOA_GENERATED_SOURCES_PUBLIC_DIR}/noa/Version.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/noa")

message(STATUS "-> noa::noa: configuring public target... done")
message(STATUS "--------------------------------------\n")
