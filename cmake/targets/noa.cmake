message(STATUS "--------------------------------------")
message(STATUS "-> noa::noa: configuring public target...")

# Creating the target and its source files.
add_library(noa STATIC)
add_library(noa::noa ALIAS noa)
target_sources(noa
    PRIVATE ${NOA_SOURCES}
    PUBLIC FILE_SET HEADERS
        BASE_DIRS "${PROJECT_SOURCE_DIR}/src"
        FILES ${NOA_HEADERS}
)

# Find or fetch unconditional dependencies.
find_package(Threads REQUIRED)
include(${PROJECT_SOURCE_DIR}/cmake/ext/fmt.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/half.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/eigen.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/fftw.cmake)

# Link compiler options and warnings, and dependencies.
target_link_libraries(noa
    PRIVATE
    prj_compiler_private_options
    prj_compiler_warnings
    $<BUILD_INTERFACE:Eigen3::Eigen>

    PUBLIC
    prj_compiler_public_options
    Threads::Threads
    fmt::fmt
    half::half
    FFTW3::fftw3
    FFTW3::fftw3f
)

# Optional OpenMP support.
if (NOA_CPU_OPENMP)
    find_package(OpenMP 4.5 REQUIRED)
    # OpenMP pragmas are included in public headers, so they need public visibility.
    # The OpenMP::OpenMP_CXX target sets INTERFACE_COMPILE_OPTIONS to CXX only, so CUDA source files
    # will not have OpenMP. Therefore, add the flags/include/lib directly regardless of the language.
    target_compile_options(noa PUBLIC ${OpenMP_CXX_FLAGS})
    target_include_directories(noa SYSTEM PUBLIC ${OpenMP_CXX_INCLUDE_DIRS})
    target_link_libraries(noa PUBLIC ${OpenMP_CXX_LIBRARIES})
endif ()

# Optional TIFF support.
if (NOA_ENABLE_TIFF)
    include(${PROJECT_SOURCE_DIR}/cmake/ext/tiff.cmake)
    target_link_libraries(noa PRIVATE TIFF::TIFF)
endif ()

# CUDA backend:
if (NOA_ENABLE_CUDA)
    include(${PROJECT_SOURCE_DIR}/cmake/ext/cuda-toolkit.cmake)
    target_link_libraries(noa
        PUBLIC
            CUDA::cuda_driver
            $<IF:$<BOOL:${NOA_CUDA_STATIC}>, CUDA::cudart_static, CUDA::cudart>
        PRIVATE
            $<IF:$<BOOL:${NOA_CUDA_STATIC}>, CUDA::cufft_static, CUDA::cufft>
            $<IF:$<BOOL:${NOA_CUDA_STATIC}>, CUDA::cublas_static, CUDA::cublas>
    )
    set_target_properties(noa PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif ()

# Set definitions.
target_compile_definitions(noa
    PUBLIC
    NOA_ERROR_POLICY=${NOA_ERROR_POLICY}
    $<$<CONFIG:Debug>:NOA_DEBUG>
    $<$<BOOL:${NOA_ENABLE_CPU}>:NOA_ENABLE_CPU>
    $<$<BOOL:${NOA_ENABLE_CUDA}>:NOA_ENABLE_CUDA>
    $<$<BOOL:${NOA_ENABLE_TIFF}>:NOA_ENABLE_TIFF>
    $<$<BOOL:${NOA_CPU_OPENMP}>:NOA_ENABLE_OPENMP>
    $<$<BOOL:${NOA_CPU_FFTW3_MULTITHREADED}>:NOA_CPU_FFTW3_MULTITHREADED>
)

# Versioning support.
set_target_properties(noa PROPERTIES
    SOVERSION ${NOA_VERSION_MAJOR} # FIXME used only for shared library
    VERSION ${NOA_VERSION}
    EXPORT_NAME noa
)

# Installation support.
install(
    TARGETS
    prj_compiler_public_options
    prj_compiler_private_options
    prj_compiler_warnings
    noa

    EXPORT noa
    FILE_SET HEADERS

    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

message(STATUS "-> noa::noa: configuring public target... done")
message(STATUS "--------------------------------------\n")
