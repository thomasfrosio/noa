message(STATUS "--------------------------------------")
message(STATUS "-> noa::noa: configuring public target...")

# ---------------------------------------------------------------------------------------
# Linking options and libraries
# ---------------------------------------------------------------------------------------
# Common:
include(${PROJECT_SOURCE_DIR}/cmake/ext/fmt.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/half.cmake)

# Interfaces gathering the dependencies.
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
        FFTW3::fftw3
        FFTW3::fftw3f
        )
endif ()

# CUDA backend:
if (NOA_ENABLE_CUDA)
    include(${PROJECT_SOURCE_DIR}/cmake/ext/cuda-toolkit.cmake)
    target_link_libraries(noa_public_libraries
        INTERFACE
        CUDA::cuda_driver
        $<IF:$<BOOL:${NOA_CUDA_STATIC}>, CUDA::cudart_static, CUDA::cudart>
        )
    target_link_libraries(noa_private_libraries
        INTERFACE
        $<IF:$<BOOL:${NOA_CUDA_STATIC}>, CUDA::cufft_static, CUDA::cufft>
        $<IF:$<BOOL:${NOA_CUDA_STATIC}>, CUDA::cublas_static, CUDA::cublas>
        )
endif ()

# ---------------------------------------------------------------------------------------
# Creating the target and set up the build
# ---------------------------------------------------------------------------------------
add_library(noa STATIC ${NOA_SOURCES} ${NOA_HEADERS})
add_library(noa::noa ALIAS noa)

# TODO Use target_sources instead
target_include_directories(noa
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
)

target_link_libraries(noa
    PRIVATE
    prj_compiler_private_options
    prj_compiler_warnings
    noa_private_libraries

    PUBLIC
    prj_compiler_public_options
    noa_public_libraries
    )

#set_target_properties(noa PROPERTIES POSITION_INDEPENDENT_CODE ON) # FIXME remove since prj_compiler_public_options should propagate it
if (NOA_ENABLE_CUDA)
    set_target_properties(noa PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

# Set definitions:
target_compile_definitions(noa
    PUBLIC
    NOA_ERROR_POLICY=${NOA_ERROR_POLICY}
    "$<$<CONFIG:DEBUG>:NOA_DEBUG>"
    "$<$<BOOL:${NOA_ENABLE_CPU}>:NOA_ENABLE_CPU>"
    "$<$<BOOL:${NOA_ENABLE_CUDA}>:NOA_ENABLE_CUDA>"
    "$<$<BOOL:${NOA_ENABLE_TIFF}>:NOA_ENABLE_TIFF>"
    "$<$<BOOL:${NOA_CPU_OPENMP}>:NOA_ENABLE_OPENMP>"
    "$<$<BOOL:${NOA_CPU_FFTW3_MULTITHREADED}>:NOA_CPU_FFTW3_MULTITHREADED>"
    )

# Since it is static library only, the SOVERSION shouldn't matter.
set_target_properties(noa PROPERTIES
    SOVERSION ${NOA_VERSION_MAJOR}
    VERSION ${NOA_VERSION}
    )

# ---------------------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------------------
install(
    TARGETS
    prj_compiler_public_options
    prj_compiler_private_options
    prj_compiler_warnings
    noa_private_libraries
    noa_public_libraries
    noa

    EXPORT noa

    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}

    ARCHIVE
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT noa_development
    )

# Install public headers:
foreach (FILE ${NOA_HEADERS})
    get_filename_component(DIR ${FILE} DIRECTORY)
    install(FILES ${FILE} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/noa/${DIR}")
endforeach ()

message(STATUS "-> noa::noa: configuring public target... done")
message(STATUS "--------------------------------------\n")
