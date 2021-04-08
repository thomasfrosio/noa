enable_testing()

# ---------------------------------------------------------------------------------------
# Sets the benchmark files
# ---------------------------------------------------------------------------------------
set(BENCHMARK_SOURCES
        EntryPoint.cpp
        Helpers.h
        ${PROJECT_SOURCE_DIR}/tests/Helpers.h
        )

# ---------------------------------------------------------------------------------------
# Sets the benchmark files using the CUDA backend
# ---------------------------------------------------------------------------------------
set(BENCHMARK_CUDA_SOURCES
        src/gpu/cuda/math/BenchCUDAArithmetics.cpp

        src/gpu/cuda/math/BenchCUDAReductions.cpp)

if (NOA_BUILD_CUDA)
    set(BENCHMARK_SOURCES ${BENCHMARK_SOURCES} ${BENCHMARK_CUDA_SOURCES})
endif ()

# ---------------------------------------------------------------------------------------
# Built target
# ---------------------------------------------------------------------------------------
include(${PROJECT_SOURCE_DIR}/ext/catch2/catch2.cmake)

add_executable(noa_benchmarks ${BENCHMARK_SOURCES})
target_link_libraries(noa_benchmarks
        PRIVATE
        prj_common_option
        prj_cxx_warnings
        Noa::noa
        Catch2::Catch2
        )

target_precompile_headers(noa_benchmarks
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src/noa/Definitions.h
        ${PROJECT_SOURCE_DIR}/src/noa/Errno.h
        ${PROJECT_SOURCE_DIR}/src/noa/Exception.h
        ${PROJECT_SOURCE_DIR}/src/noa/Logger.h
        ${PROJECT_SOURCE_DIR}/src/noa/Types.h
        )

# Global includes. Used by all targets linking to the library (PUBLIC)
# Headers in source tree can be included as well as the noa_generated_headers
target_include_directories(noa_benchmarks
        PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/benchmarks>"
        "$<INSTALL_INTERFACE:.>"
        )
