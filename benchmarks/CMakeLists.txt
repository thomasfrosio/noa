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
        src/gpu/cuda/math/BenchCUDAReductions.cpp
        )

if (NOA_ENABLE_CUDA)
    set(BENCHMARK_SOURCES ${BENCHMARK_SOURCES} ${BENCHMARK_CUDA_SOURCES})
endif ()

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa_benchmarks.cmake)
