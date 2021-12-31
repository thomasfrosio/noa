# ---------------------------------------------------------------------------------------
# Sets the benchmark files
# ---------------------------------------------------------------------------------------
set(BENCHMARK_SOURCES
        EntryPoint.cpp
        Helpers.h
        ${PROJECT_SOURCE_DIR}/tests/Helpers.h
        )

set(BENCHMARK_CPU_SOURCES
        src/cpu/fft/BenchCPUFilters.cpp
        src/cpu/fft/BenchCPURemap.cpp
        src/cpu/fft/BenchCPUResize.cpp

        )

set(BENCHMARK_CUDA_SOURCES
        src/gpu/cuda/math/BenchCUDAArithmetics.cpp
        src/gpu/cuda/math/BenchCUDAReductions.cpp
        )

if (NOA_ENABLE_CUDA)
    set(BENCHMARK_SOURCES ${BENCHMARK_SOURCES} ${BENCHMARK_CPU_SOURCES} ${BENCHMARK_CUDA_SOURCES})
endif ()

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa_benchmarks.cmake)
