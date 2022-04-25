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
#        src/cpu/fft/BenchCPURemap.cpp
#        src/cpu/fft/BenchCPUResize.cpp
#        src/cpu/fft/BenchCPUTransforms.cpp
#
#        src/cpu/signal/BenchCPUConvolve.cpp
#        src/cpu/signal/BenchCPUMedian.cpp
#        src/cpu/signal/BenchCPUShape.cpp
#
#        src/cpu/memory/BenchCPUCast.cpp
#        src/cpu/memory/BenchCPUTranspose.cpp
#
#        src/cpu/transform/BenchCPURotate.cpp
#        src/cpu/transform/BenchCPUSymmetry.cpp
#        src/cpu/transform/fft/BenchCPURotate.cpp
#        src/cpu/transform/fft/BenchCPUShift.cpp
        )

set(BENCHMARK_CUDA_SOURCES
        src/gpu/cuda/fft/BenchCUDAFilters.cpp

#        src/gpu/cuda/math/BenchCUDAArithmetics.cpp
#        src/gpu/cuda/math/BenchCUDAReductions.cpp
        )

if (NOA_ENABLE_CUDA)
    set(BENCHMARK_SOURCES ${BENCHMARK_SOURCES} ${BENCHMARK_CPU_SOURCES} ${BENCHMARK_CUDA_SOURCES})
endif ()

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa_benchmarks.cmake)
