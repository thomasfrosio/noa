# ---------------------------------------------------------------------------------------
# Sets the benchmark files
# ---------------------------------------------------------------------------------------
set(BENCHMARK_SOURCES
        EntryPoint.cpp
        Helpers.h
        ${PROJECT_SOURCE_DIR}/tests/Helpers.h
        )

set(BENCHMARK_CPU_SOURCES
        src/cpu/fft/BenchCPURemap.cpp
        src/cpu/fft/BenchCPUResize.cpp
        src/cpu/fft/BenchCPUTransforms.cpp

        src/cpu/math/BenchCPUReductions.cpp

        src/cpu/signal/BenchCPUConvolve.cpp
#        src/cpu/signal/BenchCPUMedian.cpp
#        src/cpu/signal/BenchCPUShape.cpp
#
#        src/cpu/memory/BenchCPUCast.cpp
#        src/cpu/memory/BenchCPUTranspose.cpp
#
        src/cpu/geometry/BenchCPURotate.cpp
        src/cpu/geometry/BenchCPUSymmetry.cpp
#        src/cpu/geometry/fft/BenchCPURotate.cpp
#        src/cpu/geometry/fft/BenchCPUShift.cpp
        )

set(BENCHMARK_CUDA_SOURCES
#        src/gpu/cuda/fft/BenchCUDAFilters.cpp
        src/gpu/cuda/fft/BenchCUDATransforms.cpp

#        src/gpu/cuda/math/BenchCUDAArithmetics.cpp
        src/gpu/cuda/math/BenchCUDAReductions.cpp

        src/gpu/cuda/geometry/BenchCUDARotate.cpp
        src/gpu/cuda/geometry/BenchCUDASymmetry.cpp
        )

set(BENCHMARK_UNIFIED_SOURCES
        src/unified/fft/BenchUnifiedTransforms.cpp

        )


if (NOA_ENABLE_CUDA)
    set(BENCHMARK_SOURCES
            ${BENCHMARK_SOURCES}
            ${BENCHMARK_CPU_SOURCES}
            ${BENCHMARK_CUDA_SOURCES}
            ${BENCHMARK_UNIFIED_SOURCES}
            )
endif ()

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa_benchmarks.cmake)
