# ---------------------------------------------------------------------------------------
# Sets the benchmark files
# ---------------------------------------------------------------------------------------
set(BENCHMARK_SOURCES
    EntryPoint.cpp

    src/BenchCopy.cpp
    src/BenchTransformSpectrum.cpp
)


include(${PROJECT_SOURCE_DIR}/cmake/targets/noa_benchmarks.cmake)
