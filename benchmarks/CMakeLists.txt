# ---------------------------------------------------------------------------------------
# Sets the benchmark files
# ---------------------------------------------------------------------------------------
set(BENCHMARK_SOURCES
    EntryPoint.cpp

#    src/BenchCopy.cpp
#    src/BenchTransform.cpp
    src/BenchTransformSpectrum.cpp
#    src/BenchProject.cpp
)

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa_benchmarks.cmake)
