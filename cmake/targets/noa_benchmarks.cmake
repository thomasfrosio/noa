message(STATUS "--------------------------------------")
message(STATUS "-> noa::noa_benchmarks: configuring public target...")

include(${PROJECT_SOURCE_DIR}/cmake/ext/google-benchmark.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/yaml-cpp.cmake)

add_executable(noa_benchmarks ${BENCHMARK_SOURCES})
add_executable(noa::noa_benchmarks ALIAS noa_benchmarks)

target_link_libraries(noa_benchmarks
    PRIVATE
    prj_common_option
    prj_compiler_warnings
    noa::noa
    benchmark::benchmark
    yaml-cpp::yaml-cpp
    )

target_precompile_headers(noa_benchmarks
    PRIVATE
    ${PROJECT_SOURCE_DIR}/src/noa/core/Types.hpp
    )

target_include_directories(noa_benchmarks
    PRIVATE
    ${PROJECT_SOURCE_DIR}/benchmarks)

install(TARGETS noa_benchmarks
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )

message(STATUS "-> noa::noa_benchmarks: configuring public target... done")
message(STATUS "--------------------------------------\n")
