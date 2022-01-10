message(STATUS "Configuring target: noa::noa_benchmarks")

include(${PROJECT_SOURCE_DIR}/ext/google-benchmark/google-benchmark.cmake)
include(${PROJECT_SOURCE_DIR}/ext/yaml-cpp/yaml-cpp.cmake)

add_executable(noa_benchmarks ${BENCHMARK_SOURCES})
add_executable(noa::noa_benchmarks ALIAS noa_benchmarks)

target_link_libraries(noa_benchmarks
        PRIVATE
        prj_common_option
        prj_compiler_warnings
        noa::noa_static
        benchmark::benchmark
        yaml-cpp::yaml-cpp
        )

target_precompile_headers(noa_benchmarks
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src/noa/common/Definitions.h
        ${PROJECT_SOURCE_DIR}/src/noa/common/Exception.h
        ${PROJECT_SOURCE_DIR}/src/noa/common/Logger.h
        ${PROJECT_SOURCE_DIR}/src/noa/common/Types.h
        )

target_include_directories(noa_benchmarks
        PRIVATE
        ${PROJECT_SOURCE_DIR}/benchmarks)

install(TARGETS noa_benchmarks
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

message("")
