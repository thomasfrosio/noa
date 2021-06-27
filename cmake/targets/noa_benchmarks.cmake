message(STATUS "Configuring target: noa::noa_benchmarks")

include(${PROJECT_SOURCE_DIR}/ext/catch2/catch2.cmake)

add_executable(noa_benchmarks ${BENCHMARK_SOURCES})
add_executable(noa::noa_benchmarks ALIAS noa_benchmarks)

target_link_libraries(noa_benchmarks
        PRIVATE
        prj_common_option
        prj_cxx_warnings
        noa::noa_static
        Catch2::Catch2
        )

target_precompile_headers(noa_benchmarks
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src/noa/Definitions.h
        ${PROJECT_SOURCE_DIR}/src/noa/Exception.h
        ${PROJECT_SOURCE_DIR}/src/noa/Logger.h
        ${PROJECT_SOURCE_DIR}/src/noa/Types.h
        )

target_include_directories(noa_benchmarks
        PRIVATE
        ${PROJECT_SOURCE_DIR}/benchmarks)

install(
        TARGETS noa_benchmarks
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

message("")
