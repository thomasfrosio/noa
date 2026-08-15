message(STATUS "--------------------------------------")
message(STATUS "-> noa::noa_benchs: configuring public target...")

include(${PROJECT_SOURCE_DIR}/cmake/ext/google-benchmark.cmake)

# Treat the sources as CUDA sources if CUDA is enabled
if (NOA_ENABLE_CUDA)
    set_source_files_properties(${BENCH_SOURCES} PROPERTIES LANGUAGE CUDA)

    # The preprocessing in CUDA is super annoying because it generates a few useless casts.
    # So turn this specific warning off. Note that for CPU only, we still want all warnings.
    set_source_files_properties(${BENCH_SOURCES} PROPERTIES
        COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-useless-cast>"
    )
endif ()

add_executable(noa_benchs ${BENCH_SOURCES})
add_executable(noa::noa_benchs ALIAS noa_benchs)

target_link_libraries(noa_benchs
    PRIVATE
    prj_compiler_public_options
    prj_compiler_private_options
    prj_compiler_warnings
    noa::noa
    benchmark::benchmark
)

target_include_directories(noa_benchs
    PRIVATE
    ${PROJECT_SOURCE_DIR}/tests
)

if (NOA_ENABLE_CUDA)
    set_target_properties(noa_benchs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

install(
    TARGETS noa_benchs
    EXPORT noa
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

message(STATUS "-> noa::noa_benchs: configuring public target... done")
message(STATUS "--------------------------------------\n")
