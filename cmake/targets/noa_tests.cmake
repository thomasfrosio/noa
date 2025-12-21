message(STATUS "--------------------------------------")
message(STATUS "-> noa::noa_tests: configuring public target...")

if (NOT NOA_ERROR_POLICY EQUAL 2)
    message(FATAL_ERROR "In order to built tests, the library should built with NOA_ERROR_POLICY=2, but got NOA_ERROR_POLICY=${NOA_ERROR_POLICY}")
endif ()

include(${PROJECT_SOURCE_DIR}/cmake/ext/catch2.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/yaml-cpp.cmake)

# Treat the sources as CUDA sources if CUDA is enabled
if (NOA_ENABLE_CUDA)
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
endif ()

add_executable(noa_tests ${TEST_SOURCES})
add_executable(noa::noa_tests ALIAS noa_tests)

target_link_libraries(noa_tests
    PRIVATE
    prj_compiler_public_options
    prj_compiler_private_options
    prj_compiler_warnings
    noa::noa
    Catch2::Catch2
    yaml-cpp::yaml-cpp
)

target_include_directories(noa_tests
    PRIVATE
    ${PROJECT_SOURCE_DIR}/tests
)

if (NOA_ENABLE_CUDA)
    set_target_properties(noa_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # The preprocessing in CUDA is super annoying because it generates a few useless casts.
    # So turn this specific warning off. Note that for CPU only, we still want all warnings.
    target_compile_options(noa_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: "-Wno-useless-cast" >)
endif()

install(
    TARGETS noa_tests
    EXPORT noa
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

message(STATUS "-> noa::noa_tests: configuring public target... done")
message(STATUS "--------------------------------------\n")
