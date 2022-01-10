message(STATUS "Configuring target: noa::noa_tests")

include(${PROJECT_SOURCE_DIR}/ext/catch2/catch2.cmake)
include(${PROJECT_SOURCE_DIR}/ext/yaml-cpp/yaml-cpp.cmake)

add_executable(noa_tests ${TEST_SOURCES})
add_executable(noa::noa_tests ALIAS noa_tests)

target_link_libraries(noa_tests
        PRIVATE
        prj_common_option
        prj_compiler_warnings
        noa::noa_static
        Catch2::Catch2
        yaml-cpp::yaml-cpp
        )

target_precompile_headers(noa_tests
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src/noa/common/Definitions.h
        ${PROJECT_SOURCE_DIR}/src/noa/common/Exception.h
        ${PROJECT_SOURCE_DIR}/src/noa/common/Logger.h
        ${PROJECT_SOURCE_DIR}/src/noa/common/Types.h
        )

target_include_directories(noa_tests
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests)

install(TARGETS noa_tests
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        )

message("")
