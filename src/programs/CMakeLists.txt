set(SANDBOX_SOURCES
        sandbox/Sandbox.cpp
        )

set(SANDBOX_HEADERS
        )

add_executable(sandbox ${SANDBOX_SOURCES} ${SANDBOX_HEADERS})

target_precompile_headers(sandbox
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src/noa/Base.h)

target_compile_definitions(sandbox PUBLIC "SANDBOX_DEBUG=$<CONFIG:Debug>")
target_link_libraries(sandbox
        PRIVATE
        project_options
        Noa::noa
        spdlog::spdlog)

#install(TARGETS programs
#        RUNTIME DESTINATION bin)
