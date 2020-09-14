set(AKIRA_SOURCES
        Akira.cpp
        )

set(AKIRA_HEADERS
        )

add_executable(akira ${AKIRA_SOURCES} ${AKIRA_HEADERS})

target_compile_definitions(akira PUBLIC "AKIRA_DEBUG=$<CONFIG:Debug>")
target_link_libraries(akira
        PRIVATE
        project_options
        Noa::noa
        spdlog::spdlog)

#install(TARGETS akira
#        RUNTIME DESTINATION bin)
