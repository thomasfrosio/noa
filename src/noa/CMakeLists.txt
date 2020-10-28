# =======================================================================================
# Build and install the noa library
# =======================================================================================
set(NOA_SOURCES
        # util
        util/Log.cpp

        # files
        files/File.cpp
        files/TextFile.cpp
        files/ProjectFile.cpp
        files/ImageFile.cpp
        files/MRCFile.cpp

        # managers
        managers/InputManager.cpp

        )


set(NOA_HEADERS
        # util
        util/Assert.h
        util/Exception.h
        util/Log.h
        util/String.h
        util/Traits.h
        util/OS.h

        # files
        files/File.h
        files/TextFile.h
        files/ProjectFile.h
        files/ImageFile.h
        files/MRCFile.h

        # managers
        managers/InputManager.h

        # gpu
        gpu/Base.h

        # base
        Base.h

        util/Arrays.h

        )

# ---------------------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------------------
add_library(noa ${NOA_SOURCES} ${NOA_HEADERS})
add_library(Noa::noa ALIAS noa)

target_link_libraries(noa
        PRIVATE
        project_options
        PUBLIC
        spdlog::spdlog
        )

# Hide everything by default - export manually
set_target_properties(noa PROPERTIES
        CMAKE_CXX_VISIBILITY_PRESET hidden
        CMAKE_VISIBILITY_INLINES_HIDDEN YES)

if (NOA_ENABLE_PCH)
    target_precompile_headers(noa
            PRIVATE
            ${PROJECT_SOURCE_DIR}/src/noa/Base.h)
endif ()

# NOA_DEBUG definition
target_compile_definitions(noa
        PUBLIC
        "NOA_DEBUG=$<CONFIG:Debug>")

# Global includes. Used by all targets linking to the library (PUBLIC)
# Headers in source tree can be included as well as the noa_generated_headers
target_include_directories(noa
        PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
        "$<BUILD_INTERFACE:${NOA_GENERATED_HEADERS_DIR}>"
        "$<INSTALL_INTERFACE:.>")

# ---------------------------------------------------------------------------------------
# Symbol visibility
# ---------------------------------------------------------------------------------------
# Generate the export header for noa.
#   - generates API.h
#   - ensure NOA_API is defined
include(GenerateExportHeader)
generate_export_header(noa
        EXPORT_MACRO_NAME NOA_API
        EXPORT_FILE_NAME ${NOA_GENERATED_HEADERS_DIR}/noa/API.h)

# ---------------------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------------------
# Targets:
#   - <prefix>/lib/libnoa.a
#   - header location after install: <prefix>/noa/*.h
#   - headers can be included by C++ code `#include <noa/*.h>`
install(TARGETS noa project_options spdlog
        EXPORT "${NOA_TARGETS_EXPORT_NAME}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Headers:
#   - noa/*.h -> <prefix>/include/noa/*.h
install(FILES ${NOA_HEADERS}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/noa")

# Headers:
#   - generated_headers/noa/version.h -> <prefix>/include/noa/version.h
install(FILES
        "${NOA_GENERATED_HEADERS_DIR}/noa/API.h"
        "${NOA_GENERATED_HEADERS_DIR}/noa/Version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/noa")

# Config
#   - <prefix>/lib/cmake/Foo/FooConfig.cmake
#   - <prefix>/lib/cmake/Foo/FooConfigVersion.cmake
install(FILES
        "${NOA_CONFIG_FILE}"
        "${NOA_CONFIG_VERSION_FILE}"
        DESTINATION "${NOA_INSTALL_LIBDIR}")

# Config
#   - <prefix>/lib/cmake/Foo/FooTargets.cmake
install(EXPORT "${NOA_TARGETS_EXPORT_NAME}"
        FILE "NoaTargets.cmake"
        DESTINATION "${NOA_INSTALL_LIBDIR}"
        NAMESPACE "Noa::")
