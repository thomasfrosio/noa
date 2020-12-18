# =======================================================================================
# Build and install the noa library
# =======================================================================================
set(NOA_SOURCES
        # Utilities
        util/Log.cpp

        # Files
        files/TextFile.cpp
        files/ImageFile.cpp

        # File headers
        files/headers/MRCHeader.cpp

        # Managers
        managers/InputManager.cpp

        )


set(NOA_HEADERS
        Base.h

        # Utilities
        util/Vectors.h
        util/VectorInt.h
        util/VectorFloat.h
        util/Math.h
        util/Exception.h
        util/IO.h
        util/Log.h
        util/OS.h
        util/String.h
        util/Traits.h
        util/Errno.h

        # Files
        files/File.h
        files/TextFile.h
        files/ImageFile.h


        # File headers
        files/headers/Header.h
        files/headers/AbstractHeader.h
        files/headers/MRCHeader.h

        # Managers
        managers/InputManager.h

        # GPU
        gpu/Base.h

        # CryoEM
        cryoem/Image.h
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
#   - core/*.h -> <prefix>/include/noa/*.h
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
