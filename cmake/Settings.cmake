# ---------------------------------------------------------------------------------------
# What should be build?
# ---------------------------------------------------------------------------------------
message(STATUS "Build executable: ${NOA_BUILD_APP}")
if (BUILD_SHARED_LIBS)
    message(STATUS "BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif ()

# ---------------------------------------------------------------------------------------
# Build type (default: Release)
# ---------------------------------------------------------------------------------------
# If multi-config, set CMAKE_CONFIGURATION_TYPES.
# If single-config, set CMAKE_BUILD_TYPE.
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (isMultiConfig)
    set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo" CACHE STRING "")
    message(STATUS "CMAKE_CONFIGURATION_TYPES (multi-config): ${CMAKE_CONFIGURATION_TYPES}")
else ()
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build.")
    message(STATUS "CMAKE_BUILD_TYPE (single-config): ${CMAKE_BUILD_TYPE}")

    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
endif ()
message(STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}\n")

# Make sure different configurations don't collide
set(CMAKE_DEBUG_POSTFIX "d")

# ---------------------------------------------------------------------------------------
# Build tree
# ---------------------------------------------------------------------------------------
# ├── bin                                       (optional)
# │   └── akira
# ├── doc                                       (optional)
# │   └── /* doxygen/sphinx documentation */
# ├── tests                                     (optional)
# │   └── /* various ctest tests */
# ├── include
# │   └── noa
# │       ├── */*.h
# │       └── version.h
# └── lib
#     ├── cmake
#     │   └── Noa
#     │       ├── NoaConfig.cmake
#     │       ├── NoaConfigVersion.cmake
#     │       ├── NoaTargets.cmake
#     │       ├── NoaTargets-debug.cmake
#     │       └── NoaTargets-release.cmake
#     ├── libnoa.a                              (Release)
#     └── libnoad.a                             (Debug)

include(GNUInstallDirs)

# Install directory and exportable targets
set(NOA_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/cmake/Noa")
set(NOA_TARGETS_EXPORT_NAME "NoaTargets")

# Generated folders -> available at generator time
set(NOA_GENERATED_HEADERS_DIR "${CMAKE_CURRENT_BINARY_DIR}/noa_generated_headers")
set(NOA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/noa_generated")
set(NOA_CONFIG_VERSION_FILE "${NOA_GENERATED_DIR}/NoaConfigVersion.cmake")
set(NOA_CONFIG_FILE "${NOA_GENERATED_DIR}/NoaConfig.cmake")

# The config|config-version files
include(CMakePackageConfigHelpers)
configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in"
        "${NOA_CONFIG_FILE}"
        INSTALL_DESTINATION "${NOA_INSTALL_LIBDIR}")
write_basic_package_version_file(
        "${NOA_CONFIG_VERSION_FILE}"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion)

# Always full RPATH (for shared libraries)
#  https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
if (BUILD_SHARED_LIBS)
    # use, i.e. don't skip the full RPATH for the build tree
    set(CMAKE_SKIP_BUILD_RPATH FALSE)

    # when building, don't use the install RPATH already
    # (but later on when installing)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

    # add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    # the RPATH to be used when installing, but only if it's not a system directory
    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    if ("${isSystemDir}" STREQUAL "-1")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    endif ()
endif ()

# ---------------------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------------------
configure_file(
        "${PROJECT_SOURCE_DIR}/src/noa/version.h.in"
        "${NOA_GENERATED_HEADERS_DIR}/noa/version.h"
        @ONLY)

# ---------------------------------------------------------------------------------------
# Uninstall target
# ---------------------------------------------------------------------------------------
configure_file("${PROJECT_SOURCE_DIR}/cmake/Uninstall.cmake.in"
        "${NOA_GENERATED_DIR}/Uninstall.cmake"
        IMMEDIATE @ONLY)
add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${NOA_GENERATED_DIR}/Uninstall.cmake)
