# Sets the project environment.
# Using:
#   - CMAKE_MODULE_PATH
#   - CMAKE_CONFIGURATION_TYPES or CMAKE_BUILD_TYPE
#   - PROJECT_VERSION
#
# Introduces:
#   - CMAKE_INSTALL_LIBDIR      : The standard install directories. See GNUInstallDirs.
#   - CMAKE_INSTALL_BINDIR      : The standard install directories. See GNUInstallDirs.
#   - CMAKE_INSTALL_INCLUDEDIR  : The standard install directories. See GNUInstallDirs.
#   - NOA_GENERATED_DIR         : Where to store generated project CMake files.
#   - NOA_GENERATED_HEADERS_DIR : Where to store generated noa C++ headers.
#   - NOA_INSTALL_LIBDIR        : Where to store installed project CMake files.
#   - NOA_TARGETS_EXPORT_NAME
#   - NOA_CONFIG_FILE
#   - NOA_CONFIG_VERSION_FILE
#
# Created targets:
#   - noa_uninstall             : Uninstall everything in the install_manifest.txt.
message(STATUS "--------------------------------------")

# Add the local modules.
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/find)

# Make sure different configurations don't collide.
set(CMAKE_DEBUG_POSTFIX "d")

# Generate compile_commands.json to make it easier to work with clang based tools.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Get the standard install directories.
include(GNUInstallDirs)

# Generated directories:
set(NOA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/noa_generated")
set(NOA_GENERATED_HEADERS_DIR "${CMAKE_CURRENT_BINARY_DIR}/noa_generated_headers")

# ---------------------------------------------------------------------------------------
# Build type (default: Release)
# ---------------------------------------------------------------------------------------
# If multi-config, uses CMAKE_CONFIGURATION_TYPES.
# If single-config, uses CMAKE_BUILD_TYPE.
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
message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if("CUDA" IN_LIST languages)
    message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
    if (CMAKE_CUDA_HOST_COMPILER)
        message(STATUS "CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")
    endif ()
    message(STATUS "CUDA architectures: ${NOA_CUDA_ARCH}")
endif()

# ---------------------------------------------------------------------------------------
# CMake project packaging files
# ---------------------------------------------------------------------------------------
set(NOA_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/cmake/noa")
set(NOA_TARGETS_EXPORT_NAME "NoaTargets")
set(NOA_CONFIG_VERSION_FILE "${NOA_GENERATED_DIR}/NoaConfigVersion.cmake")
set(NOA_CONFIG_FILE "${NOA_GENERATED_DIR}/NoaConfig.cmake")

include(CMakePackageConfigHelpers)
configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/settings/Config.cmake.in"
        "${NOA_CONFIG_FILE}"
        INSTALL_DESTINATION "${NOA_INSTALL_LIBDIR}")
write_basic_package_version_file(
        "${NOA_CONFIG_VERSION_FILE}"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion)
# Since we currently only support STATIC builds, the ConfigVersion is not really useful, is it?

# ---------------------------------------------------------------------------------------
# Uninstall target
# ---------------------------------------------------------------------------------------
configure_file("${PROJECT_SOURCE_DIR}/cmake/settings/Uninstall.cmake.in"
        "${NOA_GENERATED_DIR}/Uninstall.cmake"
        IMMEDIATE @ONLY)
add_custom_target(noa_uninstall
        COMMAND ${CMAKE_COMMAND} -P ${NOA_GENERATED_DIR}/Uninstall.cmake)

message(STATUS "--------------------------------------\n")
