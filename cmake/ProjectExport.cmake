# Exports the project.

# ---------------------------------------------------------------------------------------
# Configuration files.
# ---------------------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

# Directory to store the generated config files.
set(NOA_GENERATED_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/noa_generated_config")

# When a project does find_package(noa), this config file is going to be included into
# the project. The file should 1) include all of our targets, this is done by including
# the noaTargets.cmake file generated by CMake, and 2) import our target dependencies
# or at least make sure they are available to the project.
# When building and importing on the same system, we basically want to export our targets
# and their dependencies so that the project can use exactly the same binaries. CMake doesn't
# seem to allow us to export imported targets which were not built by this project. As such,
# we have to save as much information as possible about the build in the configuration file.
# When building and importing on different systems, we want the package to be relocatable.
# As such, the configuration file should find the packages that closest match the dependencies
# we used during building. This is where package manager shines...
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/utils/noaConfig.cmake.in
    ${NOA_GENERATED_CONFIG_DIR}/noaConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/noa)

# The library uses the semantic versioning: MAJOR.MINOR.PATCH
# MAJOR: Breaking change. Reset MINOR and PATCH to 0.
# MINOR: Non-breaking additions to the API (i.e. something was added).
#        W ensure that if someone had built against the previous version, they could simply
#        rebuild their application with the new version without having to change the way they
#        interact with the library. When updated, PATCH should be reset to 0.
# PATCH: Bug fix only. No API changes.
write_basic_package_version_file(
    ${NOA_GENERATED_CONFIG_DIR}/noaConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

# Install the config files.
# Our modules are used by noaConfig.cmake, so install them as well.
install(FILES
    ${NOA_GENERATED_CONFIG_DIR}/noaConfig.cmake
    ${NOA_GENERATED_CONFIG_DIR}/noaConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/noa)
install(DIRECTORY
    ${PROJECT_SOURCE_DIR}/cmake/modules
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/)

# ---------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------
# All of our targets were installed under the same export name,
# which should now be installed as well. When a project imports our config,
# our targets (but not our dependencies) will be enclosed in this noa:: namespace.
install(EXPORT noa
    FILE noaTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/noa
    NAMESPACE noa::
    )
