# CMake for the noa static library (include/) and executable (src/).

cmake_minimum_required(VERSION 3.16)
project(root CXX)

set(CMAKE_CXX_STANDARD 17)

# Options
include(cmake/Options.cmake)
setOption()

include(cmake/ProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)
include(cmake/Dependencies.cmake)
include(cmake/Cache.cmake)

# Use the compiler warnings
add_library(root_warnings INTERFACE)
include(cmake/CompilerWarnings.cmake)
set_project_warnings(root_warnings)

# Enable doxygen
include(cmake/Doxygen.cmake)
enable_doxygen()

# Google Test
include(CTest)
enable_testing()
add_subdirectory(tests)

# Noa programs
message("######## Load noa ########")
add_executable(noa src/noa/EntryPoint.cpp)
target_include_directories(noa PRIVATE include)
target_link_libraries(noa
        PRIVATE
        root_warnings
        spdlog::spdlog) # Access {fmt} via spdlog

# Unset the options from the cache
unsetOption()
