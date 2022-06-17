message(STATUS "OpenBLAS: searching for existing libraries...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")
find_package(OpenBLAS)
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "OpenBLAS: searching for existing libraries... done")

# Note: FetchContent doesn't work properly with OpenBLAS.
#       OpenBLAS added CMake support, but it is experimental.
# TODO: Add support for FetchContent or ExternalProject_Add.
