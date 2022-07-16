message(STATUS "OpenBLAS: searching for existing libraries...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

find_package(OpenBLAS REQUIRED)
message(STATUS "[output] OpenBLAS_VERSION: ${OpenBLAS_VERSION}")
message(STATUS "[output] OpenBLAS_INCLUDE_DIRS: ${OpenBLAS_INCLUDE_DIRS}")
message(STATUS "[output] OpenBLAS_LIBRARY: ${OpenBLAS_LIBRARY}")
message(STATUS "New imported target available: OpenBLAS::OpenBLAS")

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "OpenBLAS: searching for existing libraries... done")

# TODO: Add support for FetchContent or ExternalProject_Add.
