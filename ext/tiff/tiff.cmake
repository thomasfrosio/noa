message(STATUS "libtiff: searching for existing libraries...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

# https://cmake.org/cmake/help/v3.15/module/FindTIFF.html

message(STATUS "[input] NOA_TIFF_STATIC: ${NOA_TIFF_STATIC}")
message(STATUS "[input (env)] NOA_ENV_TIFF_LIBRARIES: $ENV{NOA_ENV_TIFF_LIBRARIES}")
message(STATUS "[input (env)] NOA_ENV_TIFF_INCLUDE: $ENV{NOA_ENV_TIFF_INCLUDE}")

# Whether to search for static or dynamic libraries.
set(NOA_CMAKE_FIND_LIBRARY_SUFFIXES_OLD ${CMAKE_FIND_LIBRARY_SUFFIXES})
if (NOA_TIFF_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif ()

set(NOA_TIFF_ENV $ENV{NOA_ENV_TIFF_LIBRARIES} $ENV{NOA_ENV_TIFF_INCLUDE})
if (NOA_TIFF_ENV)
    find_package(TIFF REQUIRED PATH ${NOA_TIFF_ENV} NO_DEFAULT_PATH)
else ()
    find_package(TIFF REQUIRED)
endif ()

# Reset:
set(CMAKE_FIND_LIBRARY_SUFFIXES ${NOA_CMAKE_FIND_LIBRARY_SUFFIXES_OLD})

message(STATUS "[output] TIFF_INCLUDE_DIR: ${TIFF_INCLUDE_DIR}")
message(STATUS "[output] TIFF_LIBRARY: ${TIFF_LIBRARY}")
message(STATUS "New imported target available: TIFF::TIFF")

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "libtiff: searching for existing libraries... done")
