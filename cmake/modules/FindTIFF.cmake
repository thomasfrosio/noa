# Simple wrapper around CMake's FindTIFF.cmake to add support for TIFF_STATIC.

set(_tiff_old_lib_suffix ${CMAKE_FIND_LIBRARY_SUFFIXES})
if (TIFF_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
else ()
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
endif ()

list(REMOVE_ITEM CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

find_package(TIFF ${ARGN})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_tiff_old_lib_suffix})
unset(_tiff_old_lib_suffix)
