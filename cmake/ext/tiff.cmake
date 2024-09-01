message(STATUS "TIFF: searching for existing libraries...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET TIFF::TIFF)
    message(STATUS "Target already exists: TIFF::TIFF")
else ()
    # TODO libtiff.a from apt is broken apparently, similar to this https://bugs.archlinux.org/task/77224.
    # TODO Use shared library for now, but maybe look for a better TIFF option?
    set(_tiff_old_lib_suffix ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})

    find_package(TIFF ${ARGN} REQUIRED)

    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_tiff_old_lib_suffix})
    unset(_tiff_old_lib_suffix)

    message(STATUS "New imported target available: TIFF::TIFF")
endif ()

message(STATUS "[out] TIFF_INCLUDE_DIR: ${TIFF_INCLUDE_DIR}")
message(STATUS "[out] TIFF_LIBRARIES: ${TIFF_LIBRARIES}")

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "TIFF: searching for existing libraries... done")
