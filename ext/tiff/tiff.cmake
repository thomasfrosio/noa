if (NOA_TIFF_USE_EXISTING)
    message(STATUS "tiff: searching for existing libraries...")
    # https://cmake.org/cmake/help/v3.15/module/FindTIFF.html
    # There are two cached variables in v3.15:
    #   TIFF_INCLUDE_DIR:   the directory containing the TIFF headers.
    #   TIFF_LIBRARY:       the path to the TIFF library
    find_package(TIFF)
    if (NOT TIFF_FOUND)
        message(FATAL_ERROR "The TIFF libraries were not found...")
    else ()
        message(STATUS "tiff: searching for existing libraries... found")
    endif ()
else ()
    message(FATAL_ERROR "NOA_TIFF_USE_EXISTING=OFF. This option is currently not supported.")
endif ()
message("")
