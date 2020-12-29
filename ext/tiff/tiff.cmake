message(STATUS "Finding shared dependency: libtiff")

# libtiff repository: https://gitlab.com/libtiff/libtiff

# https://cmake.org/cmake/help/v3.15/module/FindTIFF.html
# There are two cached variables in v3.15:
#   TIFF_INCLUDE_DIR:   the directory containing the TIFF headers.
#   TIFF_LIBRARY:       the path to the TIFF library
find_package(TIFF)
