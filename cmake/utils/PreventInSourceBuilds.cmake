macro(noa_prevent_in_source_builds)
    # This aims to prevent in-source builds.
    # REALPATH: make sure the user doesn't play dirty with symlinks
    get_filename_component(_srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
    get_filename_component(_bindir "${CMAKE_BINARY_DIR}" REALPATH)

    if ("${_srcdir}" STREQUAL "${_bindir}")
        message("In-source builds are disabled")
        message("Please create a separate build directory (which can be inside the source directory) and run cmake from there")
        message(FATAL_ERROR "Quitting configuration")
    endif ()
endmacro()
