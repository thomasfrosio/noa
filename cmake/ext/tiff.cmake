message(STATUS "TIFF: searching for existing libraries...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET TIFF::TIFF)
    message(STATUS "Target already exists: TIFF::TIFF")
else ()
    message(STATUS "[in] TIFF_STATIC: ${TIFF_STATIC}")
    find_package(TIFF REQUIRED)
    message(STATUS "New imported target available: TIFF::TIFF")
endif ()

message(STATUS "[out] TIFF_INCLUDE_DIR: ${TIFF_INCLUDE_DIR}")
message(STATUS "[out] TIFF_LIBRARIES: ${TIFF_LIBRARIES}")

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "TIFF: searching for existing libraries... done")
