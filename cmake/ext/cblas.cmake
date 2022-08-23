message(STATUS "CBLAS: searching for existing libraries...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET CBLAS::CBLAS)
    message(STATUS "Target already exists: CBLAS::CBLAS")
else ()
    message(STATUS "[in] BLA_STATIC: ${BLA_STATIC}")
    message(STATUS "[in] BLA_VENDOR: ${BLA_VENDOR}")
    find_package(CBLAS REQUIRED)
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "CBLAS: searching for existing libraries... done")
