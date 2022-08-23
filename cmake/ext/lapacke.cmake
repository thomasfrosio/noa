message(STATUS "LAPACKE: searching for existing libraries...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET LAPACKE::LAPACKE)
    message(STATUS "Target already exists: LAPACKE::LAPACKE")
else ()
    message(STATUS "[in] LAPACKE_STATIC: ${LAPACKE_STATIC}")
    message(STATUS "[in] BLA_STATIC: ${BLA_STATIC}")
    message(STATUS "[in] BLA_VENDOR: ${BLA_VENDOR}")

    find_package(LAPACKE REQUIRED)
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "LAPACKE: searching for existing libraries... done")
