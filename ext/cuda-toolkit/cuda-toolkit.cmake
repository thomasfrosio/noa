message(STATUS "CUDAToolkit: searching for existing libraries...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

find_package(CUDAToolkit 11 QUIET REQUIRED)

message(STATUS "CUDA Toolkit library path: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "CUDAToolkit: searching for existing libraries... done")
