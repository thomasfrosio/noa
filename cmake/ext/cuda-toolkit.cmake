message(STATUS "CUDAToolkit: searching for existing libraries...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

message(STATUS "[in] NOA_CUDA_STATIC: ${NOA_CUDA_STATIC}")

find_package(CUDAToolkit 12.6 REQUIRED)

message(STATUS "[out] CUDA Toolkit library path: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "[out] CUDA Toolkit version: ${CUDAToolkit_VERSION}")

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "CUDAToolkit: searching for existing libraries... done")
