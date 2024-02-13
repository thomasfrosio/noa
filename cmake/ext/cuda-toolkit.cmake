message(STATUS "CUDAToolkit: searching for existing libraries...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

message(STATUS "[in] NOA_CUDA_CUDART_STATIC: ${NOA_CUDA_CUDART_STATIC}")
message(STATUS "[in] NOA_CUDA_CUFFT_STATIC: ${NOA_CUDA_CUFFT_STATIC}")
message(STATUS "[in] NOA_CUDA_CURAND_STATIC: ${NOA_CUDA_CURAND_STATIC}")
message(STATUS "[in] NOA_CUDA_CUBLAS_STATIC: ${NOA_CUDA_CUBLAS_STATIC}")

find_package(CUDAToolkit 12 REQUIRED)

message(STATUS "[out] CUDA Toolkit library path: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "[out] CUDA Toolkit version: ${CUDAToolkit_VERSION}")

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "CUDAToolkit: searching for existing libraries... done")
