//
// Created by thomas on 11/01/2021.
//

#ifndef NOA_BASE_H
#define NOA_BASE_H

#endif //NOA_BASE_H

// #include <stdlib.h>
// #include <stdio.h>
//
//#define THREADS 128
//
//// Macro to catch CUDA errors in CUDA runtime calls
//#define CUDA_SAFE_CALL(call)                                          \
//do {                                                                  \
//    cudaError_t err = call;                                           \
//    if (cudaSuccess != err) {                                         \
//        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
//                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
//        exit(EXIT_FAILURE);                                           \
//    }                                                                 \
//} while (0)
//
//// Macro to catch CUDA errors in kernel launches
//#define CHECK_LAUNCH_ERROR()                                          \
//do {                                                                  \
//    /* Check synchronous errors, i.e. pre-launch */                   \
//    cudaError_t err = cudaGetLastError();                             \
//    if (cudaSuccess != err) {                                         \
//        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
//                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
//        exit(EXIT_FAILURE);                                           \
//    }                                                                 \
//    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
//    err = cudaDeviceSynchronize();                                    \
//    if (cudaSuccess != err) {                                         \
//        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
//                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
//        exit(EXIT_FAILURE);                                           \
//    }                                                                 \
//} while (0)
//
//__global__ void add5 (int *arr, int len)
//{
//    int stride = gridDim.x * blockDim.x;
//    int tid = blockDim.x * blockIdx.x + threadIdx.x;
//    for (int i = tid; i < len; i += stride) {
//        arr[i] += 5;
//    }
//}
//
//__global__ void mul5 (int *arr, int len)
//{
//    int stride = gridDim.x * blockDim.x;
//    int tid = blockDim.x * blockIdx.x + threadIdx.x;
//    for (int i = tid; i < len; i += stride) {
//        arr[i] *= 5;
//    }
//}
//
//void wrapper_1 (int len, int **d_arr, int **h_arr)
//{
//    *h_arr = (int *)malloc (sizeof(*h_arr[0]) * len);
//    if (!*h_arr) {
//        fprintf (stderr, "host alloc failed in file '%s' in line %\n",
//                 __FILE__, __LINE__);
//        exit(EXIT_FAILURE);
//    }
//    memset (*h_arr, 0x00, sizeof(*h_arr[0]) * len);
//    CUDA_SAFE_CALL (cudaMalloc((void**)d_arr, sizeof(*d_arr[0]) * len));
//    CUDA_SAFE_CALL (cudaMemset(*d_arr, 0x00, sizeof(*d_arr[0]) * len));
//    dim3 dimBlock (THREADS);
//    int threadBlocks = (len + (dimBlock.x - 1)) / dimBlock.x;
//    dim3 dimGrid(threadBlocks);
//    add5<<<dimGrid,dimBlock>>>(*d_arr, len);
//    CHECK_LAUNCH_ERROR();
//}
//
//void wrapper_2 (int len, int *d_arr, int *h_arr)
//{
//    dim3 dimBlock (THREADS);
//    int threadBlocks = (len + (dimBlock.x - 1)) / dimBlock.x;
//    dim3 dimGrid(threadBlocks);
//    mul5<<<dimGrid,dimBlock>>>(d_arr, len);
//    CHECK_LAUNCH_ERROR();
//    CUDA_SAFE_CALL (cudaMemcpy (h_arr, d_arr, sizeof (h_arr[0]) * len,
//                                cudaMemcpyDeviceToHost));
//    for (int i = 0; i < len; i++) {
//        printf ("%d: %d\n", i, h_arr[i]);
//    }
//}
//
//int main (void)
//{
//    int *d_arr, *h_arr;
//    int len = 10;
//    wrapper_1 (len, &d_arr, &h_arr);
//    wrapper_2 (len, d_arr, h_arr);
//    CUDA_SAFE_CALL (cudaFree (d_arr));
//    free (h_arr);
//    return EXIT_SUCCESS;
//}
