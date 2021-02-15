#include "KernelPtrDevice.h"

namespace Test::CUDA {
    __global__ void kernel_zero(int* d_ptr, size_t elements) {
        uint lane = blockIdx.x * blockIdx.x + threadIdx.x;
        if (lane < elements) {
            d_ptr[lane] = 0; // cudaMemset...
        }
    }

    void zero(char* d_ptr, size_t bytes) {
        uint thread_per_block = 128;
        uint block_per_grid = (bytes + thread_per_block - 1) / thread_per_block;
        cudaMemset(d_ptr, 0, bytes);

    }
    void zero(uint* d_ptr, size_t elements);
    void zero(float* d_ptr, size_t elements);
    void zero(Noa::cfloat_t* d_ptr, size_t elements);
}
