#pragma once
#include <cuda_runtime_api.h>

#include "noa/Types.h"

namespace Test::CUDA {

    inline cudaError_t zero(void* d_ptr, size_t bytes) {
        cudaError_t err = cudaMemset(d_ptr, 0, bytes);
        cudaDeviceSynchronize(); // cudaMemset is async if device memory.
        return err;
    }


}
