#include "noa/gpu/cuda/Device.hpp"

// The library (i.e. the CUDA backend) keeps track of some global resources, e.g. FFT plans or cuBLAS handles.
// While we expect users to destroy the resources they have created for the device they want to reset (this includes
// streams, arrays, textures, CUDA arrays), we have to take care of our own internal resources, namely:
//  - FFT plans
//  - cublas handles/workspaces
#include "noa/gpu/cuda/fft/Plan.hpp"
#include "noa/gpu/cuda/Blas.hpp"

namespace noa::cuda {
    void Device::reset() const {
        const DeviceGuard guard(*this);
        guard.synchronize(); // if called from noa::Device::reset(), the device is already synchronized

        noa::cuda::fft::clear_caches(*this);
        noa::cuda::cublas_clear_cache(*this);

        check(cudaDeviceReset());
    }
}
