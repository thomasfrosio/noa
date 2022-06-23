#include "noa/gpu/cuda/Device.h"

// The library (i.e. the CUDA backend) keeps track of some global resources, e.g. FFT plans or cuBLAS handles.
// While we expect users to destroy the resources they have created for the device they want to reset (this includes
// streams, arrays, textures, CUDA arrays), we have to take care of our own internal resources, namely:
//  - FFT plans
//  - cublas handles/workspaces
#include "noa/gpu/cuda/fft/Plan.h"
#include "noa/gpu/cuda/math/Blas.h"

namespace noa::cuda {
    void Device::reset() const {
        DeviceGuard guard(*this);
        guard.synchronize(); // if called noa::Device::reset(), the device is already synchronized

        fft::PlanCache::cleanup();
        math::details::cublasClearCache(id());

        NOA_THROW_IF(cudaDeviceReset());
    }
}
