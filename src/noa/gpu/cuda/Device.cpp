#include "noa/gpu/cuda/Device.h"

#include "noa/gpu/cuda/fft/Plan.h"
#ifdef NOA_ENABLE_UNIFIED
#include "noa/unified/Stream.h"
#endif

namespace noa::cuda {
    // Before resetting the device, we must ensure that all global resource on that device is deleted.
    void Device::reset() const {
        DeviceGuard guard(*this);

        fft::PlanCache::cleanup();

        // Reset the default stream for that device to the NULL stream.
        #ifdef NOA_ENABLE_UNIFIED
        noa::Stream stream{noa::Device{noa::Device::GPU, this->id(), true}, noa::Stream::DEFAULT};
        noa::Stream::current(stream);
        #endif

        NOA_THROW_IF(cudaDeviceReset());
    }
}
