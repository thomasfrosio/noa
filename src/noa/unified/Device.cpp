#include "noa/unified/Device.h"
#include "noa/unified/Stream.h"
#include "noa/cpu/Device.hpp"

namespace noa {
    void Device::synchronize() const {
        Stream::current(*this).synchronize();
        #ifdef NOA_ENABLE_CUDA
        if (is_gpu())
            cuda::Device(this->id(), cuda::Device::DeviceUnchecked{}).synchronize();
        #endif
    }

    void Device::reset() const {
        synchronize();
        Stream stream(*this, StreamMode::DEFAULT);
        Stream::set_current(stream);

        if (is_cpu()) {
            cpu::Device::reset();
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::Device(id(), cuda::Device::DeviceUnchecked{}).reset();
            #endif
        }
    }
}
