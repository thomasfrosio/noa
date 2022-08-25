#include "noa/Device.h"
#include "noa/Stream.h"
#include "noa/cpu/Device.h"

namespace noa {
    void Device::synchronize() const {
        Stream::current(*this).synchronize();
        #ifdef NOA_ENABLE_CUDA
        if (gpu())
            cuda::Device(this->id(), true).synchronize();
        #endif
    }

    void Device::reset() const {
        synchronize();
        Stream stream(*this, Stream::DEFAULT);
        Stream::current(stream);

        if (cpu()) {
            cpu::Device::reset();
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::Device(id(), true).reset();
            #endif
        }
    }
}
