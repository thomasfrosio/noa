#include "noa/unified/Device.hpp"
#include "noa/unified/Stream.hpp"

namespace noa {
    void Device::synchronize() const {
        Stream::current(*this).synchronize();
        #ifdef NOA_ENABLE_CUDA
        if (is_gpu())
            noa::cuda::Device(this->id(), Unchecked{}).synchronize();
        #endif
    }

    void Device::reset() const {
        synchronize();

        // Set the current stream to the default/null stream, potentially freeing user-created streams.
        const Stream stream(*this, Stream::DEFAULT);
        Stream::set_current(stream);

        if (is_cpu()) {
            noa::cpu::Device::reset();
        } else {
            #ifdef NOA_ENABLE_CUDA
            noa::cuda::Device(id(), Unchecked{}).reset();
            #endif
        }
    }
}
