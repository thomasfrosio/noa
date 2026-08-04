#include "noa/runtime/Device.hpp"
#include "noa/runtime/Stream.hpp"

namespace noa {
    void Device::add_reset_callback(reset_callback_type callback) {
        noa::cpu::Device::add_reset_callback(callback);
        #ifdef NOA_ENABLE_CUDA
        noa::cuda::Device::add_reset_callback(callback);
        #endif
    }

    void Device::remove_reset_callback(reset_callback_type callback) {
        noa::cpu::Device::remove_reset_callback(callback);
        #ifdef NOA_ENABLE_CUDA
        noa::cuda::Device::remove_reset_callback(callback);
        #endif
    }

    void Device::synchronize() const {
        Stream::current(*this).synchronize();
        #ifdef NOA_ENABLE_CUDA
        if (is_gpu())
            noa::cuda::Device(this->id(), Unchecked{}).synchronize();
        #endif
    }

    void Device::reset() const {
        synchronize();

        if (is_cpu()) {
            noa::cpu::Device::reset();
        } else {
            #ifdef NOA_ENABLE_CUDA
            noa::cuda::Device(id(), Unchecked{}).reset();
            #endif
        }
    }
}
