#include "noa/unified/Device.h"

namespace {
    thread_local noa::Device g_device{noa::Device::CPU, -1, true}; // per-thread current device
}

namespace noa {
    Device Device::current() {
        return g_device;
    }

    void Device::current(Device device) {
        g_device = device;
        #ifdef NOA_ENABLE_CUDA
        if (device.gpu())
            cuda::Device::current(cuda::Device(device.id(), true));
        #endif
    }
}
