#include "noa/runtime/cuda/Device.hpp"
#include "noa/runtime/cuda/Blas.hpp"

namespace {
    std::mutex g_mutex{};
    std::vector<noa::cuda::Device::reset_callback_type> g_callbacks{};
}

namespace noa::cuda {
    void Device::add_reset_callback(reset_callback_type callback) {
        if (callback == nullptr)
            return;

        auto lock = std::lock_guard(g_mutex);
        for (auto p: g_callbacks)
            if (p == callback)
                return;
        g_callbacks.push_back(callback);
    }

    void Device::remove_reset_callback(reset_callback_type callback) {
        auto lock = std::lock_guard(g_mutex);
        std::erase(g_callbacks, callback);
    }

    void Device::reset() const {
        const auto guard = DeviceGuard(*this);
        guard.synchronize(); // if called from noa::Device::reset(), the device is already synchronized

        auto lock = std::lock_guard(g_mutex);
        for (auto p: g_callbacks)
            p(this->id());

        check(cudaDeviceReset());
    }
}
