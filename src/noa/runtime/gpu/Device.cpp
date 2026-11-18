#include "noa/runtime/gpu/Device.hpp"
#include "noa/runtime/gpu/Blas.hpp"

namespace {
    std::mutex g_mutex{};
    using device_callback_t = noa::gpu::Device::device_callback_type;
    std::vector<device_callback_t> g_reset_callbacks{};
    std::vector<device_callback_t> g_synchronize_callbacks{};

    void add_callback_(std::vector<device_callback_t>& callbacks, device_callback_t callback) {
        if (callback == nullptr)
            return;

        auto lock = std::lock_guard(g_mutex);
        for (const auto p: callbacks)
            if (p == callback)
                return;
        callbacks.push_back(callback);
    }
}

#ifdef NOA_ENABLE_SYCL
namespace {
    // Per-thread device, defaults to 0, like CUDA/HIP.
    thread_local noa::i32 tl_sycl_current_device_id = 0;
}

namespace noa::gpu::details {
    auto sycl_devices() -> const std::vector<sycl::device>& {
        // Order is implementation-defined, and controlled by env vars:
        // ONEAPI_DEVICE_SELECTOR (DPC++), ACPP_VISIBILITY_MASK (AdaptiveCpp).
        static const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        return devices;
    }

    auto sycl_current_device_id() -> i32 {
        return tl_sycl_current_device_id;
    }

    void sycl_set_current_device_id(i32 id) {
        NOA_ASSERT(0 <= id and static_cast<usize>(id) < sycl_devices().size());
        tl_sycl_current_device_id = id;
    }

    auto sycl_device_context(i32 id) -> const sycl::context& {
        // One context per device so that USM pointers are valid for every queue of that device.
        static const auto contexts = [] {
            auto v = std::vector<sycl::context>{};
            for (const auto& d: sycl_devices())
                v.emplace_back(d, sycl_async_error_handler);
            return v;
        }();
        return contexts.at(static_cast<usize>(id));
    }
}
#endif

namespace noa::gpu {
    void Device::add_synchronize_callback(device_callback_type callback) {
        add_callback_(g_synchronize_callbacks, callback);
    }

    void Device::remove_synchronize_callback(device_callback_type callback) {
        auto lock = std::lock_guard(g_mutex);
        std::erase(g_synchronize_callbacks, callback);
    }

    void Device::add_reset_callback(device_callback_type callback) {
        add_callback_(g_reset_callbacks, callback);
    }

    void Device::remove_reset_callback(device_callback_type callback) {
        auto lock = std::lock_guard(g_mutex);
        std::erase(g_reset_callbacks, callback);
    }

    void Device::synchronize() const {
        const auto guard = DeviceGuard(*this);

        auto lock = std::lock_guard(g_mutex);
        for (auto p: g_synchronize_callbacks)
            p(this->id());

#if defined(NOA_ENABLE_CUDA)
        check(cudaDeviceSynchronize());
#elif defined(NOA_ENABLE_ROCM)
        check(hipDeviceSynchronize());
#elif defined(NOA_ENABLE_SYCL)
        // no-op
#endif
    }

    void Device::reset() const {
        const auto guard = DeviceGuard(*this);
        guard.synchronize(); // if called from noa::Device::reset(), the device is already synchronized

        auto lock = std::lock_guard(g_mutex);
        for (auto p: g_reset_callbacks)
            p(this->id());

#if defined(NOA_ENABLE_CUDA)
        check(cudaDeviceReset());
#elif defined(NOA_ENABLE_ROCM)
        check(hipDeviceReset());
#elif defined(NOA_ENABLE_SYCL)
        // no-op
#endif
    }
}
