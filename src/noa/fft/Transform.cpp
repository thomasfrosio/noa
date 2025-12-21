#include "noa/fft/Transform.hpp"

#include "noa/fft/cpu/Plan.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/fft/cuda/Plan.hpp"
#endif

namespace noa::fft {
    auto clear_cache(Device device) -> i32 {
        if (device.is_cpu())
            return noa::fft::cpu::clear_cache();
        #ifdef NOA_ENABLE_CUDA
        auto cuda_device = noa::cuda::Device(device.id(), Unchecked{});
        return noa::fft::cuda::clear_cache(cuda_device);
        #else
        return 0;
        #endif
    }

    auto set_cache_limit(i32 count, Device device) -> i32 {
        #ifdef NOA_ENABLE_CUDA
        if (device.is_gpu()) {
            auto cuda_device = noa::cuda::Device(device.id(), Unchecked{});
            return noa::fft::cuda::set_cache_limit(cuda_device, safe_cast<i32>(count));
        }
        #else
        (void) count;
        (void) device;
        #endif
        return -1;
    }

    auto cache_limit(Device device) -> i32 {
        #ifdef NOA_ENABLE_CUDA
        if (device.is_gpu()) {
            auto cuda_device = noa::cuda::Device(device.id(), Unchecked{});
            return noa::fft::cuda::cache_limit(cuda_device);
        }
        #else
        (void) device;
        #endif
        return -1;
    }

    auto cache_size(Device device) -> i32 {
        if (device.is_cpu())
            return noa::fft::cpu::cache_size();
        #ifdef NOA_ENABLE_CUDA
        auto cuda_device = noa::cuda::Device(device.id(), Unchecked{});
        return noa::fft::cuda::cache_size(cuda_device);
        #else
        return -1;
        #endif
    }

    auto workspace_left_to_allocate(Device device) -> isize {
        #ifdef NOA_ENABLE_CUDA
        if (device.is_gpu()) {
            auto cuda_device = noa::cuda::Device(device.id(), Unchecked{});
            return noa::fft::cuda::workspace_left_to_allocate(cuda_device);
        }
        #else
        (void) device;
        #endif
        return 0;
    }

    auto details::set_workspace(Device device, const std::shared_ptr<std::byte[]>& buffer, isize buffer_bytes) -> i32 {
        #ifdef NOA_ENABLE_CUDA
        if (device.is_gpu()) {
            auto cuda_device = noa::cuda::Device(device.id(), Unchecked{});
            return noa::fft::cuda::set_workspace(cuda_device, buffer, buffer_bytes);
        }
        #else
        (void) device;
        (void) buffer;
        (void) buffer_bytes;
        #endif
        return 0;
    }
}
