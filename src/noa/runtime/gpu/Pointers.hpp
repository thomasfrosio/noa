#pragma once

#include "noa/runtime/gpu/Device.hpp"
#include "noa/runtime/gpu/Error.hpp"
#include "noa/runtime/gpu/Runtime.hpp"

namespace noa::gpu {
    enum class MemoryKind {
        /// Unknown to the GPU runtime. Plain CPU malloc, not accessible to the device.
        UNREGISTERED = 0,

        /// Only available on the device used during the allocation.
        /// This is the fastest and recommended option for device-only usage.
        /// \see AllocatorDevice for more details.
        DEVICE = 1,

        /// Both available to the host and device(s), possibly by automatic data movements upon page-faults.
        /// Host and device accesses should be exclusive; to access the memory from the host, there should be no
        /// (pending) accesses of the memory region from the device, and vice versa.
        /// \see AllocatorUnified for more details, especially on device visibility.
        /// TODO Concurrent host/device access is likely possible (see cudaDevAttrConcurrentManagedAccess,
        ///      sycl::aspect::usm_atomic_shared_allocations).
        UNIFIED = 2,

        /// Both available to the host and device by pinning the memory (mapped for DMA/device access).
        /// In CUDA/HIP, all devices have access. In SYCL, only by the context used for the allocation
        /// (in our case, contexts are per-device).
        /// \see AllocatorPinned for more details.
        PINNED = 3,
    };

    struct PointerInfo {
        MemoryKind kind{};
        i32 device_id{-1};
        void* device_ptr{nullptr};

        template<typename T>
        [[nodiscard]] static auto from_pointer(const T* ptr) -> PointerInfo {
            void* p = const_cast<T*>(ptr);
#if defined(NOA_ENABLE_CUDA)
            cudaPointerAttributes attr;
            check(cudaPointerGetAttributes(&attr, ptr));
            switch (attr.type) {
                case cudaMemoryTypeHost:
                    // CUDA: "The pointer value through which allocated host memory may be accessed
                    // in kernels on all devices that support unified addressing is the same
                    // as the pointer value through which that memory is accessed on the host.
                    // It is not necessary to call ::cudaHostGetDevicePointer() to get the device
                    // pointer for these allocations." So devicePointer == p.
                    return {MemoryKind::PINNED, attr.device, attr.devicePointer};
                case cudaMemoryTypeDevice:
                    return {MemoryKind::DEVICE, attr.device, p};
                case cudaMemoryTypeManaged:
                    return {MemoryKind::UNIFIED, attr.device, p};
                default:
                    return {MemoryKind::UNREGISTERED, -1, nullptr};
            }
#elif defined(NOA_ENABLE_HIP)
            hipPointerAttribute_t attr{};
            if (hipPointerGetAttributes(&attr, ptr) != hipSuccess) {
                (void) hipGetLastError();
                return {MemoryKind::UNREGISTERED, -1, nullptr};
            }
            if (attr.isManaged)
                return {MemoryKind::UNIFIED, attr.device, p};
            if (attr.type == hipMemoryTypeDevice)
                return {MemoryKind::DEVICE, attr.device, p};
            if (attr.type == hipMemoryTypeHost)
                return {MemoryKind::PINNED, attr.device, attr.devicePointer};
            return {MemoryKind::UNREGISTERED, -1, nullptr};
#elif defined(NOA_ENABLE_SYCL)
            // get_pointer_type() is per context, and we have one context per device (see Stream).
            // A pointer from another device's context reports unknown, so we have to ask every context.
            // Plain malloc'ed memory is unknown in all contexts, so it is unregistered.
            for (i32 id = 0; id < Device::count(); ++id) {
                switch (sycl::get_pointer_type(ptr, details::sycl_device_context(id))) {
                    case sycl::usm::alloc::device:
                        return {MemoryKind::DEVICE, id, p};
                    case sycl::usm::alloc::shared:
                        return {MemoryKind::UNIFIED, id, p};
                    case sycl::usm::alloc::host:
                        return {MemoryKind::PINNED, id, p};
                    default: break;
                }
            }
            return {MemoryKind::UNREGISTERED, -1, nullptr};
#endif
        }

        [[nodiscard]] bool is_accessible_from_device() const { return kind != MemoryKind::UNREGISTERED; }
        [[nodiscard]] bool is_accessible_from_host() const { return kind != MemoryKind::DEVICE; }

        /// Whether device_ptr can be dereferenced from the given device ID.
        /// \note This function is fragile for the following reasons:
        /// 1. With SYCL, we assume the allocation was done using our per-device contexts (e.g. using AllocatorDevice,
        ///    AllocatorUnified, or AllocatorPinned), resulting in only having to check that the pointer's device and
        ///    the given device ID match.
        /// 2. With CUDA, PINNED memory is available to any stream and any device, since we assume all devices support
        ///    unified addressing (which seem very reasonable). UNIFIED memory may have global access or be private to
        ///    a stream. This function assumes UNIFIED memory has global visibility since most systems have a device
        ///    with cudaDevAttrConcurrentManagedAccess != 0. For older devices, this may be zero and the visibility may
        ///    indeed have been restrained to a stream (see AllocatorUnified::STREAM). Let the driver report an error
        ///    for this rare scenario.
        [[nodiscard]] bool is_accessible_from_device_id(i32 id) const {
            #if defined(NOA_ENABLE_SYCL)
            return device_id == id;
            #else
            // Assume pinned and managed memory is accessible from any device.
            // Device memory only from its own device (assuming peer access is disabled).
            return kind != MemoryKind::DEVICE or device_id == id;
            #endif
        }
    };

#ifndef NOA_ENABLE_SYCL
    /// Returns the device address of constant_pointer.
    template<typename T = void>
    auto constant_to_device_pointer(const void* constant_pointer) -> T* {
        void* device_pointer;
#   if defined(NOA_ENABLE_CUDA)
        check(cudaGetSymbolAddress(&device_pointer, constant_pointer));
#   else
        check(hipGetSymbolAddress(&device_pointer, constant_pointer));
#   endif
        return static_cast<T*>(device_pointer);
    }
#endif
}
