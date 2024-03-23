#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include <ostream>
#include "noa/core/Exception.hpp"
#include "noa/core/string/Format.hpp"

#include "noa/cpu/AllocatorHeap.hpp"
#if defined(NOA_ENABLE_CUDA)
#include "noa/gpu/cuda/AllocatorDevice.hpp"
#include "noa/gpu/cuda/AllocatorDevicePadded.hpp"
#include "noa/gpu/cuda/AllocatorManaged.hpp"
#include "noa/gpu/cuda/AllocatorPinned.hpp"
#endif

#include "noa/unified/Device.hpp"
#include "noa/unified/Stream.hpp"

namespace noa::inline types {
    /// Memory allocation depends on the device used for the allocation.
    enum class MemoryResource {
        /// No allocation can be performed.
        NONE = 0,

        /// The device default allocator.
        /// - \b Allocation: For CPUs, it refers to the standard allocator using the heap as resource and
        ///   returning at least 64-bytes aligned pointer. For GPUs, it refers to the GPU backend default
        ///   allocator using the GPU global memory as resource. In CUDA, pointers have a minimum 256-bytes
        ///   alignment. Allocations do not use the current stream.
        /// - \b Accessibility: Allocated memory is private to the device that performed the allocation,
        ///   but can be used by any stream of that device.
        DEFAULT = 1,

        /// The device asynchronous allocator.
        /// - \b Allocation: Same as DEFAULT, except if the device is a CUDA-capable device. In this case,
        ///   the current stream of the device is used to performed the allocation, which is thereby stream-
        ///   ordered. Since CUDA 11.2, it is the recommend way to allocate GPU memory. The alignment is
        ///   the same as DEFAULT.
        /// - \b Accessibility: Allocated memory is private to the device that performed the allocation,
        ///   but can be used by any stream of that device. If the device is a CUDA-capable device, one
        ///   should make sure the memory is accessed in the appropriate stream-order after allocation since
        ///   the memory is only valid when the stream reaches the allocation event.
        DEFAULT_ASYNC = 2,
        ASYNC = DEFAULT_ASYNC,

        /// "Pitch" allocator.
        /// - \b Allocation: This is equivalent to DEFAULT, except for CUDA-capable devices. In this case,
        ///   the CUDA driver will potentially pad the right side of the innermost dimension of the ND array.
        ///   The size of the innermost dimension, including the padding, is called the "pitch". "Pitched"
        ///   layouts can be useful to minimize the number of memory accesses on a given row (but can increase
        ///   the number of memory accesses for reading the whole array) and to reduce shared memory bank
        ///   conflicts. It is highly recommended to use these layouts if the application will be performing
        ///   memory copies involving 2D or 3D CUDA arrays. Allocations do not use the current stream.
        /// - \b Accessibility: Allocated memory is private to the device that performed the allocation,
        ///   but can be used by any stream of that device.
        PITCHED = 4,

        /// Page-locked (i.e. pinned) memory allocator.
        /// - \b Allocation: Pinned memory can be allocated by a CPU or a GPU device. Allocating excessive
        ///   amounts of pinned memory may degrade system performance, since it reduces the amount of memory
        ///   available to the system for paging. Thus, it is best used sparingly, e.g. to allocate staging
        ///   areas for data exchange between CPU and GPU. Allocations do not use the current stream.
        /// - \b Accessibility: Can be accessed by the CPU, and the GPU against which the allocation was
        ///   performed. If the CPU device was used for allocation, this GPU is the "current" GPU at the
        ///   time of allocation.
        PINNED = 8,

        /// Managed memory allocator.
        /// - \b Allocation: If the device is the CPU, the current GPU stream of the current GPU is used to
        ///   perform the allocation. Otherwise, the current GPU stream of the GPU device is used. While
        ///   streams are used (the memory is attached to them), the allocation itself is synchronous.
        /// - \b Accessibility: Can be accessed by the CPU. If the GPU stream used for the allocation
        ///   was the NULL stream, this is equivalent to MANAGED_GLOBAL. Otherwise, the allocated memory on
        ///   the GPU side is private to the stream and the GPU that performed the allocation.
        MANAGED = 16,
        UNIFIED = MANAGED,

        /// Managed memory allocator.
        /// - \b Allocation: Managed memory can be allocated by a CPU or a GPU device. Allocation does not
        ///   use the current stream. Note that this is much less efficient compared to a stream-private
        ///   allocation with MANAGED.
        /// - \b Accessibility: Can be accessed by any stream and any device (CPU and GPU).
        MANAGED_GLOBAL = 32,
        UNIFIED_GLOBAL = MANAGED_GLOBAL,

        /// CUDA array.
        /// - \b Allocation: This is only supported by CUDA-capable devices and is only used for textures.
        /// - \b Accessibility: Can only be accessed via texture fetching on the device it was allocated on.
        CUDA_ARRAY = 64
    };

    inline std::ostream& operator<<(std::ostream& os, MemoryResource memory_resource) {
        switch (memory_resource) {
            case MemoryResource::NONE:
                return os << "none";
            case MemoryResource::DEFAULT:
                return os << "default";
            case MemoryResource::ASYNC:
                return os << "async";
            case MemoryResource::PITCHED:
                return os << "pitched";
            case MemoryResource::PINNED:
                return os << "pinned";
            case MemoryResource::UNIFIED:
                return os << "unified";
            case MemoryResource::UNIFIED_GLOBAL:
                return os << "unified_global";
            case MemoryResource::CUDA_ARRAY:
                return os << "cuda_array";
        }
        return os;
    }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator (operator<<))
namespace fmt {
    template<> struct formatter<noa::MemoryResource> : ostream_formatter {};
}

namespace noa::inline types {
    /// Memory allocator.
    class Allocator {
    public:
        Allocator() = default;
        constexpr /*implicit*/ Allocator(MemoryResource memory_resource) : m_memory_resource(memory_resource) {}
        explicit Allocator(std::string_view name) : m_memory_resource(parse_memory_resource_(name)) {}
        /*implicit*/ Allocator(const char* name) : Allocator(std::string_view(name)) {}

        [[nodiscard]] constexpr MemoryResource resource() const noexcept { return m_memory_resource; }

    public:
        /// Allocates \p n_elements from the given \p memory_resource.
        /// \note This is intended to be used as part of the Array allocation, as the allocated resource is
        ///       converted to a shared_ptr. This is because the underlying allocators return different types
        ///       (they have different deleters), so we have to type erase them with the shared_ptr.
        template<typename T>
        static auto allocate(
                i64 n_elements,
                const Device& device,
                MemoryResource memory_resource
        ) -> std::shared_ptr<T> {
            if (!n_elements)
                return nullptr;

            switch (memory_resource) {
                case MemoryResource::NONE:
                    break;
                case MemoryResource::DEFAULT: {
                    if (device.is_cpu()) {
                        return noa::cpu::AllocatorHeap<T>::allocate(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const DeviceGuard guard(device);
                        return noa::cuda::AllocatorDevice<T>::allocate(n_elements);
                        #endif
                    }
                }
                case MemoryResource::DEFAULT_ASYNC: {
                    if (device.is_cpu()) {
                        return noa::cpu::AllocatorHeap<T>::allocate(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        return noa::cuda::AllocatorDevice<T>::allocate_async(
                                n_elements, Stream::current(device).cuda());
                        #endif
                    }
                }
                case MemoryResource::PITCHED: {
                    if (device.is_cpu()) {
                        return noa::cpu::AllocatorHeap<T>::allocate(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const DeviceGuard guard(device);
                        // AllocatorDevicePadded requires sizeof(T) <= 16 bytes.
                        if constexpr (nt::is_numeric_v<T>) {
                            return noa::cuda::AllocatorDevicePadded<T>::allocate({1, 1, 1, n_elements}).first;
                        } else {
                            return noa::cuda::AllocatorDevice<T>::allocate(n_elements);
                        }
                        #endif
                    }
                }
                case MemoryResource::PINNED: {
                    if (device.is_cpu() and not Device::is_any(DeviceType::GPU)) {
                        return noa::cpu::AllocatorHeap<T>::allocate(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const DeviceGuard guard(device.is_gpu() ? device : Device::current(DeviceType::GPU));
                        return noa::cuda::AllocatorPinned<T>::allocate(n_elements);
                        #endif
                    }
                }
                case MemoryResource::MANAGED: {
                    if (device.is_cpu() and not Device::is_any(DeviceType::GPU)) {
                        return noa::cpu::AllocatorHeap<T>::allocate(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const Device gpu = device.is_gpu() ? device : Device::current(DeviceType::GPU);
                        const DeviceGuard guard(gpu); // could be helpful when retrieving device
                        auto& cuda_stream = Stream::current(gpu).cuda();
                        return noa::cuda::AllocatorManaged<T>::allocate(n_elements, cuda_stream);
                        #endif
                    }
                }
                case MemoryResource::MANAGED_GLOBAL: {
                    if (device.is_cpu() and not Device::is_any(DeviceType::GPU)) {
                        return noa::cpu::AllocatorHeap<T>::allocate(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const DeviceGuard guard(device.is_gpu() ? device : Device::current(DeviceType::GPU));
                        return noa::cuda::AllocatorManaged<T>::allocate_global(n_elements);
                        #endif
                    }
                }
                case MemoryResource::CUDA_ARRAY:
                    panic("CUDA arrays are not supported by this allocator. Use Texture instead");
                default:
                    panic("{} is not supported by this allocator", memory_resource);
            }
        }

        template<typename T>
        static auto allocate_pitched(
                const Shape4<i64>& shape,
                const Device& device,
                MemoryResource memory_resource
        ) -> Pair<std::shared_ptr<T>, Strides4<i64>> {
            switch (memory_resource) {
                case MemoryResource::NONE:
                    break;
                case MemoryResource::PITCHED: {
                    if (device.is_cpu()) {
                        return {noa::cpu::AllocatorHeap<T>::allocate(shape.elements()), shape.strides()};
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const DeviceGuard guard(device);
                        // AllocatorDevicePadded requires sizeof(T) <= 16 bytes.
                        if constexpr (nt::is_numeric_v<T>) {
                            auto [ptr, strides] = noa::cuda::AllocatorDevicePadded<T>::allocate(shape);
                            return {std::move(ptr), strides};
                        } else {
                            return {noa::cuda::AllocatorDevice<T>::allocate(shape.elements()), shape.strides()};
                        }
                        #endif
                    }
                }
                default:
                    return {allocate<T>(shape.elements(), device, memory_resource), shape.strides()};
            }
        }

    private:
        static MemoryResource parse_memory_resource_(std::string_view name) {
            std::string str_ = ns::to_lower(ns::trim(name));
            std::replace(str_.begin(), str_.end(), '-', '_');

            if (str_ == "default") {
                return MemoryResource::DEFAULT;
            } else if (str_ == "default_async" or str_ == "async") {
                return MemoryResource::DEFAULT_ASYNC;
            } else if (str_ == "pitched") {
                return MemoryResource::PITCHED;
            } else if (str_ == "pinned") {
                return MemoryResource::PINNED;
            } else if (str_ == "managed" or str_ == "unified_private") {
                return MemoryResource::MANAGED;
            } else if (str_ == "managed_global" or str_ == "unified") {
                return MemoryResource::MANAGED_GLOBAL;
            } else if (str_ == "cuda_array") {
                return MemoryResource::CUDA_ARRAY;
            } else if (str_ == "none" or str_.empty()) {
                return MemoryResource::NONE;
            } else {
                panic("{} is not a valid memory resource", str_);
            }
        }

    private:
        MemoryResource m_memory_resource{MemoryResource::DEFAULT};
    };
}
#endif
