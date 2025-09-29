#pragma once

#include "noa/core/Error.hpp"
#include "noa/core/utils/Strings.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Pair.hpp"

#include "noa/cpu/Allocators.hpp"
#if defined(NOA_ENABLE_CUDA)
#   include "noa/gpu/cuda/Allocators.hpp"
#endif

#include "noa/unified/Device.hpp"
#include "noa/unified/Stream.hpp"

namespace noa::inline types {
    /// Memory allocator.
    class Allocator {
    public:
        /// Different types of allocators.
        /// \note Memory allocation depends on the device used for the allocation.
        enum class Enum {
            /// No allocation can be performed.
            NONE = 0,

            /// The device default allocator.
            /// - \b Allocation: For CPUs, it refers to the standard allocator using the heap as memory resource and
            ///   returning at least 256-bytes aligned pointers. For GPUs, it refers to the GPU backend's default
            ///   allocator using the GPU global memory as a resource. In CUDA, pointers have a minimum 256-bytes
            ///   alignment.
            /// - \b Accessibility: The allocated memory is private to the device that performed the allocation,
            ///   but can be used by any stream of that device.
            DEFAULT = 1,

            /// The device asynchronous allocator.
            /// - \b Allocation: Same as DEFAULT, except if the device is a CUDA-capable device. In this case,
            ///   the current stream of the device is used to perform the allocation, i.e., the allocation is
            ///   stream-ordered. Since CUDA 11.2, it has been the recommended way to allocate GPU memory.
            /// - \b Accessibility: Allocated memory is private to the device that performed the allocation,
            ///   but can be used by any stream of that device. If the device is a CUDA-capable device, one
            ///   should make sure the memory is accessed in a stream-ordered way since the memory is only
            ///   valid when the stream reaches the allocation event.
            DEFAULT_ASYNC = 2,
            ASYNC = DEFAULT_ASYNC,

            /// "Pitch" allocator.
            /// - \b Allocation: This is equivalent to DEFAULT, except for CUDA-capable devices. In this case,
            ///   the CUDA driver will potentially pad the right side of the rows to preserve a minimum alignment.
            ///   The size of the row, including the padding, is called the "pitch" in CUDA. "Pitched" layouts can be
            ///   useful to minimize the number of memory accesses on a given row, but may increase the number of
            ///   memory accesses for reading the whole array. Due to the potentially increased per-row alignment,
            ///   it can also be used to reduce shared memory bank conflicts. It is recommended to use these layouts
            ///   if the application is performing copies from/to 2d or 3d CUDA arrays, or when manipulating a stack
            ///   of row.
            /// - \b Accessibility: Allocated memory is private to the device that performed the allocation,
            ///   but can be used by any stream of that device.
            PITCHED = 3,

            /// "Pitch"-managed memory allocator.
            /// - \b Allocation: Similar to PITCHED, but is implemented on the library side (it is not supported
            ///   by CUDA itself) using the MANAGED allocator. The padding is computed to guarantee a per-row
            ///   alignment of at least 256 bytes.
            /// - \b Accessibility: Same as MANAGED.
            PITCHED_MANAGED = 4,
            PITCHED_UNIFIED = PITCHED_MANAGED,

            /// Page-locked (i.e. pinned) memory allocator.
            /// - \b Allocation: Pinned memory can be allocated by a CPU or a GPU device. Allocating excessive
            ///   amounts of pinned memory may degrade system performance, since it reduces the amount of memory
            ///   available to the system for paging. Thus, it is best used sparingly, e.g., to allocate staging
            ///   areas for data exchange between CPU and GPU. Note that accessing pinned memory from the GPU in
            ///   a non-coalesced way may result in terribly poor performance.
            /// - \b Accessibility: Can be accessed by the CPU, and the GPU against which the allocation was
            ///   performed. If the CPU device was used for allocation, this GPU is the "current" GPU at the
            ///   time of allocation. Concurrent access from the CPU and the GPU is illegal.
            PINNED = 5,

            /// Managed memory allocator.
            /// - \b Allocation: If the device is the CPU, the current GPU stream of the current GPU is used to
            ///   perform the allocation. Otherwise, the current GPU stream of the GPU device is used. While
            ///   streams are used (the memory is attached to them), the allocation itself is synchronous.
            /// - \b Accessibility: Can be accessed by the CPU. If the GPU stream used for the allocation
            ///   was the NULL stream, this is equivalent to MANAGED_GLOBAL. Otherwise, the allocated memory on
            ///   the GPU side is private to the stream and the GPU that performed the allocation. Concurrent access
            ///   from the CPU and the GPU is illegal.
            MANAGED = 6,
            UNIFIED = MANAGED,

            /// Managed memory allocator.
            /// - \b Allocation: Managed memory can be allocated by a CPU or a GPU device. In CUDA, this is much less
            ///   efficient compared to a stream-private allocation with MANAGED.
            /// - \b Accessibility: Can be accessed by any stream and any device (CPU and GPU). Concurrent access
            ///   from the CPU and the GPU is illegal.
            MANAGED_GLOBAL = 7,
            UNIFIED_GLOBAL = MANAGED_GLOBAL,

            /// CUDA array.
            /// - \b Allocation: This is only supported by CUDA-capable devices and is only used for textures.
            /// - \b Accessibility: Can only be accessed via texture fetching on the device it was allocated on.
            CUDA_ARRAY = 8
        } value{DEFAULT};

    public:
        /// Returns the number of bytes currently allocated (by the library's allocators) on the given device.
        /// \note Allocated PINNED and MANAGED memory are counted for the GPU used for the allocation, as well as for
        ///       the CPU. Allocated MANAGED_GLOBAL memory is counted for the CPU and all GPU. The counted memory
        ///       allocated as CUDA_ARRAY is only an estimate, and CUDA may allocate slightly more than that.
        [[nodiscard]] static auto bytes_currently_allocated(Device device) -> size_t {
            size_t n_bytes{};
            if (device.is_cpu()) {
                n_bytes += noa::cpu::AllocatorHeap::bytes_currently_allocated();
                #ifdef NOA_ENABLE_CUDA
                n_bytes += noa::cuda::AllocatorPinned::bytes_currently_allocated(device.id());
                n_bytes += noa::cuda::AllocatorManaged::bytes_currently_allocated(device.id());
                #endif
            } else {
                #ifdef NOA_ENABLE_CUDA
                n_bytes += noa::cuda::AllocatorDevice::bytes_currently_allocated(device.id());
                n_bytes += noa::cuda::AllocatorDevicePadded::bytes_currently_allocated(device.id());
                n_bytes += noa::cuda::AllocatorPinned::bytes_currently_allocated(device.id());
                n_bytes += noa::cuda::AllocatorManaged::bytes_currently_allocated(device.id());
                n_bytes += noa::cuda::AllocatorTexture::bytes_currently_allocated(device.id());
                #endif
            }
            return n_bytes;
        }

    public: // enum-like
        using enum Enum;
        constexpr Allocator() noexcept = default;
        constexpr /*implicit*/ Allocator(Enum value_) noexcept : value(value_) {}
        constexpr /*implicit*/ operator Enum() const noexcept { return value; }

    public: // from string
        explicit Allocator(std::string_view name) : value(parse_(name)) {}
        /*implicit*/ Allocator(const char* name) : Allocator(std::string_view(name)) {}

    public:
        /// Whether the allocator is any of the provided values.
        /// The values should be convertible to an Allocator instance,
        /// e.g. Allocator{"unified"}.is_any("pinned") == false,
        /// e.g. Allocator{"unified"}.is_any("pinned", Allocator::UNIFIED) == true.
        [[nodiscard]] constexpr auto is_any(const auto&... values) const {
            auto get_resource = []<typename T>(const T& v) {
                if constexpr (nt::any_of<T, Allocator, Enum>) {
                    return v;
                } else if constexpr (std::convertible_to<decltype(v), std::string_view>) {
                    return parse_(v);
                } else {
                    static_assert(nt::always_false<>);
                }
            };
            return ((value == get_resource(values)) or ...);
        }

        /// Allocates, using the current allocator, \p n_elements from a memory resource accessible to the \p device.
        /// \warning The underlying allocators are using malloc-like functions, thus return uninitialized memory
        ///          with the appropriate alignment requirement and size, i.e. it is undefined behavior to directly
        ///          read from these memory regions.
        /// \note This is intended to be used as part of the Array allocation, as the allocated resource is
        ///       converted to a shared_ptr. This is because the underlying allocators return different types
        ///       (they have different deleters), so we have to type erase them with the shared_ptr.
        template<typename T>
        auto allocate(
            i64 n_elements,
            const Device& device
        ) -> std::shared_ptr<T[]> {
            if (not n_elements)
                return nullptr;

            switch (value) {
                case Allocator::NONE:
                    return {};
                case Allocator::DEFAULT: {
                    if (device.is_cpu()) {
                        return noa::cpu::AllocatorHeap::allocate<T>(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const auto cuda_device = noa::cuda::Device(device.id(), noa::cuda::Device::DeviceUnchecked{});
                        return noa::cuda::AllocatorDevice::allocate<T>(n_elements, cuda_device);
                        #else
                        panic_no_gpu_backend();
                        #endif
                    }
                }
                case Allocator::DEFAULT_ASYNC: {
                    if (device.is_cpu()) {
                        return noa::cpu::AllocatorHeap::allocate<T>(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        return noa::cuda::AllocatorDevice::allocate_async<T>(
                            n_elements, Stream::current(device).cuda());
                        #else
                        panic_no_gpu_backend();
                        #endif
                    }
                }
                case Allocator::PITCHED: {
                    if (device.is_cpu()) {
                        return noa::cpu::AllocatorHeap::allocate<T>(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const auto cuda_device = noa::cuda::Device(device.id(), noa::cuda::Device::DeviceUnchecked{});
                        // AllocatorDevicePadded requires sizeof(T) <= 16 bytes.
                        if constexpr (nt::numeric<T>) {
                            auto shape = Shape<i64, 4>::from_values(1, 1, 1, n_elements);
                            return noa::cuda::AllocatorDevicePadded::allocate<T>(shape, cuda_device).first;
                        } else {
                            return noa::cuda::AllocatorDevice::allocate<T>(n_elements, cuda_device);
                        }
                        #else
                        panic_no_gpu_backend();
                        #endif
                    }
                }
                case Allocator::PINNED: {
                    if (device.is_cpu() and not Device::is_any(Device::GPU)) {
                        return noa::cpu::AllocatorHeap::allocate<T>(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const auto allocating_device = device.is_gpu() ? device : Device::current_gpu();
                        const auto cuda_device = noa::cuda::Device(allocating_device.id(), noa::cuda::Device::DeviceUnchecked{});
                        return noa::cuda::AllocatorPinned::allocate<T>(n_elements, cuda_device);
                        #else
                        panic_no_gpu_backend();
                        #endif
                    }
                }
                case Allocator::PITCHED_MANAGED:
                case Allocator::MANAGED: {
                    if (device.is_cpu() and not Device::is_any(Device::GPU)) {
                        return noa::cpu::AllocatorHeap::allocate<T>(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const auto gpu = device.is_gpu() ? device : Device::current_gpu();
                        const auto guard = DeviceGuard(gpu); // could be helpful when retrieving the device
                        return noa::cuda::AllocatorManaged::allocate<T>(n_elements, Stream::current(gpu).cuda());
                        #else
                        panic_no_gpu_backend();
                        #endif
                    }
                }
                case Allocator::MANAGED_GLOBAL: {
                    if (device.is_cpu() and not Device::is_any(Device::GPU)) {
                        return noa::cpu::AllocatorHeap::allocate<T>(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const auto guard = DeviceGuard(device.is_gpu() ? device : Device::current_gpu());
                        return noa::cuda::AllocatorManaged::allocate_global<T>(n_elements);
                        #else
                        panic_no_gpu_backend();
                        #endif
                    }
                }
                case Allocator::CUDA_ARRAY:
                    panic("CUDA arrays is not supported by this function. Use Texture instead");
            }
            std::terminate(); // unreachable
        }

        /// Similar to allocate, but for 4d shapes. If Allocator::PITCHED, the returned memory is optimized for
        /// efficient per-row access. This may be done at the cost of an extra padding of the rows to preserve the
        /// memory alignment. Of course, this padding is encoded in the returned strides.
        template<typename T>
        auto allocate_pitched(
            const Shape4<i64>& shape,
            const Device& device
        ) -> Pair<std::shared_ptr<T[]>, Strides4<i64>> {
            switch (value) {
                case Allocator::NONE:
                    return {};
                case Allocator::PITCHED: {
                    if (device.is_cpu()) {
                        return {noa::cpu::AllocatorHeap::allocate<T>(shape.n_elements()), shape.strides()};
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        // AllocatorDevicePadded requires sizeof(T) <= 16 bytes.
                        const auto cuda_device = noa::cuda::Device(device.id(), noa::cuda::Device::DeviceUnchecked{});
                        if constexpr (nt::numeric<T>) {
                            auto [ptr, strides] = noa::cuda::AllocatorDevicePadded::allocate<T>(shape, cuda_device);
                            return {std::move(ptr), strides};
                        } else {
                            return {noa::cuda::AllocatorDevice::allocate<T>(shape.n_elements(), cuda_device), shape.strides()};
                        }
                        #else
                        panic_no_gpu_backend();
                        #endif
                    }
                }
                case Allocator::PITCHED_MANAGED: {
                    if (device.is_cpu()) {
                        return {noa::cpu::AllocatorHeap::allocate<T>(shape.n_elements()), shape.strides()};
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        auto& cuda_stream = Stream::current(device).cuda();
                        auto [ptr, strides] = noa::cuda::AllocatorManagedPadded::allocate<T>(shape, cuda_stream);
                        return {std::move(ptr), strides};
                        #else
                        panic_no_gpu_backend();
                        #endif
                    }
                }
                default:
                    return {allocate<T>(shape.n_elements(), device), shape.strides()};
            }
        }

        /// Check that the pointer matches the allocator and device.
        void validate(const void* ptr, const Device& device);

    private:
        static Enum parse_(std::string_view name);
    };
}

namespace noa::inline types {
    auto operator<<(std::ostream& os, Allocator::Enum allocator) -> std::ostream&;
    inline auto operator<<(std::ostream& os, Allocator allocator) -> std::ostream& {
        return os << allocator.value;
    }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator (operator<<))
namespace fmt {
    template<> struct formatter<noa::Allocator::Enum> : ostream_formatter {};
    template<> struct formatter<noa::Allocator> : ostream_formatter {};
}
