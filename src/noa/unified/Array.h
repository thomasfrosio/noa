#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Indexing.h"
#include "noa/common/Types.h"
#include "noa/common/types/View.h"

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrDevicePadded.h"
#include "noa/gpu/cuda/memory/PtrManaged.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"
#include "noa/gpu/cuda/util/Pointers.h"
#endif

#include "noa/unified/Device.h"
#include "noa/unified/Stream.h"

namespace noa {
    /// Memory allocators.
    /// Memory allocation depends on the device used for the allocation.
    enum class Allocator {
        /// No allocation is performed.
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

        /// Managed memory allocator.
        /// - \b Allocation: Managed memory can be allocated by a CPU or a GPU device. Allocation does not
        ///   use the current stream. Note that this is much less efficient compared to a stream-private
        ///   allocation with MANAGED.
        /// - \b Accessibility: Can be accessed by any stream and any device (CPU and GPU).
        MANAGED_GLOBAL = 32,
    };

    std::ostream& operator<<(std::ostream& os, Allocator resource);
}

namespace noa {
    /// Options for Array(s).
    class ArrayOption {
    public: // Constructors
        /// Sets the array options.
        /// \param device       Device of the array. Defaults to the CPU.
        /// \param allocator    Allocator of the array. Defaults to the default device allocator.
        constexpr /*implicit*/ ArrayOption(Device device = {}, Allocator allocator = Allocator::DEFAULT)
                : m_device(device), m_allocator(allocator) {}

    public: // Setters
        constexpr ArrayOption& device(Device device) noexcept {
            m_device = device;
            return *this;
        }

        constexpr ArrayOption& allocator(Allocator allocator) noexcept {
            m_allocator = allocator;
            return *this;
        }

    public: // Getters
        [[nodiscard]] constexpr Device device() const noexcept { return m_device; }
        [[nodiscard]] constexpr Allocator allocator() const noexcept { return m_allocator; }

    private:
        Device m_device;
        Allocator m_allocator;
        // TODO Switch to uint32_t bitset?
    };
}

namespace noa {
    /// 4-dimensional array of any data type.
    /// \details
    /// - \b Resource: Arrays manage a reference-counted resource, which can be shared, moved, copied, etc.
    ///   The resource type and location depends on the ArrayOption used to create the array. Arrays are therefore
    ///   attached to a device, either the CPU or a GPU. Depending on the resource (thus the allocator used to create
    ///   the array), arrays can be interpreted as CPU or GPU and it is possible to create aliases of the same
    ///   resource for the CPU and the GPU (see as() for more details).
    /// - \b Accessors: When a backend call is needed, arrays will use the current stream of their device, they
    ///   are "stream-safe". In other words, one does not and should not synchronize the stream between each operation
    ///   involving an Array, except if the result of that operation is used in a "unsafe" way. These unsafe ways of
    ///   accessing the managed data are by get(), share() or view(). While these are often required for e.g. efficient
    ///   loop-like indexing, one must make sure the current stream of the Array's device is synchronized.
    template<typename T>
    class Array {
    public: // typedefs
        using value_t = T;
        using dim_t = size_t;
        using dim4_t = Int4<dim_t>;
        using ptr_t = T*;
        using ref_t = T&;

        static_assert(!std::is_const_v<T>);
        static_assert(!std::is_pointer_v<T>);
        static_assert(!std::is_reference_v<T>);
        static_assert(noa::traits::is_data_v<T> ||
                      noa::traits::is_intX_v<T> ||
                      noa::traits::is_floatX_v<T>);

        template<typename U>
        static constexpr bool is_indexable_v =
                std::bool_constant<noa::traits::is_int_v<U> ||
                                   noa::traits::is_same_v<U, indexing::full_extent_t> ||
                                   noa::traits::is_same_v<U, indexing::slice_t>>::value;

    public: // Constructors
        /// Creates an empty array.
        constexpr Array() = default;

        /// Creates a contiguous array.
        /// \param shape    Rightmost shape of the array.
        /// \param option   Options of the created array.
        /// \see Allocator for more details.
        constexpr explicit Array(size_t elements, ArrayOption option = {})
                : m_shape(size4_t{1, 1, 1, elements}),
                  m_stride(size4_t{elements, elements, elements, 1}),
                  m_options(option) { alloc_(); }

        /// Creates a contiguous array.
        /// \param shape    Rightmost shape of the array.
        /// \param option   Options of the created array.
        /// \see Allocator for more details.
        constexpr explicit Array(size4_t shape, ArrayOption option = {})
                : m_shape(shape), m_stride(shape.stride()), m_options(option) { alloc_(); }

        /// Creates a strided array.
        /// \param shape    Rightmost shape of the array.
        /// \param stride   Rightmost stride of the array.
        /// \param option   Options of the created array.
        /// \note An array large enough to fit the physical layout specified by \p stride is allocated.
        ///       Otherwise, this constructor behaves the same way as for contiguous arrays.
        constexpr Array(size4_t shape, size4_t stride, ArrayOption option = {})
                : m_shape(shape), m_stride(stride), m_options(option) { alloc_(true); }

        /// Creates a non-owning array from an existing allocated memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param shape        Rightmost shape of \p data.
        /// \param stride       Rightmost stride of \p data.
        /// \param option       Options of \p data.
        constexpr Array(T* data, size4_t shape, size4_t stride, ArrayOption option)
                : m_shape(shape), m_stride(stride), m_ptr(data, [](void*) {}), m_options(option) {
            validate_(data, option);
        }

        /// Creates an array from an existing allocated memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param shape        Rightmost shape of \p data.
        /// \param stride       Rightmost stride of \p data.
        /// \param option       Options of \p data.
        constexpr Array(shared_t<T[]> data, size4_t shape, size4_t stride, ArrayOption option)
                : m_shape(shape), m_stride(stride), m_ptr(std::move(data)), m_options(option) {
            validate_(data.get(), option);
        }

    public: // Getters
        /// Returns the options used to create the array.
        [[nodiscard]] constexpr ArrayOption options() const noexcept { return m_options; }

        /// Returns the device used to create the array.
        [[nodiscard]] constexpr Device device() const noexcept { return m_options.device(); }

        /// Returns the memory resource used to create the array.
        [[nodiscard]] constexpr Allocator allocator() const noexcept { return m_options.allocator(); }

        /// Whether the managed data can be accessed by CPU threads.
        /// \note While it indicates whether the managed data can be dereferenced on the CPU, it does not indicate
        ///       if the Array will be seen as a CPU (or a GPU) array. This choice is purely made on the Array's
        ///       device type. For instance, pinned memory can be dereferenced by the CPU, so this function will
        ///       returned true, but if the Array's device is a GPU, the implementations will refer to this Array
        ///       as a GPU array and will therefore prioritizing GPU access.
        [[nodiscard]] constexpr bool dereferencable() const noexcept {
            return m_options.device().cpu() || allocator() == Allocator::PINNED ||
                   allocator() == Allocator::MANAGED || allocator() == Allocator::MANAGED_GLOBAL;
        }

        /// Whether the array is empty.
        [[nodiscard]] bool empty() const noexcept { return !m_ptr || !m_shape.elements(); }

        /// Returns the rightmost shape of the array.
        [[nodiscard]] const size4_t& shape() const noexcept { return m_shape; }

        /// Returns the rightmost stride of the array.
        [[nodiscard]] const size4_t& stride() const noexcept { return m_stride; }

        /// Whether the dimensions of the array are contiguous.
        [[nodiscard]] bool4_t contiguous() const noexcept {
            return indexing::isContiguous(m_stride, m_shape);
        }

    public: // Accessors
        /// Returns the pointer to the data.
        [[nodiscard]] constexpr T* get() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr T* data() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* data() const noexcept { return m_ptr.get(); }

        /// Returns a reference of the managed resource.
        [[nodiscard]] constexpr std::shared_ptr<T[]> share() const noexcept { return m_ptr; }

        /// Returns a view of the array.
        template<typename I = size_t>
        [[nodiscard]] constexpr View<T, I> view() noexcept { return {get(), m_shape, m_stride}; }

        /// Returns a const view of the array.
        template<typename I = size_t>
        [[nodiscard]] constexpr View<const T, I> view() const noexcept { return {get(), m_shape, m_stride}; }

    public: // Deep copy
        /// Performs a deep copy of the array. The returned array is completely independent from the original one.
        Array copy(bool as_contiguous = false) const {
            return to(m_options, as_contiguous);
        }

        /// Performs a deep copy of the array. The returned array is completely independent from the original one.
        /// \details Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched memory. However, other non-contiguous memory layouts can only be copied
        ///          if the source and destination are both on the same GPU or the CPU.
        /// \param option           Output device and resource to perform the allocation of the new array.
        ///                         The current stream for that device is used to perform the copy.
        /// \param as_contiguous    Whether the output array should be made contiguous.
        Array to(ArrayOption option, bool as_contiguous = false) const;

    public: // Data reinterpretation
        /// Reinterprets the managed array of \p T as an array of U.
        /// \note This is only well defined in cases where reinterpret_cast<U*>(T*) is well defined, for instance,
        ///       when \p U is a unsigned char (to represent any data type as a array of bytes), or to switch between
        ///       complex and real floating-point numbers with the same precision.
        template<typename U>
        Array<U> as() const;

        /// Changes the side (CPU<->GPU) on which the memory should accessed.
        /// \details If the memory resource can be accessed by the CPU and/or a GPU, this function returns an array
        ///          with the device of the given \p type. This is used to control whether PINNED or MANAGED memory
        ///          should be accessed by the CPU or the GPU. MANAGED_GLOBAL memory is not attached to any particular
        ///          GPU, so the current GPU is used in that case.
        Array as(Device::Type type) const;

        /// Reshapes the array.
        /// \param shape Rightmost shape. Must contain the same number of elements as the current shape.
        /// \note Only contiguous arrays are currently supported.
        Array reshape(size4_t shape) const;

        /// Permutes the array.
        /// \param permutation  Rightmost permutation. Axes are numbered from 0 to 3, 3 being the innermost dimension.
        /// \param copy         Whether the permuted array should be copied into a contiguous array, completely
        ///                     independent from the original one.
        template<typename I>
        Array permute(Int4<I> permutation, bool copy = false) const;

    public: // Operators
        // TODO operator=(), operator+(), ..., fancy indexing etc.
        //      do not add unsafe accesses, like loop-like indexing.

    public: // Subregion
        template<typename A,
                 typename B = indexing::full_extent_t,
                 typename C = indexing::full_extent_t,
                 typename D = indexing::full_extent_t,
                 typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B> &&
                                             is_indexable_v<C> && is_indexable_v<D>>>
        constexpr Array subregion(A&& i0, B&& i1 = {}, C&& i2 = {}, D&& i3 = {}) const {
            const indexing::Subregion<int64_t> indexer =
                    indexing::Subregion<int64_t>{long4_t{m_shape}, long4_t{m_stride}}(i0, i1, i2, i3);
            return {std::shared_ptr<T[]>{m_ptr, m_ptr.get() + indexer.offset},
                    size4_t{indexer.shape}, size4_t{indexer.stride}, m_options};
        }

        constexpr Array subregion(indexing::ellipsis_t) const {
            return *this;
        }

        template<typename A, typename = std::enable_if_t<is_indexable_v<A>>>
        constexpr Array subregion(indexing::ellipsis_t, A&& i3) const {
            return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, indexing::full_extent_t{}, i3);
        }

        template<typename A, typename B,
                typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B>>>
        constexpr Array subregion(indexing::ellipsis_t, A&& i2, B&& i3) const {
            return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, i2, i3);
        }

        template<typename A, typename B, typename C,
                typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B> && is_indexable_v<C>>>
        constexpr Array subregion(indexing::ellipsis_t, A&& i1, B&& i2, C&& i3) const {
            return subregion(indexing::full_extent_t{}, i1, i2, i3);
        }

    private:
        void alloc_(bool strided = false);
        static void validate_(void* ptr, ArrayOption option);

    private:
        size4_t m_shape;
        size4_t m_stride;
        std::shared_ptr<T[]> m_ptr;
        ArrayOption m_options;
    };
}

#define NOA_UNIFIED_ARRAY_
#include "noa/unified/Array.inl"
#undef NOA_UNIFIED_ARRAY_
