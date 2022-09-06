#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Indexing.h"
#include "noa/common/Types.h"
#include "noa/common/types/View.h"

#include "noa/cpu/math/Ewise.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Permute.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/memory/Set.h"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Permute.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrDevicePadded.h"
#include "noa/gpu/cuda/memory/PtrManaged.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"
#include "noa/gpu/cuda/memory/Set.h"
#include "noa/gpu/cuda/util/Pointers.h"
#endif

#include "noa/unified/Allocator.h"
#include "noa/unified/ArrayOption.h"
#include "noa/unified/Device.h"
#include "noa/unified/Stream.h"

namespace noa {
    template<typename T>
    class Array;
}

namespace noa::details {
    template<typename T, typename U, typename BinaryOp, typename = std::enable_if_t<traits::is_data_v<U>>>
    void arrayAssign(const Array<T>& lhs, U rhs, BinaryOp binary_op);

    template<typename T, typename U, typename BinaryOp>
    void arrayAssign(const Array<T>& lhs, const Array<U>& rhs, BinaryOp binary_op);
}

namespace noa {
    /// 4-dimensional array of any data type.
    /// \details
    /// - \b Type: Arrays are usually managing "data" types, i.e. integers, floating-points or complex floating-points.
    ///   However, other types are supported, namely, small static vectors (e.g. float4_t) or static matrices
    ///   (e.g. float33_t). Array of such composite types are limited to be simple containers (e.g. arithmetics are
    ///   not supported). The managed type cannot be const-qualified, nor can it be a reference, a pointer or an extent.\n
    /// - \b Resource: Arrays manage a reference-counted resource, which can be shared, moved and copied.
    ///   The resource type and location depends on the ArrayOption used to create the array. Arrays are therefore
    ///   attached to a device, either the CPU or a GPU. Depending on the resource (thus the allocator used to create
    ///   the array), arrays can be interpreted as CPU or GPU and it is possible to create aliases of the same
    ///   resource for the CPU and the GPU (see as() for more details).\n
    /// - \b Accessors: Functions in the unified API taking or returning an Array enqueue backend calls to the current
    ///   stream of the output Array's device. As such, one does not and should not synchronize the stream between each
    ///   operation involving an Array, except if:\n
    ///     - The input and output Array(s) are used in a "unsafe" way between the function call and a synchronization
    ///       point. These unsafe ways of accessing the managed data of an Array are by get()/data()/operator[],
    ///       share() or view(). While these are often required, e.g. for efficient loop-like indexing, one must make
    ///       sure the current stream of the Array's device is synchronized.\n
    ///     - The input and output Array(s) are on the same device but used by different streams. The unified API will
    ///       always use the current stream of the device. If an array is used by a non-current stream (e.g. the current
    ///       stream was changed), one must make sure that stream-ordering is respected by synchronizing this stream
    ///       before calling the function. Note that if the arrays are on different devices, the implementation will
    ///       make sure that stream-ordering is respected.\n
    /// - \b Shape: Shape and strides are in number of elements and specified in the BDHW order (from left to right).
    ///   While column-major ordering is supported, row-major ordering is recommended. Empty dimensions have a size
    ///   of 1. If one dimension is 0, the entire array is considered empty. Arrays can be broadcast to another shape
    ///   and they follow the broadcasting rule (see indexing::broadcast()). As such, some arrays can have dimensions
    ///   with a stride of 0. Negative strides are not supported.
    template<typename T>
    class Array {
    public: // typedefs
        using value_t = T;
        using value_type = T;
        using dim_t = size_t;
        using dim4_t = Int4<dim_t>;
        using ptr_t = T*;
        using ref_t = T&;

        static_assert(!std::is_const_v<T>);
        static_assert(!std::is_pointer_v<T>);
        static_assert(!std::is_reference_v<T>);
        static_assert(traits::is_data_v<T> ||
                      traits::is_intX_v<T> ||
                      traits::is_floatX_v<T> ||
                      traits::is_floatXX_v<T>);

        template<typename U>
        static constexpr bool is_indexable_v =
                std::bool_constant<traits::is_int_v<U> ||
                                   traits::is_almost_same_v<U, indexing::full_extent_t> ||
                                   traits::is_almost_same_v<U, indexing::slice_t>>::value;

    public: // Constructors
        /// Creates an empty array.
        constexpr Array() = default;

        /// Creates a contiguous row vector.
        /// \param elements Number of elements.
        /// \param option   Options of the created array.
        /// \see Allocator for more details.
        constexpr explicit Array(size_t elements, ArrayOption option = {});

        /// Creates a contiguous array.
        /// \param shape    BDHW shape of the array.
        /// \param option   Options of the created array.
        /// \see Allocator for more details.
        constexpr explicit Array(size4_t shape, ArrayOption option = {});

        /// Creates a non-owning contiguous row vector from an existing allocated memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param elements     Number of elements in \p data.
        /// \param option       Options of \p data.
        constexpr Array(T* data, size_t elements, ArrayOption option = {});

        /// Creates a non-owning array from an existing allocated memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param shape        BDHW shape of \p data.
        /// \param strides      BDHW strides of \p data.
        /// \param option       Options of \p data.
        constexpr Array(T* data, size4_t shape, size4_t stride, ArrayOption option = {});

        /// Creates a contiguous row vector from an existing allocated memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param elements     Number of elements in \p data.
        /// \param option       Options of \p data.
        constexpr Array(shared_t<T[]> data, size_t elements, ArrayOption option = {});

        /// Creates an array from an existing allocated memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param shape        BDHW shape of \p data.
        /// \param strides      BDHW strides of \p data.
        /// \param option       Options of \p data.
        constexpr Array(shared_t<T[]> data, size4_t shape, size4_t strides, ArrayOption option = {});

    public: // Getters
        /// Returns the options used to create the array.
        [[nodiscard]] constexpr ArrayOption options() const noexcept;

        /// Returns the device used to create the array.
        [[nodiscard]] constexpr Device device() const noexcept;

        /// Returns the memory resource used to create the array.
        [[nodiscard]] constexpr Allocator allocator() const noexcept;

        /// Whether the managed data can be accessed by CPU threads.
        /// \note While it indicates whether the managed data can be dereferenced on the CPU, it does not indicate
        ///       if the Array will be seen as a CPU (or a GPU) array. This choice is purely made on the Array's
        ///       device type. For instance, pinned memory can be dereferenced by the CPU, so this function will
        ///       returned true, but if the Array's device is a GPU, the implementations will refer to this Array
        ///       as a GPU array and will therefore prioritizing GPU access.
        [[nodiscard]] constexpr bool dereferenceable() const noexcept;

        /// Whether the array is empty.
        [[nodiscard]] bool empty() const noexcept;

        /// Returns the BDHW shape of the array.
        [[nodiscard]] const size4_t& shape() const noexcept;

        /// Returns the BDHW strides of the array.
        [[nodiscard]] const size4_t& strides() const noexcept;

        /// Returns the number of elements in the array.
        [[nodiscard]] size_t elements() const noexcept;
        [[nodiscard]] size_t size() const noexcept;

        /// Whether the dimensions of the array are C or F contiguous.
        template<char ORDER = 'C'>
        [[nodiscard]] bool contiguous() const noexcept;

    public: // Accessors
        /// Returns the pointer to the data.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may create a data race.
        [[nodiscard]] constexpr T* get() const noexcept;

        /// Returns the pointer to the data.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may create a data race.
        [[nodiscard]] constexpr T* data() const noexcept;

        /// Returns a reference of the element at the specified memory \p offset.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this reference may create a data race.
        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] constexpr T& operator[](I offset) const noexcept;

        /// Returns a reference of the managed resource.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may create a data race.
        [[nodiscard]] constexpr const std::shared_ptr<T[]>& share() const noexcept;

        /// Returns a const view of the array.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this view may create a data race.
        template<typename I = size_t>
        [[nodiscard]] constexpr View<T, I> view() const noexcept;

        /// Synchronizes the current stream of the Array's device.
        /// \details It guarantees safe access to the Array's managed memory using get(), data(),
        ///          operator[], share() or view(). Stream-ordered access is safe and doesn't need
        ///          synchronization.
        const Array& eval() const;

        /// Releases the array. *this is left empty.
        Array release() noexcept;

    public: // Deep copy
        /// Performs a deep copy of the array to \p output.
        /// \details Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts and colum or row vectors. However, other non-contiguous memory
        ///          layouts can only be copied if the source and destination are both on the same GPU or on the CPU.
        /// \param[out] output  Destination. It should not overlap with *this.
        void to(const Array& output) const;

        /// Performs a deep copy of the array according \p option.
        /// \details The returned array is completely independent from the original one and is C-contiguous.
        ///          Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts and colum or row vectors. However, other non-contiguous memory
        ///          layouts can only be copied if the source and destination are both on the same GPU or on the CPU.
        /// \param option   Output device and resource to perform the allocation of the new array.
        ///                 The current stream for that device is used to perform the copy.
        Array to(ArrayOption option) const;

        /// Performs a deep copy of the array preserving the array's options.
        Array copy() const;

    public: // Data reinterpretation
        /// Reinterprets the managed array of \p T as an array of \p U.
        /// \note This is only well defined in cases where reinterpret_cast<U*>(T*) is well defined, for instance,
        ///       when \p U is a unsigned char (to represent any data type as a array of bytes), or to switch between
        ///       complex and real floating-point numbers with the same precision.
        template<typename U>
        Array<U> as() const;

        /// Changes the side (CPU<->GPU) on which the memory should be accessed.
        /// \details If the memory resource can be accessed by the CPU and/or a GPU, this function returns an array
        ///          with the device of the given \p type. This is used to control whether PINNED or MANAGED memory
        ///          should be accessed by the CPU or the GPU. MANAGED_GLOBAL memory is not attached to any particular
        ///          GPU, so the current GPU is used in that case.
        Array as(Device::Type type) const;

        /// Reshapes the array.
        /// \details This function performs a "safe" reshape by making sure the new shape contains the same number
        ///          of elements. If one wants to assign an array to an arbitrary new shape and new strides, one
        ///          can use the alias Array constructor instead.
        /// \param shape    New shape. Must contain the same number of elements as the current shape.
        /// \return An alias of the array with the new shape and strides.
        Array reshape(size4_t shape) const;

        /// Reshapes the array in a vector along a particular axis.
        /// Returns a row vector by default.
        Array flat(int axis = 3) const;

        /// Permutes the array.
        /// \param permutation  Permutation with the axes numbered from 0 to 3.
        /// \param copy         Whether the permuted array should be copied into a C-contiguous array, completely
        ///                     independent from the original one.
        /// \return The permuted array. If \p copy is false, this new array is an alias from the original array.
        Array permute(uint4_t permutation, bool copy = false) const;

    public: // Assignment operators
        /// Clears the array. Equivalent to assigning *this with an empty array.
        Array& operator=(std::nullptr_t);

        /// Fills the array with \p value.
        Array& operator=(T value);

        /// Add \p value to the array.
        Array& operator+=(T value);

        /// Subtract \p value to the array.
        Array& operator-=(T value);

        /// Multiplies the array by \p value.
        Array& operator*=(T value);

        /// Divide the array by \p value.
        Array& operator/=(T value);

        /// Element-wise addition of \p array.
        template<typename U>
        Array& operator+=(const Array<U>& array);

        /// Element-wise subtraction of \p array.
        template<typename U>
        Array& operator-=(const Array<U>& array);

        /// Element-wise multiplication of \p array.
        template<typename U>
        Array& operator*=(const Array<U>& array);

        /// Element-wise division by \p array.
        template<typename U>
        Array& operator/=(const Array<U>& array);

        template<typename U, typename = std::enable_if_t<traits::is_complex_v<T> && traits::is_scalar_v<U>>>
        Array& operator+=(U value);

        template<typename U, typename = std::enable_if_t<traits::is_complex_v<T> && traits::is_scalar_v<U>>>
        Array& operator-=(U value);

        template<typename U, typename = std::enable_if_t<traits::is_complex_v<T> && traits::is_scalar_v<U>>>
        Array& operator*=(U value);

        template<typename U, typename = std::enable_if_t<traits::is_complex_v<T> && traits::is_scalar_v<U>>>
        Array& operator/=(U value);

    public: // Subregion
        template<typename A,
                 typename B = indexing::full_extent_t,
                 typename C = indexing::full_extent_t,
                 typename D = indexing::full_extent_t,
                 typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B> &&
                                             is_indexable_v<C> && is_indexable_v<D>>>
        constexpr Array subregion(A&& i0, B&& i1 = {}, C&& i2 = {}, D&& i3 = {}) const;

        constexpr Array subregion(indexing::ellipsis_t) const;

        template<typename A, typename = std::enable_if_t<is_indexable_v<A>>>
        constexpr Array subregion(indexing::ellipsis_t, A&& i3) const;

        template<typename A, typename B,
                 typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B>>>
        constexpr Array subregion(indexing::ellipsis_t, A&& i2, B&& i3) const;

        template<typename A, typename B, typename C,
                 typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B> && is_indexable_v<C>>>
        constexpr Array subregion(indexing::ellipsis_t, A&& i1, B&& i2, C&& i3) const;

    private:
        void alloc_();
        static void validate_(void* ptr, ArrayOption option);

    private:
        size4_t m_shape;
        size4_t m_strides;
        std::shared_ptr<T[]> m_ptr;
        ArrayOption m_options;
    };
}

namespace noa::indexing {
    /// Broadcasts an array to a given shape.
    template<typename T>
    Array<T> broadcast(const Array<T>& input, size4_t shape);
}

#define NOA_UNIFIED_ARRAY_
#include "noa/unified/Array.inl"
#undef NOA_UNIFIED_ARRAY_
