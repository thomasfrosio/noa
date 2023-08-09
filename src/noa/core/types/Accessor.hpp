#pragma once

#include <cstdint>
#include <cstddef>

#include "noa/core/Assert.hpp"
#include "noa/core/Definitions.hpp"
#include "noa/core/traits/Numerics.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/utils/Indexing.hpp"

// The Accessor and AccessorReference are used bind a pointer (pointing to a memory region) and
// the strides of this memory region, i.e. should step through this memory region. This is mostly
// intended for the backends to use, so these aim for overall good performance. As such:
//  1) The size of the dimensions are not stored, so it cannot bound-check the indexes against
//     their dimension size. In a lot of cases, the input/output arrays have the same size/shape,
//     and, the size/shape is often not needed by the compute kernel (e.g. ewise/iwise).
//     Either way, it would require storing unused information.
//  2) Pointer traits. By default, the pointers are not marked with any attributes, but the "restrict"
//     traits can be added. This is useful to trigger optimizations ultimately leading to less memory
//     transfers, better use of registers, and even automatic-vectorization on the CPU.
//  3) Strides are fully dynamic (one dynamic stride per dimension) by default, but the rightmost
//     dimension can be marked contiguous. The Accessor use the rightmost convention, so already
//     assumes that the layout is ordered and that the innermost dimension is the rightmost dimension.
//     F-contiguous layouts are not supported because the backends is supposed to reevaluate
//     the layout of the arrays at runtime and reorder dimensions to the rightmost order before
//     creating the accessors, thereby transforming F-contiguous arrays to C-contiguous arrays.
//     Similarly, if the layout is fully contiguous and the indexes are not needed (e.g. ewise),
//     the backend can transform the layout to a 1D contiguous accessor, which only stores a pointer
//     and uses a fixed stride of 1.
//     One disadvantage of the contiguous case is that the API becomes a bit more "rough" because
//     the innermost stride is not stored. Thus, one should be pay attention to this when accessing
//     the strides, specially with AccessorReference. In practice, this rarely (never?) happens.

namespace noa {
    enum class PointerTraits { DEFAULT, RESTRICT }; // TODO ATOMIC?
    enum class StridesTraits { STRIDED, CONTIGUOUS }; // TODO EMPTY?

    // Empty type used to emulate the stride of a contiguous dimension.
    template<typename I>
    struct AccessorContiguousStride {
        template<typename... Ts>
        NOA_HD constexpr /* implicit */ AccessorContiguousStride(Ts&&...) noexcept {}

        NOA_HD constexpr I* data() const noexcept { return nullptr; }

        template<typename T>
        [[nodiscard]] NOA_HD constexpr I operator[](T i) const noexcept {
            NOA_ASSERT(i == 0);
            (void) i;
            return I{1};
        }
    };
}

namespace noa::details {
    template<typename InputValue, typename OutputValue>
    constexpr bool is_mutable_value_type_v =
            std::is_const_v<OutputValue> &&
            std::is_same_v<InputValue, std::remove_const_t<OutputValue>>;
}

namespace noa {
    template<typename T, size_t N, typename I,
            PointerTraits PointerTrait,
            StridesTraits StridesTrait>
    class AccessorReference;

    // Multidimensional accessor using C-style multidimensional array indexing.
    template<typename T, size_t N, typename I,
             PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED>
    class Accessor {
    public:
        static_assert(!std::is_pointer_v<T>);
        static_assert(!std::is_reference_v<T>);
        static_assert(std::is_integral_v<I>);
        static_assert(N > 0 && N <= 4);

        static constexpr bool IS_RESTRICT = PointerTrait == PointerTraits::RESTRICT;
        static constexpr bool IS_CONTIGUOUS = StridesTrait == StridesTraits::CONTIGUOUS;

        #if defined(__CUDACC__)
        using pointer_type = std::conditional_t<IS_RESTRICT, T* __restrict__, T*>;
        #else
        using pointer_type = std::conditional_t<IS_RESTRICT, T* __restrict, T*>;
        #endif

        using value_type = T;
        using mutable_value_type = T;
        using index_type = I;
        static constexpr index_type COUNT = N;
        static constexpr size_t SIZE = N;
        static constexpr int64_t SSIZE = N;

        using strided_strides_type = Strides<index_type, N>;
        using contiguous_strides_type = Strides<index_type, noa::math::max(size_t{1}, N - 1)>;
        using contiguous_stride_type = AccessorContiguousStride<index_type>;
        using strides_type =
                std::conditional_t<!IS_CONTIGUOUS, strided_strides_type,
                std::conditional_t<(N > 1), contiguous_strides_type, contiguous_stride_type>>;
        using accessor_reference_type = AccessorReference<value_type, SIZE, index_type, PointerTrait, StridesTrait>;

    public: // Constructors
        NOA_HD constexpr Accessor() = default;

        // Creates a strided or contiguous accessor.
        NOA_HD constexpr Accessor(pointer_type pointer, const Strides<index_type, SIZE>& strides) noexcept
                : m_ptr(pointer), m_strides(strides.data()) {}

        // Creates an accessor from an accessor reference.
        NOA_HD constexpr explicit Accessor(accessor_reference_type accessor_reference) noexcept
                : m_ptr(accessor_reference.get()), m_strides(accessor_reference.strides()) {}

        // Creates a contiguous 1D accessor, assuming the stride is 1.
        template<typename Void = void, typename = std::enable_if_t<(SIZE == 1) && IS_CONTIGUOUS && std::is_void_v<Void>>>
        NOA_HD constexpr explicit Accessor(pointer_type pointer) noexcept
                : m_ptr(pointer), m_strides(nullptr) {}

        // Creates a const accessor from an existing non-const accessor.
        template<typename U,
                 typename = std::enable_if_t<details::is_mutable_value_type_v<U, value_type>>>
        NOA_HD constexpr /* implicit */ Accessor(const Accessor<U, N, I, PointerTrait, StridesTrait>& accessor)
                : m_ptr(accessor.get()), m_strides(accessor.strides().data()) {}

    public: // Accessing strides
        template<size_t INDEX>
        [[nodiscard]] NOA_HD constexpr auto stride() const noexcept {
            static_assert(INDEX < N);
            if constexpr (IS_CONTIGUOUS && INDEX == SIZE - 1)
                return index_type{1};
            return m_strides[INDEX];
        }

        [[nodiscard]] NOA_HD constexpr strides_type& strides() noexcept { return m_strides; }
        [[nodiscard]] NOA_HD constexpr const strides_type& strides() const noexcept { return m_strides; }

    public:
        [[nodiscard]] NOA_HD constexpr pointer_type get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr pointer_type data() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return !is_empty(); }

        [[nodiscard]] NOA_HD constexpr accessor_reference_type to_accessor_reference() const noexcept {
            return accessor_reference_type(*this);
        }

        template<typename Int0, typename Int1,
                 typename = std::enable_if_t<
                         StridesTraits::STRIDED == StridesTrait &&
                         nt::are_int_v<Int0, Int1>>>
        [[nodiscard]] NOA_HD constexpr Accessor swap_dimensions(Int0 d0, Int1 d1) const noexcept {
            Accessor out = *this;
            noa::details::swap(out.strides()[d0], out.strides()[d1]);
            return out;
        }

    public:
        // Offsets the current pointer at dimension 0 (without changing the dimensionality of the accessor)
        // and return a reference of that new accessor.
        template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
        [[nodiscard]] NOA_HD auto offset_accessor(Int index) const noexcept {
            NOA_ASSERT(!is_empty());
            return accessor_reference_type(m_ptr + indexing::at(index, stride<0>()), strides().data());
        }

        // C-style indexing operator, on 1D accessor. 1D -> value_type&
        template<typename Int, std::enable_if_t<SIZE == 1 && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD value_type& operator[](Int index) const noexcept {
            NOA_ASSERT(!is_empty());
            return m_ptr[indexing::at(index, stride<0>())];
        }

        // C-style indexing operator, on multidimensional accessor. ND -> ND-1
        template<typename Int, std::enable_if_t<(SIZE > 1) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD auto operator[](Int index) const noexcept {
            NOA_ASSERT(!is_empty());
            using output_type = AccessorReference<value_type, (SIZE - 1), index_type, PointerTrait, StridesTrait>;
            return output_type(m_ptr + indexing::at(index, stride<0>()), strides().data() + 1);
        }

    public:
        template<typename Pointer, typename Int0,
                 std::enable_if_t<std::is_integral_v<Int0>, bool> = true>
        [[nodiscard]] NOA_HD auto
        offset_pointer(Pointer pointer, Int0 i0) const noexcept {
            pointer += indexing::at(i0, stride<0>());
            return pointer;
        }

        template<typename Pointer, typename Int0, typename Int1,
                 std::enable_if_t<(SIZE >= 2) && nt::are_int_v<Int0, Int1>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto
        offset_pointer(Pointer pointer, Int0 i0, Int1 i1) const noexcept {
            pointer += indexing::at(i0, stride<0>());
            pointer += indexing::at(i1, stride<1>());
            return pointer;
        }

        template<typename Pointer, typename Int0, typename Int1, typename Int2,
                 std::enable_if_t<(SIZE >= 3) && nt::are_int_v<Int0, Int1, Int2>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto
        offset_pointer(Pointer pointer, Int0 i0, Int1 i1, Int2 i2) const noexcept {
            pointer += indexing::at(i0, stride<0>());
            pointer += indexing::at(i1, stride<1>());
            pointer += indexing::at(i2, stride<2>());
            return pointer;
        }

        template<typename Pointer, typename Int0, typename Int1, typename Int2, typename Int3,
                 std::enable_if_t<SIZE == 4 && nt::are_int_v<Int0, Int1, Int2, Int3>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto
        offset_pointer(Pointer pointer, Int0 i0, Int1 i1, Int2 i2, Int3 i3) const noexcept {
            pointer += indexing::at(i0, stride<0>());
            pointer += indexing::at(i1, stride<1>());
            pointer += indexing::at(i2, stride<2>());
            pointer += indexing::at(i3, stride<3>());
            return pointer;
        }

        template<typename Int0,
                 std::enable_if_t<(SIZE >= 1) && std::is_integral_v<Int0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i0);
        }

        template<typename Int0, typename Int1,
                 std::enable_if_t<(SIZE >= 2) && nt::are_int_v<Int0, Int1>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0, Int1 i1) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i0, i1);
        }

        template<typename Int0, typename Int1, typename Int2,
                 std::enable_if_t<(SIZE >= 3) && nt::are_int_v<Int0, Int1, Int2>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0, Int1 i1, Int2 i2) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i0, i1, i2);
        }

        template<typename Int0, typename Int1, typename Int2, typename Int3,
                 std::enable_if_t<SIZE == 4 && nt::are_int_v<Int0, Int1, Int2, Int3>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0, Int1 i1, Int2 i2, Int3 i3) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i0, i1, i2, i3);
        }

        template<typename Int, std::enable_if_t<(SIZE >= 1) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Vec1<Int>& i0) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i0[0]);
        }

        template<typename Int, std::enable_if_t<(SIZE >= 2) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Vec2<Int>& i01) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i01[0], i01[1]);
        }

        template<typename Int, std::enable_if_t<(SIZE >= 3) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Vec3<Int>& i012) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i012[0], i012[1], i012[2]);
        }

        template<typename Int, std::enable_if_t<(SIZE >= 4) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Vec4<Int>& i0123) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i0123[0], i0123[1], i0123[2], i0123[3]);
        }

    private:
        pointer_type m_ptr{};
        NOA_NO_UNIQUE_ADDRESS strides_type m_strides{};
    };

    // Multidimensional accessor.
    // This is similar to Accessor, except that this type does not copy the strides,
    // it simply points to existing ones. Usually "AccessorReference" is not created
    // by the user, but instead by an Accessor during a ND C-style indexing.
    template<typename T, size_t N, typename I,
             PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED>
    class AccessorReference {
    public:
        static constexpr bool IS_RESTRICT = PointerTrait == PointerTraits::RESTRICT;
        static constexpr bool IS_CONTIGUOUS = StridesTrait == StridesTraits::CONTIGUOUS;

        using accessor_type = Accessor<T, N, I, PointerTrait, StridesTrait>;
        using value_type = typename accessor_type::value_type;
        using index_type = typename accessor_type::index_type;
        using pointer_type = typename accessor_type::pointer_type;
        using strides_type = std::conditional_t<
                StridesTrait == StridesTraits::CONTIGUOUS && N == 1,
                AccessorContiguousStride<index_type>, const index_type*>;

        static constexpr index_type COUNT = N;
        static constexpr size_t SIZE = N;
        static constexpr int64_t SSIZE = N;

    public:
        // Creates an empty view.
        NOA_HD constexpr AccessorReference() = default;

        // Creates a strided accessor.
        // For the contiguous case, the rightmost stride is ignored and never read from
        // the input strides pointer (so N-1 elements are read from "strides").
        // Therefore, in the 1D contiguous case, a nullptr can be passed.
        NOA_HD constexpr AccessorReference(pointer_type pointer, strides_type strides) noexcept
                : m_ptr(pointer),
                  m_strides(strides) {}

        NOA_HD constexpr explicit AccessorReference(accessor_type accessor) noexcept
                : AccessorReference(accessor.ptr, accessor.strides().data()) {}

        // Creates a const accessor from an existing non-const accessor.
        template<typename U,
                 typename = std::enable_if_t<details::is_mutable_value_type_v<U, value_type>>>
        NOA_HD constexpr /* implicit */ AccessorReference(
                const AccessorReference<U, N, I, PointerTrait, StridesTrait>& accessor)
                : m_ptr(accessor.get()), m_strides(accessor.strides().data()) {}

    public: // Accessing strides
        template<size_t INDEX>
        [[nodiscard]] NOA_HD constexpr auto stride() const noexcept {
            static_assert(INDEX < N);
            if constexpr (IS_CONTIGUOUS && INDEX == SIZE - 1)
                return index_type{1};
            return m_strides[INDEX];
        }

        [[nodiscard]] NOA_HD constexpr strides_type strides() noexcept { return m_strides; }

    public:
        [[nodiscard]] NOA_HD constexpr pointer_type get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr pointer_type data() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return !is_empty(); }

        [[nodiscard]] NOA_HD constexpr accessor_type to_accessor() const noexcept {
            return accessor_type(*this);
        }

    public:
        template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
        [[nodiscard]] NOA_HD auto offset_accessor(Int index) const noexcept {
            NOA_ASSERT(!is_empty());
            return AccessorReference(m_ptr + noa::indexing::at(index, stride<0>()), m_strides);
        }

        // Indexing operator, on 1D accessor. 1D -> ref
        template<typename Int, std::enable_if_t<SIZE == 1 && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD value_type& operator[](Int index) const noexcept {
            NOA_ASSERT(!is_empty());
            return m_ptr[noa::indexing::at(index, stride<0>())];
        }

        // Indexing operator, multidimensional accessor. ND -> ND-1
        template<typename Int, std::enable_if_t<(SIZE > 1) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD auto operator[](Int index) const noexcept {
            NOA_ASSERT(!is_empty());
            using output_type = AccessorReference<value_type, SIZE - 1, index_type, PointerTrait, StridesTrait>;
            return output_type(m_ptr + noa::indexing::at(index, stride<0>()), m_strides + 1);
        }

    public:
        template<typename Pointer, typename Int0,
                 std::enable_if_t<std::is_integral_v<Int0>, bool> = true>
        [[nodiscard]] NOA_HD auto
        offset_pointer(Pointer pointer, Int0 i0) const noexcept {
            pointer += indexing::at(i0, stride<0>());
            return pointer;
        }

        template<typename Pointer, typename Int0, typename Int1,
                 std::enable_if_t<(SIZE >= 2) && nt::are_int_v<Int0, Int1>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto
        offset_pointer(Pointer pointer, Int0 i0, Int1 i1) const noexcept {
            pointer += indexing::at(i0, stride<0>());
            pointer += indexing::at(i1, stride<1>());
            return pointer;
        }

        template<typename Pointer, typename Int0, typename Int1, typename Int2,
                 std::enable_if_t<(SIZE >= 3) && nt::are_int_v<Int0, Int1, Int2>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto
        offset_pointer(Pointer pointer, Int0 i0, Int1 i1, Int2 i2) const noexcept {
            pointer += indexing::at(i0, stride<0>());
            pointer += indexing::at(i1, stride<1>());
            pointer += indexing::at(i2, stride<2>());
            return pointer;
        }

        template<typename Pointer, typename Int0, typename Int1, typename Int2, typename Int3,
                 std::enable_if_t<SIZE == 4 && nt::are_int_v<Int0, Int1, Int2, Int3>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto
        offset_pointer(Pointer pointer, Int0 i0, Int1 i1, Int2 i2, Int3 i3) const noexcept {
            pointer += indexing::at(i0, stride<0>());
            pointer += indexing::at(i1, stride<1>());
            pointer += indexing::at(i2, stride<2>());
            pointer += indexing::at(i3, stride<3>());
            return pointer;
        }

        template<typename Int0,
                 std::enable_if_t<(SIZE >= 1) && std::is_integral_v<Int0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i0);
        }

        template<typename Int0, typename Int1,
                 std::enable_if_t<(SIZE >= 2) && nt::are_int_v<Int0, Int1>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0, Int1 i1) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i0, i1);
        }

        template<typename Int0, typename Int1, typename Int2,
                 std::enable_if_t<(SIZE >= 3) && nt::are_int_v<Int0, Int1, Int2>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0, Int1 i1, Int2 i2) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i0, i1, i2);
        }

        template<typename Int0, typename Int1, typename Int2, typename Int3,
                 std::enable_if_t<SIZE == 4 && nt::are_int_v<Int0, Int1, Int2, Int3>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0, Int1 i1, Int2 i2, Int3 i3) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i0, i1, i2, i3);
        }

        template<typename Int, std::enable_if_t<(SIZE >= 1) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Vec1<Int>& i0) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i0[0]);
        }

        template<typename Int, std::enable_if_t<(SIZE >= 2) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Vec2<Int>& i01) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i01[0], i01[1]);
        }

        template<typename Int, std::enable_if_t<(SIZE >= 3) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Vec3<Int>& i012) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i012[0], i012[1], i012[2]);
        }

        template<typename Int, std::enable_if_t<(SIZE >= 4) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Vec4<Int>& i0123) const noexcept {
            NOA_ASSERT(!is_empty());
            return *offset_pointer(m_ptr, i0123[0], i0123[1], i0123[2], i0123[3]);
        }

    private:
        pointer_type m_ptr{};
        NOA_NO_UNIQUE_ADDRESS strides_type m_strides{};
    };

    template<typename T, size_t N>
    using AccessorI64 = Accessor<T, N, int64_t>;
    template<typename T, size_t N>
    using AccessorI32 = Accessor<T, N, int32_t>;
    template<typename T, size_t N>
    using AccessorU64 = Accessor<T, N, uint64_t>;
    template<typename T, size_t N>
    using AccessorU32 = Accessor<T, N, uint32_t>;

    template<typename T, size_t N, typename I, StridesTraits StridesTrait = StridesTraits::STRIDED>
    using AccessorRestrict = Accessor<T, N, I, PointerTraits::RESTRICT, StridesTrait>;
    template<typename T, size_t N, StridesTraits StridesTrait = StridesTraits::STRIDED>
    using AccessorRestrictI64 = AccessorRestrict<T, N, int64_t, StridesTrait>;
    template<typename T, size_t N, StridesTraits StridesTrait = StridesTraits::STRIDED>
    using AccessorRestrictI32 = AccessorRestrict<T, N, int32_t, StridesTrait>;
    template<typename T, size_t N, StridesTraits StridesTrait = StridesTraits::STRIDED>
    using AccessorRestrictU64 = AccessorRestrict<T, N, uint64_t, StridesTrait>;
    template<typename T, size_t N, StridesTraits StridesTrait = StridesTraits::STRIDED>
    using AccessorRestrictU32 = AccessorRestrict<T, N, uint32_t, StridesTrait>;

    template<typename T, size_t N, typename I, PointerTraits PointerTrait = PointerTraits::DEFAULT>
    using AccessorContiguous = Accessor<T, N, I, PointerTrait, StridesTraits::CONTIGUOUS>;
    template<typename T, size_t N, PointerTraits PointerTrait = PointerTraits::DEFAULT>
    using AccessorContiguousI64 = AccessorContiguous<T, N, int64_t, PointerTrait>;
    template<typename T, size_t N, PointerTraits PointerTrait = PointerTraits::DEFAULT>
    using AccessorContiguousI32 = AccessorContiguous<T, N, int32_t, PointerTrait>;
    template<typename T, size_t N, PointerTraits PointerTrait = PointerTraits::DEFAULT>
    using AccessorContiguousU64 = AccessorContiguous<T, N, uint64_t, PointerTrait>;
    template<typename T, size_t N, PointerTraits PointerTrait = PointerTraits::DEFAULT>
    using AccessorContiguousU32 = AccessorContiguous<T, N, uint32_t, PointerTrait>;

    template<typename T, size_t N, typename I>
    using AccessorRestrictContiguous = Accessor<T, N, I, PointerTraits::RESTRICT, StridesTraits::CONTIGUOUS>;
    template<typename T, size_t N>
    using AccessorRestrictContiguousI64 = AccessorRestrictContiguous<T, N, int64_t>;
    template<typename T, size_t N>
    using AccessorRestrictContiguousI32 = AccessorRestrictContiguous<T, N, int32_t>;
    template<typename T, size_t N>
    using AccessorRestrictContiguousU64 = AccessorRestrictContiguous<T, N, uint64_t>;
    template<typename T, size_t N>
    using AccessorRestrictContiguousU32 = AccessorRestrictContiguous<T, N, uint32_t>;
}
