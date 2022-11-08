#pragma once

#include <cstdint>
#include <cstddef>

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/SafeCast.h"
#include "noa/common/Indexing.h"
#include "noa/common/types/Int2.h"
#include "noa/common/types/Int3.h"
#include "noa/common/types/Int4.h"

namespace noa {
    enum class AccessorTraits { DEFAULT, RESTRICT };

    template<typename Value, int NDIM, typename Index, AccessorTraits TRAITS>
    class AccessorReference;

    /// Multidimensional accessor using C-style multidimensional array indexing.
    /// \details This class in meant to provide multidimensional-indexing on contiguous/strided data.
    ///          It only keeps track of a pointer and one stride for each dimension. As opposed, to View,
    ///          it cannot bound check the index against the dimension size.
    /// \note The only reason the Accessor doesn't keep track of the dimension size is that in a lot of
    ///       cases, the input/output arrays have the same size/shape, so kernels would have to transfer
    ///       redundant information to the device, which could cause a performance hit for CUDA kernels.
    ///       If this is not an issue, prefer to use View.
    template<typename Value, int NDIM, typename Index, AccessorTraits TRAITS = AccessorTraits::DEFAULT>
    class Accessor {
    public:
        static_assert(!std::is_pointer_v<Value>);
        static_assert(!std::is_reference_v<Value>);
        static_assert(std::is_integral_v<Index>);
        static_assert(NDIM > 0 && NDIM <= 4);

        #if defined(__CUDACC__)
        using pointer_type = std::conditional_t<TRAITS == AccessorTraits::RESTRICT, Value* __restrict__, Value*>;
        #else
        using pointer_type = std::conditional_t<TRAITS == AccessorTraits::RESTRICT, Value* __restrict, Value*>;
        #endif

        using value_type = Value;
        using index_type = Index;
        static constexpr index_type COUNT = NDIM;

    public:
        /// Creates an empty view.
        NOA_HD constexpr Accessor() = default;

        /// Creates a strided accessor.
        template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
        NOA_HD constexpr Accessor(pointer_type pointer, const Int* strides) noexcept : m_ptr(pointer) {
            NOA_ASSERT(strides != nullptr);
            for (index_type i = 0; i < COUNT; ++i) {
                NOA_ASSERT(isSafeCast<index_type>(strides[i]));
                m_strides[i] = static_cast<index_type>(strides[i]);
            }
        }

        template<typename Int, typename = std::enable_if_t<COUNT == 1 && traits::is_int_v<Int>>>
        NOA_HD constexpr Accessor(pointer_type pointer, Int stride) noexcept
                : Accessor(pointer, &stride) {}

        template<typename Int,
                 typename = std::enable_if_t<(COUNT == 2 && traits::is_int2_v<Int>) ||
                                             (COUNT == 3 && traits::is_int3_v<Int>) ||
                                             (COUNT == 4 && traits::is_int4_v<Int>)>>
        NOA_HD constexpr Accessor(pointer_type pointer, const Int& strides) noexcept
                : Accessor(pointer, strides.get()) {}

        /// Creates a const accessor from an existing non-const accessor.
        template<typename MutableValue,
                 typename = std::enable_if_t<std::is_const_v<value_type> &&
                                             std::is_same_v<MutableValue, std::remove_const_t<value_type>>>>
        NOA_HD constexpr /* implicit */ Accessor(const Accessor<MutableValue, COUNT, index_type, TRAITS>& accessor)
                : Accessor(accessor.data(), accessor.strides()) {}

    public:
        template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
        [[nodiscard]] NOA_HD constexpr index_type& stride(Int dim) noexcept {
            NOA_ASSERT(static_cast<index_type>(dim) < COUNT);
            return m_strides[dim];
        }

        template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
        [[nodiscard]] NOA_HD constexpr const index_type& stride(Int dim) const noexcept {
            NOA_ASSERT(static_cast<index_type>(dim) < COUNT);
            return m_strides[dim];
        }

        [[nodiscard]] NOA_HD constexpr index_type* strides() noexcept { return m_strides; }
        [[nodiscard]] NOA_HD constexpr const index_type* strides() const noexcept { return m_strides; }
        [[nodiscard]] NOA_HD constexpr pointer_type data() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr pointer_type get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr bool empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return !empty(); }

        template<typename Int0, typename Int1,
                 typename = std::enable_if_t<std::is_integral_v<Int0> && std::is_integral_v<Int1>>>
        [[nodiscard]] NOA_HD constexpr Accessor swap(Int0 d0, Int1 d1) const noexcept {
            NOA_ASSERT(static_cast<index_type>(d0) < COUNT && static_cast<index_type>(d1) < COUNT);
            Accessor out = *this;
            const auto tmp = out.stride(d0); // swap
            out.stride(d0) = out.stride(d1);
            out.stride(d1) = tmp;
            return out;
        }

    public:
        /// Offsets the current pointer at dimension 0 (without changing the dimensionality of the accessor)
        /// and return a reference of that new accessor.
        template<typename Int, typename std::enable_if_t<std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD auto offset(Int index) const noexcept {
            NOA_ASSERT((traits::is_uint_v<Int> || index >= Int{0}) && isSafeCast<index_type>(index));
            using output_type = AccessorReference<value_type, COUNT, index_type, TRAITS>;
            return output_type(m_ptr + m_strides[0] * static_cast<index_type>(index), strides());
        }

        /// Indexing operator, on 1D accessor. 1D -> ref
        template<typename Int, typename std::enable_if_t<COUNT == 1 && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD value_type& operator[](Int index) const noexcept {
            NOA_ASSERT(m_ptr != nullptr &&
                       (traits::is_uint_v<Int> || index >= Int{0}) &&
                       isSafeCast<index_type>(index));
            return m_ptr[m_strides[0] * static_cast<index_type>(index)];
        }

        /// Indexing operator, multidimensional accessor. ND -> ND-1
        template<typename Int, typename std::enable_if_t<(COUNT > 1) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD auto operator[](Int index) const noexcept {
            NOA_ASSERT(m_ptr != nullptr &&
                       (traits::is_uint_v<Int> || index >= Int{0}) &&
                       isSafeCast<index_type>(index));
            // While returning an Accessor adds no overhead on the CPU, on the GPU it does.
            // Returning an AccessorReference removes a few instructions.
            using output_type = AccessorReference<value_type, COUNT - 1, index_type, TRAITS>;
            return output_type(m_ptr + stride(0) * static_cast<index_type>(index), m_strides + 1);
        }

    public:
        template<typename Pointer, typename Int0,
                 typename std::enable_if_t<std::is_integral_v<Int0>, bool> = true>
        [[nodiscard]] NOA_HD auto
        offsetPointer(Pointer pointer, Int0 i0) const noexcept {
            pointer += indexing::at(i0, m_strides[0]);
            return pointer;
        }

        template<typename Pointer, typename Int0, typename Int1,
                 typename std::enable_if_t<(COUNT >= 2) && traits::are_int_v<Int0, Int1>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto
        offsetPointer(Pointer pointer, Int0 i0, Int1 i1) const noexcept {
            pointer += indexing::at(i0, m_strides[0]);
            pointer += indexing::at(i1, m_strides[1]);
            return pointer;
        }

        template<typename Pointer, typename Int0, typename Int1, typename Int2,
                 typename std::enable_if_t<(COUNT >= 3) && traits::are_int_v<Int0, Int1, Int2>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto
        offsetPointer(Pointer pointer, Int0 i0, Int1 i1, Int2 i2) const noexcept {
            pointer += indexing::at(i0, m_strides[0]);
            pointer += indexing::at(i1, m_strides[1]);
            pointer += indexing::at(i2, m_strides[2]);
            return pointer;
        }

        template<typename Pointer, typename Int0, typename Int1, typename Int2, typename Int3,
                 typename std::enable_if_t<COUNT == 4 && traits::are_int_v<Int0, Int1, Int2, Int3>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto
        offsetPointer(Pointer pointer, Int0 i0, Int1 i1, Int2 i2, Int3 i3) const noexcept {
            pointer += indexing::at(i0, m_strides[0]);
            pointer += indexing::at(i1, m_strides[1]);
            pointer += indexing::at(i2, m_strides[2]);
            pointer += indexing::at(i3, m_strides[3]);
            return pointer;
        }

        template<typename Int0,
                 typename std::enable_if_t<(COUNT >= 1) && std::is_integral_v<Int0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i0);
        }

        template<typename Int0, typename Int1,
                 typename std::enable_if_t<(COUNT >= 2) && traits::are_int_v<Int0, Int1>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0, Int1 i1) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i0, i1);
        }

        template<typename Int0, typename Int1, typename Int2,
                 typename std::enable_if_t<(COUNT >= 3) && traits::are_int_v<Int0, Int1, Int2>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0, Int1 i1, Int2 i2) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i0, i1, i2);
        }

        template<typename Int0, typename Int1, typename Int2, typename Int3,
                 typename std::enable_if_t<COUNT == 4 && traits::are_int_v<Int0, Int1, Int2, Int3>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0, Int1 i1, Int2 i2, Int3 i3) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i0, i1, i2, i3);
        }

        template<typename Int, typename std::enable_if_t<(COUNT >= 2) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Int2<Int>& i01) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i01[0], i01[1]);
        }

        template<typename Int, typename std::enable_if_t<(COUNT >= 3) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Int3<Int>& i012) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i012[0], i012[1], i012[2]);
        }

        template<typename Int, typename std::enable_if_t<(COUNT >= 4) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Int4<Int>& i0123) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i0123[0], i0123[1], i0123[2], i0123[3]);
        }

    private:
        pointer_type m_ptr{};
        index_type m_strides[(size_t) COUNT]{};
    };

    /// Multidimensional accessor.
    /// \warning This class does not copy the strides. It is simpler and safer to use Accessor instead.
    template<typename Value, int NDIM, typename Index, AccessorTraits TRAITS = AccessorTraits::DEFAULT>
    class AccessorReference {
    public:
        static_assert(!std::is_pointer_v<Value>);
        static_assert(!std::is_reference_v<Value>);
        static_assert(std::is_integral_v<Index>);
        static_assert(NDIM > 0 && NDIM <= 4);

        #if defined(__CUDACC__)
        using pointer_type = std::conditional_t<TRAITS == AccessorTraits::RESTRICT, Value* __restrict__, Value*>;
        #else
        using pointer_type = std::conditional_t<TRAITS == AccessorTraits::RESTRICT, Value* __restrict, Value*>;
        #endif

        using value_type = Value;
        using index_type = Index;
        static constexpr index_type COUNT = NDIM;

    public:
        /// Creates an empty view.
        NOA_HD constexpr AccessorReference() = default;

        /// Creates a strided accessor.
        NOA_HD constexpr AccessorReference(pointer_type pointer, const index_type* strides) noexcept
                : m_ptr(pointer), m_strides(strides) {}

        /// Creates a const accessor from an existing non-const accessor.
        template<typename MutableValue,
                 typename = std::enable_if_t<std::is_const_v<value_type> &&
                                             std::is_same_v<MutableValue, std::remove_const_t<value_type>>>>
        NOA_HD constexpr /* implicit */
        AccessorReference(const AccessorReference<MutableValue, COUNT, index_type, TRAITS>& accessor)
                : m_ptr(accessor.data()), m_strides(accessor.strides()) {}

    public:
        template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
        [[nodiscard]] NOA_HD constexpr index_type stride(Int dim) const noexcept {
            NOA_ASSERT(static_cast<index_type>(dim) < COUNT);
            return m_strides[dim];
        }

        [[nodiscard]] NOA_HD constexpr const index_type* strides() const noexcept { return m_strides; }
        [[nodiscard]] NOA_HD constexpr pointer_type data() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr pointer_type get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr bool empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return !empty(); }

        template<typename Int0, typename Int1, typename = std::enable_if_t<traits::are_int_v<Int0, Int1>>>
        [[nodiscard]] NOA_HD constexpr auto swap(Int0 d0, Int1 d1) const noexcept {
            using output_type = Accessor<value_type, COUNT, index_type, TRAITS>;
            return output_type(m_ptr, strides()).swap(d0, d1);
        }

    public:
        template<typename Int, typename std::enable_if_t<std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD auto offset(Int index) const noexcept {
            NOA_ASSERT((traits::is_uint_v<Int> || index >= Int{0}) && isSafeCast<index_type>(index));
            return AccessorReference(m_ptr + m_strides[0] * static_cast<index_type>(index), strides());
        }

        /// Indexing operator, on 1D accessor. 1D -> ref
        template<typename Int, typename std::enable_if_t<COUNT == 1 && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD value_type& operator[](Int index) const noexcept {
            NOA_ASSERT(m_ptr != nullptr &&
                       (traits::is_uint_v<Int> || index >= Int{0}) &&
                       isSafeCast<index_type>(index));
            return m_ptr[m_strides[0] * static_cast<index_type>(index)];
        }

        /// Indexing operator, multidimensional accessor. ND -> ND-1
        template<typename Int, typename std::enable_if_t<(COUNT > 1) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD auto operator[](Int index) const noexcept {
            NOA_ASSERT(m_ptr != nullptr &&
                       (traits::is_uint_v<Int> || index >= Int{0}) &&
                       isSafeCast<index_type>(index));
            using output_type = AccessorReference<value_type, COUNT - 1, index_type, TRAITS>;
            return output_type(m_ptr + stride(0) * static_cast<index_type>(index), m_strides + 1);
        }

    public:
        template<typename Pointer, typename Int0,
                 typename std::enable_if_t<std::is_integral_v<Int0>, bool> = true>
        [[nodiscard]] NOA_HD auto
        offsetPointer(Pointer pointer, Int0 i0) const noexcept {
            pointer += indexing::at(i0, m_strides[0]);
            return pointer;
        }

        template<typename Pointer, typename Int0, typename Int1,
                 typename std::enable_if_t<(COUNT >= 2) && traits::are_int_v<Int0, Int1>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto
        offsetPointer(Pointer pointer, Int0 i0, Int1 i1) const noexcept {
            pointer += indexing::at(i0, m_strides[0]);
            pointer += indexing::at(i1, m_strides[1]);
            return pointer;
        }

        template<typename Pointer, typename Int0, typename Int1, typename Int2,
                 typename std::enable_if_t<(COUNT >= 3) && traits::are_int_v<Int0, Int1, Int2>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto
        offsetPointer(Pointer pointer, Int0 i0, Int1 i1, Int2 i2) const noexcept {
            pointer += indexing::at(i0, m_strides[0]);
            pointer += indexing::at(i1, m_strides[1]);
            pointer += indexing::at(i2, m_strides[2]);
            return pointer;
        }

        template<typename Pointer, typename Int0, typename Int1, typename Int2, typename Int3,
                 typename std::enable_if_t<COUNT == 4 && traits::are_int_v<Int0, Int1, Int2, Int3>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto
        offsetPointer(Pointer pointer, Int0 i0, Int1 i1, Int2 i2, Int3 i3) const noexcept {
            pointer += indexing::at(i0, m_strides[0]);
            pointer += indexing::at(i1, m_strides[1]);
            pointer += indexing::at(i2, m_strides[2]);
            pointer += indexing::at(i3, m_strides[3]);
            return pointer;
        }

        template<typename Int0,
                 typename std::enable_if_t<(COUNT >= 1) && std::is_integral_v<Int0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i0);
        }

        template<typename Int0, typename Int1,
                 typename std::enable_if_t<(COUNT >= 2) && traits::are_int_v<Int0, Int1>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0, Int1 i1) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i0, i1);
        }

        template<typename Int0, typename Int1, typename Int2,
                 typename std::enable_if_t<(COUNT >= 3) && traits::are_int_v<Int0, Int1, Int2>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0, Int1 i1, Int2 i2) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i0, i1, i2);
        }

        template<typename Int0, typename Int1, typename Int2, typename Int3,
                 typename std::enable_if_t<COUNT == 4 && traits::are_int_v<Int0, Int1, Int2, Int3>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(Int0 i0, Int1 i1, Int2 i2, Int3 i3) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i0, i1, i2, i3);
        }

        template<typename Int, typename std::enable_if_t<(COUNT >= 2) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Int2<Int>& i01) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i01[0], i01[1]);
        }

        template<typename Int, typename std::enable_if_t<(COUNT >= 3) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Int3<Int>& i012) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i012[0], i012[1], i012[2]);
        }

        template<typename Int, typename std::enable_if_t<(COUNT >= 4) && std::is_integral_v<Int>, bool> = true>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(const Int4<Int>& i0123) const noexcept {
            NOA_ASSERT(!empty());
            return *offsetPointer(m_ptr, i0123[0], i0123[1], i0123[2], i0123[3]);
        }

    private:
        pointer_type m_ptr{};
        const index_type* m_strides{};
    };

    template<typename Value, int NDIM, typename Index>
    using AccessorRestrict = Accessor<Value, NDIM, Index, AccessorTraits::RESTRICT>;

    template<typename Value, int NDIM, typename Index>
    using AccessorReferenceRestrict = AccessorReference<Value, NDIM, Index, AccessorTraits::RESTRICT>;
}

