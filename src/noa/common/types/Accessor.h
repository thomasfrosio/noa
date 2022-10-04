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

    template<typename T, int N, typename I, AccessorTraits TRAITS>
    class AccessorReference;

    /// Multidimensional accessor using C-style multidimensional array indexing.
    /// \details This class in meant to provide multidimensional-indexing on contiguous/strided data.
    ///          It only keeps track of a pointer and one stride for each dimension. As opposed, to View,
    ///          it cannot bound check the index against the dimension size.
    /// \note The only reason the Accessor doesn't keep track of the dimension size is that in a lot of
    ///       cases, the input/output arrays have the same size/shape, so kernels would have to transfer
    ///       redundant information to the device, which could cause a performance hit for CUDA kernels.
    ///       If this is not an issue, prefer to use View.
    template<typename T, int N, typename I, AccessorTraits TRAITS = AccessorTraits::DEFAULT>
    class Accessor {
    public:
        static_assert(!std::is_pointer_v<T>);
        static_assert(!std::is_reference_v<T>);
        static_assert(std::is_integral_v<I>);
        static_assert(N > 0 && N <= 4);

        #if defined(__CUDACC__)
        using ptr_type = std::conditional_t<TRAITS == AccessorTraits::RESTRICT, T* __restrict__, T*>;
        #else
        using ptr_type = std::conditional_t<TRAITS == AccessorTraits::RESTRICT, T* __restrict, T*>;
        #endif

        using value_type = T;
        using index_type = I;
        static constexpr index_type COUNT = N;

    public:
        /// Creates an empty view.
        NOA_HD constexpr Accessor() = default;

        /// Creates a strided accessor.
        template<typename I0, typename = std::enable_if_t<std::is_integral_v<I0>>>
        NOA_HD constexpr Accessor(ptr_type pointer, const I0* strides) noexcept : m_ptr(pointer) {
            NOA_ASSERT(strides != nullptr);
            for (int i = 0; i < N; ++i) {
                NOA_ASSERT(isSafeCast<index_type>(strides[i]));
                m_strides[i] = static_cast<index_type>(strides[i]);
            }
        }

        template<typename I0, typename = std::enable_if_t<N == 1 && traits::is_int_v<I0>>>
        NOA_HD constexpr Accessor(ptr_type pointer, I0 stride) noexcept
                : Accessor(pointer, &stride) {}

        template<typename I0,
                 typename = std::enable_if_t<(N == 2 && traits::is_int2_v<I0>) ||
                                             (N == 3 && traits::is_int3_v<I0>) ||
                                             (N == 4 && traits::is_int4_v<I0>)>>
        NOA_HD constexpr Accessor(ptr_type pointer, const I0& strides) noexcept
                : Accessor(pointer, strides.get()) {}

        /// Creates a const accessor from an existing non-const accessor.
        template<typename U, typename = std::enable_if_t<std::is_const_v<T> && std::is_same_v<U, std::remove_const_t<T>>>>
        NOA_HD constexpr /* implicit */ Accessor(const Accessor<U, N, index_type, TRAITS>& accessor)
                : Accessor(accessor.data(), accessor.strides()) {}

    public:
        template<typename I0, typename = std::enable_if_t<std::is_integral_v<I0>>>
        [[nodiscard]] NOA_HD constexpr index_type& stride(I0 dim) noexcept {
            NOA_ASSERT(static_cast<index_type>(dim) < N);
            return m_strides[dim];
        }

        template<typename I0, typename = std::enable_if_t<std::is_integral_v<I0>>>
        [[nodiscard]] NOA_HD constexpr const index_type& stride(I0 dim) const noexcept {
            NOA_ASSERT(static_cast<index_type>(dim) < N);
            return m_strides[dim];
        }

        [[nodiscard]] NOA_HD constexpr index_type* strides() noexcept { return m_strides; }
        [[nodiscard]] NOA_HD constexpr const index_type* strides() const noexcept { return m_strides; }
        [[nodiscard]] NOA_HD constexpr ptr_type data() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr ptr_type get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr bool empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return !empty(); }

        template<typename I0, typename I1, typename = std::enable_if_t<std::is_integral_v<I0> && std::is_integral_v<I1>>>
        [[nodiscard]] NOA_HD constexpr Accessor swap(I0 d0, I1 d1) const noexcept {
            NOA_ASSERT(static_cast<index_type>(d0) < N && static_cast<index_type>(d1) < N);
            Accessor out = *this;
            const auto tmp = out.stride(d0); // swap
            out.stride(d0) = out.stride(d1);
            out.stride(d1) = tmp;
            return out;
        }

    public:
        /// Offsets the current pointer at dimension 0 (without changing the dimensionality of the accessor)
        /// and return a reference of that new accessor.
        template<typename I0, typename std::enable_if_t<std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD auto offset(I0 index) const noexcept {
            NOA_ASSERT((traits::is_uint_v<I0> || index >= I0{0}) && isSafeCast<index_type>(index));
            using output_type = AccessorReference<T, N, index_type, TRAITS>;
            return output_type(m_ptr + m_strides[0] * static_cast<index_type>(index), strides());
        }

        /// Indexing operator, on 1D accessor. 1D -> ref
        template<typename I0, typename std::enable_if_t<N == 1 && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD T& operator[](I0 index) const noexcept {
            NOA_ASSERT(m_ptr != nullptr && (traits::is_uint_v<I0> || index >= I0{0}) && isSafeCast<index_type>(index));
            return m_ptr[m_strides[0] * static_cast<index_type>(index)];
        }

        /// Indexing operator, multidimensional accessor. ND -> ND-1
        template<typename I0, typename std::enable_if_t<(N > 1) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD auto operator[](I0 index) const noexcept {
            NOA_ASSERT(m_ptr != nullptr && (traits::is_uint_v<I0> || index >= I0{0}) && isSafeCast<index_type>(index));
            // While returning an Accessor adds no overhead on the CPU, on the GPU it does.
            // Returning a reference removes a few instructions.
            using output_type = AccessorReference<T, N - 1, index_type, TRAITS>;
            return output_type(m_ptr + stride(0) * static_cast<index_type>(index), m_strides + 1);
        }

    public:
        template<typename I0,
                typename std::enable_if_t<(N >= 1) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0) const noexcept {
            NOA_ASSERT(!empty());
            return m_ptr[indexing::at(i0, m_strides[0])];
        }

        template<typename I0, typename I1,
                typename std::enable_if_t<(N >= 2) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0, I1 i1) const noexcept {
            NOA_ASSERT(!empty());
            ptr_type tmp = m_ptr;
            tmp += indexing::at(i0, m_strides[0]);
            tmp += indexing::at(i1, m_strides[1]);
            return *tmp;
        }

        template<typename I0, typename std::enable_if_t<(N >= 2) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(const Int2<I0>& i01) const noexcept {
            return (*this)(i01[0], i01[1]);
        }

        template<typename I0, typename I1, typename I2,
                typename std::enable_if_t<(N >= 3) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0, I1 i1, I2 i2) const noexcept {
            NOA_ASSERT(!empty());
            ptr_type tmp = m_ptr;
            tmp += indexing::at(i0, m_strides[0]);
            tmp += indexing::at(i1, m_strides[1]);
            tmp += indexing::at(i2, m_strides[2]);
            return *tmp;
        }

        template<typename I0, typename std::enable_if_t<(N >= 3) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(const Int3<I0>& i012) const noexcept {
            return (*this)(i012[0], i012[1], i012[2]);
        }

        template<typename I0, typename I1, typename I2, typename I3,
                typename std::enable_if_t<N == 4 && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
            NOA_ASSERT(!empty());
            ptr_type tmp = m_ptr;
            tmp += indexing::at(i0, m_strides[0]);
            tmp += indexing::at(i1, m_strides[1]);
            tmp += indexing::at(i2, m_strides[2]);
            tmp += indexing::at(i3, m_strides[3]);
            return *tmp;
        }

        template<typename I0, typename std::enable_if_t<(N >= 4) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(const Int4<I0>& i0123) const noexcept {
            return (*this)(i0123[0], i0123[1], i0123[2], i0123[3]);
        }

    private:
        ptr_type m_ptr{};
        index_type m_strides[(size_t) N]{};
    };

    /// Multidimensional accessor.
    /// \warning This class does not copy the strides. It is simpler and safer to use Accessor instead.
    template<typename T, int N, typename I, AccessorTraits TRAITS = AccessorTraits::DEFAULT>
    class AccessorReference {
    public:
        static_assert(!std::is_pointer_v<T>);
        static_assert(!std::is_reference_v<T>);
        static_assert(std::is_integral_v<I>);
        static_assert(N > 0 && N <= 4);

        #if defined(__CUDACC__)
        using ptr_type = std::conditional_t<TRAITS == AccessorTraits::RESTRICT, T* __restrict__, T*>;
        #else
        using ptr_type = std::conditional_t<TRAITS == AccessorTraits::RESTRICT, T* __restrict, T*>;
        #endif

        using value_type = T;
        using index_type = I;
        static constexpr index_type COUNT = N;

    public:
        /// Creates an empty view.
        NOA_HD constexpr AccessorReference() = default;

        /// Creates a strided accessor.
        NOA_HD constexpr AccessorReference(ptr_type pointer, const index_type* strides) noexcept
                : m_ptr(pointer), m_strides(strides) {}

        /// Creates a const accessor from an existing non-const accessor.
        template<typename U, typename = std::enable_if_t<std::is_const_v<T> && std::is_same_v<U, std::remove_const_t<T>>>>
        NOA_HD constexpr /* implicit */ AccessorReference(const AccessorReference<U, N, index_type, TRAITS>& accessor)
                : m_ptr(accessor.data()), m_strides(accessor.strides()) {}

    public:
        template<typename I0, typename = std::enable_if_t<std::is_integral_v<I0>>>
        [[nodiscard]] NOA_HD constexpr index_type stride(I0 dim) const noexcept {
            NOA_ASSERT(static_cast<index_type>(dim) < N);
            return m_strides[dim];
        }

        [[nodiscard]] NOA_HD constexpr const index_type* strides() const noexcept { return m_strides; }
        [[nodiscard]] NOA_HD constexpr ptr_type data() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr ptr_type get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr bool empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return !empty(); }

        template<typename I0, typename I1, typename = std::enable_if_t<std::is_integral_v<I0> && std::is_integral_v<I1>>>
        [[nodiscard]] NOA_HD constexpr auto swap(I0 d0, I1 d1) const noexcept {
            using output_type = Accessor<value_type, N, index_type, TRAITS>;
            return output_type(m_ptr, strides()).swap(d0, d1);
        }

    public:
        template<typename I0, typename std::enable_if_t<std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD auto offset(I0 index) const noexcept {
            NOA_ASSERT((traits::is_uint_v<I0> || index >= I0{0}) && isSafeCast<index_type>(index));
            return AccessorReference(m_ptr + m_strides[0] * static_cast<index_type>(index), strides());
        }

        /// Indexing operator, on 1D accessor. 1D -> ref
        template<typename I0, typename std::enable_if_t<N == 1 && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD T& operator[](I0 index) const noexcept {
            NOA_ASSERT(m_ptr != nullptr && (traits::is_uint_v<I0> || index >= I0{0}) && isSafeCast<index_type>(index));
            return m_ptr[m_strides[0] * static_cast<index_type>(index)];
        }

        /// Indexing operator, multidimensional accessor. ND -> ND-1
        template<typename I0, typename std::enable_if_t<(N > 1) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD auto operator[](I0 index) const noexcept {
            NOA_ASSERT(m_ptr != nullptr && (traits::is_uint_v<I0> || index >= I0{0}) && isSafeCast<index_type>(index));
            using output_type = AccessorReference<value_type, N - 1, index_type, TRAITS>;
            return output_type(m_ptr + stride(0) * static_cast<index_type>(index), m_strides + 1);
        }

    public:
        template<typename I0,
                typename std::enable_if_t<(N >= 1) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0) const noexcept {
            NOA_ASSERT(!empty());
            return m_ptr[indexing::at(i0, m_strides[0])];
        }

        template<typename I0, typename I1,
                typename std::enable_if_t<(N >= 2) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0, I1 i1) const noexcept {
            NOA_ASSERT(!empty());
            ptr_type tmp = m_ptr;
            tmp += indexing::at(i0, m_strides[0]);
            tmp += indexing::at(i1, m_strides[1]);
            return *tmp;
        }

        template<typename I0, typename std::enable_if_t<(N >= 2) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(const Int2<I0>& i01) const noexcept {
            return (*this)(i01[0], i01[1]);
        }

        template<typename I0, typename I1, typename I2,
                 typename std::enable_if_t<(N >= 3) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0, I1 i1, I2 i2) const noexcept {
            NOA_ASSERT(!empty());
            ptr_type tmp = m_ptr;
            tmp += indexing::at(i0, m_strides[0]);
            tmp += indexing::at(i1, m_strides[1]);
            tmp += indexing::at(i2, m_strides[2]);
            return *tmp;
        }

        template<typename I0, typename std::enable_if_t<(N >= 3) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(const Int3<I0>& i012) const noexcept {
            return (*this)(i012[0], i012[1], i012[2]);
        }

        template<typename I0, typename I1, typename I2, typename I3,
                 typename std::enable_if_t<N == 4 && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
            // Note: We add each dimension offset to the pointer to reduce the risk of integer overflow.
            //       On the CPU, indexing should use 8 bytes values (this is genuinely more efficient).
            //       On the GPU-CUDA, we mostly use 4 bytes values, so integer overflow is actual relevant.
            // Note: Using the C-style multidimensional indexing operator[] gives identical code on the
            //       GPU using 4 bytes values, but on the CPU is adds an extra "move" instruction. See
            //       https://godbolt.org/z/rYY71aYK3, hence these operator(), which are as efficient as
            //       it can be...
            NOA_ASSERT(!empty());
            ptr_type tmp = m_ptr;
            tmp += indexing::at(i0, m_strides[0]);
            tmp += indexing::at(i1, m_strides[1]);
            tmp += indexing::at(i2, m_strides[2]);
            tmp += indexing::at(i3, m_strides[3]);
            return *tmp;
        }

        template<typename I0, typename std::enable_if_t<(N >= 4) && std::is_integral_v<I0>, bool> = true>
        [[nodiscard]] NOA_HD constexpr T& operator()(const Int4<I0>& i0123) const noexcept {
            return (*this)(i0123[0], i0123[1], i0123[2], i0123[3]);
        }

    private:
        ptr_type m_ptr{};
        const index_type* m_strides{};
    };

    template<typename T, int N, typename I>
    using AccessorRestrict = Accessor<T, N, I, AccessorTraits::RESTRICT>;

    template<typename T, int N, typename I>
    using AccessorReferenceRestrict = AccessorReference<T, N, I, AccessorTraits::RESTRICT>;
}

