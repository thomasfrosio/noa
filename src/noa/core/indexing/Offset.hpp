#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Enums.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/utils/ClampCast.hpp"
#include "noa/core/utils/SafeCast.hpp"

namespace noa::indexing {
    /// Returns the memory offset corresponding to the given 4D indexes.
    /// \param i0,i1,i2,i3  Multi-dimensional indexes.
    /// \param strides      Strides associated with these indexes.
    template<typename I0, typename I1, typename I2, typename I3, typename Offset,
             nt::enable_if_bool_t<nt::are_int_v<I0, I1, I2, I3>> = true>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
            I0 i0, I1 i1, I2 i2, I3 i3,
            const Strides4<Offset>& strides
    ) noexcept -> Offset {
        static_assert(sizeof(Offset) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        NOA_ASSERT(is_safe_cast<Offset>(i0) && is_safe_cast<Offset>(i1) &&
                   is_safe_cast<Offset>(i2) && is_safe_cast<Offset>(i3));

        return static_cast<Offset>(i0) * strides[0] +
               static_cast<Offset>(i1) * strides[1] +
               static_cast<Offset>(i2) * strides[2] +
               static_cast<Offset>(i3) * strides[3];
    }

    /// Returns the memory offset corresponding to the given 4D indexes.
    template<typename Index, typename Offset>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
            const Vec4<Index>& index,
            const Strides4<Offset>& strides
    ) noexcept {
        return offset_at(index[0], index[1], index[2], index[3], strides);
    }

    /// Returns the memory offset corresponding to the given 3d indexes.
    /// \param i0,i1,i2     Multi-dimensional indexes.
    /// \param strides      Strides associated with these indexes. Only the first 3 values are used.
    template<typename I0, typename I1, typename I2, typename Offset, size_t N,
             nt::enable_if_bool_t<nt::are_int_v<I0, I1, I2> && (N >= 3)> = true>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
            I0 i0, I1 i1, I2 i2,
            const Strides<Offset, N>& strides)
    noexcept -> Offset {
        static_assert(sizeof(Offset) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        NOA_ASSERT(is_safe_cast<Offset>(i0) && is_safe_cast<Offset>(i1) && is_safe_cast<Offset>(i2));

        return static_cast<Offset>(i0) * strides[0] +
               static_cast<Offset>(i1) * strides[1] +
               static_cast<Offset>(i2) * strides[2];
    }

    /// Returns the memory offset corresponding to the given 3d indexes.
    template<typename Index, typename Offset, size_t N,
             nt::enable_if_bool_t<nt::is_int_v<Index> && (N >= 3)> = true>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
            const Vec3<Index>& index,
            const Strides<Offset, N>& strides
    ) noexcept {
        return offset_at(index[0], index[1], index[2], strides);
    }

    /// Returns the memory offset corresponding to the given 2d indexes.
    /// \param i0,i1        Multi-dimensional indexes.
    /// \param strides      Strides associated with these indexes. Only the first 2 values are used.
    template<typename I0, typename I1, typename Offset, size_t N,
             nt::enable_if_bool_t<nt::are_int_v<I0, I1> && (N >= 2)> = true>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
            I0 i0, I1 i1,
            const Strides<Offset, N>& strides)
    noexcept -> Offset {
        static_assert(sizeof(Offset) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        NOA_ASSERT(is_safe_cast<Offset>(i0) && is_safe_cast<Offset>(i1));

        return static_cast<Offset>(i0) * strides[0] +
               static_cast<Offset>(i1) * strides[1];
    }

    /// Returns the memory offset corresponding to the given 3d indexes.
    template<typename Index, typename Offset, size_t N,
             nt::enable_if_bool_t<nt::is_int_v<Index> && (N >= 2)> = true>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
            const Vec2<Index>& index,
            const Strides<Offset, N>& strides
    ) noexcept {
        return offset_at(index[0], index[1], strides);
    }

    /// Returns the memory offset corresponding to the given 1D indexes.
    /// \param i0           Index.
    /// \param strides      Strides associated with these indexes. Only the first value is used.
    template<typename Index, typename Strides,
             nt::enable_if_bool_t<nt::is_int_v<Index> && (nt::is_stridesX_v<Strides> || nt::is_int_v<Strides>)> = true>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(Index i0, Strides strides) noexcept {
        using offset_t = nt::value_type_t<Strides>;
        static_assert(sizeof(offset_t) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        NOA_ASSERT(is_safe_cast<offset_t>(i0));

        if constexpr (nt::is_int_v<Strides>) {
            return static_cast<offset_t>(i0) * strides;
        } else {
            return static_cast<offset_t>(i0) * strides[0];
        }
    }

    /// Returns the memory offset corresponding to the given 3d indexes.
    template<typename Index, typename Offset, size_t N,
             nt::enable_if_bool_t<nt::is_int_v<Index> && (N >= 1)> = true>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
            const Vec1<Index>& index,
            const Strides<Offset, N>& strides
    ) noexcept {
        return offset_at(index[0], strides);
    }

    /// If \p idx is out-of-bound, computes a valid index, i.e. [0, size-1], according to \p MODE.
    /// Otherwise, returns \p idx. \p size should be > 0.
    template<Border MODE, typename SInt, nt::enable_if_bool_t<std::is_signed_v<SInt>> = true>
    [[nodiscard]] NOA_IHD constexpr SInt index_at(SInt idx, SInt size) {
        static_assert(MODE == Border::CLAMP || MODE == Border::PERIODIC ||
                      MODE == Border::MIRROR || MODE == Border::REFLECT);
        NOA_ASSERT(size > 0);

        // a % b == a - b * (a / b) == a + b * (-a / b)
        // Having a < 0 is well-defined since C++11.
        if constexpr (MODE == Border::CLAMP) {
            if (idx < 0)
                idx = 0;
            else if (idx >= size)
                idx = size - 1;
        } else if constexpr (MODE == Border::PERIODIC) {
            // 0 1 2 3 0 1 2 3 0 1 2 3 |  0 1 2 3  | 0 1 2 3 0 1 2 3 0 1 2 3
            SInt rem = idx % size; // FIXME maybe enclose this, at the expense of two jumps?
            idx = rem < 0 ? rem + size : rem;
        } else if constexpr (MODE == Border::MIRROR) {
            // 0 1 2 3 3 2 1 0 0 1 2 3 3 2 1 0 |  0 1 2 3  | 3 2 1 0 0 1 2 3 3 2 1 0
            if (idx < 0)
                idx = -idx - 1;
            if (idx >= size) {
                SInt period = 2 * size;
                idx %= period;
                if (idx >= size)
                    idx = period - idx - 1;
            }
        } else if constexpr (MODE == Border::REFLECT) {
            // 0 1 2 3 2 1 0 1 2 3 2 1 |  0 1 2 3  | 2 1 0 1 2 3 2 1 0
            if (idx < 0)
                idx = -idx;
            if (idx >= size) {
                SInt period = 2 * size - 2;
                idx %= period;
                if (idx >= size)
                    idx = period - idx;
            }
        }
        return idx;
    }

    /// Returns the 2d rightmost indexes corresponding to
    /// the given memory offset in a contiguous layout.
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Vec2<T> offset2index(T offset, T size) noexcept {
        NOA_ASSERT(size > 0);
        const auto i0 = offset / size;
        const auto i1 = offset - i0 * size;
        return {i0, i1};
    }

    /// Returns the 3d rightmost indexes corresponding to
    /// the given memory offset in a contiguous layout.
    /// \param offset   Linear memory offset.
    /// \param s0,s1    DH sizes.
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Vec3<T> offset2index(T offset, T s0, T s1) noexcept {
        NOA_ASSERT(s0 > 0 && s1 > 0);
        const auto i0 = offset / (s0 * s1);
        offset -= i0 * s0 * s1;
        const auto i1 = offset / s1;
        offset -= i1 * s1;
        return {i0, i1, offset};
    }

    /// Returns the 4D rightmost indexes corresponding to
    /// the given memory offset in a contiguous layout.
    /// \param offset   Linear memory offset.
    /// \param s0,s1,s2 DHW sizes.
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Vec4<T> offset2index(T offset, T s0, T s1, T s2) noexcept {
        NOA_ASSERT(s0 > 0 && s1 > 0 && s2 > 0);
        const auto i0 = offset / (s0 * s1 * s2);
        offset -= i0 * s0 * s1 * s2;
        const auto i1 = offset / (s1 * s2);
        offset -= i1 * s1 * s2;
        const auto i2 = offset / s2;
        offset -= i2 * s2;
        return {i0, i1, i2, offset};
    }

    /// Returns the multidimensional indexes corresponding to a memory \p offset, assuming BDHW C-contiguity.
    /// \param offset   Memory offset within the array.
    /// \param shape    Shape of the array.
    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr Vec<T, N> offset2index(T offset, Shape<T, N> shape) noexcept {
        if constexpr (N == 1) {
            return Vec<T, N>{offset};
        } else if constexpr (N == 2) {
            return offset2index(offset, shape[1]);
        } else if constexpr (N == 3) {
            return offset2index(offset, shape[1], shape[2]);
        } else {
            return offset2index(offset, shape[1], shape[2], shape[3]);
        }
    }

    /// Returns the multidimensional indexes corresponding to a memory \p offset.
    /// \details Given a memory layout (i.e. strides and shape), this function computes the ND logical indexes
    ///          pointing at the given memory \p offset. Broadcasting is not supported, so the strides should
    ///          be greater than 0. Otherwise, any ordering is supported.
    /// \param offset   Memory offset within the array.
    /// \param strides  Strides of the array.
    /// \param shape    Shape of the array.
    template<bool ASSUME_RIGHTMOST = false, typename T, size_t N>
    [[nodiscard]] NOA_IHD constexpr auto offset2index(
            T offset,
            const Strides<T, N>& strides,
            const Shape<T, N>& shape
    ) noexcept {
        NOA_ASSERT(noa::all(shape > 0));
        Vec<T, N> out{0};
        T remain = offset;

        if constexpr (ASSUME_RIGHTMOST) {
            for (size_t i = 0; i < N; ++i) {
                if (shape[i] > 1) { // if empty, ignore it.
                    NOA_ASSERT(strides[i] > 0);
                    out[i] = remain / strides[i]; // single-divide optimization should kick in
                    remain %= strides[i]; // or remain -= out[i] * stride
                }
            }
        } else {
            const auto rightmost_order = order(strides, shape);
            for (size_t i = 0; i < N; ++i) {
                const auto idx = rightmost_order[i];
                if (shape[idx] > 1) {
                    NOA_ASSERT(strides[idx] > 0);
                    out[idx] = remain / strides[idx];
                    remain %= strides[idx];
                }
            }
        }

        NOA_ASSERT(remain == 0);
        return out;
    }
}
