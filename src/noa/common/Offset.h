/// \file noa/common/types/Index.h
/// \brief Indexing utilities.
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/types/IntX.h"

namespace noa {
    /// Returns the memory offset corresponding to the given 4D indexes.
    /// \param i0,i1,i2,i3  Multi-dimensional indexes.
    /// \param stride       Strides associated with these indexes.
    template<typename T, typename U, typename V, typename W, typename Z>
    NOA_FHD constexpr auto at(T i0, U i1, V i2, W i3, Int4<Z> stride) noexcept {
        static_assert(sizeof(Z) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        static_assert(std::is_integral_v<T> || std::is_integral_v<U> ||
                      std::is_integral_v<V> || std::is_integral_v<W>);
        return static_cast<Z>(i0) * static_cast<Z>(stride[0]) +
               static_cast<Z>(i1) * static_cast<Z>(stride[1]) +
               static_cast<Z>(i2) * static_cast<Z>(stride[2]) +
               static_cast<Z>(i3) * static_cast<Z>(stride[3]);
    }

    /// Returns the memory offset corresponding to the given 4D indexes.
    /// \param index    Multi-dimensional indexes.
    /// \param stride   Strides associated with these indexes.
    template<typename T, typename U>
    NOA_FHD constexpr auto at(Int4<T> index, Int4<U> stride) noexcept {
        return at(index[0], index[1], index[2], index[3], stride);
    }

    /// Returns the memory offset corresponding to the given 3D indexes.
    /// \tparam W       Int3 or Int4.
    /// \param i0,i1,i2 Multi-dimensional indexes.
    /// \param stride   Strides associated with these indexes. Only the first 3 values are used.
    template<typename T, typename U, typename V, typename W,
             typename = std::enable_if_t<noa::traits::is_int4_v<W> || noa::traits::is_int3_v<W>>>
    NOA_FHD constexpr auto at(T i0, U i1, V i2, W stride) noexcept {
        using value_t = noa::traits::value_type_t<W>;
        static_assert(sizeof(W) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        static_assert(std::is_integral_v<T> || std::is_integral_v<U> || std::is_integral_v<V>);
        return static_cast<value_t>(i0) * static_cast<value_t>(stride[0]) +
               static_cast<value_t>(i1) * static_cast<value_t>(stride[1]) +
               static_cast<value_t>(i2) * static_cast<value_t>(stride[2]);
    }

    /// Returns the memory offset corresponding to the given 3D indexes.
    /// \tparam U       Int3 or Int4.
    /// \param index    Multi-dimensional indexes.
    /// \param stride   Strides associated with these indexes. Only the first 3 values are used.
    template<typename T, typename U,
             typename = std::enable_if_t<noa::traits::is_int4_v<U> || noa::traits::is_int3_v<U>>>
    NOA_FHD constexpr auto at(Int3<T> index, U stride) noexcept {
        return at(index[0], index[1], index[2], stride);
    }

    /// Returns the memory offset corresponding to the given 2D indexes.
    /// \tparam W       Int2, Int3, or Int4.
    /// \param i0,i1    Multi-dimensional indexes.
    /// \param stride   Strides associated with these indexes. Only the first 2 values are used.
    template<typename T, typename U, typename V,
             typename = std::enable_if_t<noa::traits::is_intX_v<V>>>
    NOA_FHD constexpr auto at(T i0, U i1, V stride) noexcept {
        using value_t = noa::traits::value_type_t<V>;
        static_assert(sizeof(value_t) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        static_assert(std::is_integral_v<T> || std::is_integral_v<U>);
        return static_cast<value_t>(i0) * static_cast<value_t>(stride[0]) +
               static_cast<value_t>(i1) * static_cast<value_t>(stride[1]);
    }

    /// Returns the memory offset corresponding to the given 2D indexes.
    /// \tparam W       Int2, Int3, or Int4.
    /// \param i0,i1    Multi-dimensional indexes.
    /// \param stride   Strides associated with these indexes. Only the first 2 values are used.
    template<typename T, typename U,
             typename = std::enable_if_t<noa::traits::is_int4_v<U> || noa::traits::is_int3_v<U>>>
    NOA_FHD constexpr auto at(Int2<T> index, U stride) noexcept {
        return at(index[0], index[1], stride);
    }

    /// Returns the 2D rightmost indexes corresponding to the given memory offset.
    /// \param offset   Linear memory offset.
    /// \param pitch    Pitch of the innermost dimension.
    /// \return         {index in the outermost dimension,
    ///                  index in the innermost dimension}.
    template<typename T>
    NOA_FHD constexpr Int2<T> indexes(T offset, T pitch) noexcept {
        const T i0 = offset / pitch;
        const T i1 = offset - i0 * pitch;
        return {i0, i1};
    }

    /// Returns the 3D rightmost indexes corresponding to the given memory offset.
    /// \param offset   Linear memory offset.
    /// \param p0       Pitch of the outermost dimension.
    /// \param p1       Pitch of the innermost dimension.
    /// \return         {index in the outermost dimension,
    ///                  index in the second-most dimension,
    ///                  index in the innermost dimension}.
    template<typename T>
    NOA_FHD constexpr Int3<T> indexes(T offset, T p0, T p1) noexcept {
        const T i0 = offset / (p0 * p1);
        const T tmp = offset - i0 * p0 * p1; // remove the offset to section
        const T i1 = tmp / p1;
        const T i2 = tmp - i1 * p1;
        return {i0, i1, i2};
    }

    /// Whether or not the dimensions are contiguous.
    /// \param shape    Rightmost shape.
    /// \param stride   Rightmost stride.
    template<typename T, typename = std::enable_if_t<noa::traits::is_intX_v<T>>>
    NOA_FHD auto isContiguous(T stride, T shape) {
        return stride == shape.strides();
    }
}
