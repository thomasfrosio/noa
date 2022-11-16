#pragma once

#include "noa/unified/Array.h"

namespace noa::math::details {
    template<typename S, typename T, typename U>
    constexpr bool is_valid_find_v =
            traits::is_any_v<T, uint32_t, uint64_t, int32_t, int64_t, half_t, float, double> &&
            traits::is_any_v<U, uint32_t, uint64_t, int32_t, int64_t> &&
            traits::is_any_v<S, noa::math::first_min_t, noa::math::first_max_t, noa::math::last_min_t, noa::math::last_max_t>;
}

namespace noa::math {
    /// Returns the memory offset(s) of a particular kind of value(s).
    /// \tparam S               Any of {first|last}_{min|max}_t.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float or double.
    /// \tparam U               (u)int32_t or (u)int64_t.
    /// \param searcher         Abstract search functor.
    /// \param[in] input        Input array.
    /// \param[out] offsets     Contiguous vector where the memory offset(s) are saved.
    /// \param batch            Whether each batch in \p input should be segmented. If true, the offset of the
    ///                         result value in each batch is returned and these offsets are relative to the
    ///                         beginning of each batch. If false, a single value is returned in \p offsets.
    /// \param swap_layout      Whether the function is allowed to reorder the input for fastest search.
    ///                         If true, the search is done following the optimal strides order of \p input,
    ///                         so the first minimum value might be different if the array is e.g. column-major
    ///                         or row-major. Otherwise, the search is done in the rightmost order. Note that if
    ///                         \p batch is true, the batch dimension is cannot swapped.
    /// \note One can retrieve the multidimensional indexes from the offset using indexing::indexes(offset, input).
    /// \note If \p input is on the CPU, \p offsets should be dereferenceable by the CPU.
    ///       If it is on the GPU, it can be on any device, including the CPU.
    template<typename S, typename T, typename U, typename = std::enable_if_t<details::is_valid_find_v<S, T, U>>>
    void find(S searcher, const Array<T>& input, const Array<U>& offsets, bool batch = true, bool swap_layout = false);

    /// Returns the memory offset of a particular kind of value.
    /// \tparam S               Any of {first|last}_{min|max}_t.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float or double.
    /// \tparam offset_         (u)int32_t or (u)int64_t.
    /// \param searcher         Abstract search functor.
    /// \param[in] input        Input array.
    /// \param swap_layout      Whether the function is allowed to reorder the input for fastest search.
    ///                         If true, the search is done following the optimal strides order of \p input,
    ///                         so the first minimum value might be different if the array is e.g. column-major
    ///                         or row-major. Otherwise, the search is done in the rightmost order.
    template<typename offset_t = dim_t, typename S, typename T,
             typename = std::enable_if_t<details::is_valid_find_v<S, T, offset_t>>>
    [[nodiscard]] offset_t find(S searcher, const Array<T>& input, bool swap_layout = false);
}

#define NOA_UNIFIED_MATH_FIND_
#include "noa/unified/math/Find.inl"
#undef NOA_UNIFIED_MATH_FIND_
