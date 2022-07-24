/// \file noa/cpu/math/Find.h
/// \brief Find index of min/max values.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021
#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/common/Functors.h"

namespace noa::cpu::math::details {
    template<typename S, typename T, typename U>
    constexpr bool is_valid_find_v =
            traits::is_any_v<T, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half_t, float, double> &&
            traits::is_any_v<U, uint32_t, uint64_t, int32_t, int64_t> &&
            traits::is_any_v<S, noa::math::first_min_t, noa::math::first_max_t, noa::math::last_min_t, noa::math::last_max_t>;
}

namespace noa::cpu::math {
    /// Returns the memory offset(s) of a particular kind of value(s).
    /// \details The search is done following the optimal strides order of \p input, so the first minimum
    ///          value might be different if the array is e.g. column-major or row-major.
    /// \tparam S               Any of {first|last}_{min|max}_t.
    /// \tparam T               Any data type.
    /// \tparam U               (u)int32_t or (u)int64_t.
    /// \param searcher         Abstract search functor.
    /// \param[in] input        On the \b host. Input array.
    /// \param strides          BDHW strides of \p input.
    /// \param shape            BDHW shape of \p input.
    /// \param[out] offsets     On the \b host. Memory offset(s).
    /// \param batch            Whether each batch in \p input should be segmented. If true, the offset of the
    ///                         result value in each batch is returned and these offsets are relative to the
    ///                         beginning of each batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note One can retrieve the multi-dimensional indexes from the offset using noa::indexing::indexes(...).
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename S, typename T, typename U, typename = std::enable_if_t<details::is_valid_find_v<S, T, U>>>
    void find(S searcher, const shared_t<T[]>& input, size4_t strides, size4_t shape,
              const shared_t<U[]>& offsets, bool batch, Stream& stream);

    /// Returns the memory offset of a particular kind of value.
    /// \tparam S               Any of {first|last}_{min|max}_t.
    /// \tparam T               Any data type.
    /// \tparam offset_         (u)int32_t or (u)int64_t.
    /// \param searcher         Abstract search functor.
    /// \param[in] input        On the \b host. Input array.
    /// \param strides          BDHW strides of \p input.
    /// \param shape            BDHW shape of \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<typename offset_t = size_t, typename S, typename T,
             typename = std::enable_if_t<details::is_valid_find_v<S, T, offset_t>>>
    offset_t find(S searcher, const shared_t<T[]>& input, size4_t strides, size4_t shape, Stream& stream);

    /// Returns the index of a particular kind of value.
    /// \tparam S               Any of {first|last}_{min|max}_t.
    /// \tparam T               Any data type.
    /// \tparam offset_t        (u)int32_t or (u)int64_t.
    /// \param searcher         Abstract search functor.
    /// \param[in] input        On the \b host. Input array.
    /// \param elements         Number of elements in \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<typename offset_t = size_t, typename S, typename T,
             typename = std::enable_if_t<details::is_valid_find_v<S, T, offset_t>>>
    offset_t find(S searcher, const shared_t<T[]>& input, size_t elements, Stream& stream);
}
