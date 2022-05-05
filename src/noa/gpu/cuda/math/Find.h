/// \file noa/gpu/cuda/math/Find.h
/// \brief Find offsets for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/common/Functors.h"

namespace noa::cuda::math::details {
    template<typename T, typename U>
    constexpr bool is_valid_find_v =
            traits::is_any_v<T, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half_t, float, double> &&
            traits::is_any_v<U, int32_t, int64_t, uint32_t, uint64_t>;
}

namespace noa::cuda::math {
    /// Returns the memory offset(s) of the first minimum value(s).
    /// \tparam T               Any data type.
    /// \tparam U               (u)int32_t or (u)int64_t.
    /// \param[in] input        On the \b device. Input array.
    /// \param stride           Rightmost stride of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] offsets     On the \b device or \b host. Memory offset(s).
    /// \param batch            Whether \p input should be segmented by its outermost dimension.
    ///                         If true, the offset of minimum value in each batch is returned and
    ///                         these offsets are relative to the beginning of the batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note One can retrieve the multi-dimensional indexes from the offset using noa::indexing::indexes(...).
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_find_v<T, U>>>
    void find(noa::math::min_t, const shared_t<T[]>& input, size4_t stride, size4_t shape,
              const shared_t<U[]>& offsets, bool batch, Stream& stream);

    /// Returns the index of the minimum value.
    /// \tparam T               Any data type.
    /// \tparam offset_t        (u)int32_t or (u)int64_t.
    /// \param[in] input        On the \b device. Input array.
    /// \param elements         Number of elements in \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<typename offset_t = size_t, typename T, typename = std::enable_if_t<details::is_valid_find_v<T, offset_t>>>
    offset_t find(noa::math::min_t, const shared_t<T[]>& input, size_t elements, Stream& stream);

    /// Returns the memory offset(s) of the first maximum value(s).
    /// \tparam T               Any data type.
    /// \tparam U               (u)int32_t or (u)int64_t.
    /// \param[in] input        On the \b device. Input array.
    /// \param stride           Rightmost stride of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] offsets     On the \b device or \b host. Memory offset(s).
    /// \param batch            Whether \p input should be segmented by its outermost dimension.
    ///                         If true, the offset of maximum value in each batch is returned and
    ///                         these offsets are relative to the beginning of the batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note One can retrieve the multi-dimensional indexes from the offset using noa::indexing::indexes(...).
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_find_v<T, U>>>
    void find(noa::math::max_t, const shared_t<T[]>& input, size4_t stride, size4_t shape,
              const shared_t<U[]>& offsets, bool batch, Stream& stream);

    /// Returns the index of the maximum value.
    /// \tparam T               Any data type.
    /// \tparam offset_t        (u)int32_t or (u)int64_t.
    /// \param[in] input        On the \b device. Input array.
    /// \param elements         Number of elements in \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<typename offset_t = size_t, typename T, typename = std::enable_if_t<details::is_valid_find_v<T, offset_t>>>
    offset_t find(noa::math::max_t, const shared_t<T[]>& input, size_t elements, Stream& stream);
}
