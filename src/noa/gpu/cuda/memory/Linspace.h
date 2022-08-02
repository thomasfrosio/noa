#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory {
    template<typename T>
    std::tuple<size_t, T, T> linspaceStep(size_t elements, T start, T stop, bool endpoint = true) {
        const size_t count = elements - static_cast<size_t>(endpoint);
        const T delta = stop - start;
        const T step = delta / static_cast<T>(count);
        return {count, delta, step};
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param[in] src          On the \b device. Array with evenly spaced values.
    /// \param elements         Number of elements to set.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint         Whether the stop is the last simple. Otherwise, it is not included.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \return                 Size of spacing between samples.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T> && !traits::is_bool_v<T>>>
    T linspace(const shared_t<T[]>& src, size_t elements,
               T start, T stop, bool endpoint, Stream& stream);

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param[in] src          On the \b device. Array with evenly spaced values.
    /// \param elements         Number of elements to set.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \return                 Size of spacing between samples.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T> && !traits::is_bool_v<T>>>
    inline T linspace(const shared_t<T[]>& src, size_t elements,
                      T start, T stop, Stream& stream) {
        return linspace(src, elements, start, stop, true, stream);
    }

    /// Returns evenly spaced values within a given interval, in the rightmost order.
    /// \tparam T               Any floating-point or complex type.
    /// \param[in] src          On the \b device. Array with evenly spaced values.
    /// \param strides          BDHW strides, in elements, of \p src.
    /// \param shape            BDHW shape of \p src.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint         Whether the stop is the last simple. Otherwise, it is not included.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \return                 Size of spacing between samples.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T> && !traits::is_bool_v<T>>>
    T linspace(const shared_t<T[]>& src, size4_t strides, size4_t shape,
               T start, T stop, bool endpoint, Stream& stream);

    /// Returns evenly spaced values within a given interval, in the rightmost order.
    /// \tparam T               Any floating-point or complex type.
    /// \param[in] src          On the \b device. Array with evenly spaced values.
    /// \param strides          BDHW strides, in elements, of \p src.
    /// \param shape            BDHW shape of \p src.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \return                 Size of spacing between samples.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T> && !traits::is_bool_v<T>>>
    inline T linspace(const shared_t<T[]>& src, size4_t strides, size4_t shape,
                      T start, T stop, Stream& stream) {
        return linspace(src, strides, shape, start, stop, true, stream);
    }
}
