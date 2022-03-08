#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory {
    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param[in] src          On the \b device. Array with evenly spaced values.
    /// \param elements         Number of elements to set.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint         Whether the stop is the last simple. Otherwise, it is not included.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void linspace(T* src, size_t elements, T start, T stop, bool endpoint, Stream& stream);

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param[in] src          On the \b device. Array with evenly spaced values.
    /// \param elements         Number of elements to set.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void linspace(T* src, size_t elements, T start, T stop, Stream& stream) {
        linspace(src, elements, start, stop, true, stream);
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param[in] src          On the \b device. Array with evenly spaced values.
    /// \param stride           Rightmost strides, in elements, of \p src.
    /// \param shape            Rightmost shape of \p src.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint         Whether the stop is the last simple. Otherwise, it is not included.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void linspace(T* src, size4_t stride, size4_t shape, T start, T stop, bool endpoint, Stream& stream);

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param[in] src          On the \b device. Array with evenly spaced values.
    /// \param stride           Rightmost strides, in elements, of \p src.
    /// \param shape            Rightmost shape of \p src.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void linspace(T* src, size4_t stride, size4_t shape, T start, T stop, Stream& stream) {
        linspace(src, stride, shape, start, stop, true, stream);
    }
}
