#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    /// Returns evenly spaced values within a given interval.
    /// \tparam T           Any built-in or complex type.
    /// \param[in,out] src  On the \b host. Array with evenly spaced values.
    /// \param elements     Number of elements to set.
    /// \param start        Start of interval.
    /// \param step         Spacing between values.
    template<typename T>
    NOA_HOST void arange(T* src, size_t elements, T start = T(0), T step = T(1)) {
        NOA_PROFILE_FUNCTION();
        T value = start;
        for (size_t i = 0; i < elements; ++i, value += step)
            src[i] = value;
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T           Any built-in or complex type.
    /// \param[in,out] src  On the \b host. Array with evenly spaced values.
    /// \param stride       Rightmost strides, in elements, of \p src.
    /// \param shape        Rightmost shape of \p src.
    /// \param start        Start of interval.
    /// \param step         Spacing between values.
    template<typename T>
    NOA_HOST void arange(T* src, size4_t stride, size4_t shape, T start = T(0), T step = T(1)) {
        NOA_PROFILE_FUNCTION();
        T value = start;
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j)
                for (size_t k = 0; k < shape[2]; ++k)
                    for (size_t l = 0; l < shape[3]; ++l, value += step)
                        src[at(i, j, k, l, stride)] = value;
        }
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any built-in or complex type.
    /// \param[in,out] src      On the \b host. Array with evenly spaced values.
    /// \param elements         Number of elements to set.
    /// \param start            Start of interval.
    /// \param step             Spacing between values.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void arange(T* src, size_t elements, T start, T step, Stream& stream) {
        stream.enqueue([=]() {
            arange(src, elements, start, step);
        });
    }

    /// Returns evenly spaced values within a given interval, starting from 0, with a step of 1.
    /// \tparam T               Any built-in or complex type.
    /// \param[in,out] src      On the \b host. Array with evenly spaced values.
    /// \param elements         Number of elements to set.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void arange(T* src, size_t elements, Stream& stream) {
        arange(src, elements, T(0), T(1), stream);
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any built-in or complex type.
    /// \param[in,out] src      On the \b host. Array with evenly spaced values.
    /// \param stride           Rightmost strides, in elements, of \p src.
    /// \param shape            Rightmost shape of \p src.
    /// \param start            Start of interval.
    /// \param step             Spacing between values.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void arange(T* src, size4_t stride, size4_t shape, T start, T step, Stream& stream) {
        stream.enqueue([=]() {
            arange(src, stride, shape, start, step);
        });
    }

    /// Returns evenly spaced values within a given interval, starting from 0, with a step of 1.
    /// \tparam T               Any built-in or complex type.
    /// \param[in,out] src      On the \b host. Array with evenly spaced values.
    /// \param stride           Rightmost strides, in elements, of \p src.
    /// \param shape            Rightmost shape of \p src.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void arange(T* src, size4_t stride, size4_t shape, Stream& stream) {
        arange(src, stride, shape, T(0), T(1), stream);
    }
}
