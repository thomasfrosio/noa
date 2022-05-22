#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    /// Returns evenly spaced values within a given interval.
    /// \tparam T       Any floating-point or complex type.
    /// \param[in] src  On the \b host. Array with evenly spaced values.
    /// \param elements Number of elements to set.
    /// \param start    Start of interval.
    /// \param stop     The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint Whether the stop is the last sample. Otherwise, it is not included.
    /// \return         Size of spacing between samples.
    template<typename T>
    T linspace(T* src, size_t elements, T start, T stop, bool endpoint = true) {
        if (elements <= 1) {
            if (elements)
                *src = start;
            return T(0);
        }
        const size_t count = elements - static_cast<size_t>(endpoint);
        const T delta = stop - start;
        const T step = delta / static_cast<T>(count);
        for (size_t i = 0; i < count; ++i)
            src[i] = start + static_cast<T>(i) * step;
        if (endpoint)
            src[elements - 1] = stop;
        return step;
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T       Any floating-point or complex type.
    /// \param src      On the \b host. Array with evenly spaced values.
    /// \param stride   Rightmost strides, in elements, of \p src.
    /// \param shape    Rightmost shape of \p src.
    /// \param start    Start of interval.
    /// \param stop     The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint Whether the stop is the last sample. Otherwise, it is not included.
    /// \return         Size of spacing between samples.
    template<typename T>
    T linspace(T* src, size4_t stride, size4_t shape,
               T start, T stop, bool endpoint = true) {
        if (all(indexing::isContiguous(stride, shape)))
            return linspace(src, shape.elements(), start, stop, endpoint);

        const size_t elements = shape.elements();
        if (elements <= 1) {
            if (elements)
                *src = start;
            return T(0);
        }
        const size_t count = elements - static_cast<size_t>(endpoint);
        const T delta = stop - start;
        const T step = delta / static_cast<T>(count);
        size_t inc = 0;
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                for (size_t k = 0; k < shape[2]; ++k)
                    for (size_t l = 0; l < shape[3]; ++l, ++inc)
                        src[indexing::at(i, j, k, l, stride)] = start + static_cast<T>(inc) * step;
        if (endpoint)
            src[indexing::at(shape - 1, stride)] = stop;
        return step;
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param[in] src          On the \b host. Array with evenly spaced values.
    /// \param elements         Number of elements to set.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint         Whether the stop is the last sample. Otherwise, it is not included.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void linspace(const shared_t<T[]>& src, size_t elements,
                         T start, T stop, bool endpoint, Stream& stream) {
        stream.enqueue([=]() {
            linspace(src.get(), elements, start, stop, endpoint);
        });
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param[in] src          On the \b host. Array with evenly spaced values.
    /// \param elements         Number of elements to set.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void linspace(const shared_t<T[]>& src, size_t elements,
                         T start, T stop, Stream& stream) {
        linspace(src, elements, start, stop, true, stream);
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param src              On the \b host. Array with evenly spaced values.
    /// \param stride           Rightmost strides, in elements, of \p src.
    /// \param shape            Rightmost shape of \p src.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint         Whether the stop is the last sample. Otherwise, it is not included.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void linspace(const shared_t<T[]>& src, size4_t stride, size4_t shape,
                         T start, T stop, bool endpoint, Stream& stream) {
        stream.enqueue([=]() {
            linspace(src.get(), stride, shape, start, stop, endpoint);
        });
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param src              On the \b host. Array with evenly spaced values.
    /// \param stride           Rightmost strides, in elements, of \p src.
    /// \param shape            Rightmost shape of \p src.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void linspace(const shared_t<T[]>& src, size4_t stride, size4_t shape,
                         T start, T stop, Stream& stream) {
        linspace(src, stride, shape, start, stop, true, stream);
    }
}
