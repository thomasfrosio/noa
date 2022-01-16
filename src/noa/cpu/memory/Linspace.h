#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    /// Returns evenly spaced values within a given interval.
    /// \tparam T       Any floating-point or complex type.
    /// \param[in] src  On the \b host. Array with evenly spaced values.
    /// \param pitch    Pitch, in elements, of \p src.
    /// \param elements Number of elements to set.
    /// \param batches  Number of batches.
    /// \param start    Start of interval.
    /// \param stop     The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint Whether the stop is the last simple. Otherwise, it is not included.
    /// \return         Size of spacing between samples.
    template<typename T>
    NOA_HOST T linspace(T* src, size_t pitch, size_t elements, size_t batches,
                        T start, T stop, bool endpoint = true) {
        NOA_PROFILE_FUNCTION();
        if (elements <= 1) {
            if (elements)
                *src = start;
            return T(0);
        }
        const size_t count = elements - static_cast<size_t>(endpoint);
        const T delta = stop - start;
        const T step = delta / static_cast<T>(count);
        for (size_t batch = 0; batch < batches; ++batch) {
            T* i_src = src + batch * pitch;
            for (size_t i = 0; i < count; ++i)
                i_src[i] = start + static_cast<T>(i) * step;
            if (endpoint)
                i_src[elements - 1] = stop;
        }
        return step;
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T       Any floating-point or complex type.
    /// \param src      On the \b host. Array with evenly spaced values.
    /// \param pitch    Pitch, in elements, of \p src.
    /// \param shape    Physical {fast,medium,slow} shape of \p src.
    /// \param batches  Number of batches.
    /// \param start    Start of interval.
    /// \param stop     The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint Whether the stop is the last simple. Otherwise, it is not included.
    /// \return         Size of spacing between samples.
    template<typename T>
    NOA_HOST T linspace(T* src, size3_t pitch, size3_t shape, size_t batches,
                        T start, T stop, bool endpoint = true) {
        NOA_PROFILE_FUNCTION();
        const size_t elements = noa::elements(shape);
        const size_t offset = noa::elements(pitch);
        if (elements <= 1) {
            if (elements)
                for (size_t batch = 0; batch < batches; ++batch)
                    src[batch * offset] = start;
            return T(0);
        }
        const size_t count = elements - static_cast<size_t>(endpoint);
        const T delta = stop - start;
        const T step = delta / static_cast<T>(count);
        for (size_t batch = 0; batch < batches; ++batch) {
            T* i_src = src + batch * offset;
            size_t i = 0;
            for (size_t z = 0; z < shape.z; ++z)
                for (size_t y = 0; y < shape.y; ++y)
                    for (size_t x = 0; x < shape.x; ++x, ++i)
                        i_src[index(x, y, z, pitch)] = start + static_cast<T>(i) * step;
            if (endpoint)
                i_src[index(shape - 1, pitch)] = stop;
        }
        return step;
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param[in] src          On the \b host. Array with evenly spaced values.
    /// \param pitch            Pitch, in elements, of \p src.
    /// \param elements         Number of elements to set.
    /// \param batches          Number of batches.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint         Whether the stop is the last simple. Otherwise, it is not included.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void linspace(T* src, size_t pitch, size_t elements, size_t batches,
                           T start, T stop, bool endpoint, Stream& stream) {
        stream.enqueue([=]() {
            linspace(src, pitch, elements, batches, start, stop, endpoint);
        });
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param[in] src          On the \b host. Array with evenly spaced values.
    /// \param pitch            Pitch, in elements, of \p src.
    /// \param elements         Number of elements to set.
    /// \param batches          Number of batches.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void linspace(T* src, size_t pitch, size_t elements, size_t batches,
                           T start, T stop, Stream& stream) {
        linspace(src, pitch, elements, batches, start, stop, true, stream);
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param src              On the \b host. Array with evenly spaced values.
    /// \param pitch            Pitch, in elements, of \p src.
    /// \param shape            Physical {fast,medium,slow} shape of \p src.
    /// \param batches          Number of batches.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint         Whether the stop is the last simple. Otherwise, it is not included.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void linspace(T* src, size3_t pitch, size3_t shape, size_t batches,
                           T start, T stop, bool endpoint, Stream& stream) {
        stream.enqueue([=]() {
            linspace(src, pitch, shape, batches, start, stop, endpoint);
        });
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any floating-point or complex type.
    /// \param src              On the \b host. Array with evenly spaced values.
    /// \param pitch            Pitch, in elements, of \p src.
    /// \param shape            Physical {fast,medium,slow} shape of \p src.
    /// \param batches          Number of batches.
    /// \param start            Start of interval.
    /// \param stop             The end value of the sequence.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void linspace(T* src, size3_t pitch, size3_t shape, size_t batches,
                           T start, T stop, Stream& stream) {
        linspace(src, pitch, shape, batches, start, stop, true, stream);
    }
}
