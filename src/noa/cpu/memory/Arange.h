#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    /// Returns evenly spaced values within a given interval.
    /// \tparam T       Any built-in or complex type.
    /// \param[in] src  On the \b host. Array with evenly spaced values.
    /// \param pitch    Pitch, in elements, of \p src.
    /// \param elements Number of elements to set.
    /// \param batches  Number of batches.
    /// \param start    Start of interval.
    /// \param step     Spacing between values.
    template<typename T>
    NOA_HOST void arange(T* src, size_t pitch, size_t elements, size_t batches, T start = T(0), T step = T(1)) {
        NOA_PROFILE_FUNCTION();
        for (size_t batch = 0; batch < batches; ++batch) {
            T* i_src = src + batch * pitch;
            T value = start;
            for (size_t i = 0; i < elements; ++i, value += step)
                i_src[i] = value;
        }
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T       Any built-in or complex type.
    /// \param src      On the \b host. Array with evenly spaced values.
    /// \param pitch    Pitch, in elements, of \p src.
    /// \param shape    Physical {fast,medium,slow} shape of \p src.
    /// \param batches  Number of batches.
    /// \param start    Start of interval.
    /// \param step     Spacing between values.
    template<typename T>
    NOA_HOST void arange(T* src, size3_t pitch, size3_t shape, size_t batches, T start = T(0), T step = T(1)) {
        NOA_PROFILE_FUNCTION();
        for (size_t batch = 0; batch < batches; ++batch) {
            T* i_src = src + batch * elements(pitch);
            T value = start;
            for (size_t z = 0; z < shape.z; ++z)
                for (size_t y = 0; y < shape.y; ++y)
                    for (size_t x = 0; x < shape.x; ++x, value += step)
                        i_src[index(x, y, z, pitch)] = value;
        }
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any built-in or complex type.
    /// \param[in] src          On the \b host. Array with evenly spaced values.
    /// \param pitch            Pitch, in elements, of \p src.
    /// \param elements         Number of elements to set.
    /// \param batches          Number of batches.
    /// \param start            Start of interval.
    /// \param step             Spacing between values.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void arange(T* src, size_t pitch, size_t elements, size_t batches, T start, T step, Stream& stream) {
        stream.enqueue([=](){
            arange(src, pitch, elements, batches, start, step);
        });
    }

    /// Returns evenly spaced values within a given interval, starting from 0, with a step of 1.
    /// \tparam T               Any built-in or complex type.
    /// \param[in] src          On the \b host. Array with evenly spaced values.
    /// \param pitch            Pitch, in elements, of \p src.
    /// \param elements         Number of elements to set.
    /// \param batches          Number of batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void arange(T* src, size_t pitch, size_t elements, size_t batches, Stream& stream) {
        arange(src, pitch, elements, batches, T(0), T(1), stream);
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any built-in or complex type.
    /// \param src              On the \b host. Array with evenly spaced values.
    /// \param pitch            Pitch, in elements, of \p src.
    /// \param shape            Physical {fast,medium,slow} shape of \p src.
    /// \param batches          Number of batches.
    /// \param start            Start of interval.
    /// \param step             Spacing between values.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void arange(T* src, size3_t pitch, size3_t shape, size_t batches, T start, T step, Stream& stream) {
        stream.enqueue([=](){
            arange(src, pitch, shape, batches, start, step);
        });
    }

    /// Returns evenly spaced values within a given interval, starting from 0, with a step of 1.
    /// \tparam T               Any built-in or complex type.
    /// \param src              On the \b host. Array with evenly spaced values.
    /// \param pitch            Pitch, in elements, of \p src.
    /// \param shape            Physical {fast,medium,slow} shape of \p src.
    /// \param batches          Number of batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void arange(T* src, size3_t pitch, size3_t shape, size_t batches, Stream& stream) {
        arange(src, pitch, shape, batches, T(0), T(1), stream);
    }
}
