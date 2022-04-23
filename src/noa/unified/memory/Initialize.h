#pragma once

#include "noa/cpu/memory/Arange.h"
#include "noa/cpu/memory/Linspace.h"
#include "noa/cpu/memory/Set.h"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Arange.h"
#include "noa/gpu/cuda/memory/Linspace.h"
#include "noa/gpu/cuda/memory/Set.h"
#endif

#include "noa/unified/Array.h"

namespace noa::memory {
    /// Sets an array with a given value.
    /// \tparam T           Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param value        The value to assign.
    template<typename T>
    void fill(const Array<T>& output, T value) {
        NOA_PROFILE_FUNCTION();
        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::set(output.share(), output.stride(), output.shape(), value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::set(output.share(), output.stride(), output.shape(), value, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns an array filled with a given value.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param value    The value to assign.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> fill(size4_t shape, T value, ArrayOption option = {}) {
        NOA_PROFILE_FUNCTION();
        Array<T> out{shape, option};
        fill(out, value);
        return out;
    }

    /// Returns an array filled with zeros.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> zeros(size4_t shape, ArrayOption option = {}) {
        return fill(shape, T{0}, option); // TODO add calloc
    }

    /// Returns an array filled with ones.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> ones(size4_t shape, ArrayOption option = {}) {
        return fill(shape, T{1}, option);
    }

    /// Returns an uninitialized array.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> empty(size4_t shape, ArrayOption option = {}) {
        NOA_PROFILE_FUNCTION();
        return Array<T>{shape, option};
    }
}

namespace noa::memory {
    /// Returns evenly spaced values within a given interval.
    /// \tparam T           Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param start        Start of interval.
    /// \param step         Spacing between values.
    /// \note Depending on the current stream, this function may be asynchronous and may return before completion.
    template<typename T>
    void arange(const Array<T>& output, T start = T(0), T step = T(1)) {
        NOA_PROFILE_FUNCTION();
        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::arange(output.share(), output.stride(), output.shape(), start, step, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::arange(output.share(), output.stride(), output.shape(), start, step, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns an array with evenly spaced values within a given interval.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param start    Start of interval.
    /// \param step     Spacing between values.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> arange(size4_t shape, T start = T(0), T step = T(1), ArrayOption option = {}) {
        NOA_PROFILE_FUNCTION();
        Array<T> out{shape, option};
        arange(out, start, step);
        return out;
    }

    /// Returns an array with evenly spaced values within a given interval.
    /// \tparam T       Any data type.
    /// \param shape    Number of elements.
    /// \param start    Start of interval.
    /// \param step     Spacing between values.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> arange(size_t elements, T start = T(0), T step = T(1), ArrayOption option = {}) {
        NOA_PROFILE_FUNCTION();
        Array<T> out{elements, option};
        arange(out, start, step);
        return out;
    }
}

namespace noa::memory {
    /// Returns evenly spaced values within a given interval.
    /// \tparam T           Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param start        Start of interval.
    /// \param stop         The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint     Whether the stop is the last simple. Otherwise, it is not included.
    template<typename T>
    void linspace(const Array<T>& output, T start, T stop, bool endpoint = true) {
        NOA_PROFILE_FUNCTION();
        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::linspace(output.share(), output.stride(), output.shape(),
                                  start, stop, endpoint, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::linspace(output.share(), output.stride(), output.shape(),
                                   start, stop, endpoint, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns an array with evenly spaced values within a given interval.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param start    Start of interval.
    /// \param stop     The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint Whether the stop is the last simple. Otherwise, it is not included.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> linspace(size4_t shape, T start, T stop, bool endpoint = true, ArrayOption option = {}) {
        NOA_PROFILE_FUNCTION();
        Array<T> out{shape, option};
        linspace(out, start, stop, endpoint);
        return out;
    }

    /// Returns an array with evenly spaced values within a given interval.
    /// \tparam T       Any data type.
    /// \param shape    Number of elements.
    /// \param start    Start of interval.
    /// \param stop     The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint Whether the stop is the last simple. Otherwise, it is not included.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> linspace(size_t elements, T start, T stop, bool endpoint = true, ArrayOption option = {}) {
        NOA_PROFILE_FUNCTION();
        Array<T> out{elements, option};
        linspace(out, start, stop, endpoint);
        return out;
    }
}
