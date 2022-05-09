#pragma once

#include "noa/unified/Array.h"

namespace noa::memory {
    /// Sets an array with a given value.
    /// \tparam T           Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param value        The value to assign.
    template<typename T>
    void fill(const Array<T>& output, T value);

    /// Returns an array filled with a given value.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param value    The value to assign.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> fill(size4_t shape, T value, ArrayOption option = {});

    /// Returns an array filled with zeros.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> zeros(size4_t shape, ArrayOption option = {});

    /// Returns an array filled with ones.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> ones(size4_t shape, ArrayOption option = {});

    /// Returns an uninitialized array.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> empty(size4_t shape, ArrayOption option = {});

    /// Returns an uninitialized contiguous array with the
    /// same shape and options as \p array.
    template<typename T>
    Array<T> like(const Array<T>& array);
}

namespace noa::memory {
    /// Returns evenly spaced values within a given interval.
    /// \tparam T           Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param start        Start of interval.
    /// \param step         Spacing between values.
    /// \note Depending on the current stream, this function may be asynchronous and may return before completion.
    template<typename T>
    void arange(const Array<T>& output, T start = T{0}, T step = T{1});

    /// Returns an array with evenly spaced values within a given interval.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param start    Start of interval.
    /// \param step     Spacing between values.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> arange(size4_t shape, T start = T{0}, T step = T{1}, ArrayOption option = {});

    /// Returns an array with evenly spaced values within a given interval.
    /// \tparam T       Any data type.
    /// \param shape    Number of elements.
    /// \param start    Start of interval.
    /// \param step     Spacing between values.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> arange(size_t elements, T start = T{0}, T step = T{1}, ArrayOption option = {});
}

namespace noa::memory {
    /// Returns evenly spaced values within a given interval.
    /// \tparam T           Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param start        Start of interval.
    /// \param stop         The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint     Whether the stop is the last simple. Otherwise, it is not included.
    template<typename T>
    void linspace(const Array<T>& output, T start, T stop, bool endpoint = true);

    /// Returns an array with evenly spaced values within a given interval.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param start    Start of interval.
    /// \param stop     The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint Whether the stop is the last simple. Otherwise, it is not included.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> linspace(size4_t shape, T start, T stop, bool endpoint = true, ArrayOption option = {});

    /// Returns an array with evenly spaced values within a given interval.
    /// \tparam T       Any data type.
    /// \param shape    Number of elements.
    /// \param start    Start of interval.
    /// \param stop     The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint Whether the stop is the last simple. Otherwise, it is not included.
    /// \param option   Options of the created array.
    template<typename T>
    Array<T> linspace(size_t elements, T start, T stop, bool endpoint = true, ArrayOption option = {});
}

#define NOA_UNIFIED_FACTORY_
#include "noa/unified/memory/Factory.inl"
#undef NOA_UNIFIED_FACTORY_
