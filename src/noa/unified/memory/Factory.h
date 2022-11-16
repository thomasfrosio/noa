#pragma once

#include "noa/unified/Array.h"

namespace noa::memory {
    /// Sets an array with a given value.
    /// \tparam Value       Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param value        The value to assign.
    template<typename Value>
    void fill(const Array<Value>& output, Value value);

    /// Returns an array filled with a given value.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param value    The value to assign.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> fill(dim4_t shape, Value value, ArrayOption option = {});

    /// Returns an array filled with zeros.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> zeros(dim4_t shape, ArrayOption option = {});

    /// Returns an array filled with ones.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> ones(dim4_t shape, ArrayOption option = {});

    /// Returns an uninitialized array.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> empty(dim4_t shape, ArrayOption option = {});

    /// Returns an uninitialized contiguous array with the
    /// same shape and options as \p array.
    template<typename Value>
    [[nodiscard]] Array<Value> like(const Array<Value>& array);
}

namespace noa::memory {
    /// Returns evenly spaced values within a given interval, in the rightmost order.
    /// \tparam Value       Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param start        Start of interval.
    /// \param step         Spacing between values.
    template<typename Value>
    void arange(const Array<Value>& output, Value start = Value{0}, Value step = Value{1});

    /// Returns an array with evenly spaced values within a given interval, in the rightmost order.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param start    Start of interval.
    /// \param step     Spacing between values.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> arange(dim4_t shape, Value start = Value{0}, Value step = Value{1},
                                      ArrayOption option = {});

    /// Returns an array with evenly spaced values within a given interval, in the rightmost order.
    /// \tparam Value   Any data type.
    /// \param shape    Number of elements.
    /// \param start    Start of interval.
    /// \param step     Spacing between values.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> arange(dim_t elements, Value start = Value{0}, Value step = Value{1},
                                      ArrayOption option = {});
}

namespace noa::memory {
    /// Returns evenly spaced values within a given interval, in the rightmost order.
    /// \tparam Value       Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param start        Start of interval.
    /// \param stop         The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint     Whether the stop is the last simple. Otherwise, it is not included.
    template<typename Value>
    Value linspace(const Array<Value>& output, Value start, Value stop, bool endpoint = true);

    /// Returns an array with evenly spaced values within a given interval, in the rightmost order.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param start    Start of interval.
    /// \param stop     The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint Whether the stop is the last simple. Otherwise, it is not included.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> linspace(dim4_t shape, Value start, Value stop,
                                        bool endpoint = true, ArrayOption option = {});

    /// Returns an array with evenly spaced values within a given interval, in the rightmost order.
    /// \tparam Value   Any data type.
    /// \param shape    Number of elements.
    /// \param start    Start of interval.
    /// \param stop     The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint Whether the stop is the last simple. Otherwise, it is not included.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> linspace(dim_t elements, Value start, Value stop,
                                        bool endpoint = true, ArrayOption option = {});
}

namespace noa::memory {
    /// Returns a tiled sequence [0, elements), in the rightmost order.
    /// \tparam Value       Any restricted scalar.
    /// \param[out] output  Array with the tiled sequence.
    /// \param tile         Tile shape in each dimension.
    ///                     If the tile is equal to the shape of \p output,
    ///                     this is equivalent to `arange` with a start of 0 and step of 1.
    template<typename Value>
    void iota(const Array<Value>& output, dim4_t tile);

    /// Returns a tiled sequence [0, elements), in the rightmost order.
    /// \tparam Value   Any restricted scalar.
    /// \param shape    Shape of the array.
    /// \param tile     Tile shape in each dimension. If the tile is equal to \p shape,
    ///                 this is equivalent to `arange` with a start of 0 and step of 1.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> iota(dim4_t shape, dim4_t tile, ArrayOption option = {});

    /// Returns a 1D tiled sequence [0, elements).
    /// \tparam Value   Any restricted scalar.
    /// \param tile     Tile size. If the tile is equal to \p elements,
    ///                 this is equivalent to `arange` with a start of 0 and step of 1.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> iota(dim_t elements, dim_t tile, ArrayOption option = {});
}

#define NOA_UNIFIED_FACTORY_
#include "noa/unified/memory/Factory.inl"
#undef NOA_UNIFIED_FACTORY_
