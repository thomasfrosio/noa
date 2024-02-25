#pragma once

#include "noa/core/Arange.hpp"
#include "noa/core/Iota.hpp"
#include "noa/core/Linspace.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa {
    /// Sets an array with a given value.
    /// \tparam Value       Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param value        The value to assign.
    template<typename Output, typename Value>
    requires (nt::is_varray_v<Output> and std::is_same_v<nt::value_type_t<Output>, Value>)
    void fill(const Output& output, Value value) {
        check(not output.is_empty(), "Empty array detected");
        ewise({}, forward_as_tuple(output), Fill{static_cast<nt::value_type_t<Output>>(value)});
    }

    /// Returns an array filled with a given value.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param value    The value to assign.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> fill(const Shape4<i64>& shape, Value value, ArrayOption option = {}) {
        if constexpr (nt::is_numeric_v<Value> or nt::is_vec_v<Value> or nt::is_mat_v<Value>) {
            if (value == Value{0} and option.device.is_cpu() and
                (not Device::is_any(DeviceType::GPU) or (option.allocator.resource() == MemoryResource::DEFAULT or
                                                         option.allocator.resource() == MemoryResource::DEFAULT_ASYNC or
                                                         option.allocator.resource() == MemoryResource::PITCHED))) {
                return Array<Value>(noa::cpu::AllocatorHeap<Value>::calloc(shape.elements()),
                                    shape, shape.strides(), option);
            }
        }
        Array<Value> out(shape, option);
        fill(out, value);
        return out;
    }

    /// Returns an array filled with a given value.
    template<typename Value>
    [[nodiscard]] Array<Value> fill(i64 elements, Value value, ArrayOption option = {}) {
        return fill(Shape4<i64>{1, 1, 1, elements}, value, option);
    }

    /// Returns an array filled with zeros.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> zeros(const Shape4<i64>& shape, ArrayOption option = {}) {
        return fill(shape, Value{0}, option);
    }

    /// Returns an array filled with zeros.
    template<typename Value>
    [[nodiscard]] Array<Value> zeros(i64 elements, ArrayOption option = {}) {
        return fill(elements, Value{0}, option);
    }

    /// Returns an array filled with ones.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> ones(const Shape4<i64>& shape, ArrayOption option = {}) {
        return fill(shape, Value{1}, option);
    }

    /// Returns an array filled with ones.
    template<typename Value>
    [[nodiscard]] Array<Value> ones(i64 elements, ArrayOption option = {}) {
        return fill(elements, Value{1}, option);
    }

    /// Returns an uninitialized array.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> empty(const Shape4<i64>& shape, ArrayOption option = {}) {
        return Array<Value>(shape, option);
    }

    /// Returns an uninitialized array.
    template<typename Value>
    [[nodiscard]] Array<Value> empty(i64 elements, ArrayOption option = {}) {
        return Array<Value>(elements, option);
    }

    /// Returns an uninitialized contiguous array with the same shape and options as \p array.
    /// The value type can be set explicitly. By default, it is set to the mutable value type of \p array.
    template<typename Value = Empty, typename Input>
    [[nodiscard]] auto like(const Input& array) {
        using value_t = std::conditional_t<std::is_empty_v<Value>, nt::mutable_value_type_t<Input>, Value>;
        return Array<value_t>(array.shape(), array.options());
    }
}

namespace noa {
    /// Returns evenly spaced values within a given interval, in the BDHW order.
    /// \tparam Value       Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param start        Start of interval.
    /// \param step         Spacing between values.
    template<typename Output, typename Value = nt::value_type_t<Output>>
    requires nt::is_varray_of_any_v<Output, Value>
    void arange(const Output& output, Value start = Value{0}, Value step = Value{1}) {
        check(not output.is_empty(), "Empty array detected");
        if (output.are_contiguous())
            iwise(output.shape(), output.device(), Arange4d(output.accessor(), output.shape(), start, step));
        else {
            const auto n_elements = output.elements();
            iwise(Shape{n_elements}, output.device(), Arange1d(output.accessor_contiguous_1d(), start, step));
        }
    }

    /// Returns an array with evenly spaced values within a given interval, in the BDHW order.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param start    Start of interval.
    /// \param step     Spacing between values.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> arange(
            const Shape4<i64>& shape,
            Value start = Value{0}, Value step = Value{1},
            ArrayOption option = {}
    ) {
        Array<Value> out(shape, option);
        arange(out, start, step);
        return out;
    }

    /// Returns an array with evenly spaced values within a given interval, in the BDHW order.
    /// \tparam Value   Any data type.
    /// \param shape    Number of elements.
    /// \param start    Start of interval.
    /// \param step     Spacing between values.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> arange(
            i64 elements,
            Value start = Value{0}, Value step = Value{1},
            ArrayOption option = {}
    ) {
        Array<Value> out(elements, option);
        arange(out, start, step);
        return out;
    }
}

namespace noa {
    /// Returns evenly spaced values within a given interval, in the BDHW order.
    /// \tparam Value       Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param start        Start of interval.
    /// \param stop         The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint     Whether the stop is the last simple. Otherwise, it is not included.
    template<typename Output, typename Value, typename = std::enable_if_t<
             nt::is_varray_of_any_v<Output, Value>>>
    Value linspace(const Output& output, Value start, Value stop, bool endpoint = true) {
        check(not output.is_empty(), "Empty array detected");

        const auto n_elements = output.elements();
        const auto linspace = Linspace<Value, i64>::from_range(start, stop, n_elements, endpoint);

        if (output.are_contiguous()) {
            iwise(output.shape(), output.device(),
                  Linspace4d(output.accessor(), output.shape(), linspace));
        } else {

            iwise(Shape{n_elements}, output.device(),
                  Linspace1d(output.accessor_contiguous_1d(), linspace));
        }

        return linspace.step;
    }

    /// Returns an array with evenly spaced values within a given interval, in the BDHW order.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param start    Start of interval.
    /// \param stop     The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint Whether the stop is the last sample. Otherwise, it is not included.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> linspace(
            const Shape4<i64>& shape,
            Value start, Value stop,
            bool endpoint = true,
            ArrayOption option = {}
    ) {
        Array<Value> out(shape, option);
        linspace(out, start, stop, endpoint);
        return out;
    }

    /// Returns an array with evenly spaced values within a given interval, in the BDHW order.
    /// \tparam Value   Any data type.
    /// \param shape    Number of elements.
    /// \param start    Start of interval.
    /// \param stop     The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint Whether the stop is the last simple. Otherwise, it is not included.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> linspace(
            i64 elements, Value start, Value stop,
            bool endpoint = true,
            ArrayOption option = {}
    ) {
        Array<Value> out(elements, option);
        linspace(out, start, stop, endpoint);
        return out;
    }
}

namespace noa {
    /// Returns a tiled sequence [0, elements), in the rightmost order.
    /// \param[out] output  Array with the tiled sequence.
    /// \param tile         Tile shape in each dimension.
    ///                     If the tile is equal to the shape of \p output,
    ///                     this is equivalent to `arange` with a start of 0 and step of 1.
    template<typename Output, typename = std::enable_if_t<nt::is_varray_v<Output>>>
    void iota(const Output& output, const Vec4<i64>& tile) {
        check(not output.is_empty(), "Empty array detected");

        if (output.are_contiguous()) {
            iwise(output.shape(), output.device(),
                  Iota4d(output.accessor(), output.shape(), tile));
        } else {
            const auto n_elements = output.elements();
            iwise(Shape{n_elements}, output.device(),
                  Iota1d(output.accessor_contiguous_1d(), tile));
        }
    }

    /// Returns a tiled sequence [0, elements), in the rightmost order.
    /// \tparam Value   Any restricted scalar.
    /// \param shape    Shape of the array.
    /// \param tile     Tile shape in each dimension. If the tile is equal to \p shape,
    ///                 this is equivalent to `arange` with a start of 0 and step of 1.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> iota(const Shape4<i64>& shape, const Vec4<i64>& tile, ArrayOption option = {}) {
        Array<Value> out(shape, option);
        iota(out, tile);
        return out;
    }

    /// Returns a 1D tiled sequence [0, elements).
    /// \tparam Value   Any restricted scalar.
    /// \param tile     Tile size. If the tile is equal to \p elements,
    ///                 this is equivalent to `arange` with a start of 0 and step of 1.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> iota(i64 elements, i64 tile, ArrayOption option = {}) {
        Array<Value> out(elements, option);
        iota(out, Vec4<i64>{1, 1, 1, tile});
        return out;
    }
}
