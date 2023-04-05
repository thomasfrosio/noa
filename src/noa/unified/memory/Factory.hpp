#pragma once

#include "noa/cpu/memory/Arange.hpp"
#include "noa/cpu/memory/Linspace.hpp"
#include "noa/cpu/memory/Iota.hpp"
#include "noa/cpu/memory/Set.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Arange.hpp"
#include "noa/gpu/cuda/memory/Linspace.hpp"
#include "noa/gpu/cuda/memory/Iota.hpp"
#include "noa/gpu/cuda/memory/Set.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::memory {
    /// Sets an array with a given value.
    /// \tparam Value       Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param value        The value to assign.
    template<typename Output, typename Value, typename = std::enable_if_t<
             noa::traits::is_array_or_view_v<Output> &&
             std::is_same_v<noa::traits::value_type_t<Output>, Value>>>
    void fill(const Output& output, Value value) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=](){
                cpu::memory::set(output.get(), output.strides(), output.shape(), value, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_set_v<Value>) {
                auto& cuda_stream = stream.cuda();
                cuda::memory::set(output.get(), output.strides(), output.shape(), value, cuda_stream);
                cuda_stream.enqueue_attach(output.share());
            } else {
                NOA_THROW("The CUDA backend does not support this type ({})", noa::string::human<Value>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns an array filled with a given value.
    /// \tparam Value   Any data type.
    /// \param shape    Shape of the array.
    /// \param value    The value to assign.
    /// \param option   Options of the created array.
    template<typename Value>
    [[nodiscard]] Array<Value> fill(const Shape4<i64>& shape, Value value, ArrayOption option = {}) {
        // Try to shortcut with calloc().
        if constexpr (noa::traits::is_numeric_v<Value> ||
                      noa::traits::is_vecX_v<Value> ||
                      noa::traits::is_matXX_v<Value>) {
            if (value == Value{0} && option.device().is_cpu() &&
                (!Device::is_any(DeviceType::GPU) || (option.allocator() == Allocator::DEFAULT ||
                                                      option.allocator() == Allocator::DEFAULT_ASYNC ||
                                                      option.allocator() == Allocator::PITCHED))) {
                return Array<Value>(cpu::memory::PtrHost<Value>::calloc(shape.elements()),
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
    template<typename Input>
    [[nodiscard]] auto like(const Input& array) {
        using value_t = noa::traits::mutable_value_type_t<Input>;
        return Array<value_t>(array.shape(), array.options());
    }
}

namespace noa::memory {
    /// Returns evenly spaced values within a given interval, in the BDHW order.
    /// \tparam Value       Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param start        Start of interval.
    /// \param step         Spacing between values.
    template<typename Output, typename Value = noa::traits::value_type_t<Output>,
             typename = std::enable_if_t<noa::traits::is_array_or_view_of_any_v<Output, Value>>>
    void arange(const Output& output, Value start = Value{0}, Value step = Value{1}) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=](){
                cpu::memory::arange(output.get(), output.strides(), output.shape(), start, step, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (traits::is_restricted_numeric_v<Value>) {
                auto& cuda_stream = stream.cuda();
                cuda::memory::arange(output.get(), output.strides(), output.shape(), start, step, cuda_stream);
                cuda_stream.enqueue_attach(output.share());
            } else {
                NOA_THROW("The CUDA backend does not support this type ({})", noa::string::human<Value>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
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
            ArrayOption option = {}) {
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
            ArrayOption option = {}) {
        Array<Value> out(elements, option);
        arange(out, start, step);
        return out;
    }
}

namespace noa::memory {
    /// Returns evenly spaced values within a given interval, in the BDHW order.
    /// \tparam Value       Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param start        Start of interval.
    /// \param stop         The end value of the sequence, unless \p endpoint is false.
    /// \param endpoint     Whether the stop is the last simple. Otherwise, it is not included.
    template<typename Output, typename Value, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_any_v<Output, Value>>>
    Value linspace(const Output& output, Value start, Value stop, bool endpoint = true) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=](){
                cpu::memory::linspace(output.get(), output.strides(), output.shape(),
                                      start, stop, endpoint, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (traits::is_restricted_numeric_v<Value>) {
                auto& cuda_stream = stream.cuda();
                cuda::memory::linspace(output.get(), output.strides(), output.shape(),
                                       start, stop, endpoint, cuda_stream);
                cuda_stream.enqueue_attach(output.share());
            } else {
                NOA_THROW("The CUDA backend does not support this type ({})", noa::string::human<Value>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        auto[a_, b_, step] = noa::algorithm::memory::linspace_step(output.shape().elements(), start, stop, endpoint);
        return step;
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
            const Shape4<i64>& shape, Value start, Value stop,
            bool endpoint = true, ArrayOption option = {}) {
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
            bool endpoint = true, ArrayOption option = {}) {
        Array<Value> out(elements, option);
        linspace(out, start, stop, endpoint);
        return out;
    }
}

namespace noa::memory {
    /// Returns a tiled sequence [0, elements), in the rightmost order.
    /// \param[out] output  Array with the tiled sequence.
    /// \param tile         Tile shape in each dimension.
    ///                     If the tile is equal to the shape of \p output,
    ///                     this is equivalent to `arange` with a start of 0 and step of 1.
    template<typename Output, typename = std::enable_if_t<noa::traits::is_array_or_view_v<Output>>>
    void iota(const Output& output, const Vec4<i64>& tile) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=](){
                cpu::memory::iota(output.get(), output.strides(), output.shape(), tile, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (noa::traits::is_array_or_view_of_numeric_v<Output>) {
                auto& cuda_stream = stream.cuda();
                cuda::memory::iota(output.get(), output.strides(), output.shape(), tile, cuda_stream);
                cuda_stream.enqueue_attach(output.share());
            } else {
                NOA_THROW("The CUDA backend does not support this type ({})",
                          noa::string::human<noa::traits::value_type_t<Output>>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
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
