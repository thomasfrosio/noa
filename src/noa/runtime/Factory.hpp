#pragma once

#include "noa/runtime/core/Iwise.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/Ewise.hpp"
#include "noa/runtime/Iwise.hpp"

namespace noa {
    /// Sets an array with a given value.
    /// \param[out] output  Array with evenly spaced values.
    /// \param value        The value to assign.
    template<nt::writable_varray_decay Output>
    void fill(Output&& output, nt::mutable_value_type_t<Output> value) {
        #ifdef NOA_ENABLE_CUDA
        using value_t = nt::mutable_value_type_t<Output>;
        if constexpr (nt::trivial_zero<value_t>) {
            if (output.device().is_gpu() and output.is_contiguous() and value_t{} == value) {
                auto& cuda_stream = Stream::current(output.device()).gpu();
                noa::cuda::fill_with_zeroes(output.get(), output.ssize(), cuda_stream);
                return;
            }
        }
        #endif
        ewise({}, std::forward<Output>(output), Fill{value});
    }

    /// Returns an array filled with a given value.
    /// \tparam T       Any data type.
    /// \param shape    Shape of the array.
    /// \param value    The value to assign.
    /// \param option   Options of the created array.
    template<typename T>
    [[nodiscard]] auto fill(const Shape4& shape, T value, ArrayOption option = {}) -> Array<T> {
        if constexpr (nt::trivial_zero<T>) {
            if (value == T{} and option.device.is_cpu() and
                (not Device::is_any_gpu() or
                 option.allocator.is_any(Allocator::DEFAULT, Allocator::ASYNC, Allocator::PITCHED))) {
                return Array<T>(noa::cpu::AllocatorHeap::calloc<T>(shape.n_elements()),
                                shape, shape.strides(), option);
            }
        }
        auto out = Array<T>(shape, option);
        fill(out, value);
        return out;
    }

    /// Returns an array filled with a given value.
    template<typename T>
    [[nodiscard]] auto fill(isize elements, T value, ArrayOption option = {}) -> Array<T> {
        return fill(Shape4{1, 1, 1, elements}, value, option);
    }

    /// Returns an array filled with zeros.
    /// \tparam T       Any data type.
    /// \param shape    Shape of the array.
    /// \param option   Options of the created array.
    template<typename T>
    [[nodiscard]] auto zeros(const Shape4& shape, ArrayOption option = {}) -> Array<T> {
        return fill(shape, T{}, option);
    }

    /// Returns an array filled with zeros.
    template<typename T>
    [[nodiscard]] auto zeros(isize elements, ArrayOption option = {}) -> Array<T> {
        return fill(elements, T{}, option);
    }

    /// Returns an array filled with ones.
    /// \tparam T       Any data type.
    /// \param shape    Shape of the array.
    /// \param option   Options of the created array.
    template<typename T>
    [[nodiscard]] auto ones(const Shape4& shape, ArrayOption option = {}) -> Array<T> {
        return fill(shape, T{1}, option);
    }

    /// Returns an array filled with ones.
    template<typename T>
    [[nodiscard]] auto ones(isize elements, ArrayOption option = {}) -> Array<T> {
        return fill(elements, T{1}, option);
    }

    /// Returns an uninitialized array.
    /// \tparam T       Any data type.
    /// \param shape    Shape of the array.
    /// \param option   Options of the created array.
    template<typename T>
    [[nodiscard]] auto empty(const Shape4& shape, ArrayOption option = {}) -> Array<T> {
        return Array<T>(shape, option);
    }

    /// Returns an uninitialized array.
    template<typename T>
    [[nodiscard]] auto empty(isize elements, ArrayOption option = {}) -> Array<T> {
        return Array<T>(elements, option);
    }

    /// Returns an uninitialized contiguous array with the same shape and options as \p array.
    /// The value type can be set explicitly. By default, it is set to the mutable value type of \p array.
    template<typename T = Empty, nt::varray Input>
    [[nodiscard]] auto like(const Input& array) {
        using value_t = std::conditional_t<nt::empty<T>, nt::mutable_value_type_t<Input>, T>;
        return Array<value_t>(array.shape(), array.options());
    }
}

namespace noa {
    /// Returns evenly spaced values within a given interval, in the BDHW order.
    /// \tparam T           Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param params       Arange parameters.
    template<nt::writable_varray_decay Output, typename T = nt::value_type_t<Output>>
    void arange(Output&& output, Arange<T> params = {}) {
        check(not output.is_empty(), "Empty array detected");
        if (output.is_contiguous()) {
            auto accessor = nd::to_accessor_contiguous_1d(output);
            using op_t = nd::IwiseRange<1, decltype(accessor), isize, Arange<T>>;
            iwise(Shape{output.n_elements()}, output.device(),
                  op_t(accessor, Shape<isize, 1>{}, params),
                  std::forward<Output>(output));
        } else {
            auto accessor = nd::to_accessor(output);
            using op_t = nd::IwiseRange<4, decltype(accessor), isize, Arange<T>>;
            iwise(output.shape(), output.device(),
                  op_t(accessor, output.shape(), params),
                  std::forward<Output>(output));
        }
    }

    /// Returns an array with evenly spaced values within a given interval, in the BDHW order.
    /// \tparam T       Any data type.
    /// \param shape    Shape of the array.
    /// \param params   Arange parameters.
    /// \param option   Options of the created array.
    template<typename T = void, typename U = T>
    [[nodiscard]] auto arange(
        const Shape4& shape,
        Arange<U> params = Arange<U>{},
        ArrayOption option = {}
    ) {
        using type = std::conditional_t<std::is_void_v<T>, U, T>;
        auto out = Array<type>(shape, option);
        arange(out, params);
        return out;
    }

    /// Returns an array with evenly spaced values within a given interval, in the BDHW order.
    /// \tparam T           Any data type.
    /// \param n_elements   Number of elements.
    /// \param params       Arange parameters.
    /// \param option       Options of the created array.
    template<typename T = void, typename U = T>
    [[nodiscard]] auto arange(
        isize n_elements,
        Arange<U> params = Arange<U>{},
        ArrayOption option = {}
    ) {
        using type = std::conditional_t<std::is_void_v<T>, U, T>;
        auto out = Array<type>(n_elements, option);
        arange(out, params);
        return out;
    }
}

namespace noa {
    /// Returns evenly spaced values within a given interval, in the BDHW order.
    /// \tparam T           Any data type.
    /// \param[out] output  Array with evenly spaced values.
    /// \param params       Linspace parameters.
    template<nt::writable_varray_decay Output, typename T = nt::value_type_t<Output>>
    auto linspace(Output&& output, Linspace<T> params) {
        check(not output.is_empty(), "Empty array detected");

        const auto n_elements = output.n_elements();
        auto linspace = params.for_size(n_elements);

        if (output.is_contiguous()) {
            auto accessor = nd::to_accessor_contiguous_1d(output);
            using op_t = nd::IwiseRange<1, decltype(accessor), isize, decltype(linspace)>;
            iwise(Shape{n_elements}, output.device(),
                  op_t(accessor, Shape<isize, 1>{}, linspace),
                  std::forward<Output>(output));
        } else {
            auto accessor = nd::to_accessor(output);
            using op_t = nd::IwiseRange<4, decltype(accessor), isize, decltype(linspace)>;
            iwise(output.shape(), output.device(),
                  op_t(accessor, output.shape(), linspace),
                  std::forward<Output>(output));
        }
        return static_cast<nt::mutable_value_type_t<Output>>(linspace.step);
    }

    /// Returns an array with evenly spaced values within a given interval, in the BDHW order.
    /// \tparam T       Any data type.
    /// \param shape    Shape of the array.
    /// \param params   Linspace parameters.
    /// \param option   Options of the created array.
    template<typename T = void, typename U = T>
    [[nodiscard]] auto linspace(
        const Shape4& shape,
        Linspace<U> params,
        ArrayOption option = {}
    ) {
        using type = std::conditional_t<std::is_void_v<T>, U, T>;
        auto out = Array<type>(shape, option);
        linspace(out, params);
        return out;
    }

    /// Returns an array with evenly spaced values within a given interval, in the BDHW order.
    /// \tparam T           Any data type.
    /// \param n_elements   Number of elements.
    /// \param params       Linspace parameters.
    /// \param option       Options of the created array.
    template<typename T = void, typename U = T>
    [[nodiscard]] auto linspace(
        isize n_elements,
        Linspace<U> params,
        ArrayOption option = {}
    ) {
        using type = std::conditional_t<std::is_void_v<T>, U, T>;
        auto out = Array<type>(n_elements, option);
        linspace(out, params);
        return out;
    }
}

namespace noa {
    /// Returns a tiled sequence [0, elements), in the rightmost order.
    /// \param[out] output  Array with the tiled sequence.
    /// \param tile         Tile shape in each dimension.
    ///                     If the tile is equal to the shape of \p output,
    ///                     this is equivalent to `arange` with a start of 0 and step of 1.
    template<nt::writable_varray_decay Output, nt::integer T>
    void iota(Output&& output, const Vec<T, 4>& tile) {
        check(not output.is_empty(), "Empty array detected");

        auto shape = output.shape().template as_safe<T>();
        if (output.is_contiguous()) {
            auto accessor = nd::to_accessor_contiguous(output);
            using op_t = nd::Iota<4, decltype(accessor), T>;
            iwise(shape, output.device(),
                  op_t(accessor, shape, tile),
                  std::forward<Output>(output));
        } else {
            auto accessor = nd::to_accessor(output);
            using op_t = nd::Iota<4, decltype(accessor), T>;
            iwise(shape, output.device(),
                  op_t(accessor, shape, tile),
                  std::forward<Output>(output));
        }
    }

    /// Returns a tiled sequence [0, elements), in the rightmost order.
    /// \tparam T       Anything convertible from an integer.
    /// \param shape    Shape of the array.
    /// \param tile     Tile shape in each dimension. If the tile is equal to \p shape,
    ///                 this is equivalent to `arange` with a start of 0 and step of 1.
    /// \param option   Options of the created array.
    template<typename T, nt::integer U>
    [[nodiscard]] auto iota(const Shape4& shape, const Vec<U, 4>& tile, ArrayOption option = {}) -> Array<T> {
        auto out = Array<T>(shape, option);
        iota(out, tile);
        return out;
    }

    /// Returns a 1D tiled sequence [0, elements).
    /// \tparam T           Anything convertible from an integer.
    /// \param n_elements   Number of elements.
    /// \param tile         Tile size. If the tile is equal to \p elements,
    ///                     this is equivalent to `arange` with a start of 0 and step of 1.
    /// \param option       Options of the created array.
    template<typename T, nt::integer U>
    [[nodiscard]] auto iota(isize n_elements, U tile, ArrayOption option = {}) -> Array<T> {
        auto out = Array<T>(n_elements, option);
        iota(out, Vec<U, 4>{1, 1, 1, tile});
        return out;
    }
}
