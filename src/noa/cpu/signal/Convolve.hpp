#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/Ewise.hpp"
#include "noa/core/utils/Misc.hpp"
#include "noa/cpu/AllocatorHeap.hpp"
#include "noa/cpu/Iwise.hpp"
#include "noa/cpu/Ewise.hpp"

namespace noa::cpu::signal::guts {
    template<size_t DIM, typename InputAccessor, typename OutputAccessor, typename FilterAccessor>
    class Convolution {
    public:
        using shape_type = Shape<i64, DIM>;
        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using filter_accessor_type = FilterAccessor;
        using input_value_type = nt::mutable_value_type_t<input_accessor_type>;
        using output_value_type = nt::value_type_t<output_accessor_type>;
        using filter_value_type = nt::mutable_value_type_t<filter_accessor_type>;

    public:
        Convolution(
                const input_accessor_type& input,
                const output_accessor_type& output,
                const filter_accessor_type& filter,
                const shape_type& shape,
                const shape_type& filter_shape
        ) : m_input(input), m_output(output), m_filter(filter),
            m_shape(shape), m_filter_shape(filter_shape),
            m_halo(filter_shape / 2) {}

        constexpr void operator()(i64 i, i64 j, i64 k, i64 l) const {
            filter_value_type conv{};

            if constexpr (DIM == 1) {
                for (i64 wl{}; wl < m_filter_shape[0]; ++wl) {
                    const i64 il = l - m_halo[0] + wl;
                    if (il >= 0 and il < m_shape[0])
                        conv += static_cast<filter_value_type>(m_input(i, j, k, il)) * m_filter[wl];
                }
            } else if constexpr (DIM == 2) {
                for (i64 wk{}; wk < m_filter_shape[0]; ++wk) {
                    const i64 ik = k - m_halo[0] + wk;
                    if (ik < 0 or ik >= m_shape[0])
                        continue;
                    const i64 tmp = wk * m_filter_shape[1];
                    for (i64 wl{}; wl < m_filter_shape[1]; ++wl) {
                        const i64 il = l - m_halo[1] + wl;
                        if (il >= 0 and il < m_shape[1])
                            conv += static_cast<filter_value_type>(m_input(i, j, ik, il)) * m_filter[tmp + wl];
                    }
                }
            } else if constexpr (DIM == 3) {
                for (i64 wj{}; wj < m_filter_shape[0]; ++wj) {
                    const i64 ij = j - m_halo[0] + wj;
                    if (ij < 0 or ij >= m_shape[0])
                        continue;
                    const i64 tmp_z = wj * m_filter_shape[1] * m_filter_shape[2];
                    for (i64 wk{}; wk < m_filter_shape[1]; ++wk) {
                        const i64 ik = k - m_halo[1] + wk;
                        if (ik < 0 or ik >= m_shape[1])
                            continue;
                        const i64 tmp = tmp_z + wk * m_filter_shape[2];
                        for (i64 wl{}; wl < m_filter_shape[2]; ++wl) {
                            const i64 il = l - m_halo[2] + wl;
                            if (il >= 0 and il < m_shape[2])
                                conv += static_cast<filter_value_type>(m_input(i, ij, ik, il)) * m_filter[tmp + wl];
                        }
                    }
                }
            } else {
                static_assert(nt::always_false<>);
            }
            m_output(i, j, k, l) = static_cast<output_value_type>(conv);
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        filter_accessor_type m_filter;
        shape_type m_shape;
        shape_type m_filter_shape;
        shape_type m_halo;
    };

    enum class ConvolutionSeparableDim {
        DEPTH = 1, HEIGHT = 2, WIDTH = 3
    };

    template<ConvolutionSeparableDim DIM, typename InputAccessor, typename OutputAccessor, typename FilterAccessor>
    class ConvolutionSeparable {
    public:
        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using filter_accessor_type = FilterAccessor;
        using input_value_type = nt::mutable_value_type_t<input_accessor_type>;
        using output_value_type = nt::value_type_t<output_accessor_type>;
        using filter_value_type = nt::mutable_value_type_t<filter_accessor_type>;

    public:
        ConvolutionSeparable(
                const input_accessor_type& input,
                const output_accessor_type& output,
                const filter_accessor_type& filter,
                i64 dim_size, i64 filter_size
        ) : m_input(input), m_output(output), m_filter(filter),
            m_dim_size(dim_size), m_filter_size(filter_size),
            m_halo(filter_size / 2) {}

        constexpr void operator()(i64 i, i64 j, i64 k, i64 l) const {
            filter_value_type conv{};

            if constexpr (DIM == ConvolutionSeparableDim::WIDTH) {
                for (i64 wl{}; wl < m_filter_size; ++wl) {
                    const i64 il = l - m_halo + wl;
                    if (il >= 0 and il < m_dim_size)
                        conv += static_cast<filter_value_type>(m_input(i, j, k, il)) * m_filter[wl];
                }
            } else if constexpr (DIM == ConvolutionSeparableDim::HEIGHT) {
                for (i64 wk{}; wk < m_filter_size; ++wk) {
                    const i64 ik = k - m_halo + wk;
                    if (ik >= 0 and ik < m_dim_size)
                        conv += static_cast<filter_value_type>(m_input(i, j, ik, l)) * m_filter[wk];
                }
            } else if constexpr (DIM == ConvolutionSeparableDim::DEPTH) {
                for (i64 wj{}; wj < m_filter_size; ++wj) {
                    const i64 ij = j - m_halo + wj;
                    if (ij >= 0 and ij < m_dim_size)
                        conv += static_cast<filter_value_type>(m_input(i, ij, k, l)) * m_filter[wj];
                }
            }
            m_output(i, j, k, l) = static_cast<output_value_type>(conv);
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        filter_accessor_type m_filter;
        i64 m_dim_size;
        i64 m_filter_size;
        i64 m_halo;
    };

    // If half precision, convert filter and do the accumulation is single-precision.
    // Without this, half precision would be ~10times slower. With this preprocessing, it is only 2times slower.
    template<typename T>
    auto get_filter(const T* filter, i64 filter_size) {
        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        using buffer_t = AllocatorHeap<compute_t>::alloc_unique_type;
        using accessor_t = AccessorRestrictContiguous<const compute_t, 1, i64>;
        buffer_t buffer{};

        if constexpr (std::is_same_v<f16, T>) {
            buffer = AllocatorHeap<compute_t>::allocate(filter_size);
            for (size_t i{}; i < static_cast<size_t>(filter_size); ++i)
                buffer[i] = static_cast<compute_t>(filter[i]);
            return Pair{accessor_t(buffer.get()), std::move(buffer)};
        } else {
            return Pair{accessor_t(filter), std::move(buffer)};
        }
    }

    template<ConvolutionSeparableDim DIM, typename T, typename U, typename V>
    void launch_convolve_separable(
            const T* input, const Strides4<i64>& input_strides,
            U* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const V* filter, i64 filter_size, i64 threads
    ) {
        using input_accessor_t = AccessorRestrict<const T, 4, i64>;
        using output_accessor_t = AccessorRestrict<U, 4, i64>;
        const auto [filter_accessor, filter_buffer] = guts::get_filter(filter, filter_size);
        const auto dim_size = shape.filter(to_underlying(DIM))[0];

        auto kernel = ConvolutionSeparable<DIM, input_accessor_t, output_accessor_t, decltype(filter_accessor)>(
                input_accessor_t(input, input_strides),
                output_accessor_t(output, output_strides),
                filter_accessor, dim_size, filter_size);
        iwise(shape, kernel, threads);
    }
}

namespace noa::cpu::signal {
    template<typename T, typename U, typename V>
    void convolve(
            const T* input, Strides4<i64> input_strides,
            U* output, Strides4<i64> output_strides, const Shape4<i64>& shape,
            const V* filter, const Shape3<i64>& filter_shape, i64 threads
    ) {
        const auto n_dimensions_to_convolve = sum(filter_shape > 1);
        const auto ndim = filter_shape.ndim();
        if (n_dimensions_to_convolve == 1) {
            if (filter_shape[0] > 1) {
                guts::launch_convolve_separable<guts::ConvolutionSeparableDim::DEPTH>(
                        input, input_strides, output, output_strides, shape,
                        filter, filter_shape[0], threads);
            } else if (filter_shape[1] > 1) {
                guts::launch_convolve_separable<guts::ConvolutionSeparableDim::HEIGHT>(
                        input, input_strides, output, output_strides, shape,
                        filter, filter_shape[1], threads);
            } else {
                guts::launch_convolve_separable<guts::ConvolutionSeparableDim::WIDTH>(
                        input, input_strides, output, output_strides, shape,
                        filter, filter_shape[2], threads);
            }
        } else if (ndim == 2) {
            const auto filter_shape_2d = filter_shape.pop_front();
            const auto input_accessor = AccessorRestrict<const T, 4, i64>(input, input_strides);
            const auto output_accessor = AccessorRestrict<U, 4, i64>(output, output_strides);
            const auto [filter_accessor, filter_buffer] = guts::get_filter(filter, filter_shape_2d.n_elements());
            const auto shape_2d = shape.filter(2, 3);
            auto kernel = guts::Convolution(input_accessor, output_accessor, filter_accessor, shape_2d, filter_shape_2d);
            iwise(shape, kernel, threads);

        } else if (ndim == 3) {
            const auto input_accessor = AccessorRestrict<const T, 4, i64>(input, input_strides);
            const auto output_accessor = AccessorRestrict<U, 4, i64>(output, output_strides);
            const auto [filter_accessor, filter_buffer] = guts::get_filter(filter, filter_shape.n_elements());
            const auto shape_3d = shape.filter(1, 2, 3);
            auto kernel = guts::Convolution(input_accessor, output_accessor, filter_accessor, shape_3d, filter_shape);
            iwise(shape, kernel, threads);

        } else if (all(filter_shape == 1)) {
            auto order = ni::order(output_strides, shape);
            if (vany(NotEqual{}, order, Vec4<i64>{0, 1, 2, 3})) {
                input_strides = ni::reorder(input_strides, order);
                output_strides = ni::reorder(output_strides, order);
            }
            const auto input_accessor = AccessorRestrict<const T, 4, i64>(input, input_strides);
            const auto output_accessor = AccessorRestrict<U, 4, i64>(output, output_strides);
            const auto value = AccessorValue<T>(static_cast<T>(filter[0]));
            return ewise(shape, Multiply{}, make_tuple(input_accessor, value), make_tuple(output_accessor), threads);

        } else {
            panic("unreachable");
        }
    }

    template<typename T, typename U, typename V> requires nt::are_real_v<T, U, V>
    void convolve_separable(
            const T* input, const Strides4<i64>& input_strides,
            U* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const V* filter_depth, i64 filter_depth_size,
            const V* filter_height, i64 filter_height_size,
            const V* filter_width, i64 filter_width_size,
            V* tmp, Strides4<i64> tmp_strides, i64 threads
    ) {
        if (filter_depth_size <= 0)
            filter_depth = nullptr;
        if (filter_height_size <= 0)
            filter_height = nullptr;
        if (filter_width_size <= 0)
            filter_width = nullptr;

        // Allocate temp buffer if necessary.
        i32 count = 0;
        if (filter_depth)
            count += 1;
        if (filter_height)
            count += 1;
        if (filter_width)
            count += 1;
        using allocator_t = noa::cpu::AllocatorHeap<V>;
        typename allocator_t::alloc_unique_type buffer{};
        if (not tmp and count > 1) {
            buffer = allocator_t::allocate(shape.n_elements());
            tmp = buffer.get();
            tmp_strides = shape.strides();
        }

        if (filter_depth and filter_height and filter_width) {
            guts::launch_convolve_separable<guts::ConvolutionSeparableDim::DEPTH>(
                    input, input_strides, output, output_strides, shape,
                    filter_depth, filter_depth_size, threads);
            guts::launch_convolve_separable<guts::ConvolutionSeparableDim::HEIGHT>(
                    output, output_strides, tmp, tmp_strides, shape,
                    filter_height, filter_height_size, threads);
            guts::launch_convolve_separable<guts::ConvolutionSeparableDim::WIDTH>(
                    tmp, tmp_strides, output, output_strides, shape,
                    filter_width, filter_width_size, threads);

        } else if (filter_depth and filter_height) {
            guts::launch_convolve_separable<guts::ConvolutionSeparableDim::DEPTH>(
                    input, input_strides, tmp, tmp_strides, shape,
                    filter_depth, filter_depth_size, threads);
            guts::launch_convolve_separable<guts::ConvolutionSeparableDim::HEIGHT>(
                    tmp, tmp_strides, output, output_strides, shape,
                    filter_height, filter_height_size, threads);

        } else if (filter_height and filter_width) {
            guts::launch_convolve_separable<guts::ConvolutionSeparableDim::HEIGHT>(
                    input, input_strides, tmp, tmp_strides, shape,
                    filter_height, filter_height_size, threads);
            guts::launch_convolve_separable<guts::ConvolutionSeparableDim::WIDTH>(
                    tmp, tmp_strides, output, output_strides, shape,
                    filter_width, filter_width_size, threads);

        } else if (filter_depth and filter_width) {
            guts::launch_convolve_separable<guts::ConvolutionSeparableDim::DEPTH>(
                    input, input_strides, tmp, tmp_strides, shape,
                    filter_depth, filter_depth_size, threads);
            guts::launch_convolve_separable<guts::ConvolutionSeparableDim::WIDTH>(
                    tmp, tmp_strides, output, output_strides, shape,
                    filter_width, filter_width_size, threads);

        } else if (filter_depth) {
            guts::launch_convolve_separable<guts::ConvolutionSeparableDim::DEPTH>(
                    input, input_strides, output, output_strides, shape,
                    filter_depth, filter_depth_size, threads);
        } else if (filter_height) {
            guts::launch_convolve_separable<guts::ConvolutionSeparableDim::HEIGHT>(
                    input, input_strides, output, output_strides, shape,
                    filter_height, filter_height_size, threads);
        } else if (filter_width) {
            guts::launch_convolve_separable<guts::ConvolutionSeparableDim::WIDTH>(
                    input, input_strides, output, output_strides, shape,
                    filter_width, filter_width_size, threads);
        }
    }
}
