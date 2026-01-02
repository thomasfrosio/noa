#pragma once

#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Shape.hpp"

#include "noa/runtime/cpu/Allocators.hpp"
#include "noa/runtime/cpu/Ewise.hpp"
#include "noa/runtime/cpu/Iwise.hpp"

namespace noa::signal::cpu::details {
    template<usize DIM, Border BORDER, typename InputAccessor, typename OutputAccessor, typename FilterAccessor>
    class Convolution {
    public:
        using shape_type = Shape<isize, DIM>;
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

        constexpr void operator()(isize i, isize j, isize k, isize l) const {
            filter_value_type conv{};

            if constexpr (DIM == 1) {
                for (isize wl{}; wl < m_filter_shape[0]; ++wl) {
                    const isize il = l - m_halo[0] + wl;
                    if constexpr (BORDER == Border::ZERO) {
                        if (il >= 0 and il < m_shape[0])
                            conv += static_cast<filter_value_type>(m_input(i, j, k, il)) * m_filter[wl];
                    } else {
                        const auto idx = index_at<BORDER>(il, m_shape[0]);
                        conv += static_cast<filter_value_type>(m_input(i, j, k, idx)) * m_filter[wl];
                    }
                }
            } else if constexpr (DIM == 2) {
                if constexpr (BORDER == Border::ZERO) {
                    for (isize wk{}; wk < m_filter_shape[0]; ++wk) {
                        const isize ik = k - m_halo[0] + wk;
                        if (ik < 0 or ik >= m_shape[0])
                            continue;
                        const isize tmp = wk * m_filter_shape[1];
                        for (isize wl{}; wl < m_filter_shape[1]; ++wl) {
                            const isize il = l - m_halo[1] + wl;
                            if (il >= 0 and il < m_shape[1])
                                conv += static_cast<filter_value_type>(m_input(i, j, ik, il)) * m_filter[tmp + wl];
                        }
                    }
                } else {
                    for (isize wk{}; wk < m_filter_shape[0]; ++wk) {
                        const isize ik = index_at<BORDER>(k - m_halo[0] + wk, m_shape[0]);
                        const isize tmp = wk * m_filter_shape[1];
                        for (isize wl{}; wl < m_filter_shape[1]; ++wl) {
                            const isize il = index_at<BORDER>(l - m_halo[1] + wl, m_shape[1]);
                            conv += static_cast<filter_value_type>(m_input(i, j, ik, il)) * m_filter[tmp + wl];
                        }
                    }
                }
            } else if constexpr (DIM == 3) {
                if constexpr (BORDER == Border::ZERO) {
                    for (isize wj{}; wj < m_filter_shape[0]; ++wj) {
                        const isize ij = j - m_halo[0] + wj;
                        if (ij < 0 or ij >= m_shape[0])
                            continue;
                        const isize tmp_z = wj * m_filter_shape[1] * m_filter_shape[2];
                        for (isize wk{}; wk < m_filter_shape[1]; ++wk) {
                            const isize ik = k - m_halo[1] + wk;
                            if (ik < 0 or ik >= m_shape[1])
                                continue;
                            const isize tmp = tmp_z + wk * m_filter_shape[2];
                            for (isize wl{}; wl < m_filter_shape[2]; ++wl) {
                                const isize il = l - m_halo[2] + wl;
                                if (il >= 0 and il < m_shape[2])
                                    conv += static_cast<filter_value_type>(m_input(i, ij, ik, il)) * m_filter[tmp + wl];
                            }
                        }
                    }
                } else {
                    for (isize wj{}; wj < m_filter_shape[0]; ++wj) {
                        const isize ij = index_at<BORDER>(j - m_halo[0] + wj, m_shape[0]);
                        const isize tmp_z = wj * m_filter_shape[1] * m_filter_shape[2];
                        for (isize wk{}; wk < m_filter_shape[1]; ++wk) {
                            const isize ik = index_at<BORDER>(k - m_halo[1] + wk, m_shape[1]);
                            const isize tmp = tmp_z + wk * m_filter_shape[2];
                            for (isize wl{}; wl < m_filter_shape[2]; ++wl) {
                                const isize il = index_at<BORDER>(l - m_halo[2] + wl, m_shape[2]);
                                conv += static_cast<filter_value_type>(m_input(i, ij, ik, il)) * m_filter[tmp + wl];
                            }
                        }
                    }
                }
            } else {
                static_assert(nt::always_false<output_value_type>);
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

    template<ConvolutionSeparableDim DIM, Border BORDER,
             typename InputAccessor, typename OutputAccessor, typename FilterAccessor>
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
            isize dim_size, isize filter_size
        ) : m_input(input), m_output(output), m_filter(filter),
            m_dim_size(dim_size), m_filter_size(filter_size),
            m_halo(filter_size / 2) {}

        constexpr void operator()(isize i, isize j, isize k, isize l) const {
            filter_value_type conv{};
            if constexpr (DIM == ConvolutionSeparableDim::WIDTH) {
                for (isize wl{}; wl < m_filter_size; ++wl) {
                    const isize il = l - m_halo + wl;
                    if constexpr (BORDER == Border::ZERO) {
                        if (il >= 0 and il < m_dim_size)
                            conv += static_cast<filter_value_type>(m_input(i, j, k, il)) * m_filter[wl];
                    } else {
                        const auto idx = index_at<BORDER>(il, m_dim_size);
                        conv += static_cast<filter_value_type>(m_input(i, j, k, idx)) * m_filter[wl];
                    }
                }
            } else if constexpr (DIM == ConvolutionSeparableDim::HEIGHT) {
                for (isize wk{}; wk < m_filter_size; ++wk) {
                    const isize ik = k - m_halo + wk;
                    if constexpr (BORDER == Border::ZERO) {
                        if (ik >= 0 and ik < m_dim_size)
                            conv += static_cast<filter_value_type>(m_input(i, j, ik, l)) * m_filter[wk];
                    } else {
                        const auto idx = index_at<BORDER>(ik, m_dim_size);
                        conv += static_cast<filter_value_type>(m_input(i, j, idx, l)) * m_filter[wk];
                    }
                }
            } else if constexpr (DIM == ConvolutionSeparableDim::DEPTH) {
                for (isize wj{}; wj < m_filter_size; ++wj) {
                    const isize ij = j - m_halo + wj;
                    if constexpr (BORDER == Border::ZERO) {
                        if (ij >= 0 and ij < m_dim_size)
                            conv += static_cast<filter_value_type>(m_input(i, ij, k, l)) * m_filter[wj];
                    } else {
                        const auto idx = index_at<BORDER>(ij, m_dim_size);
                        conv += static_cast<filter_value_type>(m_input(i, idx, k, l)) * m_filter[wj];
                    }
                }
            }
            m_output(i, j, k, l) = static_cast<output_value_type>(conv);
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        filter_accessor_type m_filter;
        isize m_dim_size;
        isize m_filter_size;
        isize m_halo;
    };

    // If half precision, convert filter and do the accumulation is single-precision.
    // Without this, half precision would be ~10times slower. With this preprocessing, it is only 2times slower.
    template<typename T>
    auto get_filter(const T* filter, isize filter_size) {
        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        using buffer_t = noa::cpu::AllocatorHeap::allocate_type<compute_t>;
        using accessor_t = AccessorRestrictContiguous<const compute_t, 1, isize>;
        buffer_t buffer{};

        if constexpr (std::is_same_v<f16, T>) {
            buffer = noa::cpu::AllocatorHeap::allocate<compute_t>(filter_size);
            for (usize i{}; i < static_cast<usize>(filter_size); ++i)
                buffer[i] = static_cast<compute_t>(filter[i]);
            return Pair{accessor_t(buffer.get()), std::move(buffer)};
        } else {
            return Pair{accessor_t(filter), std::move(buffer)};
        }
    }

    template<ConvolutionSeparableDim DIM, Border BORDER, typename T, typename U, typename V>
    void launch_convolve_separable(
        const T* input, const Strides4& input_strides,
        U* output, const Strides4& output_strides, const Shape4& shape,
        const V* filter, isize filter_size, i32 n_threads
    ) {
        using input_accessor_t = AccessorRestrict<const T, 4, isize>;
        using output_accessor_t = AccessorRestrict<U, 4, isize>;
        const auto [filter_accessor, filter_buffer] = details::get_filter(filter, filter_size);
        const auto dim_size = shape.filter(to_underlying(DIM))[0];

        auto kernel = ConvolutionSeparable<DIM, BORDER, input_accessor_t, output_accessor_t, decltype(filter_accessor)>(
            input_accessor_t(input, input_strides),
            output_accessor_t(output, output_strides),
            filter_accessor, dim_size, filter_size
        );
        noa::cpu::iwise(shape, kernel, n_threads);
    }
}

namespace noa::signal::cpu {
    template<Border BORDER, typename T, typename U, typename V>
    void convolve(
        const T* input, Strides4 input_strides,
        U* output, Strides4 output_strides, const Shape4& shape,
        const V* filter, const Shape3& filter_shape, i32 n_threads
    ) {
        const auto n_dimensions_to_convolve = sum(filter_shape.cmp_gt(1));
        const auto ndim = filter_shape.ndim();
        if (n_dimensions_to_convolve == 1) {
            if (filter_shape[0] > 1) {
                details::launch_convolve_separable<details::ConvolutionSeparableDim::DEPTH, BORDER>(
                    input, input_strides, output, output_strides, shape,
                    filter, filter_shape[0], n_threads);
            } else if (filter_shape[1] > 1) {
                details::launch_convolve_separable<details::ConvolutionSeparableDim::HEIGHT, BORDER>(
                    input, input_strides, output, output_strides, shape,
                    filter, filter_shape[1], n_threads);
            } else {
                details::launch_convolve_separable<details::ConvolutionSeparableDim::WIDTH, BORDER>(
                    input, input_strides, output, output_strides, shape,
                    filter, filter_shape[2], n_threads);
            }
        } else if (ndim == 2) {
            using input_accessor_t = AccessorRestrict<const T, 4, isize>;
            using output_accessor_t = AccessorRestrict<U, 4, isize>;
            const auto input_accessor = input_accessor_t(input, input_strides);
            const auto output_accessor = output_accessor_t(output, output_strides);

            const auto filter_shape_2d = filter_shape.pop_front();
            const auto [filter_accessor, filter_buffer] = details::get_filter(filter, filter_shape_2d.n_elements());
            using filter_accessor_t = std::decay_t<decltype(filter_accessor)>;

            using op_t = details::Convolution<2, BORDER, input_accessor_t, output_accessor_t, filter_accessor_t>;
            auto kernel = op_t(input_accessor, output_accessor, filter_accessor, shape.filter(2, 3), filter_shape_2d);
            noa::cpu::iwise(shape, kernel, n_threads);

        } else if (ndim == 3) {
            using input_accessor_t = AccessorRestrict<const T, 4, isize>;
            using output_accessor_t = AccessorRestrict<U, 4, isize>;
            const auto input_accessor = input_accessor_t(input, input_strides);
            const auto output_accessor = output_accessor_t(output, output_strides);

            const auto [filter_accessor, filter_buffer] = details::get_filter(filter, filter_shape.n_elements());
            using filter_accessor_t = std::decay_t<decltype(filter_accessor)>;

            using op_t = details::Convolution<3, BORDER, input_accessor_t, output_accessor_t, filter_accessor_t>;
            auto kernel = op_t(input_accessor, output_accessor, filter_accessor, shape.filter(1, 2, 3), filter_shape);
            noa::cpu::iwise(shape, kernel, n_threads);

        } else if (filter_shape == 1) {
            nd::permute_all_to_rightmost_order<true>(output_strides, shape, input_strides, output_strides);
            const auto input_accessor = AccessorRestrict<const T, 4, isize>(input, input_strides);
            const auto output_accessor = AccessorRestrict<U, 4, isize>(output, output_strides);
            const auto value = AccessorValue<T>(static_cast<T>(filter[0]));
            return noa::cpu::ewise(shape, Multiply{}, make_tuple(input_accessor, value), make_tuple(output_accessor), n_threads);

        } else {
            panic();
        }
    }

    template<Border BORDER, typename T, typename U, typename V> requires nt::are_real_v<T, U, V>
    void convolve_separable(
        const T* input, const Strides4& input_strides,
        U* output, const Strides4& output_strides, const Shape4& shape,
        const V* filter_depth, isize filter_depth_size,
        const V* filter_height, isize filter_height_size,
        const V* filter_width, isize filter_width_size,
        V* tmp, Strides4 tmp_strides, i32 n_threads
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
        noa::cpu::AllocatorHeap::allocate_type<V> buffer{};
        if (not tmp and count > 1) {
            buffer = noa::cpu::AllocatorHeap::allocate<V>(shape.n_elements());
            tmp = buffer.get();
            tmp_strides = shape.strides();
        }

        if (filter_depth and filter_height and filter_width) {
            details::launch_convolve_separable<details::ConvolutionSeparableDim::DEPTH, BORDER>(
                input, input_strides, output, output_strides, shape,
                filter_depth, filter_depth_size, n_threads);
            details::launch_convolve_separable<details::ConvolutionSeparableDim::HEIGHT, BORDER>(
                output, output_strides, tmp, tmp_strides, shape,
                filter_height, filter_height_size, n_threads);
            details::launch_convolve_separable<details::ConvolutionSeparableDim::WIDTH, BORDER>(
                tmp, tmp_strides, output, output_strides, shape,
                filter_width, filter_width_size, n_threads);

        } else if (filter_depth and filter_height) {
            details::launch_convolve_separable<details::ConvolutionSeparableDim::DEPTH, BORDER>(
                input, input_strides, tmp, tmp_strides, shape,
                filter_depth, filter_depth_size, n_threads);
            details::launch_convolve_separable<details::ConvolutionSeparableDim::HEIGHT, BORDER>(
                tmp, tmp_strides, output, output_strides, shape,
                filter_height, filter_height_size, n_threads);

        } else if (filter_height and filter_width) {
            details::launch_convolve_separable<details::ConvolutionSeparableDim::HEIGHT, BORDER>(
                input, input_strides, tmp, tmp_strides, shape,
                filter_height, filter_height_size, n_threads);
            details::launch_convolve_separable<details::ConvolutionSeparableDim::WIDTH, BORDER>(
                tmp, tmp_strides, output, output_strides, shape,
                filter_width, filter_width_size, n_threads);

        } else if (filter_depth and filter_width) {
            details::launch_convolve_separable<details::ConvolutionSeparableDim::DEPTH, BORDER>(
                input, input_strides, tmp, tmp_strides, shape,
                filter_depth, filter_depth_size, n_threads);
            details::launch_convolve_separable<details::ConvolutionSeparableDim::WIDTH, BORDER>(
                tmp, tmp_strides, output, output_strides, shape,
                filter_width, filter_width_size, n_threads);

        } else if (filter_depth) {
            details::launch_convolve_separable<details::ConvolutionSeparableDim::DEPTH, BORDER>(
                input, input_strides, output, output_strides, shape,
                filter_depth, filter_depth_size, n_threads);
        } else if (filter_height) {
            details::launch_convolve_separable<details::ConvolutionSeparableDim::HEIGHT, BORDER>(
                input, input_strides, output, output_strides, shape,
                filter_height, filter_height_size, n_threads);
        } else if (filter_width) {
            details::launch_convolve_separable<details::ConvolutionSeparableDim::WIDTH, BORDER>(
                input, input_strides, output, output_strides, shape,
                filter_width, filter_width_size, n_threads);
        }
    }
}
