#pragma once

#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cpu/Allocators.hpp"
#include "noa/runtime/cpu/Iwise.hpp"

namespace noa::signal::cpu::details {
    template<usize B, usize R, Border BORDER, typename InputAccessor, typename OutputAccessor, typename FilterAccessor>
    class Convolution {
    public:
        using shape_type = Shape<isize, R>;
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

        constexpr void operator()(const Vec<isize, B + R>& output_indices) const {
            const auto [batches, indices] = output_indices.template split<B>();
            auto input_r = m_input[batches];

            filter_value_type conv{};
            if constexpr (R == 1) {
                for (isize wl{}; wl < m_filter_shape[0]; ++wl) {
                    const isize il = indices[0] - m_halo[0] + wl;
                    if constexpr (BORDER == Border::ZERO) {
                        if (il >= 0 and il < m_shape[0])
                            conv += static_cast<filter_value_type>(input_r(il)) * m_filter[wl];
                    } else {
                        const auto idx = index_at<BORDER>(il, m_shape[0]);
                        conv += static_cast<filter_value_type>(input_r(idx)) * m_filter[wl];
                    }
                }
            } else if constexpr (R == 2) {
                if constexpr (BORDER == Border::ZERO) {
                    for (isize wk{}; wk < m_filter_shape[0]; ++wk) {
                        const isize ik = indices[0] - m_halo[0] + wk;
                        if (ik < 0 or ik >= m_shape[0])
                            continue;
                        const isize tmp = wk * m_filter_shape[1];
                        for (isize wl{}; wl < m_filter_shape[1]; ++wl) {
                            const isize il = indices[1] - m_halo[1] + wl;
                            if (il >= 0 and il < m_shape[1])
                                conv += static_cast<filter_value_type>(input_r(ik, il)) * m_filter[tmp + wl];
                        }
                    }
                } else {
                    for (isize wk{}; wk < m_filter_shape[0]; ++wk) {
                        const isize ik = index_at<BORDER>(indices[0] - m_halo[0] + wk, m_shape[0]);
                        const isize tmp = wk * m_filter_shape[1];
                        for (isize wl{}; wl < m_filter_shape[1]; ++wl) {
                            const isize il = index_at<BORDER>(indices[1] - m_halo[1] + wl, m_shape[1]);
                            conv += static_cast<filter_value_type>(input_r(ik, il)) * m_filter[tmp + wl];
                        }
                    }
                }
            } else if constexpr (R == 3) {
                if constexpr (BORDER == Border::ZERO) {
                    for (isize wj{}; wj < m_filter_shape[0]; ++wj) {
                        const isize ij = indices[0] - m_halo[0] + wj;
                        if (ij < 0 or ij >= m_shape[0])
                            continue;
                        const isize tmp_z = wj * m_filter_shape[1] * m_filter_shape[2];
                        for (isize wk{}; wk < m_filter_shape[1]; ++wk) {
                            const isize ik = indices[1] - m_halo[1] + wk;
                            if (ik < 0 or ik >= m_shape[1])
                                continue;
                            const isize tmp = tmp_z + wk * m_filter_shape[2];
                            for (isize wl{}; wl < m_filter_shape[2]; ++wl) {
                                const isize il = indices[2] - m_halo[2] + wl;
                                if (il >= 0 and il < m_shape[2])
                                    conv += static_cast<filter_value_type>(input_r(ij, ik, il)) * m_filter[tmp + wl];
                            }
                        }
                    }
                } else {
                    for (isize wj{}; wj < m_filter_shape[0]; ++wj) {
                        const isize ij = index_at<BORDER>(indices[0] - m_halo[0] + wj, m_shape[0]);
                        const isize tmp_z = wj * m_filter_shape[1] * m_filter_shape[2];
                        for (isize wk{}; wk < m_filter_shape[1]; ++wk) {
                            const isize ik = index_at<BORDER>(indices[1] - m_halo[1] + wk, m_shape[1]);
                            const isize tmp = tmp_z + wk * m_filter_shape[2];
                            for (isize wl{}; wl < m_filter_shape[2]; ++wl) {
                                const isize il = index_at<BORDER>(indices[2] - m_halo[2] + wl, m_shape[2]);
                                conv += static_cast<filter_value_type>(input_r(ij, ik, il)) * m_filter[tmp + wl];
                            }
                        }
                    }
                }
            } else {
                static_assert(nt::always_false<output_value_type>);
            }
            m_output(output_indices) = static_cast<output_value_type>(conv);
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        filter_accessor_type m_filter;
        shape_type m_shape;
        shape_type m_filter_shape;
        shape_type m_halo;
    };

    template<usize N, usize I, Border BORDER, typename InputAccessor, typename OutputAccessor, typename FilterAccessor>
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

        constexpr void operator()(const Vec<isize, N>& output_indices) const {
            filter_value_type conv{};
            auto input_indices = output_indices;
            for (isize i{}; i < m_filter_size; ++i) {
                input_indices[I] = output_indices[I] - m_halo + i;
                if constexpr (BORDER == Border::ZERO) {
                    if (input_indices[I] >= 0 and input_indices[I] < m_dim_size)
                        conv += static_cast<filter_value_type>(m_input(input_indices)) * m_filter[i];
                } else {
                    input_indices[I] = index_at<BORDER>(input_indices[I], m_dim_size);
                    conv += static_cast<filter_value_type>(m_input(input_indices)) * m_filter[i];
                }
            }
            m_output(output_indices) = static_cast<output_value_type>(conv);
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
    // Without this, half precision is ~10x slower. With this preprocessing, it is only 2x slower.
    template<typename T, usize R>
    auto get_filter(const T* filter, const Shape<isize, R>& filter_shape) {
        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        using buffer_t = noa::cpu::AllocatorHeap::allocate_type<compute_t>;
        using accessor_t = AccessorRestrictContiguous<const compute_t, R, isize>;
        buffer_t buffer{};

        if constexpr (std::is_same_v<f16, T>) {
            const auto filter_size = static_cast<usize>(filter_shape.n_elements());
            buffer = noa::cpu::AllocatorHeap::allocate<compute_t>();
            for (usize i{}; i < filter_size; ++i)
                buffer[i] = static_cast<compute_t>(filter[i]);
            return Pair{accessor_t(buffer.get()), std::move(buffer)};
        } else {
            return Pair{accessor_t(filter), std::move(buffer)};
        }
    }

    template<usize I, Border BORDER, typename T, typename U, typename V, usize N>
    void launch_convolve_separable(
        const T* input, const Strides<isize, N>& input_strides,
        U* output, const Strides<isize, N>& output_strides, const Shape<isize, N>& shape,
        const V* filter, isize filter_size, i32 n_threads
    ) {
        using input_accessor_t = AccessorRestrict<const T, N, isize>;
        using output_accessor_t = AccessorRestrict<U, N, isize>;
        const auto [filter_accessor, filter_buffer] = details::get_filter(filter, Shape{filter_size});
        const auto dim_size = shape[I];

        auto kernel = ConvolutionSeparable<N, I, BORDER, input_accessor_t, output_accessor_t, decltype(filter_accessor)>(
            input_accessor_t(input, input_strides),
            output_accessor_t(output, output_strides),
            filter_accessor, dim_size, filter_size
        );
        noa::cpu::iwise(shape, kernel, n_threads);
    }
}

namespace noa::signal::cpu {
    template<Border BORDER, typename T, typename U, typename V, usize N, usize R>
    void convolve(
        const T* input, Strides<isize, N> input_strides,
        U* output, Strides<isize, N> output_strides, const Shape<isize, N>& shape,
        const V* filter, const Shape<isize, R>& filter_shape, i32 n_threads
    ) {
        const auto n_dimensions_to_convolve = sum(filter_shape.cmp_gt(1));
        if (n_dimensions_to_convolve == 1) {
            if (filter_shape[0] > 1) {
                details::launch_convolve_separable<0, BORDER>(
                    input, input_strides, output, output_strides, shape,
                    filter, filter_shape[0], n_threads);
            } else if (filter_shape[1] > 1) {
                details::launch_convolve_separable<1, BORDER>(
                    input, input_strides, output, output_strides, shape,
                    filter, filter_shape[1], n_threads);
            } else {
                details::launch_convolve_separable<2, BORDER>(
                    input, input_strides, output, output_strides, shape,
                    filter, filter_shape[2], n_threads);
            }
            return;
        }

        const auto rank = filter_shape.rank();
        if (rank == 2) {
            using input_accessor_t = AccessorRestrict<const T, N, isize>;
            using output_accessor_t = AccessorRestrict<U, N, isize>;
            const auto input_accessor = input_accessor_t(input, input_strides);
            const auto output_accessor = output_accessor_t(output, output_strides);

            const auto filter_shape_2d = filter_shape.pop_front();
            const auto [filter_accessor, filter_buffer] = details::get_filter(filter, filter_shape_2d);
            using filter_accessor_t = std::decay_t<decltype(filter_accessor)>;

            constexpr auto B = N - 2;
            using op_t = details::Convolution<B, 2, BORDER, input_accessor_t, output_accessor_t, filter_accessor_t>;
            auto kernel = op_t(input_accessor, output_accessor, filter_accessor, shape.template pop_front<B>(), filter_shape_2d);
            noa::cpu::iwise(shape, kernel, n_threads);

        } else if (rank == 3) {
            using input_accessor_t = AccessorRestrict<const T, N, isize>;
            using output_accessor_t = AccessorRestrict<U, N, isize>;
            const auto input_accessor = input_accessor_t(input, input_strides);
            const auto output_accessor = output_accessor_t(output, output_strides);

            const auto [filter_accessor, filter_buffer] = details::get_filter(filter, filter_shape);
            using filter_accessor_t = std::decay_t<decltype(filter_accessor)>;

            constexpr auto B = N - 3;
            using op_t = details::Convolution<B, 3, BORDER, input_accessor_t, output_accessor_t, filter_accessor_t>;
            auto kernel = op_t(input_accessor, output_accessor, filter_accessor, shape.template pop_front<B>(), filter_shape);
            noa::cpu::iwise(shape, kernel, n_threads);

        } else {
            panic();
        }
    }

    template<Border BORDER, typename T, typename U, typename V, usize N> requires nt::are_real_v<T, U, V>
    void convolve_separable(
        const T* input, const Strides<isize, N>& input_strides,
        U* output, const Strides<isize, N>& output_strides, const Shape<isize, N>& shape,
        const V* filter_depth, isize filter_depth_size,
        const V* filter_height, isize filter_height_size,
        const V* filter_width, isize filter_width_size,
        V* tmp, Strides<isize, N> tmp_strides, i32 n_threads
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

        constexpr auto D = N - 3;
        constexpr auto H = N - 2;
        constexpr auto W = N - 1;
        if constexpr (N >= 3) {
            if (filter_depth and filter_height and filter_width) {
                details::launch_convolve_separable<D, BORDER>(
                    input, input_strides, output, output_strides, shape,
                    filter_depth, filter_depth_size, n_threads);
                details::launch_convolve_separable<H, BORDER>(
                    output, output_strides, tmp, tmp_strides, shape,
                    filter_height, filter_height_size, n_threads);
                details::launch_convolve_separable<W, BORDER>(
                    tmp, tmp_strides, output, output_strides, shape,
                    filter_width, filter_width_size, n_threads);
                return;
            }
        }

        if constexpr (N >= 2) {
            if constexpr (N >= 3) {
                if (filter_depth and filter_height) {
                    details::launch_convolve_separable<D, BORDER>(
                        input, input_strides, tmp, tmp_strides, shape,
                        filter_depth, filter_depth_size, n_threads);
                    details::launch_convolve_separable<H, BORDER>(
                        tmp, tmp_strides, output, output_strides, shape,
                        filter_height, filter_height_size, n_threads);
                    return;
                }
                if (filter_depth and filter_width) {
                    details::launch_convolve_separable<D, BORDER>(
                        input, input_strides, tmp, tmp_strides, shape,
                        filter_depth, filter_depth_size, n_threads);
                    details::launch_convolve_separable<W, BORDER>(
                        tmp, tmp_strides, output, output_strides, shape,
                        filter_width, filter_width_size, n_threads);
                    return;
                }
            }
            if (filter_height and filter_width) {
                details::launch_convolve_separable<H, BORDER>(
                    input, input_strides, tmp, tmp_strides, shape,
                    filter_height, filter_height_size, n_threads);
                details::launch_convolve_separable<W, BORDER>(
                    tmp, tmp_strides, output, output_strides, shape,
                    filter_width, filter_width_size, n_threads);
                return;
            }
        }

        if constexpr (N >= 3) {
            if (filter_depth) {
                details::launch_convolve_separable<D, BORDER>(
                    input, input_strides, output, output_strides, shape,
                    filter_depth, filter_depth_size, n_threads);
                return;
            }
        }
        if constexpr (N >= 2) {
            if (filter_height) {
                details::launch_convolve_separable<H, BORDER>(
                    input, input_strides, output, output_strides, shape,
                    filter_height, filter_height_size, n_threads);
                return;
            }
        }
        if (filter_width) {
            details::launch_convolve_separable<W, BORDER>(
                input, input_strides, output, output_strides, shape,
                filter_width, filter_width_size, n_threads);
        }
    }
}
