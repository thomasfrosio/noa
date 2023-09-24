#include "noa/cpu/Ewise.hpp"
#include "noa/cpu/memory/AllocatorHeap.hpp"
#include "noa/cpu/signal/Convolve.hpp"
#include "noa/cpu/utils/Iwise.hpp"

namespace {
    using namespace ::noa;

    template<typename T, size_t DIM>
    class Convolution {
    public:
        using value_type = T;
        using shape_type = Shape<i64, DIM>;
        using compute_type = std::conditional_t<std::is_same_v<f16, value_type>, f32, value_type>;
        using input_accessor_type = AccessorRestrict<const value_type, 4, i64>;
        using output_accessor_type = AccessorRestrict<value_type, 4, i64>;
        using filter_accessor_type = AccessorRestrictContiguous<const compute_type, 1, i64>;

    public:
        Convolution(const input_accessor_type& input,
                    const output_accessor_type& output,
                    const filter_accessor_type& filter,
                    const shape_type& shape,
                    const shape_type& filter_shape)
                : m_input(input), m_output(output), m_filter(filter),
                  m_shape(shape), m_filter_shape(filter_shape),
                  m_halo(filter_shape / 2) {}

        constexpr void operator()(i64 i, i64 j, i64 k, i64 l) const noexcept {
            compute_type conv{0};
            if constexpr (DIM == 1) {
                for (i64 wl = 0; wl < m_filter_shape[0]; ++wl) {
                    const i64 il = l - m_halo[0] + wl;
                    if (il >= 0 && il < m_shape[0])
                        conv += static_cast<compute_type>(m_input(i, j, k, il)) * m_filter[wl];
                }
            } else if constexpr (DIM == 2) {
                for (i64 wk = 0; wk < m_filter_shape[0]; ++wk) {
                    const i64 ik = k - m_halo[0] + wk;
                    if (ik < 0 || ik >= m_shape[0])
                        continue;
                    const i64 tmp = wk * m_filter_shape[1];
                    for (i64 wl = 0; wl < m_filter_shape[1]; ++wl) {
                        const i64 il = l - m_halo[1] + wl;
                        if (il >= 0 && il < m_shape[1])
                            conv += static_cast<compute_type>(m_input(i, j, ik, il)) * m_filter[tmp + wl];
                    }
                }
            } else if constexpr (DIM == 3) {
                for (i64 wj = 0; wj < m_filter_shape[0]; ++wj) {
                    const i64 ij = j - m_halo[0] + wj;
                    if (ij < 0 || ij >= m_shape[0])
                        continue;
                    const i64 tmp_z = wj * m_filter_shape[1] * m_filter_shape[2];
                    for (i64 wk = 0; wk < m_filter_shape[1]; ++wk) {
                        const i64 ik = k - m_halo[1] + wk;
                        if (ik < 0 || ik >= m_shape[1])
                            continue;
                        const i64 tmp = tmp_z + wk * m_filter_shape[2];
                        for (i64 wl = 0; wl < m_filter_shape[2]; ++wl) {
                            const i64 il = l - m_halo[2] + wl;
                            if (il >= 0 && il < m_shape[2])
                                conv += static_cast<compute_type>(m_input(i, ij, ik, il)) * m_filter[tmp + wl];
                        }
                    }
                }
            } else {
                static_assert(nt::always_false_v<value_type>);
            }
            m_output(i, j, k, l) = static_cast<value_type>(conv);
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

    template<typename T, ConvolutionSeparableDim DIM>
    class ConvolutionSeparable {
    public:
        using value_type = T;
        using compute_type = std::conditional_t<std::is_same_v<f16, value_type>, f32, value_type>;
        using input_accessor_type = AccessorRestrict<const value_type, 4, i64>;
        using output_accessor_type = AccessorRestrict<value_type, 4, i64>;
        using filter_accessor_type = AccessorRestrictContiguous<const compute_type, 1, i64>;

    public:
        ConvolutionSeparable(const input_accessor_type& input,
                             const output_accessor_type& output,
                             const filter_accessor_type& filter,
                             i64 dim_size, i64 filter_size)
                : m_input(input), m_output(output), m_filter(filter),
                  m_dim_size(dim_size), m_filter_size(filter_size),
                  m_halo(filter_size / 2) {}

        constexpr void operator()(i64 i, i64 j, i64 k, i64 l) const noexcept {
            compute_type conv = 0;
            if constexpr (DIM == ConvolutionSeparableDim::WIDTH) {
                for (i64 wl = 0; wl < m_filter_size; ++wl) {
                    const i64 il = l - m_halo + wl;
                    if (il >= 0 && il < m_dim_size)
                        conv += static_cast<compute_type>(m_input(i, j, k, il)) * m_filter[wl];
                }
            } else if constexpr (DIM == ConvolutionSeparableDim::HEIGHT) {
                for (i64 wk = 0; wk < m_filter_size; ++wk) {
                    const i64 ik = k - m_halo + wk;
                    if (ik >= 0 && ik < m_dim_size)
                        conv += static_cast<compute_type>(m_input(i, j, ik, l)) * m_filter[wk];
                }
            } else if constexpr (DIM == ConvolutionSeparableDim::DEPTH) {
                for (i64 wj = 0; wj < m_filter_size; ++wj) {
                    const i64 ij = j - m_halo + wj;
                    if (ij >= 0 && ij < m_dim_size)
                        conv += static_cast<compute_type>(m_input(i, ij, k, l)) * m_filter[wj];
                }
            }
            m_output(i, j, k, l) = static_cast<T>(conv);
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
    auto get_filter_(const T* filter, i64 filter_size) {
        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        using buffer_t = typename noa::cpu::memory::AllocatorHeap<compute_t>::alloc_unique_type;
        using accessor_t = AccessorRestrictContiguous<const compute_t, 1, i64>;
        buffer_t buffer{};

        if constexpr (std::is_same_v<f16, T>) {
            buffer = noa::cpu::memory::AllocatorHeap<compute_t>::allocate(filter_size);
            for (size_t i = 0; i < static_cast<size_t>(filter_size); ++i)
                buffer[i] = static_cast<compute_t>(filter[i]);
            return std::pair{accessor_t{buffer.get()}, std::move(buffer)};
        } else {
            return std::pair{accessor_t{filter}, std::move(buffer)};
        }
    }

    template<ConvolutionSeparableDim DIM, typename T, typename U>
    void launch_convolve_separable_(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const U* filter, i64 filter_size, i64 threads) {

        if (filter_size == 1) {
            return noa::cpu::ewise_binary(
                    input, input_strides, static_cast<T>(filter[0]),
                    output, output_strides, shape,
                    noa::multiply_t{}, threads);
        }

        const auto input_accessor = AccessorRestrict<const T, 4, i64>(input, input_strides);
        const auto output_accessor = AccessorRestrict<T, 4, i64>(output, output_strides);
        const auto [filter_accessor, filter_buffer] = get_filter_(filter, filter_size);
        const auto dim_size = shape.filter(noa::to_underlying(DIM))[0];

        auto kernel = ConvolutionSeparable<T, DIM>(
                input_accessor, output_accessor, filter_accessor, dim_size, filter_size);
        noa::cpu::utils::iwise_4d(shape, kernel, threads);
    }
}

namespace noa::cpu::signal {
    template<typename T, typename U, typename>
    void convolve_1d(const T* input, const Strides4<i64>& input_strides,
                     T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                     const U* filter, const Shape1<i64>& filter_shape, i64 threads) {
        NOA_ASSERT(input != output && noa::all(shape > 0));
        NOA_ASSERT(noa::all((filter_shape % 2) == 1));
        if (noa::all(filter_shape == 1)) {
            return noa::cpu::ewise_binary(
                    input, input_strides, static_cast<T>(filter[0]),
                    output, output_strides, shape,
                    noa::multiply_t{}, threads);
        }

        const auto input_accessor = AccessorRestrict<const T, 4, i64>(input, input_strides);
        const auto output_accessor = AccessorRestrict<T, 4, i64>(output, output_strides);
        const auto [filter_accessor, filter_buffer] = get_filter_(filter, filter_shape.elements());
        const auto shape_1d = shape.filter(3);

        auto kernel = Convolution(input_accessor, output_accessor, filter_accessor, shape_1d, filter_shape);
        noa::cpu::utils::iwise_4d(shape, kernel, threads);
    }

    template<typename T, typename U, typename>
    void convolve_2d(const T* input, const Strides4<i64>& input_strides,
                     T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                     const U* filter, const Shape2<i64>& filter_shape, i64 threads) {
        NOA_ASSERT(input != output && noa::all(shape > 0));
        NOA_ASSERT(noa::all((filter_shape % 2) == 1));
        if (noa::all(filter_shape == 1)) {
            return noa::cpu::ewise_binary(
                    input, input_strides, static_cast<T>(filter[0]),
                    output, output_strides, shape,
                    noa::multiply_t{}, threads);
        }

        const auto input_accessor = AccessorRestrict<const T, 4, i64>(input, input_strides);
        const auto output_accessor = AccessorRestrict<T, 4, i64>(output, output_strides);
        const auto [filter_accessor, filter_buffer] = get_filter_(filter, filter_shape.elements());
        const auto shape_2d = shape.filter(2, 3);

        auto kernel = Convolution(input_accessor, output_accessor, filter_accessor, shape_2d, filter_shape);
        noa::cpu::utils::iwise_4d(shape, kernel, threads);
    }

    template<typename T, typename U, typename>
    void convolve_3d(const T* input, const Strides4<i64>& input_strides,
                     T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                     const U* filter, const Shape3<i64>& filter_shape, i64 threads) {
        NOA_ASSERT(input != output && noa::all(shape > 0));
        NOA_ASSERT(noa::all((filter_shape % 2) == 1));
        if (noa::all(filter_shape == 1)) {
            return noa::cpu::ewise_binary(
                    input, input_strides, static_cast<T>(filter[0]),
                    output, output_strides, shape,
                    noa::multiply_t{}, threads);
        }

        const auto input_accessor = AccessorRestrict<const T, 4, i64>(input, input_strides);
        const auto output_accessor = AccessorRestrict<T, 4, i64>(output, output_strides);
        const auto [filter_accessor, filter_buffer] = get_filter_(filter, filter_shape.elements());
        const auto shape_3d = shape.filter(1, 2, 3);

        auto kernel = Convolution(input_accessor, output_accessor, filter_accessor, shape_3d, filter_shape);
        noa::cpu::utils::iwise_4d(shape, kernel, threads);
    }

    template<typename T, typename U, typename>
    void convolve_separable(const T* input, const Strides4<i64>& input_strides,
                            T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                            const U* filter_depth, i64 filter_depth_size,
                            const U* filter_height, i64 filter_height_size,
                            const U* filter_width, i64 filter_width_size,
                            T* tmp, Strides4<i64> tmp_strides, i64 threads) {
        NOA_ASSERT(input != output && noa::all(shape > 0));

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
        using allocator_t = noa::cpu::memory::AllocatorHeap<T>;
        typename allocator_t::alloc_unique_type buffer{};
        if (!tmp && count > 1) {
            buffer = allocator_t::allocate(shape.elements());
            tmp = buffer.get();
            tmp_strides = shape.strides();
        }

        NOA_ASSERT(!filter_depth || filter_depth_size % 2);
        NOA_ASSERT(!filter_height || filter_height_size % 2);
        NOA_ASSERT(!filter_width || filter_width_size % 2);

        if (filter_depth && filter_height && filter_width) {
            launch_convolve_separable_<ConvolutionSeparableDim::DEPTH>(
                    input, input_strides, output, output_strides, shape,
                    filter_depth, filter_depth_size, threads);
            launch_convolve_separable_<ConvolutionSeparableDim::HEIGHT>(
                    output, output_strides, tmp, tmp_strides, shape,
                    filter_height, filter_height_size, threads);
            launch_convolve_separable_<ConvolutionSeparableDim::WIDTH>(
                    tmp, tmp_strides, output, output_strides, shape,
                    filter_width, filter_width_size, threads);

        } else if (filter_depth && filter_height) {
            launch_convolve_separable_<ConvolutionSeparableDim::DEPTH>(
                    input, input_strides, tmp, tmp_strides, shape,
                    filter_depth, filter_depth_size, threads);
            launch_convolve_separable_<ConvolutionSeparableDim::HEIGHT>(
                    tmp, tmp_strides, output, output_strides, shape,
                    filter_height, filter_height_size, threads);

        } else if (filter_height && filter_width) {
            launch_convolve_separable_<ConvolutionSeparableDim::HEIGHT>(
                    input, input_strides, tmp, tmp_strides, shape,
                    filter_height, filter_height_size, threads);
            launch_convolve_separable_<ConvolutionSeparableDim::WIDTH>(
                    tmp, tmp_strides, output, output_strides, shape,
                    filter_width, filter_width_size, threads);

        } else if (filter_depth && filter_width) {
            launch_convolve_separable_<ConvolutionSeparableDim::DEPTH>(
                    input, input_strides, tmp, tmp_strides, shape,
                    filter_depth, filter_depth_size, threads);
            launch_convolve_separable_<ConvolutionSeparableDim::WIDTH>(
                    tmp, tmp_strides, output, output_strides, shape,
                    filter_width, filter_width_size, threads);

        } else if (filter_depth) {
            launch_convolve_separable_<ConvolutionSeparableDim::DEPTH>(
                    input, input_strides, output, output_strides, shape,
                    filter_depth, filter_depth_size, threads);
        } else if (filter_height) {
            launch_convolve_separable_<ConvolutionSeparableDim::HEIGHT>(
                    input, input_strides, output, output_strides, shape,
                    filter_height, filter_height_size, threads);
        } else if (filter_width) {
            launch_convolve_separable_<ConvolutionSeparableDim::WIDTH>(
                    input, input_strides, output, output_strides, shape,
                    filter_width, filter_width_size, threads);
        }
    }

    template<typename T, typename U, typename>
    void convolve(const T* input, const Strides4<i64>& input_strides,
                  T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                  const U* filter, const Shape3<i64>& filter_shape, i64 threads)  {
        NOA_ASSERT(noa::all(filter_shape > 0) && noa::all(shape > 0));

        // If there's a single dimension, use separable convolution kernels:
        const auto dimensions_to_convolve = noa::math::sum((filter_shape > 1).as<i32>());
        const auto ndim = filter_shape.ndim();

        if (dimensions_to_convolve == 1) {
            if (filter_shape[0] > 1) {
                launch_convolve_separable_<ConvolutionSeparableDim::DEPTH>(
                        input, input_strides, output, output_strides, shape,
                        filter, filter_shape[0], threads);
            } else if (filter_shape[1] > 1) {
                launch_convolve_separable_<ConvolutionSeparableDim::HEIGHT>(
                        input, input_strides, output, output_strides, shape,
                        filter, filter_shape[1], threads);
            } else {
                launch_convolve_separable_<ConvolutionSeparableDim::WIDTH>(
                        input, input_strides, output, output_strides, shape,
                        filter, filter_shape[2], threads);
            }
        } else if (ndim == 2) {
            return convolve_2d(input, input_strides, output, output_strides,
                               shape, filter, filter_shape.pop_front(), threads);
        } else if (ndim == 3) {
            return convolve_3d(input, input_strides, output, output_strides,
                               shape, filter, filter_shape, threads);
        } else if (noa::all(filter_shape == 1)) {
            return noa::cpu::ewise_binary(
                    input, input_strides, static_cast<T>(filter[0]),
                    output, output_strides, shape,
                    noa::multiply_t{}, threads);
        } else {
            NOA_THROW("DEV: unreachable");
        }
    }

    #define NOA_INSTANTIATE_CONV_(T, U)                 \
    template void convolve_1d<T, U, void>(              \
        const T*, const Strides4<i64>&,                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const U*, const Shape1<i64>&, i64);             \
    template void convolve_2d<T, U, void>(              \
        const T*, const Strides4<i64>&,                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const U*, const Shape2<i64>&, i64);             \
    template void convolve_3d<T, U, void>(              \
        const T*, const Strides4<i64>&,                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const U*, const Shape3<i64>&, i64);             \
    template void convolve_separable<T, U, void>(       \
        const T*, const Strides4<i64>&,                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const U*, i64, const U*, i64, const U*, i64,    \
        T*, Strides4<i64>, i64);                        \
    template void convolve<T, U, void>(                 \
        const T*, const Strides4<i64>&,                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const U*, const Shape3<i64>&, i64)

//    NOA_INSTANTIATE_CONV_(f16, f16);
//    NOA_INSTANTIATE_CONV_(f16, f32);
    NOA_INSTANTIATE_CONV_(f32, f32);
    NOA_INSTANTIATE_CONV_(f64, f64);
}
