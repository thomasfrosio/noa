#pragma once

#include <algorithm> // std::nth_element

#include "noa/runtime/cpu/Copy.hpp"
#include "noa/runtime/cpu/Allocators.hpp"
#include "noa/runtime/cpu/Iwise.hpp"

namespace noa::signal::cpu::details {
    // If index is out of bound, apply mirror (d c b | a b c d | c b a).
    // This requires shape >= window/2+1. noa::index_at() would work, but this
    // is slightly more efficient because we know the size of the halo is less than the size.
    constexpr auto median_filter_get_mirror_index(isize index, isize size) -> isize {
        if (index < 0)
            index *= -1;
        else if (index >= size)
            index = 2 * (size - 1) - index;
        return index;
    }

    template<usize N, usize I, Border MODE, typename Input, typename Output, typename Buffer>
    class MedianFilter1d {
    public:
        MedianFilter1d(
            const Input& input,
            const Output& output,
            Buffer buffer,
            const Shape<isize, N>& shape,
            isize window
        ) : m_input(input), m_output(output), m_buffer(buffer),
            m_dim_size(shape[I]), m_window(window),
            m_window_half(window / 2) {}

        using remove_default_init = bool;
        void init(nt::compute_handle auto& handle) noexcept {
            // Before starting the loop, offset to the thread workspace.
            m_buffer = Buffer(m_buffer.get() + handle.thread().gid() * m_window);
        }

        void operator()(const Vec<isize, N>& output_indices) const noexcept {
            using compute_type = Buffer::value_type;
            using output_type = Output::value_type;
            auto input_indices = output_indices;

            // Gather the window.
            if constexpr (MODE == Border::REFLECT) {
                for (isize i{}; i < m_window; ++i) {
                    input_indices[I] = median_filter_get_mirror_index(i, output_indices[I] - m_window_half + i);
                    m_buffer[i] = static_cast<compute_type>(m_input(input_indices));
                }
            } else { // Border::ZERO
                for (isize i{}; i < m_window; ++i) {
                    input_indices[I] = output_indices[I] - m_window_half + i;
                    if (input_indices[I] < 0 or input_indices[I] >= m_dim_size)
                        m_buffer[i] = compute_type{};
                    else
                        m_buffer[i] = static_cast<compute_type>(m_input(input_indices));
                }
            }

            // Sort the elements in the m_window to get the median.
            std::nth_element(m_buffer.get(), m_buffer.get() + m_window_half, m_buffer.get() + m_window);
            m_output(output_indices) = static_cast<output_type>(m_buffer[m_window_half]);
        }

    private:
        Input m_input;
        Output m_output;
        Buffer m_buffer{};
        isize m_dim_size;
        isize m_window;
        isize m_window_half;
    };

    template<usize N, usize I, usize J, Border MODE, typename Input, typename Output, typename Buffer>
    class MedianFilter2d {
    public:
        MedianFilter2d(
            const Input& input,
            const Output& output,
            Buffer buffer,
            const Shape<isize, N>& shape,
            isize window
        ) : m_input(input), m_output(output), m_buffer(buffer),
            m_shape(shape.filter(I, J)),
            m_window_1d(window),
            m_window_1d_half(window / 2),
            m_window_size(window * window),
            m_window_half((window * window) / 2) {}

        using remove_default_init = bool;
        void init(nt::compute_handle auto& handle) noexcept {
            // Before starting the loop, offset to the thread buffer.
            m_buffer = Buffer(m_buffer.get() + handle.thread().gid() * m_window_size);
        }

        void operator()(const Vec<isize, N>& output_indices) const noexcept {
            using compute_type = Buffer::value_type;
            using output_type = Output::value_type;
            auto input_indices = output_indices;

            if constexpr (MODE == Border::REFLECT) {
                for (isize i{}; i < m_window_1d; ++i) {
                    input_indices[I] = median_filter_get_mirror_index(output_indices[I] - m_window_1d_half + i, m_shape[0]);
                    for (isize j{}; j < m_window_1d; ++j) {
                        input_indices[J] = median_filter_get_mirror_index(output_indices[J] - m_window_1d_half + j, m_shape[1]);
                        m_buffer[i * m_window_1d + j] = static_cast<compute_type>(m_input(input_indices));
                    }
                }
            } else { // Border::ZERO
                for (isize i{}; i < m_window_1d; ++i) {
                    input_indices[I] = output_indices[I] - m_window_1d_half + i;
                    for (isize j{}; j < m_window_1d; ++j) {
                        input_indices[J] = output_indices[J] - m_window_1d_half + j;
                        if (input_indices[I] < 0 or input_indices[I] >= m_shape[0] or
                            input_indices[J] < 0 or input_indices[J] >= m_shape[1]) {
                            m_buffer[i * m_window_1d + j] = compute_type{};
                        } else {
                            m_buffer[i * m_window_1d + j] = static_cast<compute_type>(m_input(input_indices));
                        }
                    }
                }
            }

            // Sort the elements in the m_window to get the median.
            std::nth_element(m_buffer.get(), m_buffer.get() + m_window_half, m_buffer.get() + m_window_size);
            m_output(output_indices) = static_cast<output_type>(m_buffer[m_window_half]);
        }

    private:
        Input m_input;
        Output m_output;
        Buffer m_buffer{};
        Shape2 m_shape;
        isize m_window_1d;
        isize m_window_1d_half;
        isize m_window_size;
        isize m_window_half;
    };

    template<usize N, usize I, usize J, usize K, Border MODE, typename Input, typename Output, typename Buffer>
    class MedianFilter3d {
    public:
        MedianFilter3d(
            const Input& input,
            const Output& output,
            const Buffer& buffer,
            const Shape4& shape,
            isize window
        ) : m_input(input), m_output(output), m_buffer(buffer),
            m_shape(shape.filter(I, J, K)),
            m_window_1d(window),
            m_window_1d_half(window / 2),
            m_window_size(window * window * window),
            m_window_half((window * window * window) / 2) {}

        using remove_default_init = bool;
        void init(nt::compute_handle auto& handle) noexcept {
            // Before starting the loop, offset to the thread buffer.
            m_buffer = Buffer(m_buffer.get() + handle.thread().gid() * m_window_size);
        }

        void operator()(const Vec<isize, N>& output_indices) const noexcept {
            using compute_type = Buffer::value_type;
            using output_type = Output::value_type;
            auto input_indices = output_indices;

            if constexpr (MODE == Border::REFLECT) {
                for (isize i{}; i < m_window_1d; ++i) {
                    input_indices[I] = median_filter_get_mirror_index(output_indices[I] - m_window_1d_half + i, m_shape[0]);
                    for (isize j{}; j < m_window_1d; ++j) {
                        input_indices[J] = median_filter_get_mirror_index(output_indices[J] - m_window_1d_half + j, m_shape[1]);
                        for (isize k{}; k < m_window_1d; ++k) {
                            input_indices[K] = median_filter_get_mirror_index(output_indices[K] - m_window_1d_half + k, m_shape[2]);
                            m_buffer[(i * m_window_1d + j) * m_window_1d + k] = static_cast<compute_type>(m_input(input_indices));
                        }
                    }
                }
            } else { // Border::ZERO
                for (isize i{}; i < m_window_1d; ++i) {
                    input_indices[I] = output_indices[I] - m_window_1d_half + i;
                    for (isize j{}; j < m_window_1d; ++j) {
                        input_indices[J] = output_indices[J] - m_window_1d_half + j;
                        for (isize k{}; k < m_window_1d; ++k) {
                            input_indices[K] = output_indices[K] - m_window_1d_half + k;
                            const isize idx = (i * m_window_1d + j) * m_window_1d + k;
                            if (input_indices[I] < 0 or input_indices[I] >= m_shape[0] or
                                input_indices[J] < 0 or input_indices[J] >= m_shape[1] or
                                input_indices[K] < 0 or input_indices[K] >= m_shape[2]) {
                                m_buffer[idx] = compute_type{};
                            } else {
                                m_buffer[idx] = static_cast<compute_type>(m_input(input_indices));
                            }
                        }
                    }
                }
            }

            // Sort the elements in the window to get the median.
            std::nth_element(m_buffer.get(), m_buffer.get() + m_window_half, m_buffer.get() + m_window_size);
            m_output(output_indices) = static_cast<output_type>(m_buffer[m_window_half]);
        }

    private:
        Input m_input;
        Output m_output;
        Buffer m_buffer{};
        Shape3 m_shape;
        isize m_window_1d;
        isize m_window_1d_half;
        isize m_window_size;
        isize m_window_half;
    };
}

namespace noa::signal::cpu {
    template<typename T, typename U, usize N>
    void median_filter_1d(
        const T* input, const Strides<isize, N>& input_strides,
        U* output, const Strides<isize, N>& output_strides,
        const Shape<isize, N>& shape, Border border_mode,
        isize window_size, i32 n_threads
    ) {
        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        const auto buffer = noa::cpu::AllocatorHeap::allocate<compute_t>(window_size * n_threads);

        using input_t = AccessorRestrict<const T, N, isize>;
        using output_t = AccessorRestrict<U, N, isize>;
        using buffer_t = AccessorRestrictContiguous<compute_t, 1, isize>;

        switch (border_mode) {
            case Border::REFLECT: {
                NOA_ASSERT(window_size / 2 + 1 <= shape[N - 1]);
                auto op = details::MedianFilter1d<N, N - 1, Border::REFLECT, input_t, output_t, buffer_t>(
                    input_t(input, input_strides),
                    output_t(output, output_strides),
                    buffer_t(buffer.get()),
                    shape, window_size);
                return noa::cpu::iwise(shape, op, n_threads);
            }
            case Border::ZERO: {
                auto op = details::MedianFilter1d<N, N - 1, Border::ZERO, input_t, output_t, buffer_t>(
                    input_t(input, input_strides),
                    output_t(output, output_strides),
                    buffer_t(buffer.get()),
                    shape, window_size);
                return noa::cpu::iwise(shape, op, n_threads);
            }
            default:
                panic("Border not supported. Should be {} or {}, got {}",
                      Border::ZERO, Border::REFLECT, border_mode);
        }
    }

    template<typename T, typename U, usize N>
    void median_filter_2d(
        const T* input, const Strides<isize, N>& input_strides,
        U* output, const Strides<isize, N>& output_strides,
        const Shape<isize, N>& shape, Border border_mode, isize window_size, i32 n_threads
    ) {
        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        const auto buffer = noa::cpu::AllocatorHeap::allocate<compute_t>(window_size * window_size * n_threads);

        using input_t = AccessorRestrict<const T, N, isize>;
        using output_t = AccessorRestrict<U, N, isize>;
        using buffer_t = AccessorRestrictContiguous<compute_t, 1, isize>;

        switch (border_mode) {
            case Border::REFLECT: {
                auto op = details::MedianFilter2d<N, N - 2, N - 1, Border::REFLECT, input_t, output_t, buffer_t>(
                    input_t(input, input_strides),
                    output_t(output, output_strides),
                    buffer_t(buffer.get()),
                    shape, window_size);
                return noa::cpu::iwise(shape, op, n_threads);
            }
            case Border::ZERO: {
                NOA_ASSERT(window_size / 2 + 1 <= shape[N - 2]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[N - 1]);
                auto op = details::MedianFilter2d<N, N - 2, N - 1, Border::ZERO, input_t, output_t, buffer_t>(
                    input_t(input, input_strides),
                    output_t(output, output_strides),
                    buffer_t(buffer.get()),
                    shape, window_size);
                return noa::cpu::iwise(shape, op, n_threads);
            }
            default:
                panic("Border not supported. Should be {} or {}, got {}",
                      Border::ZERO, Border::REFLECT, border_mode);
        }
    }

    template<typename T, typename U, usize N>
    void median_filter_3d(
        const T* input, const Strides<isize, N>& input_strides,
        U* output, const Strides<isize, N>& output_strides,
        const Shape<isize, N>& shape, Border border_mode, isize window_size, i32 n_threads
    ) {
        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        const auto buffer = noa::cpu::AllocatorHeap::allocate<compute_t>(window_size * window_size * window_size * n_threads);

        using input_t = AccessorRestrict<const T, N, isize>;
        using output_t = AccessorRestrict<U, N, isize>;
        using buffer_t = AccessorRestrictContiguous<compute_t, 1, isize>;

        switch (border_mode) {
            case Border::REFLECT: {
                NOA_ASSERT(window_size / 2 + 1 <= shape[N - 3]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[N - 2]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[N - 1]);
                auto op = details::MedianFilter3d<N, N - 3, N - 2, N - 1, Border::REFLECT, input_t, output_t, buffer_t>(
                    input_t(input, input_strides),
                    output_t(output, output_strides),
                    buffer_t(buffer.get()),
                    shape, window_size);
                return noa::cpu::iwise(shape, op, n_threads);
            }
            case Border::ZERO: {
                auto op = details::MedianFilter3d<N, N - 3, N - 2, N - 1, Border::ZERO, input_t, output_t, buffer_t>(
                    input_t(input, input_strides),
                    output_t(output, output_strides),
                    buffer_t(buffer.get()),
                    shape, window_size);
                return noa::cpu::iwise(shape, op, n_threads);
            }
            default:
                panic("Border not supported. Should be {} or {}, got {}",
                      Border::ZERO, Border::REFLECT, border_mode);
        }
    }
}
