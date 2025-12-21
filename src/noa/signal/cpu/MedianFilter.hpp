#pragma once

#include <algorithm> // std::nth_element

#include "noa/runtime/cpu/Copy.hpp"
#include "noa/runtime/cpu/Allocators.hpp"
#include "noa/runtime/cpu/Iwise.hpp"

namespace noa::signal::cpu::details {
    // If index is out of bound, apply mirror (d c b | a b c d | c b a).
    // This requires shape >= window/2+1. noa::indexing::index_at() would work, but this
    // is slightly more efficient because we know the size of the halo is less than the size.
    constexpr auto median_filter_get_mirror_index(isize index, isize size) -> isize {
        if (index < 0)
            index *= -1;
        else if (index >= size)
            index = 2 * (size - 1) - index;
        return index;
    }

    template<Border MODE, typename InputAccessor, typename OutputAccessor, typename BufferAccessor>
    class MedianFilter1d {
    public:
        MedianFilter1d(
            const InputAccessor& input,
            const OutputAccessor& output,
            BufferAccessor buffer,
            const Shape4& shape,
            isize window
        ) : m_input(input), m_output(output), m_buffer(buffer),
            m_width(shape.width()), m_window(window),
            m_window_half(window / 2) {}

        void init(isize thread) noexcept {
            // Before starting the loop, offset to the thread workspace.
            m_buffer = BufferAccessor(m_buffer.get() + thread * m_window);
        }

        void operator()(isize i, isize j, isize k, isize l) const noexcept {
            using compute_type = BufferAccessor::value_type;
            using output_type = OutputAccessor::value_type;

            // Gather the window.
            if constexpr (MODE == Border::REFLECT) {
                for (isize wl{}; wl < m_window; ++wl) {
                    const isize il = median_filter_get_mirror_index(l - m_window_half + wl, m_width);
                    m_buffer[wl] = static_cast<compute_type>(m_input(i, j, k, il));
                }
            } else { // Border::ZERO
                for (isize wl{}; wl < m_window; ++wl) {
                    const isize il = l - m_window_half + wl;
                    if (il < 0 or il >= m_width)
                        m_buffer[wl] = compute_type{};
                    else
                        m_buffer[wl] = static_cast<compute_type>(m_input(i, j, k, il));
                }
            }

            // Sort the elements in the m_window to get the median.
            std::nth_element(m_buffer.get(), m_buffer.get() + m_window_half, m_buffer.get() + m_window);
            m_output(i, j, k, l) = static_cast<output_type>(m_buffer[m_window_half]);
        }

    private:
        InputAccessor m_input;
        OutputAccessor m_output;
        BufferAccessor m_buffer{};
        isize m_width;
        isize m_window;
        isize m_window_half;
    };

    template<Border MODE, typename InputAccessor, typename OutputAccessor, typename BufferAccessor>
    class MedianFilter2d {
    public:
        MedianFilter2d(
            const InputAccessor& input,
            const OutputAccessor& output,
            BufferAccessor buffer,
            const Shape4& shape,
            isize window
        ) : m_input(input), m_output(output), m_buffer(buffer),
            m_shape(shape.filter(2, 3)),
            m_window_1d(window),
            m_window_1d_half(window / 2),
            m_window_size(window * window),
            m_window_half((window * window) / 2) {}

        void init(isize thread) noexcept {
            // Before starting the loop, offset to the thread buffer.
            m_buffer = BufferAccessor(m_buffer.get() + thread * m_window_size);
        }

        void operator()(isize i, isize j, isize k, isize l) const noexcept {
            using compute_type = BufferAccessor::value_type;
            using output_type = OutputAccessor::value_type;

            if constexpr (MODE == Border::REFLECT) {
                for (isize wk{}; wk < m_window_1d; ++wk) {
                    const isize ik = median_filter_get_mirror_index(k - m_window_1d_half + wk, m_shape[0]);
                    for (isize wl{}; wl < m_window_1d; ++wl) {
                        const isize il = median_filter_get_mirror_index(l - m_window_1d_half + wl, m_shape[1]);
                        m_buffer[wk * m_window_1d + wl] = static_cast<compute_type>(m_input(i, j, ik, il));
                    }
                }
            } else { // Border::ZERO
                for (isize wk{}; wk < m_window_1d; ++wk) {
                    const isize ik = k - m_window_1d_half + wk;
                    for (isize wl{}; wl < m_window_1d; ++wl) {
                        const isize il = l - m_window_1d_half + wl;
                        if (ik < 0 or ik >= m_shape[0] or il < 0 or il >= m_shape[1]) {
                            m_buffer[wk * m_window_1d + wl] = compute_type{};
                        } else {
                            m_buffer[wk * m_window_1d + wl] = static_cast<compute_type>(m_input(i, j, ik, il));
                        }
                    }
                }
            }

            // Sort the elements in the m_window to get the median.
            std::nth_element(m_buffer.get(), m_buffer.get() + m_window_half, m_buffer.get() + m_window_size);
            m_output(i, j, k, l) = static_cast<output_type>(m_buffer[m_window_half]);
        }

    private:
        InputAccessor m_input;
        OutputAccessor m_output;
        BufferAccessor m_buffer{};
        Shape2 m_shape;
        isize m_window_1d;
        isize m_window_1d_half;
        isize m_window_size;
        isize m_window_half;
    };

    template<Border MODE, typename InputAccessor, typename OutputAccessor, typename BufferAccessor>
    class MedianFilter3d {
    public:
        MedianFilter3d(
            const InputAccessor& input,
            const OutputAccessor& output,
            const BufferAccessor& buffer,
            const Shape4& shape,
            isize window
        ) : m_input(input), m_output(output), m_buffer(buffer),
            m_shape(shape.pop_front()),
            m_window_1d(window),
            m_window_1d_half(window / 2),
            m_window_size(window * window * window),
            m_window_half((window * window * window) / 2) {}

        void init(isize thread) noexcept {
            // Before starting the loop, offset to the thread buffer.
            m_buffer = BufferAccessor(m_buffer.get() + thread * m_window_size);
        }

        void operator()(isize i, isize j, isize k, isize l) const noexcept {
            using compute_type = BufferAccessor::value_type;
            using output_type = OutputAccessor::value_type;

            if constexpr (MODE == Border::REFLECT) {
                for (isize wj{}; wj < m_window_1d; ++wj) {
                    const isize ij = median_filter_get_mirror_index(j - m_window_1d_half + wj, m_shape[0]);
                    for (isize wk{}; wk < m_window_1d; ++wk) {
                        const isize ik = median_filter_get_mirror_index(k - m_window_1d_half + wk, m_shape[1]);
                        for (isize wl{}; wl < m_window_1d; ++wl) {
                            const isize il = median_filter_get_mirror_index(l - m_window_1d_half + wl, m_shape[2]);
                            m_buffer[(wj * m_window_1d + wk) * m_window_1d + wl] =
                                    static_cast<compute_type>(m_input(i, ij, ik, il));
                        }
                    }
                }
            } else { // Border::ZERO
                for (isize wj{}; wj < m_window_1d; ++wj) {
                    const isize ij = j - m_window_1d_half + wj;
                    for (isize wk{}; wk < m_window_1d; ++wk) {
                        const isize ik = k - m_window_1d_half + wk;
                        for (isize wl{}; wl < m_window_1d; ++wl) {
                            const isize il = l - m_window_1d_half + wl;
                            const isize idx = (wj * m_window_1d + wk) * m_window_1d + wl;
                            if (ij < 0 or ij >= m_shape[0] or
                                ik < 0 or ik >= m_shape[1] or
                                il < 0 or il >= m_shape[2]) {
                                m_buffer[idx] = compute_type{};
                            } else {
                                m_buffer[idx] = static_cast<compute_type>(m_input(i, ij, ik, il));
                            }
                        }
                    }
                }
            }

            // Sort the elements in the window to get the median.
            std::nth_element(m_buffer.get(), m_buffer.get() + m_window_half, m_buffer.get() + m_window_size);
            m_output(i, j, k, l) = static_cast<output_type>(m_buffer[m_window_half]);
        }

    private:
        InputAccessor m_input;
        OutputAccessor m_output;
        BufferAccessor m_buffer{};
        Shape3 m_shape;
        isize m_window_1d;
        isize m_window_1d_half;
        isize m_window_size;
        isize m_window_half;
    };
}

namespace noa::signal::cpu {
    template<typename T, typename U>
    void median_filter_1d(
        const T* input, const Strides4& input_strides,
        U* output, const Strides4& output_strides,
        const Shape4& shape, Border border_mode,
        isize window_size, isize n_threads
    ) {
        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        const auto buffer = noa::cpu::AllocatorHeap::allocate<compute_t>(window_size * n_threads);

        using input_accessor_t = AccessorRestrict<const T, 4, isize>;
        using output_accessor_t = AccessorRestrict<U, 4, isize>;
        using buffer_accessor_t = AccessorRestrictContiguous<compute_t, 1, isize>;

        switch (border_mode) {
            case Border::REFLECT: {
                NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                auto op = details::MedianFilter1d<Border::REFLECT, input_accessor_t, output_accessor_t, buffer_accessor_t>(
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    buffer_accessor_t(buffer.get()),
                    shape, window_size);
                return noa::cpu::iwise(shape, op, n_threads);
            }
            case Border::ZERO: {
                auto op = details::MedianFilter1d<Border::ZERO, input_accessor_t, output_accessor_t, buffer_accessor_t>(
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    buffer_accessor_t(buffer.get()),
                    shape, window_size);
                return noa::cpu::iwise(shape, op, n_threads);
            }
            default:
                panic("Border not supported. Should be {} or {}, got {}",
                      Border::ZERO, Border::REFLECT, border_mode);
        }
    }

    template<typename T, typename U>
    void median_filter_2d(
        const T* input, Strides4 input_strides,
        U* output, Strides4 output_strides,
        Shape4 shape, Border border_mode, isize window_size, isize n_threads
    ) {
        const auto order_2d = ni::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (order_2d != Vec<isize, 2>{0, 1}) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
        }

        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        const auto buffer = noa::cpu::AllocatorHeap::allocate<compute_t>(window_size * window_size * n_threads);

        using input_accessor_t = AccessorRestrict<const T, 4, isize>;
        using output_accessor_t = AccessorRestrict<U, 4, isize>;
        using buffer_accessor_t = AccessorRestrictContiguous<compute_t, 1, isize>;

        switch (border_mode) {
            case Border::REFLECT: {
                auto op = details::MedianFilter2d<Border::REFLECT, input_accessor_t, output_accessor_t, buffer_accessor_t>(
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    buffer_accessor_t(buffer.get()),
                    shape, window_size);
                return noa::cpu::iwise(shape, op, n_threads);
            }
            case Border::ZERO: {
                NOA_ASSERT(window_size / 2 + 1 <= shape[2]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                auto op = details::MedianFilter2d<Border::ZERO, input_accessor_t, output_accessor_t, buffer_accessor_t>(
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    buffer_accessor_t(buffer.get()),
                    shape, window_size);
                return noa::cpu::iwise(shape, op, n_threads);
            }
            default:
                panic("Border not supported. Should be {} or {}, got {}",
                      Border::ZERO, Border::REFLECT, border_mode);
        }
    }

    template<typename T, typename U>
    void median_filter_3d(
        const T* input, Strides4 input_strides,
        U* output, Strides4 output_strides,
        Shape4 shape, Border border_mode, isize window_size, isize n_threads
    ) {
        const auto order_3d = ni::order(output_strides.pop_front(), shape.pop_front());
        if (order_3d != Vec<isize, 3>{0, 1, 2}) {
            const auto order = (order_3d + 1).push_front(0);
            input_strides = ni::reorder(input_strides, order);
            output_strides = ni::reorder(output_strides, order);
            shape = ni::reorder(shape, order);
        }

        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        const auto buffer = noa::cpu::AllocatorHeap::allocate<compute_t>(window_size * window_size * window_size * n_threads);

        using input_accessor_t = AccessorRestrict<const T, 4, isize>;
        using output_accessor_t = AccessorRestrict<U, 4, isize>;
        using buffer_accessor_t = AccessorRestrictContiguous<compute_t, 1, isize>;

        switch (border_mode) {
            case Border::REFLECT: {
                NOA_ASSERT(window_size / 2 + 1 <= shape[1]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[2]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                auto op = details::MedianFilter3d<Border::REFLECT, input_accessor_t, output_accessor_t, buffer_accessor_t>(
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    buffer_accessor_t(buffer.get()),
                    shape, window_size);
                return noa::cpu::iwise(shape, op, n_threads);
            }
            case Border::ZERO: {
                auto op = details::MedianFilter3d<Border::ZERO, input_accessor_t, output_accessor_t, buffer_accessor_t>(
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    buffer_accessor_t(buffer.get()),
                    shape, window_size);
                return noa::cpu::iwise(shape, op, n_threads);
            }
            default:
                panic("Border not supported. Should be {} or {}, got {}",
                      Border::ZERO, Border::REFLECT, border_mode);
        }
    }
}
