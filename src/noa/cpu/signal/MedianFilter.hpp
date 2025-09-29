#pragma once

#include <algorithm> // std::nth_element

#include "noa/cpu/Copy.hpp"
#include "noa/cpu/Allocators.hpp"
#include "noa/cpu/Iwise.hpp"

namespace noa::cpu::signal::guts {
    // If index is out of bound, apply mirror (d c b | a b c d | c b a).
    // This requires shape >= window/2+1. noa::indexing::index_at() would work, but this
    // is slightly more efficient because we know the size of the halo is less than the size.
    constexpr i64 median_filter_get_mirror_index(i64 index, i64 size) {
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
            const Shape4<i64>& shape,
            i64 window
        ) : m_input(input), m_output(output), m_buffer(buffer),
            m_width(shape.width()), m_window(window),
            m_window_half(window / 2) {}

        void init(i64 thread) noexcept {
            // Before starting the loop, offset to the thread workspace.
            m_buffer = BufferAccessor(m_buffer.get() + thread * m_window);
        }

        void operator()(i64 i, i64 j, i64 k, i64 l) const noexcept {
            using compute_type = BufferAccessor::value_type;
            using output_type = OutputAccessor::value_type;

            // Gather the window.
            if constexpr (MODE == Border::REFLECT) {
                for (i64 wl{}; wl < m_window; ++wl) {
                    const i64 il = median_filter_get_mirror_index(l - m_window_half + wl, m_width);
                    m_buffer[wl] = static_cast<compute_type>(m_input(i, j, k, il));
                }
            } else { // Border::ZERO
                for (i64 wl{}; wl < m_window; ++wl) {
                    const i64 il = l - m_window_half + wl;
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
        i64 m_width;
        i64 m_window;
        i64 m_window_half;
    };

    template<Border MODE, typename InputAccessor, typename OutputAccessor, typename BufferAccessor>
    class MedianFilter2d {
    public:
        MedianFilter2d(
            const InputAccessor& input,
            const OutputAccessor& output,
            BufferAccessor buffer,
            const Shape4<i64>& shape,
            i64 window
        ) : m_input(input), m_output(output), m_buffer(buffer),
            m_shape(shape.filter(2, 3)),
            m_window_1d(window),
            m_window_1d_half(window / 2),
            m_window_size(window * window),
            m_window_half((window * window) / 2) {}

        void init(i64 thread) noexcept {
            // Before starting the loop, offset to the thread buffer.
            m_buffer = BufferAccessor(m_buffer.get() + thread * m_window_size);
        }

        void operator()(i64 i, i64 j, i64 k, i64 l) const noexcept {
            using compute_type = BufferAccessor::value_type;
            using output_type = OutputAccessor::value_type;

            if constexpr (MODE == Border::REFLECT) {
                for (i64 wk{}; wk < m_window_1d; ++wk) {
                    const i64 ik = median_filter_get_mirror_index(k - m_window_1d_half + wk, m_shape[0]);
                    for (i64 wl{}; wl < m_window_1d; ++wl) {
                        const i64 il = median_filter_get_mirror_index(l - m_window_1d_half + wl, m_shape[1]);
                        m_buffer[wk * m_window_1d + wl] = static_cast<compute_type>(m_input(i, j, ik, il));
                    }
                }
            } else { // Border::ZERO
                for (i64 wk{}; wk < m_window_1d; ++wk) {
                    const i64 ik = k - m_window_1d_half + wk;
                    for (i64 wl{}; wl < m_window_1d; ++wl) {
                        const i64 il = l - m_window_1d_half + wl;
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
        Shape2<i64> m_shape;
        i64 m_window_1d;
        i64 m_window_1d_half;
        i64 m_window_size;
        i64 m_window_half;
    };

    template<Border MODE, typename InputAccessor, typename OutputAccessor, typename BufferAccessor>
    class MedianFilter3d {
    public:
        MedianFilter3d(
            const InputAccessor& input,
            const OutputAccessor& output,
            const BufferAccessor& buffer,
            const Shape4<i64>& shape,
            i64 window
        ) : m_input(input), m_output(output), m_buffer(buffer),
            m_shape(shape.pop_front()),
            m_window_1d(window),
            m_window_1d_half(window / 2),
            m_window_size(window * window * window),
            m_window_half((window * window * window) / 2) {}

        void init(i64 thread) noexcept {
            // Before starting the loop, offset to the thread buffer.
            m_buffer = BufferAccessor(m_buffer.get() + thread * m_window_size);
        }

        void operator()(i64 i, i64 j, i64 k, i64 l) const noexcept {
            using compute_type = BufferAccessor::value_type;
            using output_type = OutputAccessor::value_type;

            if constexpr (MODE == Border::REFLECT) {
                for (i64 wj{}; wj < m_window_1d; ++wj) {
                    const i64 ij = median_filter_get_mirror_index(j - m_window_1d_half + wj, m_shape[0]);
                    for (i64 wk{}; wk < m_window_1d; ++wk) {
                        const i64 ik = median_filter_get_mirror_index(k - m_window_1d_half + wk, m_shape[1]);
                        for (i64 wl{}; wl < m_window_1d; ++wl) {
                            const i64 il = median_filter_get_mirror_index(l - m_window_1d_half + wl, m_shape[2]);
                            m_buffer[(wj * m_window_1d + wk) * m_window_1d + wl] =
                                    static_cast<compute_type>(m_input(i, ij, ik, il));
                        }
                    }
                }
            } else { // Border::ZERO
                for (i64 wj{}; wj < m_window_1d; ++wj) {
                    const i64 ij = j - m_window_1d_half + wj;
                    for (i64 wk{}; wk < m_window_1d; ++wk) {
                        const i64 ik = k - m_window_1d_half + wk;
                        for (i64 wl{}; wl < m_window_1d; ++wl) {
                            const i64 il = l - m_window_1d_half + wl;
                            const i64 idx = (wj * m_window_1d + wk) * m_window_1d + wl;
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
        Shape3<i64> m_shape;
        i64 m_window_1d;
        i64 m_window_1d_half;
        i64 m_window_size;
        i64 m_window_half;
    };
}

namespace noa::cpu::signal {
    template<typename T, typename U>
    void median_filter_1d(
        const T* input, const Strides4<i64>& input_strides,
        U* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, Border border_mode,
        i64 window_size, i64 n_threads
    ) {
        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        const auto buffer = AllocatorHeap::allocate<compute_t>(window_size * n_threads);

        using input_accessor_t = AccessorRestrictI64<const T, 4>;
        using output_accessor_t = AccessorRestrictI64<U, 4>;
        using buffer_accessor_t = AccessorRestrictContiguousI64<compute_t, 1>;

        switch (border_mode) {
            case Border::REFLECT: {
                NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                auto op = guts::MedianFilter1d<Border::REFLECT, input_accessor_t, output_accessor_t, buffer_accessor_t>(
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    buffer_accessor_t(buffer.get()),
                    shape, window_size);
                return iwise(shape, op, n_threads);
            }
            case Border::ZERO: {
                auto op = guts::MedianFilter1d<Border::ZERO, input_accessor_t, output_accessor_t, buffer_accessor_t>(
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    buffer_accessor_t(buffer.get()),
                    shape, window_size);
                return iwise(shape, op, n_threads);
            }
            default:
                panic("Border not supported. Should be {} or {}, got {}",
                      Border::ZERO, Border::REFLECT, border_mode);
        }
    }

    template<typename T, typename U>
    void median_filter_2d(
        const T* input, Strides4<i64> input_strides,
        U* output, Strides4<i64> output_strides,
        Shape4<i64> shape, Border border_mode, i64 window_size, i64 n_threads
    ) {
        const auto order_2d = ni::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (any(order_2d != Vec2<i64>{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
        }

        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        const auto buffer = AllocatorHeap::allocate<compute_t>(window_size * window_size * n_threads);

        using input_accessor_t = AccessorRestrictI64<const T, 4>;
        using output_accessor_t = AccessorRestrictI64<U, 4>;
        using buffer_accessor_t = AccessorRestrictContiguousI64<compute_t, 1>;

        switch (border_mode) {
            case Border::REFLECT: {
                auto op = guts::MedianFilter2d<Border::REFLECT, input_accessor_t, output_accessor_t, buffer_accessor_t>(
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    buffer_accessor_t(buffer.get()),
                    shape, window_size);
                return iwise(shape, op, n_threads);
            }
            case Border::ZERO: {
                NOA_ASSERT(window_size / 2 + 1 <= shape[2]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                auto op = guts::MedianFilter2d<Border::ZERO, input_accessor_t, output_accessor_t, buffer_accessor_t>(
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    buffer_accessor_t(buffer.get()),
                    shape, window_size);
                return iwise(shape, op, n_threads);
            }
            default:
                panic("Border not supported. Should be {} or {}, got {}",
                      Border::ZERO, Border::REFLECT, border_mode);
        }
    }

    template<typename T, typename U>
    void median_filter_3d(
        const T* input, Strides4<i64> input_strides,
        U* output, Strides4<i64> output_strides,
        Shape4<i64> shape, Border border_mode, i64 window_size, i64 n_threads
    ) {
        const auto order_3d = ni::order(output_strides.pop_front(), shape.pop_front());
        if (any(order_3d != Vec3<i64>{0, 1, 2})) {
            const auto order = (order_3d + 1).push_front(0);
            input_strides = ni::reorder(input_strides, order);
            output_strides = ni::reorder(output_strides, order);
            shape = ni::reorder(shape, order);
        }

        using compute_t = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        const auto buffer = AllocatorHeap::allocate<compute_t>(window_size * window_size * window_size * n_threads);

        using input_accessor_t = AccessorRestrictI64<const T, 4>;
        using output_accessor_t = AccessorRestrictI64<U, 4>;
        using buffer_accessor_t = AccessorRestrictContiguousI64<compute_t, 1>;

        switch (border_mode) {
            case Border::REFLECT: {
                NOA_ASSERT(window_size / 2 + 1 <= shape[1]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[2]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                auto op = guts::MedianFilter3d<Border::REFLECT, input_accessor_t, output_accessor_t, buffer_accessor_t>(
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    buffer_accessor_t(buffer.get()),
                    shape, window_size);
                return iwise(shape, op, n_threads);
            }
            case Border::ZERO: {
                auto op = guts::MedianFilter3d<Border::ZERO, input_accessor_t, output_accessor_t, buffer_accessor_t>(
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    buffer_accessor_t(buffer.get()),
                    shape, window_size);
                return iwise(shape, op, n_threads);
            }
            default:
                panic("Border not supported. Should be {} or {}, got {}",
                      Border::ZERO, Border::REFLECT, border_mode);
        }
    }
}
