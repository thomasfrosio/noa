#include <algorithm> // std::nth_element

#include "noa/cpu/memory/Copy.hpp"
#include "noa/cpu/memory/AllocatorHeap.hpp"
#include "noa/cpu/signal/Median.hpp"
#include "noa/cpu/utils/Iwise.hpp"

namespace {
    using namespace noa;

    // If index is out of bound, apply mirror (d c b | a b c d | c b a).
    // This requires shape >= window/2+1. noa::indexing::at() would work, but this
    // is slightly more efficient because we know the size of the halo is less than the size.
    i64 get_mirror_index_(i64 index, i64 size) {
        if (index < 0)
            index *= -1;
        else if (index >= size)
            index = 2 * (size - 1) - index;
        return index;
    }

    template<typename T, BorderMode MODE>
    class MedianFilter1D {
    public:
        using value_type = T;
        using compute_type = std::conditional_t<std::is_same_v<f16, value_type>, f32, value_type>;
        using input_accessor_type = AccessorRestrict<const value_type, 4, i64>;
        using output_accessor_type = AccessorRestrict<value_type, 4, i64>;
        using buffer_accessor_type = AccessorRestrictContiguous<compute_type, 1, i64>;

    public:
        MedianFilter1D(const input_accessor_type& input,
                       const output_accessor_type& output,
                       const Shape4<i64>& shape, i64 window,
                       buffer_accessor_type buffer)
                : m_input(input), m_output(output), m_buffer(buffer),
                  m_width(shape.width()), m_window(window),
                  m_window_half(window / 2) {}

        void initialize(i64 thread) noexcept {
            // Before starting the loop, offset to the thread buffer.
            m_buffer = buffer_accessor_type(m_buffer.get() + thread * m_window);
        }

        void operator()(i64 i, i64 j, i64 k, i64 l) const noexcept {
            // Gather the window.
            if constexpr (MODE == BorderMode::REFLECT) {
                for (i64 wl = 0; wl < m_window; ++wl) {
                    const i64 il = get_mirror_index_(l - m_window_half + wl, m_width);
                    m_buffer[wl] = static_cast<compute_type>(m_input(i, j, k, il));
                }
            } else { // BorderMode::ZERO
                for (i64 wl = 0; wl < m_window; ++wl) {
                    const i64 il = l - m_window_half + wl;
                    if (il < 0 || il >= m_width)
                        m_buffer[wl] = compute_type{0};
                    else
                        m_buffer[wl] = static_cast<compute_type>(m_input(i, j, k, il));
                }
            }

            // Sort the elements in the m_window to get the median.
            std::nth_element(m_buffer.get(), m_buffer.get() + m_window_half, m_buffer.get() + m_window);
            m_output(i, j, k, l) = static_cast<value_type>(m_buffer[m_window_half]);
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        buffer_accessor_type m_buffer{};
        i64 m_width;
        i64 m_window;
        i64 m_window_half;
    };

    template<typename T, BorderMode MODE>
    class MedianFilter2D {
    public:
        using value_type = T;
        using compute_type = std::conditional_t<std::is_same_v<f16, value_type>, f32, value_type>;
        using input_accessor_type = AccessorRestrict<const value_type, 4, i64>;
        using output_accessor_type = AccessorRestrict<value_type, 4, i64>;
        using buffer_accessor_type = AccessorRestrictContiguous<compute_type, 1, i64>;

    public:
        MedianFilter2D(const input_accessor_type& input,
                       const output_accessor_type& output,
                       const Shape4<i64>& shape, i64 window,
                       buffer_accessor_type buffer)
                : m_input(input), m_output(output), m_buffer(buffer),
                  m_shape(shape.filter(2, 3)),
                  m_window_1d(window),
                  m_window_1d_half(window / 2),
                  m_window_size(window * window),
                  m_window_half((window * window) / 2) {}

        void initialize(i64 thread) noexcept {
            // Before starting the loop, offset to the thread buffer.
            m_buffer = buffer_accessor_type(m_buffer.get() + thread * m_window_size);
        }

        void operator()(i64 i, i64 j, i64 k, i64 l) const noexcept {
            if constexpr (MODE == BorderMode::REFLECT) {
                for (i64 wk = 0; wk < m_window_1d; ++wk) {
                    const i64 ik = get_mirror_index_(k - m_window_1d_half + wk, m_shape[0]);
                    for (i64 wl = 0; wl < m_window_1d; ++wl) {
                        const i64 il = get_mirror_index_(l - m_window_1d_half + wl, m_shape[1]);
                        m_buffer[wk * m_window_1d + wl] = static_cast<compute_type>(m_input(i, j, ik, il));
                    }
                }
            } else { // BorderMode::ZERO
                for (i64 wk = 0; wk < m_window_1d; ++wk) {
                    const i64 ik = k - m_window_1d_half + wk;
                    for (i64 wl = 0; wl < m_window_1d; ++wl) {
                        const i64 il = l - m_window_1d_half + wl;
                        if (ik < 0 || ik >= m_shape[0] || il < 0 || il >= m_shape[1]) {
                            m_buffer[wk * m_window_1d + wl] = compute_type{0};
                        } else {
                            m_buffer[wk * m_window_1d + wl] = static_cast<compute_type>(m_input(i, j, ik, il));
                        }
                    }
                }
            }

            // Sort the elements in the m_window to get the median.
            std::nth_element(m_buffer.get(), m_buffer.get() + m_window_half, m_buffer.get() + m_window_size);
            m_output(i, j, k, l) = static_cast<T>(m_buffer[m_window_half]);
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        buffer_accessor_type m_buffer{};
        Shape2<i64> m_shape;
        i64 m_window_1d;
        i64 m_window_1d_half;
        i64 m_window_size;
        i64 m_window_half;
    };

    template<typename T, BorderMode MODE>
    class MedianFilter3D {
    public:
        using value_type = T;
        using compute_type = std::conditional_t<std::is_same_v<f16, value_type>, f32, value_type>;
        using input_accessor_type = AccessorRestrict<const value_type, 4, i64>;
        using output_accessor_type = AccessorRestrict<value_type, 4, i64>;
        using buffer_accessor_type = AccessorRestrictContiguous<compute_type, 1, i64>;

    public:
        MedianFilter3D(const input_accessor_type& input,
                       const output_accessor_type& output,
                       const Shape4<i64>& shape, i64 window,
                       buffer_accessor_type buffer)
                : m_input(input), m_output(output), m_buffer(buffer),
                  m_shape(shape.pop_front()),
                  m_window_1d(window),
                  m_window_1d_half(window / 2),
                  m_window_size(window * window * window),
                  m_window_half((window * window * window) / 2) {}

        void initialize(i64 thread) noexcept {
            // Before starting the loop, offset to the thread buffer.
            m_buffer = buffer_accessor_type(m_buffer.get() + thread * m_window_size);
        }

        void operator()(i64 i, i64 j, i64 k, i64 l) const noexcept {
            if constexpr (MODE == BorderMode::REFLECT) {
                for (i64 wj = 0; wj < m_window_1d; ++wj) {
                    const i64 ij = get_mirror_index_(j - m_window_1d_half + wj, m_shape[0]);
                    for (i64 wk = 0; wk < m_window_1d; ++wk) {
                        const i64 ik = get_mirror_index_(k - m_window_1d_half + wk, m_shape[1]);
                        for (i64 wl = 0; wl < m_window_1d; ++wl) {
                            const i64 il = get_mirror_index_(l - m_window_1d_half + wl, m_shape[2]);
                            m_buffer[(wj * m_window_1d + wk) * m_window_1d + wl] =
                                    static_cast<compute_type>(m_input(i, ij, ik, il));
                        }
                    }
                }
            } else { // BorderMode::ZERO
                for (i64 wj = 0; wj < m_window_1d; ++wj) {
                    const i64 ij = j - m_window_1d_half + wj;
                    for (i64 wk = 0; wk < m_window_1d; ++wk) {
                        const i64 ik = k - m_window_1d_half + wk;
                        for (i64 wl = 0; wl < m_window_1d; ++wl) {
                            const i64 il = l - m_window_1d_half + wl;
                            const i64 idx = (wj * m_window_1d + wk) * m_window_1d + wl;
                            if (ij < 0 || ij >= m_shape[0] ||
                                ik < 0 || ik >= m_shape[1] ||
                                il < 0 || il >= m_shape[2]) {
                                m_buffer[idx] = compute_type{0};
                            } else {
                                m_buffer[idx] = static_cast<compute_type>(m_input(i, ij, ik, il));
                            }
                        }
                    }
                }
            }

            // Sort the elements in the window to get the median.
            std::nth_element(m_buffer.get(), m_buffer.get() + m_window_half, m_buffer.get() + m_window_size);
            m_output(i, j, k, l) = static_cast<value_type>(m_buffer[m_window_half]);
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        buffer_accessor_type m_buffer{};
        Shape3<i64> m_shape;
        i64 m_window_1d;
        i64 m_window_1d_half;
        i64 m_window_size;
        i64 m_window_half;
    };

    static_assert(nt::is_detected_v<nt::has_initialize, MedianFilter1D<f32, BorderMode::REFLECT>>);
    static_assert(nt::is_detected_v<nt::has_initialize, MedianFilter2D<f32, BorderMode::REFLECT>>);
    static_assert(nt::is_detected_v<nt::has_initialize, MedianFilter3D<f32, BorderMode::REFLECT>>);
}

namespace noa::cpu::signal {
    template<typename T, typename>
    void median_filter_1d(const T* input, const Strides4<i64>& input_strides,
                          T* output, const Strides4<i64>& output_strides,
                          const Shape4<i64>& shape, BorderMode border_mode, i64 window_size, i64 threads) {
        NOA_ASSERT(input != output && noa::all(shape > 0));
        if (window_size == 1)
            return noa::cpu::memory::copy(input, input_strides, output, output_strides, shape, threads);

        NOA_ASSERT(window_size % 2);

        // If half precision, do the sort in single-precision.
        using Value = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        const auto buffer = cpu::memory::AllocatorHeap<Value>::allocate(window_size * threads);

        const auto input_accessor = AccessorRestrict<const T, 4, i64>(input, input_strides);
        const auto output_accessor = AccessorRestrict<T, 4, i64>(output, output_strides);
        const auto buffer_accessor = AccessorRestrictContiguous<Value, 1, i64>(buffer.get());

        switch (border_mode) {
            case BorderMode::REFLECT: {
                NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                auto kernel = MedianFilter1D<T, BorderMode::REFLECT>(
                        input_accessor, output_accessor, shape, window_size, buffer_accessor);
                return noa::cpu::utils::iwise_4d(shape, kernel, threads);
            }
            case BorderMode::ZERO: {
                auto kernel = MedianFilter1D<T, BorderMode::ZERO>(
                        input_accessor, output_accessor, shape, window_size, buffer_accessor);
                return noa::cpu::utils::iwise_4d(shape, kernel, threads);
            }
            default:
                NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                          BorderMode::ZERO, BorderMode::REFLECT, border_mode);
        }
    }

    template<typename T, typename>
    void median_filter_2d(const T* input, Strides4<i64> input_strides,
                          T* output, Strides4<i64> output_strides,
                          Shape4<i64> shape, BorderMode border_mode, i64 window_size, i64 threads) {
        NOA_ASSERT(input != output && noa::all(shape > 0));
        if (window_size == 1)
            return memory::copy(input, input_strides, output, output_strides, shape, threads);

        NOA_ASSERT(window_size % 2);

        const auto order_2d = noa::indexing::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (noa::any(order_2d != Vec2<i64>{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
        }

        // If half precision, do the sort in single-precision.
        using Value = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        const auto buffer = cpu::memory::AllocatorHeap<Value>::allocate(window_size * window_size * threads);

        const auto input_accessor = AccessorRestrict<const T, 4, i64>(input, input_strides);
        const auto output_accessor = AccessorRestrict<T, 4, i64>(output, output_strides);
        const auto buffer_accessor = AccessorRestrictContiguous<Value, 1, i64>(buffer.get());

        switch (border_mode) {
            case BorderMode::REFLECT: {
                NOA_ASSERT(window_size / 2 + 1 <= shape[2]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                auto kernel = MedianFilter2D<T, BorderMode::REFLECT>(
                        input_accessor, output_accessor, shape, window_size, buffer_accessor);
                return noa::cpu::utils::iwise_4d(shape, kernel, threads);
            }
            case BorderMode::ZERO: {
                auto kernel = MedianFilter2D<T, BorderMode::ZERO>(
                        input_accessor, output_accessor, shape, window_size, buffer_accessor);
                return noa::cpu::utils::iwise_4d(shape, kernel, threads);
            }
            default:
                NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                          BorderMode::ZERO, BorderMode::REFLECT, border_mode);
        }
    }

    template<typename T, typename>
    void median_filter_3d(const T* input, Strides4<i64> input_strides,
                          T* output, Strides4<i64> output_strides,
                          Shape4<i64> shape, BorderMode border_mode, i64 window_size, i64 threads) {
        NOA_ASSERT(input != output && noa::all(shape > 0));
        if (window_size == 1)
            return noa::cpu::memory::copy(input, input_strides, output, output_strides, shape, threads);

        NOA_ASSERT(window_size % 2);

        const auto order_3d = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
        if (noa::any(order_3d != Vec3<i64>{0, 1, 2})) {
            const auto order = (order_3d + 1).push_front(0);
            input_strides = noa::indexing::reorder(input_strides, order);
            output_strides = noa::indexing::reorder(output_strides, order);
            shape = noa::indexing::reorder(shape, order);
        }

        // If half precision, do the sort in single-precision.
        using Value = std::conditional_t<std::is_same_v<f16, T>, f32, T>;
        const auto buffer = cpu::memory::AllocatorHeap<Value>::allocate(window_size * window_size * window_size * threads);

        const auto input_accessor = AccessorRestrict<const T, 4, i64>(input, input_strides);
        const auto output_accessor = AccessorRestrict<T, 4, i64>(output, output_strides);
        const auto buffer_accessor = AccessorRestrictContiguous<Value, 1, i64>(buffer.get());

        switch (border_mode) {
            case BorderMode::REFLECT: {
                NOA_ASSERT(window_size / 2 + 1 <= shape[1]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[2]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                auto kernel = MedianFilter3D<T, BorderMode::REFLECT>(
                        input_accessor, output_accessor, shape, window_size, buffer_accessor);
                return noa::cpu::utils::iwise_4d(shape, kernel, threads);
            }
            case BorderMode::ZERO: {
                auto kernel = MedianFilter3D<T, BorderMode::ZERO>(
                        input_accessor, output_accessor, shape, window_size, buffer_accessor);
                return noa::cpu::utils::iwise_4d(shape, kernel, threads);
            }
            default:
                NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                          BorderMode::ZERO, BorderMode::REFLECT, border_mode);
        }
    }

    #define NOA_INSTANTIATE_MEDFILT_(T)                                                                                                             \
    template void median_filter_1d<T, void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, BorderMode, i64, i64);    \
    template void median_filter_2d<T, void>(const T*, Strides4<i64>, T*, Strides4<i64>, Shape4<i64>, BorderMode, i64, i64);                         \
    template void median_filter_3d<T, void>(const T*, Strides4<i64>, T*, Strides4<i64>, Shape4<i64>, BorderMode, i64, i64)

    NOA_INSTANTIATE_MEDFILT_(f16);
    NOA_INSTANTIATE_MEDFILT_(f32);
    NOA_INSTANTIATE_MEDFILT_(f64);
    NOA_INSTANTIATE_MEDFILT_(i32);
    NOA_INSTANTIATE_MEDFILT_(i64);
    NOA_INSTANTIATE_MEDFILT_(u32);
    NOA_INSTANTIATE_MEDFILT_(u64);
}
