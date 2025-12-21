#pragma once

#include "noa/runtime/core/Math.hpp"

namespace noa::signal::details {
    template<typename Derived>
    class Window {
    public:
        constexpr explicit Window(isize n_elements, bool half_window = false) :
            m_n_elements(n_elements),
            m_offset(half_window ? n_elements / 2 : 0),
            m_center(window_center_coordinate_(n_elements)) {}

        [[nodiscard]] constexpr auto n_elements() const noexcept -> isize { return m_n_elements; }
        [[nodiscard]] constexpr auto offset() const noexcept -> isize { return m_offset; }
        [[nodiscard]] constexpr auto center() const noexcept -> f64 { return m_center; }

        template<nt::real Real>
        constexpr void generate(Real* output, bool normalize = true) const noexcept {
            f64 sum{};
            for (isize i = m_offset; i < m_n_elements; ++i) {
                const f64 n = static_cast<f64>(i);
                const f64 window = static_cast<const Derived*>(this)->window(n, m_center);
                sum += window;
                output[i - m_offset] = static_cast<Real>(window);
            }
            if (normalize)
                window_normalize_(output, m_n_elements - m_offset, sum);
        }

        [[nodiscard]] constexpr auto sample(isize index) const noexcept -> f64 {
            const f64 n = static_cast<f64>(index + m_offset);
            return static_cast<const Derived*>(this)->window(n, m_center);
        }

    private:
        [[nodiscard]] static constexpr auto window_center_coordinate_(isize n_elements) noexcept -> f64 {
            auto half = static_cast<f64>(n_elements / 2);
            if (is_even(n_elements))
                half -= 0.5;
            return half;
        }

        template<typename Real>
        static constexpr void window_normalize_(Real* output, isize n_elements, f64 sum) noexcept {
            const auto sum_real = static_cast<Real>(sum);
            for (isize i = 0; i < n_elements; ++i)
                output[i] /= sum_real;
        }

    private:
        isize m_n_elements;
        isize m_offset;
        f64 m_center;
    };

    struct WindowGaussian : Window<WindowGaussian> {
        f64 sig2;
        constexpr explicit WindowGaussian(isize elements, bool half_window, f64 stddev) :
            Window(elements, half_window), sig2(2 * stddev * stddev) {}

        [[nodiscard]] auto window(f64 i, f64 center) const noexcept -> f64 {
            i -= center;
            return exp(-(i * i) / sig2);
        }
    };

    struct WindowBlackman : Window<WindowBlackman> {
        constexpr explicit WindowBlackman(isize elements, bool half_window) :
            Window(elements, half_window) {}

        [[nodiscard]] auto window(f64 i, f64) const noexcept -> f64 {
            constexpr auto PI = Constant<f64>::PI;
            const auto norm = static_cast<f64>(n_elements() - 1);
            return 0.42 -
                   0.5 * cos(2 * PI * i / norm) +
                   0.08 * cos(4 * PI * i / norm);
        }
    };

    struct WindowSinc : Window<WindowSinc> {
        f64 constant;
        constexpr explicit WindowSinc(isize elements, bool half_window, f64 constant_) :
            Window(elements, half_window), constant(constant_) {}

        [[nodiscard]] auto window(f64 i, f64 center) const noexcept -> f64 {
            constexpr auto PI = Constant<f64>::PI;
            i -= center;
            return i == 0 ? constant : sin(PI * i * constant) / (PI * i);
        }
    };
}

namespace noa::signal {
    struct WindowOptions {
        /// Whether to normalize the sum of the full-window to 1.
        bool normalize{};

        /// Whether to only compute the second half of the window.
        /// If true, the output array only has \c (n_elements-1)//2+1 n_elements.
        bool half_window{};
    };

    /// Computes the gaussian window.
    /// \param[out] output  Output array where to save the (half-)window.
    /// \param elements     Number of elements in the full-window.
    /// \param stddev       Standard deviation of the gaussian.
    /// \param options      Window options.
    template<nt::real T>
    constexpr void window_gaussian(T* output, isize elements, f64 stddev, WindowOptions options = {}) {
        if (elements <= 0)
            return;
        details::WindowGaussian(elements, options.half_window, stddev).generate(output, options.normalize);
    }

    /// Samples the gaussian window at a particular index.
    /// \param index        Index where to sample.
    /// \param elements     Number of elements in the window.
    /// \param stddev       Standard deviation of the gaussian.
    /// \param options      Window options. Note that `normalize` is ignored.
    [[nodiscard]] constexpr auto window_gaussian(isize index, isize elements, f64 stddev, WindowOptions options = {}) -> f64 {
        if (elements <= 0)
            return 0.;
        return details::WindowGaussian(elements, options.half_window, stddev).sample(index);
    }

    /// Computes the blackman window.
    /// \param[out] output  Output array where to save the (half-)window.
    /// \param elements     Number of elements in the full-window.
    /// \param options      Window options.
    template<nt::real T>
    constexpr void window_blackman(T* output, isize elements, WindowOptions options = {}) {
        if (elements <= 0)
            return;
        details::WindowBlackman(elements, options.half_window).generate(output, options.normalize);
    }

    /// Samples the blackman window at a particular index.
    /// \param index        Index where to sample.
    /// \param elements     Number of elements in the window.
    /// \param options      Window options. Note that `normalize` is ignored.
    [[nodiscard]] constexpr auto window_blackman(isize index, isize elements, WindowOptions options = {}) -> f64 {
        if (elements <= 0)
            return 0;
        return details::WindowBlackman(elements, options.half_window).sample(index);
    }

    /// Computes the sinc window.
    /// \param[out] output  Output array where to save the (half-)window.
    /// \param elements     Number of elements in the full-window.
    /// \param constant     Additional constant factor. \c sin(constant*pi*x)/(pi*x) is computed.
    /// \param options      Window options.
    template<nt::real T>
    constexpr void window_sinc(T* output, isize elements, f64 constant, WindowOptions options = {}) {
        if (elements <= 0)
            return;
        details::WindowSinc(elements, options.half_window, constant).generate(output, options.normalize);
    }

    /// Samples the sinc window at a particular index.
    /// \param index        Index where to sample.
    /// \param elements     Number of elements in the window. Should be >= 1.
    /// \param constant     Additional constant factor. \c sin(constant*pi*x)/(pi*x) is computed.
    /// \param options      Window options. Note that `normalize` is ignored.
    [[nodiscard]] constexpr auto window_sinc(isize index, isize elements, f64 constant, WindowOptions options = {}) -> f64 {
        if (elements <= 0)
            return 0;
        return details::WindowSinc(elements, options.half_window, constant).sample(index);
    }
}
