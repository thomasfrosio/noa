#pragma once

#include <numeric>

#include "noa/core/Types.hpp"
#include "noa/core/Math.hpp"

namespace noa::signal::details {
    template<typename Derived> // CRTP
    class Window {
    public:
        constexpr explicit Window(i64 elements, bool half_window = false) :
                m_elements(elements),
                m_offset(half_window ? elements / 2 : 0),
                m_center(window_center_coordinate_(elements)) {}

        [[nodiscard]] constexpr auto elements() const noexcept -> i64 { return m_elements; }
        [[nodiscard]] constexpr auto offset() const noexcept -> i64 { return m_offset; }
        [[nodiscard]] constexpr auto center() const noexcept -> f64 { return m_center; }

        template<typename Real, typename = std::enable_if_t<nt::is_real_v<Real>>>
        constexpr void generate(Real* output, bool normalize = true) const noexcept {
            f64 sum{0};
            for (i64 i = m_offset; i < m_elements; ++i) {
                const f64 n = static_cast<f64>(i);
                const f64 window = static_cast<const Derived*>(this)->window(n, m_center);
                sum += window;
                output[i - m_offset] = static_cast<Real>(window);
            }
            if (normalize)
                window_normalize_(output, m_elements - m_offset, sum);
        }

        [[nodiscard]] constexpr f64 sample(i64 index) const noexcept {
            const f64 n = static_cast<f64>(index + m_offset);
            return static_cast<const Derived*>(this)->window(n, m_center);
        }

    private:
        [[nodiscard]] static constexpr f64 window_center_coordinate_(i64 elements) noexcept {
            auto half = static_cast<f64>(elements / 2);
            if (!(elements % 2)) // even
                half -= 0.5;
            return half;
        }

        template<typename Real>
        static constexpr void window_normalize_(Real* output, i64 elements, f64 sum) noexcept {
            const auto sum_real = static_cast<Real>(sum);
            for (i64 i = 0; i < elements; ++i)
                output[i] /= sum_real;
        }

    private:
        i64 m_elements;
        i64 m_offset;
        f64 m_center;
    };

    struct WindowGaussian : public Window<WindowGaussian> {
        f64 sig2;
        constexpr explicit WindowGaussian(i64 elements, bool half_window, f64 stddev)
                : Window(elements, half_window), sig2(2 * stddev * stddev) {}

        [[nodiscard]] f64 window(f64 i, f64 center) const noexcept {
            i -= center;
            return noa::math::exp(-(i * i) / sig2);
        }
    };

    struct WindowBlackman : public Window<WindowBlackman> {
        constexpr explicit WindowBlackman(i64 elements, bool half_window)
                : Window(elements, half_window) {}

        [[nodiscard]] f64 window(f64 i, f64) const noexcept {
            constexpr auto PI = noa::math::Constant<f64>::PI;
            const auto norm = static_cast<f64>(elements() - 1);
            return 0.42 -
                   0.5 * noa::math::cos(2 * PI * i / norm) +
                   0.08 * noa::math::cos(4 * PI * i / norm);
        }
    };

    struct WindowSinc : public Window<WindowSinc> {
        f64 constant;
        constexpr explicit WindowSinc(i64 elements, bool half_window, f64 constant_)
                : Window(elements, half_window), constant(constant_) {}

        [[nodiscard]] f64 window(f64 i, f64 center) const noexcept {
            constexpr auto PI = noa::math::Constant<f64>::PI;
            i -= center;
            return i == 0 ? constant : noa::math::sin(PI * i * constant) / (PI * i);
        }
    };
}

namespace noa::signal {
    /// Computes the gaussian window.
    /// \param[out] output  Output array where to save the (half-)window.
    /// \param elements     Number of elements in the full-window.
    /// \param stddev       Standard deviation of the gaussian.
    /// \param normalize    Whether to normalize the sum of the full-window to 1.
    /// \param half_window  Whether to only compute the second half of the window.
    ///                     If true, only (elements-1)//2+1 are written in \p output.
    template<typename Real, typename = std::enable_if_t<nt::is_real_v<Real>>>
    constexpr void window_gaussian(
            Real* output, i64 elements, f64 stddev,
            bool normalize = false, bool half_window = false
    ) {
        if (elements <= 0)
            return;
        details::WindowGaussian(elements, half_window, stddev).generate(output, normalize);
    }

    /// Samples the gaussian window at a particular index.
    /// \param index        Index where to sample.
    /// \param elements     Number of elements in the window.
    /// \param stddev       Standard deviation of the gaussian.
    /// \param half_window  Whether to compute the second half of the window.
    [[nodiscard]] constexpr f64 window_gaussian(i64 index, i64 elements, f64 stddev, bool half_window = false) {
        if (elements <= 0)
            return 0.;
        return details::WindowGaussian(elements, half_window, stddev).sample(index);
    }

    /// Computes the blackman window.
    /// \param[out] output  Output array where to save the (half-)window.
    /// \param elements     Number of elements in the full-window.
    /// \param normalize    Whether to normalize the sum of the full-window to 1.
    /// \param half_window  Whether to only compute the second half of the window.
    ///                     If true, only (elements-1)//2+1 are written in \p output.
    template<typename Real>
    constexpr void window_blackman(Real* output, i64 elements, bool normalize = false, bool half_window = false) {
        if (elements <= 0)
            return;
        details::WindowBlackman(elements, half_window).generate(output, normalize);
    }

    /// Samples the blackman window at a particular index.
    /// \param index        Index where to sample.
    /// \param elements     Number of elements in the window.
    /// \param half_window  Whether to compute the second half of the window.
    [[nodiscard]] constexpr f64 window_blackman(i64 index, i64 elements, bool half_window = false) {
        if (elements <= 0)
            return 0;
        return details::WindowBlackman(elements, half_window).sample(index);
    }

    /// Computes the sinc window.
    /// \param[out] output  Output array where to save the (half-)window.
    /// \param elements     Number of elements in the full-window.
    /// \param constant     Additional constant factor. \c sin(constant*pi*x)/(pi*x) is computed.
    /// \param normalize    Whether to normalize the sum of the full-window to 1.
    /// \param half_window  Whether to only compute the second half of the window.
    ///                     If true, only (elements-1)//2+1 are written in \p output.
    template<typename Real>
    constexpr void window_sinc(
            Real* output, i64 elements, f64 constant,
            bool normalize = false, bool half_window = false
    ) {
        if (elements <= 0)
            return;
        details::WindowSinc(elements, half_window, constant).generate(output, normalize);
    }

    /// Samples the sinc window at a particular index.
    /// \param index        Index where to sample.
    /// \param elements     Number of elements in the window. Should be >= 1.
    /// \param constant     Additional constant factor. \c sin(constant*pi*x)/(pi*x) is computed.
    /// \param half_window  Whether to compute the second half of the window.
    [[nodiscard]] constexpr f64 window_sinc(i64 index, i64 elements, f64 constant, bool half_window = false) {
        if (elements <= 0)
            return 0;
        return details::WindowSinc(elements, half_window, constant).sample(index);
    }
}
