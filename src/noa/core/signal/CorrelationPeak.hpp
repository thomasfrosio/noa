#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/math/LeastSquare.hpp"

namespace noa::signal {
    template<std::floating_point Real, std::signed_integral SInt, size_t N>
    constexpr NOA_HD auto peak_parabola_1d(const Real* window_values, const Vec<SInt, N>& window_radius) {
        // Add sub-pixel position by fitting a 1D parabola to the peak and its two adjacent points.
        f64 peak_value{0};
        Vec<f64, N> peak_coordinate{};
        const Real* current_window = window_values;
        for (size_t i = 0; i < N; ++i) {
            const auto window_size = window_radius[i] * 2 + 1;

            if (window_radius[i] == 1) {
                const auto [x, y] = lstsq_fit_quadratic_vertex_3points(
                        current_window[0], current_window[1], current_window[2]);
                peak_coordinate[i] = static_cast<f64>(x);
                peak_value += static_cast<f64>(y);
            } else {
                f64 a{}, b{}, c{};
                lstsq_fit_quadratic(current_window, window_size, &a, &b, &c);
                if (a != 0) { // This can fail if all values in output are equal.
                    const auto x = clamp(-b / (2 * a), 0.5, static_cast<f64>(window_size) - 1.5);
                    const auto y = a * x * x + b * x + c;
                    peak_coordinate[i] = x - static_cast<f64>(window_radius[i]);
                    peak_value += y;
                }
            }
            current_window += window_size;
        }
        return Pair{peak_value / N, peak_coordinate};
    }
}
