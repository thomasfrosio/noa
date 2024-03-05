#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/math/LeastSquare.hpp"

namespace noa::signal::guts {
    struct CrossCorrelationL2Norm {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;

        NOA_FHD constexpr void init(const auto& lhs, const auto& rhs, f64& lhs_sum, f64& rhs_sum) const {
            lhs_sum += static_cast<f64>(abs_squared(lhs));
            rhs_sum += static_cast<f64>(abs_squared(rhs));
        }

        NOA_FHD constexpr void join(f64 lhs_isum, f64 rhs_isum, f64& lhs_sum, f64& rhs_sum) const {
            lhs_sum += lhs_isum;
            rhs_sum += rhs_isum;
        }

        template<typename T>
        NOA_FHD constexpr void final(f64 lhs_sum, f64 rhs_sum, T& lhs_norm, T& rhs_norm) const {
            lhs_norm = static_cast<T>(sqrt(lhs_sum));
            rhs_norm = static_cast<T>(sqrt(rhs_sum));
        }
    };

    struct CrossCorrelationScore {
        using allow_vectorization = bool;

        template<typename T>
        NOA_FHD constexpr void init(auto lhs, auto rhs, T& sum) const {
            if constexpr (nt::is_complex_v<decltype(rhs)>) {
                sum += static_cast<T>(lhs * conj(rhs));
            } else {
                sum += static_cast<T>(lhs * rhs);
            }
        }

        NOA_FHD constexpr void init(const auto& lhs, const auto& rhs, auto lhs_norm, auto rhs_norm, auto& sum) const {
            init(lhs / lhs_norm, rhs / rhs_norm, sum);
        }

        NOA_FHD constexpr void join(auto isum, auto& sum) const {
            sum += isum;
        }
    };

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
