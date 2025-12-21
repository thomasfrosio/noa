#pragma once

#include "noa/runtime/core/Math.hpp"
#include "noa/runtime/core/Pair.hpp"
#include "noa/runtime/core/Span.hpp"

// More details at: https://www.codeproject.com/Articles/63170/Least-Squares-Regression-for-Quadratic-Curve-Fitti

namespace noa::signal::details {
    template<nt::any_of<f32, f64> T>
    struct QuadraticCurve {
        using value_type = T;
        value_type a, b, c;
    };

    /// Least squares regression for quadratic curve fitting.
    /// Returns {a, b, c}, as in ``y(x) = ax^2 + bx + c``, where x is an integral number from 0 to y.size()-1.
    template<nt::real T, nt::integer I>
    NOA_HD constexpr auto lstsq_fit_quadratic(SpanContiguous<const T, 1, I> y) -> QuadraticCurve<f64> {
        if (not y or y.size() < 3)
            return {};

        f64 sy{};
        f64 syx{};
        f64 syx2{};

        for (isize i{}; auto& e: y) {
            const auto iy = static_cast<f64>(e);
            const auto ix = static_cast<f64>(i++);
            sy += iy;
            syx += iy * ix;
            syx2 += iy * ix * ix;
        }

        const auto n = static_cast<f64>(y.ssize() - 1);
        const auto sx0 = static_cast<f64>(y.ssize());
        const f64 sx1 = n * (n + 1) / 2;
        const f64 sx2 = n * (n + 1) * (2 * n + 1) / 6;
        const f64 sx3 = n * n * (n + 1) * (n + 1) / 4;
        const f64 sx4 = ((6 * n * n * n * n * n) +
                         (15 * n * n * n * n) +
                         (10 * n * n * n) - n) / 30;

        const f64 tmp0 = sx2 * sx0 - sx1 * sx1;
        const f64 tmp1 = sx3 * sx0 - sx1 * sx2;
        const f64 tmp2 = sx3 * sx1 - sx2 * sx2;

        return {
            .a = (syx2 * (tmp0) - syx * (tmp1) + sy * (tmp2)) /
                 (sx4 * (tmp0) - sx3 * (tmp1) + sx2 * (tmp2)),
            .b = (sx4 * (syx * sx0 - sy * sx1) -
                  sx3 * (syx2 * sx0 - sy * sx2) +
                  sx2 * (syx2 * sx1 - syx * sx2)) /
                 (sx4 * (tmp0) - sx3 * (tmp1) + sx2 * (tmp2)),
            .c = (sx4 * (sx2 * sy - sx1 * syx) -
                  sx3 * (sx3 * sy - sx1 * syx2) +
                  sx2 * (sx3 * syx - sx2 * syx2)) /
                 (sx4 * (tmp0) - sx3 * (tmp1) + sx2 * (tmp2)),
        };
    }

    /// Given values at three successive positions, y0, y1, and y2, where y1 is the peak value,
    /// this fits a parabola to the values and returns the offset of the peak from the center position,
    /// a number between -0.5 and 0.5.
    template<nt::real T>
    NOA_IHD constexpr auto lstsq_fit_quadratic_vertex_3points(T y0, T y1, T y2) noexcept {
        // https://stackoverflow.com/a/717791
        const T a = 2 * y1 - y0 - y2;
        const T b = y0 - y2;
        const T c = y1;

        if (abs(a) < static_cast<T>(1e-6))
            return Pair{T{}, y1};

        const T x = -b / (2 * a);
        const T y = c + b * b / (8 * a);
        return Pair{x, y};
    }
}
