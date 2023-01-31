#pragma once

#include <tuple>
#include <cstddef>

#include "noa/core/math/Generic.hpp"
#include "noa/core/math/Comparison.hpp"
#include "noa/core/utils/Pair.hpp"

// More details at: https://www.codeproject.com/Articles/63170/Least-Squares-Regression-for-Quadratic-Curve-Fitti

namespace noa::math {
    /// Least squares regression for quadratic curve fitting.
    /// Returns {a, b, c}, as in ``y = ax^2 + bx + c``, where x is an integral number from 0 to size-1.
    /// For large sizes, prefer to use the more stable math::lstsq().
    template<bool ACCURATE_SUM = true, typename Real, typename Int>
    constexpr NOA_HD void lstsq_fit_quadratic(const Real* y, Int size, double* a, double* b, double* c) {
        static_assert(noa::traits::is_real_v<Real> && noa::traits::is_int_v<Int>);
        if (!y || size < 3)
            return;

        double sy{0};
        double syx{0};
        double syx2{0};

        [[maybe_unused]] double ey{0};
        [[maybe_unused]] double eyx{0};
        [[maybe_unused]] double eyx2{0};

        auto sum_ = [](double& s, double& e, double value) {
            if constexpr (ACCURATE_SUM) {
                auto t = s + value;
                e += noa::math::abs(s) >= noa::math::abs(value) ? (s - t) + value : (value - t) + s;
                s = t;
            } else {
                s += value;
            }
        };

        // FIXME Add CUDA warp reduction?
        for (Int i = 0; i < size; ++i) {
            const auto iy = static_cast<double>(y[i]);
            const auto ix = static_cast<double>(i);
            sum_(sy, ey, iy);
            sum_(syx, eyx, iy * ix);
            sum_(syx2, eyx2, iy * ix * ix);
        }

        const auto n = static_cast<double>(size - 1);
        const auto sx0 = static_cast<double>(size);
        const double sx1 = n * (n + 1) / 2;
        const double sx2 = n * (n + 1) * (2 * n + 1) / 6;
        const double sx3 = n * n * (n + 1) * (n + 1) / 4;
        const double sx4 = ((6 * n * n * n * n * n) +
                            (15 * n * n * n * n) +
                            (10 * n * n * n) - n) / 30;

        const double tmp0 = sx2 * sx0 - sx1 * sx1;
        const double tmp1 = sx3 * sx0 - sx1 * sx2;
        const double tmp2 = sx3 * sx1 - sx2 * sx2;

        *a = (syx2 * (tmp0) - syx * (tmp1) + sy * (tmp2)) /
            (sx4 * (tmp0) - sx3 * (tmp1) + sx2 * (tmp2));

        *b = (sx4 * (syx * sx0 - sy * sx1) -
             sx3 * (syx2 * sx0 - sy * sx2) +
             sx2 * (syx2 * sx1 - syx * sx2)) /
            (sx4 * (tmp0) - sx3 * (tmp1) + sx2 * (tmp2));

        if (c) {
            *c = (sx4 * (sx2 * sy - sx1 * syx) -
                  sx3 * (sx3 * sy - sx1 * syx2) +
                  sx2 * (sx3 * syx - sx2 * syx2)) /
                 (sx4 * (tmp0) - sx3 * (tmp1) + sx2 * (tmp2));
        }
    }

    template<bool ACCURATE_SUM = true, typename Real, typename Int>
    auto lstsq_fit_quadratic(const Real* y, Int size) {
        double a{}, b{}, c{};
        lstsq_fit_quadratic<ACCURATE_SUM>(y, size, &a, &b, &c);
        return std::tuple{a, b, c};
    }

    /// This is equivalent to math::lstsqFitQuadratic() if one wants to get the vertex for three points.
    template<typename Real>
    constexpr NOA_IHD auto lstsq_fit_quadratic_vertex_3points(Real y0, Real y1, Real y2) noexcept {
        // https://stackoverflow.com/a/717791
        const Real a = 2 * y1 - y0 - y2; // * -0.5
        const Real b = y0 - y2; // * -0.5
        const Real c = y1;

        const Real x = noa::math::clamp(-b / (2 * a), Real{-0.5}, Real{0.5});
        const Real y = c + b * b / (8 * a);

        return Pair{x, y};
    }
}
