#pragma once
#include <tuple>
#include <cstddef>
#include "noa/common/math/Generic.h"

// More details at: https://www.codeproject.com/Articles/63170/Least-Squares-Regression-for-Quadratic-Curve-Fitti

namespace noa::math {
    /// Least squares regression for quadratic curve fitting.
    /// Returns {a, b, c}, as in ``y = ax^2 + bx + c``, where x is an integral number from 0 to size-1.
    /// For large sizes, prefer to use the more stable math::lstsq().
    template<bool ACCURATE_SUM = true, typename T>
    std::tuple<double, double, double>
    lstsqFitQuadratic(const T* y, size_t size) {
        if (!y || size < 3)
            return {};

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

        for (size_t i = 0; i < size; ++i) {
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

        const double a = (syx2 * (tmp0) - syx * (tmp1) + sy * (tmp2)) /
                         (sx4 * (tmp0) - sx3 * (tmp1) + sx2 * (tmp2));

        const double b = (sx4 * (syx * sx0 - sy * sx1) -
                          sx3 * (syx2 * sx0 - sy * sx2) +
                          sx2 * (syx2 * sx1 - syx * sx2)) /
                         (sx4 * (tmp0) - sx3 * (tmp1) + sx2 * (tmp2));

        const double c = (sx4 * (sx2 * sy - sx1 * syx) -
                          sx3 * (sx3 * sy - sx1 * syx2) +
                          sx2 * (sx3 * syx - sx2 * syx2)) /
                         (sx4 * (tmp0) - sx3 * (tmp1) + sx2 * (tmp2));

        return {a, b, c};
    }
}
