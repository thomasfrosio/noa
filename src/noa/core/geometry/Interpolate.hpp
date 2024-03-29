#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/Types.hpp"

namespace noa::geometry::interpolate::details {
    template<typename Value, typename Coord>
    constexpr bool is_valid_interp_v = nt::is_real_or_complex_v<Value> && nt::is_real_v<Coord>;

    template<typename Value, typename Coord>
    using enable_if_valid_interp_t = std::enable_if_t<details::is_valid_interp_v<Value, Coord>>;

    template<typename Coord, typename Value>
    constexpr NOA_IHD void bspline_weights(Coord ratio, Value* w0, Value* w1, Value* w2, Value* w3) {
        constexpr Coord ONE_SIXTH = static_cast<Coord>(1) / static_cast<Coord>(6);
        constexpr Coord TWO_THIRD = static_cast<Coord>(2) / static_cast<Coord>(3);
        const Coord one_minus = 1 - ratio;
        const Coord one_squared = one_minus * one_minus;
        const Coord squared = ratio * ratio;

        *w0 = static_cast<Value>(ONE_SIXTH * one_squared * one_minus);
        *w1 = static_cast<Value>(TWO_THIRD - static_cast<Coord>(0.5) * squared * (2 - ratio));
        *w2 = static_cast<Value>(TWO_THIRD - static_cast<Coord>(0.5) * one_squared * (2 - one_minus));
        *w3 = static_cast<Value>(ONE_SIXTH * squared * ratio);
    }
}

// Linear interpolation:
namespace noa::geometry::interpolate {
    template<typename Value, typename Coord, typename = details::enable_if_valid_interp_t<Value, Coord>>
    constexpr NOA_FHD Value lerp_1d(Value v0, Value v1, Coord r) noexcept {
        using value_t = nt::value_type_t<Value>;
        return static_cast<value_t>(r) * (v1 - v0) + v0;
    }

    template<typename Value, typename Coord, typename = details::enable_if_valid_interp_t<Value, Coord>>
    constexpr NOA_FHD Value lerp_2d(Value v00, Value v01, Value v10, Value v11, Coord rx, Coord ry) noexcept {
        const Value tmp1 = lerp_1d(v00, v01, rx);
        const Value tmp2 = lerp_1d(v10, v11, rx);
        return lerp_1d(tmp1, tmp2, ry); // https://godbolt.org/z/eGcThbaGG
    }

    template<typename Value, typename Coord, typename = details::enable_if_valid_interp_t<Value, Coord>>
    constexpr NOA_FHD Value lerp_3d(Value v000, Value v001, Value v010, Value v011,
                                    Value v100, Value v101, Value v110, Value v111,
                                    Coord rx, Coord ry, Coord rz) noexcept {
        const Value tmp1 = lerp_2d(v000, v001, v010, v011, rx, ry);
        const Value tmp2 = lerp_2d(v100, v101, v110, v111, rx, ry);
        return lerp_1d(tmp1, tmp2, rz);
    }
}

// Linear interpolation with cosine weighting of the fraction:
namespace noa::geometry::interpolate {
    template<typename Value, typename Coord, typename = details::enable_if_valid_interp_t<Value, Coord>>
    constexpr NOA_IHD Value cosine_1d(Value v0, Value v1, Coord r) {
        constexpr Coord PI = ::noa::math::Constant<Coord>::PI;
        const Coord tmp = (static_cast<Coord>(1) - ::noa::math::cos(r * PI)) / static_cast<Coord>(2);
        return lerp_1d(v0, v1, tmp);
    }

    template<typename Value, typename Coord, typename = details::enable_if_valid_interp_t<Value, Coord>>
    constexpr NOA_IHD Value cosine_2d(Value v00, Value v01, Value v10, Value v11, Coord rx, Coord ry) {
        constexpr Coord PI = ::noa::math::Constant<Coord>::PI;
        const Coord tmp1 = (static_cast<Coord>(1) - ::noa::math::cos(rx * PI)) / static_cast<Coord>(2);
        const Coord tmp2 = (static_cast<Coord>(1) - ::noa::math::cos(ry * PI)) / static_cast<Coord>(2);
        return lerp_2d(v00, v01, v10, v11, tmp1, tmp2);
    }

    template<typename Value, typename Coord, typename = details::enable_if_valid_interp_t<Value, Coord>>
    constexpr NOA_IHD Value cosine_3d(Value v000, Value v001, Value v010, Value v011,
                                      Value v100, Value v101, Value v110, Value v111,
                                      Coord rx, Coord ry, Coord rz) {
        constexpr Coord PI = ::noa::math::Constant<Coord>::PI;
        const Coord tmp1 = (static_cast<Coord>(1) - ::noa::math::cos(rx * PI)) / static_cast<Coord>(2);
        const Coord tmp2 = (static_cast<Coord>(1) - ::noa::math::cos(ry * PI)) / static_cast<Coord>(2);
        const Coord tmp3 = (static_cast<Coord>(1) - ::noa::math::cos(rz * PI)) / static_cast<Coord>(2);
        return lerp_3d(v000, v001, v010, v011, v100, v101, v110, v111, tmp1, tmp2, tmp3);
    }
}

// Cubic interpolation:
namespace noa::geometry::interpolate {
    template<typename Value, typename Coord, typename = details::enable_if_valid_interp_t<Value, Coord>>
    constexpr NOA_HD Value cubic_1d(Value v0, Value v1, Value v2, Value v3, Coord r) {
        const Value a0 = v3 - v2 - v0 + v1;
        const Value a1 = v0 - v1 - a0;
        const Value a2 = v2 - v0;
        // a3 = v1

        using real_t = nt::value_type_t<Value>;
        const auto r1 = static_cast<real_t>(r);
        const auto r2 = r1 * r1;
        const auto r3 = r2 * r1;
        return a0 * r3 + a1 * r2 + a2 * r1 + v1;
    }

    template<typename Value, typename Coord, typename = details::enable_if_valid_interp_t<Value, Coord>>
    constexpr NOA_HD Value cubic_2d(Value v[4][4], Coord rx, Coord ry) {
        const Value a0 = cubic_1d(v[0][0], v[0][1], v[0][2], v[0][3], rx);
        const Value a1 = cubic_1d(v[1][0], v[1][1], v[1][2], v[1][3], rx);
        const Value a2 = cubic_1d(v[2][0], v[2][1], v[2][2], v[2][3], rx);
        const Value a3 = cubic_1d(v[3][0], v[3][1], v[3][2], v[3][3], rx);
        return cubic_1d(a0, a1, a2, a3, ry);
    }

    template<typename Value, typename Coord, typename = details::enable_if_valid_interp_t<Value, Coord>>
    constexpr NOA_HD Value cubic_3d(Value v[4][4][4], Coord rx, Coord ry, Coord rz) {
        const Value a0 = cubic_2d(v[0], rx, ry);
        const Value a1 = cubic_2d(v[1], rx, ry);
        const Value a2 = cubic_2d(v[2], rx, ry);
        const Value a3 = cubic_2d(v[3], rx, ry);
        return cubic_1d(a0, a1, a2, a3, rz);
    }
}

// Cubic B-spline interpolation:
// Cubic B-spline curves are not constrained to pass through the data, i.e. data points are simply referred
// to as control points. As such, these curves are not really interpolating. For instance, a ratio of 0
// puts only 2/3 of the total weight on \a v1 and 1/3 on \a v0 and \a v2. One solution to this is to
// counter-weight in anticipation of a cubic B-spline function to force the function to go through the
// original data points. Combined with this filter, this function performs an actual interpolation of the data.
// http://www2.cs.uregina.ca/~anima/408/Notes/Interpolation/UniformBSpline.htm
// http://www.dannyruijters.nl/cubicinterpolation/
namespace noa::geometry::interpolate {
    template<typename Value, typename Coord, typename = details::enable_if_valid_interp_t<Value, Coord>>
    constexpr NOA_IHD Value cubic_bspline_1d(Value v0, Value v1, Value v2, Value v3, Coord r) {
        using real_t = nt::value_type_t<Value>;
        real_t w0, w1, w2, w3;
        details::bspline_weights(r, &w0, &w1, &w2, &w3);
        return v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3;
    }

    template<typename Value, typename Coord, typename = details::enable_if_valid_interp_t<Value, Coord>>
    constexpr NOA_HD Value cubic_bspline_2d(Value v[4][4], Coord rx, Coord ry) {
        using real_t = nt::value_type_t<Value>;
        real_t w0, w1, w2, w3;
        details::bspline_weights(rx, &w0, &w1, &w2, &w3);
        const Value a0 = v[0][0] * w0 + v[0][1] * w1 + v[0][2] * w2 + v[0][3] * w3;
        const Value a1 = v[1][0] * w0 + v[1][1] * w1 + v[1][2] * w2 + v[1][3] * w3;
        const Value a2 = v[2][0] * w0 + v[2][1] * w1 + v[2][2] * w2 + v[2][3] * w3;
        const Value a3 = v[3][0] * w0 + v[3][1] * w1 + v[3][2] * w2 + v[3][3] * w3;

        return cubic_bspline_1d(a0, a1, a2, a3, ry);
    }

    template<typename Value, typename Coord, typename = details::enable_if_valid_interp_t<Value, Coord>>
    constexpr NOA_HD Value cubic_bspline_3d(Value v[4][4][4], Coord rx, Coord ry, Coord rz) {
        using real_t = nt::value_type_t<Value>;
        real_t wx0, wx1, wx2, wx3;
        real_t wy0, wy1, wy2, wy3;
        details::bspline_weights(rx, &wx0, &wx1, &wx2, &wx3);
        details::bspline_weights(ry, &wy0, &wy1, &wy2, &wy3);

        Value x0, x1, x2, x3;
        x0 = v[0][0][0] * wx0 + v[0][0][1] * wx1 + v[0][0][2] * wx2 + v[0][0][3] * wx3;
        x1 = v[0][1][0] * wx0 + v[0][1][1] * wx1 + v[0][1][2] * wx2 + v[0][1][3] * wx3;
        x2 = v[0][2][0] * wx0 + v[0][2][1] * wx1 + v[0][2][2] * wx2 + v[0][2][3] * wx3;
        x3 = v[0][3][0] * wx0 + v[0][3][1] * wx1 + v[0][3][2] * wx2 + v[0][3][3] * wx3;
        const Value y0 = x0 * wy0 + x1 * wy1 + x2 * wy2 + x3 * wy3;

        x0 = v[1][0][0] * wx0 + v[1][0][1] * wx1 + v[1][0][2] * wx2 + v[1][0][3] * wx3;
        x1 = v[1][1][0] * wx0 + v[1][1][1] * wx1 + v[1][1][2] * wx2 + v[1][1][3] * wx3;
        x2 = v[1][2][0] * wx0 + v[1][2][1] * wx1 + v[1][2][2] * wx2 + v[1][2][3] * wx3;
        x3 = v[1][3][0] * wx0 + v[1][3][1] * wx1 + v[1][3][2] * wx2 + v[1][3][3] * wx3;
        const Value y1 = x0 * wy0 + x1 * wy1 + x2 * wy2 + x3 * wy3;

        x0 = v[2][0][0] * wx0 + v[2][0][1] * wx1 + v[2][0][2] * wx2 + v[2][0][3] * wx3;
        x1 = v[2][1][0] * wx0 + v[2][1][1] * wx1 + v[2][1][2] * wx2 + v[2][1][3] * wx3;
        x2 = v[2][2][0] * wx0 + v[2][2][1] * wx1 + v[2][2][2] * wx2 + v[2][2][3] * wx3;
        x3 = v[2][3][0] * wx0 + v[2][3][1] * wx1 + v[2][3][2] * wx2 + v[2][3][3] * wx3;
        const Value y2 = x0 * wy0 + x1 * wy1 + x2 * wy2 + x3 * wy3;

        x0 = v[3][0][0] * wx0 + v[3][0][1] * wx1 + v[3][0][2] * wx2 + v[3][0][3] * wx3;
        x1 = v[3][1][0] * wx0 + v[3][1][1] * wx1 + v[3][1][2] * wx2 + v[3][1][3] * wx3;
        x2 = v[3][2][0] * wx0 + v[3][2][1] * wx1 + v[3][2][2] * wx2 + v[3][2][3] * wx3;
        x3 = v[3][3][0] * wx0 + v[3][3][1] * wx1 + v[3][3][2] * wx2 + v[3][3][3] * wx3;
        const Value y3 = x0 * wy0 + x1 * wy1 + x2 * wy2 + x3 * wy3;

        return cubic_bspline_1d(y0, y1, y2, y3, rz);
    }
}
