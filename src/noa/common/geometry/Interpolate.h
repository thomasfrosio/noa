#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/Types.h"

namespace noa::geometry::interpolate::details {
    template<typename data_t, typename coord_t>
    constexpr bool is_valid_interp_v = traits::is_float_or_complex_v<data_t> && traits::is_float_v<coord_t>;

    template<typename data_t, typename coord_t>
    using enable_if_valid_interp_t = std::enable_if_t<details::is_valid_interp_v<data_t, coord_t>>;

    template<typename coord_t, typename data_t>
    constexpr NOA_IHD void bsplineWeights(coord_t ratio, data_t* w0, data_t* w1, data_t* w2, data_t* w3) {
        constexpr coord_t ONE_SIXTH = static_cast<coord_t>(1) / static_cast<coord_t>(6);
        constexpr coord_t TWO_THIRD = static_cast<coord_t>(2) / static_cast<coord_t>(3);
        const coord_t one_minus = 1.0f - ratio;
        const coord_t one_squared = one_minus * one_minus;
        const coord_t squared = ratio * ratio;

        *w0 = static_cast<data_t>(ONE_SIXTH * one_squared * one_minus);
        *w1 = static_cast<data_t>(TWO_THIRD - static_cast<coord_t>(0.5) * squared * (2 - ratio));
        *w2 = static_cast<data_t>(TWO_THIRD - static_cast<coord_t>(0.5) * one_squared * (2 - one_minus));
        *w3 = static_cast<data_t>(ONE_SIXTH * squared * ratio);
    }
}

// Linear interpolation:
namespace noa::geometry::interpolate {
    template<typename data_t, typename coord_t, typename = details::enable_if_valid_interp_t<data_t, coord_t>>
    constexpr NOA_FHD data_t lerp1D(data_t v0, data_t v1, coord_t r) noexcept {
        using value_t = traits::value_type_t<data_t>;
        return static_cast<value_t>(r) * (v1 - v0) + v0;
    }

    template<typename data_t, typename coord_t, typename = details::enable_if_valid_interp_t<data_t, coord_t>>
    constexpr NOA_FHD data_t lerp2D(data_t v00, data_t v01, data_t v10, data_t v11, coord_t rx, coord_t ry) noexcept {
        const data_t tmp1 = lerp1D(v00, v01, rx);
        const data_t tmp2 = lerp1D(v10, v11, rx);
        return lerp1D(tmp1, tmp2, ry); // https://godbolt.org/z/eGcThbaGG
    }

    template<typename data_t, typename coord_t, typename = details::enable_if_valid_interp_t<data_t, coord_t>>
    constexpr NOA_FHD data_t lerp3D(data_t v000, data_t v001, data_t v010, data_t v011,
                                    data_t v100, data_t v101, data_t v110, data_t v111,
                                    coord_t rx, coord_t ry, coord_t rz) noexcept {
        const data_t tmp1 = lerp2D(v000, v001, v010, v011, rx, ry);
        const data_t tmp2 = lerp2D(v100, v101, v110, v111, rx, ry);
        return lerp1D(tmp1, tmp2, rz);
    }
}

// Linear interpolation with cosine weighting of the fraction:
namespace noa::geometry::interpolate {
    template<typename data_t, typename coord_t, typename = details::enable_if_valid_interp_t<data_t, coord_t>>
    constexpr NOA_IHD data_t cosine1D(data_t v0, data_t v1, coord_t r) {
        constexpr coord_t PI = ::noa::math::Constants<coord_t>::PI;
        const coord_t tmp = (static_cast<coord_t>(1) - ::noa::math::cos(r * PI)) / static_cast<coord_t>(2);
        return lerp1D(v0, v1, tmp);
    }

    template<typename data_t, typename coord_t, typename = details::enable_if_valid_interp_t<data_t, coord_t>>
    constexpr NOA_IHD data_t cosine2D(data_t v00, data_t v01, data_t v10, data_t v11, coord_t rx, coord_t ry) {
        constexpr coord_t PI = ::noa::math::Constants<coord_t>::PI;
        const coord_t tmp1 = (static_cast<coord_t>(1) - ::noa::math::cos(rx * PI)) / static_cast<coord_t>(2);
        const coord_t tmp2 = (static_cast<coord_t>(1) - ::noa::math::cos(ry * PI)) / static_cast<coord_t>(2);
        return lerp2D(v00, v01, v10, v11, tmp1, tmp2);
    }

    template<typename data_t, typename coord_t, typename = details::enable_if_valid_interp_t<data_t, coord_t>>
    constexpr NOA_IHD data_t cosine3D(data_t v000, data_t v001, data_t v010, data_t v011,
                                      data_t v100, data_t v101, data_t v110, data_t v111,
                                      coord_t rx, coord_t ry, coord_t rz) {
        constexpr coord_t PI = ::noa::math::Constants<coord_t>::PI;
        const coord_t tmp1 = (static_cast<coord_t>(1) - ::noa::math::cos(rx * PI)) / static_cast<coord_t>(2);
        const coord_t tmp2 = (static_cast<coord_t>(1) - ::noa::math::cos(ry * PI)) / static_cast<coord_t>(2);
        const coord_t tmp3 = (static_cast<coord_t>(1) - ::noa::math::cos(rz * PI)) / static_cast<coord_t>(2);
        return lerp3D(v000, v001, v010, v011, v100, v101, v110, v111, tmp1, tmp2, tmp3);
    }
}

// Cubic interpolation:
namespace noa::geometry::interpolate {
    template<typename data_t, typename coord_t, typename = details::enable_if_valid_interp_t<data_t, coord_t>>
    constexpr NOA_HD data_t cubic1D(data_t v0, data_t v1, data_t v2, data_t v3, coord_t r) {
        const data_t a0 = v3 - v2 - v0 + v1;
        const data_t a1 = v0 - v1 - a0;
        const data_t a2 = v2 - v0;
        // a3 = v1

        using real_t = traits::value_type_t<data_t>;
        const auto r1 = static_cast<real_t>(r);
        const auto r2 = r1 * r1;
        const auto r3 = r2 * r1;
        return a0 * r3 + a1 * r2 + a2 * r1 + v1;
    }

    template<typename data_t, typename coord_t, typename = details::enable_if_valid_interp_t<data_t, coord_t>>
    constexpr NOA_HD data_t cubic2D(data_t v[4][4], coord_t rx, coord_t ry) {
        const data_t a0 = cubic1D(v[0][0], v[0][1], v[0][2], v[0][3], rx);
        const data_t a1 = cubic1D(v[1][0], v[1][1], v[1][2], v[1][3], rx);
        const data_t a2 = cubic1D(v[2][0], v[2][1], v[2][2], v[2][3], rx);
        const data_t a3 = cubic1D(v[3][0], v[3][1], v[3][2], v[3][3], rx);
        return cubic1D(a0, a1, a2, a3, ry);
    }

    template<typename data_t, typename coord_t, typename = details::enable_if_valid_interp_t<data_t, coord_t>>
    constexpr NOA_HD data_t cubic3D(data_t v[4][4][4], coord_t rx, coord_t ry, coord_t rz) {
        const data_t a0 = cubic2D(v[0], rx, ry);
        const data_t a1 = cubic2D(v[1], rx, ry);
        const data_t a2 = cubic2D(v[2], rx, ry);
        const data_t a3 = cubic2D(v[3], rx, ry);
        return cubic1D(a0, a1, a2, a3, rz);
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
    template<typename data_t, typename coord_t, typename = details::enable_if_valid_interp_t<data_t, coord_t>>
    constexpr NOA_IHD data_t cubicBSpline1D(data_t v0, data_t v1, data_t v2, data_t v3, coord_t r) {
        using real_t = traits::value_type_t<data_t>;
        real_t w0, w1, w2, w3;
        details::bsplineWeights(r, &w0, &w1, &w2, &w3);
        return v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3;
    }

    template<typename data_t, typename coord_t, typename = details::enable_if_valid_interp_t<data_t, coord_t>>
    constexpr NOA_HD data_t cubicBSpline2D(data_t v[4][4], coord_t rx, coord_t ry) {
        using real_t = traits::value_type_t<data_t>;
        real_t w0, w1, w2, w3;
        details::bsplineWeights(rx, &w0, &w1, &w2, &w3);
        const data_t a0 = v[0][0] * w0 + v[0][1] * w1 + v[0][2] * w2 + v[0][3] * w3;
        const data_t a1 = v[1][0] * w0 + v[1][1] * w1 + v[1][2] * w2 + v[1][3] * w3;
        const data_t a2 = v[2][0] * w0 + v[2][1] * w1 + v[2][2] * w2 + v[2][3] * w3;
        const data_t a3 = v[3][0] * w0 + v[3][1] * w1 + v[3][2] * w2 + v[3][3] * w3;

        return cubicBSpline1D(a0, a1, a2, a3, ry);
    }

    template<typename data_t, typename coord_t, typename = details::enable_if_valid_interp_t<data_t, coord_t>>
    constexpr NOA_HD data_t cubicBSpline3D(data_t v[4][4][4], coord_t rx, coord_t ry, coord_t rz) {
        using real_t = traits::value_type_t<data_t>;
        real_t wx0, wx1, wx2, wx3;
        real_t wy0, wy1, wy2, wy3;
        details::bsplineWeights(rx, &wx0, &wx1, &wx2, &wx3);
        details::bsplineWeights(ry, &wy0, &wy1, &wy2, &wy3);

        data_t x0, x1, x2, x3;
        x0 = v[0][0][0] * wx0 + v[0][0][1] * wx1 + v[0][0][2] * wx2 + v[0][0][3] * wx3;
        x1 = v[0][1][0] * wx0 + v[0][1][1] * wx1 + v[0][1][2] * wx2 + v[0][1][3] * wx3;
        x2 = v[0][2][0] * wx0 + v[0][2][1] * wx1 + v[0][2][2] * wx2 + v[0][2][3] * wx3;
        x3 = v[0][3][0] * wx0 + v[0][3][1] * wx1 + v[0][3][2] * wx2 + v[0][3][3] * wx3;
        const data_t y0 = x0 * wy0 + x1 * wy1 + x2 * wy2 + x3 * wy3;

        x0 = v[1][0][0] * wx0 + v[1][0][1] * wx1 + v[1][0][2] * wx2 + v[1][0][3] * wx3;
        x1 = v[1][1][0] * wx0 + v[1][1][1] * wx1 + v[1][1][2] * wx2 + v[1][1][3] * wx3;
        x2 = v[1][2][0] * wx0 + v[1][2][1] * wx1 + v[1][2][2] * wx2 + v[1][2][3] * wx3;
        x3 = v[1][3][0] * wx0 + v[1][3][1] * wx1 + v[1][3][2] * wx2 + v[1][3][3] * wx3;
        const data_t y1 = x0 * wy0 + x1 * wy1 + x2 * wy2 + x3 * wy3;

        x0 = v[2][0][0] * wx0 + v[2][0][1] * wx1 + v[2][0][2] * wx2 + v[2][0][3] * wx3;
        x1 = v[2][1][0] * wx0 + v[2][1][1] * wx1 + v[2][1][2] * wx2 + v[2][1][3] * wx3;
        x2 = v[2][2][0] * wx0 + v[2][2][1] * wx1 + v[2][2][2] * wx2 + v[2][2][3] * wx3;
        x3 = v[2][3][0] * wx0 + v[2][3][1] * wx1 + v[2][3][2] * wx2 + v[2][3][3] * wx3;
        const data_t y2 = x0 * wy0 + x1 * wy1 + x2 * wy2 + x3 * wy3;

        x0 = v[3][0][0] * wx0 + v[3][0][1] * wx1 + v[3][0][2] * wx2 + v[3][0][3] * wx3;
        x1 = v[3][1][0] * wx0 + v[3][1][1] * wx1 + v[3][1][2] * wx2 + v[3][1][3] * wx3;
        x2 = v[3][2][0] * wx0 + v[3][2][1] * wx1 + v[3][2][2] * wx2 + v[3][2][3] * wx3;
        x3 = v[3][3][0] * wx0 + v[3][3][1] * wx1 + v[3][3][2] * wx2 + v[3][3][3] * wx3;
        const data_t y3 = x0 * wy0 + x1 * wy1 + x2 * wy2 + x3 * wy3;

        return cubicBSpline1D(y0, y1, y2, y3, rz);
    }
}
