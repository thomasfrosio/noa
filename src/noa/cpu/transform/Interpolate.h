/// \file noa/cpu/transform/Interpolate.h
/// \brief Interpolate values using different methods of interpolation.
/// \author Thomas - ffyr2w
/// \date 3 Jul 2020
/// \details The values are entered in the x, then y, then z order.
///          The naming convention is vX for 1D, vYX for 2D and vZYX for 3D.
///          For cubic interpolations, the 4x4 square and 4x4x4 cube are entered in the [z][y][x] order.

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Math.h"
#include "noa/common/Types.h"
#include "noa/common/traits/BaseTypes.h"

// With nearest neighbor, the coordinate is directly rounded (e.g. array[floor(x+0.5)]),
// so there's no need to have a function for it.

#define NOA_ENABLE_IF_FP_ \
std::enable_if_t<(noa::traits::is_float_v<T> || noa::traits::is_complex_v<T>) && noa::traits::is_float_v<R>>

namespace noa::cpu::transform::details::bspline {
    template<typename R, typename T,
             typename = std::enable_if_t<noa::traits::is_float_v<T> && noa::traits::is_float_v<R>>>
    constexpr NOA_IH void weights(R ratio, T* w0, T* w1, T* w2, T* w3);
}

namespace noa::cpu::transform {
    /// Returns the linear interpolation.
    /// \tparam T   Type of the data. float, double, cfloat_t or cdouble_t.
    /// \tparam R   Type of the coordinate. float or double
    /// \param v0   Value at the first point of the interpolated line.
    /// \param v1   Value at the second point of the interpolated line.
    /// \param r    Coordinate/fraction where to interpolate. O points at \a v1, 1 points at \a v2.
    /// \return     Interpolated value.
    template<typename T, typename R, typename = NOA_ENABLE_IF_FP_>
    constexpr NOA_IH T linear1D(T v0, T v1, R r) noexcept {
        return static_cast<noa::traits::value_type_t<T>>(r) * (v1 - v0) + v0;
    }

    /// Returns the bilinear interpolation.
    /// This function assumes an unit square, i.e. the (x,y) coordinates, \a rx and \a ry, are between 0 and 1.
    template<typename T, typename R, typename = NOA_ENABLE_IF_FP_>
    constexpr NOA_IH T linear2D(T v00, T v01, T v10, T v11, R rx, R ry) noexcept {
        T tmp1 = linear1D(v00, v01, rx);
        T tmp2 = linear1D(v10, v11, rx);
        return linear1D(tmp1, tmp2, ry); // https://godbolt.org/z/eGcThbaGG
    }

    /// Returns the trilinear interpolation.
    /// This function assumes an unit cube, i.e. the (x,y,z) coordinates, \a rx, \a ry and \a rz, are between 0 and 1.
    template<typename T, typename R, typename = NOA_ENABLE_IF_FP_>
    constexpr NOA_IH T linear3D(T v000, T v001, T v010, T v011,
                                T v100, T v101, T v110, T v111,
                                R rx, R ry, R rz) noexcept {
        T tmp1 = linear2D(v000, v001, v010, v011, rx, ry);
        T tmp2 = linear2D(v100, v101, v110, v111, rx, ry);
        return linear1D(tmp1, tmp2, rz);
    }

    /// Returns the linear interpolation with cosine smoothing.
    /// \tparam T   Type of the data. float, double, cfloat_t or cdouble_t.
    /// \tparam R   Type of the coordinate. float or double
    /// \param v0   Value at the first point of the interpolated line.
    /// \param v1   Value at the second point of the interpolated line.
    /// \param r    Coordinate/fraction where to interpolate. O points at \a v1, 1 points at \a v2.
    /// \return     Interpolated value.
    template<typename T, typename R, typename = NOA_ENABLE_IF_FP_>
    constexpr NOA_IH T cosine1D(T v0, T v1, R r) {
        R tmp = (static_cast<R>(1) - math::cos(r * math::Constants<R>::PI)) / static_cast<R>(2);
        return linear1D(v0, v1, tmp);
    }

    /// Returns the bilinear interpolation with cosine smoothing.
    /// This function assumes an unit square, i.e. the (x,y) coordinates, \a rx and \a ry, are between 0 and 1.
    template<typename T, typename R, typename = NOA_ENABLE_IF_FP_>
    constexpr NOA_IH T cosine2D(T v00, T v01, T v10, T v11, R rx, R ry) {
        R tmp1 = (static_cast<R>(1) - math::cos(rx * math::Constants<R>::PI)) / static_cast<R>(2);
        R tmp2 = (static_cast<R>(1) - math::cos(ry * math::Constants<R>::PI)) / static_cast<R>(2);
        return linear2D(v00, v01, v10, v11, tmp1, tmp2);
    }

    /// Returns the trilinear interpolation with cosine smoothing.
    /// This function assumes an unit cube, i.e. the (x,y,z) coordinates, \a rx, \a ry and \a rz, are between 0 and 1.
    template<typename T, typename R, typename = NOA_ENABLE_IF_FP_>
    constexpr NOA_IH T cosine3D(T v000, T v001, T v010, T v011,
                                T v100, T v101, T v110, T v111,
                                R rx, R ry, R rz) {
        R tmp1 = (static_cast<R>(1) - math::cos(rx * math::Constants<R>::PI)) / static_cast<R>(2);
        R tmp2 = (static_cast<R>(1) - math::cos(ry * math::Constants<R>::PI)) / static_cast<R>(2);
        R tmp3 = (static_cast<R>(1) - math::cos(rz * math::Constants<R>::PI)) / static_cast<R>(2);
        return linear3D(v000, v001, v010, v011, v100, v101, v110, v111, tmp1, tmp2, tmp3);
    }

    /// Returns the 1D cubic interpolated value.
    /// \tparam T   Type of the data. float, double, cfloat_t or cdouble_t.
    /// \tparam R   Type of the coordinate. float or double
    /// \param v0   Value at the first point of the interpolated line.
    /// \param v1   Value at the second point of the interpolated line.
    /// \param v2   Value at the third point of the interpolated line.
    /// \param v3   Value at the fourth point of the interpolated line.
    /// \param r    Coordinate/fraction where to interpolate. O points at \a v1, 1 points at \a v2.
    /// \return     Interpolated value.
    template<typename T, typename R, typename = NOA_ENABLE_IF_FP_>
    constexpr NOA_HOST T cubic1D(T v0, T v1, T v2, T v3, R r) {
        T a0 = v3 - v2 - v0 + v1;
        T a1 = v0 - v1 - a0;
        T a2 = v2 - v0;
        T a3 = v1;

        auto r1 = static_cast<noa::traits::value_type_t<T>>(r);
        auto r2 = r1 * r1;
        auto r3 = r2 * r1;
        return a0 * r3 + a1 * r2 + a2 * r1 + a3;
    }

    /// Returns the 2D bicubic interpolated value.
    /// \tparam T   Type of the data. float, double, cfloat_t or cdouble_t.
    /// \tparam R   Type of the coordinates. float or double
    /// \param v    Values at the "edges" of the unit square, in the [y][x] order.
    /// \param rx   Coordinate/fraction in x where to interpolate. O points at v[y][1], 1 points at v[y][2].
    /// \param ry   Coordinate/fraction in y where to interpolate. O points at v[1][x], 1 points at v[2][x].
    /// \return     Interpolated value.
    template<typename T, typename R, typename = NOA_ENABLE_IF_FP_>
    constexpr NOA_HOST T cubic2D(T v[4][4], R rx, R ry) {
        T a0 = cubic1D(v[0][0], v[0][1], v[0][2], v[0][3], rx);
        T a1 = cubic1D(v[1][0], v[1][1], v[1][2], v[1][3], rx);
        T a2 = cubic1D(v[2][0], v[2][1], v[2][2], v[2][3], rx);
        T a3 = cubic1D(v[3][0], v[3][1], v[3][2], v[3][3], rx);
        return cubic1D(a0, a1, a2, a3, ry);
    }

    /// Returns the 3D cubic interpolated value.
    /// \tparam T   Type of the data. float, double, cfloat_t or cdouble_t.
    /// \tparam R   Type of the coordinate. float or double.
    /// \param v    Values at the "edges" of the unit cube, in the [z][y][x] order.
    /// \param rx   Coordinate/fraction in x where to interpolate. O points at v[z][y][1], 1 points at v[z][y][2].
    /// \param ry   Coordinate/fraction in y where to interpolate. O points at v[z][1][x], 1 points at v[z][2][x].
    /// \param rz   Coordinate/fraction in z where to interpolate. O points at v[1][y][x], 1 points at v[2][y][x].
    /// \return     Interpolated value.
    template<typename T, typename R, typename = NOA_ENABLE_IF_FP_>
    constexpr NOA_HOST T cubic3D(T v[4][4][4], R rx, R ry, R rz) {
        T a0 = cubic2D(v[0], rx, ry);
        T a1 = cubic2D(v[1], rx, ry);
        T a2 = cubic2D(v[2], rx, ry);
        T a3 = cubic2D(v[3], rx, ry);
        return cubic1D(a0, a1, a2, a3, rz);
    }

    /// Returns the 1D cubic B-spline "interpolated" value.
    /// \tparam T   Type of the data. float, double, cfloat or cdouble_t.
    /// \tparam R   Type of the coordinate. float or double.
    /// \param v0   Value at the first control point.
    /// \param v1   Value at the second control point.
    /// \param v2   Value at the third control point.
    /// \param v3   Value at the fourth control point.
    /// \param r    Coordinate/fraction where to "interpolate". O points at \a v1, 1 points at \a v2.
    /// \return     "Interpolated" value.
    ///
    /// \note Cubic B-spline curves are not constrained to pass through the data, i.e. data points are simply referred
    ///       to as control points. As such, these curves are not really interpolating. For instance, a ration of 0
    ///       puts only 2/3 of the total weight on \a v1 and 1/3 on \a v0 and \a v2. One solution to this is to
    ///       counter-weight in anticipation of a cubic B-spline function to force the function to go through the
    ///       original data points. Combined with this filter, this function performs an actual interpolation of the data.
    ///
    /// \see noa::transform::bspline::prefilter()
    /// \see http://www2.cs.uregina.ca/~anima/408/Notes/Interpolation/UniformBSpline.htm
    /// \see http://www.dannyruijters.nl/cubicinterpolation/
    template<typename T, typename R, typename = NOA_ENABLE_IF_FP_>
    constexpr NOA_IH T cubicBSpline1D(T v0, T v1, T v2, T v3, R r) {
        using real_t = noa::traits::value_type_t<T>;
        real_t w0, w1, w2, w3;
        details::bspline::weights(r, &w0, &w1, &w2, &w3);
        return v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3;
    }

    /// Returns the 2D cubic B-spline "interpolated" value.
    /// \tparam T   Type of the data. float, double, cfloat or cdouble_t.
    /// \tparam R   Type of the coordinate. float or double.
    /// \param v    Values at the "edges" of the unit square, in the [y][x] order.
    /// \param rx   Coordinate/fraction in x where to "interpolate". O points at v[y][1], 1 points at v[y][2].
    /// \param ry   Coordinate/fraction in y where to "interpolate". O points at v[1][x], 1 points at v[2][x].
    /// \return     "Interpolated" value.
    ///
    /// \see noa::transform::cubicBSpline1D() for more details.
    template<typename T, typename R, typename = NOA_ENABLE_IF_FP_>
    constexpr NOA_HOST T cubicBSpline2D(T v[4][4], R rx, R ry) {
        using real_t = noa::traits::value_type_t<T>;
        real_t w0, w1, w2, w3;
        details::bspline::weights(rx, &w0, &w1, &w2, &w3);
        T a0 = v[0][0] * w0 + v[0][1] * w1 + v[0][2] * w2 + v[0][3] * w3;
        T a1 = v[1][0] * w0 + v[1][1] * w1 + v[1][2] * w2 + v[1][3] * w3;
        T a2 = v[2][0] * w0 + v[2][1] * w1 + v[2][2] * w2 + v[2][3] * w3;
        T a3 = v[3][0] * w0 + v[3][1] * w1 + v[3][2] * w2 + v[3][3] * w3;

        return cubicBSpline1D(a0, a1, a2, a3, ry);
    }

    /// Returns the 3D cubic B-spline "interpolated" value.
    /// \tparam T   Type of the data. float, double, cfloat or cdouble_t.
    /// \tparam R   Type of the coordinate. float or double.
    /// \param v    Values at the "edges" of the unit cube, in the [z][y][x] order.
    /// \param rx   Coordinate/fraction in x where to "interpolate". O points at v[z][y][1], 1 points at v[z][y][2].
    /// \param ry   Coordinate/fraction in y where to "interpolate". O points at v[z][1][x], 1 points at v[z][2][x].
    /// \param rz   Coordinate/fraction in z where to "interpolate". O points at v[1][y][x], 1 points at v[2][y][x].
    /// \return     "Interpolated" value.
    ///
    /// \see noa::transform::cubicBSpline1D() for more details.
    template<typename T, typename R, typename = NOA_ENABLE_IF_FP_>
    constexpr NOA_HOST T cubicBSpline3D(T v[4][4][4], R rx, R ry, R rz) {
        using real_t = noa::traits::value_type_t<T>;
        real_t wx0, wx1, wx2, wx3;
        real_t wy0, wy1, wy2, wy3;
        details::bspline::weights(rx, &wx0, &wx1, &wx2, &wx3);
        details::bspline::weights(ry, &wy0, &wy1, &wy2, &wy3);

        T x0, x1, x2, x3;
        x0 = v[0][0][0] * wx0 + v[0][0][1] * wx1 + v[0][0][2] * wx2 + v[0][0][3] * wx3;
        x1 = v[0][1][0] * wx0 + v[0][1][1] * wx1 + v[0][1][2] * wx2 + v[0][1][3] * wx3;
        x2 = v[0][2][0] * wx0 + v[0][2][1] * wx1 + v[0][2][2] * wx2 + v[0][2][3] * wx3;
        x3 = v[0][3][0] * wx0 + v[0][3][1] * wx1 + v[0][3][2] * wx2 + v[0][3][3] * wx3;
        const T y0 = x0 * wy0 + x1 * wy1 + x2 * wy2 + x3 * wy3;

        x0 = v[1][0][0] * wx0 + v[1][0][1] * wx1 + v[1][0][2] * wx2 + v[1][0][3] * wx3;
        x1 = v[1][1][0] * wx0 + v[1][1][1] * wx1 + v[1][1][2] * wx2 + v[1][1][3] * wx3;
        x2 = v[1][2][0] * wx0 + v[1][2][1] * wx1 + v[1][2][2] * wx2 + v[1][2][3] * wx3;
        x3 = v[1][3][0] * wx0 + v[1][3][1] * wx1 + v[1][3][2] * wx2 + v[1][3][3] * wx3;
        const T y1 = x0 * wy0 + x1 * wy1 + x2 * wy2 + x3 * wy3;

        x0 = v[2][0][0] * wx0 + v[2][0][1] * wx1 + v[2][0][2] * wx2 + v[2][0][3] * wx3;
        x1 = v[2][1][0] * wx0 + v[2][1][1] * wx1 + v[2][1][2] * wx2 + v[2][1][3] * wx3;
        x2 = v[2][2][0] * wx0 + v[2][2][1] * wx1 + v[2][2][2] * wx2 + v[2][2][3] * wx3;
        x3 = v[2][3][0] * wx0 + v[2][3][1] * wx1 + v[2][3][2] * wx2 + v[2][3][3] * wx3;
        const T y2 = x0 * wy0 + x1 * wy1 + x2 * wy2 + x3 * wy3;

        x0 = v[3][0][0] * wx0 + v[3][0][1] * wx1 + v[3][0][2] * wx2 + v[3][0][3] * wx3;
        x1 = v[3][1][0] * wx0 + v[3][1][1] * wx1 + v[3][1][2] * wx2 + v[3][1][3] * wx3;
        x2 = v[3][2][0] * wx0 + v[3][2][1] * wx1 + v[3][2][2] * wx2 + v[3][2][3] * wx3;
        x3 = v[3][3][0] * wx0 + v[3][3][1] * wx1 + v[3][3][2] * wx2 + v[3][3][3] * wx3;
        const T y3 = x0 * wy0 + x1 * wy1 + x2 * wy2 + x3 * wy3;

        return cubicBSpline1D(y0, y1, y2, y3, rz);
    }
}

namespace noa::cpu::transform::bspline {
    /// Applies a 1D prefilter to \a inputs so that the cubic B-spline values will pass through the sample data.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] inputs       On the \p host.Input arrays. One per batch.
    /// \param[out] outputs     On the \p host. Output arrays. One per batch. Can be equal to \a inputs.
    /// \param size             Size, in elements, of \a inputs and \a outputs, of one batch.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    ///
    /// \see noa::transform::bspline::prefilter2D() for more details.
    /// \see http://www.dannyruijters.nl/cubicinterpolation/ for more details.
    template<typename T>
    NOA_HOST void prefilter1D(const T* inputs, T* outputs, size_t size, uint batches);

    /// Applies a 2D prefilter to \a inputs so that the cubic B-spline values will pass through the sample data.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] inputs       On the \p host.Input arrays. One per batch.
    /// \param[out] outputs     On the \p host. Output arrays. One per batch. Can be equal to \a inputs.
    /// \param shape            Logical {fast, medium} shape of \a inputs and \a outputs, ignoring the batches.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    ///
    /// \details From Danny Ruijters:
    ///          "When the approach described above is directly applied, it will result in smoothened images.
    ///          This is caused by the fact that the cubic B-spline filtering yields a function that does not
    ///          pass through its coefficients (i.e. texture values). In order to wind up with a cubic B-spline
    ///          interpolated image that passes through the original samples, we need to pre-filter the texture".
    /// \see http://www.dannyruijters.nl/cubicinterpolation/ for more details.
    template<typename T>
    NOA_HOST void prefilter2D(const T* inputs, T* outputs, size2_t shape, uint batches);

    /// Applies a 3D prefilter to \a inputs so that the cubic B-spline values will pass through the sample data.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] inputs       On the \p host.Input arrays. One per batch.
    /// \param[out] outputs     On the \p host. Output arrays. One per batch. Can be equal to \a inputs.
    /// \param shape            Logical {fast, medium, slow} shape of \a inputs and \a outputs, ignoring the batches.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    ///
    /// \see noa::transform::bspline::prefilter2D() for more details.
    /// \see http://www.dannyruijters.nl/cubicinterpolation/ for more details.
    template<typename T>
    NOA_HOST void prefilter3D(const T* inputs, T* outputs, size3_t shape, uint batches);

    /// Applies a prefilter to \a inputs so that the cubic B-spline values will pass through the sample data.
    template<typename T>
    NOA_IH void prefilter(const T* inputs, T* outputs, size3_t shape, uint batches) {
        uint ndim = getNDim(shape);
        switch (ndim) {
            case 1:
                prefilter1D(inputs, outputs, shape.x, batches);
                break;
            case 2:
                prefilter2D(inputs, outputs, size2_t(shape.x, shape.y), batches);
                break;
            case 3:
                prefilter3D(inputs, outputs, shape, batches);
                break;
            default:
                NOA_THROW("DEV: This should not have happened...");
        }
    }
}

// -- Implementation -- //

namespace noa::cpu::transform::details::bspline {
    template<typename R, typename T, typename>
    constexpr void weights(R ratio, T* w0, T* w1, T* w2, T* w3) {
        constexpr R one_sixth = static_cast<R>(1) / static_cast<R>(6);
        constexpr R two_third = static_cast<R>(2) / static_cast<R>(3);
        const R one_minus = 1.0f - ratio;
        const R one_squared = one_minus * one_minus;
        const R squared = ratio * ratio;

        *w0 = static_cast<T>(one_sixth * one_squared * one_minus);
        *w1 = static_cast<T>(two_third - static_cast<R>(0.5f) * squared * (2 - ratio));
        *w2 = static_cast<T>(two_third - static_cast<R>(0.5f) * one_squared * (2 - one_minus));
        *w3 = static_cast<T>(one_sixth * squared * ratio);
    }
}

#undef NOA_ENABLE_IF_FP_
