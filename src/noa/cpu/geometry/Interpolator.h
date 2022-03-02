/// \file noa/cpu/geometry/Interpolator.h
/// \brief 1D, 2D and 3D interpolators.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Math.h"
#include "noa/cpu/geometry/Interpolate.h"

// On of the main difference between these Interpolators and what we can find on other cryoEM packages,
// is that the interpolation window can be partially OOB, that is, elements that are OOB are replaced
// according to a BorderMode. cryoEM packages usually check that all elements are in bound and if there's
// even one element OOB, they don't interpolate.
// Note: These Interpolators are for real space interpolation or redundant and centered Fourier transforms.

namespace noa::cpu::geometry {
    /// Interpolates 1D data.
    /// \details Simple helper to interpolate data, given a InterpMode and BorderMode.
    ///          Supported interpolation methods: INTERP_NEAREST, INTERP_LINEAR, INTERP_COSINE,
    ///                                           INTERP_CUBIC, INTERP_CUBIC_BSPLINE
    ///          Supported border modes: BORDER_ZERO, BORDER_VALUE, BORDER_CLAMP,
    ///                                  BORDER_PERIODIC, BORDER_MIRROR, BORDER_REFLECT
    ///
    /// \tparam T Type of the interpolated data. float, double, cfloat_t or cdouble_t.
    /// \note The coordinate system matches the indexing. The coordinate is the floating-point passed to `get<>()`.
    ///       For instance the first data sample at index 0 is located at the coordinate 0 and the coordinate 0.5
    ///       is just in between the first and second element. As such, the fractional part of the coordinate
    ///       corresponds to the ratio/weight used by the interpolation function (e.g. linear1D()). In other words,
    ///       the coordinate system locates the data between -0.5 and N-1 + 0.5.
    template<typename T>
    class Interpolator1D {
    public:
        /// Empty state. Use reset() to set data points.
        Interpolator1D() = default;

        /// Sets the data points.
        /// \param[in] input    On the \b host. Input contiguous 1D array.
        /// \param stride       Stride, in elements, of \p input.
        /// \param shape        Shape, in elements, of \p input.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST Interpolator1D(const T* input, size_t stride, size_t shape, T value) noexcept;

        /// Returns the interpolated value at the coordinate \p x.
        /// \tparam BORDER      Border/addressing mode.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \param x            Coordinate to interpolate at.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float x) const;

        /// Returns the interpolated value at the coordinate \p x.
        /// \tparam BORDER      Border/addressing mode.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \param x            Coordinate to interpolate at.
        /// \param offset       Temporary memory offset to apply to the underlying array.
        ///                     This is used for instance to change batches.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float x, size_t offset) const;

    private:
        const T* m_data{};
        int m_stride{};
        int m_shape{};
        T m_value{};
    private:
        template<BorderMode BORDER>
        NOA_HOST T nearest_(const T* data, float x) const;
        template<BorderMode BORDER, bool COSINE>
        NOA_HOST T linear_(const T* data, float x) const;
        template<BorderMode BORDER, bool BSPLINE>
        NOA_HOST T cubic_(const T* data, float x) const;
    };

    /// Interpolates 2D data.
    /// \details Simple helper to interpolate data, given a InterpMode and BorderMode.
    ///          Supported interpolation methods: INTERP_NEAREST, INTERP_LINEAR, INTERP_COSINE,
    ///                                           INTERP_CUBIC, INTERP_CUBIC_BSPLINE
    ///          Supported border modes: BORDER_ZERO, BORDER_VALUE, BORDER_CLAMP,
    ///                                  BORDER_PERIODIC, BORDER_MIRROR, BORDER_REFLECT
    ///
    /// \tparam T Type of the interpolated data. float, double, cfloat_t or cdouble_t.
    /// \see Interpolator1D for more details.
    /// \note With INTERP_CUBIC_BSPLINE, in order to have interpolation, i.e. computed line goes through
    ///       the input data, one should prefilter the data using the bspline::prefilter() function.
    template<typename T>
    class Interpolator2D {
    public:
        /// Empty state. Use reset() to set data points.
        Interpolator2D() = default;

        /// Sets the data points.
        /// \param[in] input    On the \b host. Input 2D array.
        /// \param stride       Rightmost stride, in elements, of \p input.
        /// \param shape        Rightmost shape, in elements, of \p input.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST Interpolator2D(const T* input, size2_t stride, size2_t shape, T value) noexcept;

        /// Returns the interpolated value at the coordinate \p x, \p y.
        /// \tparam BORDER      Border/addressing mode.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \param coords       Rightmost coordinates.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float2_t coords) const;

        /// Returns the interpolated value at the coordinate \p x, \p y.
        /// \tparam BORDER      Border/addressing mode.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \param coords       Rightmost coordinates.
        /// \param offset       Temporary memory offset to apply to the underlying array.
        ///                     This is used for instance to change batches.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float2_t coords, size_t offset) const;

    private:
        const T* m_data{};
        int2_t m_stride{};
        int2_t m_shape{};
        T m_value{};
    private:
        template<BorderMode BORDER>
        NOA_HOST T nearest_(const T* data, float y, float x) const;
        template<BorderMode BORDER, bool COSINE>
        NOA_HOST T linear_(const T* data, float y, float x) const;
        template<BorderMode BORDER, bool BSPLINE>
        NOA_HOST T cubic_(const T* data, float y, float x) const;
    };

    /// Interpolates 3D data.
    /// \details Simple helper to interpolate data, given a InterpMode and BorderMode.
    ///          Supported interpolation methods: INTERP_NEAREST, INTERP_LINEAR, INTERP_COSINE,
    ///                                           INTERP_CUBIC, INTERP_CUBIC_BSPLINE
    ///          Supported border modes: BORDER_ZERO, BORDER_VALUE, BORDER_CLAMP,
    ///                                  BORDER_PERIODIC, BORDER_MIRROR, BORDER_REFLECT
    ///
    /// \tparam T Type of the interpolated data. float, double, cfloat_t or cdouble_t.
    /// \see Interpolator1D for more details.
    /// \note With INTERP_CUBIC_BSPLINE, in order to have interpolation, i.e. computed line goes through
    ///       the input data, one should prefilter the data using the bspline::prefilter() function.
    template<typename T>
    class Interpolator3D {
    public:
        /// Empty state. Use reset() to set data points.
        Interpolator3D() = default;

        /// Sets the data points.
        /// \param[in] input    On the \b host. Input 3D array.
        /// \param stride       Rightmost stride, in elements, of \p input.
        /// \param shape        Rightmost shape, in elements, of \p input.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST Interpolator3D(const T* input, size3_t stride, size3_t shape, T value) noexcept;

        /// Returns the interpolated value at the coordinate \p x, \p y, \p z.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \tparam BORDER      Border/addressing mode.
        /// \param coords       Rightmost coordinates.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float3_t coords) const;

        /// Returns the interpolated value at the coordinate \p x, \p y, \p z.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \tparam BORDER      Border/addressing mode.
        /// \param coords       Rightmost coordinates.
        /// \param offset       Temporary memory offset to apply to the underlying array.
        ///                     This is used for instance to change batches.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float3_t coords, size_t offset) const;

    private:
        const T* m_data{};
        int3_t m_stride{};
        int3_t m_shape{};
        T m_value{};
    private:
        template<BorderMode BORDER>
        NOA_HOST T nearest_(const T* data, float z, float y, float x) const;
        template<BorderMode BORDER, bool COSINE>
        NOA_HOST T linear_(const T* data, float z, float y, float x) const;
        template<BorderMode BORDER, bool BSPLINE>
        NOA_HOST T cubic_(const T* data, float z, float y, float x) const;
    };
}

// Implementation:
namespace noa::cpu::geometry {
    template<typename T>
    Interpolator1D<T>::Interpolator1D(const T* input, size_t stride, size_t shape, T value) noexcept
            : m_data(input), m_stride(static_cast<int>(stride)), m_shape(static_cast<int>(shape)), m_value(value) {}

    template<typename T>
    template<BorderMode BORDER>
    T Interpolator1D<T>::nearest_(const T* data, float x) const {
        T out;
        auto idx = static_cast<int>(noa::math::round(x));
        if constexpr (BORDER == BORDER_ZERO) {
            out = idx >= 0 && idx < m_shape ? data[idx * m_stride] : static_cast<T>(0);
        } else if constexpr (BORDER == BORDER_VALUE) {
            out = idx >= 0 && idx < m_shape ? data[idx * m_stride] : static_cast<T>(m_value);
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            out = data[getBorderIndex<BORDER>(idx, m_shape) * m_stride];
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return out;
    }

    template<typename T>
    template<BorderMode BORDER, bool COSINE>
    T Interpolator1D<T>::linear_(const T* data, float x) const {
        int idx0 = static_cast<int>(noa::math::floor(x));
        int idx1 = idx0 + 1;
        T values[2];
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond[2] = {idx0 >= 0 && idx0 < m_shape, idx1 >= 0 && idx1 < m_shape};
            if constexpr (BORDER == BORDER_ZERO) {
                values[0] = cond[0] ? data[idx0 * m_stride] : static_cast<T>(0);
                values[1] = cond[1] ? data[idx1 * m_stride] : static_cast<T>(0);
            } else {
                values[0] = cond[0] ? data[idx0 * m_stride] : static_cast<T>(m_value);
                values[1] = cond[1] ? data[idx1 * m_stride] : static_cast<T>(m_value);
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            values[0] = data[getBorderIndex<BORDER>(idx0, m_shape) * m_stride];
            values[1] = data[getBorderIndex<BORDER>(idx1, m_shape) * m_stride];
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        float fraction = x - static_cast<float>(idx0);
        if constexpr (COSINE)
            return cosine1D(values[0], values[1], fraction);
        else
            return linear1D(values[0], values[1], fraction);
    }

    template<typename T>
    template<BorderMode BORDER, bool BSPLINE>
    T Interpolator1D<T>::cubic_(const T* data, float x) const {
        int idx1 = static_cast<int>(noa::math::floor(x));
        int idx0 = idx1 - 1;
        int idx2 = idx1 + 1;
        int idx3 = idx1 + 2;
        T values[4];
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond[4] = {idx0 >= 0 && idx0 < m_shape,
                            idx1 >= 0 && idx1 < m_shape,
                            idx2 >= 0 && idx2 < m_shape,
                            idx3 >= 0 && idx3 < m_shape};
            if constexpr (BORDER == BORDER_ZERO) {
                values[0] = cond[0] ? data[idx0 * m_stride] : static_cast<T>(0);
                values[1] = cond[1] ? data[idx1 * m_stride] : static_cast<T>(0);
                values[2] = cond[2] ? data[idx2 * m_stride] : static_cast<T>(0);
                values[3] = cond[3] ? data[idx3 * m_stride] : static_cast<T>(0);
            } else {
                values[0] = cond[0] ? data[idx0 * m_stride] : static_cast<T>(m_value);
                values[1] = cond[1] ? data[idx1 * m_stride] : static_cast<T>(m_value);
                values[2] = cond[2] ? data[idx2 * m_stride] : static_cast<T>(m_value);
                values[3] = cond[3] ? data[idx3 * m_stride] : static_cast<T>(m_value);
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            values[0] = data[getBorderIndex<BORDER>(idx0, m_shape) * m_stride];
            values[1] = data[getBorderIndex<BORDER>(idx1, m_shape) * m_stride];
            values[2] = data[getBorderIndex<BORDER>(idx2, m_shape) * m_stride];
            values[3] = data[getBorderIndex<BORDER>(idx3, m_shape) * m_stride];
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        float fraction = x - static_cast<float>(idx1);
        if constexpr (BSPLINE)
            return cubicBSpline1D(values[0], values[1], values[2], values[3], fraction);
        else
            return cubic1D(values[0], values[1], values[2], values[3], fraction);
    }

    template<typename T>
    template<InterpMode INTERP, BorderMode BORDER>
    T Interpolator1D<T>::get(float x) const {
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(m_data, x);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(m_data, x);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(m_data, x);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(m_data, x);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(m_data, x);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    template<typename T>
    template<InterpMode INTERP, BorderMode BORDER>
    T Interpolator1D<T>::get(float x, size_t offset) const {
        const T* data = m_data + offset;
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(data, x);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(data, x);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(data, x);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(data, x);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(data, x);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    // -- 2D -- //

    template<typename T>
    Interpolator2D<T>::Interpolator2D(const T* input, size2_t stride, size2_t shape, T value) noexcept
            : m_data(input), m_stride(stride), m_shape(shape), m_value(value) {}

    template<typename T>
    template<BorderMode BORDER>
    T Interpolator2D<T>::nearest_(const T* data, float y, float x) const {
        T out;
        int2_t idx(noa::math::round(y), noa::math::round(x));
        if constexpr (BORDER == BORDER_ZERO) {
            if (idx[1] < 0 || idx[1] >= m_shape[1] || idx[0] < 0 || idx[0] >= m_shape[0])
                out = static_cast<T>(0);
            else
                out = data[at(idx, m_stride)];
        } else if constexpr (BORDER == BORDER_VALUE) {
            if (idx[1] < 0 || idx[1] >= m_shape[1] || idx[0] < 0 || idx[0] >= m_shape[0])
                out = static_cast<T>(m_value);
            else
                out = data[at(idx, m_stride)];
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            idx[0] = getBorderIndex<BORDER>(idx[0], m_shape[0]);
            idx[1] = getBorderIndex<BORDER>(idx[1], m_shape[1]);
            out = data[at(idx, m_stride)];
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return out;
    }

    template<typename T>
    template<BorderMode BORDER, bool COSINE>
    T Interpolator2D<T>::linear_(const T* data, float y, float x) const {
        int2_t idx0(noa::math::floor(y), noa::math::floor(x));
        int2_t idx1(idx0 + 1);
        T values[4]; // v00, v10, v01, v11
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond_y[2] = {idx0[0] >= 0 && idx0[0] < m_shape[0], idx1[0] >= 0 && idx1[0] < m_shape[0]};
            bool cond_x[2] = {idx0[1] >= 0 && idx0[1] < m_shape[1], idx1[1] >= 0 && idx1[1] < m_shape[1]};
            if constexpr (BORDER == BORDER_ZERO) {
                constexpr T ZERO = static_cast<T>(0);
                values[0] = cond_y[0] && cond_x[0] ? data[idx0[0] * m_stride[0] + idx0[1] * m_stride[1]] : ZERO; // v00
                values[1] = cond_y[0] && cond_x[1] ? data[idx0[0] * m_stride[0] + idx1[1] * m_stride[1]] : ZERO; // v01
                values[2] = cond_y[1] && cond_x[0] ? data[idx1[0] * m_stride[0] + idx0[1] * m_stride[1]] : ZERO; // v10
                values[3] = cond_y[1] && cond_x[1] ? data[idx1[0] * m_stride[0] + idx1[1] * m_stride[1]] : ZERO; // v11
            } else {
                values[0] = cond_y[0] && cond_x[0] ? data[idx0[0] * m_stride[0] + idx0[1] * m_stride[1]] : m_value;
                values[1] = cond_y[0] && cond_x[1] ? data[idx0[0] * m_stride[0] + idx1[1] * m_stride[1]] : m_value;
                values[2] = cond_y[1] && cond_x[0] ? data[idx1[0] * m_stride[0] + idx0[1] * m_stride[1]] : m_value;
                values[3] = cond_y[1] && cond_x[1] ? data[idx1[0] * m_stride[0] + idx1[1] * m_stride[1]] : m_value;
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            int tmp[4] = {getBorderIndex<BORDER>(idx0[1], m_shape[1]),
                          getBorderIndex<BORDER>(idx1[1], m_shape[1]),
                          getBorderIndex<BORDER>(idx0[0], m_shape[0]),
                          getBorderIndex<BORDER>(idx1[0], m_shape[0])};
            values[0] = data[tmp[2] * m_stride[0] + tmp[0] * m_stride[1]]; // v00
            values[1] = data[tmp[2] * m_stride[0] + tmp[1] * m_stride[1]]; // v01
            values[2] = data[tmp[3] * m_stride[0] + tmp[0] * m_stride[1]]; // v10
            values[3] = data[tmp[3] * m_stride[0] + tmp[1] * m_stride[1]]; // v11
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        float2_t fraction(x - static_cast<float>(idx0[1]), y - static_cast<float>(idx0[0]));
        if constexpr (COSINE)
            return cosine2D(values[0], values[1], values[2], values[3], fraction[0], fraction[1]);
        else
            return linear2D(values[0], values[1], values[2], values[3], fraction[0], fraction[1]);
    }

    template<typename T>
    template<BorderMode BORDER, bool BSPLINE>
    T Interpolator2D<T>::cubic_(const T* data, float y, float x) const {
        int2_t idx(noa::math::floor(y), noa::math::floor(x));
        T square[4][4]; // [y][x]
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond_y[4] = {idx[0] - 1 >= 0 && idx[0] - 1 < m_shape[0],
                              idx[0] + 0 >= 0 && idx[0] + 0 < m_shape[0],
                              idx[0] + 1 >= 0 && idx[0] + 1 < m_shape[0],
                              idx[0] + 2 >= 0 && idx[0] + 2 < m_shape[0]};
            bool cond_x[4] = {idx[1] - 1 >= 0 && idx[1] - 1 < m_shape[1],
                              idx[1] + 0 >= 0 && idx[1] + 0 < m_shape[1],
                              idx[1] + 1 >= 0 && idx[1] + 1 < m_shape[1],
                              idx[1] + 2 >= 0 && idx[1] + 2 < m_shape[1]};
            constexpr int offset[4] = {-1, 0, 1, 2};
            for (int j = 0; j < 4; ++j) {
                int off_y = (idx[0] + offset[j]) * m_stride[0];
                for (int i = 0; i < 4; ++i) {
                    int off_x = (idx[1] + offset[i]) * m_stride[1];
                    if constexpr (BORDER == BORDER_ZERO)
                        square[j][i] = cond_x[i] && cond_y[j] ? data[off_y + off_x] : static_cast<T>(0);
                    else
                        square[j][i] = cond_x[i] && cond_y[j] ? data[off_y + off_x] : m_value;
                }
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            int tmp_y[4] = {getBorderIndex<BORDER>(idx[0] - 1, m_shape[0]),
                            getBorderIndex<BORDER>(idx[0] + 0, m_shape[0]),
                            getBorderIndex<BORDER>(idx[0] + 1, m_shape[0]),
                            getBorderIndex<BORDER>(idx[0] + 2, m_shape[0])};
            int tmp_x[4] = {getBorderIndex<BORDER>(idx[1] - 1, m_shape[1]),
                            getBorderIndex<BORDER>(idx[1] + 0, m_shape[1]),
                            getBorderIndex<BORDER>(idx[1] + 1, m_shape[1]),
                            getBorderIndex<BORDER>(idx[1] + 2, m_shape[1])};
            for (int j = 0; j < 4; ++j) {
                int offset = tmp_y[j] * m_stride[0];
                for (int i = 0; i < 4; ++i) {
                    square[j][i] = data[offset + tmp_x[i] * m_stride[1]];
                }
            }
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        float2_t fraction(x - static_cast<float>(idx[1]), y - static_cast<float>(idx[0]));
        if constexpr (BSPLINE)
            return cubicBSpline2D(square, fraction[0], fraction[1]);
        else
            return cubic2D(square, fraction[0], fraction[1]);
    }

    template<typename T>
    template<InterpMode INTERP, BorderMode BORDER>
    T Interpolator2D<T>::get(float2_t coords) const {
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(m_data, coords[0], coords[1]);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(m_data, coords[0], coords[1]);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(m_data, coords[0], coords[1]);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(m_data, coords[0], coords[1]);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(m_data, coords[0], coords[1]);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    template<typename T>
    template<InterpMode INTERP, BorderMode BORDER>
    T Interpolator2D<T>::get(float2_t coords, size_t offset) const {
        const T* data = m_data + offset;
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(data, coords[0], coords[1]);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(data, coords[0], coords[1]);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(data, coords[0], coords[1]);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(data, coords[0], coords[1]);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(data, coords[0], coords[1]);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    // -- 3D -- //

    template<typename T>
    Interpolator3D<T>::Interpolator3D(const T* input, size3_t stride, size3_t shape, T value) noexcept
            : m_data(input), m_stride(stride), m_shape(shape), m_value(value) {}

    template<typename T>
    template<BorderMode BORDER>
    T Interpolator3D<T>::nearest_(const T* data, float z, float y, float x) const {
        T out;
        int3_t idx(noa::math::round(z), noa::math::round(y), noa::math::round(x));
        if constexpr (BORDER == BORDER_ZERO) {
            if (idx[2] < 0 || idx[2] >= m_shape[2] ||
                idx[1] < 0 || idx[1] >= m_shape[1] ||
                idx[0] < 0 || idx[0] >= m_shape[0])
                out = static_cast<T>(0);
            else
                out = data[at(idx, m_stride)];
        } else if constexpr (BORDER == BORDER_VALUE) {
            if (idx[2] < 0 || idx[2] >= m_shape[2] ||
                idx[1] < 0 || idx[1] >= m_shape[1] ||
                idx[0] < 0 || idx[0] >= m_shape[0])
                out = static_cast<T>(m_value);
            else
                out = data[at(idx, m_stride)];
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            idx[2] = getBorderIndex<BORDER>(idx[2], m_shape[2]);
            idx[1] = getBorderIndex<BORDER>(idx[1], m_shape[1]);
            idx[0] = getBorderIndex<BORDER>(idx[0], m_shape[0]);
            out = data[at(idx, m_stride)];
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return out;
    }

    template<typename T>
    template<BorderMode BORDER, bool COSINE>
    T Interpolator3D<T>::linear_(const T* data, float z, float y, float x) const {
        int3_t idx[2];
        idx[0] = int3_t(noa::math::floor(z), noa::math::floor(y), noa::math::floor(x));
        idx[1] = idx[0] + 1;

        T values[8];
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond_z[2] = {idx[0][0] >= 0 && idx[0][0] < m_shape[0], idx[1][0] >= 0 && idx[1][0] < m_shape[0]};
            bool cond_y[2] = {idx[0][1] >= 0 && idx[0][1] < m_shape[1], idx[1][1] >= 0 && idx[1][1] < m_shape[1]};
            bool cond_x[2] = {idx[0][2] >= 0 && idx[0][2] < m_shape[2], idx[1][2] >= 0 && idx[1][2] < m_shape[2]};

            T cval;
            if constexpr (BORDER == BORDER_ZERO)
                cval = static_cast<T>(0);
            else
                cval = m_value;
            int off_z[2] = {idx[0][0] * m_stride[0], idx[1][0] * m_stride[0]};
            int off_y[2] = {idx[0][1] * m_stride[1], idx[1][1] * m_stride[1]};
            int off_x[2] = {idx[0][2] * m_stride[2], idx[1][2] * m_stride[2]};
            values[0] = cond_z[0] && cond_y[0] && cond_x[0] ? data[off_z[0] + off_y[0] + off_x[0]] : cval; // v000
            values[1] = cond_z[0] && cond_y[0] && cond_x[1] ? data[off_z[0] + off_y[0] + off_x[1]] : cval; // v001
            values[2] = cond_z[0] && cond_y[1] && cond_x[0] ? data[off_z[0] + off_y[1] + off_x[0]] : cval; // v010
            values[3] = cond_z[0] && cond_y[1] && cond_x[1] ? data[off_z[0] + off_y[1] + off_x[1]] : cval; // v011
            values[4] = cond_z[1] && cond_y[0] && cond_x[0] ? data[off_z[1] + off_y[0] + off_x[0]] : cval; // v100
            values[5] = cond_z[1] && cond_y[0] && cond_x[1] ? data[off_z[1] + off_y[0] + off_x[1]] : cval; // v101
            values[6] = cond_z[1] && cond_y[1] && cond_x[0] ? data[off_z[1] + off_y[1] + off_x[0]] : cval; // v110
            values[7] = cond_z[1] && cond_y[1] && cond_x[1] ? data[off_z[1] + off_y[1] + off_x[1]] : cval; // v111

        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            int tmp[6] = {getBorderIndex<BORDER>(idx[0][2], m_shape[2]),
                          getBorderIndex<BORDER>(idx[1][2], m_shape[2]),
                          getBorderIndex<BORDER>(idx[0][1], m_shape[1]),
                          getBorderIndex<BORDER>(idx[1][1], m_shape[1]),
                          getBorderIndex<BORDER>(idx[0][0], m_shape[0]),
                          getBorderIndex<BORDER>(idx[1][0], m_shape[0])};
            values[0] = data[tmp[4] * m_stride[0] + tmp[2] * m_stride[1] + tmp[0] * m_stride[2]]; // v000
            values[1] = data[tmp[4] * m_stride[0] + tmp[2] * m_stride[1] + tmp[1] * m_stride[2]]; // v001
            values[2] = data[tmp[4] * m_stride[0] + tmp[3] * m_stride[1] + tmp[0] * m_stride[2]]; // v010
            values[3] = data[tmp[4] * m_stride[0] + tmp[3] * m_stride[1] + tmp[1] * m_stride[2]]; // v011
            values[4] = data[tmp[5] * m_stride[0] + tmp[2] * m_stride[1] + tmp[0] * m_stride[2]]; // v100
            values[5] = data[tmp[5] * m_stride[0] + tmp[2] * m_stride[1] + tmp[1] * m_stride[2]]; // v101
            values[6] = data[tmp[5] * m_stride[0] + tmp[3] * m_stride[1] + tmp[0] * m_stride[2]]; // v110
            values[7] = data[tmp[5] * m_stride[0] + tmp[3] * m_stride[1] + tmp[1] * m_stride[2]]; // v111
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        float3_t fraction(x - static_cast<float>(idx[0][2]),
                          y - static_cast<float>(idx[0][1]),
                          z - static_cast<float>(idx[0][0]));
        if constexpr (COSINE)
            return cosine3D(values[0], values[1], values[2], values[3],
                            values[4], values[5], values[6], values[7],
                            fraction[0], fraction[1], fraction[2]);
        else
            return linear3D(values[0], values[1], values[2], values[3],
                            values[4], values[5], values[6], values[7],
                            fraction[0], fraction[1], fraction[2]);
    }

    template<typename T>
    template<BorderMode BORDER, bool BSPLINE>
    T Interpolator3D<T>::cubic_(const T* data, float z, float y, float x) const {
        int3_t idx(noa::math::floor(z), noa::math::floor(y), noa::math::floor(x));
        T values[4][4][4]; // [z][y][x]
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond_z[4] = {idx[0] - 1 >= 0 && idx[0] - 1 < m_shape[0],
                              idx[0] + 0 >= 0 && idx[0] + 0 < m_shape[0],
                              idx[0] + 1 >= 0 && idx[0] + 1 < m_shape[0],
                              idx[0] + 2 >= 0 && idx[0] + 2 < m_shape[0]};
            bool cond_y[4] = {idx[1] - 1 >= 0 && idx[1] - 1 < m_shape[1],
                              idx[1] + 0 >= 0 && idx[1] + 0 < m_shape[1],
                              idx[1] + 1 >= 0 && idx[1] + 1 < m_shape[1],
                              idx[1] + 2 >= 0 && idx[1] + 2 < m_shape[1]};
            bool cond_x[4] = {idx[2] - 1 >= 0 && idx[2] - 1 < m_shape[2],
                              idx[2] + 0 >= 0 && idx[2] + 0 < m_shape[2],
                              idx[2] + 1 >= 0 && idx[2] + 1 < m_shape[2],
                              idx[2] + 2 >= 0 && idx[2] + 2 < m_shape[2]};
            T cval;
            if constexpr (BORDER == BORDER_ZERO)
                cval = static_cast<T>(0);
            else
                cval = m_value;
            constexpr int offset[4] = {-1, 0, 1, 2};
            for (int i = 0; i < 4; ++i) {
                int off_z = (idx[0] + offset[i]) * m_stride[0];
                for (int j = 0; j < 4; ++j) {
                    int off = off_z + (idx[1] + offset[j]) * m_stride[1];
                    for (int k = 0; k < 4; ++k) {
                        values[i][j][k] = cond_z[i] && cond_y[j] && cond_x[k] ?
                                          data[off + (idx[2] + offset[k]) * m_stride[2]] : cval;
                    }
                }
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            int tmp_z[4] = {getBorderIndex<BORDER>(idx[0] - 1, m_shape[0]),
                            getBorderIndex<BORDER>(idx[0] + 0, m_shape[0]),
                            getBorderIndex<BORDER>(idx[0] + 1, m_shape[0]),
                            getBorderIndex<BORDER>(idx[0] + 2, m_shape[0])};
            int tmp_y[4] = {getBorderIndex<BORDER>(idx[1] - 1, m_shape[1]),
                            getBorderIndex<BORDER>(idx[1] + 0, m_shape[1]),
                            getBorderIndex<BORDER>(idx[1] + 1, m_shape[1]),
                            getBorderIndex<BORDER>(idx[1] + 2, m_shape[1])};
            int tmp_x[4] = {getBorderIndex<BORDER>(idx[2] - 1, m_shape[2]),
                            getBorderIndex<BORDER>(idx[2] + 0, m_shape[2]),
                            getBorderIndex<BORDER>(idx[2] + 1, m_shape[2]),
                            getBorderIndex<BORDER>(idx[2] + 2, m_shape[2])};
            for (int i = 0; i < 4; ++i) {
                int off_z = tmp_z[i] * m_stride[0];
                for (int j = 0; j < 4; ++j) {
                    int offset = off_z + tmp_y[j] * m_stride[1];
                    for (int k = 0; k < 4; ++k) {
                        values[i][j][k] = data[offset + tmp_x[k] * m_stride[2]];
                    }
                }
            }
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        float3_t fraction(x - static_cast<float>(idx[2]),
                          y - static_cast<float>(idx[1]),
                          z - static_cast<float>(idx[0]));
        if constexpr (BSPLINE)
            return cubicBSpline3D(values, fraction[0], fraction[1], fraction[2]);
        else
            return cubic3D(values, fraction[0], fraction[1], fraction[2]);
    }

    template<typename T>
    template<InterpMode INTERP, BorderMode BORDER>
    T Interpolator3D<T>::get(float3_t coords) const {
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(m_data, coords[0], coords[1], coords[2]);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(m_data, coords[0], coords[1], coords[2]);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(m_data, coords[0], coords[1], coords[2]);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(m_data, coords[0], coords[1], coords[2]);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(m_data, coords[0], coords[1], coords[2]);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    template<typename T>
    template<InterpMode INTERP, BorderMode BORDER>
    T Interpolator3D<T>::get(float3_t coords, size_t offset) const {
        const T* data = m_data + offset;
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(data, coords[0], coords[1], coords[2]);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(data, coords[0], coords[1], coords[2]);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(data, coords[0], coords[1], coords[2]);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(data, coords[0], coords[1], coords[2]);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(data, coords[0], coords[1], coords[2]);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }
}
