/// \file noa/cpu/transform/Interpolator.h
/// \brief 1D, 2D and 3D interpolators.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Math.h"
#include "noa/cpu/transform/Interpolate.h"

namespace noa::cpu::transform {
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
    ///       corresponds to the radio/weight used by the interpolation function (e.g. linear1D()). In other words,
    ///       the coordinate system locates the data between -0.5 and N-1 + 0.5.
    template<typename T>
    class Interpolator1D {
    public:
        /// Empty state. Use reset() to set data points.
        Interpolator1D() = default;

        /// Sets the data points.
        /// \param[in] input    On the \b host. Input contiguous 1D array.
        /// \param size         Size, in elements, of \p input.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST Interpolator1D(const T* input, size_t size, T value = static_cast<T>(0)) noexcept;

        /// Resets the data points.
        /// \param[in] input    On the \b host. Input contiguous 1D array.
        /// \param size         Size, in elements, of \p input.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST void reset(const T* input, size_t size, T value = static_cast<T>(0)) noexcept;

        /// Returns the interpolated value at the coordinate \p x.
        /// \tparam BORDER      Border/addressing mode.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \param x            Coordinate to interpolate at.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float x);

    private:
        const T* m_data{};
        int m_size{};
        T m_value{};
    private:
        template<BorderMode BORDER> NOA_HOST T nearest_(float x);
        template<BorderMode BORDER, bool COSINE> NOA_HOST T linear_(float x);
        template<BorderMode BORDER, bool BSPLINE> NOA_HOST T cubic_(float x);
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
        /// \param shape        Logical {fast, medium} shape, in elements, of \p input.
        /// \param pitch        Pitch, in elements, of \p shape.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST Interpolator2D(const T* input, size2_t shape, size_t pitch, T value = static_cast<T>(0)) noexcept;

        /// Resets the data points.
        /// \param[in] input    On the \b host. Input 2D array.
        /// \param shape        Logical {fast, medium} shape, in elements, of \p input.
        /// \param pitch        Pitch, in elements, of \p shape.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST void reset(const T* input, size2_t shape, size_t pitch, T value = static_cast<T>(0)) noexcept;

        /// Returns the interpolated value at the coordinate \p x, \p y.
        /// \tparam BORDER      Border/addressing mode.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \param x            Coordinate in the first dimension.
        /// \param y            Coordinate in the second dimension.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float x, float y);

    private:
        const T* m_data{};
        int2_t m_shape{};
        int m_pitch{};
        T m_value{};
    private:
        template<BorderMode BORDER> NOA_HOST T nearest_(float x, float y);
        template<BorderMode BORDER, bool COSINE> NOA_HOST T linear_(float x, float y);
        template<BorderMode BORDER, bool BSPLINE> NOA_HOST T cubic_(float x, float y);
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
        /// \param shape        Logical {fast, medium, slow} shape, in elements, of \p input.
        /// \param pitch        Pitch, in elements, of \p shape.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST Interpolator3D(const T* input, size3_t shape, size_t pitch, T value = static_cast<T>(0)) noexcept;

        /// Resets the data points.
        /// \param[in] input    On the \b host. Input 3D array.
        /// \param shape        Logical {fast, medium, slow} shape, in elements, of \p input.
        /// \param pitch        Pitch, in elements, of \p shape.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST void reset(const T* input, size3_t shape, size_t pitch, T value = static_cast<T>(0)) noexcept;

        /// Returns the interpolated value at the coordinate \p x, \p y, \p z.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \tparam BORDER      Border/addressing mode.
        /// \param x            Coordinate in the first dimension.
        /// \param y            Coordinate in the second dimension.
        /// \param z            Coordinate in the third dimension.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float x, float y, float z);
    private:
        const T* m_data{};
        int3_t m_shape{};
        int m_pitch{};
        int m_page{};
        T m_value{};
    private:
        template<BorderMode BORDER> NOA_HOST T nearest_(float x, float y, float z);
        template<BorderMode BORDER, bool COSINE> NOA_HOST T linear_(float x, float y, float z);
        template<BorderMode BORDER, bool BSPLINE> NOA_HOST T cubic_(float x, float y, float z);
    };
}

// Implementation:
namespace noa::cpu::transform {
    template<typename T>
    Interpolator1D<T>::Interpolator1D(const T* input, size_t size, T value) noexcept
            : m_data(input), m_size(static_cast<int>(size)), m_value(value) {}

    template<typename T>
    void Interpolator1D<T>::reset(const T* input, size_t size, T value) noexcept {
        m_data = input;
        m_size = static_cast<int>(size);
        m_value = value;
    }

    template<typename T>
    template<BorderMode BORDER>
    T Interpolator1D<T>::nearest_(float x) {
        T out;
        auto idx = static_cast<int>(noa::math::round(x));
        if constexpr (BORDER == BORDER_ZERO) {
            out = idx >= 0 && idx < m_size ? m_data[idx] : static_cast<T>(0);
        } else if constexpr (BORDER == BORDER_VALUE) {
            out = idx >= 0 && idx < m_size ? m_data[idx] : static_cast<T>(m_value);
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            out = m_data[getBorderIndex<BORDER>(idx, m_size)];
        }
        return out;
    }

    template<typename T>
    template<BorderMode BORDER, bool COSINE>
    T Interpolator1D<T>::linear_(float x) {
        int idx0 = static_cast<int>(noa::math::floor(x));
        int idx1 = idx0 + 1;
        T values[2];
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond[2] = {idx0 >= 0 && idx0 < m_size, idx1 >= 0 && idx1 < m_size};
            if constexpr (BORDER == BORDER_ZERO) {
                values[0] = cond[0] ? m_data[idx0] : static_cast<T>(0);
                values[1] = cond[1] ? m_data[idx1] : static_cast<T>(0);
            } else {
                values[0] = cond[0] ? m_data[idx0] : static_cast<T>(m_value);
                values[1] = cond[1] ? m_data[idx1] : static_cast<T>(m_value);
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            values[0] = m_data[getBorderIndex<BORDER>(idx0, m_size)];
            values[1] = m_data[getBorderIndex<BORDER>(idx1, m_size)];
        }
        float fraction = x - static_cast<float>(idx0);
        if constexpr (COSINE)
            return cosine1D(values[0], values[1], fraction);
        else
            return linear1D(values[0], values[1], fraction);
    }

    template<typename T>
    template<BorderMode BORDER, bool BSPLINE>
    T Interpolator1D<T>::cubic_(float x) {
        int idx1 = static_cast<int>(noa::math::floor(x));
        int idx0 = idx1 - 1;
        int idx2 = idx1 + 1;
        int idx3 = idx1 + 2;
        T values[4];
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond[4] = {idx0 >= 0 && idx0 < m_size,
                            idx1 >= 0 && idx1 < m_size,
                            idx2 >= 0 && idx2 < m_size,
                            idx3 >= 0 && idx3 < m_size};
            if constexpr (BORDER == BORDER_ZERO) {
                values[0] = cond[0] ? m_data[idx0] : static_cast<T>(0);
                values[1] = cond[1] ? m_data[idx1] : static_cast<T>(0);
                values[2] = cond[2] ? m_data[idx2] : static_cast<T>(0);
                values[3] = cond[3] ? m_data[idx3] : static_cast<T>(0);
            } else {
                values[0] = cond[0] ? m_data[idx0] : static_cast<T>(m_value);
                values[1] = cond[1] ? m_data[idx1] : static_cast<T>(m_value);
                values[2] = cond[2] ? m_data[idx2] : static_cast<T>(m_value);
                values[3] = cond[3] ? m_data[idx3] : static_cast<T>(m_value);
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            values[0] = m_data[getBorderIndex<BORDER>(idx0, m_size)];
            values[1] = m_data[getBorderIndex<BORDER>(idx1, m_size)];
            values[2] = m_data[getBorderIndex<BORDER>(idx2, m_size)];
            values[3] = m_data[getBorderIndex<BORDER>(idx3, m_size)];
        }
        float fraction = x - static_cast<float>(idx1);
        if constexpr (BSPLINE)
            return cubicBSpline1D(values[0], values[1], values[2], values[3], fraction);
        else
            return cubic1D(values[0], values[1], values[2], values[3], fraction);
    }

    template<typename T>
    template<InterpMode INTERP, BorderMode BORDER>
    T Interpolator1D<T>::get(float x) {
        static_assert(BORDER == BORDER_ZERO || BORDER == BORDER_VALUE || BORDER == BORDER_CLAMP ||
                      BORDER == BORDER_PERIODIC || BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT);
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(x);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(x);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(x);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(x);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(x);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    template<typename T>
    Interpolator2D<T>::Interpolator2D(const T* input, size2_t shape, size_t pitch, T value) noexcept
            : m_data(input), m_shape(shape), m_pitch(static_cast<int>(pitch)), m_value(value) {}

    template<typename T>
    void Interpolator2D<T>::reset(const T* input, size2_t shape, size_t pitch, T value) noexcept {
        m_data = input;
        m_shape = shape;
        m_pitch = static_cast<int>(pitch);
        m_value = value;
    }

    template<typename T>
    template<BorderMode BORDER>
    T Interpolator2D<T>::nearest_(float x, float y) {
        T out;
        int2_t idx(noa::math::round(x), noa::math::round(y));
        if constexpr (BORDER == BORDER_ZERO) {
            if (idx.x < 0 || idx.x >= m_shape.x || idx.y < 0 || idx.y >= m_shape.y)
                out = static_cast<T>(0);
            else
                out = m_data[idx.y * m_pitch + idx.x];
        } else if constexpr (BORDER == BORDER_VALUE) {
            if (idx.x < 0 || idx.x >= m_shape.x || idx.y < 0 || idx.y >= m_shape.y)
                out = static_cast<T>(m_value);
            else
                out = m_data[idx.y * m_pitch + idx.x];
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            idx.x = getBorderIndex<BORDER>(idx.x, m_shape.x);
            idx.y = getBorderIndex<BORDER>(idx.y, m_shape.y);
            out = m_data[idx.y * m_pitch + idx.x];
        }
        return out;
    }

    template<typename T>
    template<BorderMode BORDER, bool COSINE>
    T Interpolator2D<T>::linear_(float x, float y) {
        int2_t idx0(noa::math::floor(x), noa::math::floor(y));
        int2_t idx1(idx0 + 1);
        T values[4]; // v00, v10, v01, v11
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond_x[2] = {idx0.x >= 0 && idx0.x < m_shape.x, idx1.x >= 0 && idx1.x < m_shape.x};
            bool cond_y[2] = {idx0.y >= 0 && idx0.y < m_shape.y, idx1.y >= 0 && idx1.y < m_shape.y};
            if constexpr (BORDER == BORDER_ZERO) {
                values[0] = cond_y[0] && cond_x[0] ? m_data[idx0.y * m_pitch + idx0.x] : static_cast<T>(0); // v00
                values[1] = cond_y[0] && cond_x[1] ? m_data[idx0.y * m_pitch + idx1.x] : static_cast<T>(0); // v01
                values[2] = cond_y[1] && cond_x[0] ? m_data[idx1.y * m_pitch + idx0.x] : static_cast<T>(0); // v10
                values[3] = cond_y[1] && cond_x[1] ? m_data[idx1.y * m_pitch + idx1.x] : static_cast<T>(0); // v11
            } else {
                values[0] = cond_y[0] && cond_x[0] ? m_data[idx0.y * m_pitch + idx0.x] : static_cast<T>(m_value);
                values[1] = cond_y[0] && cond_x[1] ? m_data[idx0.y * m_pitch + idx1.x] : static_cast<T>(m_value);
                values[2] = cond_y[1] && cond_x[0] ? m_data[idx1.y * m_pitch + idx0.x] : static_cast<T>(m_value);
                values[3] = cond_y[1] && cond_x[1] ? m_data[idx1.y * m_pitch + idx1.x] : static_cast<T>(m_value);
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            int tmp[4] = {getBorderIndex<BORDER>(idx0.x, m_shape.x),
                          getBorderIndex<BORDER>(idx1.x, m_shape.x),
                          getBorderIndex<BORDER>(idx0.y, m_shape.y),
                          getBorderIndex<BORDER>(idx1.y, m_shape.y)};
            values[0] = m_data[tmp[2] * m_pitch + tmp[0]]; // v00
            values[1] = m_data[tmp[2] * m_pitch + tmp[1]]; // v01
            values[2] = m_data[tmp[3] * m_pitch + tmp[0]]; // v10
            values[3] = m_data[tmp[3] * m_pitch + tmp[1]]; // v11
        }
        float2_t fraction(x - static_cast<float>(idx0.x), y - static_cast<float>(idx0.y));
        if constexpr (COSINE)
            return cosine2D(values[0], values[1], values[2], values[3], fraction.x, fraction.y);
        else
            return linear2D(values[0], values[1], values[2], values[3], fraction.x, fraction.y);
    }

    template<typename T>
    template<BorderMode BORDER, bool BSPLINE>
    T Interpolator2D<T>::cubic_(float x, float y) {
        int2_t idx(noa::math::floor(x), noa::math::floor(y));
        T square[4][4]; // [y][x]
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond_x[4] = {idx.x - 1 >= 0 && idx.x - 1 < m_shape.x,
                              idx.x + 0 >= 0 && idx.x + 0 < m_shape.x,
                              idx.x + 1 >= 0 && idx.x + 1 < m_shape.x,
                              idx.x + 2 >= 0 && idx.x + 2 < m_shape.x};
            bool cond_y[4] = {idx.y - 1 >= 0 && idx.y - 1 < m_shape.y,
                              idx.y + 0 >= 0 && idx.y + 0 < m_shape.y,
                              idx.y + 1 >= 0 && idx.y + 1 < m_shape.y,
                              idx.y + 2 >= 0 && idx.y + 2 < m_shape.y};
            constexpr int offset[4] = {-1, 0, 1, 2};
            for (int j = 0; j < 4; j++) {
                int off_y = (idx.y + offset[j]) * m_pitch;
                for (int i = 0; i < 4; i++) {
                    if constexpr (BORDER == BORDER_ZERO)
                        square[j][i] = cond_x[i] && cond_y[j] ?
                                       m_data[off_y + idx.x + offset[i]] : static_cast<T>(0);
                    else
                        square[j][i] = cond_x[i] && cond_y[j] ?
                                       m_data[off_y + idx.x + offset[i]] : static_cast<T>(m_value);
                }
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            int tmp_x[4] = {getBorderIndex<BORDER>(idx.x - 1, m_shape.x),
                            getBorderIndex<BORDER>(idx.x + 0, m_shape.x),
                            getBorderIndex<BORDER>(idx.x + 1, m_shape.x),
                            getBorderIndex<BORDER>(idx.x + 2, m_shape.x)};
            int tmp_y[4] = {getBorderIndex<BORDER>(idx.y - 1, m_shape.y),
                            getBorderIndex<BORDER>(idx.y + 0, m_shape.y),
                            getBorderIndex<BORDER>(idx.y + 1, m_shape.y),
                            getBorderIndex<BORDER>(idx.y + 2, m_shape.y)};
            for (int j = 0; j < 4; ++j) {
                int offset = tmp_y[j] * m_pitch;
                for (int i = 0; i < 4; ++i) {
                    square[j][i] = m_data[offset + tmp_x[i]];
                }
            }
        }
        float2_t fraction(x - static_cast<float>(idx.x), y - static_cast<float>(idx.y));
        if constexpr (BSPLINE)
            return cubicBSpline2D(square, fraction.x, fraction.y);
        else
            return cubic2D(square, fraction.x, fraction.y);
    }

    template<typename T>
    template<InterpMode INTERP, BorderMode BORDER>
    T Interpolator2D<T>::get(float x, float y) {
        static_assert(BORDER == BORDER_ZERO || BORDER == BORDER_VALUE || BORDER == BORDER_CLAMP ||
                      BORDER == BORDER_PERIODIC || BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT);
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(x, y);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(x, y);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(x, y);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(x, y);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(x, y);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    template<typename T>
    Interpolator3D<T>::Interpolator3D(const T* input, size3_t shape, size_t pitch, T value) noexcept
            : m_data(input), m_shape(shape), m_pitch(static_cast<int>(pitch)),
              m_page(static_cast<int>(pitch * shape.y)), m_value(value) {}

    template<typename T>
    void Interpolator3D<T>::reset(const T* input, size3_t shape, size_t pitch, T value) noexcept {
        m_data = input;
        m_shape = shape;
        m_pitch = static_cast<int>(pitch);
        m_page = static_cast<int>(pitch * shape.y);
        m_value = value;
    }

    template<typename T>
    template<BorderMode BORDER>
    T Interpolator3D<T>::nearest_(float x, float y, float z) {
        T out;
        int3_t idx(noa::math::round(x), noa::math::round(y), noa::math::round(z));
        if constexpr (BORDER == BORDER_ZERO) {
            if (idx.x < 0 || idx.x >= m_shape.x ||
                idx.y < 0 || idx.y >= m_shape.y ||
                idx.z < 0 || idx.z >= m_shape.z)
                out = static_cast<T>(0);
            else
                out = m_data[idx.z * m_page + idx.y * m_pitch + idx.x];
        } else if constexpr (BORDER == BORDER_VALUE) {
            if (idx.x < 0 || idx.x >= m_shape.x ||
                idx.y < 0 || idx.y >= m_shape.y ||
                idx.z < 0 || idx.z >= m_shape.z)
                out = static_cast<T>(m_value);
            else
                out = m_data[idx.z * m_page + idx.y * m_pitch + idx.x];
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            idx.x = getBorderIndex<BORDER>(idx.x, m_shape.x);
            idx.y = getBorderIndex<BORDER>(idx.y, m_shape.y);
            idx.z = getBorderIndex<BORDER>(idx.z, m_shape.z);
            out = m_data[idx.z * m_page + idx.y * m_pitch + idx.x];
        }
        return out;
    }

    template<typename T>
    template<BorderMode BORDER, bool COSINE>
    T Interpolator3D<T>::linear_(float x, float y, float z) {
        int3_t idx[2];
        idx[0] = int3_t(noa::math::floor(x), noa::math::floor(y), noa::math::floor(z));
        idx[1] = idx[0] + 1;

        T values[8];
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond_x[2] = {idx[0].x >= 0 && idx[0].x < m_shape.x, idx[1].x >= 0 && idx[1].x < m_shape.x};
            bool cond_y[2] = {idx[0].y >= 0 && idx[0].y < m_shape.y, idx[1].y >= 0 && idx[1].y < m_shape.y};
            bool cond_z[2] = {idx[0].z >= 0 && idx[0].z < m_shape.z, idx[1].z >= 0 && idx[1].z < m_shape.z};

            T cval;
            if constexpr (BORDER == BORDER_ZERO)
                cval = static_cast<T>(0);
            else
                cval = static_cast<T>(m_value);
            int off_y[2] = {idx[0].y * m_pitch, idx[1].y * m_pitch};
            int off_z[2] = {idx[0].z * m_page, idx[1].z * m_page};
            values[0] = cond_z[0] && cond_y[0] && cond_x[0] ? m_data[off_z[0] + off_y[0] + idx[0].x] : cval; // v000
            values[1] = cond_z[0] && cond_y[0] && cond_x[1] ? m_data[off_z[0] + off_y[0] + idx[1].x] : cval; // v001
            values[2] = cond_z[0] && cond_y[1] && cond_x[0] ? m_data[off_z[0] + off_y[1] + idx[0].x] : cval; // v010
            values[3] = cond_z[0] && cond_y[1] && cond_x[1] ? m_data[off_z[0] + off_y[1] + idx[1].x] : cval; // v011
            values[4] = cond_z[1] && cond_y[0] && cond_x[0] ? m_data[off_z[1] + off_y[0] + idx[0].x] : cval; // v100
            values[5] = cond_z[1] && cond_y[0] && cond_x[1] ? m_data[off_z[1] + off_y[0] + idx[1].x] : cval; // v101
            values[6] = cond_z[1] && cond_y[1] && cond_x[0] ? m_data[off_z[1] + off_y[1] + idx[0].x] : cval; // v110
            values[7] = cond_z[1] && cond_y[1] && cond_x[1] ? m_data[off_z[1] + off_y[1] + idx[1].x] : cval; // v111

        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            int tmp[6] = {getBorderIndex<BORDER>(idx[0].x, m_shape.x),
                          getBorderIndex<BORDER>(idx[1].x, m_shape.x),
                          getBorderIndex<BORDER>(idx[0].y, m_shape.y),
                          getBorderIndex<BORDER>(idx[1].y, m_shape.y),
                          getBorderIndex<BORDER>(idx[0].z, m_shape.z),
                          getBorderIndex<BORDER>(idx[1].z, m_shape.z)};
            values[0] = m_data[tmp[4] * m_page + tmp[2] * m_pitch + tmp[0]]; // v000
            values[1] = m_data[tmp[4] * m_page + tmp[2] * m_pitch + tmp[1]]; // v001
            values[2] = m_data[tmp[4] * m_page + tmp[3] * m_pitch + tmp[0]]; // v010
            values[3] = m_data[tmp[4] * m_page + tmp[3] * m_pitch + tmp[1]]; // v011
            values[4] = m_data[tmp[5] * m_page + tmp[2] * m_pitch + tmp[0]]; // v100
            values[5] = m_data[tmp[5] * m_page + tmp[2] * m_pitch + tmp[1]]; // v101
            values[6] = m_data[tmp[5] * m_page + tmp[3] * m_pitch + tmp[0]]; // v110
            values[7] = m_data[tmp[5] * m_page + tmp[3] * m_pitch + tmp[1]]; // v111
        }
        float3_t fraction(x - static_cast<float>(idx[0].x),
                          y - static_cast<float>(idx[0].y),
                          z - static_cast<float>(idx[0].z));
        if constexpr (COSINE)
            return cosine3D(values[0], values[1], values[2], values[3],
                            values[4], values[5], values[6], values[7],
                            fraction.x, fraction.y, fraction.z);
        else
            return linear3D(values[0], values[1], values[2], values[3],
                            values[4], values[5], values[6], values[7],
                            fraction.x, fraction.y, fraction.z);
    }

    template<typename T>
    template<BorderMode BORDER, bool BSPLINE>
    T Interpolator3D<T>::cubic_(float x, float y, float z) {
        int3_t idx(noa::math::floor(x), noa::math::floor(y), noa::math::floor(z));
        T values[4][4][4]; // [z][y][x]
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond_x[4] = {idx.x - 1 >= 0 && idx.x - 1 < m_shape.x,
                              idx.x + 0 >= 0 && idx.x + 0 < m_shape.x,
                              idx.x + 1 >= 0 && idx.x + 1 < m_shape.x,
                              idx.x + 2 >= 0 && idx.x + 2 < m_shape.x};
            bool cond_y[4] = {idx.y - 1 >= 0 && idx.y - 1 < m_shape.y,
                              idx.y + 0 >= 0 && idx.y + 0 < m_shape.y,
                              idx.y + 1 >= 0 && idx.y + 1 < m_shape.y,
                              idx.y + 2 >= 0 && idx.y + 2 < m_shape.y};
            bool cond_z[4] = {idx.z - 1 >= 0 && idx.z - 1 < m_shape.z,
                              idx.z + 0 >= 0 && idx.z + 0 < m_shape.z,
                              idx.z + 1 >= 0 && idx.z + 1 < m_shape.z,
                              idx.z + 2 >= 0 && idx.z + 2 < m_shape.z};
            T cval;
            if constexpr (BORDER == BORDER_ZERO)
                cval = static_cast<T>(0);
            else
                cval = static_cast<T>(m_value);
            constexpr int offset[4] = {-1, 0, 1, 2};
            for (int k = 0; k < 4; ++k) {
                int off_z = (idx.z + offset[k]) * m_page;
                for (int j = 0; j < 4; ++j) {
                    int off = off_z + (idx.y + offset[j]) * m_pitch;
                    for (int i = 0; i < 4; ++i) {
                        values[k][j][i] = cond_z[k] && cond_y[j] && cond_x[i] ?
                                          m_data[off + idx.x + offset[i]] : cval;
                    }
                }
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            int tmp_x[4] = {getBorderIndex<BORDER>(idx.x - 1, m_shape.x),
                            getBorderIndex<BORDER>(idx.x + 0, m_shape.x),
                            getBorderIndex<BORDER>(idx.x + 1, m_shape.x),
                            getBorderIndex<BORDER>(idx.x + 2, m_shape.x)};
            int tmp_y[4] = {getBorderIndex<BORDER>(idx.y - 1, m_shape.y),
                            getBorderIndex<BORDER>(idx.y + 0, m_shape.y),
                            getBorderIndex<BORDER>(idx.y + 1, m_shape.y),
                            getBorderIndex<BORDER>(idx.y + 2, m_shape.y)};
            int tmp_z[4] = {getBorderIndex<BORDER>(idx.z - 1, m_shape.z),
                            getBorderIndex<BORDER>(idx.z + 0, m_shape.z),
                            getBorderIndex<BORDER>(idx.z + 1, m_shape.z),
                            getBorderIndex<BORDER>(idx.z + 2, m_shape.z)};
            for (int k = 0; k < 4; ++k) {
                int off_z = tmp_z[k] * m_page;
                for (int j = 0; j < 4; ++j) {
                    int offset = off_z + tmp_y[j] * m_pitch;
                    for (int i = 0; i < 4; ++i) {
                        values[k][j][i] = m_data[offset + tmp_x[i]];
                    }
                }
            }
        }
        float3_t fraction(x - static_cast<float>(idx.x),
                          y - static_cast<float>(idx.y),
                          z - static_cast<float>(idx.z));
        if constexpr (BSPLINE)
            return cubicBSpline3D(values, fraction.x, fraction.y, fraction.z);
        else
            return cubic3D(values, fraction.x, fraction.y, fraction.z);
    }

    template<typename T>
    template<InterpMode INTERP, BorderMode BORDER>
    T Interpolator3D<T>::get(float x, float y, float z) {
        static_assert(BORDER == BORDER_ZERO || BORDER == BORDER_VALUE || BORDER == BORDER_CLAMP ||
                      BORDER == BORDER_PERIODIC || BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT);
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(x, y, z);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(x, y, z);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(x, y, z);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(x, y, z);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(x, y, z);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }
}
