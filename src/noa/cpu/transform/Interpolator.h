/// \file noa/cpu/transform/Interpolator.h
/// \brief 1D, 2D and 3D interpolators.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Math.h"
#include "noa/cpu/transform/Interpolate.h"

// On of the main difference between these Interpolators and what we can find on other cryoEM packages,
// is that the interpolation window can be partially OOB, that is, elements that are OOB are replaced
// according to a BorderMode. cryoEM packages usually check that all elements are in bound and if there's
// even one element OOB, they don't interpolate.
// Note: These Interpolators are for real space interpolation or redundant and centered Fourier transforms.

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
    ///       corresponds to the ratio/weight used by the interpolation function (e.g. linear1D()). In other words,
    ///       the coordinate system locates the data between -0.5 and N-1 + 0.5.
    template<typename T>
    class Interpolator1D {
    public:
        /// Empty state. Use reset() to set data points.
        Interpolator1D() = default;

        /// Sets the data points.
        /// \param[in] inputs   On the \b host. Input contiguous 1D array.
        /// \param pitch        Pitch, in elements, of \p inputs.
        /// \param size         Size, in elements, of \p inputs.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST Interpolator1D(const T* input, size_t pitch, size_t size, T value) noexcept;

        /// Resets the data points.
        /// \param[in] inputs   On the \b host. Input contiguous 1D array.
        /// \param pitch        Pitch, in elements, of \p inputs.
        /// \param size         Size, in elements, of \p inputs.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST void reset(const T* input, size_t pitch, size_t size, T value) noexcept;

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
        /// \param batch        Batch to use.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float x, size_t batch) const;

    private:
        const T* m_data{};
        int m_pitch{};
        int m_size{};
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
        /// \param[in] inputs   On the \b host. Input 2D array.
        /// \param pitch        Pitch, in elements, of \p inputs.
        /// \param shape        Logical {fast, medium} shape, in elements, of \p input.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST Interpolator2D(const T* input, size2_t pitch, size2_t shape, T value) noexcept;

        /// Resets the data points.
        /// \param[in] inputs   On the \b host. Input 2D array.
        /// \param pitch        Pitch, in elements, of \p inputs.
        /// \param shape        Logical {fast, medium} shape, in elements, of \p input.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST void reset(const T* input, size2_t pitch, size2_t shape, T value) noexcept;

        /// Returns the interpolated value at the coordinate \p x, \p y.
        /// \tparam BORDER      Border/addressing mode.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \param coords       (x, y) coordinates.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float2_t coords) const;

        /// Returns the interpolated value at the coordinate \p x, \p y.
        /// \tparam BORDER      Border/addressing mode.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \param coords       (x, y) coordinates.
        /// \param batch        Batch to use.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float2_t coords, size_t batch) const;

    private:
        const T* m_data{};
        int2_t m_pitch{};
        int2_t m_shape{};
        size_t m_elements{};
        T m_value{};
    private:
        template<BorderMode BORDER>
        NOA_HOST T nearest_(const T* data, float x, float y) const;
        template<BorderMode BORDER, bool COSINE>
        NOA_HOST T linear_(const T* data, float x, float y) const;
        template<BorderMode BORDER, bool BSPLINE>
        NOA_HOST T cubic_(const T* data, float x, float y) const;
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
        /// \param[in] inputs   On the \b host. Input 3D array.
        /// \param pitch        Pitch, in elements, of \p inputs.
        /// \param shape        Logical {fast, medium, slow} shape, in elements, of \p inputs.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST Interpolator3D(const T* input, size3_t pitch, size3_t shape, T value) noexcept;

        /// Resets the data points.
        /// \param[in] inputs   On the \b host. Input 3D array.
        /// \param pitch        Pitch, in elements, of \p inputs.
        /// \param shape        Logical {fast, medium, slow} shape, in elements, of \p inputs.
        /// \param value        Constant value to use for out-of-bound elements. Only use with BORDER_VALUE.
        NOA_HOST void reset(const T* input, size3_t pitch, size3_t shape, T value) noexcept;

        /// Returns the interpolated value at the coordinate \p x, \p y, \p z.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \tparam BORDER      Border/addressing mode.
        /// \param coords       (x, y, z) coordinates.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float3_t coords) const;

        /// Returns the interpolated value at the coordinate \p x, \p y, \p z.
        /// \tparam INTERP      Interpolation/filter mode.
        /// \tparam BORDER      Border/addressing mode.
        /// \param coords       (x, y, z) coordinates.
        /// \param batch        Batch to use.
        /// \return             Interpolated value.
        template<InterpMode INTERP = INTERP_LINEAR, BorderMode BORDER = BORDER_ZERO>
        NOA_HOST T get(float3_t coords, size_t batch) const;

    private:
        const T* m_data{};
        int3_t m_pitch{};
        int3_t m_shape{};
        size_t m_elements{};
        T m_value{};
        int m_page{};
    private:
        template<BorderMode BORDER>
        NOA_HOST T nearest_(const T* data, float x, float y, float z) const;
        template<BorderMode BORDER, bool COSINE>
        NOA_HOST T linear_(const T* data, float x, float y, float z) const;
        template<BorderMode BORDER, bool BSPLINE>
        NOA_HOST T cubic_(const T* data, float x, float y, float z) const;
    };
}

// Implementation:
namespace noa::cpu::transform {
    template<typename T>
    Interpolator1D<T>::Interpolator1D(const T* input, size_t pitch, size_t size, T value) noexcept
            : m_data(input), m_pitch(static_cast<int>(pitch)), m_size(static_cast<int>(size)), m_value(value) {}

    template<typename T>
    void Interpolator1D<T>::reset(const T* input, size_t pitch, size_t size, T value) noexcept {
        m_data = input;
        m_pitch = static_cast<int>(pitch);
        m_size = static_cast<int>(size);
        m_value = value;
    }

    template<typename T>
    template<BorderMode BORDER>
    T Interpolator1D<T>::nearest_(const T* data, float x) const {
        T out;
        auto idx = static_cast<int>(noa::math::round(x));
        if constexpr (BORDER == BORDER_ZERO) {
            out = idx >= 0 && idx < m_size ? data[idx] : static_cast<T>(0);
        } else if constexpr (BORDER == BORDER_VALUE) {
            out = idx >= 0 && idx < m_size ? data[idx] : static_cast<T>(m_value);
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            out = data[getBorderIndex<BORDER>(idx, m_size)];
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
            bool cond[2] = {idx0 >= 0 && idx0 < m_size, idx1 >= 0 && idx1 < m_size};
            if constexpr (BORDER == BORDER_ZERO) {
                values[0] = cond[0] ? data[idx0] : static_cast<T>(0);
                values[1] = cond[1] ? data[idx1] : static_cast<T>(0);
            } else {
                values[0] = cond[0] ? data[idx0] : static_cast<T>(m_value);
                values[1] = cond[1] ? data[idx1] : static_cast<T>(m_value);
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            values[0] = data[getBorderIndex<BORDER>(idx0, m_size)];
            values[1] = data[getBorderIndex<BORDER>(idx1, m_size)];
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
            bool cond[4] = {idx0 >= 0 && idx0 < m_size,
                            idx1 >= 0 && idx1 < m_size,
                            idx2 >= 0 && idx2 < m_size,
                            idx3 >= 0 && idx3 < m_size};
            if constexpr (BORDER == BORDER_ZERO) {
                values[0] = cond[0] ? data[idx0] : static_cast<T>(0);
                values[1] = cond[1] ? data[idx1] : static_cast<T>(0);
                values[2] = cond[2] ? data[idx2] : static_cast<T>(0);
                values[3] = cond[3] ? data[idx3] : static_cast<T>(0);
            } else {
                values[0] = cond[0] ? data[idx0] : static_cast<T>(m_value);
                values[1] = cond[1] ? data[idx1] : static_cast<T>(m_value);
                values[2] = cond[2] ? data[idx2] : static_cast<T>(m_value);
                values[3] = cond[3] ? data[idx3] : static_cast<T>(m_value);
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            values[0] = data[getBorderIndex<BORDER>(idx0, m_size)];
            values[1] = data[getBorderIndex<BORDER>(idx1, m_size)];
            values[2] = data[getBorderIndex<BORDER>(idx2, m_size)];
            values[3] = data[getBorderIndex<BORDER>(idx3, m_size)];
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
    T Interpolator1D<T>::get(float x, size_t batch) const {
        const T* data = m_data + batch * static_cast<size_t>(m_pitch);
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
    Interpolator2D<T>::Interpolator2D(const T* input, size2_t pitch, size2_t shape, T value) noexcept
            : m_data(input), m_pitch(pitch), m_shape(shape), m_elements(elements(pitch)), m_value(value) {}

    template<typename T>
    void Interpolator2D<T>::reset(const T* input, size2_t pitch, size2_t shape, T value) noexcept {
        m_data = input;
        m_pitch = int2_t{pitch};
        m_shape = int2_t{shape};
        m_elements = elements(pitch);
        m_value = value;
    }

    template<typename T>
    template<BorderMode BORDER>
    T Interpolator2D<T>::nearest_(const T* data, float x, float y) const {
        T out;
        int2_t idx(noa::math::round(x), noa::math::round(y));
        if constexpr (BORDER == BORDER_ZERO) {
            if (idx.x < 0 || idx.x >= m_shape.x || idx.y < 0 || idx.y >= m_shape.y)
                out = static_cast<T>(0);
            else
                out = data[idx.y * m_pitch.x + idx.x];
        } else if constexpr (BORDER == BORDER_VALUE) {
            if (idx.x < 0 || idx.x >= m_shape.x || idx.y < 0 || idx.y >= m_shape.y)
                out = static_cast<T>(m_value);
            else
                out = data[idx.y * m_pitch.x + idx.x];
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            idx.x = getBorderIndex<BORDER>(idx.x, m_shape.x);
            idx.y = getBorderIndex<BORDER>(idx.y, m_shape.y);
            out = data[idx.y * m_pitch.x + idx.x];
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return out;
    }

    template<typename T>
    template<BorderMode BORDER, bool COSINE>
    T Interpolator2D<T>::linear_(const T* data, float x, float y) const {
        int2_t idx0(noa::math::floor(x), noa::math::floor(y));
        int2_t idx1(idx0 + 1);
        T values[4]; // v00, v10, v01, v11
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            bool cond_x[2] = {idx0.x >= 0 && idx0.x < m_shape.x, idx1.x >= 0 && idx1.x < m_shape.x};
            bool cond_y[2] = {idx0.y >= 0 && idx0.y < m_shape.y, idx1.y >= 0 && idx1.y < m_shape.y};
            if constexpr (BORDER == BORDER_ZERO) {
                values[0] = cond_y[0] && cond_x[0] ? data[idx0.y * m_pitch.x + idx0.x] : static_cast<T>(0); // v00
                values[1] = cond_y[0] && cond_x[1] ? data[idx0.y * m_pitch.x + idx1.x] : static_cast<T>(0); // v01
                values[2] = cond_y[1] && cond_x[0] ? data[idx1.y * m_pitch.x + idx0.x] : static_cast<T>(0); // v10
                values[3] = cond_y[1] && cond_x[1] ? data[idx1.y * m_pitch.x + idx1.x] : static_cast<T>(0); // v11
            } else {
                values[0] = cond_y[0] && cond_x[0] ? data[idx0.y * m_pitch.x + idx0.x] : static_cast<T>(m_value);
                values[1] = cond_y[0] && cond_x[1] ? data[idx0.y * m_pitch.x + idx1.x] : static_cast<T>(m_value);
                values[2] = cond_y[1] && cond_x[0] ? data[idx1.y * m_pitch.x + idx0.x] : static_cast<T>(m_value);
                values[3] = cond_y[1] && cond_x[1] ? data[idx1.y * m_pitch.x + idx1.x] : static_cast<T>(m_value);
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            int tmp[4] = {getBorderIndex<BORDER>(idx0.x, m_shape.x),
                          getBorderIndex<BORDER>(idx1.x, m_shape.x),
                          getBorderIndex<BORDER>(idx0.y, m_shape.y),
                          getBorderIndex<BORDER>(idx1.y, m_shape.y)};
            values[0] = data[tmp[2] * m_pitch.x + tmp[0]]; // v00
            values[1] = data[tmp[2] * m_pitch.x + tmp[1]]; // v01
            values[2] = data[tmp[3] * m_pitch.x + tmp[0]]; // v10
            values[3] = data[tmp[3] * m_pitch.x + tmp[1]]; // v11
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        float2_t fraction(x - static_cast<float>(idx0.x), y - static_cast<float>(idx0.y));
        if constexpr (COSINE)
            return cosine2D(values[0], values[1], values[2], values[3], fraction.x, fraction.y);
        else
            return linear2D(values[0], values[1], values[2], values[3], fraction.x, fraction.y);
    }

    template<typename T>
    template<BorderMode BORDER, bool BSPLINE>
    T Interpolator2D<T>::cubic_(const T* data, float x, float y) const {
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
                int off_y = (idx.y + offset[j]) * m_pitch.x;
                for (int i = 0; i < 4; i++) {
                    if constexpr (BORDER == BORDER_ZERO)
                        square[j][i] = cond_x[i] && cond_y[j] ?
                                       data[off_y + idx.x + offset[i]] : static_cast<T>(0);
                    else
                        square[j][i] = cond_x[i] && cond_y[j] ?
                                       data[off_y + idx.x + offset[i]] : static_cast<T>(m_value);
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
                int offset = tmp_y[j] * m_pitch.x;
                for (int i = 0; i < 4; ++i) {
                    square[j][i] = data[offset + tmp_x[i]];
                }
            }
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        float2_t fraction(x - static_cast<float>(idx.x), y - static_cast<float>(idx.y));
        if constexpr (BSPLINE)
            return cubicBSpline2D(square, fraction.x, fraction.y);
        else
            return cubic2D(square, fraction.x, fraction.y);
    }

    template<typename T>
    template<InterpMode INTERP, BorderMode BORDER>
    T Interpolator2D<T>::get(float2_t coords) const {
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(m_data, coords.x, coords.y);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(m_data, coords.x, coords.y);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(m_data, coords.x, coords.y);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(m_data, coords.x, coords.y);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(m_data, coords.x, coords.y);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    template<typename T>
    template<InterpMode INTERP, BorderMode BORDER>
    T Interpolator2D<T>::get(float2_t coords, size_t batch) const {
        const T* data = m_data + batch * m_elements;
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(data, coords.x, coords.y);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(data, coords.x, coords.y);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(data, coords.x, coords.y);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(data, coords.x, coords.y);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(data, coords.x, coords.y);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    // -- 3D -- //

    template<typename T>
    Interpolator3D<T>::Interpolator3D(const T* input, size3_t pitch, size3_t shape, T value) noexcept
            : m_data(input), m_pitch(pitch), m_shape(shape), m_elements(elements(pitch)),
            m_value(value), m_page(m_pitch.x * m_pitch.y) {}

    template<typename T>
    void Interpolator3D<T>::reset(const T* input, size3_t pitch, size3_t shape, T value) noexcept {
        m_data = input;
        m_pitch = int3_t{pitch};
        m_shape = int3_t{shape};
        m_elements = elements(pitch);
        m_value = value;
        m_page = m_pitch.x * m_pitch.y;
    }

    template<typename T>
    template<BorderMode BORDER>
    T Interpolator3D<T>::nearest_(const T* data, float x, float y, float z) const {
        T out;
        int3_t idx(noa::math::round(x), noa::math::round(y), noa::math::round(z));
        if constexpr (BORDER == BORDER_ZERO) {
            if (idx.x < 0 || idx.x >= m_shape.x ||
                idx.y < 0 || idx.y >= m_shape.y ||
                idx.z < 0 || idx.z >= m_shape.z)
                out = static_cast<T>(0);
            else
                out = data[(idx.z * m_pitch.y + idx.y) * m_pitch.x + idx.x];
        } else if constexpr (BORDER == BORDER_VALUE) {
            if (idx.x < 0 || idx.x >= m_shape.x ||
                idx.y < 0 || idx.y >= m_shape.y ||
                idx.z < 0 || idx.z >= m_shape.z)
                out = static_cast<T>(m_value);
            else
                out = data[(idx.z * m_pitch.y + idx.y) * m_pitch.x + idx.x];
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            idx.x = getBorderIndex<BORDER>(idx.x, m_shape.x);
            idx.y = getBorderIndex<BORDER>(idx.y, m_shape.y);
            idx.z = getBorderIndex<BORDER>(idx.z, m_shape.z);
            out = data[(idx.z * m_pitch.y + idx.y) * m_pitch.x + idx.x];
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return out;
    }

    template<typename T>
    template<BorderMode BORDER, bool COSINE>
    T Interpolator3D<T>::linear_(const T* data, float x, float y, float z) const {
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
            int off_y[2] = {idx[0].y * m_pitch.x, idx[1].y * m_pitch.x};
            int off_z[2] = {idx[0].z * m_page, idx[1].z * m_page};
            values[0] = cond_z[0] && cond_y[0] && cond_x[0] ? data[off_z[0] + off_y[0] + idx[0].x] : cval; // v000
            values[1] = cond_z[0] && cond_y[0] && cond_x[1] ? data[off_z[0] + off_y[0] + idx[1].x] : cval; // v001
            values[2] = cond_z[0] && cond_y[1] && cond_x[0] ? data[off_z[0] + off_y[1] + idx[0].x] : cval; // v010
            values[3] = cond_z[0] && cond_y[1] && cond_x[1] ? data[off_z[0] + off_y[1] + idx[1].x] : cval; // v011
            values[4] = cond_z[1] && cond_y[0] && cond_x[0] ? data[off_z[1] + off_y[0] + idx[0].x] : cval; // v100
            values[5] = cond_z[1] && cond_y[0] && cond_x[1] ? data[off_z[1] + off_y[0] + idx[1].x] : cval; // v101
            values[6] = cond_z[1] && cond_y[1] && cond_x[0] ? data[off_z[1] + off_y[1] + idx[0].x] : cval; // v110
            values[7] = cond_z[1] && cond_y[1] && cond_x[1] ? data[off_z[1] + off_y[1] + idx[1].x] : cval; // v111

        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            int tmp[6] = {getBorderIndex<BORDER>(idx[0].x, m_shape.x),
                          getBorderIndex<BORDER>(idx[1].x, m_shape.x),
                          getBorderIndex<BORDER>(idx[0].y, m_shape.y),
                          getBorderIndex<BORDER>(idx[1].y, m_shape.y),
                          getBorderIndex<BORDER>(idx[0].z, m_shape.z),
                          getBorderIndex<BORDER>(idx[1].z, m_shape.z)};
            values[0] = data[(tmp[4] * m_pitch.y + tmp[2]) * m_pitch.x + tmp[0]]; // v000
            values[1] = data[(tmp[4] * m_pitch.y + tmp[2]) * m_pitch.x + tmp[1]]; // v001
            values[2] = data[(tmp[4] * m_pitch.y + tmp[3]) * m_pitch.x + tmp[0]]; // v010
            values[3] = data[(tmp[4] * m_pitch.y + tmp[3]) * m_pitch.x + tmp[1]]; // v011
            values[4] = data[(tmp[5] * m_pitch.y + tmp[2]) * m_pitch.x + tmp[0]]; // v100
            values[5] = data[(tmp[5] * m_pitch.y + tmp[2]) * m_pitch.x + tmp[1]]; // v101
            values[6] = data[(tmp[5] * m_pitch.y + tmp[3]) * m_pitch.x + tmp[0]]; // v110
            values[7] = data[(tmp[5] * m_pitch.y + tmp[3]) * m_pitch.x + tmp[1]]; // v111
        } else {
            static_assert(noa::traits::always_false_v<T>);
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
    T Interpolator3D<T>::cubic_(const T* data, float x, float y, float z) const {
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
                    int off = off_z + (idx.y + offset[j]) * m_pitch.x;
                    for (int i = 0; i < 4; ++i) {
                        values[k][j][i] = cond_z[k] && cond_y[j] && cond_x[i] ?
                                          data[off + idx.x + offset[i]] : cval;
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
                    int offset = off_z + tmp_y[j] * m_pitch.x;
                    for (int i = 0; i < 4; ++i) {
                        values[k][j][i] = data[offset + tmp_x[i]];
                    }
                }
            }
        } else {
            static_assert(noa::traits::always_false_v<T>);
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
    T Interpolator3D<T>::get(float3_t coords) const {
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(m_data, coords.x, coords.y, coords.z);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(m_data, coords.x, coords.y, coords.z);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(m_data, coords.x, coords.y, coords.z);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(m_data, coords.x, coords.y, coords.z);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(m_data, coords.x, coords.y, coords.z);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    template<typename T>
    template<InterpMode INTERP, BorderMode BORDER>
    T Interpolator3D<T>::get(float3_t coords, size_t batch) const {
        const T* data = m_data + batch * m_elements;
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(data, coords.x, coords.y, coords.z);
        } else if constexpr (INTERP == INTERP_LINEAR) {
            return linear_<BORDER, false>(data, coords.x, coords.y, coords.z);
        } else if constexpr (INTERP == INTERP_COSINE) {
            return linear_<BORDER, true>(data, coords.x, coords.y, coords.z);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(data, coords.x, coords.y, coords.z);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE) {
            return cubic_<BORDER, true>(data, coords.x, coords.y, coords.z);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }
}
