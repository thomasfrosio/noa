#pragma once

#include "noa/common/Types.h"
#include "noa/common/geometry/Interpolate.h"

// On of the main difference between these interpolations and what we can find on other cryoEM packages,
// is that the interpolation window can be partially out-of-bound (OOB), that is, elements that are OOB
// are replaced according to a BorderMode. cryoEM packages usually check that all elements are in bound
// and if there's even one element OOB, they don't interpolate.
// Note: The interpolators below are for real space interpolation, or redundant and centered Fourier transforms.

// The coordinate system matches the indexing. This is quite standard.
// For instance the first data sample at index 0 is located at the coordinate 0 and the coordinate 0.5
// is just in between the first and second element. As such, the fractional part of the coordinate
// corresponds to the ratio/weight used by the interpolation function. In other words,
// the coordinate system locates the data between -0.5 and N-1 + 0.5.

namespace noa::geometry {
    // Interpolates 2D data.
    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE,
             typename Data, typename Index = int64_t, typename Coord = float,
             int ACCESSOR_NDIM = 2, AccessorTraits ACCESSOR_TRAITS = AccessorTraits::DEFAULT>
    class Interpolator2D {
    public:
        static_assert(traits::is_any_v<Index, int32_t, int64_t>);
        static_assert(traits::is_float_or_complex_v<Data> && !std::is_const_v<Data>);
        static_assert(traits::is_float_v<Coord> && !std::is_const_v<Coord>);
        static_assert(ACCESSOR_NDIM == 2 || ACCESSOR_NDIM == 3);

        using data_t = Data;
        using index_t = Index;
        using index2_t = Int2<index_t>;
        using coord_t = Coord;
        using coord2_t = Float2<coord_t>;
        using accessor_t = Accessor<const data_t, ACCESSOR_NDIM, index_t, ACCESSOR_TRAITS>;

    public:
        constexpr Interpolator2D() = default;

        constexpr NOA_HD Interpolator2D(accessor_t data, index2_t shape, Data cvalue = Data{0}) noexcept
                : m_data(data), m_shape(shape), m_cvalue(cvalue) {}

        template<typename void_t = void, typename = std::enable_if_t<ACCESSOR_NDIM == 2 && std::is_same_v<void_t, void>>>
        constexpr NOA_HD data_t operator()(coord2_t coordinate) const noexcept {
            if constexpr (INTERP_MODE == INTERP_NEAREST) {
                return nearest_(m_data, coordinate);
            } else if constexpr (INTERP_MODE == INTERP_LINEAR ||
                               INTERP_MODE == INTERP_LINEAR_FAST ||
                               INTERP_MODE == INTERP_COSINE ||
                               INTERP_MODE == INTERP_COSINE_FAST) {
                return linear_(m_data, coordinate);
            } else if constexpr (INTERP_MODE == INTERP_CUBIC ||
                               INTERP_MODE == INTERP_CUBIC_BSPLINE ||
                               INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST) {
                return cubic_(m_data, coordinate);
            } else {
                static_assert(traits::always_false_v<data_t>);
            }
        }

        template<typename I = index_t, typename = std::enable_if_t<ACCESSOR_NDIM == 3 && traits::is_int_v<I>>>
        constexpr NOA_HD data_t operator()(coord2_t coordinate, I batch) const noexcept {
            if constexpr (INTERP_MODE == INTERP_NEAREST) {
                return nearest_(m_data[batch], coordinate);
            } else if constexpr (INTERP_MODE == INTERP_LINEAR ||
                                 INTERP_MODE == INTERP_LINEAR_FAST ||
                                 INTERP_MODE == INTERP_COSINE ||
                                 INTERP_MODE == INTERP_COSINE_FAST) {
                return linear_(m_data[batch], coordinate);
            } else if constexpr (INTERP_MODE == INTERP_CUBIC ||
                                 INTERP_MODE == INTERP_CUBIC_BSPLINE ||
                                 INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST) {
                return cubic_(m_data[batch], coordinate);
            } else {
                static_assert(traits::always_false_v<data_t>);
            }
        }

    private:
        template<typename accessor2d_t>
        constexpr NOA_HD data_t nearest_(accessor2d_t accessor, coord2_t coordinate) const noexcept {
            static_assert(accessor2d_t::COUNT == 2);
            const index2_t idx(::noa::math::round(coordinate));

            if constexpr (BORDER_MODE == BORDER_ZERO) {
                if (idx[1] < 0 || idx[1] >= m_shape[1] ||
                    idx[0] < 0 || idx[0] >= m_shape[0])
                    return data_t{0};
                else
                    return accessor(idx[0], idx[1]);
            } else if constexpr (BORDER_MODE == BORDER_VALUE) {
                if (idx[1] < 0 || idx[1] >= m_shape[1] ||
                    idx[0] < 0 || idx[0] >= m_shape[0])
                    return static_cast<data_t>(m_cvalue);
                else
                    return accessor(idx[0], idx[1]);
            } else if constexpr (BORDER_MODE == BORDER_CLAMP || BORDER_MODE == BORDER_PERIODIC ||
                                 BORDER_MODE == BORDER_MIRROR || BORDER_MODE == BORDER_REFLECT) {
                return accessor(indexing::at<BORDER_MODE>(idx[0], m_shape[0]),
                                indexing::at<BORDER_MODE>(idx[1], m_shape[1]));
            } else {
                static_assert(traits::always_false_v<data_t>);
            }
        }

        template<typename accessor2d_t>
        constexpr NOA_HD data_t linear_(accessor2d_t accessor, coord2_t coordinate) const noexcept {
            static_assert(accessor2d_t::COUNT == 2);
            const index2_t idx0(::noa::math::floor(coordinate));
            const index2_t idx1(idx0 + 1);

            data_t values[4]; // v00, v10, v01, v11
            if constexpr (BORDER_MODE == BORDER_ZERO || BORDER_MODE == BORDER_VALUE) {
                const bool cond_y[2] = {idx0[0] >= 0 && idx0[0] < m_shape[0], idx1[0] >= 0 && idx1[0] < m_shape[0]};
                const bool cond_x[2] = {idx0[1] >= 0 && idx0[1] < m_shape[1], idx1[1] >= 0 && idx1[1] < m_shape[1]};

                if constexpr (BORDER_MODE == BORDER_ZERO) {
                    values[0] = cond_y[0] && cond_x[0] ? accessor(idx0[0], idx0[1]) : data_t{0}; // v00
                    values[1] = cond_y[0] && cond_x[1] ? accessor(idx0[0], idx1[1]) : data_t{0}; // v01
                    values[2] = cond_y[1] && cond_x[0] ? accessor(idx1[0], idx0[1]) : data_t{0}; // v10
                    values[3] = cond_y[1] && cond_x[1] ? accessor(idx1[0], idx1[1]) : data_t{0}; // v11
                } else {
                    values[0] = cond_y[0] && cond_x[0] ? accessor(idx0[0], idx0[1]) : m_cvalue;
                    values[1] = cond_y[0] && cond_x[1] ? accessor(idx0[0], idx1[1]) : m_cvalue;
                    values[2] = cond_y[1] && cond_x[0] ? accessor(idx1[0], idx0[1]) : m_cvalue;
                    values[3] = cond_y[1] && cond_x[1] ? accessor(idx1[0], idx1[1]) : m_cvalue;
                }

            } else if constexpr (BORDER_MODE == BORDER_CLAMP || BORDER_MODE == BORDER_PERIODIC ||
                                 BORDER_MODE == BORDER_MIRROR || BORDER_MODE == BORDER_REFLECT) {
                const index_t tmp[4] = {indexing::at<BORDER_MODE>(idx0[1], m_shape[1]),
                                        indexing::at<BORDER_MODE>(idx1[1], m_shape[1]),
                                        indexing::at<BORDER_MODE>(idx0[0], m_shape[0]),
                                        indexing::at<BORDER_MODE>(idx1[0], m_shape[0])};
                values[0] = accessor(tmp[2], tmp[0]); // v00
                values[1] = accessor(tmp[2], tmp[1]); // v01
                values[2] = accessor(tmp[3], tmp[0]); // v10
                values[3] = accessor(tmp[3], tmp[1]); // v11

            } else {
                static_assert(traits::always_false_v<data_t>);
            }

            const coord2_t fraction{coordinate - static_cast<coord2_t>(idx0)};
            if constexpr (INTERP_MODE == INTERP_COSINE || INTERP_MODE == INTERP_COSINE_FAST)
                return interpolate::cosine2D(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
            else
                return interpolate::lerp2D(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
        }

        template<typename accessor2d_t>
        constexpr NOA_HD data_t cubic_(accessor2d_t accessor, coord2_t coordinate) const noexcept {
            static_assert(accessor2d_t::COUNT == 2);
            const index2_t idx(::noa::math::floor(coordinate));

            data_t square[4][4]; // [y][x]
            if constexpr (BORDER_MODE == BORDER_ZERO || BORDER_MODE == BORDER_VALUE) {
                const bool cond_y[4] = {idx[0] - 1 >= 0 && idx[0] - 1 < m_shape[0],
                                        idx[0] + 0 >= 0 && idx[0] + 0 < m_shape[0],
                                        idx[0] + 1 >= 0 && idx[0] + 1 < m_shape[0],
                                        idx[0] + 2 >= 0 && idx[0] + 2 < m_shape[0]};
                const bool cond_x[4] = {idx[1] - 1 >= 0 && idx[1] - 1 < m_shape[1],
                                        idx[1] + 0 >= 0 && idx[1] + 0 < m_shape[1],
                                        idx[1] + 1 >= 0 && idx[1] + 1 < m_shape[1],
                                        idx[1] + 2 >= 0 && idx[1] + 2 < m_shape[1]};
                constexpr index_t offset[4] = {-1, 0, 1, 2};
                for (index_t j = 0; j < 4; ++j) {
                    const index_t idx_y = idx[0] + offset[j];
                    for (index_t i = 0; i < 4; ++i) {
                        const index_t idx_x = idx[1] + offset[i];
                        if constexpr (BORDER_MODE == BORDER_ZERO)
                            square[j][i] = cond_x[i] && cond_y[j] ? accessor(idx_y, idx_x) : data_t{0};
                        else
                            square[j][i] = cond_x[i] && cond_y[j] ? accessor(idx_y, idx_x) : m_cvalue;
                    }
                }
            } else if constexpr (BORDER_MODE == BORDER_CLAMP || BORDER_MODE == BORDER_PERIODIC ||
                                 BORDER_MODE == BORDER_MIRROR || BORDER_MODE == BORDER_REFLECT) {
                const index_t tmp_y[4] = {indexing::at<BORDER_MODE>(idx[0] - 1, m_shape[0]),
                                          indexing::at<BORDER_MODE>(idx[0] + 0, m_shape[0]),
                                          indexing::at<BORDER_MODE>(idx[0] + 1, m_shape[0]),
                                          indexing::at<BORDER_MODE>(idx[0] + 2, m_shape[0])};
                const index_t tmp_x[4] = {indexing::at<BORDER_MODE>(idx[1] - 1, m_shape[1]),
                                          indexing::at<BORDER_MODE>(idx[1] + 0, m_shape[1]),
                                          indexing::at<BORDER_MODE>(idx[1] + 1, m_shape[1]),
                                          indexing::at<BORDER_MODE>(idx[1] + 2, m_shape[1])};
                for (index_t j = 0; j < 4; ++j)
                    for (index_t i = 0; i < 4; ++i)
                        square[j][i] = accessor(tmp_y[j], tmp_x[i]);

            } else {
                static_assert(traits::always_false_v<data_t>);
            }
            const coord2_t fraction{coordinate - static_cast<coord2_t>(idx)};
            if constexpr (INTERP_MODE == INTERP_CUBIC_BSPLINE || INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST)
                return interpolate::cubicBSpline2D(square, fraction[1], fraction[0]);
            else
                return interpolate::cubic2D(square, fraction[1], fraction[0]);
        }

    private:
        accessor_t m_data{};
        index2_t m_shape{};
        data_t m_cvalue{};
    };

    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE, typename coord_t = float, typename cvalue_t = float,
             typename data_t, typename index_t, int NDIM, AccessorTraits TRAITS,
             typename = std::enable_if_t<traits::is_almost_same_v<data_t, cvalue_t>>>
    constexpr auto interpolator2D(const Accessor<data_t, NDIM, index_t, TRAITS>& accessor,
                                  Int2<index_t> shape,
                                  cvalue_t cvalue = cvalue_t{0}) {
        using mutable_data_t = std::remove_cv_t<data_t>;
        using interpolator_t = Interpolator2D<BORDER_MODE, INTERP_MODE, mutable_data_t, index_t, coord_t, NDIM, TRAITS>;
        return interpolator_t(accessor, shape, cvalue);
    }

    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE, typename coord_t = float, typename cvalue_t = float,
             typename data_t, typename index_t, int NDIM, AccessorTraits TRAITS,
             typename = std::enable_if_t<traits::is_almost_same_v<data_t, cvalue_t>>>
    constexpr auto interpolator2D(const AccessorReference<data_t, NDIM, index_t, TRAITS>& accessor,
                                  Int2<index_t> shape,
                                  cvalue_t cvalue = cvalue_t{0}) {
        using mutable_data_t = std::remove_cv_t<data_t>;
        using interpolator_t = Interpolator2D<BORDER_MODE, INTERP_MODE, mutable_data_t, index_t, coord_t, NDIM, TRAITS>;
        return interpolator_t({accessor.data(), accessor.strides()}, shape, cvalue);
    }
}

namespace noa::geometry {
    // Interpolates 2D data.
    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE,
            typename Data, typename Index = int64_t, typename Coord = float,
            int ACCESSOR_NDIM = 3, AccessorTraits ACCESSOR_TRAITS = AccessorTraits::DEFAULT>
    class Interpolator3D {
    public:
        static_assert(traits::is_any_v<Index, int32_t, int64_t>);
        static_assert(traits::is_float_or_complex_v<Data> && !std::is_const_v<Data>);
        static_assert(traits::is_float_v<Coord> && !std::is_const_v<Coord>);
        static_assert(ACCESSOR_NDIM == 3 || ACCESSOR_NDIM == 4);

        using data_t = Data;
        using index_t = Index;
        using index3_t = Int3<index_t>;
        using coord_t = Coord;
        using coord3_t = Float3<coord_t>;
        using accessor_t = Accessor<const data_t, ACCESSOR_NDIM, index_t, ACCESSOR_TRAITS>;

    public:
        constexpr Interpolator3D() = default;

        constexpr NOA_HD Interpolator3D(accessor_t data, index3_t shape, Data cvalue = Data{0}) noexcept
                : m_data(data), m_shape(shape), m_cvalue(cvalue) {}

        template<typename void_t = void, typename = std::enable_if_t<ACCESSOR_NDIM == 3 && std::is_same_v<void_t, void>>>
        constexpr NOA_HD data_t operator()(coord3_t coordinate) const noexcept {
            if constexpr (INTERP_MODE == INTERP_NEAREST) {
                return nearest_(m_data, coordinate);
            } else if constexpr (INTERP_MODE == INTERP_LINEAR ||
                                 INTERP_MODE == INTERP_LINEAR_FAST ||
                                 INTERP_MODE == INTERP_COSINE ||
                                 INTERP_MODE == INTERP_COSINE_FAST) {
                return linear_(m_data, coordinate);
            } else if constexpr (INTERP_MODE == INTERP_CUBIC ||
                                 INTERP_MODE == INTERP_CUBIC_BSPLINE ||
                                 INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST) {
                return cubic_(m_data, coordinate);
            } else {
                static_assert(traits::always_false_v<data_t>);
            }
        }

        template<typename I = index_t, typename = std::enable_if_t<ACCESSOR_NDIM == 4 && traits::is_int_v<I>>>
        constexpr NOA_HD data_t operator()(coord3_t coordinate, I batch) const noexcept {
            if constexpr (INTERP_MODE == INTERP_NEAREST) {
                return nearest_(m_data[batch], coordinate);
            } else if constexpr (INTERP_MODE == INTERP_LINEAR ||
                                 INTERP_MODE == INTERP_LINEAR_FAST ||
                                 INTERP_MODE == INTERP_COSINE ||
                                 INTERP_MODE == INTERP_COSINE_FAST) {
                return linear_(m_data[batch], coordinate);
            } else if constexpr (INTERP_MODE == INTERP_CUBIC ||
                                 INTERP_MODE == INTERP_CUBIC_BSPLINE ||
                                 INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST) {
                return cubic_(m_data[batch], coordinate);
            } else {
                static_assert(traits::always_false_v<data_t>);
            }
        }

    private:
        template<typename accessor3d_t>
        constexpr NOA_HD data_t nearest_(accessor3d_t accessor, coord3_t coordinate) const noexcept {
            static_assert(accessor3d_t::COUNT == 3);

            const index3_t idx(noa::math::round(coordinate));
            if constexpr (BORDER_MODE == BORDER_ZERO) {
                if (idx[2] < 0 || idx[2] >= m_shape[2] ||
                    idx[1] < 0 || idx[1] >= m_shape[1] ||
                    idx[0] < 0 || idx[0] >= m_shape[0])
                    return data_t{0};
                else
                    return accessor(idx[0], idx[1], idx[2]);
            } else if constexpr (BORDER_MODE == BORDER_VALUE) {
                if (idx[2] < 0 || idx[2] >= m_shape[2] ||
                    idx[1] < 0 || idx[1] >= m_shape[1] ||
                    idx[0] < 0 || idx[0] >= m_shape[0])
                    return static_cast<data_t>(m_cvalue);
                else
                    return accessor(idx[0], idx[1], idx[2]);
            } else if constexpr (BORDER_MODE == BORDER_CLAMP || BORDER_MODE == BORDER_PERIODIC ||
                                 BORDER_MODE == BORDER_MIRROR || BORDER_MODE == BORDER_REFLECT) {
                return accessor(indexing::at<BORDER_MODE>(idx[0], m_shape[0]),
                                indexing::at<BORDER_MODE>(idx[1], m_shape[1]),
                                indexing::at<BORDER_MODE>(idx[2], m_shape[2]));
            } else {
                static_assert(traits::always_false_v<data_t>);
            }
        }

        template<typename accessor3d_t>
        constexpr NOA_HD data_t linear_(accessor3d_t accessor, coord3_t coordinate) const noexcept {
            static_assert(accessor3d_t::COUNT == 3);
            index3_t idx[2];
            idx[0] = index3_t{noa::math::floor(coordinate)};
            idx[1] = idx[0] + 1;

            data_t values[8];
            if constexpr (BORDER_MODE == BORDER_ZERO || BORDER_MODE == BORDER_VALUE) {
                const bool cond_z[2] = {idx[0][0] >= 0 && idx[0][0] < m_shape[0], idx[1][0] >= 0 && idx[1][0] < m_shape[0]};
                const bool cond_y[2] = {idx[0][1] >= 0 && idx[0][1] < m_shape[1], idx[1][1] >= 0 && idx[1][1] < m_shape[1]};
                const bool cond_x[2] = {idx[0][2] >= 0 && idx[0][2] < m_shape[2], idx[1][2] >= 0 && idx[1][2] < m_shape[2]};

                data_t cval;
                if constexpr (BORDER_MODE == BORDER_ZERO)
                    cval = data_t{0};
                else
                    cval = m_cvalue;
                const index_t off_z[2] = {idx[0][0] * accessor.stride(0), idx[1][0] * accessor.stride(0)};
                const index_t off_y[2] = {idx[0][1] * accessor.stride(1), idx[1][1] * accessor.stride(1)};
                const index_t off_x[2] = {idx[0][2] * accessor.stride(2), idx[1][2] * accessor.stride(2)};
                values[0] = cond_z[0] && cond_y[0] && cond_x[0] ? accessor.get()[off_z[0] + off_y[0] + off_x[0]] : cval; // v000
                values[1] = cond_z[0] && cond_y[0] && cond_x[1] ? accessor.get()[off_z[0] + off_y[0] + off_x[1]] : cval; // v001
                values[2] = cond_z[0] && cond_y[1] && cond_x[0] ? accessor.get()[off_z[0] + off_y[1] + off_x[0]] : cval; // v010
                values[3] = cond_z[0] && cond_y[1] && cond_x[1] ? accessor.get()[off_z[0] + off_y[1] + off_x[1]] : cval; // v011
                values[4] = cond_z[1] && cond_y[0] && cond_x[0] ? accessor.get()[off_z[1] + off_y[0] + off_x[0]] : cval; // v100
                values[5] = cond_z[1] && cond_y[0] && cond_x[1] ? accessor.get()[off_z[1] + off_y[0] + off_x[1]] : cval; // v101
                values[6] = cond_z[1] && cond_y[1] && cond_x[0] ? accessor.get()[off_z[1] + off_y[1] + off_x[0]] : cval; // v110
                values[7] = cond_z[1] && cond_y[1] && cond_x[1] ? accessor.get()[off_z[1] + off_y[1] + off_x[1]] : cval; // v111

            } else if constexpr (BORDER_MODE == BORDER_CLAMP || BORDER_MODE == BORDER_PERIODIC ||
                                 BORDER_MODE == BORDER_MIRROR || BORDER_MODE == BORDER_REFLECT) {
                const index_t tmp[6] = {indexing::at<BORDER_MODE>(idx[0][2], m_shape[2]),
                                        indexing::at<BORDER_MODE>(idx[1][2], m_shape[2]),
                                        indexing::at<BORDER_MODE>(idx[0][1], m_shape[1]),
                                        indexing::at<BORDER_MODE>(idx[1][1], m_shape[1]),
                                        indexing::at<BORDER_MODE>(idx[0][0], m_shape[0]),
                                        indexing::at<BORDER_MODE>(idx[1][0], m_shape[0])};
                values[0] = accessor(tmp[4], tmp[2], tmp[0]); // v000
                values[1] = accessor(tmp[4], tmp[2], tmp[1]); // v001
                values[2] = accessor(tmp[4], tmp[3], tmp[0]); // v010
                values[3] = accessor(tmp[4], tmp[3], tmp[1]); // v011
                values[4] = accessor(tmp[5], tmp[2], tmp[0]); // v100
                values[5] = accessor(tmp[5], tmp[2], tmp[1]); // v101
                values[6] = accessor(tmp[5], tmp[3], tmp[0]); // v110
                values[7] = accessor(tmp[5], tmp[3], tmp[1]); // v111
            } else {
                static_assert(traits::always_false_v<data_t>);
            }

            const coord3_t fraction{coordinate - static_cast<coord3_t>(idx[0])};
            if constexpr (INTERP_MODE == INTERP_COSINE || INTERP_MODE == INTERP_CUBIC_BSPLINE) {
                return interpolate::cosine3D(values[0], values[1], values[2], values[3],
                                             values[4], values[5], values[6], values[7],
                                             fraction[2], fraction[1], fraction[0]);
            } else {
                return interpolate::lerp3D(values[0], values[1], values[2], values[3],
                                           values[4], values[5], values[6], values[7],
                                           fraction[2], fraction[1], fraction[0]);
            }
        }

        template<typename accessor3d_t>
        constexpr NOA_HD data_t cubic_(accessor3d_t accessor, coord3_t coordinate) const noexcept {
            static_assert(accessor3d_t::COUNT == 3);

            const index3_t idx(noa::math::floor(coordinate));
            data_t values[4][4][4]; // [z][y][x]
            if constexpr (BORDER_MODE == BORDER_ZERO || BORDER_MODE == BORDER_VALUE) {
                const bool cond_z[4] = {idx[0] - 1 >= 0 && idx[0] - 1 < m_shape[0],
                                        idx[0] + 0 >= 0 && idx[0] + 0 < m_shape[0],
                                        idx[0] + 1 >= 0 && idx[0] + 1 < m_shape[0],
                                        idx[0] + 2 >= 0 && idx[0] + 2 < m_shape[0]};
                const bool cond_y[4] = {idx[1] - 1 >= 0 && idx[1] - 1 < m_shape[1],
                                        idx[1] + 0 >= 0 && idx[1] + 0 < m_shape[1],
                                        idx[1] + 1 >= 0 && idx[1] + 1 < m_shape[1],
                                        idx[1] + 2 >= 0 && idx[1] + 2 < m_shape[1]};
                const bool cond_x[4] = {idx[2] - 1 >= 0 && idx[2] - 1 < m_shape[2],
                                        idx[2] + 0 >= 0 && idx[2] + 0 < m_shape[2],
                                        idx[2] + 1 >= 0 && idx[2] + 1 < m_shape[2],
                                        idx[2] + 2 >= 0 && idx[2] + 2 < m_shape[2]};
                data_t cval;
                if constexpr (BORDER_MODE == BORDER_ZERO)
                    cval = data_t{0};
                else
                    cval = m_cvalue;
                constexpr index_t offset[4] = {-1, 0, 1, 2};
                for (index_t i = 0; i < 4; ++i) {
                    const index_t idx_z = idx[0] + offset[i];
                    for (index_t j = 0; j < 4; ++j) {
                        const index_t idx_y = idx[1] + offset[j];
                        for (index_t k = 0; k < 4; ++k) {
                            values[i][j][k] = cond_z[i] && cond_y[j] && cond_x[k] ?
                                              accessor(idx_z, idx_y, idx[2] + offset[k]) : cval;
                        }
                    }
                }
            } else if constexpr (BORDER_MODE == BORDER_CLAMP || BORDER_MODE == BORDER_PERIODIC ||
                                 BORDER_MODE == BORDER_MIRROR || BORDER_MODE == BORDER_REFLECT) {
                const index_t tmp_z[4] = {indexing::at<BORDER_MODE>(idx[0] - 1, m_shape[0]),
                                          indexing::at<BORDER_MODE>(idx[0] + 0, m_shape[0]),
                                          indexing::at<BORDER_MODE>(idx[0] + 1, m_shape[0]),
                                          indexing::at<BORDER_MODE>(idx[0] + 2, m_shape[0])};
                const index_t tmp_y[4] = {indexing::at<BORDER_MODE>(idx[1] - 1, m_shape[1]),
                                          indexing::at<BORDER_MODE>(idx[1] + 0, m_shape[1]),
                                          indexing::at<BORDER_MODE>(idx[1] + 1, m_shape[1]),
                                          indexing::at<BORDER_MODE>(idx[1] + 2, m_shape[1])};
                const index_t tmp_x[4] = {indexing::at<BORDER_MODE>(idx[2] - 1, m_shape[2]),
                                          indexing::at<BORDER_MODE>(idx[2] + 0, m_shape[2]),
                                          indexing::at<BORDER_MODE>(idx[2] + 1, m_shape[2]),
                                          indexing::at<BORDER_MODE>(idx[2] + 2, m_shape[2])};
                for (index_t i = 0; i < 4; ++i)
                    for (index_t j = 0; j < 4; ++j)
                        for (index_t k = 0; k < 4; ++k)
                            values[i][j][k] = accessor(tmp_z[i], tmp_y[j], tmp_x[k]);

            } else {
                static_assert(traits::always_false_v<data_t>);
            }

            const coord3_t fraction{coordinate - static_cast<coord3_t>(idx)};
            if constexpr (INTERP_MODE == INTERP_CUBIC_BSPLINE || INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST)
                return interpolate::cubicBSpline3D(values, fraction[2], fraction[1], fraction[0]);
            else
                return interpolate::cubic3D(values, fraction[2], fraction[1], fraction[0]);
        }

    private:
        accessor_t m_data{};
        index3_t m_shape{};
        data_t m_cvalue{};
    };

    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE, typename coord_t = float, typename cvalue_t = float,
             typename data_t, typename index_t, int NDIM, AccessorTraits TRAITS,
             typename = std::enable_if_t<traits::is_almost_same_v<data_t, cvalue_t>>>
    constexpr auto interpolator3D(const Accessor<data_t, NDIM, index_t, TRAITS>& accessor,
                                  Int3<index_t> shape,
                                  cvalue_t cvalue = cvalue_t{0}) {
        using mutable_data_t = std::remove_cv_t<data_t>;
        using interpolator_t = Interpolator3D<BORDER_MODE, INTERP_MODE, mutable_data_t, index_t, coord_t, NDIM, TRAITS>;
        return interpolator_t(accessor, shape, cvalue);
    }

    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE, typename coord_t = float, typename cvalue_t = float,
             typename data_t, typename index_t, int NDIM, AccessorTraits TRAITS,
             typename = std::enable_if_t<traits::is_almost_same_v<data_t, cvalue_t>>>
    constexpr auto interpolator3D(const AccessorReference<data_t, NDIM, index_t, TRAITS>& accessor,
                                  Int3<index_t> shape,
                                  cvalue_t cvalue = cvalue_t{0}) {
        using mutable_data_t = std::remove_cv_t<data_t>;
        using interpolator_t = Interpolator3D<BORDER_MODE, INTERP_MODE, mutable_data_t, index_t, coord_t, NDIM, TRAITS>;
        return interpolator_t({accessor.data(), accessor.strides()}, shape, cvalue);
    }
}
