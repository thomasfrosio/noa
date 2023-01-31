#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/geometry/Interpolate.hpp"

// This "interpolator" has the same usage that the classic interpolators.
// The difference is that the input is represented by a single constant value,
// more efficiently than what is possible with the classic interpolators.

namespace noa::geometry {
    // Interpolates 2D data.
    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE,
             typename Data, typename Index = int64_t, typename Coord = float>
    class InterpolatorValue2D {
    public:
        static_assert(traits::is_any_v<Index, int32_t, int64_t>);
        static_assert(traits::is_float_or_complex_v<Data> && !std::is_const_v<Data>);
        static_assert(traits::is_float_v<Coord> && !std::is_const_v<Coord>);

        using data_type = Data;
        using index_type = Index;
        using coord_type = Coord;

        using index2_type = Int2<index_type>;
        using coord2_type = Float2<coord_type>;
        using data_or_empty_type = std::conditional_t<BORDER_MODE == BORDER_VALUE, data_type, empty_t>;

    public:
        constexpr InterpolatorValue2D() = default;

        constexpr NOA_HD InterpolatorValue2D(data_type data, index2_type shape, data_type cvalue = data_type{0}) noexcept
                : m_data(data), m_shape(shape) {
            if constexpr (!std::is_empty_v<data_or_empty_type>)
                m_cvalue = cvalue;
        }

        // 2D interpolation on a given batch.
        // The batch is ignored, effectively broadcasting the batch dimension.
        template<typename Int = index_type, typename = std::enable_if_t<traits::is_int_v<Int>>>
        constexpr NOA_HD data_type operator()(coord2_type coordinate, Int = Int{0}) const noexcept {
            if constexpr (BORDER_MODE != BORDER_ZERO && BORDER_MODE != BORDER_VALUE)
                return m_data;

            if constexpr (INTERP_MODE == INTERP_NEAREST) {
                return nearest_(coordinate);
            } else if constexpr (INTERP_MODE == INTERP_LINEAR ||
                                 INTERP_MODE == INTERP_LINEAR_FAST ||
                                 INTERP_MODE == INTERP_COSINE ||
                                 INTERP_MODE == INTERP_COSINE_FAST) {
                return linear_(coordinate);
            } else if constexpr (INTERP_MODE == INTERP_CUBIC ||
                                 INTERP_MODE == INTERP_CUBIC_BSPLINE ||
                                 INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST) {
                return cubic_(coordinate);
            } else {
                static_assert(traits::always_false_v<data_type>);
            }
        }

        constexpr NOA_FHD data_type at(index_type, index_type, index_type) const noexcept {
            return m_data;
        }

        constexpr NOA_FHD data_type at(index_type, index_type) const noexcept {
            return m_data;
        }

    private:
        constexpr NOA_HD data_type nearest_(coord2_type coordinate) const noexcept {
            const index2_type index(::noa::math::round(coordinate));
            if (index[1] < 0 || index[1] >= m_shape[1] ||
                index[0] < 0 || index[0] >= m_shape[0]) {
                if constexpr (BORDER_MODE == BORDER_ZERO)
                    return data_type{0};
                else if constexpr (BORDER_MODE == BORDER_VALUE)
                    return m_cvalue;
            } else {
                return m_data;
            }
        }

        constexpr NOA_HD data_type linear_(coord2_type coordinate) const noexcept {
            const index2_type idx0(::noa::math::floor(coordinate));
            const index2_type idx1(idx0 + 1);
            if (all(idx0 >= 0 && idx1 < m_shape))
                return m_data;

            const bool cond_y[2] = {idx0[0] >= 0 && idx0[0] < m_shape[0], idx1[0] >= 0 && idx1[0] < m_shape[0]};
            const bool cond_x[2] = {idx0[1] >= 0 && idx0[1] < m_shape[1], idx1[1] >= 0 && idx1[1] < m_shape[1]};
            data_type values[4]; // v00, v10, v01, v11
            if constexpr (BORDER_MODE == BORDER_ZERO) {
                values[0] = cond_y[0] && cond_x[0] ? m_data : data_type{0}; // v00
                values[1] = cond_y[0] && cond_x[1] ? m_data : data_type{0}; // v01
                values[2] = cond_y[1] && cond_x[0] ? m_data : data_type{0}; // v10
                values[3] = cond_y[1] && cond_x[1] ? m_data : data_type{0}; // v11
            } else {
                values[0] = cond_y[0] && cond_x[0] ? m_data : m_cvalue;
                values[1] = cond_y[0] && cond_x[1] ? m_data : m_cvalue;
                values[2] = cond_y[1] && cond_x[0] ? m_data : m_cvalue;
                values[3] = cond_y[1] && cond_x[1] ? m_data : m_cvalue;
            }

            const coord2_type fraction{coordinate - static_cast<coord2_type>(idx0)};
            if constexpr (INTERP_MODE == INTERP_COSINE || INTERP_MODE == INTERP_COSINE_FAST)
                return interpolate::cosine2D(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
            else
                return interpolate::lerp2D(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
        }

        constexpr NOA_HD data_type cubic_(coord2_type coordinate) const noexcept {
            const index2_type idx(::noa::math::floor(coordinate));
            if (all(idx - 1 >= 0 && idx + 2 < m_shape))
                return m_data;

            data_type square[4][4]; // [y][x]
            const bool cond_y[4] = {idx[0] - 1 >= 0 && idx[0] - 1 < m_shape[0],
                                    idx[0] + 0 >= 0 && idx[0] + 0 < m_shape[0],
                                    idx[0] + 1 >= 0 && idx[0] + 1 < m_shape[0],
                                    idx[0] + 2 >= 0 && idx[0] + 2 < m_shape[0]};
            const bool cond_x[4] = {idx[1] - 1 >= 0 && idx[1] - 1 < m_shape[1],
                                    idx[1] + 0 >= 0 && idx[1] + 0 < m_shape[1],
                                    idx[1] + 1 >= 0 && idx[1] + 1 < m_shape[1],
                                    idx[1] + 2 >= 0 && idx[1] + 2 < m_shape[1]};

            data_type oob_value;
            if constexpr (BORDER_MODE == BORDER_ZERO)
                oob_value = data_type{0};
            else
                oob_value = m_cvalue;

            for (index_type j = 0; j < 4; ++j)
                for (index_type i = 0; i < 4; ++i)
                    square[j][i] = cond_x[i] && cond_y[j] ? m_data : oob_value;

            const coord2_type fraction{coordinate - static_cast<coord2_type>(idx)};
            if constexpr (INTERP_MODE == INTERP_CUBIC_BSPLINE || INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST)
                return interpolate::cubicBSpline2D(square, fraction[1], fraction[0]);
            else
                return interpolate::cubic2D(square, fraction[1], fraction[0]);
        }

    private:
        data_type m_data{};
        index2_type m_shape{};
        data_or_empty_type m_cvalue{};
    };

    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE, typename Coord = float,
             typename Data, typename Index,
             typename CValue = traits::remove_ref_cv_t<Data>,
             typename = std::enable_if_t<traits::is_almost_same_v<Data, CValue>>>
    constexpr auto interpolatorValue2D(Data data,
                                       Int2<Index> shape,
                                       CValue cvalue = CValue{0}) {
        using mutable_data_t = std::remove_cv_t<Data>;
        using interpolator_t = InterpolatorValue2D<BORDER_MODE, INTERP_MODE, mutable_data_t, Index, Coord>;
        return interpolator_t(data, shape, cvalue);
    }
}

namespace noa::geometry {
    // Interpolates 3D data.
    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE,
             typename Data, typename Index = int64_t, typename Coord = float>
    class InterpolatorValue3D {
    public:
        static_assert(traits::is_any_v<Index, int32_t, int64_t>);
        static_assert(traits::is_float_or_complex_v<Data> && !std::is_const_v<Data>);
        static_assert(traits::is_float_v<Coord> && !std::is_const_v<Coord>);

        using data_type = Data;
        using index_type = Index;
        using coord_type = Coord;

        using index3_type = Int3<index_type>;
        using coord3_type = Float3<coord_type>;
        using data_or_empty_type = std::conditional_t<BORDER_MODE == BORDER_VALUE, data_type, empty_t>;

    public:
        constexpr InterpolatorValue3D() = default;

        constexpr NOA_HD InterpolatorValue3D(data_type data, index3_type shape, data_type cvalue = data_type{0}) noexcept
                : m_data(data), m_shape(shape) {
            if constexpr (!std::is_empty_v<data_or_empty_type>)
                m_cvalue = cvalue;
        }

        // 3D interpolation on a given batch.
        // The batch is ignored if the accessor is 3D, effectively broadcasting the batch dimension.
        template<typename Int = index_type, typename = std::enable_if_t<traits::is_int_v<Int>>>
        constexpr NOA_HD data_type operator()(coord3_type coordinate, Int = Int{0}) const noexcept {
            if constexpr (BORDER_MODE != BORDER_ZERO && BORDER_MODE != BORDER_VALUE)
                return m_data;

            if constexpr (INTERP_MODE == INTERP_NEAREST) {
                return nearest_(coordinate);
            } else if constexpr (INTERP_MODE == INTERP_LINEAR ||
                                 INTERP_MODE == INTERP_LINEAR_FAST ||
                                 INTERP_MODE == INTERP_COSINE ||
                                 INTERP_MODE == INTERP_COSINE_FAST) {
                return linear_(coordinate);
            } else if constexpr (INTERP_MODE == INTERP_CUBIC ||
                                 INTERP_MODE == INTERP_CUBIC_BSPLINE ||
                                 INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST) {
                return cubic_(coordinate);
            } else {
                static_assert(traits::always_false_v<data_type>);
            }
        }

        constexpr NOA_FHD data_type at(index_type, index_type, index_type, index_type) const noexcept {
            return m_data;
        }

        constexpr NOA_FHD data_type at(index_type, index_type, index_type) const noexcept {
            return m_data;
        }

    private:
        constexpr NOA_HD data_type nearest_(coord3_type coordinate) const noexcept {
            const index3_type index(noa::math::round(coordinate));
            if constexpr (BORDER_MODE == BORDER_ZERO) {
                if (index[2] < 0 || index[2] >= m_shape[2] ||
                    index[1] < 0 || index[1] >= m_shape[1] ||
                    index[0] < 0 || index[0] >= m_shape[0])
                    return data_type{0};
                else
                    return m_data;
            } else if constexpr (BORDER_MODE == BORDER_VALUE) {
                if (index[2] < 0 || index[2] >= m_shape[2] ||
                    index[1] < 0 || index[1] >= m_shape[1] ||
                    index[0] < 0 || index[0] >= m_shape[0])
                    return m_cvalue;
                else
                    return m_data;
            }
        }

        constexpr NOA_HD data_type linear_(coord3_type coordinate) const noexcept {
            index3_type idx[2];
            idx[0] = index3_type(noa::math::floor(coordinate));
            idx[1] = idx[0] + 1;
            if (all(idx[0] >= 0 && idx[1] < m_shape))
                return m_data;

            data_type values[8];
            const bool cond_z[2] = {idx[0][0] >= 0 && idx[0][0] < m_shape[0], idx[1][0] >= 0 && idx[1][0] < m_shape[0]};
            const bool cond_y[2] = {idx[0][1] >= 0 && idx[0][1] < m_shape[1], idx[1][1] >= 0 && idx[1][1] < m_shape[1]};
            const bool cond_x[2] = {idx[0][2] >= 0 && idx[0][2] < m_shape[2], idx[1][2] >= 0 && idx[1][2] < m_shape[2]};

            // TODO Might be more efficient to do two 2D interpolations and a final 1D...
            data_type cval;
            if constexpr (BORDER_MODE == BORDER_ZERO)
                cval = data_type{0};
            else
                cval = m_cvalue;
            values[0] = cond_z[0] && cond_y[0] && cond_x[0] ? m_data : cval; // v000
            values[1] = cond_z[0] && cond_y[0] && cond_x[1] ? m_data : cval; // v001
            values[2] = cond_z[0] && cond_y[1] && cond_x[0] ? m_data : cval; // v010
            values[3] = cond_z[0] && cond_y[1] && cond_x[1] ? m_data : cval; // v011
            values[4] = cond_z[1] && cond_y[0] && cond_x[0] ? m_data : cval; // v100
            values[5] = cond_z[1] && cond_y[0] && cond_x[1] ? m_data : cval; // v101
            values[6] = cond_z[1] && cond_y[1] && cond_x[0] ? m_data : cval; // v110
            values[7] = cond_z[1] && cond_y[1] && cond_x[1] ? m_data : cval; // v111

            const coord3_type fraction{coordinate - static_cast<coord3_type>(idx[0])};
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

        constexpr NOA_HD data_type cubic_(coord3_type coordinate) const noexcept {
            const index3_type idx(noa::math::floor(coordinate));
            if (all(idx - 1 >= 0 && idx + 2 < m_shape))
                return m_data;

            data_type values[4][4][4]; // [z][y][x]
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
            data_type cval;
            if constexpr (BORDER_MODE == BORDER_ZERO)
                cval = data_type{0};
            else
                cval = m_cvalue;
            for (index_type i = 0; i < 4; ++i)
                for (index_type j = 0; j < 4; ++j)
                    for (index_type k = 0; k < 4; ++k)
                        values[i][j][k] = cond_z[i] && cond_y[j] && cond_x[k] ? m_data : cval;

            const coord3_type fraction{coordinate - static_cast<coord3_type>(idx)};
            if constexpr (INTERP_MODE == INTERP_CUBIC_BSPLINE || INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST)
                return interpolate::cubicBSpline3D(values, fraction[2], fraction[1], fraction[0]);
            else
                return interpolate::cubic3D(values, fraction[2], fraction[1], fraction[0]);
        }

    private:
        data_type m_data{};
        index3_type m_shape{};
        data_or_empty_type m_cvalue{};
    };

    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE, typename Coord = float,
             typename Data, typename Index,
             typename CValue = traits::remove_ref_cv_t<Data>,
             typename = std::enable_if_t<traits::is_almost_same_v<Data, CValue>>>
    constexpr auto interpolatorValue3D(Data data,
                                       Int3<Index> shape,
                                       CValue cvalue = CValue{0}) {
        using mutable_data_t = std::remove_cv_t<Data>;
        using interpolator_t = InterpolatorValue3D<BORDER_MODE, INTERP_MODE, mutable_data_t, Index, Coord>;
        return interpolator_t(data, shape, cvalue);
    }
}
