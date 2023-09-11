#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/geometry/Interpolate.hpp"
#include "noa/core/traits/Interpolator.hpp"

// This "interpolator" has the same usage that the classic interpolators.
// The difference is that the input is represented by a single constant value,
// more efficiently than what is possible with the classic interpolators.

namespace noa::geometry {
    // Interpolates 2D data.
    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE,
             typename Value, typename Index, typename Coord>
    class InterpolatorValue2D {
    public:
        static_assert(nt::is_any_v<Index, i32, i64>);
        static_assert(nt::is_real_or_complex_v<Value> && !std::is_const_v<Value>);
        static_assert(nt::is_real_v<Coord> && !std::is_const_v<Coord>);

        using value_type = Value;
        using mutable_value_type = Value;
        using index_type = Index;
        using coord_type = Coord;

        using index2_type = Vec2<index_type>;
        using coord2_type = Vec2<coord_type>;
        using shape2_type = Shape2<index_type>;
        using value_or_empty_type = std::conditional_t<BORDER_MODE == BorderMode::VALUE, value_type, Empty>;

    public:
        constexpr InterpolatorValue2D() = default;

        constexpr NOA_HD InterpolatorValue2D(
                value_type data, shape2_type shape,
                value_type cvalue = value_type{0}) noexcept
                : m_data(data), m_shape(shape.vec()) {
            if constexpr (!std::is_empty_v<value_or_empty_type>)
                m_cvalue = cvalue;
        }

        // 2D interpolation on a given batch.
        // The batch is ignored, effectively broadcasting the batch dimension.
        template<typename Int = index_type, typename = std::enable_if_t<nt::is_int_v<Int>>>
        constexpr NOA_HD value_type operator()(coord2_type coordinate, Int = Int{0}) const noexcept {
            if constexpr (BORDER_MODE != BorderMode::ZERO && BORDER_MODE != BorderMode::VALUE)
                return m_data;

            if constexpr (INTERP_MODE == InterpMode::NEAREST) {
                return nearest_(coordinate);
            } else if constexpr (INTERP_MODE == InterpMode::LINEAR ||
                                 INTERP_MODE == InterpMode::LINEAR_FAST ||
                                 INTERP_MODE == InterpMode::COSINE ||
                                 INTERP_MODE == InterpMode::COSINE_FAST) {
                return linear_(coordinate);
            } else if constexpr (INTERP_MODE == InterpMode::CUBIC ||
                                 INTERP_MODE == InterpMode::CUBIC_BSPLINE ||
                                 INTERP_MODE == InterpMode::CUBIC_BSPLINE_FAST) {
                return cubic_(coordinate);
            } else {
                static_assert(nt::always_false_v<value_type>);
            }
        }

        constexpr NOA_FHD value_type at(index_type, index_type, index_type) const noexcept {
            return m_data;
        }

        constexpr NOA_FHD value_type at(index_type, index_type) const noexcept {
            return m_data;
        }

    private:
        constexpr NOA_HD value_type nearest_(coord2_type coordinate) const noexcept {
            const index2_type index(noa::math::round(coordinate));
            if (index[1] < 0 || index[1] >= m_shape[1] ||
                index[0] < 0 || index[0] >= m_shape[0]) {
                if constexpr (BORDER_MODE == BorderMode::ZERO)
                    return value_type{0};
                else if constexpr (BORDER_MODE == BorderMode::VALUE)
                    return m_cvalue;
            } else {
                return m_data;
            }
        }

        constexpr NOA_HD value_type linear_(coord2_type coordinate) const noexcept {
            const index2_type idx0(noa::math::floor(coordinate));
            const index2_type idx1(idx0 + 1);
            if (noa::all(idx0 >= 0 && idx1 < m_shape))
                return m_data;

            const bool cond_y[2] = {idx0[0] >= 0 && idx0[0] < m_shape[0], idx1[0] >= 0 && idx1[0] < m_shape[0]};
            const bool cond_x[2] = {idx0[1] >= 0 && idx0[1] < m_shape[1], idx1[1] >= 0 && idx1[1] < m_shape[1]};
            value_type values[4]; // v00, v10, v01, v11
            if constexpr (BORDER_MODE == BorderMode::ZERO) {
                values[0] = cond_y[0] && cond_x[0] ? m_data : value_type{0}; // v00
                values[1] = cond_y[0] && cond_x[1] ? m_data : value_type{0}; // v01
                values[2] = cond_y[1] && cond_x[0] ? m_data : value_type{0}; // v10
                values[3] = cond_y[1] && cond_x[1] ? m_data : value_type{0}; // v11
            } else {
                values[0] = cond_y[0] && cond_x[0] ? m_data : m_cvalue;
                values[1] = cond_y[0] && cond_x[1] ? m_data : m_cvalue;
                values[2] = cond_y[1] && cond_x[0] ? m_data : m_cvalue;
                values[3] = cond_y[1] && cond_x[1] ? m_data : m_cvalue;
            }

            const coord2_type fraction{coordinate - static_cast<coord2_type>(idx0)};
            if constexpr (INTERP_MODE == InterpMode::COSINE || INTERP_MODE == InterpMode::COSINE_FAST)
                return interpolate::cosine_2d(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
            else
                return interpolate::lerp_2d(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
        }

        constexpr NOA_HD value_type cubic_(coord2_type coordinate) const noexcept {
            const index2_type idx(noa::math::floor(coordinate));
            if (noa::all(idx - 1 >= 0 && idx + 2 < m_shape))
                return m_data;

            value_type square[4][4]; // [y][x]
            const bool cond_y[4] = {idx[0] - 1 >= 0 && idx[0] - 1 < m_shape[0],
                                    idx[0] + 0 >= 0 && idx[0] + 0 < m_shape[0],
                                    idx[0] + 1 >= 0 && idx[0] + 1 < m_shape[0],
                                    idx[0] + 2 >= 0 && idx[0] + 2 < m_shape[0]};
            const bool cond_x[4] = {idx[1] - 1 >= 0 && idx[1] - 1 < m_shape[1],
                                    idx[1] + 0 >= 0 && idx[1] + 0 < m_shape[1],
                                    idx[1] + 1 >= 0 && idx[1] + 1 < m_shape[1],
                                    idx[1] + 2 >= 0 && idx[1] + 2 < m_shape[1]};

            value_type oob_value;
            if constexpr (BORDER_MODE == BorderMode::ZERO)
                oob_value = value_type{0};
            else
                oob_value = m_cvalue;

            for (index_type j = 0; j < 4; ++j)
                for (index_type i = 0; i < 4; ++i)
                    square[j][i] = cond_x[i] && cond_y[j] ? m_data : oob_value;

            const coord2_type fraction{coordinate - static_cast<coord2_type>(idx)};
            if constexpr (INTERP_MODE == InterpMode::CUBIC_BSPLINE || INTERP_MODE == InterpMode::CUBIC_BSPLINE_FAST)
                return interpolate::cubic_bspline_2d(square, fraction[1], fraction[0]);
            else
                return interpolate::cubic_2d(square, fraction[1], fraction[0]);
        }

    private:
        value_type m_data{};
        index2_type m_shape{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_cvalue{};
    };

    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE, typename Coord = f32,
             typename Value, typename Index,
             typename CValue = nt::remove_ref_cv_t<Value>,
             typename = std::enable_if_t<nt::is_almost_same_v<Value, CValue>>>
    constexpr auto interpolator_value_2d(
            Value data,
            const Shape2<Index>& shape,
            CValue cvalue = CValue{0}
    ) {
        using mutable_data_t = std::remove_cv_t<Value>;
        using interpolator_t = InterpolatorValue2D<BORDER_MODE, INTERP_MODE, mutable_data_t, Index, Coord>;
        return interpolator_t(data, shape, cvalue);
    }
}

namespace noa::geometry {
    // Interpolates 3D data.
    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE,
             typename Value, typename Index, typename Coord>
    class InterpolatorValue3D {
    public:
        static_assert(nt::is_any_v<Index, i32, i64>);
        static_assert(nt::is_real_or_complex_v<Value> && !std::is_const_v<Value>);
        static_assert(nt::is_real_v<Coord> && !std::is_const_v<Coord>);

        using value_type = Value;
        using mutable_value_type = Value;
        using index_type = Index;
        using coord_type = Coord;

        using index3_type = Vec3<index_type>;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;
        using value_or_empty_type = std::conditional_t<BORDER_MODE == BorderMode::VALUE, value_type, Empty>;

    public:
        constexpr InterpolatorValue3D() = default;

        constexpr NOA_HD InterpolatorValue3D(
                value_type data, shape3_type shape,
                value_type cvalue = value_type{0}) noexcept
                : m_data(data), m_shape(shape.vec()) {
            if constexpr (!std::is_empty_v<value_or_empty_type>)
                m_cvalue = cvalue;
        }

        // 3D interpolation on a given batch.
        // The batch is ignored if the accessor is 3D, effectively broadcasting the batch dimension.
        template<typename Int = index_type, typename = std::enable_if_t<nt::is_int_v<Int>>>
        constexpr NOA_HD value_type operator()(coord3_type coordinate, Int = Int{0}) const noexcept {
            if constexpr (BORDER_MODE != BorderMode::ZERO && BORDER_MODE != BorderMode::VALUE)
                return m_data;

            if constexpr (INTERP_MODE == InterpMode::NEAREST) {
                return nearest_(coordinate);
            } else if constexpr (INTERP_MODE == InterpMode::LINEAR ||
                                 INTERP_MODE == InterpMode::LINEAR_FAST ||
                                 INTERP_MODE == InterpMode::COSINE ||
                                 INTERP_MODE == InterpMode::COSINE_FAST) {
                return linear_(coordinate);
            } else if constexpr (INTERP_MODE == InterpMode::CUBIC ||
                                 INTERP_MODE == InterpMode::CUBIC_BSPLINE ||
                                 INTERP_MODE == InterpMode::CUBIC_BSPLINE_FAST) {
                return cubic_(coordinate);
            } else {
                static_assert(nt::always_false_v<value_type>);
            }
        }

        constexpr NOA_FHD value_type at(index_type, index_type, index_type, index_type) const noexcept {
            return m_data;
        }

        constexpr NOA_FHD value_type at(index_type, index_type, index_type) const noexcept {
            return m_data;
        }

    private:
        constexpr NOA_HD value_type nearest_(coord3_type coordinate) const noexcept {
            const index3_type index(noa::math::round(coordinate));
            if constexpr (BORDER_MODE == BorderMode::ZERO) {
                if (index[2] < 0 || index[2] >= m_shape[2] ||
                    index[1] < 0 || index[1] >= m_shape[1] ||
                    index[0] < 0 || index[0] >= m_shape[0])
                    return value_type{0};
                else
                    return m_data;
            } else if constexpr (BORDER_MODE == BorderMode::VALUE) {
                if (index[2] < 0 || index[2] >= m_shape[2] ||
                    index[1] < 0 || index[1] >= m_shape[1] ||
                    index[0] < 0 || index[0] >= m_shape[0])
                    return m_cvalue;
                else
                    return m_data;
            }
        }

        constexpr NOA_HD value_type linear_(coord3_type coordinate) const noexcept {
            index3_type idx[2];
            idx[0] = index3_type(noa::math::floor(coordinate));
            idx[1] = idx[0] + 1;
            if (noa::all(idx[0] >= 0 && idx[1] < m_shape))
                return m_data;

            value_type values[8];
            const bool cond_z[2] = {idx[0][0] >= 0 && idx[0][0] < m_shape[0], idx[1][0] >= 0 && idx[1][0] < m_shape[0]};
            const bool cond_y[2] = {idx[0][1] >= 0 && idx[0][1] < m_shape[1], idx[1][1] >= 0 && idx[1][1] < m_shape[1]};
            const bool cond_x[2] = {idx[0][2] >= 0 && idx[0][2] < m_shape[2], idx[1][2] >= 0 && idx[1][2] < m_shape[2]};

            // TODO Might be more efficient to do two 2D interpolations and a final 1D...
            value_type cval;
            if constexpr (BORDER_MODE == BorderMode::ZERO)
                cval = value_type{0};
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
            if constexpr (INTERP_MODE == InterpMode::COSINE || INTERP_MODE == InterpMode::CUBIC_BSPLINE) {
                return interpolate::cosine_3d(values[0], values[1], values[2], values[3],
                                              values[4], values[5], values[6], values[7],
                                              fraction[2], fraction[1], fraction[0]);
            } else {
                return interpolate::lerp_3d(values[0], values[1], values[2], values[3],
                                            values[4], values[5], values[6], values[7],
                                            fraction[2], fraction[1], fraction[0]);
            }
        }

        constexpr NOA_HD value_type cubic_(coord3_type coordinate) const noexcept {
            const index3_type idx(noa::math::floor(coordinate));
            if (noa::all(idx - 1 >= 0 && idx + 2 < m_shape))
                return m_data;

            value_type values[4][4][4]; // [z][y][x]
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
            value_type cval;
            if constexpr (BORDER_MODE == BorderMode::ZERO)
                cval = value_type{0};
            else
                cval = m_cvalue;
            for (index_type i = 0; i < 4; ++i)
                for (index_type j = 0; j < 4; ++j)
                    for (index_type k = 0; k < 4; ++k)
                        values[i][j][k] = cond_z[i] && cond_y[j] && cond_x[k] ? m_data : cval;

            const coord3_type fraction{coordinate - static_cast<coord3_type>(idx)};
            if constexpr (INTERP_MODE == InterpMode::CUBIC_BSPLINE || INTERP_MODE == InterpMode::CUBIC_BSPLINE_FAST)
                return interpolate::cubic_bspline_3d(values, fraction[2], fraction[1], fraction[0]);
            else
                return interpolate::cubic_3d(values, fraction[2], fraction[1], fraction[0]);
        }

    private:
        value_type m_data{};
        index3_type m_shape{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_cvalue{};
    };

    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE, typename Coord = f32,
             typename Value, typename Index,
             typename CValue = nt::remove_ref_cv_t<Value>,
             typename = std::enable_if_t<nt::is_almost_same_v<Value, CValue>>>
    constexpr auto interpolator_value_3d(
            Value data,
            const Shape3<Index>& shape,
            CValue cvalue = CValue{0}
    ) {
        using mutable_data_t = std::remove_cv_t<Value>;
        using interpolator_t = InterpolatorValue3D<BORDER_MODE, INTERP_MODE, mutable_data_t, Index, Coord>;
        return interpolator_t(data, shape, cvalue);
    }
}

namespace noa::traits {
    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE,
             typename Value, typename Index, typename Coord>
    struct proclaim_is_interpolator_2d<
            noa::geometry::InterpolatorValue2D<BORDER_MODE, INTERP_MODE, Value, Index, Coord>
    > : std::true_type {};

    template<BorderMode BORDER_MODE, InterpMode INTERP_MODE,
            typename Value, typename Index, typename Coord>
    struct proclaim_is_interpolator_3d<
            noa::geometry::InterpolatorValue3D<BORDER_MODE, INTERP_MODE, Value, Index, Coord>
    > : std::true_type {};
}
