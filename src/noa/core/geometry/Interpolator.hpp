#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Enums.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/geometry/Interpolate.hpp"
#include "noa/core/indexing/Offset.hpp"

/// Features:
///     - 2d or 3d interpolation on real or complex data, with various interpolation and border modes.
///     - The data is wrapped into an Accessor, thus supporting different memory layouts and pointer traits.
///     - The data can be batched. If the input is not batched, the batch dimension is automatically broadcast.
///     - Interpolation can be skipped using the .at() member functions, in case integral coordinates are used
///       (in which case the interpolators are simple wrappers around the accessors).
///     - The coordinate and index/offset types are templated, e.g. interpolation can be single or double precision.
///     - These interpolators can work on the CPU or GPU.
///     - The input can be a single value (passed on the stack), using InterpolatorValue2d or InterpolatorValue3d.
///       This constant value is stored in the interpolator and the interpolation logic is simplified whenever possible.
///       Note that this is just an optimization since the accessors can represent a fully broadcast input too.
///
/// Out-of-bounds:
///     One of the main differences between these interpolations and what we can find in other cryoEM packages,
///     is that the interpolation window can be partially out-of-bound (OOB), that is, elements that are OOB
///     are replaced according to a Border. cryoEM packages usually check that all elements are in-bound
///     and if there's even one element OOB, they don't interpolate.
///     Note: The interpolators below are for real space interpolation, or redundant and centered Fourier transforms.
///
/// Coordinate system:
///     The coordinate system matches the indexing. This is quite standard.
///     For instance, the first data sample at index 0 is located at the coordinate 0 and the coordinate 0.5
///     is just in between the first and second element. As such, the fractional part of the coordinate
///     corresponds to the ratio/weight used by the interpolation function. In other words,
///     the coordinate system locates the data between -0.5 and N-1 + 0.5.

// TODO Merge InterpolatorValue with Interpolator, when Accessor will have an empty mode?

namespace noa::geometry {
    /// Interpolates 2d data.
    template<Border BORDER, Interp INTERP,
             typename Value, typename Offset, typename Coord,
             size_t ACCESSOR_NDIM = 2,
             PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED>
    class Interpolator2d {
    public:
        static_assert(nt::is_any_v<Offset, int32_t, int64_t, uint32_t, uint64_t>);
        static_assert(nt::is_real_or_complex_v<Value> && !std::is_const_v<Value>);
        static_assert(nt::is_real_v<Coord> && !std::is_const_v<Coord>);
        static_assert(ACCESSOR_NDIM == 2 || ACCESSOR_NDIM == 3);

        using value_type = Value;
        using mutable_value_type = Value;
        using offset_type = Offset;
        using coord_type = Coord;

        using index_type = std::make_signed_t<offset_type>;
        using index2_type = Vec2<index_type>;
        using coord2_type = Vec2<coord_type>;
        using shape2_type = Shape2<index_type>;
        using value_or_empty_type = std::conditional_t<BORDER == Border::VALUE, value_type, Empty>;
        using accessor_type = Accessor<const value_type, ACCESSOR_NDIM, offset_type, PointerTrait, StridesTrait>;

    public:
        constexpr Interpolator2d() = default;

        NOA_HD constexpr Interpolator2d(
                accessor_type data,
                shape2_type shape,
                value_type cvalue = value_type{0}
        ) noexcept
                : m_data(data), m_shape(shape.vec)
        {
            if constexpr (!std::is_empty_v<value_or_empty_type>)
                m_cvalue = cvalue;
        }

        /// 2d interpolation on a given batch.
        /// The batch is ignored if the accessor is 2d, effectively broadcasting the batch dimension.
        template<typename Int = index_type, typename = std::enable_if_t<nt::is_int_v<Int>>>
        NOA_HD constexpr value_type operator()(coord2_type coordinate, Int batch = Int{0}) const noexcept {
            if constexpr (ACCESSOR_NDIM == 2) {
                if constexpr (INTERP == Interp::NEAREST) {
                    return nearest_(m_data, coordinate);
                } else if constexpr (INTERP == Interp::LINEAR ||
                                     INTERP == Interp::LINEAR_FAST ||
                                     INTERP == Interp::COSINE ||
                                     INTERP == Interp::COSINE_FAST) {
                    return linear_(m_data, coordinate);
                } else if constexpr (INTERP == Interp::CUBIC ||
                                     INTERP == Interp::CUBIC_BSPLINE ||
                                     INTERP == Interp::CUBIC_BSPLINE_FAST) {
                    return cubic_(m_data, coordinate);
                } else {
                    static_assert(nt::always_false_v<value_type>);
                }
            } else {
                if constexpr (INTERP == Interp::NEAREST) {
                    return nearest_(m_data[batch], coordinate);
                } else if constexpr (INTERP == Interp::LINEAR ||
                                     INTERP == Interp::LINEAR_FAST ||
                                     INTERP == Interp::COSINE ||
                                     INTERP == Interp::COSINE_FAST) {
                    return linear_(m_data[batch], coordinate);
                } else if constexpr (INTERP == Interp::CUBIC ||
                                     INTERP == Interp::CUBIC_BSPLINE ||
                                     INTERP == Interp::CUBIC_BSPLINE_FAST) {
                    return cubic_(m_data[batch], coordinate);
                } else {
                    static_assert(nt::always_false_v<value_type>);
                }
            }
        }

        template<typename Void = void, typename = std::enable_if_t<ACCESSOR_NDIM == 3 && std::is_void_v<Void>>>
        NOA_FHD constexpr value_type at(index_type batch, index_type y, index_type x) const noexcept {
            return m_data(batch, y, x);
        }

        template<typename Void = void, typename = std::enable_if_t<ACCESSOR_NDIM == 2 && std::is_void_v<Void>>>
        NOA_FHD constexpr value_type at(index_type y, index_type x) const noexcept {
            return m_data(y, x);
        }

    private:
        template<typename Accessor2d>
        NOA_HD constexpr value_type nearest_(Accessor2d accessor, coord2_type coordinate) const noexcept {
            static_assert(Accessor2d::COUNT == 2);
            const index2_type index = round(coordinate).template as<index_type>();

            if constexpr (BORDER == Border::ZERO) {
                if (index[1] < 0 || index[1] >= m_shape[1] ||
                    index[0] < 0 || index[0] >= m_shape[0])
                    return value_type{0};
                else
                    return accessor(index[0], index[1]);
            } else if constexpr (BORDER == Border::VALUE) {
                if (index[1] < 0 || index[1] >= m_shape[1] ||
                    index[0] < 0 || index[0] >= m_shape[0])
                    return m_cvalue;
                else
                    return accessor(index[0], index[1]);
            } else if constexpr (BORDER == Border::CLAMP || BORDER == Border::PERIODIC ||
                                 BORDER == Border::MIRROR || BORDER == Border::REFLECT) {
                return accessor(offset_at<BORDER>(index[0], m_shape[0]),
                                offset_at<BORDER>(index[1], m_shape[1]));
            } else {
                static_assert(nt::always_false_v<value_type>);
            }
        }

        template<typename Accessor2d>
        NOA_HD constexpr value_type linear_(Accessor2d accessor, coord2_type coordinate) const noexcept {
            static_assert(Accessor2d::COUNT == 2);
            const index2_type idx0 = coordinate.template as<index_type>();
            const index2_type idx1 = idx0 + 1;

            value_type values[4]; // v00, v10, v01, v11
            if constexpr (BORDER == Border::ZERO || BORDER == Border::VALUE) {
                const bool cond_y[2] = {idx0[0] >= 0 && idx0[0] < m_shape[0], idx1[0] >= 0 && idx1[0] < m_shape[0]};
                const bool cond_x[2] = {idx0[1] >= 0 && idx0[1] < m_shape[1], idx1[1] >= 0 && idx1[1] < m_shape[1]};

                if constexpr (BORDER == Border::ZERO) {
                    values[0] = cond_y[0] && cond_x[0] ? accessor(idx0[0], idx0[1]) : value_type{0}; // v00
                    values[1] = cond_y[0] && cond_x[1] ? accessor(idx0[0], idx1[1]) : value_type{0}; // v01
                    values[2] = cond_y[1] && cond_x[0] ? accessor(idx1[0], idx0[1]) : value_type{0}; // v10
                    values[3] = cond_y[1] && cond_x[1] ? accessor(idx1[0], idx1[1]) : value_type{0}; // v11
                } else {
                    values[0] = cond_y[0] && cond_x[0] ? accessor(idx0[0], idx0[1]) : m_cvalue;
                    values[1] = cond_y[0] && cond_x[1] ? accessor(idx0[0], idx1[1]) : m_cvalue;
                    values[2] = cond_y[1] && cond_x[0] ? accessor(idx1[0], idx0[1]) : m_cvalue;
                    values[3] = cond_y[1] && cond_x[1] ? accessor(idx1[0], idx1[1]) : m_cvalue;
                }

            } else if constexpr (BORDER == Border::CLAMP || BORDER == Border::PERIODIC ||
                                 BORDER == Border::MIRROR || BORDER == Border::REFLECT) {
                const index_type tmp[4] = {offset_at<BORDER>(idx0[1], m_shape[1]),
                                           offset_at<BORDER>(idx1[1], m_shape[1]),
                                           offset_at<BORDER>(idx0[0], m_shape[0]),
                                           offset_at<BORDER>(idx1[0], m_shape[0])};
                values[0] = accessor(tmp[2], tmp[0]); // v00
                values[1] = accessor(tmp[2], tmp[1]); // v01
                values[2] = accessor(tmp[3], tmp[0]); // v10
                values[3] = accessor(tmp[3], tmp[1]); // v11

            } else {
                static_assert(nt::always_false_v<value_type>);
            }

            const coord2_type fraction{coordinate - static_cast<coord2_type>(idx0)};
            if constexpr (INTERP == Interp::COSINE || INTERP == Interp::COSINE_FAST)
                return interpolate_cosine_2d(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
            else
                return lerp_2d(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
        }

        template<typename Accessor2d>
        NOA_HD constexpr value_type cubic_(Accessor2d accessor, coord2_type coordinate) const noexcept {
            static_assert(Accessor2d::COUNT == 2);
            const index2_type idx = floor(coordinate).template as<index_type>();

            value_type square[4][4]; // [y][x]
            if constexpr (BORDER == Border::ZERO || BORDER == Border::VALUE) {
                const bool cond_y[4] = {idx[0] - 1 >= 0 && idx[0] - 1 < m_shape[0],
                                        idx[0] + 0 >= 0 && idx[0] + 0 < m_shape[0],
                                        idx[0] + 1 >= 0 && idx[0] + 1 < m_shape[0],
                                        idx[0] + 2 >= 0 && idx[0] + 2 < m_shape[0]};
                const bool cond_x[4] = {idx[1] - 1 >= 0 && idx[1] - 1 < m_shape[1],
                                        idx[1] + 0 >= 0 && idx[1] + 0 < m_shape[1],
                                        idx[1] + 1 >= 0 && idx[1] + 1 < m_shape[1],
                                        idx[1] + 2 >= 0 && idx[1] + 2 < m_shape[1]};
                constexpr index_type offset[4] = {-1, 0, 1, 2};
                for (index_type j = 0; j < 4; ++j) {
                    const index_type idx_y = idx[0] + offset[j];
                    for (index_type i = 0; i < 4; ++i) {
                        const index_type idx_x = idx[1] + offset[i];
                        if constexpr (BORDER == Border::ZERO)
                            square[j][i] = cond_x[i] && cond_y[j] ? accessor(idx_y, idx_x) : value_type{0};
                        else
                            square[j][i] = cond_x[i] && cond_y[j] ? accessor(idx_y, idx_x) : m_cvalue;
                    }
                }
            } else if constexpr (BORDER == Border::CLAMP || BORDER == Border::PERIODIC ||
                                 BORDER == Border::MIRROR || BORDER == Border::REFLECT) {
                const index_type tmp_y[4] = {offset_at<BORDER>(idx[0] - 1, m_shape[0]),
                                             offset_at<BORDER>(idx[0] + 0, m_shape[0]),
                                             offset_at<BORDER>(idx[0] + 1, m_shape[0]),
                                             offset_at<BORDER>(idx[0] + 2, m_shape[0])};
                const index_type tmp_x[4] = {offset_at<BORDER>(idx[1] - 1, m_shape[1]),
                                             offset_at<BORDER>(idx[1] + 0, m_shape[1]),
                                             offset_at<BORDER>(idx[1] + 1, m_shape[1]),
                                             offset_at<BORDER>(idx[1] + 2, m_shape[1])};
                for (index_type j = 0; j < 4; ++j)
                    for (index_type i = 0; i < 4; ++i)
                        square[j][i] = accessor(tmp_y[j], tmp_x[i]);

            } else {
                static_assert(nt::always_false_v<value_type>);
            }
            const coord2_type fraction{coordinate - static_cast<coord2_type>(idx)};
            if constexpr (INTERP == Interp::CUBIC_BSPLINE || INTERP == Interp::CUBIC_BSPLINE_FAST)
                return interpolate_cubic_bspline_2d(square, fraction[1], fraction[0]);
            else
                return interpolate_cubic_2d(square, fraction[1], fraction[0]);
        }

    private:
        accessor_type m_data{};
        index2_type m_shape{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_cvalue{};
    };

    template<Border BORDER, Interp INTERP, typename Coord = float,
             typename Value, typename Offset, typename Index, size_t NDIM,
             PointerTraits PointerTrait, StridesTraits StridesTrait,
             typename CValue = nt::remove_ref_cv_t<Value>,
             typename = std::enable_if_t<nt::is_almost_same_v<Value, CValue> &&
                                         std::is_same_v<std::make_signed_t<Offset>, Index>>>
    constexpr auto interpolator_2d(
            const Accessor<Value, NDIM, Offset, PointerTrait, StridesTrait>& accessor,
            Shape2<Index> shape,
            CValue cvalue = CValue{0}
    ) {
        using mutable_value_t = std::remove_cv_t<Value>;
        using interpolator_t = Interpolator2d<
                BORDER, INTERP, mutable_value_t,
                Offset, Coord, NDIM, PointerTrait, StridesTrait>;
        return interpolator_t(accessor, shape, cvalue);
    }
}

namespace noa::geometry {
    /// Interpolates 3d data.
    template<Border BORDER, Interp INTERP,
             typename Value, typename Offset, typename Coord,
             size_t ACCESSOR_NDIM = 3,
             PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED>
    class Interpolator3d {
    public:
        static_assert(nt::is_any_v<Offset, int32_t, int64_t, uint32_t, uint64_t>);
        static_assert(nt::is_real_or_complex_v<Value> && !std::is_const_v<Value>);
        static_assert(nt::is_real_v<Coord> && !std::is_const_v<Coord>);
        static_assert(ACCESSOR_NDIM == 3 || ACCESSOR_NDIM == 4);

        using value_type = Value;
        using mutable_value_type = Value;
        using offset_type = Offset;
        using coord_type = Coord;

        using index_type = std::make_signed_t<offset_type>;
        using index3_type = Vec3<index_type>;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;
        using value_or_empty_type = std::conditional_t<BORDER == Border::VALUE, value_type, Empty>;
        using accessor_type = Accessor<const value_type, ACCESSOR_NDIM, offset_type, PointerTrait, StridesTrait>;

    public:
        constexpr Interpolator3d() = default;

        NOA_HD constexpr Interpolator3d(
                accessor_type data,
                shape3_type shape,
                value_type cvalue = value_type{0}
        ) noexcept
                : m_data(data), m_shape(shape.vec)
        {
            if constexpr (!std::is_empty_v<value_or_empty_type>)
                m_cvalue = cvalue;
        }

        /// 3d interpolation on a given batch.
        /// The batch is ignored if the accessor is 3d, effectively broadcasting the batch dimension.
        template<typename Int = index_type, typename = std::enable_if_t<nt::is_int_v<Int>>>
        NOA_HD constexpr value_type operator()(coord3_type coordinate, Int batch = Int{0}) const noexcept {
            if constexpr (ACCESSOR_NDIM == 3) {
                if constexpr (INTERP == Interp::NEAREST) {
                    return nearest_(m_data, coordinate);
                } else if constexpr (INTERP == Interp::LINEAR ||
                                     INTERP == Interp::LINEAR_FAST ||
                                     INTERP == Interp::COSINE ||
                                     INTERP == Interp::COSINE_FAST) {
                    return linear_(m_data, coordinate);
                } else if constexpr (INTERP == Interp::CUBIC ||
                                     INTERP == Interp::CUBIC_BSPLINE ||
                                     INTERP == Interp::CUBIC_BSPLINE_FAST) {
                    return cubic_(m_data, coordinate);
                } else {
                    static_assert(nt::always_false_v<value_type>);
                }
            } else {
                if constexpr (INTERP == Interp::NEAREST) {
                    return nearest_(m_data[batch], coordinate);
                } else if constexpr (INTERP == Interp::LINEAR ||
                                     INTERP == Interp::LINEAR_FAST ||
                                     INTERP == Interp::COSINE ||
                                     INTERP == Interp::COSINE_FAST) {
                    return linear_(m_data[batch], coordinate);
                } else if constexpr (INTERP == Interp::CUBIC ||
                                     INTERP == Interp::CUBIC_BSPLINE ||
                                     INTERP == Interp::CUBIC_BSPLINE_FAST) {
                    return cubic_(m_data[batch], coordinate);
                } else {
                    static_assert(nt::always_false_v<value_type>);
                }
            }
        }

        template<typename Void = void, typename = std::enable_if_t<ACCESSOR_NDIM == 4 && std::is_void_v<Void>>>
        NOA_FHD constexpr value_type at(index_type batch, index_type z, index_type y, index_type x) const noexcept {
            return m_data(batch, z, y, x);
        }

        template<typename Void = void, typename = std::enable_if_t<ACCESSOR_NDIM == 3 && std::is_void_v<Void>>>
        NOA_FHD constexpr value_type at(index_type z, index_type y, index_type x) const noexcept {
            return m_data(z, y, x);
        }

    private:
        template<typename Accessor3d>
        NOA_HD constexpr value_type nearest_(Accessor3d accessor, coord3_type coordinate) const noexcept {
            static_assert(Accessor3d::COUNT == 3);

            const index3_type index = round(coordinate).template as<index_type>();
            if constexpr (BORDER == Border::ZERO) {
                if (index[2] < 0 || index[2] >= m_shape[2] ||
                    index[1] < 0 || index[1] >= m_shape[1] ||
                    index[0] < 0 || index[0] >= m_shape[0])
                    return value_type{0};
                else
                    return accessor(index[0], index[1], index[2]);
            } else if constexpr (BORDER == Border::VALUE) {
                if (index[2] < 0 || index[2] >= m_shape[2] ||
                    index[1] < 0 || index[1] >= m_shape[1] ||
                    index[0] < 0 || index[0] >= m_shape[0])
                    return m_cvalue;
                else
                    return accessor(index[0], index[1], index[2]);
            } else if constexpr (BORDER == Border::CLAMP || BORDER == Border::PERIODIC ||
                                 BORDER == Border::MIRROR || BORDER == Border::REFLECT) {
                return accessor(offset_at<BORDER>(index[0], m_shape[0]),
                                offset_at<BORDER>(index[1], m_shape[1]),
                                offset_at<BORDER>(index[2], m_shape[2]));
            } else {
                static_assert(nt::always_false_v<value_type>);
            }
        }

        template<typename Accessor3d>
        NOA_HD constexpr value_type linear_(Accessor3d accessor, coord3_type coordinate) const noexcept {
            static_assert(Accessor3d::COUNT == 3);
            index3_type idx[2];
            idx[0] = floor(coordinate).template as<index_type>();
            idx[1] = idx[0] + 1;

            value_type values[8];
            if constexpr (BORDER == Border::ZERO || BORDER == Border::VALUE) {
                const bool cond_z[2] = {idx[0][0] >= 0 && idx[0][0] < m_shape[0], idx[1][0] >= 0 && idx[1][0] < m_shape[0]};
                const bool cond_y[2] = {idx[0][1] >= 0 && idx[0][1] < m_shape[1], idx[1][1] >= 0 && idx[1][1] < m_shape[1]};
                const bool cond_x[2] = {idx[0][2] >= 0 && idx[0][2] < m_shape[2], idx[1][2] >= 0 && idx[1][2] < m_shape[2]};

                // TODO Might be more efficient to do two 2d interpolations and a final 1D...
                value_type cval;
                if constexpr (BORDER == Border::ZERO)
                    cval = value_type{0};
                else
                    cval = m_cvalue;
                values[0] = cond_z[0] && cond_y[0] && cond_x[0] ? accessor(idx[0][0], idx[0][1], idx[0][2]) : cval; // v000
                values[1] = cond_z[0] && cond_y[0] && cond_x[1] ? accessor(idx[0][0], idx[0][1], idx[1][2]) : cval; // v001
                values[2] = cond_z[0] && cond_y[1] && cond_x[0] ? accessor(idx[0][0], idx[1][1], idx[0][2]) : cval; // v010
                values[3] = cond_z[0] && cond_y[1] && cond_x[1] ? accessor(idx[0][0], idx[1][1], idx[1][2]) : cval; // v011
                values[4] = cond_z[1] && cond_y[0] && cond_x[0] ? accessor(idx[1][0], idx[0][1], idx[0][2]) : cval; // v100
                values[5] = cond_z[1] && cond_y[0] && cond_x[1] ? accessor(idx[1][0], idx[0][1], idx[1][2]) : cval; // v101
                values[6] = cond_z[1] && cond_y[1] && cond_x[0] ? accessor(idx[1][0], idx[1][1], idx[0][2]) : cval; // v110
                values[7] = cond_z[1] && cond_y[1] && cond_x[1] ? accessor(idx[1][0], idx[1][1], idx[1][2]) : cval; // v111

            } else if constexpr (BORDER == Border::CLAMP || BORDER == Border::PERIODIC ||
                                 BORDER == Border::MIRROR || BORDER == Border::REFLECT) {
                const offset_type tmp[6] = {static_cast<offset_type>(offset_at<BORDER>(idx[0][2], m_shape[2])),
                                            static_cast<offset_type>(offset_at<BORDER>(idx[1][2], m_shape[2])),
                                            static_cast<offset_type>(offset_at<BORDER>(idx[0][1], m_shape[1])),
                                            static_cast<offset_type>(offset_at<BORDER>(idx[1][1], m_shape[1])),
                                            static_cast<offset_type>(offset_at<BORDER>(idx[0][0], m_shape[0])),
                                            static_cast<offset_type>(offset_at<BORDER>(idx[1][0], m_shape[0]))};
                values[0] = accessor(tmp[4], tmp[2], tmp[0]); // v000
                values[1] = accessor(tmp[4], tmp[2], tmp[1]); // v001
                values[2] = accessor(tmp[4], tmp[3], tmp[0]); // v010
                values[3] = accessor(tmp[4], tmp[3], tmp[1]); // v011
                values[4] = accessor(tmp[5], tmp[2], tmp[0]); // v100
                values[5] = accessor(tmp[5], tmp[2], tmp[1]); // v101
                values[6] = accessor(tmp[5], tmp[3], tmp[0]); // v110
                values[7] = accessor(tmp[5], tmp[3], tmp[1]); // v111
            } else {
                static_assert(nt::always_false_v<value_type>);
            }

            const coord3_type fraction{coordinate - static_cast<coord3_type>(idx[0])};
            if constexpr (INTERP == Interp::COSINE || INTERP == Interp::CUBIC_BSPLINE) {
                return interpolate_cosine_3d(
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7],
                        fraction[2], fraction[1], fraction[0]);
            } else {
                return interpolate_lerp_3d(
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7],
                        fraction[2], fraction[1], fraction[0]);
            }
        }

        template<typename Accessor3d>
        NOA_HD constexpr value_type cubic_(Accessor3d accessor, coord3_type coordinate) const noexcept {
            static_assert(Accessor3d::COUNT == 3);

            const index3_type idx = floor(coordinate).template as<index_type>();
            value_type values[4][4][4]; // [z][y][x]
            if constexpr (BORDER == Border::ZERO || BORDER == Border::VALUE) {
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
                if constexpr (BORDER == Border::ZERO)
                    cval = value_type{0};
                else
                    cval = m_cvalue;
                constexpr index_type offset[4] = {-1, 0, 1, 2};
                for (index_type i = 0; i < 4; ++i) {
                    const index_type idx_z = idx[0] + offset[i];
                    for (index_type j = 0; j < 4; ++j) {
                        const index_type idx_y = idx[1] + offset[j];
                        for (index_type k = 0; k < 4; ++k) {
                            values[i][j][k] = cond_z[i] && cond_y[j] && cond_x[k] ?
                                              accessor(idx_z, idx_y, idx[2] + offset[k]) : cval;
                        }
                    }
                }
            } else if constexpr (BORDER == Border::CLAMP || BORDER == Border::PERIODIC ||
                                 BORDER == Border::MIRROR || BORDER == Border::REFLECT) {
                const index_type tmp_z[4] = {offset_at<BORDER>(idx[0] - 1, m_shape[0]),
                                             offset_at<BORDER>(idx[0] + 0, m_shape[0]),
                                             offset_at<BORDER>(idx[0] + 1, m_shape[0]),
                                             offset_at<BORDER>(idx[0] + 2, m_shape[0])};
                const index_type tmp_y[4] = {offset_at<BORDER>(idx[1] - 1, m_shape[1]),
                                             offset_at<BORDER>(idx[1] + 0, m_shape[1]),
                                             offset_at<BORDER>(idx[1] + 1, m_shape[1]),
                                             offset_at<BORDER>(idx[1] + 2, m_shape[1])};
                const index_type tmp_x[4] = {offset_at<BORDER>(idx[2] - 1, m_shape[2]),
                                             offset_at<BORDER>(idx[2] + 0, m_shape[2]),
                                             offset_at<BORDER>(idx[2] + 1, m_shape[2]),
                                             offset_at<BORDER>(idx[2] + 2, m_shape[2])};
                for (index_type i = 0; i < 4; ++i)
                    for (index_type j = 0; j < 4; ++j)
                        for (index_type k = 0; k < 4; ++k)
                            values[i][j][k] = accessor(tmp_z[i], tmp_y[j], tmp_x[k]);

            } else {
                static_assert(nt::always_false_v<value_type>);
            }

            const coord3_type fraction{coordinate - static_cast<coord3_type>(idx)};
            if constexpr (INTERP == Interp::CUBIC_BSPLINE || INTERP == Interp::CUBIC_BSPLINE_FAST)
                return interpolate_cubic_bspline_3d(values, fraction[2], fraction[1], fraction[0]);
            else
                return interpolate_cubic_3d(values, fraction[2], fraction[1], fraction[0]);
        }

    private:
        accessor_type m_data{};
        index3_type m_shape{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_cvalue{};
    };

    template<Border BORDER, Interp INTERP, typename Coord = float,
             typename Value, typename Offset, typename Index, size_t NDIM,
             PointerTraits PointerTrait, StridesTraits StridesTrait,
             typename CValue = nt::remove_ref_cv_t<Value>,
             typename = std::enable_if_t<nt::is_almost_same_v<Value, CValue> &&
                                         std::is_same_v<std::make_signed_t<Offset>, Index>>>
    constexpr auto interpolator_3d(
            const Accessor <Value, NDIM, Offset, PointerTrait, StridesTrait>& accessor,
            Shape3 <Index> shape,
            CValue cvalue = CValue{0}
    ) {
        using mutable_value_t = std::remove_cv_t<Value>;
        using interpolator_t = Interpolator3d<
                BORDER, INTERP, mutable_value_t,
                Offset, Coord, NDIM, PointerTrait, StridesTrait>;
        return interpolator_t(accessor, shape, cvalue);
    }
}

namespace noa::geometry {
    /// Interpolates 2d data.
    template<Border BORDER, Interp INTERP,
             typename Value, typename Index, typename Coord>
    class InterpolatorValue2d {
    public:
        static_assert(nt::is_any_v<Index, int32_t, int64_t>);
        static_assert(nt::is_real_or_complex_v<Value> && !std::is_const_v<Value>);
        static_assert(nt::is_real_v<Coord> && !std::is_const_v<Coord>);

        using value_type = Value;
        using mutable_value_type = Value;
        using index_type = Index;
        using coord_type = Coord;

        using index2_type = Vec2<index_type>;
        using coord2_type = Vec2<coord_type>;
        using shape2_type = Shape2<index_type>;
        using value_or_empty_type = std::conditional_t<BORDER == Border::VALUE, value_type, Empty>;

    public:
        constexpr InterpolatorValue2d() = default;

        NOA_HD constexpr InterpolatorValue2d(
                value_type data,
                shape2_type shape,
                value_type cvalue = value_type{0}
        ) noexcept
                : m_data(data), m_shape(shape.vec())
        {
            if constexpr (!std::is_empty_v<value_or_empty_type>)
                m_cvalue = cvalue;
        }

        /// 2d interpolation on a given batch.
        /// The batch is ignored, effectively broadcasting the batch dimension.
        template<typename Int = index_type, typename = std::enable_if_t<nt::is_int_v<Int>>>
        NOA_HD constexpr value_type operator()(coord2_type coordinate, Int = Int{0}) const noexcept {
            if constexpr (BORDER != Border::ZERO && BORDER != Border::VALUE)
                return m_data;

            if constexpr (INTERP == Interp::NEAREST) {
                return nearest_(coordinate);
            } else if constexpr (INTERP == Interp::LINEAR ||
                                 INTERP == Interp::LINEAR_FAST ||
                                 INTERP == Interp::COSINE ||
                                 INTERP == Interp::COSINE_FAST) {
                return linear_(coordinate);
            } else if constexpr (INTERP == Interp::CUBIC ||
                                 INTERP == Interp::CUBIC_BSPLINE ||
                                 INTERP == Interp::CUBIC_BSPLINE_FAST) {
                return cubic_(coordinate);
            } else {
                static_assert(nt::always_false_v<value_type>);
            }
        }

        NOA_FHD constexpr value_type at(index_type, index_type, index_type) const noexcept {
            return m_data;
        }

        NOA_FHD constexpr value_type at(index_type, index_type) const noexcept {
            return m_data;
        }

    private:
        NOA_HD constexpr value_type nearest_(coord2_type coordinate) const noexcept {
            const index2_type index = round(coordinate).template as<index_type>();
            if (index[1] < 0 || index[1] >= m_shape[1] ||
                index[0] < 0 || index[0] >= m_shape[0]) {
                if constexpr (BORDER == Border::ZERO)
                    return value_type{0};
                else if constexpr (BORDER == Border::VALUE)
                    return m_cvalue;
            } else {
                return m_data;
            }
        }

        NOA_HD constexpr value_type linear_(coord2_type coordinate) const noexcept {
            const index2_type idx0 = floor(coordinate).template as<index_type>();
            const index2_type idx1 = idx0 + 1;
            if (noa::all(idx0 >= 0 && idx1 < m_shape))
                return m_data;

            const bool cond_y[2] = {idx0[0] >= 0 && idx0[0] < m_shape[0], idx1[0] >= 0 && idx1[0] < m_shape[0]};
            const bool cond_x[2] = {idx0[1] >= 0 && idx0[1] < m_shape[1], idx1[1] >= 0 && idx1[1] < m_shape[1]};
            value_type values[4]; // v00, v10, v01, v11
            if constexpr (BORDER == Border::ZERO) {
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
            if constexpr (INTERP == Interp::COSINE || INTERP == Interp::COSINE_FAST)
                return interpolate_cosine_2d(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
            else
                return lerp_2d(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
        }

        NOA_HD constexpr value_type cubic_(coord2_type coordinate) const noexcept {
            const index2_type idx = floor(coordinate).template as<index_type>();
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
            if constexpr (BORDER == Border::ZERO)
                oob_value = value_type{0};
            else
                oob_value = m_cvalue;

            for (index_type j = 0; j < 4; ++j)
                for (index_type i = 0; i < 4; ++i)
                    square[j][i] = cond_x[i] && cond_y[j] ? m_data : oob_value;

            const coord2_type fraction{coordinate - static_cast<coord2_type>(idx)};
            if constexpr (INTERP == Interp::CUBIC_BSPLINE || INTERP == Interp::CUBIC_BSPLINE_FAST)
                return interpolate_cubic_bspline_2d(square, fraction[1], fraction[0]);
            else
                return interpolate_cubic_2d(square, fraction[1], fraction[0]);
        }

    private:
        value_type m_data{};
        index2_type m_shape{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_cvalue{};
    };

    template<Border BORDER, Interp INTERP, typename Coord = float,
             typename Value, typename Index,
             typename CValue = nt::remove_ref_cv_t<Value>,
             typename = std::enable_if_t<nt::is_almost_same_v<Value, CValue>>>
    constexpr auto interpolator_value_2d(
            Value data,
            const Shape2<Index>& shape,
            CValue cvalue = CValue{0}
    ) {
        using mutable_data_t = std::remove_cv_t<Value>;
        using interpolator_t = InterpolatorValue2d<BORDER, INTERP, mutable_data_t, Index, Coord>;
        return interpolator_t(data, shape, cvalue);
    }
}

namespace noa::geometry {
    // Interpolates 3d data.
    template<Border BORDER, Interp INTERP,
             typename Value, typename Index, typename Coord>
    class InterpolatorValue3d {
    public:
        static_assert(nt::is_any_v<Index, int32_t, int64_t>);
        static_assert(nt::is_real_or_complex_v<Value> && !std::is_const_v<Value>);
        static_assert(nt::is_real_v<Coord> && !std::is_const_v<Coord>);

        using value_type = Value;
        using mutable_value_type = Value;
        using index_type = Index;
        using coord_type = Coord;

        using index3_type = Vec3<index_type>;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;
        using value_or_empty_type = std::conditional_t<BORDER == Border::VALUE, value_type, Empty>;

    public:
        constexpr InterpolatorValue3d() = default;

        NOA_HD constexpr InterpolatorValue3d(
                value_type data,
                shape3_type shape,
                value_type cvalue = value_type{0}
        ) noexcept
                : m_data(data), m_shape(shape.vec())
        {
            if constexpr (!std::is_empty_v<value_or_empty_type>)
                m_cvalue = cvalue;
        }

        // 3d interpolation on a given batch.
        // The batch is ignored if the accessor is 3d, effectively broadcasting the batch dimension.
        template<typename Int = index_type, typename = std::enable_if_t<nt::is_int_v<Int>>>
        NOA_HD constexpr value_type operator()(coord3_type coordinate, Int = Int{0}) const noexcept {
            if constexpr (BORDER != Border::ZERO && BORDER != Border::VALUE)
                return m_data;

            if constexpr (INTERP == Interp::NEAREST) {
                return nearest_(coordinate);
            } else if constexpr (INTERP == Interp::LINEAR ||
                                 INTERP == Interp::LINEAR_FAST ||
                                 INTERP == Interp::COSINE ||
                                 INTERP == Interp::COSINE_FAST) {
                return linear_(coordinate);
            } else if constexpr (INTERP == Interp::CUBIC ||
                                 INTERP == Interp::CUBIC_BSPLINE ||
                                 INTERP == Interp::CUBIC_BSPLINE_FAST) {
                return cubic_(coordinate);
            } else {
                static_assert(nt::always_false_v<value_type>);
            }
        }

        NOA_FHD constexpr value_type at(index_type, index_type, index_type, index_type) const noexcept {
            return m_data;
        }

        NOA_FHD constexpr value_type at(index_type, index_type, index_type) const noexcept {
            return m_data;
        }

    private:
        NOA_HD constexpr value_type nearest_(coord3_type coordinate) const noexcept {
            const index3_type index = round(coordinate).template as<index_type>();
            if constexpr (BORDER == Border::ZERO) {
                if (index[2] < 0 || index[2] >= m_shape[2] ||
                    index[1] < 0 || index[1] >= m_shape[1] ||
                    index[0] < 0 || index[0] >= m_shape[0])
                    return value_type{0};
                else
                    return m_data;
            } else if constexpr (BORDER == Border::VALUE) {
                if (index[2] < 0 || index[2] >= m_shape[2] ||
                    index[1] < 0 || index[1] >= m_shape[1] ||
                    index[0] < 0 || index[0] >= m_shape[0])
                    return m_cvalue;
                else
                    return m_data;
            }
        }

        NOA_HD constexpr value_type linear_(coord3_type coordinate) const noexcept {
            index3_type idx[2];
            idx[0] = floor(coordinate).template as<index_type>();
            idx[1] = idx[0] + 1;
            if (noa::all(idx[0] >= 0 && idx[1] < m_shape))
                return m_data;

            value_type values[8];
            const bool cond_z[2] = {idx[0][0] >= 0 && idx[0][0] < m_shape[0], idx[1][0] >= 0 && idx[1][0] < m_shape[0]};
            const bool cond_y[2] = {idx[0][1] >= 0 && idx[0][1] < m_shape[1], idx[1][1] >= 0 && idx[1][1] < m_shape[1]};
            const bool cond_x[2] = {idx[0][2] >= 0 && idx[0][2] < m_shape[2], idx[1][2] >= 0 && idx[1][2] < m_shape[2]};

            // TODO Might be more efficient to do two 2d interpolations and a final 1D...
            value_type cval;
            if constexpr (BORDER == Border::ZERO)
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
            if constexpr (INTERP == Interp::COSINE || INTERP == Interp::CUBIC_BSPLINE) {
                return interpolate_cosine_3d(
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7],
                        fraction[2], fraction[1], fraction[0]);
            } else {
                return lerp_3d(
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7],
                        fraction[2], fraction[1], fraction[0]);
            }
        }

        NOA_HD constexpr value_type cubic_(coord3_type coordinate) const noexcept {
            const index3_type idx = floor(coordinate).template as<index_type>();
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
            if constexpr (BORDER == Border::ZERO)
                cval = value_type{0};
            else
                cval = m_cvalue;
            for (index_type i = 0; i < 4; ++i)
                for (index_type j = 0; j < 4; ++j)
                    for (index_type k = 0; k < 4; ++k)
                        values[i][j][k] = cond_z[i] && cond_y[j] && cond_x[k] ? m_data : cval;

            const coord3_type fraction{coordinate - static_cast<coord3_type>(idx)};
            if constexpr (INTERP == Interp::CUBIC_BSPLINE || INTERP == Interp::CUBIC_BSPLINE_FAST)
                return interpolate_cubic_bspline_3d(values, fraction[2], fraction[1], fraction[0]);
            else
                return interpolate_cubic_3d(values, fraction[2], fraction[1], fraction[0]);
        }

    private:
        value_type m_data{};
        index3_type m_shape{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_cvalue{};
    };

    template<Border BORDER, Interp INTERP, typename Coord = float,
             typename Value, typename Index,
             typename CValue = nt::remove_ref_cv_t<Value>,
             typename = std::enable_if_t<nt::is_almost_same_v<Value, CValue>>>
    constexpr auto interpolator_value_3d(
            Value data,
            const Shape3<Index>& shape,
            CValue cvalue = CValue{0}
    ) {
        using mutable_data_t = std::remove_cv_t<Value>;
        using interpolator_t = InterpolatorValue3d<BORDER, INTERP, mutable_data_t, Index, Coord>;
        return interpolator_t(data, shape, cvalue);
    }
}

namespace noa::traits {
    template<typename T> struct proclaim_is_interpolator_2d : std::false_type {};
    template<typename T> struct proclaim_is_interpolator_3d : std::false_type {};

    template<typename T> using is_interpolator_2d = std::bool_constant<proclaim_is_interpolator_2d<remove_ref_cv_t<T>>::value>;
    template<typename T> using is_interpolator_3d = std::bool_constant<proclaim_is_interpolator_3d<remove_ref_cv_t<T>>::value>;

    template<typename T> constexpr bool is_interpolator_2d_v = is_interpolator_2d<T>::value;
    template<typename T> constexpr bool is_interpolator_3d_v = is_interpolator_3d<T>::value;
    template<typename T> constexpr bool is_interpolator_v = is_interpolator_2d_v<T> || is_interpolator_3d_v<T>;
    template<typename... Ts> constexpr bool are_interpolator_v = bool_and<is_interpolator_v<Ts>...>::value;

    template<Border BORDER, Interp INTERP,
             typename Value, typename Offset, typename Coord,
             size_t ACCESSOR_NDIM, PointerTraits PointerTrait, StridesTraits StridesTrait>
    struct proclaim_is_interpolator_2d<
            noa::geometry::Interpolator2d<
                    BORDER, INTERP,
                    Value, Offset, Coord,
                    ACCESSOR_NDIM, PointerTrait, StridesTrait>
    > : std::true_type {};

    template<Border BORDER, Interp INTERP,
            typename Value, typename Offset, typename Coord,
            size_t ACCESSOR_NDIM, PointerTraits PointerTrait, StridesTraits StridesTrait>
    struct proclaim_is_interpolator_3d<
            noa::geometry::Interpolator3d<
                    BORDER, INTERP,
                    Value, Offset, Coord,
                    ACCESSOR_NDIM, PointerTrait, StridesTrait>
    > : std::true_type {};

    template<Border BORDER, Interp INTERP, typename Value, typename Index, typename Coord>
    struct proclaim_is_interpolator_2d<noa::geometry::InterpolatorValue2d<BORDER, INTERP, Value, Index, Coord>> : std::true_type {};

    template<Border BORDER, Interp INTERP, typename Value, typename Index, typename Coord>
    struct proclaim_is_interpolator_3d<noa::geometry::InterpolatorValue3d<BORDER, INTERP, Value, Index, Coord>> : std::true_type {};
}
