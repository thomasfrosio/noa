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
///     - The data is wrapped into an accessor, thus supporting different memory layouts and pointer traits.
///       AccessorValue is supported, and the interpolator has special logic to simplify the interpolation
///       whenever possible.
///     - The data can be batched. If the input is not batched, the batch dimension is automatically broadcasted.
///     - Interpolation can be skipped using the .at() member functions, in case integral coordinates are used
///       (in which case the interpolators are simple wrappers around the accessors).
///     - The coordinate and index/offset types are templated, e.g. interpolation can be single or double precision.
///     - These interpolators can work on the CPU or GPU.
///
/// Out-of-bounds:
///     One of the main differences between these interpolations and what we can find in other cryoEM packages,
///     is that the interpolation window can be partially out-of-bound (OOB), that is, elements that are OOB
///     are replaced according to a Border. cryoEM packages usually check that all elements are in-bound
///     and if there's even one element OOB, they don't interpolate.
///     Note: The interpolators below are for real space interpolation, or redundant and centered Fourier transforms.
///           Half Fourier transforms are currently not supported and are handled separately.
///
/// Coordinate system:
///     The coordinate system matches the indexing. This is quite standard.
///     For instance, the first data sample at index 0 is located at the coordinate 0 and the coordinate 0.5
///     is just in between the first and second element. As such, the fractional part of the coordinate
///     corresponds to the ratio/weight used by the interpolation function. In other words,
///     the coordinate system locates the data between -0.5 and N-1 + 0.5.

namespace noa::geometry {
    /// Interpolates 2d data.
    template<Border BORDER, Interp INTERP, typename Accessor>
    class Interpolator2d {
    public:
        using accessor_type = Accessor;
        using value_type = typename Accessor::value_type;
        using mutable_value_type = typename Accessor::mutable_value_type;
        using offset_type = typename Accessor::index_type;
        static constexpr size_t ACCESSOR_NDIM = Accessor::SIZE;

        static_assert(nt::is_any_v<offset_type, i32, u32, i64, u64>);
        static_assert(nt::is_real_or_complex_v<value_type>);
        static_assert(ACCESSOR_NDIM == 2 or ACCESSOR_NDIM == 3 or nt::is_accessor_value_v<accessor_type>);

        using index_type = std::make_signed_t<offset_type>;
        using index2_type = Vec2<index_type>;
        using shape2_type = Shape2<index_type>;
        using value_or_empty_type = std::conditional_t<BORDER == Border::VALUE, value_type, Empty>;

    public:
        constexpr Interpolator2d() = default;

        /// Stores a copy of accessor, shape and cvalue.
        /// The created instance can then be used to interpolate the nd-data.
        NOA_HD constexpr Interpolator2d(
                const accessor_type& accessor,
                shape2_type shape,
                value_type cvalue = value_type{0}
        ) noexcept : m_accessor(accessor), m_shape(shape.vec) {
            if constexpr (not std::is_empty_v<value_or_empty_type>)
                m_cvalue = cvalue;
        }

        /// 2d interpolation on a given batch.
        /// The batch is ignored if the accessor is 2d, effectively broadcasting the batch dimension.
        template<typename T, typename I = index_type> requires (nt::is_any_v<T, f32, f64> and nt::is_int_v<I>)
        NOA_HD constexpr value_type operator()(const Vec2<T>& coordinate, I batch = I{0}) const noexcept {
            if constexpr (ACCESSOR_NDIM == 2 or nt::is_accessor_value_v<accessor_type>) {
                if constexpr (INTERP == Interp::NEAREST) {
                    return nearest_(m_accessor, coordinate);
                } else if constexpr (INTERP == Interp::LINEAR or
                                     INTERP == Interp::LINEAR_FAST or
                                     INTERP == Interp::COSINE or
                                     INTERP == Interp::COSINE_FAST) {
                    return linear_(m_accessor, coordinate);
                } else if constexpr (INTERP == Interp::CUBIC or
                                     INTERP == Interp::CUBIC_BSPLINE or
                                     INTERP == Interp::CUBIC_BSPLINE_FAST) {
                    return cubic_(m_accessor, coordinate);
                } else {
                    static_assert(nt::always_false_v<value_type>);
                }
            } else {
                if constexpr (INTERP == Interp::NEAREST) {
                    return nearest_(m_accessor[batch], coordinate);
                } else if constexpr (INTERP == Interp::LINEAR or
                                     INTERP == Interp::LINEAR_FAST or
                                     INTERP == Interp::COSINE or
                                     INTERP == Interp::COSINE_FAST) {
                    return linear_(m_accessor[batch], coordinate);
                } else if constexpr (INTERP == Interp::CUBIC or
                                     INTERP == Interp::CUBIC_BSPLINE or
                                     INTERP == Interp::CUBIC_BSPLINE_FAST) {
                    return cubic_(m_accessor[batch], coordinate);
                } else {
                    static_assert(nt::always_false_v<value_type>);
                }
            }
        }

        template<typename... Integers>
        requires (nt::are_int_v<Integers...> and
                  (ACCESSOR_NDIM == sizeof...(Integers) or
                   nt::is_accessor_value_v<accessor_type>))
        NOA_HD constexpr value_type at(Integers... indices) const noexcept {
            return m_accessor(indices...);
        }

    private:
        NOA_HD constexpr value_type nearest_(const auto& accessor, const auto& coordinate) const noexcept {
            const index2_type index = round(coordinate).template as<index_type>();

            if constexpr (BORDER == Border::ZERO) {
                if (index[1] < 0 or index[1] >= m_shape[1] or
                    index[0] < 0 or index[0] >= m_shape[0])
                    return value_type{0};
                else
                    return accessor(index[0], index[1]);
            } else if constexpr (BORDER == Border::VALUE) {
                if (index[1] < 0 or index[1] >= m_shape[1] or
                    index[0] < 0 or index[0] >= m_shape[0])
                    return m_cvalue;
                else
                    return accessor(index[0], index[1]);
            } else if constexpr (BORDER == Border::CLAMP or BORDER == Border::PERIODIC or
                                 BORDER == Border::MIRROR or BORDER == Border::REFLECT) {
                return accessor(offset_at<BORDER>(index[0], m_shape[0]),
                                offset_at<BORDER>(index[1], m_shape[1]));
            } else {
                static_assert(nt::always_false_v<value_type>);
            }
        }

        template<typename Accessor2d, typename T> requires (not nt::is_accessor_value_v<Accessor2d>)
        NOA_HD constexpr value_type linear_(const Accessor2d& accessor, const Vec2<T>& coordinate) const noexcept {
            static_assert(Accessor2d::COUNT == 2);
            const index2_type idx0 = coordinate.template as<index_type>();
            const index2_type idx1 = idx0 + 1;

            value_type values[4]; // v00, v10, v01, v11
            if constexpr (BORDER == Border::ZERO or BORDER == Border::VALUE) {
                const bool cond_y[2]{idx0[0] >= 0 and idx0[0] < m_shape[0], idx1[0] >= 0 and idx1[0] < m_shape[0]};
                const bool cond_x[2]{idx0[1] >= 0 and idx0[1] < m_shape[1], idx1[1] >= 0 and idx1[1] < m_shape[1]};

                if constexpr (BORDER == Border::ZERO) {
                    values[0] = cond_y[0] and cond_x[0] ? accessor(idx0[0], idx0[1]) : value_type{0}; // v00
                    values[1] = cond_y[0] and cond_x[1] ? accessor(idx0[0], idx1[1]) : value_type{0}; // v01
                    values[2] = cond_y[1] and cond_x[0] ? accessor(idx1[0], idx0[1]) : value_type{0}; // v10
                    values[3] = cond_y[1] and cond_x[1] ? accessor(idx1[0], idx1[1]) : value_type{0}; // v11
                } else {
                    values[0] = cond_y[0] and cond_x[0] ? accessor(idx0[0], idx0[1]) : m_cvalue;
                    values[1] = cond_y[0] and cond_x[1] ? accessor(idx0[0], idx1[1]) : m_cvalue;
                    values[2] = cond_y[1] and cond_x[0] ? accessor(idx1[0], idx0[1]) : m_cvalue;
                    values[3] = cond_y[1] and cond_x[1] ? accessor(idx1[0], idx1[1]) : m_cvalue;
                }

            } else if constexpr (BORDER == Border::CLAMP or BORDER == Border::PERIODIC or
                                 BORDER == Border::MIRROR or BORDER == Border::REFLECT) {
                const index_type tmp[4]{
                        offset_at<BORDER>(idx0[1], m_shape[1]),
                        offset_at<BORDER>(idx1[1], m_shape[1]),
                        offset_at<BORDER>(idx0[0], m_shape[0]),
                        offset_at<BORDER>(idx1[0], m_shape[0])
                };
                values[0] = accessor(tmp[2], tmp[0]); // v00
                values[1] = accessor(tmp[2], tmp[1]); // v01
                values[2] = accessor(tmp[3], tmp[0]); // v10
                values[3] = accessor(tmp[3], tmp[1]); // v11

            } else {
                static_assert(nt::always_false_v<value_type>);
            }

            const auto fraction{coordinate - Vec2<T>::from_value(idx0)};
            if constexpr (INTERP == Interp::COSINE or INTERP == Interp::COSINE_FAST)
                return interpolate_cosine_2d(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
            else
                return lerp_2d(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
        }

        template<typename AccessorValue, typename T> requires (nt::is_accessor_value_v<AccessorValue>)
        NOA_HD constexpr value_type linear_(const AccessorValue& accessor, const Vec2<T>& coordinate) const noexcept {
            const index2_type idx0 = floor(coordinate).template as<index_type>();
            const index2_type idx1 = idx0 + 1;
            if (all(idx0 >= 0 and idx1 < m_shape))
                return accessor();

            const bool cond_y[2]{idx0[0] >= 0 and idx0[0] < m_shape[0], idx1[0] >= 0 and idx1[0] < m_shape[0]};
            const bool cond_x[2]{idx0[1] >= 0 and idx0[1] < m_shape[1], idx1[1] >= 0 and idx1[1] < m_shape[1]};
            value_type values[4]; // v00, v10, v01, v11
            if constexpr (BORDER == Border::ZERO) {
                values[0] = cond_y[0] and cond_x[0] ? accessor() : value_type{0}; // v00
                values[1] = cond_y[0] and cond_x[1] ? accessor() : value_type{0}; // v01
                values[2] = cond_y[1] and cond_x[0] ? accessor() : value_type{0}; // v10
                values[3] = cond_y[1] and cond_x[1] ? accessor() : value_type{0}; // v11
            } else {
                values[0] = cond_y[0] and cond_x[0] ? accessor() : m_cvalue;
                values[1] = cond_y[0] and cond_x[1] ? accessor() : m_cvalue;
                values[2] = cond_y[1] and cond_x[0] ? accessor() : m_cvalue;
                values[3] = cond_y[1] and cond_x[1] ? accessor() : m_cvalue;
            }

            const auto fraction{coordinate - Vec2<T>::from_value(idx0)};
            if constexpr (INTERP == Interp::COSINE or INTERP == Interp::COSINE_FAST)
                return interpolate_cosine_2d(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
            else
                return lerp_2d(values[0], values[1], values[2], values[3], fraction[1], fraction[0]);
        }


        template<typename Accessor2d, typename T> requires (not nt::is_accessor_value_v<Accessor2d>)
        NOA_HD constexpr value_type cubic_(const Accessor2d& accessor, const Vec2<T>& coordinate) const noexcept {
            static_assert(Accessor2d::COUNT == 2);
            const index2_type idx = floor(coordinate).template as<index_type>();

            value_type square[4][4]; // [y][x]
            if constexpr (BORDER == Border::ZERO or BORDER == Border::VALUE) {
                const bool cond_y[4]{
                        idx[0] - 1 >= 0 and idx[0] - 1 < m_shape[0],
                        idx[0] + 0 >= 0 and idx[0] + 0 < m_shape[0],
                        idx[0] + 1 >= 0 and idx[0] + 1 < m_shape[0],
                        idx[0] + 2 >= 0 and idx[0] + 2 < m_shape[0]
                };
                const bool cond_x[4]{
                        idx[1] - 1 >= 0 and idx[1] - 1 < m_shape[1],
                        idx[1] + 0 >= 0 and idx[1] + 0 < m_shape[1],
                        idx[1] + 1 >= 0 and idx[1] + 1 < m_shape[1],
                        idx[1] + 2 >= 0 and idx[1] + 2 < m_shape[1]
                };
                constexpr index_type offset[4]{-1, 0, 1, 2};
                for (index_type j = 0; j < 4; ++j) {
                    const index_type idx_y = idx[0] + offset[j];
                    for (index_type i = 0; i < 4; ++i) {
                        const index_type idx_x = idx[1] + offset[i];
                        if constexpr (BORDER == Border::ZERO)
                            square[j][i] = cond_x[i] and cond_y[j] ? accessor(idx_y, idx_x) : value_type{0};
                        else
                            square[j][i] = cond_x[i] and cond_y[j] ? accessor(idx_y, idx_x) : m_cvalue;
                    }
                }
            } else if constexpr (BORDER == Border::CLAMP or BORDER == Border::PERIODIC or
                                 BORDER == Border::MIRROR or BORDER == Border::REFLECT) {
                const index_type tmp_y[4]{
                        offset_at<BORDER>(idx[0] - 1, m_shape[0]),
                        offset_at<BORDER>(idx[0] + 0, m_shape[0]),
                        offset_at<BORDER>(idx[0] + 1, m_shape[0]),
                        offset_at<BORDER>(idx[0] + 2, m_shape[0])
                };
                const index_type tmp_x[4]{
                        offset_at<BORDER>(idx[1] - 1, m_shape[1]),
                        offset_at<BORDER>(idx[1] + 0, m_shape[1]),
                        offset_at<BORDER>(idx[1] + 1, m_shape[1]),
                        offset_at<BORDER>(idx[1] + 2, m_shape[1])
                };
                for (index_type j = 0; j < 4; ++j)
                    for (index_type i = 0; i < 4; ++i)
                        square[j][i] = accessor(tmp_y[j], tmp_x[i]);

            } else {
                static_assert(nt::always_false_v<value_type>);
            }
            const auto fraction{coordinate - Vec2<T>::from_value(idx)};
            if constexpr (INTERP == Interp::CUBIC_BSPLINE or INTERP == Interp::CUBIC_BSPLINE_FAST)
                return interpolate_cubic_bspline_2d(square, fraction[1], fraction[0]);
            else
                return interpolate_cubic_2d(square, fraction[1], fraction[0]);
        }

        template<typename AccessorValue, typename T> requires nt::is_accessor_value_v<AccessorValue>
        NOA_HD constexpr value_type cubic_(const AccessorValue& accessor, const Vec2<T>& coordinate) const noexcept {
            const index2_type idx = floor(coordinate).template as<index_type>();
            if (all(idx - 1 >= 0 and idx + 2 < m_shape))
                return accessor();

            value_type square[4][4]; // [y][x]
            const bool cond_y[4]{
                    idx[0] - 1 >= 0 and idx[0] - 1 < m_shape[0],
                    idx[0] + 0 >= 0 and idx[0] + 0 < m_shape[0],
                    idx[0] + 1 >= 0 and idx[0] + 1 < m_shape[0],
                    idx[0] + 2 >= 0 and idx[0] + 2 < m_shape[0]
            };
            const bool cond_x[4]{
                    idx[1] - 1 >= 0 and idx[1] - 1 < m_shape[1],
                    idx[1] + 0 >= 0 and idx[1] + 0 < m_shape[1],
                    idx[1] + 1 >= 0 and idx[1] + 1 < m_shape[1],
                    idx[1] + 2 >= 0 and idx[1] + 2 < m_shape[1]
            };

            value_type oob_value;
            if constexpr (BORDER == Border::ZERO)
                oob_value = value_type{0};
            else
                oob_value = m_cvalue;

            for (index_type j = 0; j < 4; ++j)
                for (index_type i = 0; i < 4; ++i)
                    square[j][i] = cond_x[i] and cond_y[j] ? accessor() : oob_value;

            const auto fraction{coordinate - Vec2<T>::from_value(idx)};
            if constexpr (INTERP == Interp::CUBIC_BSPLINE or INTERP == Interp::CUBIC_BSPLINE_FAST)
                return interpolate_cubic_bspline_2d(square, fraction[1], fraction[0]);
            else
                return interpolate_cubic_2d(square, fraction[1], fraction[0]);
        }

    private:
        accessor_type m_accessor{};
        index2_type m_shape{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_cvalue{};
    };

    /// Interpolates 3d data.
    template<Border BORDER, Interp INTERP, typename Accessor>
    class Interpolator3d {
    public:
        using accessor_type = Accessor;
        using value_type = typename Accessor::value_type;
        using mutable_value_type = typename Accessor::mutable_value_type;
        using offset_type = typename Accessor::index_type;
        static constexpr size_t ACCESSOR_NDIM = Accessor::SIZE;

        static_assert(nt::is_any_v<offset_type,i32, u32, i64, u64>);
        static_assert(nt::is_real_or_complex_v<value_type>);
        static_assert(ACCESSOR_NDIM == 3 or ACCESSOR_NDIM == 4 or nt::is_accessor_value_v<accessor_type>);

        using index_type = std::make_signed_t<offset_type>;
        using index3_type = Vec3<index_type>;
        using shape3_type = Shape3<index_type>;
        using value_or_empty_type = std::conditional_t<BORDER == Border::VALUE, value_type, Empty>;

    public:
        constexpr Interpolator3d() = default;

        NOA_HD constexpr Interpolator3d(
                const accessor_type& data,
                shape3_type shape,
                value_type cvalue = value_type{0}
        ) noexcept : m_accessor(data), m_shape(shape.vec) {
            if constexpr (not std::is_empty_v<value_or_empty_type>)
                m_cvalue = cvalue;
        }

        /// 3d interpolation on a given batch.
        /// The batch is ignored if the accessor is 3d, effectively broadcasting the batch dimension.
        template<typename T, typename I = index_type> requires nt::is_int_v<I>
        NOA_HD constexpr value_type operator()(const Vec3<T>& coordinate, I batch = I{0}) const noexcept {
            if constexpr (ACCESSOR_NDIM == 3 or nt::is_accessor_value_v<Accessor>) {
                if constexpr (INTERP == Interp::NEAREST) {
                    return nearest_(m_accessor, coordinate);
                } else if constexpr (INTERP == Interp::LINEAR or
                                     INTERP == Interp::LINEAR_FAST or
                                     INTERP == Interp::COSINE or
                                     INTERP == Interp::COSINE_FAST) {
                    return linear_(m_accessor, coordinate);
                } else if constexpr (INTERP == Interp::CUBIC or
                                     INTERP == Interp::CUBIC_BSPLINE or
                                     INTERP == Interp::CUBIC_BSPLINE_FAST) {
                    return cubic_(m_accessor, coordinate);
                } else {
                    static_assert(nt::always_false_v<value_type>);
                }
            } else {
                if constexpr (INTERP == Interp::NEAREST) {
                    return nearest_(m_accessor[batch], coordinate);
                } else if constexpr (INTERP == Interp::LINEAR or
                                     INTERP == Interp::LINEAR_FAST or
                                     INTERP == Interp::COSINE or
                                     INTERP == Interp::COSINE_FAST) {
                    return linear_(m_accessor[batch], coordinate);
                } else if constexpr (INTERP == Interp::CUBIC or
                                     INTERP == Interp::CUBIC_BSPLINE or
                                     INTERP == Interp::CUBIC_BSPLINE_FAST) {
                    return cubic_(m_accessor[batch], coordinate);
                } else {
                    static_assert(nt::always_false_v<value_type>);
                }
            }
        }

        template<typename... Integers>
        requires (nt::are_int_v<Integers...> and
                  (ACCESSOR_NDIM == sizeof...(Integers) or
                   nt::is_accessor_value_v<accessor_type>))
        NOA_HD constexpr value_type at(Integers... indices) const noexcept {
            return m_accessor(indices...);
        }

    private:
        template<typename Accessor3d, typename T>
        NOA_HD constexpr value_type nearest_(const Accessor3d& accessor, const Vec3<T>& coordinate) const noexcept {
            static_assert(Accessor3d::COUNT == 3);

            const index3_type index = round(coordinate).template as<index_type>();
            if constexpr (BORDER == Border::ZERO) {
                if (index[2] < 0 or index[2] >= m_shape[2] or
                    index[1] < 0 or index[1] >= m_shape[1] or
                    index[0] < 0 or index[0] >= m_shape[0])
                    return value_type{0};
                else
                    return accessor(index[0], index[1], index[2]);
            } else if constexpr (BORDER == Border::VALUE) {
                if (index[2] < 0 or index[2] >= m_shape[2] or
                    index[1] < 0 or index[1] >= m_shape[1] or
                    index[0] < 0 or index[0] >= m_shape[0])
                    return m_cvalue;
                else
                    return accessor(index[0], index[1], index[2]);
            } else if constexpr (BORDER == Border::CLAMP or BORDER == Border::PERIODIC or
                                 BORDER == Border::MIRROR or BORDER == Border::REFLECT) {
                return accessor(offset_at<BORDER>(index[0], m_shape[0]),
                                offset_at<BORDER>(index[1], m_shape[1]),
                                offset_at<BORDER>(index[2], m_shape[2]));
            } else {
                static_assert(nt::always_false_v<value_type>);
            }
        }

        template<typename Accessor3d, typename T> requires (not nt::is_accessor_value_v<Accessor3d>)
        NOA_HD constexpr value_type linear_(const Accessor3d& accessor, const Vec3<T>& coordinate) const noexcept {
            static_assert(Accessor3d::COUNT == 3);
            index3_type idx[2];
            idx[0] = floor(coordinate).template as<index_type>();
            idx[1] = idx[0] + 1;

            value_type values[8];
            if constexpr (BORDER == Border::ZERO or BORDER == Border::VALUE) {
                const bool cond_z[2]{idx[0][0] >= 0 and idx[0][0] < m_shape[0], idx[1][0] >= 0 and idx[1][0] < m_shape[0]};
                const bool cond_y[2]{idx[0][1] >= 0 and idx[0][1] < m_shape[1], idx[1][1] >= 0 and idx[1][1] < m_shape[1]};
                const bool cond_x[2]{idx[0][2] >= 0 and idx[0][2] < m_shape[2], idx[1][2] >= 0 and idx[1][2] < m_shape[2]};

                // TODO Might be more efficient to do two 2d interpolations and a final 1D...
                value_type cval;
                if constexpr (BORDER == Border::ZERO)
                    cval = value_type{0};
                else
                    cval = m_cvalue;
                values[0] = cond_z[0] and cond_y[0] and cond_x[0] ? accessor(idx[0][0], idx[0][1], idx[0][2]) : cval; // v000
                values[1] = cond_z[0] and cond_y[0] and cond_x[1] ? accessor(idx[0][0], idx[0][1], idx[1][2]) : cval; // v001
                values[2] = cond_z[0] and cond_y[1] and cond_x[0] ? accessor(idx[0][0], idx[1][1], idx[0][2]) : cval; // v010
                values[3] = cond_z[0] and cond_y[1] and cond_x[1] ? accessor(idx[0][0], idx[1][1], idx[1][2]) : cval; // v011
                values[4] = cond_z[1] and cond_y[0] and cond_x[0] ? accessor(idx[1][0], idx[0][1], idx[0][2]) : cval; // v100
                values[5] = cond_z[1] and cond_y[0] and cond_x[1] ? accessor(idx[1][0], idx[0][1], idx[1][2]) : cval; // v101
                values[6] = cond_z[1] and cond_y[1] and cond_x[0] ? accessor(idx[1][0], idx[1][1], idx[0][2]) : cval; // v110
                values[7] = cond_z[1] and cond_y[1] and cond_x[1] ? accessor(idx[1][0], idx[1][1], idx[1][2]) : cval; // v111

            } else if constexpr (BORDER == Border::CLAMP or BORDER == Border::PERIODIC or
                                 BORDER == Border::MIRROR or BORDER == Border::REFLECT) {
                const offset_type tmp[6]{
                        static_cast<offset_type>(offset_at<BORDER>(idx[0][2], m_shape[2])),
                        static_cast<offset_type>(offset_at<BORDER>(idx[1][2], m_shape[2])),
                        static_cast<offset_type>(offset_at<BORDER>(idx[0][1], m_shape[1])),
                        static_cast<offset_type>(offset_at<BORDER>(idx[1][1], m_shape[1])),
                        static_cast<offset_type>(offset_at<BORDER>(idx[0][0], m_shape[0])),
                        static_cast<offset_type>(offset_at<BORDER>(idx[1][0], m_shape[0]))
                };
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

            const auto fraction{coordinate - Vec3<T>::from_value(idx[0])};
            if constexpr (INTERP == Interp::COSINE or INTERP == Interp::CUBIC_BSPLINE) {
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

        template<typename AccessorValue, typename T> requires nt::is_accessor_value_v<AccessorValue>
        NOA_HD constexpr value_type linear_(const AccessorValue& accessor, const Vec3<T>& coordinate) const noexcept {
            index3_type idx[2];
            idx[0] = floor(coordinate).template as<index_type>();
            idx[1] = idx[0] + 1;
            if (all(idx[0] >= 0 and idx[1] < m_shape))
                return accessor();

            value_type values[8];
            const bool cond_z[2]{idx[0][0] >= 0 and idx[0][0] < m_shape[0], idx[1][0] >= 0 and idx[1][0] < m_shape[0]};
            const bool cond_y[2]{idx[0][1] >= 0 and idx[0][1] < m_shape[1], idx[1][1] >= 0 and idx[1][1] < m_shape[1]};
            const bool cond_x[2]{idx[0][2] >= 0 and idx[0][2] < m_shape[2], idx[1][2] >= 0 and idx[1][2] < m_shape[2]};

            // TODO Might be more efficient to do two 2d interpolations and a final 1D...
            value_type cval;
            if constexpr (BORDER == Border::ZERO)
                cval = value_type{0};
            else
                cval = m_cvalue;
            values[0] = cond_z[0] and cond_y[0] and cond_x[0] ? accessor() : cval; // v000
            values[1] = cond_z[0] and cond_y[0] and cond_x[1] ? accessor() : cval; // v001
            values[2] = cond_z[0] and cond_y[1] and cond_x[0] ? accessor() : cval; // v010
            values[3] = cond_z[0] and cond_y[1] and cond_x[1] ? accessor() : cval; // v011
            values[4] = cond_z[1] and cond_y[0] and cond_x[0] ? accessor() : cval; // v100
            values[5] = cond_z[1] and cond_y[0] and cond_x[1] ? accessor() : cval; // v101
            values[6] = cond_z[1] and cond_y[1] and cond_x[0] ? accessor() : cval; // v110
            values[7] = cond_z[1] and cond_y[1] and cond_x[1] ? accessor() : cval; // v111

            const auto fraction{coordinate - Vec3<T>::from_value(idx[0])};
            if constexpr (INTERP == Interp::COSINE or INTERP == Interp::CUBIC_BSPLINE) {
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

        template<typename Accessor3d, typename T> requires (not nt::is_accessor_value_v<Accessor3d>)
        NOA_HD constexpr value_type cubic_(const Accessor3d& accessor, const Vec3<T>& coordinate) const noexcept {
            static_assert(Accessor3d::COUNT == 3);

            const index3_type idx = floor(coordinate).template as<index_type>();
            value_type values[4][4][4]; // [z][y][x]
            if constexpr (BORDER == Border::ZERO or BORDER == Border::VALUE) {
                const bool cond_z[4]{
                        idx[0] - 1 >= 0 and idx[0] - 1 < m_shape[0],
                        idx[0] + 0 >= 0 and idx[0] + 0 < m_shape[0],
                        idx[0] + 1 >= 0 and idx[0] + 1 < m_shape[0],
                        idx[0] + 2 >= 0 and idx[0] + 2 < m_shape[0]
                };
                const bool cond_y[4]{
                        idx[1] - 1 >= 0 and idx[1] - 1 < m_shape[1],
                        idx[1] + 0 >= 0 and idx[1] + 0 < m_shape[1],
                        idx[1] + 1 >= 0 and idx[1] + 1 < m_shape[1],
                        idx[1] + 2 >= 0 and idx[1] + 2 < m_shape[1]
                };
                const bool cond_x[4]{
                        idx[2] - 1 >= 0 and idx[2] - 1 < m_shape[2],
                        idx[2] + 0 >= 0 and idx[2] + 0 < m_shape[2],
                        idx[2] + 1 >= 0 and idx[2] + 1 < m_shape[2],
                        idx[2] + 2 >= 0 and idx[2] + 2 < m_shape[2]
                };
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
                            values[i][j][k] = cond_z[i] and cond_y[j] and cond_x[k] ?
                                              accessor(idx_z, idx_y, idx[2] + offset[k]) : cval;
                        }
                    }
                }
            } else if constexpr (BORDER == Border::CLAMP or BORDER == Border::PERIODIC or
                                 BORDER == Border::MIRROR or BORDER == Border::REFLECT) {
                const index_type tmp_z[4]{
                        offset_at<BORDER>(idx[0] - 1, m_shape[0]),
                        offset_at<BORDER>(idx[0] + 0, m_shape[0]),
                        offset_at<BORDER>(idx[0] + 1, m_shape[0]),
                        offset_at<BORDER>(idx[0] + 2, m_shape[0])
                };
                const index_type tmp_y[4]{
                        offset_at<BORDER>(idx[1] - 1, m_shape[1]),
                        offset_at<BORDER>(idx[1] + 0, m_shape[1]),
                        offset_at<BORDER>(idx[1] + 1, m_shape[1]),
                        offset_at<BORDER>(idx[1] + 2, m_shape[1])
                };
                const index_type tmp_x[4]{
                        offset_at<BORDER>(idx[2] - 1, m_shape[2]),
                        offset_at<BORDER>(idx[2] + 0, m_shape[2]),
                        offset_at<BORDER>(idx[2] + 1, m_shape[2]),
                        offset_at<BORDER>(idx[2] + 2, m_shape[2])
                };
                for (index_type i = 0; i < 4; ++i)
                    for (index_type j = 0; j < 4; ++j)
                        for (index_type k = 0; k < 4; ++k)
                            values[i][j][k] = accessor(tmp_z[i], tmp_y[j], tmp_x[k]);

            } else {
                static_assert(nt::always_false_v<value_type>);
            }

            const auto fraction{coordinate - Vec3<T>::from_value(idx)};
            if constexpr (INTERP == Interp::CUBIC_BSPLINE or INTERP == Interp::CUBIC_BSPLINE_FAST)
                return interpolate_cubic_bspline_3d(values, fraction[2], fraction[1], fraction[0]);
            else
                return interpolate_cubic_3d(values, fraction[2], fraction[1], fraction[0]);
        }

        template<typename AccessorValue, typename T> requires nt::is_accessor_value_v<AccessorValue>
        NOA_HD constexpr value_type cubic_(const AccessorValue& accessor, const Vec3<T>& coordinate) const noexcept {
            const index3_type idx = floor(coordinate).template as<index_type>();
            if (all(idx - 1 >= 0 and idx + 2 < m_shape))
                return accessor();

            value_type values[4][4][4]; // [z][y][x]
            const bool cond_z[4]{
                    idx[0] - 1 >= 0 and idx[0] - 1 < m_shape[0],
                    idx[0] + 0 >= 0 and idx[0] + 0 < m_shape[0],
                    idx[0] + 1 >= 0 and idx[0] + 1 < m_shape[0],
                    idx[0] + 2 >= 0 and idx[0] + 2 < m_shape[0]
            };
            const bool cond_y[4]{
                    idx[1] - 1 >= 0 and idx[1] - 1 < m_shape[1],
                    idx[1] + 0 >= 0 and idx[1] + 0 < m_shape[1],
                    idx[1] + 1 >= 0 and idx[1] + 1 < m_shape[1],
                    idx[1] + 2 >= 0 and idx[1] + 2 < m_shape[1]
            };
            const bool cond_x[4]{
                    idx[2] - 1 >= 0 and idx[2] - 1 < m_shape[2],
                    idx[2] + 0 >= 0 and idx[2] + 0 < m_shape[2],
                    idx[2] + 1 >= 0 and idx[2] + 1 < m_shape[2],
                    idx[2] + 2 >= 0 and idx[2] + 2 < m_shape[2]
            };
            value_type cval;
            if constexpr (BORDER == Border::ZERO)
                cval = value_type{0};
            else
                cval = m_cvalue;
            for (index_type i = 0; i < 4; ++i)
                for (index_type j = 0; j < 4; ++j)
                    for (index_type k = 0; k < 4; ++k)
                        values[i][j][k] = cond_z[i] and cond_y[j] and cond_x[k] ? accessor() : cval;

            const auto fraction{coordinate - Vec3<T>::from_value(idx)};
            if constexpr (INTERP == Interp::CUBIC_BSPLINE or INTERP == Interp::CUBIC_BSPLINE_FAST)
                return interpolate_cubic_bspline_3d(values, fraction[2], fraction[1], fraction[0]);
            else
                return interpolate_cubic_3d(values, fraction[2], fraction[1], fraction[0]);
        }

    private:
        accessor_type m_accessor{};
        index3_type m_shape{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_cvalue{};
    };
}

namespace noa::traits {
    template<Border BORDER, Interp INTERP, typename Accessor>
    struct proclaim_is_interpolator<noa::geometry::Interpolator2d<BORDER, INTERP, Accessor>> : std::true_type {};

    template<Border BORDER, Interp INTERP, typename Accessor>
    struct proclaim_is_interpolator<noa::geometry::Interpolator3d<BORDER, INTERP, Accessor>> : std::true_type {};

    template<Border BORDER, Interp INTERP, typename Accessor>
    struct proclaim_is_interpolator_nd<noa::geometry::Interpolator2d<BORDER, INTERP, Accessor>, 2> : std::true_type {};

    template<Border BORDER, Interp INTERP, typename Accessor>
    struct proclaim_is_interpolator_nd<noa::geometry::Interpolator3d<BORDER, INTERP, Accessor>, 3> : std::true_type {};
}
