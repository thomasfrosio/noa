#pragma once

#include "noa/core/geometry/Polar.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Mat.hpp"
#include "noa/core/types/Vec.hpp"

/// Simple geometric shapes (eg ellipse or rectangle).
///     - Edges can be sharp (ie binary) or smooth (using a raised-cosine edge).
///     - 3 operators are available. All take a 1d/2d/3d coordinate, where to render the shape and:
///         i.   An Empty type (defaulted) so the call site can be generic.
///         ii.  A rotation matrix encoding the inverse transformation to apply to the shape.
///              The center of rotation is the center of the shape (so for a sphere the rotation has no effect).
///         iii. An affine matrix encoding the inverse transformation to apply to the shape.
///     - These are masks, the shape has a constant value and the background is set to 0.
///       The inverse of the mask can be computed, in which case the background is set to the constant value
///       and the shape is 0 (of course, the smooth edge is also inverted).

namespace noa::geometry {
    /// Line/Ray/Window defined by a center and radius, possibly with a smooth raised-cosine edge.
    template<typename Value, bool IsSmooth, bool IsInverted = false, typename Coord = f32>
    class Line {
    public:
        static constexpr bool is_inverted = IsInverted;
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using coord_type = Coord;
        using coord_or_empty_type = std::conditional_t<is_smooth, coord_type, Empty>;

    public:
        NOA_HD constexpr Line(
                coord_type center,
                coord_type radius,
                value_type cvalue
        ) noexcept requires is_smooth
                : m_cvalue(cvalue), m_center(center), m_radius(radius) {}

        NOA_HD constexpr Line(
                coord_type center,
                coord_type radius,
                coord_type edge_size,
                value_type cvalue
        ) noexcept requires (not is_smooth)
                : m_cvalue(cvalue), m_center(center), m_radius(radius), m_edge_size(edge_size) {}

        [[nodiscard]] NOA_HD constexpr value_type operator()(std::integral auto index) const noexcept {
            return (*this)(static_cast<coord_type>(index));
        }

        [[nodiscard]] NOA_HD constexpr value_type operator()(coord_type coordinate) const noexcept {
            coordinate -= m_center;
            const auto dst = abs(coordinate);

            if constexpr (is_smooth) {
                constexpr auto PI = Constant<coord_type>::PI;
                if (dst > m_radius + m_edge_size) {
                    return static_cast<value_type>(is_inverted) * m_cvalue;
                } else if (dst <= m_radius) {
                    return static_cast<value_type>(not is_inverted) * m_cvalue;
                } else {
                    const auto weight = cos(PI * (dst - m_radius) / m_edge_size);
                    if constexpr (is_inverted)
                        return static_cast<value_type>((coord_type{1} - weight) * coord_type{0.5}) * m_cvalue;
                    else
                        return static_cast<value_type>((coord_type{1} + weight) * coord_type{0.5}) * m_cvalue;
                }
            } else {
                if (dst > m_radius)
                    return static_cast<value_type>(is_inverted) * m_cvalue;
                else
                    return static_cast<value_type>(not is_inverted) * m_cvalue;
            }
        }

    private:
        value_type m_cvalue;
        coord_type m_center;
        coord_type m_radius;
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_edge_size{};
    };

    /// Sphere defined by a center and radius, possibly with a smooth raised-cosine edge.
    template<size_t NDIM, typename Value, bool IsSmooth, bool IsInverted = false, typename Coord = f32>
    class Sphere {
    public:
        static constexpr bool is_inverted = IsInverted;
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using coord_type = Coord;
        using coord_or_empty_type = std::conditional_t<is_smooth, coord_type, Empty>;
        using vector_type = Vec<coord_type, NDIM>;
        using rotation_type = std::conditional_t<NDIM == 2, Mat22<coord_type>, Mat33<coord_type>>;
        using affine_type = std::conditional_t<NDIM == 2, Mat23<coord_type>, Mat34<coord_type>>;

        NOA_HD constexpr Sphere(
                vector_type center,
                coord_type radius,
                value_type cvalue
        ) noexcept requires (not is_smooth)
                : m_center(center), m_radius_sqd(radius * radius), m_cvalue(cvalue) {}

        NOA_HD constexpr Sphere(
                vector_type center,
                coord_type radius,
                coord_type edge_size,
                value_type cvalue
        ) noexcept requires is_smooth
                : m_center(center),
                  m_radius_sqd(radius * radius),
                  m_cvalue(cvalue),
                  m_radius(radius),
                  m_edge_size(edge_size),
                  m_radius_edge_sqd((radius + edge_size) * (radius + edge_size)) {}

    public:
        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector_type coordinates,
                Empty = Empty{}
        ) const noexcept {
            return get_(coordinates - m_center);
        }

        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector_type coordinates,
                const rotation_type& inv_transform
        ) const noexcept {
            coordinates -= m_center;
            coordinates = inv_transform * coordinates;
            return get_(coordinates);
        }

        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector_type coordinates,
                const affine_type& inv_transform
        ) const noexcept {
            return get_(inv_transform * coordinates.push_back(1) - m_center);
        }

    private:
        [[nodiscard]] NOA_HD constexpr value_type get_(vector_type coordinates) const noexcept {
            const auto dst_sqd = dot(coordinates, coordinates);
            if constexpr (is_smooth) {
                constexpr auto PI = Constant<coord_type>::PI;
                if (dst_sqd > m_radius_edge_sqd) {
                    return static_cast<value_type>(is_inverted) * m_cvalue;
                } else if (dst_sqd <= m_radius_sqd) {
                    return static_cast<value_type>(not is_inverted) * m_cvalue;
                } else {
                    const auto dst = sqrt(dst_sqd);
                    const auto weight = cos(PI * (dst - m_radius) / m_edge_size);
                    if constexpr (is_inverted)
                        return static_cast<value_type>((coord_type{1} - weight) * coord_type{0.5}) * m_cvalue;
                    else
                        return static_cast<value_type>((coord_type{1} + weight) * coord_type{0.5}) * m_cvalue;
                }
            } else {
                if (dst_sqd > m_radius_sqd)
                    return static_cast<value_type>(is_inverted) * m_cvalue;
                else
                    return static_cast<value_type>(not is_inverted) * m_cvalue;
            }
        }

    private:
        vector_type m_center;
        coord_type m_radius_sqd;
        value_type m_cvalue;
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_radius{};
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_edge_size{};
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_radius_edge_sqd{};
    };

    /// Cylinder defined by a center, radius and length, possibly with a smooth raised-cosine edge.
    template<typename Value, bool IsSmooth, bool IsInverted = false, typename Coord = f32>
    class Cylinder {
    public:
        static constexpr bool is_inverted = IsInverted;
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using coord_type = Coord;
        using coord_or_empty_type = std::conditional_t<is_smooth, coord_type, Empty>;
        using vector3_type = Vec3<coord_type>;
        using vector2_type = Vec2<coord_type>;
        using rotation_type = Mat33<coord_type>;
        using affine_type = Mat34<coord_type>;

        NOA_HD constexpr Cylinder(
                vector3_type center,
                coord_type radius,
                coord_type length,
                value_type cvalue
        ) noexcept requires (not is_smooth)
                : m_center(center),
                  m_radius_sqd(radius * radius),
                  m_length(length),
                  m_cvalue(cvalue) {}

        NOA_HD constexpr Cylinder(
                vector3_type center,
                coord_type radius,
                coord_type length,
                coord_type edge_size,
                value_type cvalue
        ) noexcept requires is_smooth
                : m_center(center),
                  m_radius_sqd(radius * radius),
                  m_length(length),
                  m_cvalue(cvalue),
                  m_radius(radius),
                  m_radius_edge_sqd((radius + edge_size) * (radius + edge_size)),
                  m_edge_size(edge_size),
                  m_length_edge(length + edge_size) {}

        NOA_HD constexpr Cylinder(
                vector3_type center,
                vector2_type length_radius,
                value_type cvalue
        ) noexcept requires (not is_smooth)
                : Cylinder(center, length_radius[1], length_radius[0], cvalue) {}

        NOA_HD constexpr Cylinder(
                vector3_type center,
                vector3_type length_radius,
                value_type cvalue
        ) noexcept requires (not is_smooth)
                : Cylinder(center, length_radius[1], length_radius[0], cvalue) {
            NOA_ASSERT(length_radius[1] == length_radius[2]);
        }

        NOA_HD constexpr Cylinder(
                vector3_type center, vector2_type length_radius,
                coord_type edge_size, value_type cvalue
        ) noexcept requires is_smooth
                : Cylinder(center, length_radius[1], length_radius[0], edge_size, cvalue) {}

        NOA_HD constexpr Cylinder(
                vector3_type center, vector3_type length_radius,
                coord_type edge_size, value_type cvalue
        ) noexcept requires is_smooth
                : Cylinder(center, length_radius[1], length_radius[0], edge_size, cvalue) {
            NOA_ASSERT(length_radius[1] == length_radius[2]);
        }

        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector3_type coordinates,
                Empty = Empty{}
        ) const noexcept {
            return get_(coordinates - m_center);
        }

        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector3_type coordinates,
                const rotation_type& inv_transform
        ) const noexcept {
            coordinates -= m_center;
            coordinates = inv_transform * coordinates;
            return get_(coordinates);
        }

        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector3_type coordinates,
                const affine_type& inv_transform
        ) const noexcept {
            return get_(inv_transform * coordinates.push_back(1) - m_center);
        }

    private:
        [[nodiscard]] NOA_HD constexpr value_type get_(vector3_type coordinates) const noexcept {
            const auto dst_z = abs(coordinates[0]);
            vector2_type tmp{coordinates[1], coordinates[2]};
            const coord_type dst_yx_sqd = dot(tmp, tmp);

            if constexpr (is_smooth) {
                if (dst_z > m_length_edge or dst_yx_sqd > m_radius_edge_sqd) {
                    return static_cast<value_type>(is_inverted) * m_cvalue;
                } else {
                    constexpr auto PI = Constant<coord_type>::PI;
                    coord_type mask;
                    if (dst_yx_sqd <= m_radius_sqd) {
                        mask = coord_type{1};
                    } else {
                        const auto dst_yx = (sqrt(dst_yx_sqd) - m_radius) / m_edge_size;
                        mask = (coord_type{1} + cos(PI * dst_yx)) * coord_type{0.5};
                    }

                    if (dst_z > m_length) {
                        const auto dst = (dst_z - m_length) / m_edge_size;
                        mask *= (coord_type{1} + cos(PI * dst)) * coord_type{0.5};
                    }

                    if constexpr (is_inverted)
                        return static_cast<value_type>(coord_type{1} - mask) * m_cvalue;
                    else
                        return static_cast<value_type>(mask) * m_cvalue;
                }
            } else {
                if (dst_z > m_length or dst_yx_sqd > m_radius_sqd)
                    return static_cast<value_type>(is_inverted) * m_cvalue;
                else
                    return static_cast<value_type>(not is_inverted) * m_cvalue;
            }
        }

    private:
        vector3_type m_center;
        coord_type m_radius_sqd;
        coord_type m_length;
        value_type m_cvalue;
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_radius{};
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_radius_edge_sqd{};
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_edge_size{};
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_length_edge{};
    };

    /// Rectangle defined by a center and radius, possibly with a smooth raised-cosine edge.
    template<size_t NDIM, typename Value, bool IsSmooth, bool IsInverted = false, typename Coord = f32>
    class Rectangle {
    public:
        static constexpr bool is_inverted = IsInverted;
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using coord_type = Coord;
        using vector_type = Vec<coord_type, NDIM>;
        using coord_or_empty_type = std::conditional_t<is_smooth, coord_type, Empty>;
        using vector_or_empty_type = std::conditional_t<is_smooth, vector_type, Empty>;
        using rotation_type = std::conditional_t<NDIM == 2, Mat22<coord_type>, Mat33<coord_type>>;
        using affine_type = std::conditional_t<NDIM == 2, Mat23<coord_type>, Mat34<coord_type>>;

        NOA_HD constexpr Rectangle(
                vector_type center, vector_type radius, value_type cvalue
        ) noexcept requires (not is_smooth)
                : m_center(center), m_radius(radius), m_cvalue(cvalue) {}

        NOA_HD constexpr Rectangle(
                vector_type center,
                vector_type radius,
                coord_type edge_size,
                value_type cvalue
        ) noexcept requires is_smooth
                : m_center(center),
                  m_radius(radius),
                  m_cvalue(cvalue),
                  m_radius_edge(radius + edge_size),
                  m_edge_size(edge_size) {}

        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector_type coords,
                Empty = Empty{}
        ) const noexcept {
            return get_(coords - m_center);
        }

        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector_type coords,
                const rotation_type& inv_transform
        ) const noexcept {
            coords -= m_center;
            coords = inv_transform * coords;
            return get_(coords);
        }

        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector_type coords,
                const affine_type& inv_transform
        ) const noexcept {
            return get_(inv_transform * coords.push_back(1) - m_center);
        }

    private:
        [[nodiscard]] NOA_HD constexpr value_type get_(vector_type coords) const noexcept {
            coords = abs(coords);

            if constexpr (IsSmooth) {
                if (any(m_radius_edge < coords)) {
                    return static_cast<value_type>(is_inverted) * m_cvalue;
                } else if (all(coords <= m_radius)) {
                    return static_cast<value_type>(not is_inverted) * m_cvalue;
                } else {
                    coord_type mask_value{1};
                    for (size_t i = 0; i < NDIM; ++i) {
                        if (m_radius[i] < coords[i] and coords[i] <= m_radius_edge[i]) {
                            constexpr auto PI = Constant<coord_type>::PI;
                            const auto distance = (coords[i] - m_radius[i]) / m_edge_size;
                            mask_value *= (coord_type{1} + cos(PI * distance)) * coord_type{0.5};
                        }
                    }
                    if constexpr (is_inverted)
                        return static_cast<value_type>(coord_type{1} - mask_value) * m_cvalue;
                    else
                        return static_cast<value_type>(mask_value) * m_cvalue;
                }
            } else {
                if (all(coords <= m_radius))
                    return static_cast<value_type>(not is_inverted) * m_cvalue;
                else
                    return static_cast<value_type>(is_inverted) * m_cvalue;
            }
        }

    private:
        vector_type m_center;
        vector_type m_radius;
        value_type m_cvalue;
        NOA_NO_UNIQUE_ADDRESS vector_or_empty_type m_radius_edge{};
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_edge_size{};
    };

    /// Ellipse defined by a center and radius, possibly with a smooth raised-cosine edge.
    template<size_t NDIM, typename Value, bool IsSmooth = true, bool IsInverted = false, typename Coord = f32>
    class Ellipse {
    public:
        static constexpr bool is_inverted = IsInverted;
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using coord_type = Coord;
        using vector_type = Vec<coord_type, NDIM>;
        using coord_or_empty_type = std::conditional_t<is_smooth, coord_type, Empty>;
        using rotation_type = std::conditional_t<NDIM == 2, Mat22<coord_type>, Mat33<coord_type>>;
        using affine_type = std::conditional_t<NDIM == 2, Mat23<coord_type>, Mat34<coord_type>>;

        NOA_HD constexpr Ellipse(
                vector_type center, vector_type radius, value_type cvalue
        ) noexcept requires (not is_smooth)
                : m_center(center), m_radius(radius), m_cvalue(cvalue) {}

        NOA_HD constexpr Ellipse(
                vector_type center,
                vector_type radius,
                coord_type edge_size,
                value_type cvalue
        ) noexcept requires (not is_smooth)
                : m_center(center), m_radius(radius * radius), m_cvalue(cvalue), m_edge_size(edge_size) {}

        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector_type coordinates,
                Empty = Empty{}
        ) const noexcept {
            return get_(coordinates - m_center);
        }

        // TODO for 2d we could also accept the rotation angle and add it to phi.
        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector_type coordinates,
                const rotation_type& inv_transform
        ) const noexcept {
            coordinates -= m_center;
            coordinates = inv_transform * coordinates;
            return get_(coordinates);
        }

        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector_type coordinates,
                const affine_type& inv_transform
        ) const noexcept {
            return get_(inv_transform * coordinates.push_back(1) - m_center);
        }

    private:
        [[nodiscard]] NOA_HD constexpr value_type get_(vector_type coordinates) const noexcept {
            if constexpr (IsSmooth) {
                coord_type irho, erho;
                if constexpr (NDIM == 2) {
                    // Current spherical coordinate:
                    irho = noa::geometry::cartesian2rho(coordinates);
                    const coord_type iphi = noa::geometry::cartesian2phi<false>(coordinates);

                    // Radius of the ellipse at (iphi, itheta):
                    const coord_type cos2phi = pow(cos(iphi), coord_type{2});
                    const coord_type sin2phi = pow(sin(iphi), coord_type{2});
                    erho = coord_type{1} / sqrt(cos2phi / m_radius[1] + sin2phi / m_radius[0]);

                } else {
                    irho = noa::geometry::cartesian2rho(coordinates);
                    const coord_type iphi = noa::geometry::cartesian2phi<false>(coordinates);
                    const coord_type itheta = noa::geometry::cartesian2theta(coordinates);

                    const coord_type cos2phi = pow(cos(iphi), coord_type{2});
                    const coord_type sin2phi = pow(sin(iphi), coord_type{2});
                    const coord_type cos2theta = pow(cos(itheta), coord_type{2});
                    const coord_type sin2theta = pow(sin(itheta), coord_type{2});
                    erho = coord_type{1} / sqrt(cos2phi * sin2theta / m_radius[2] +
                                                sin2phi * sin2theta / m_radius[1] +
                                                cos2theta / m_radius[0]);
                }

                constexpr coord_type PI = Constant<coord_type>::PI;
                if (irho > erho + m_edge_size) {
                    return static_cast<value_type>(is_inverted) * m_cvalue;
                } else if (irho <= erho) {
                    return static_cast<value_type>(not is_inverted) * m_cvalue;
                } else {
                    const auto distance = (irho - erho) / m_edge_size;
                    if constexpr (is_inverted) {
                        return static_cast<value_type>((coord_type{1} - cos(PI * distance)) * coord_type{0.5}) * m_cvalue;
                    } else {
                        return static_cast<value_type>((coord_type{1} + cos(PI * distance)) * coord_type{0.5}) * m_cvalue;
                    }
                }
            } else {
                coordinates /= m_radius;
                const coord_type rho = dot(coordinates, coordinates);
                if constexpr (is_inverted)
                    return static_cast<value_type>(rho > 1) * m_cvalue;
                else
                    return static_cast<value_type>(rho <= 1) * m_cvalue;
            }
        }

    private:
        vector_type m_center;
        vector_type m_radius;
        value_type m_cvalue;
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_edge_size{};
    };
}
