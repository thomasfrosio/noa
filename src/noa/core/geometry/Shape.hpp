#pragma once

#include "noa/core/geometry/Polar.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa::geometry {
    /// Line defined by a center and radius.
    /// \tparam INVERT  Whether the line should be filled with zeros instead of ones.
    /// \tparam ICoord  Input, storage and computation type.
    /// \tparam OCoord  Output type.
    template<typename OCoord, bool INVERT = false, typename ICoord = float>
    class Line {
    public:
        using value_type = OCoord;
        using compute_type = ICoord;

        constexpr NOA_IHD Line(compute_type center,
                               compute_type radius,
                               value_type cvalue = value_type{1}) noexcept
                : m_cvalue(cvalue), m_center(center), m_radius(radius) {}

        [[nodiscard]] constexpr NOA_IHD value_type operator()(compute_type coord) const noexcept {
            coord -= m_center;
            const auto dst = math::abs(coord);

            if (dst > m_radius)
                return static_cast<value_type>(INVERT) * m_cvalue;
            else
                return static_cast<value_type>(!INVERT) * m_cvalue;
        }

    private:
        value_type m_cvalue;
        compute_type m_center;
        compute_type m_radius;
    };

    /// Smooth line, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the line should be filled with zeros instead of ones.
    /// \tparam ICoord  Input, storage and computation type.
    /// \tparam OCoord  Output type.
    template<typename OCoord, bool INVERT = false, typename ICoord = float>
    class LineSmooth {
    public:
        using value_type = OCoord;
        using compute_type = ICoord;

        constexpr NOA_IHD LineSmooth(compute_type center, compute_type radius, compute_type edge_size,
                                     value_type cvalue = value_type{1}) noexcept
                : m_cvalue(cvalue), m_center(center), m_radius(radius), m_edge_size(edge_size) {}

        [[nodiscard]] constexpr NOA_IHD value_type operator()(compute_type coord) const noexcept {
            coord -= m_center;
            const auto dst = math::abs(coord);

            constexpr auto PI = math::Constant<compute_type>::PI;
            if (dst > m_radius + m_edge_size) {
                return static_cast<value_type>(INVERT) * m_cvalue;
            } else if (dst <= m_radius) {
                return static_cast<value_type>(!INVERT) * m_cvalue;
            } else {
                const auto weight = math::cos(PI * (dst - m_radius) / m_edge_size);
                if constexpr (INVERT)
                    return static_cast<value_type>((compute_type{1} - weight) * compute_type{0.5}) * m_cvalue;
                else
                    return static_cast<value_type>((compute_type{1} + weight) * compute_type{0.5}) * m_cvalue;
            }
        }

    private:
        value_type m_cvalue;
        compute_type m_center;
        compute_type m_radius;
        compute_type m_edge_size;
    };
}

namespace noa::signal {
    /// Sphere defined by a center and radius.
    /// \tparam INVERT  Whether the sphere should be filled with zeros instead of ones.
    /// \tparam ICoord  Input, storage and computation type.
    /// \tparam OCoord  Output type.
    template<int NDIM, typename OCoord, bool INVERT = false, typename ICoord = float>
    class Sphere {
    public:
        using value_type = OCoord;
        using compute_type = ICoord;
        using vector_type = Vec<compute_type, NDIM>;
        using rotation_type = std::conditional_t<NDIM == 2, Mat22<compute_type>, Mat33<compute_type>>;
        using affine_type = std::conditional_t<NDIM == 2, Mat23<compute_type>, Mat34<compute_type>>;

        constexpr NOA_IHD Sphere(vector_type center, compute_type radius,
                                 value_type cvalue = value_type{1}) noexcept
                : m_center(center), m_radius_sqd(radius * radius), m_cvalue(cvalue) {}

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, traits::Empty = traits::Empty{}) const noexcept {
            return get_(coords - m_center);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, const rotation_type& inv_transform) const noexcept {
            coords -= m_center;
            coords = inv_transform * coords;
            return get_(coords);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, const affine_type& inv_transform) const noexcept {
            using affine_vector_t = Vec<compute_type, NDIM + 1>;
            affine_vector_t affine_coords{1}; // {y, x, 1} or {z, y, x, 1}
            for (size_t i = 0; i < vector_type::COUNT; ++i)
                affine_coords[i] = coords[i];
            return get_(inv_transform * affine_coords - m_center);
        }

    private:
        [[nodiscard]] constexpr NOA_IHD value_type get_(vector_type coords) const noexcept {
            const auto dst_sqd = math::dot(coords, coords);
            if (dst_sqd > m_radius_sqd)
                return static_cast<value_type>(INVERT) * m_cvalue;
            else
                return static_cast<value_type>(!INVERT) * m_cvalue;
        }

    private:
        vector_type m_center;
        compute_type m_radius_sqd;
        value_type m_cvalue;
    };

    /// Smooth sphere, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the sphere should be filled with zeros instead of ones.
    /// \tparam ICoord  Input, storage and computation type.
    /// \tparam OCoord  Output type.
    template<int NDIM, typename OCoord, bool INVERT = false, typename ICoord = float>
    class SphereSmooth {
    public:
        using value_type = OCoord;
        using compute_type = ICoord;
        using vector_type = std::conditional_t<NDIM == 2, Vec2<compute_type>, Vec3<compute_type>>;
        using rotation_type = std::conditional_t<NDIM == 2, Mat22<compute_type>, Mat33<compute_type>>;
        using affine_type = std::conditional_t<NDIM == 2, Mat23<compute_type>, Mat34<compute_type>>;

        constexpr NOA_IHD SphereSmooth(
                vector_type center,
                compute_type radius,
                compute_type edge_size,
                value_type cvalue = value_type{1}) noexcept
                : m_center(center),
                  m_radius(radius),
                  m_radius_sqd(radius * radius),
                  m_edge_size(edge_size),
                  m_radius_edge_sqd((radius + edge_size) * (radius + edge_size)),
                  m_cvalue(cvalue) {}

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, traits::Empty = traits::Empty{}) const noexcept {
            return get_(coords - m_center);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, const rotation_type& inv_transform) const noexcept {
            coords -= m_center;
            coords = inv_transform * coords;
            return get_(coords);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, const affine_type& inv_transform) const noexcept {
            using affine_vector_t = std::conditional_t<NDIM == 2, Vec3<compute_type>, Vec4<compute_type>>;
            affine_vector_t affine_coords{1}; // {y, x, 1} or {z, y, x, 1}
            for (size_t i = 0; i < vector_type::COUNT; ++i)
                affine_coords[i] = coords[i];
            return get_(inv_transform * affine_coords - m_center);
        }

    private:
        [[nodiscard]] constexpr NOA_IHD value_type get_(vector_type coords) const noexcept {
            constexpr auto PI = math::Constant<compute_type>::PI;
            const auto dst_sqd = math::dot(coords, coords);
            if (dst_sqd > m_radius_edge_sqd) {
                return static_cast<value_type>(INVERT) * m_cvalue;
            } else if (dst_sqd <= m_radius_sqd) {
                return static_cast<value_type>(!INVERT) * m_cvalue;
            } else {
                const auto dst = math::sqrt(dst_sqd);
                const auto weight = math::cos(PI * (dst - m_radius) / m_edge_size);
                if constexpr (INVERT)
                    return static_cast<value_type>((compute_type{1} - weight) * compute_type{0.5}) * m_cvalue;
                else
                    return static_cast<value_type>((compute_type{1} + weight) * compute_type{0.5}) * m_cvalue;
            }
        }

    private:
        vector_type m_center;
        compute_type m_radius;
        compute_type m_radius_sqd;
        compute_type m_edge_size;
        compute_type m_radius_edge_sqd;
        value_type m_cvalue;
    };
}

namespace noa::signal {
    /// Cylinder defined by a center and radius.
    /// \tparam INVERT  Whether the cylinder should be filled with zeros instead of ones.
    /// \tparam ICoord  Input, storage and computation type.
    /// \tparam OCoord  Output type.
    template<typename OCoord, bool INVERT = false, typename ICoord = float>
    class Cylinder {
    public:
        using value_type = OCoord;
        using compute_type = ICoord;
        using vector3_type = Vec3<compute_type>;
        using vector2_type = Vec2<compute_type>;
        using rotation_type = Mat33<compute_type>;
        using affine_type = Mat34<compute_type>;

        constexpr NOA_IHD Cylinder(vector3_type center, compute_type radius, compute_type length,
                                   value_type cvalue = value_type{1}) noexcept
                : m_center(center), m_radius_sqd(radius * radius), m_length(length), m_cvalue(cvalue) {}

        constexpr NOA_IHD Cylinder(vector3_type center, vector2_type length_radius,
                                   value_type cvalue = value_type{1}) noexcept
                : Cylinder(center, length_radius[1], length_radius[0], cvalue) {}

        constexpr NOA_IHD Cylinder(vector3_type center, vector3_type length_radius,
                                   value_type cvalue = value_type{1}) noexcept
                : Cylinder(center, length_radius[1], length_radius[0], cvalue) {
            NOA_ASSERT(length_radius[1] == length_radius[2]);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector3_type coords, traits::Empty = traits::Empty{}) const noexcept {
            return get_(coords - m_center);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector3_type coords, const rotation_type& inv_transform) const noexcept {
            coords -= m_center;
            coords = inv_transform * coords;
            return get_(coords);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector3_type coords, const affine_type& inv_transform) const noexcept {
            Vec4<compute_type> affine_coords{coords[0], coords[1], coords[2], 1};
            return get_(inv_transform * affine_coords - m_center);
        }

    private:
        [[nodiscard]] constexpr NOA_IHD value_type get_(vector3_type coords) const noexcept {
            const auto dst_z = math::abs(coords[0]);
            vector2_type tmp{coords[1], coords[2]};
            const compute_type dst_yx_sqd = math::dot(tmp, tmp);

            if (dst_z > m_length || dst_yx_sqd > m_radius_sqd)
                return static_cast<value_type>(INVERT) * m_cvalue;
            else
                return static_cast<value_type>(!INVERT) * m_cvalue;
        }

    private:
        vector3_type m_center;
        compute_type m_radius_sqd;
        compute_type m_length;
        value_type m_cvalue;
    };

    /// Smooth cylinder, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the cylinder should be filled with zeros instead of ones.
    /// \tparam ICoord  Input, storage and computation type.
    /// \tparam OCoord  Output type.
    template<typename OCoord, bool INVERT = false, typename ICoord = float>
    class CylinderSmooth {
    public:
        using value_type = OCoord;
        using compute_type = ICoord;
        using vector3_type = Vec3<compute_type>;
        using vector2_type = Vec2<compute_type>;
        using rotation_type = Mat33<compute_type>;
        using affine_type = Mat34<compute_type>;

        constexpr NOA_IHD CylinderSmooth(vector3_type center, compute_type radius,
                                         compute_type length, compute_type edge_size,
                                         value_type cvalue = value_type{1}) noexcept
                : m_center(center), m_radius(radius), m_radius_sqd(radius * radius),
                  m_radius_edge_sqd((radius + edge_size) * (radius + edge_size)),
                  m_edge_size(edge_size), m_length(length), m_length_edge(length + edge_size),
                  m_cvalue(cvalue) {}

        constexpr NOA_IHD CylinderSmooth(vector3_type center, vector2_type length_radius,
                                         compute_type edge_size, value_type cvalue = value_type{1}) noexcept
                : CylinderSmooth(center, length_radius[1], length_radius[0], edge_size, cvalue) {}

        constexpr NOA_IHD CylinderSmooth(vector3_type center, vector3_type length_radius,
                                         compute_type edge_size, value_type cvalue = value_type{1}) noexcept
                : CylinderSmooth(center, length_radius[1], length_radius[0], edge_size, cvalue) {
            NOA_ASSERT(length_radius[1] == length_radius[2]);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector3_type coords, traits::Empty = traits::Empty{}) const noexcept {
            return get_(coords - m_center);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector3_type coords, const rotation_type& inv_transform) const noexcept {
            coords -= m_center;
            coords = inv_transform * coords;
            return get_(coords);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector3_type coords, const affine_type& inv_transform) const noexcept {
            Vec4<compute_type> affine_coords{coords[0], coords[1], coords[2], 1};
            return get_(inv_transform * affine_coords - m_center);
        }

    private:
        [[nodiscard]] constexpr NOA_IHD value_type get_(vector3_type coords) const noexcept {
            const auto dst_z = math::abs(coords[0]);
            vector2_type tmp{coords[1], coords[2]};
            const compute_type dst_yx_sqd = math::dot(tmp, tmp);

            if (dst_z > m_length_edge || dst_yx_sqd > m_radius_edge_sqd) {
                return static_cast<value_type>(INVERT) * m_cvalue;
            } else {
                constexpr auto PI = math::Constant<compute_type>::PI;
                compute_type mask;
                if (dst_yx_sqd <= m_radius_sqd) {
                    mask = compute_type{1};
                } else {
                    const auto dst_yx = (math::sqrt(dst_yx_sqd) - m_radius) / m_edge_size;
                    mask = (compute_type{1} + math::cos(PI * dst_yx)) * compute_type{0.5};
                }

                if (dst_z > m_length) {
                    const auto dst = (dst_z - m_length) / m_edge_size;
                    mask *= (compute_type{1} + math::cos(PI * dst)) * compute_type{0.5};
                }

                if constexpr (INVERT)
                    return static_cast<value_type>(compute_type{1} - mask) * m_cvalue;
                else
                    return static_cast<value_type>(mask) * m_cvalue;
            }
        }

    private:
        vector3_type m_center;
        compute_type m_radius;
        compute_type m_radius_sqd;
        compute_type m_radius_edge_sqd;

        compute_type m_edge_size;
        compute_type m_length;
        compute_type m_length_edge;
        value_type m_cvalue;
    };
}

namespace noa::signal {
    /// Rectangle defined by a center and radius.
    /// \tparam INVERT  Whether the rectangle should be filled with zeros instead of ones.
    /// \tparam ICoord  Input, storage and computation type.
    /// \tparam OCoord  Output type.
    template<int NDIM, typename OCoord, bool INVERT = false, typename ICoord = float>
    class Rectangle {
    public:
        using value_type = OCoord;
        using compute_type = ICoord;
        using vector_type = std::conditional_t<NDIM == 2, Vec2<compute_type>, Vec3<compute_type>>;
        using rotation_type = std::conditional_t<NDIM == 2, Mat22<compute_type>, Mat33<compute_type>>;
        using affine_type = std::conditional_t<NDIM == 2, Mat23<compute_type>, Mat34<compute_type>>;

        constexpr NOA_IHD Rectangle(vector_type center, vector_type radius,
                                    value_type cvalue = value_type{1}) noexcept
                : m_center(center), m_radius(radius), m_cvalue(cvalue) {}

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, traits::Empty = traits::Empty{}) const noexcept {
            return get_(coords - m_center);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, const rotation_type& inv_transform) const noexcept {
            coords -= m_center;
            coords = inv_transform * coords;
            return get_(coords);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, const affine_type& inv_transform) const noexcept {
            using affine_vector_t = std::conditional_t<NDIM == 2, Vec3<compute_type>, Vec4<compute_type>>;
            affine_vector_t affine_coords{1}; // {y, x, 1} or {z, y, x, 1}
            for (size_t i = 0; i < vector_type::COUNT; ++i)
                affine_coords[i] = coords[i];
            return get_(inv_transform * affine_coords - m_center);
        }

    private:
        [[nodiscard]] constexpr NOA_IHD value_type get_(vector_type coords) const noexcept {
            coords = math::abs(coords);
            if (all(coords <= m_radius))
                return static_cast<value_type>(!INVERT) * m_cvalue;
            else
                return static_cast<value_type>(INVERT) * m_cvalue;
        }

    private:
        vector_type m_center;
        vector_type m_radius;
        value_type m_cvalue;
    };

    /// Smooth rectangle, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the rectangle should be filled with zeros instead of ones.
    /// \tparam ICoord  Input, storage and computation type.
    /// \tparam OCoord  Output type.
    template<int NDIM, typename OCoord, bool INVERT = false, typename ICoord = float>
    class RectangleSmooth {
    public:
        using value_type = OCoord;
        using compute_type = ICoord;
        using vector_type = std::conditional_t<NDIM == 2, Vec2<compute_type>, Vec3<compute_type>>;
        using rotation_type = std::conditional_t<NDIM == 2, Mat22<compute_type>, Mat33<compute_type>>;
        using affine_type = std::conditional_t<NDIM == 2, Mat23<compute_type>, Mat34<compute_type>>;

        constexpr NOA_IHD RectangleSmooth(vector_type center, vector_type radius, compute_type edge_size,
                                          value_type cvalue = value_type{1}) noexcept
                : m_center(center), m_radius(radius), m_radius_edge(radius + edge_size),
                  m_edge_size(edge_size), m_cvalue(cvalue) {}

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, traits::Empty = traits::Empty{}) const noexcept {
            return get_(coords - m_center);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, const rotation_type& inv_transform) const noexcept {
            coords -= m_center;
            coords = inv_transform * coords;
            return get_(coords);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, const affine_type& inv_transform) const noexcept {
            using affine_vector_t = std::conditional_t<NDIM == 2, Vec3<compute_type>, Vec4<compute_type>>;
            affine_vector_t affine_coords{1}; // {y, x, 1} or {z, y, x, 1}
            for (size_t i = 0; i < vector_type::COUNT; ++i)
                affine_coords[i] = coords[i];
            return get_(inv_transform * affine_coords - m_center);
        }

    private:
        [[nodiscard]] constexpr NOA_IHD value_type get_(vector_type coords) const noexcept {
            coords = math::abs(coords);

            if (any(m_radius_edge < coords)) {
                return static_cast<value_type>(INVERT) * m_cvalue;
            } else if (all(coords <= m_radius)) {
                return static_cast<value_type>(!INVERT) * m_cvalue;
            } else {
                compute_type mask_value{1};
                for (int i = 0; i < NDIM; ++i) {
                    if (m_radius[i] < coords[i] && coords[i] <= m_radius_edge[i]) {
                        constexpr auto PI = math::Constant<compute_type>::PI;
                        const auto distance = (coords[i] - m_radius[i]) / m_edge_size;
                        mask_value *= (compute_type{1} + math::cos(PI * distance)) * compute_type{0.5};
                    }
                }
                if constexpr (INVERT)
                    return static_cast<value_type>(compute_type{1} - mask_value) * m_cvalue;
                else
                    return static_cast<value_type>(mask_value) * m_cvalue;
            }
        }

    private:
        vector_type m_center;
        vector_type m_radius;
        vector_type m_radius_edge;
        compute_type m_edge_size;
        value_type m_cvalue;
    };
}

namespace noa::signal {
    /// Ellipse defined by a center and radius.
    /// \tparam INVERT  Whether the ellipse should be filled with zeros instead of ones.
    /// \tparam ICoord  Input, storage and computation type.
    /// \tparam OCoord  Output type.
    template<int NDIM, typename OCoord, bool INVERT = false, typename ICoord = float>
    class Ellipse {
    public:
        using value_type = OCoord;
        using compute_type = ICoord;
        using vector_type = std::conditional_t<NDIM == 2, Vec2<compute_type>, Vec3<compute_type>>;
        using rotation_type = std::conditional_t<NDIM == 2, Mat22<compute_type>, Mat33<compute_type>>;
        using affine_type = std::conditional_t<NDIM == 2, Mat23<compute_type>, Mat34<compute_type>>;

        constexpr NOA_IHD Ellipse(vector_type center, vector_type radius,
                                  value_type cvalue = value_type{1}) noexcept
                : m_center(center), m_radius(radius), m_cvalue(cvalue) {}

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, traits::Empty = traits::Empty{}) const noexcept {
            return get_(coords - m_center);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, const rotation_type& inv_transform) const noexcept {
            coords -= m_center;
            coords = inv_transform * coords;
            return get_(coords);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, const affine_type& inv_transform) const noexcept {
            using affine_vector_t = std::conditional_t<NDIM == 2, Vec3<compute_type>, Vec4<compute_type>>;
            affine_vector_t affine_coords{1}; // {y, x, 1} or {z, y, x, 1}
            for (size_t i = 0; i < vector_type::COUNT; ++i)
                affine_coords[i] = coords[i];
            return get_(inv_transform * affine_coords - m_center);
        }

    private:
        [[nodiscard]] constexpr NOA_IHD value_type get_(vector_type coords) const noexcept {
            coords /= m_radius;
            const compute_type rho = math::dot(coords, coords);
            if constexpr (INVERT)
                return static_cast<value_type>(rho > 1) * m_cvalue;
            else
                return static_cast<value_type>(rho <= 1) * m_cvalue;
        }

    private:
        vector_type m_center;
        vector_type m_radius;
        value_type m_cvalue;
    };

    /// Smooth ellipse, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the ellipse should be filled with zeros instead of ones.
    /// \tparam ICoord  Input, storage and computation type.
    /// \tparam OCoord  Output type.
    template<int NDIM, typename OCoord, bool INVERT = false, typename ICoord = float>
    class EllipseSmooth {
    public:
        using value_type = OCoord;
        using compute_type = ICoord;
        using vector_type = std::conditional_t<NDIM == 2, Vec2<compute_type>, Vec3<compute_type>>;
        using rotation_type = std::conditional_t<NDIM == 2, Mat22<compute_type>, Mat33<compute_type>>;
        using affine_type = std::conditional_t<NDIM == 2, Mat23<compute_type>, Mat34<compute_type>>;

        constexpr NOA_IHD EllipseSmooth(vector_type center, vector_type radius, compute_type edge_size,
                                        value_type cvalue = value_type{1}) noexcept
                : m_center(center), m_radius(radius * radius), m_edge_size(edge_size), m_cvalue(cvalue) {}

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, traits::Empty = traits::Empty{}) const noexcept {
            return get_(coords - m_center);
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, const rotation_type& inv_transform) const noexcept {
            coords -= m_center;
            coords = inv_transform * coords;
            return get_(coords); // for 2D we could accept the rotation angle and add it to phi.
        }

        [[nodiscard]] constexpr NOA_IHD value_type
        operator()(vector_type coords, const affine_type& inv_transform) const noexcept {
            using affine_vector_t = std::conditional_t<NDIM == 2, Vec3<compute_type>, Vec4<compute_type>>;
            affine_vector_t affine_coords{1}; // {y, x, 1} or {z, y, x, 1}
            for (size_t i = 0; i < vector_type::COUNT; ++i)
                affine_coords[i] = coords[i];
            return get_(inv_transform * affine_coords - m_center);
        }

    private:
        [[nodiscard]] constexpr NOA_IHD value_type get_(vector_type coords) const noexcept {
            compute_type irho, erho;
            if constexpr (NDIM == 2) {
                // Current spherical coordinate:
                irho = geometry::cartesian2rho(coords);
                const compute_type iphi = geometry::cartesian2phi<false>(coords);

                // Radius of the ellipse at (iphi, itheta):
                const compute_type cos2phi = math::pow(math::cos(iphi), compute_type{2});
                const compute_type sin2phi = math::pow(math::sin(iphi), compute_type{2});
                erho = compute_type{1} / math::sqrt(cos2phi / m_radius[1] + sin2phi / m_radius[0]);

            } else {
                irho = geometry::cartesian2rho(coords);
                const compute_type iphi = geometry::cartesian2phi<false>(coords);
                const compute_type itheta = geometry::cartesian2theta(coords);

                const compute_type cos2phi = math::pow(math::cos(iphi), compute_type{2});
                const compute_type sin2phi = math::pow(math::sin(iphi), compute_type{2});
                const compute_type cos2theta = math::pow(math::cos(itheta), compute_type{2});
                const compute_type sin2theta = math::pow(math::sin(itheta), compute_type{2});
                erho = compute_type{1} / math::sqrt(cos2phi * sin2theta / m_radius[2] +
                                                    sin2phi * sin2theta / m_radius[1] +
                                                    cos2theta / m_radius[0]);
            }

            constexpr compute_type PI = math::Constant<compute_type>::PI;
            if (irho > erho + m_edge_size) {
                return static_cast<value_type>(INVERT) * m_cvalue;
            } else if (irho <= erho) {
                return static_cast<value_type>(!INVERT) * m_cvalue;
            } else {
                const auto distance = (irho - erho) / m_edge_size;
                if constexpr (INVERT) {
                    return static_cast<value_type>(
                            (compute_type{1} - math::cos(PI * distance)) * compute_type{0.5}) * m_cvalue;
                } else {
                    return static_cast<value_type>(
                            (compute_type{1} + math::cos(PI * distance)) * compute_type{0.5}) * m_cvalue;
                }
            }
        }

    private:
        vector_type m_center;
        vector_type m_radius;
        compute_type m_edge_size{};
        value_type m_cvalue;
    };
}
