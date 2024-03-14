#pragma once

#include "noa/core/geometry/Polar.hpp"
#include "noa/core/geometry/Quaternion.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Mat.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::geometry::guts {
    template<typename GeometricShape, size_t N, typename Coord, typename Xform>
    [[nodiscard]] NOA_FHD constexpr auto draw_geometric_shape_at(
            const GeometricShape& geometric_shape,
            const Vec<Coord, N>& coordinates,
            const Vec<Coord, N>& center,
            const Xform& inverse_transform = Xform{}
    ) noexcept {
        if constexpr (std::is_empty_v<Xform>) {
            return geometric_shape.draw_at(coordinates - center);
        } else if constexpr ((N == 2 and std::is_same_v<Xform, Mat22<Coord>>) or
                             (N == 3 and std::is_same_v<Xform, Mat33<Coord>>)) {
            return geometric_shape.draw_at(inverse_transform * (coordinates - center));
        } else if constexpr ((N == 2 and nt::is_any_v<Xform, Mat23<Coord>>) or
                             (N == 3 and nt::is_any_v<Xform, Mat34<Coord>>)) {
            return geometric_shape.draw_at(inverse_transform * coordinates.push_back(1) - center);
        } else if constexpr ((N == 2 and nt::is_any_v<Xform, Mat33<Coord>>) or
                             (N == 3 and nt::is_any_v<Xform, Mat44<Coord>>)) {
            return geometric_shape.draw_at((inverse_transform * coordinates.push_back(1)).pop_back() - center);
        } else if constexpr (N == 3 and nt::is_any_v<Xform, Quaternion<Coord>>) {
            return geometric_shape.draw_at(inverse_transform.rotate(coordinates - center));
        } else {
            static_assert(nt::always_false_v<Xform>, "Transformation type is not supported");
        }
    }
}

namespace noa::geometry {
    template<typename Value, typename Coord, bool IsSmooth>
    class DrawLine {
    public:
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using coord_type = Coord;
        using coord_or_empty_type = std::conditional_t<is_smooth, coord_type, Empty>;

    public:
        NOA_HD constexpr DrawLine(
                coord_type center,
                coord_type radius,
                value_type cvalue,
                bool is_inverted
        ) noexcept requires (not is_smooth)
                : m_cvalue(cvalue), m_center(center), m_radius(radius), m_is_inverted(is_inverted) {}

        NOA_HD constexpr DrawLine(
                coord_type center,
                coord_type radius,
                coord_type edge_size,
                value_type cvalue,
                bool is_inverted
        ) noexcept requires is_smooth
                : m_cvalue(cvalue), m_center(center), m_radius(radius),
                  m_edge_size(edge_size), m_is_inverted(is_inverted) {}

        [[nodiscard]] NOA_HD constexpr value_type operator()(std::integral auto index, Empty = {}) const noexcept {
            return draw_at(static_cast<coord_type>(index) - m_center);
        }

        [[nodiscard]] NOA_HD constexpr value_type draw_at(coord_type centered_coordinate) const noexcept {
            const auto dst = abs(centered_coordinate);

            if constexpr (is_smooth) {
                constexpr auto PI = Constant<coord_type>::PI;
                if (dst > m_radius + m_edge_size) {
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                } else if (dst <= m_radius) {
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
                } else {
                    const auto weight = cos(PI * (dst - m_radius) / m_edge_size);
                    if (m_is_inverted)
                        return static_cast<value_type>((coord_type{1} - weight) * coord_type{0.5}) * m_cvalue;
                    else
                        return static_cast<value_type>((coord_type{1} + weight) * coord_type{0.5}) * m_cvalue;
                }
            } else {
                if (dst > m_radius)
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                else
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
            }
        }

    private:
        value_type m_cvalue;
        coord_type m_center;
        coord_type m_radius;
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_edge_size{};
        bool m_is_inverted;
    };

    template<size_t NDIM, typename Value, typename Coord, bool IsSmooth>
    class DrawSphere {
    public:
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using coord_type = Coord;
        using coord_or_empty_type = std::conditional_t<is_smooth, coord_type, Empty>;
        using vector_type = Vec<coord_type, NDIM>;

        NOA_HD constexpr DrawSphere(
                vector_type center,
                coord_type radius,
                value_type cvalue,
                bool is_inverted
        ) noexcept requires (not is_smooth)
                : m_center(center),
                  m_radius_sqd(radius * radius),
                  m_cvalue(cvalue),
                  m_is_inverted(is_inverted) {}

        NOA_HD constexpr DrawSphere(
                vector_type center,
                coord_type radius,
                coord_type edge_size,
                value_type cvalue,
                bool is_inverted
        ) noexcept requires is_smooth
                : m_center(center),
                  m_radius_sqd(radius * radius),
                  m_cvalue(cvalue),
                  m_radius(radius),
                  m_edge_size(edge_size),
                  m_radius_edge_sqd((radius + edge_size) * (radius + edge_size)),
                  m_is_inverted(is_inverted) {}

    public:
        template<typename T = Empty>
        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector_type coordinates,
                const T& inverse_transform = T{}
        ) const noexcept {
            return guts::draw_geometric_shape_at(*this, coordinates, m_center, inverse_transform);
        }

        [[nodiscard]] NOA_HD constexpr value_type draw_at(vector_type centered_coordinates) const noexcept {
            const auto dst_sqd = dot(centered_coordinates, centered_coordinates);
            if constexpr (is_smooth) {
                constexpr auto PI = Constant<coord_type>::PI;
                if (dst_sqd > m_radius_edge_sqd) {
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                } else if (dst_sqd <= m_radius_sqd) {
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
                } else {
                    const auto dst = sqrt(dst_sqd);
                    const auto weight = cos(PI * (dst - m_radius) / m_edge_size);
                    if (m_is_inverted)
                        return static_cast<value_type>((coord_type{1} - weight) * coord_type{0.5}) * m_cvalue;
                    else
                        return static_cast<value_type>((coord_type{1} + weight) * coord_type{0.5}) * m_cvalue;
                }
            } else {
                if (dst_sqd > m_radius_sqd)
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                else
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
            }
        }

    private:
        vector_type m_center;
        coord_type m_radius_sqd;
        value_type m_cvalue;
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_radius{};
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_edge_size{};
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_radius_edge_sqd{};
        bool m_is_inverted;
    };

    template<typename Value, typename Coord, bool IsSmooth>
    class DrawCylinder {
    public:
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using coord_type = Coord;
        using coord_or_empty_type = std::conditional_t<is_smooth, coord_type, Empty>;
        using vector3_type = Vec3<coord_type>;
        using vector2_type = Vec2<coord_type>;

        NOA_HD constexpr DrawCylinder(
                vector3_type center,
                coord_type radius,
                coord_type length,
                value_type cvalue,
                bool is_inverted
        ) noexcept requires (not is_smooth)
                : m_center(center),
                  m_radius_sqd(radius * radius),
                  m_length(length),
                  m_cvalue(cvalue),
                  m_is_inverted(is_inverted) {}

        NOA_HD constexpr DrawCylinder(
                vector3_type center,
                coord_type radius,
                coord_type length,
                coord_type edge_size,
                value_type cvalue,
                bool is_inverted
        ) noexcept requires is_smooth
                : m_center(center),
                  m_radius_sqd(radius * radius),
                  m_length(length),
                  m_cvalue(cvalue),
                  m_radius(radius),
                  m_radius_edge_sqd((radius + edge_size) * (radius + edge_size)),
                  m_edge_size(edge_size),
                  m_length_edge(length + edge_size),
                  m_is_inverted(is_inverted) {}

        NOA_HD constexpr DrawCylinder(
                vector3_type center,
                vector2_type length_radius,
                value_type cvalue,
                bool is_inverted
        ) noexcept requires (not is_smooth)
                : DrawCylinder(center, length_radius[1], length_radius[0], cvalue, is_inverted) {}

        NOA_HD constexpr DrawCylinder(
                vector3_type center,
                vector3_type length_radius,
                value_type cvalue,
                bool is_inverted
        ) noexcept requires (not is_smooth)
                : DrawCylinder(center, length_radius[1], length_radius[0], cvalue, is_inverted) {
            NOA_ASSERT(length_radius[1] == length_radius[2]);
        }

        NOA_HD constexpr DrawCylinder(
                vector3_type center, vector2_type length_radius,
                coord_type edge_size, value_type cvalue,
                bool is_inverted
        ) noexcept requires is_smooth
                : DrawCylinder(center, length_radius[1], length_radius[0], edge_size, cvalue, is_inverted) {}

        NOA_HD constexpr DrawCylinder(
                vector3_type center, vector3_type length_radius,
                coord_type edge_size, value_type cvalue,
                bool is_inverted
        ) noexcept requires is_smooth
                : DrawCylinder(center, length_radius[1], length_radius[0], edge_size, cvalue, is_inverted) {
            NOA_ASSERT(length_radius[1] == length_radius[2]);
        }

        template<typename T = Empty>
        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector3_type coordinates,
                const T& inverse_transform = T{}
        ) const noexcept {
            return guts::draw_geometric_shape_at(*this, coordinates, m_center, inverse_transform);
        }

        [[nodiscard]] NOA_HD constexpr value_type draw_at(vector3_type centered_coordinates) const noexcept {
            const auto dst_z = abs(centered_coordinates[0]);
            vector2_type tmp{centered_coordinates[1], centered_coordinates[2]};
            const coord_type dst_yx_sqd = dot(tmp, tmp);

            if constexpr (is_smooth) {
                if (dst_z > m_length_edge or dst_yx_sqd > m_radius_edge_sqd) {
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
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

                    if (m_is_inverted)
                        return static_cast<value_type>(coord_type{1} - mask) * m_cvalue;
                    else
                        return static_cast<value_type>(mask) * m_cvalue;
                }
            } else {
                if (dst_z > m_length or dst_yx_sqd > m_radius_sqd)
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                else
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
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
        bool m_is_inverted;
    };

    template<size_t NDIM, typename Value, typename Coord, bool IsSmooth>
    class DrawRectangle {
    public:
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using coord_type = Coord;
        using vector_type = Vec<coord_type, NDIM>;
        using coord_or_empty_type = std::conditional_t<is_smooth, coord_type, Empty>;
        using vector_or_empty_type = std::conditional_t<is_smooth, vector_type, Empty>;

        NOA_HD constexpr DrawRectangle(
                vector_type center, vector_type radius, value_type cvalue, bool is_inverted
        ) noexcept requires (not is_smooth)
                : m_center(center), m_radius(radius), m_cvalue(cvalue), m_is_inverted(is_inverted) {}

        NOA_HD constexpr DrawRectangle(
                vector_type center,
                vector_type radius,
                coord_type edge_size,
                value_type cvalue,
                bool is_inverted
        ) noexcept requires is_smooth
                : m_center(center),
                  m_radius(radius),
                  m_cvalue(cvalue),
                  m_radius_edge(radius + edge_size),
                  m_edge_size(edge_size),
                  m_is_inverted(is_inverted) {}

        template<typename T = Empty>
        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector_type coordinates,
                const T& inverse_transform = T{}
        ) const noexcept {
            return guts::draw_geometric_shape_at(*this, coordinates, m_center, inverse_transform);
        }

        [[nodiscard]] NOA_HD constexpr value_type draw_at(vector_type coords) const noexcept {
            coords = abs(coords);

            if constexpr (IsSmooth) {
                if (any(m_radius_edge < coords)) {
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                } else if (all(coords <= m_radius)) {
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
                } else {
                    coord_type mask_value{1};
                    for (size_t i = 0; i < NDIM; ++i) {
                        if (m_radius[i] < coords[i] and coords[i] <= m_radius_edge[i]) {
                            constexpr auto PI = Constant<coord_type>::PI;
                            const auto distance = (coords[i] - m_radius[i]) / m_edge_size;
                            mask_value *= (coord_type{1} + cos(PI * distance)) * coord_type{0.5};
                        }
                    }
                    if (m_is_inverted)
                        return static_cast<value_type>(coord_type{1} - mask_value) * m_cvalue;
                    else
                        return static_cast<value_type>(mask_value) * m_cvalue;
                }
            } else {
                if (all(coords <= m_radius))
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
                else
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
            }
        }

    private:
        vector_type m_center;
        vector_type m_radius;
        value_type m_cvalue;
        NOA_NO_UNIQUE_ADDRESS vector_or_empty_type m_radius_edge{};
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_edge_size{};
        bool m_is_inverted;
    };

    template<size_t NDIM, typename Value, typename Coord, bool IsSmooth>
    class DrawEllipse {
    public:
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using coord_type = Coord;
        using vector_type = Vec<coord_type, NDIM>;
        using coord_or_empty_type = std::conditional_t<is_smooth, coord_type, Empty>;

        NOA_HD constexpr DrawEllipse(
                vector_type center, vector_type radius, value_type cvalue, bool is_inverted
        ) noexcept requires (not is_smooth)
                : m_center(center), m_radius(radius), m_cvalue(cvalue), m_is_inverted(is_inverted) {}

        NOA_HD constexpr DrawEllipse(
                vector_type center,
                vector_type radius,
                coord_type edge_size,
                value_type cvalue,
                bool is_inverted
        ) noexcept requires is_smooth
                : m_center(center), m_radius(radius * radius), m_cvalue(cvalue),
                  m_edge_size(edge_size), m_is_inverted(is_inverted) {}

        // TODO for 2d we could also accept the rotation angle and add it to phi.
        template<typename T = Empty>
        [[nodiscard]] NOA_HD constexpr value_type operator()(
                vector_type coordinates,
                const T& inverse_transform = T{}
        ) const noexcept {
            return guts::draw_geometric_shape_at(*this, coordinates, m_center, inverse_transform);
        }

        [[nodiscard]] NOA_HD constexpr value_type draw_at(vector_type coordinates) const noexcept {
            if constexpr (IsSmooth) {
                coord_type irho, erho;
                if constexpr (NDIM == 2) {
                    // Current spherical coordinate:
                    irho = cartesian2rho(coordinates);
                    const coord_type iphi = cartesian2phi<false>(coordinates);

                    // Radius of the ellipse at (iphi, itheta):
                    const coord_type cos2phi = pow(cos(iphi), coord_type{2});
                    const coord_type sin2phi = pow(sin(iphi), coord_type{2});
                    erho = coord_type{1} / sqrt(cos2phi / m_radius[1] + sin2phi / m_radius[0]);

                } else {
                    irho = cartesian2rho(coordinates);
                    const coord_type iphi = cartesian2phi<false>(coordinates);
                    const coord_type itheta = cartesian2theta(coordinates);

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
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                } else if (irho <= erho) {
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
                } else {
                    const auto distance = (irho - erho) / m_edge_size;
                    if (m_is_inverted) {
                        return static_cast<value_type>((coord_type{1} - cos(PI * distance)) * coord_type{0.5}) * m_cvalue;
                    } else {
                        return static_cast<value_type>((coord_type{1} + cos(PI * distance)) * coord_type{0.5}) * m_cvalue;
                    }
                }
            } else {
                coordinates /= m_radius;
                const coord_type rho = dot(coordinates, coordinates);
                if (m_is_inverted)
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
        bool m_is_inverted;
    };
}

namespace noa::geometry {
    template<size_t N,
             typename Index, typename Coord,
             typename DrawOp, typename Xform, typename BinaryOp,
             typename InputAccessor, typename OutputAccessor>
    requires ((N == 2 or N == 3) and
              nt::is_sint_v<Index> and nt::is_any_v<Coord, f32, f64> and
              nt::are_accessor_pure_nd<N + 1, InputAccessor, OutputAccessor>::value)
    class DrawShape {
    public:
        using index_type = Index;
        using coord_type = Coord;
        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using output_value_type = nt::value_type_t<output_accessor_type>;
        using draw_op_type = DrawOp;
        using binary_op_type = BinaryOp;

        using xform_single_type = std::remove_const_t<std::remove_pointer_t<Xform>>;
        static constexpr bool has_multiple_xforms = std::is_pointer_v<Xform>;
        using xform_type = std::conditional_t<has_multiple_xforms, const xform_single_type*, xform_single_type>;

        using index_nd_type = Vec<index_type, N>;
        using coord_nd_type = Vec<coord_type, N>;

    public:
        DrawShape(
                const input_accessor_type& input,
                const output_accessor_type& output,
                const draw_op_type& draw_op,
                const xform_type& inverse_xform,
                const binary_op_type& binary_op
        ) : m_input(input),
            m_output(output),
            m_inverse_xform(inverse_xform),
            m_draw_op(draw_op),
            m_binary_op(binary_op) {}

        NOA_HD constexpr void operator()(index_type batch, index_type ik, index_type il) const noexcept requires (N == 2) {
            draw_(batch, index_nd_type{ik, il});
        }

        NOA_HD constexpr void operator()(index_type batch, index_type ij, index_type ik, index_type il) const noexcept requires (N == 3) {
            draw_(batch, index_nd_type{ij, ik, il});
        }

    private:
        NOA_HD constexpr auto get_shape_(index_type batch, const index_nd_type& indices) {
            if constexpr (has_multiple_xforms) {
               return m_draw_op(coord_nd_type::from_vec(indices), m_inverse_xform[batch]);
            } else {
               return m_draw_op(coord_nd_type::from_vec(indices), m_inverse_xform);
            }
        }

        NOA_HD constexpr auto draw_(index_type batch, const index_nd_type& indices) {
            const auto shape = get_shape_(batch, indices);
            if (m_input) {
                auto value = m_binary_op(m_input(indices.push_front(batch)), shape);
                m_output(indices) = static_cast<output_value_type>(value);
            } else {
                m_output(indices) = static_cast<output_value_type>(shape);
            }
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        draw_op_type m_draw_op;
        binary_op_type m_binary_op;
        NOA_NO_UNIQUE_ADDRESS xform_type m_inverse_xform;
    };
}
