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
    template<size_t N, typename Index,
             typename DrawOp,
             typename Xform,
             typename BinaryOp,
             typename InputAccessor,
             typename OutputAccessor>
    requires ((N == 2 or N == 3) and nt::is_sint_v<Index> and
              nt::are_accessor_pure_nd<N + 1, InputAccessor, OutputAccessor>::value and
              (nt::is_accessor_nd<Xform, 1>::value or std::is_empty_v<Xform>))
    class DrawShape {
    public:
        using index_type = Index;
        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using draw_op_type = DrawOp;
        using binary_op_type = BinaryOp;
        using xform_type = Xform;

        using output_value_type = nt::value_type_t<output_accessor_type>;
        using coord_type = draw_op_type::value_type;
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
            m_draw_op(draw_op),
            m_inverse_xform(inverse_xform),
            m_binary_op(binary_op) {}

        NOA_HD constexpr void operator()(index_type batch, index_type ik, index_type il) const requires (N == 2) {
            draw_(batch, ik, il);
        }

        NOA_HD constexpr void operator()(index_type batch, index_type ij, index_type ik, index_type il) const requires (N == 3) {
            draw_(batch, ij, ik, il);
        }

    private:
        NOA_HD constexpr auto get_shape_(index_type batch, const index_nd_type& indices) const {
            if constexpr (std::is_empty_v<Xform>) {
                return m_draw_op(coord_nd_type::from_vec(indices));
            } else {
                return m_draw_op(coord_nd_type::from_vec(indices), m_inverse_xform[batch]);
            }
        }

        NOA_HD constexpr auto default_op_(const auto& input, auto shape) const {
            using value_t = decltype(input);
            using scalar_t = nt::value_type_t<value_t>;
            if constexpr (nt::is_complex_v<value_t> and nt::is_real_v<output_value_type>) {
                return abs_squared(input) * static_cast<scalar_t>(shape);
            } else if constexpr (nt::is_real_or_complex_v<value_t>) {
                return input * static_cast<scalar_t>(shape);
            } else if constexpr (nt::is_int_v<value_t>) {
                return input * static_cast<scalar_t>(round(shape));
            } else {
                static_assert(nt::always_false_v<value_t>,
                              "Default operator does not support this input value type. "
                              "Please provide an operator to support this type. "
                              "Note that the shape type is f64 for non-default-supported types");
            }
        }

        NOA_HD constexpr auto draw_(index_type batch, auto... indices) const {
            const auto shape = get_shape_(batch, index_nd_type{indices...});
            if (m_input) {
                auto input = m_input(batch, indices...);
                if constexpr (std::is_empty_v<BinaryOp>) {
                    m_output(batch, indices...) = static_cast<output_value_type>(default_op_(input, shape));
                } else {
                    m_output(batch, indices...) = static_cast<output_value_type>(m_binary_op(input, shape));
                }
            } else {
                m_output(batch, indices...) = static_cast<output_value_type>(shape);
            }
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        draw_op_type m_draw_op;
        NOA_NO_UNIQUE_ADDRESS xform_type m_inverse_xform;
        NOA_NO_UNIQUE_ADDRESS binary_op_type m_binary_op;
    };
}

namespace noa::geometry {
    template<typename Value, bool IsSmooth>
    requires nt::is_any<Value, f32, f64>::value
    class DrawLine {
    public:
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using value_or_empty_type = std::conditional_t<is_smooth, value_type, Empty>;

    public:
        NOA_HD constexpr DrawLine(
                value_type center,
                value_type radius,
                value_type cvalue,
                bool is_inverted,
                value_type = {}
        ) noexcept requires (not is_smooth)
                : m_cvalue(cvalue), m_center(center), m_radius(radius), m_is_inverted(is_inverted) {}

        NOA_HD constexpr DrawLine(
                value_type center,
                value_type radius,
                value_type cvalue,
                bool is_inverted,
                value_type smoothness
        ) noexcept requires is_smooth
                : m_cvalue(cvalue), m_center(center), m_radius(radius),
                  m_smoothness(smoothness), m_is_inverted(is_inverted) {}

        [[nodiscard]] NOA_HD constexpr value_type operator()(std::integral auto index, Empty = {}) const noexcept {
            return draw_at(static_cast<value_type>(index) - m_center);
        }

        [[nodiscard]] NOA_HD constexpr value_type draw_at(value_type centered_coordinate) const noexcept {
            const auto dst = abs(centered_coordinate);

            if constexpr (is_smooth) {
                constexpr auto PI = Constant<value_type>::PI;
                if (dst > m_radius + m_smoothness) {
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                } else if (dst <= m_radius) {
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
                } else {
                    const auto weight = cos(PI * (dst - m_radius) / m_smoothness);
                    if (m_is_inverted)
                        return (value_type{1} - weight) * value_type{0.5} * m_cvalue;
                    else
                        return (value_type{1} + weight) * value_type{0.5} * m_cvalue;
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
        value_type m_center;
        value_type m_radius;
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_smoothness{};
        bool m_is_inverted;
    };

    template<size_t NDIM, typename Value, bool IsSmooth>
    requires nt::is_any<Value, f32, f64>::value
    class DrawSphere {
    public:
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using value_or_empty_type = std::conditional_t<is_smooth, value_type, Empty>;
        using vector_type = Vec<value_type, NDIM>;

        NOA_HD constexpr DrawSphere(
                vector_type center,
                value_type radius,
                value_type cvalue,
                bool is_inverted,
                value_type = {}
        ) noexcept requires (not is_smooth)
                : m_center(center),
                  m_radius_sqd(radius * radius),
                  m_cvalue(cvalue),
                  m_is_inverted(is_inverted) {}

        NOA_HD constexpr DrawSphere(
                vector_type center,
                value_type radius,
                value_type cvalue,
                bool is_inverted,
                value_type smoothness
        ) noexcept requires is_smooth
                : m_center(center),
                  m_radius_sqd(radius * radius),
                  m_cvalue(cvalue),
                  m_radius(radius),
                  m_smoothness(smoothness),
                  m_radius_edge_sqd((radius + smoothness) * (radius + smoothness)),
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
                constexpr auto PI = Constant<value_type>::PI;
                if (dst_sqd > m_radius_edge_sqd) {
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                } else if (dst_sqd <= m_radius_sqd) {
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
                } else {
                    const auto dst = sqrt(dst_sqd);
                    const auto weight = cos(PI * (dst - m_radius) / m_smoothness);
                    if (m_is_inverted)
                        return (value_type{1} - weight) * value_type{0.5} * m_cvalue;
                    else
                        return (value_type{1} + weight) * value_type{0.5} * m_cvalue;
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
        value_type m_radius_sqd;
        value_type m_cvalue;
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_radius{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_smoothness{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_radius_edge_sqd{};
        bool m_is_inverted;
    };

    template<typename Value, bool IsSmooth>
    requires nt::is_any<Value, f32, f64>::value
    class DrawCylinder {
    public:
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using value_or_empty_type = std::conditional_t<is_smooth, value_type, Empty>;
        using vector3_type = Vec3<value_type>;
        using vector2_type = Vec2<value_type>;

        NOA_HD constexpr DrawCylinder(
                vector3_type center,
                vector2_type radius_length,
                value_type cvalue,
                bool is_inverted,
                value_type = {}
        ) noexcept requires (not is_smooth)
                : m_center(center),
                  m_radius_sqd(radius_length[0] * radius_length[0]),
                  m_length(radius_length[1]),
                  m_cvalue(cvalue),
                  m_is_inverted(is_inverted) {}

        NOA_HD constexpr DrawCylinder(
                vector3_type center,
                vector2_type radius_length,
                value_type cvalue,
                bool is_inverted,
                value_type smoothness
        ) noexcept requires is_smooth
                : m_center(center),
                  m_radius_sqd(radius_length[0] * radius_length[0]),
                  m_length(radius_length[1]),
                  m_cvalue(cvalue),
                  m_radius(radius_length[0]),
                  m_radius_edge_sqd((radius_length[0] + smoothness) * (radius_length[0] + smoothness)),
                  m_edge_size(smoothness),
                  m_length_edge(radius_length[1] + smoothness),
                  m_is_inverted(is_inverted) {}

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
            const value_type dst_yx_sqd = dot(tmp, tmp);

            if constexpr (is_smooth) {
                if (dst_z > m_length_edge or dst_yx_sqd > m_radius_edge_sqd) {
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                } else {
                    constexpr auto PI = Constant<value_type>::PI;
                    value_type mask;
                    if (dst_yx_sqd <= m_radius_sqd) {
                        mask = value_type{1};
                    } else {
                        const auto dst_yx = (sqrt(dst_yx_sqd) - m_radius) / m_edge_size;
                        mask = (value_type{1} + cos(PI * dst_yx)) * value_type{0.5};
                    }

                    if (dst_z > m_length) {
                        const auto dst = (dst_z - m_length) / m_edge_size;
                        mask *= (value_type{1} + cos(PI * dst)) * value_type{0.5};
                    }

                    if (m_is_inverted)
                        return (value_type{1} - mask) * m_cvalue;
                    else
                        return mask * m_cvalue;
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
        value_type m_radius_sqd;
        value_type m_length;
        value_type m_cvalue;
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_radius{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_radius_edge_sqd{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_edge_size{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_length_edge{};
        bool m_is_inverted;
    };

    template<size_t NDIM, typename Value, bool IsSmooth>
    requires nt::is_any<Value, f32, f64>::value
    class DrawRectangle {
    public:
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using vector_type = Vec<value_type, NDIM>;
        using value_or_empty_type = std::conditional_t<is_smooth, value_type, Empty>;
        using vector_or_empty_type = std::conditional_t<is_smooth, vector_type, Empty>;

        NOA_HD constexpr DrawRectangle(
                vector_type center,
                vector_type radius,
                value_type cvalue,
                bool is_inverted,
                value_type = {}
        ) noexcept requires (not is_smooth)
                : m_center(center), m_radius(radius), m_cvalue(cvalue), m_is_inverted(is_inverted) {}

        NOA_HD constexpr DrawRectangle(
                vector_type center,
                vector_type radius,
                value_type cvalue,
                bool is_inverted,
                value_type smoothness
        ) noexcept requires is_smooth
                : m_center(center),
                  m_radius(radius),
                  m_cvalue(cvalue),
                  m_radius_edge(radius + smoothness),
                  m_smoothness(smoothness),
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
                    value_type mask_value{1};
                    for (size_t i = 0; i < NDIM; ++i) {
                        if (m_radius[i] < coords[i] and coords[i] <= m_radius_edge[i]) {
                            constexpr auto PI = Constant<value_type>::PI;
                            const auto distance = (coords[i] - m_radius[i]) / m_smoothness;
                            mask_value *= (value_type{1} + cos(PI * distance)) * value_type{0.5};
                        }
                    }
                    if (m_is_inverted)
                        return (value_type{1} - mask_value) * m_cvalue;
                    else
                        return mask_value * m_cvalue;
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
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_smoothness{};
        bool m_is_inverted;
    };

    template<size_t NDIM, typename Value, bool IsSmooth>
    requires nt::is_any<Value, f32, f64>::value
    class DrawEllipse {
    public:
        static constexpr bool is_smooth = IsSmooth;
        using value_type = Value;
        using vector_type = Vec<value_type, NDIM>;
        using value_or_empty_type = std::conditional_t<is_smooth, value_type, Empty>;

        NOA_HD constexpr DrawEllipse(
                vector_type center,
                vector_type radius,
                value_type cvalue,
                bool is_inverted,
                value_type = {}
        ) noexcept requires (not is_smooth)
                : m_center(center), m_radius(radius), m_cvalue(cvalue), m_is_inverted(is_inverted) {}

        NOA_HD constexpr DrawEllipse(
                vector_type center,
                vector_type radius,
                value_type cvalue,
                bool is_inverted,
                value_type smoothness
        ) noexcept requires is_smooth
                : m_center(center), m_radius(radius * radius), m_cvalue(cvalue),
                  m_smoothness(smoothness), m_is_inverted(is_inverted) {}

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
                value_type irho, erho;
                if constexpr (NDIM == 2) {
                    // Current spherical coordinate:
                    irho = cartesian2rho(coordinates);
                    const value_type iphi = cartesian2phi<false>(coordinates);

                    // Radius of the ellipse at (iphi, itheta):
                    const value_type cos2phi = pow(cos(iphi), value_type{2});
                    const value_type sin2phi = pow(sin(iphi), value_type{2});
                    erho = value_type{1} / sqrt(cos2phi / m_radius[1] + sin2phi / m_radius[0]);

                } else {
                    irho = cartesian2rho(coordinates);
                    const value_type iphi = cartesian2phi<false>(coordinates);
                    const value_type itheta = cartesian2theta(coordinates);

                    const value_type cos2phi = pow(cos(iphi), value_type{2});
                    const value_type sin2phi = pow(sin(iphi), value_type{2});
                    const value_type cos2theta = pow(cos(itheta), value_type{2});
                    const value_type sin2theta = pow(sin(itheta), value_type{2});
                    erho = value_type{1} / sqrt(cos2phi * sin2theta / m_radius[2] +
                                                sin2phi * sin2theta / m_radius[1] +
                                                cos2theta / m_radius[0]);
                }

                constexpr value_type PI = Constant<value_type>::PI;
                if (irho > erho + m_smoothness) {
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                } else if (irho <= erho) {
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
                } else {
                    const auto distance = (irho - erho) / m_smoothness;
                    if (m_is_inverted) {
                        return (value_type{1} - cos(PI * distance)) * value_type{0.5} * m_cvalue;
                    } else {
                        return (value_type{1} + cos(PI * distance)) * value_type{0.5} * m_cvalue;
                    }
                }
            } else {
                coordinates /= m_radius;
                const value_type rho = dot(coordinates, coordinates);
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
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_smoothness{};
        bool m_is_inverted;
    };
}
