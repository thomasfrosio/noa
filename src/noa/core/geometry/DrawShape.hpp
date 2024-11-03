#pragma once

#include "noa/core/geometry/Polar.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Mat.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/geometry/Transform.hpp"

namespace noa::geometry::guts {
    template<typename GeometricShape, size_t N, typename Coord, typename Xform>
    [[nodiscard]] constexpr auto draw_geometric_shape_at(
        const GeometricShape& geometric_shape,
        const Vec<Coord, N>& coordinates,
        const Vec<Coord, N>& center,
        const Xform& inverse_transform = Xform{}
    ) {
        if constexpr (nt::empty<Xform> or nt::mat_of_shape<Xform, N, N> or nt::quaternion<Xform>) {
            return geometric_shape.draw_at(transform_vector(inverse_transform, coordinates - center));
        } else if constexpr (nt::mat_of_shape<Xform, N, N + 1> or nt::mat_of_shape<Xform, N + 1, N + 1>) {
            return geometric_shape.draw_at(transform_vector(inverse_transform, coordinates) - center);
        } else {
            static_assert(nt::always_false<>, "Transformation type is not supported");
        }
    }

    template<size_t N,
             nt::sinteger Index,
             typename DrawOp,
             typename BinaryOp,
             nt::batched_parameter Xform,
             nt::readable_nd_optional<N + 1> Input,
             nt::writable_nd<N + 1> Output>
    requires (N == 2 or N == 3)
    class DrawShape {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using draw_op_type = DrawOp;
        using binary_op_type = BinaryOp;
        using xform_type = Xform;

        using output_value_type = nt::value_type_t<output_type>;
        using draw_vec_type = Vec<nt::value_type_t<draw_op_type>, N>;

    public:
        constexpr DrawShape(
            const input_type& input,
            const output_type& output,
            const draw_op_type& draw_op,
            const xform_type& inverse_xform,
            const binary_op_type& binary_op
        ) :
            m_input(input),
            m_output(output),
            m_draw_op(draw_op),
            m_inverse_xform(inverse_xform),
            m_binary_op(binary_op) {}

        template<nt::same_as<index_type>... I> requires (N == sizeof...(I))
        NOA_HD constexpr void operator()(index_type batch, I... indices) const {
            draw_(batch, indices...);
        }

    private:
        template<typename T>
        NOA_HD constexpr auto default_op_(const T& input, auto shape) const {
            using scalar_t = nt::value_type_t<T>;
            if constexpr (nt::complex<T> and nt::real<output_value_type>) {
                return abs_squared(input) * static_cast<scalar_t>(shape);
            } else if constexpr (nt::real_or_complex<T>) {
                return input * static_cast<scalar_t>(shape);
            } else if constexpr (nt::integer<T>) {
                return input * static_cast<scalar_t>(round(shape));
            } else {
                static_assert(nt::always_false<>);
            }
        }

        NOA_HD constexpr auto draw_(index_type batch, auto... indices) const {
            const auto shape = m_draw_op(draw_vec_type::from_values(indices...), m_inverse_xform[batch]);
            if (m_input) {
                auto input = m_input(batch, indices...);
                if constexpr (nt::same_as<BinaryOp, Empty>) {
                    m_output(batch, indices...) = static_cast<output_value_type>(default_op_(input, shape));
                } else {
                    m_output(batch, indices...) = static_cast<output_value_type>(m_binary_op(input, shape));
                }
            } else {
                m_output(batch, indices...) = static_cast<nt::value_type_t<output_value_type>>(shape);
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        draw_op_type m_draw_op;
        NOA_NO_UNIQUE_ADDRESS xform_type m_inverse_xform;
        NOA_NO_UNIQUE_ADDRESS binary_op_type m_binary_op;
    };

    template<nt::any_of<f32, f64> Value, bool IsSmooth>
    class DrawLine {
    public:
        static constexpr bool IS_SMOOTH = IsSmooth;
        using value_type = Value;
        using value_or_empty_type = std::conditional_t<IS_SMOOTH, value_type, Empty>;

    public:
        constexpr DrawLine() = default;

        constexpr DrawLine(
            value_type center,
            value_type radius,
            value_type cvalue,
            bool is_inverted,
            value_type = {}
        ) requires (not IS_SMOOTH):
            m_cvalue(cvalue),
            m_center(center),
            m_radius(radius),
            m_is_inverted(is_inverted) {}

        constexpr DrawLine(
            value_type center,
            value_type radius,
            value_type cvalue,
            bool is_inverted,
            value_type smoothness
        ) requires IS_SMOOTH:
            m_cvalue(cvalue),
            m_center(center),
            m_radius(radius),
            m_smoothness(smoothness),
            m_is_inverted(is_inverted) {}

        [[nodiscard]] constexpr value_type operator()(nt::integer auto index, Empty = {}) const {
            return draw_at(static_cast<value_type>(index) - m_center);
        }

        [[nodiscard]] constexpr value_type draw_at(value_type centered_coordinate) const {
            const auto dst = abs(centered_coordinate);
            if constexpr (IS_SMOOTH) {
                constexpr auto PI = Constant<value_type>::PI;
                if (dst > m_radius + m_smoothness)
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                if (dst <= m_radius)
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
                auto weight = cos(PI * (dst - m_radius) / m_smoothness);
                if (m_is_inverted)
                    weight *= -1;
                return (value_type{1} + weight) * value_type{0.5} * m_cvalue;
            } else {
                if (dst > m_radius)
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
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

    template<size_t N, nt::any_of<f32, f64> T, bool IsSmooth>
    class DrawSphere {
    public:
        static constexpr bool IS_SMOOTH = IsSmooth;
        using value_type = T;
        using value_or_empty_type = std::conditional_t<IS_SMOOTH, value_type, Empty>;
        using vector_type = Vec<value_type, N>;

        constexpr DrawSphere() = default;

        constexpr DrawSphere(
            vector_type center,
            value_type radius,
            value_type cvalue,
            bool is_inverted,
            value_type = {}
        ) requires (not IS_SMOOTH):
            m_center(center),
            m_radius_sqd(radius * radius),
            m_cvalue(cvalue),
            m_is_inverted(is_inverted) {}

        constexpr DrawSphere(
            vector_type center,
            value_type radius,
            value_type cvalue,
            bool is_inverted,
            value_type smoothness
        ) requires IS_SMOOTH:
            m_center(center),
            m_radius_sqd(radius * radius),
            m_cvalue(cvalue),
            m_radius(radius),
            m_smoothness(smoothness),
            m_radius_edge_sqd((radius + smoothness) * (radius + smoothness)),
            m_is_inverted(is_inverted) {}

    public:
        template<typename X = Empty>
        [[nodiscard]] constexpr value_type operator()(
            vector_type coordinates,
            const X& inverse_transform = X{}
        ) const {
            return guts::draw_geometric_shape_at(*this, coordinates, m_center, inverse_transform);
        }

        [[nodiscard]] constexpr value_type draw_at(vector_type centered_coordinates) const {
            const auto dst_sqd = dot(centered_coordinates, centered_coordinates);
            if constexpr (IS_SMOOTH) {
                constexpr auto PI = Constant<value_type>::PI;
                if (dst_sqd > m_radius_edge_sqd)
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                if (dst_sqd <= m_radius_sqd)
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
                auto weight = cos(PI * (sqrt(dst_sqd) - m_radius) / m_smoothness);
                if (m_is_inverted)
                    weight *= -1;
                return (value_type{1} + weight) * value_type{0.5} * m_cvalue;
            } else {
                if (dst_sqd > m_radius_sqd)
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
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

    template<nt::any_of<f32, f64> Value, bool IsSmooth>
    class DrawCylinder {
    public:
        static constexpr bool IS_SMOOTH = IsSmooth;
        using value_type = Value;
        using value_or_empty_type = std::conditional_t<IS_SMOOTH, value_type, Empty>;
        using vector3_type = Vec3<value_type>;
        using vector2_type = Vec2<value_type>;

        constexpr DrawCylinder() = default;

        constexpr DrawCylinder(
            vector3_type center,
            vector2_type radius_length,
            value_type cvalue,
            bool is_inverted,
            value_type = {}
        ) requires (not IS_SMOOTH):
            m_center(center),
            m_radius_sqd(radius_length[0] * radius_length[0]),
            m_length(radius_length[1]),
            m_cvalue(cvalue),
            m_is_inverted(is_inverted) {}

        constexpr DrawCylinder(
            vector3_type center,
            vector2_type radius_length,
            value_type cvalue,
            bool is_inverted,
            value_type smoothness
        ) requires IS_SMOOTH:
            m_center(center),
            m_radius_sqd(radius_length[0] * radius_length[0]),
            m_length(radius_length[1]),
            m_cvalue(cvalue),
            m_radius(radius_length[0]),
            m_radius_edge_sqd((radius_length[0] + smoothness) * (radius_length[0] + smoothness)),
            m_edge_size(smoothness),
            m_length_edge(radius_length[1] + smoothness),
            m_is_inverted(is_inverted) {}

        template<typename T = Empty>
        [[nodiscard]] constexpr value_type operator()(
            vector3_type coordinates,
            const T& inverse_transform = T{}
        ) const {
            return guts::draw_geometric_shape_at(*this, coordinates, m_center, inverse_transform);
        }

        [[nodiscard]] constexpr value_type draw_at(vector3_type centered_coordinates) const {
            const auto dst_z = abs(centered_coordinates[0]);
            vector2_type tmp{centered_coordinates[1], centered_coordinates[2]};
            const value_type dst_yx_sqd = dot(tmp, tmp);

            if constexpr (IS_SMOOTH) {
                if (dst_z > m_length_edge or dst_yx_sqd > m_radius_edge_sqd)
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;

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
                return mask * m_cvalue;

            } else {
                if (dst_z > m_length or dst_yx_sqd > m_radius_sqd)
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
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

    template<size_t N, nt::any_of<f32, f64> Value, bool IsSmooth>
    class DrawRectangle {
    public:
        static constexpr bool IS_SMOOTH = IsSmooth;
        using value_type = Value;
        using vector_type = Vec<value_type, N>;
        using value_or_empty_type = std::conditional_t<IS_SMOOTH, value_type, Empty>;
        using vector_or_empty_type = std::conditional_t<IS_SMOOTH, vector_type, Empty>;

        constexpr DrawRectangle() = default;

        constexpr DrawRectangle(
            vector_type center,
            vector_type radius,
            value_type cvalue,
            bool is_inverted,
            value_type = {}
        ) requires (not IS_SMOOTH):
            m_center(center),
            m_radius(radius),
            m_cvalue(cvalue),
            m_is_inverted(is_inverted) {}

        constexpr DrawRectangle(
            vector_type center,
            vector_type radius,
            value_type cvalue,
            bool is_inverted,
            value_type smoothness
        ) requires IS_SMOOTH:
            m_center(center),
            m_radius(radius),
            m_cvalue(cvalue),
            m_radius_edge(radius + smoothness),
            m_smoothness(smoothness),
            m_is_inverted(is_inverted) {}

        template<typename T = Empty>
        [[nodiscard]] constexpr value_type operator()(
                vector_type coordinates,
                const T& inverse_transform = T{}
        ) const {
            return guts::draw_geometric_shape_at(*this, coordinates, m_center, inverse_transform);
        }

        [[nodiscard]] constexpr value_type draw_at(vector_type coords) const {
            coords = abs(coords);

            if constexpr (IS_SMOOTH) {
                if (any(m_radius_edge < coords))
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                if (all(coords <= m_radius))
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;

                value_type mask_value{1};
                for (size_t i{}; i < N; ++i) {
                    if (m_radius[i] < coords[i] and coords[i] <= m_radius_edge[i]) {
                        constexpr auto PI = Constant<value_type>::PI;
                        const auto distance = (coords[i] - m_radius[i]) / m_smoothness;
                        mask_value *= (value_type{1} + cos(PI * distance)) * value_type{0.5};
                    }
                }
                if (m_is_inverted)
                    return (value_type{1} - mask_value) * m_cvalue;
                return mask_value * m_cvalue;

            } else {
                if (all(coords <= m_radius))
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;
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

    template<size_t N, nt::any_of<f32, f64> Value, bool IsSmooth>
    class DrawEllipse {
    public:
        static constexpr bool IS_SMOOTH = IsSmooth;
        using value_type = Value;
        using vector_type = Vec<value_type, N>;
        using value_or_empty_type = std::conditional_t<IS_SMOOTH, value_type, Empty>;

        constexpr DrawEllipse() = default;

        constexpr DrawEllipse(
            vector_type center,
            vector_type radius,
            value_type cvalue,
            bool is_inverted,
            value_type = {}
        ) requires (not IS_SMOOTH):
            m_center(center),
            m_radius(radius),
            m_cvalue(cvalue),
            m_is_inverted(is_inverted) {}

        constexpr DrawEllipse(
            vector_type center,
            vector_type radius,
            value_type cvalue,
            bool is_inverted,
            value_type smoothness
        ) requires IS_SMOOTH:
            m_center(center),
            m_radius(radius * radius),
            m_cvalue(cvalue),
            m_smoothness(smoothness),
            m_is_inverted(is_inverted) {}

        // TODO for 2d we could also accept the rotation angle and add it to phi.
        template<typename T = Empty>
        [[nodiscard]] constexpr value_type operator()(
            vector_type coordinates,
            const T& inverse_transform = T{}
        ) const {
            return guts::draw_geometric_shape_at(*this, coordinates, m_center, inverse_transform);
        }

        [[nodiscard]] constexpr value_type draw_at(vector_type coordinates) const {
            if constexpr (IS_SMOOTH) {
                value_type irho, erho;
                if constexpr (N == 2) {
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
                if (irho > erho + m_smoothness)
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                if (irho <= erho)
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;

                const auto distance = (irho - erho) / m_smoothness;
                if (m_is_inverted)
                    return (value_type{1} - cos(PI * distance)) * value_type{0.5} * m_cvalue;
                return (value_type{1} + cos(PI * distance)) * value_type{0.5} * m_cvalue;

            } else {
                coordinates /= m_radius;
                const value_type rho = dot(coordinates, coordinates);
                if (m_is_inverted)
                    return static_cast<value_type>(rho > 1) * m_cvalue;
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
