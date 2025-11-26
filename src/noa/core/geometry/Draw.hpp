#pragma once

#include "noa/core/geometry/Polar.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/geometry/Transform.hpp"

namespace noa::traits {
    template<typename T, size_t N, typename Coord>
    concept drawable_transform =
        nt::empty<T> or
        (std::same_as<value_type_t<T>, Coord> and
         (nt::mat_of_shape<T, N, N> or
          nt::mat_of_shape<T, N, N + 1> or
          nt::mat_of_shape<T, N + 1, N + 1>
          or (N == 3 and nt::quaternion<T>)));

    template<typename T, size_t N, typename Coord, typename Xform>
    concept drawable = nt::any_of<Coord, f32, f64> and N == T::SIZE and drawable_transform<Xform, N, Coord> and
        requires (const T& object, const Vec<Coord, N>& vec, const Xform& xform) {
            /// Given centered coordinates, returns the drawing.
            /// Coordinates are centered onto the object center.
            { object.draw_at_centered(vec) } -> std::same_as<Coord>;

            /// Given output coordinates and an inverse transform, returns the drawing.
            /// This can usually be implemented using the nd::draw_at function and T::draw_at_centered.
            { object.draw_at(vec, xform) } -> std::same_as<Coord>;

            /// Object center.
            { object.center() } -> nt::almost_same_as<Vec<Coord, N>>;
        };
}

namespace noa::geometry {
    /// Draws at the transformed coordinates.
    template<typename Drawable, size_t N, typename Coord, typename Xform>
    requires nt::drawable<Drawable, N, Coord, Xform>
    [[nodiscard]] constexpr auto draw_at(
        const Drawable& drawable,
        const Vec<Coord, N>& coordinates,
        const Xform& inverse_transform = Xform{}
    ) {
        if constexpr (nt::empty<Xform> or nt::mat_of_shape<Xform, N, N> or nt::quaternion<Xform>) {
            return drawable.draw_at_centered(transform_vector(inverse_transform, coordinates - drawable.center()));
        } else if constexpr (nt::mat_of_shape<Xform, N, N + 1> or nt::mat_of_shape<Xform, N + 1, N + 1>) {
            return drawable.draw_at_centered(transform_vector(inverse_transform, coordinates) - drawable.center());
        } else {
            static_assert(nt::always_false<Xform>, "Transformation type is not supported");
        }
    }
}

namespace noa::geometry {
    template<size_t N, nt::any_of<f32, f64> T, bool IsSmooth>
    class DrawSphere {
    public:
        static constexpr bool IS_SMOOTH = IsSmooth;
        static constexpr size_t SIZE = N;
        using value_type = T;
        using value_or_empty_type = std::conditional_t<IS_SMOOTH, value_type, Empty>;
        using vector_type = Vec<value_type, N>;

        constexpr DrawSphere() = default;

        constexpr DrawSphere(
            vector_type center,
            value_type radius,
            value_type cvalue,
            bool is_inverted
        ) requires (not IS_SMOOTH):
            m_center(center),
            m_radius_sqd(radius * radius),
            m_cvalue(cvalue),
            m_is_inverted(is_inverted) {}

        constexpr DrawSphere(
            vector_type center,
            value_type radius,
            value_type cvalue,
            value_type smoothness,
            bool is_inverted
        ) requires IS_SMOOTH:
            m_center(center),
            m_radius_sqd(radius * radius),
            m_cvalue(cvalue),
            m_radius(radius),
            m_smoothness(smoothness),
            m_radius_edge_sqd((radius + smoothness) * (radius + smoothness)),
            m_is_inverted(is_inverted) {}

        public:
        [[nodiscard]] constexpr auto center() const -> const vector_type& { return m_center; }

        template<nt::drawable_transform<N, value_type> Xform = Empty>
        [[nodiscard]] constexpr auto draw_at(
            const vector_type& coordinates,
            const Xform& inverse_transform = Xform{}
        ) const -> value_type {
            return ng::draw_at(*this, coordinates, inverse_transform);
        }

        [[nodiscard]] constexpr auto draw_at_centered(const vector_type& centered_coordinates) const -> value_type {
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
        vector_type m_center{};
        value_type m_radius_sqd{};
        value_type m_cvalue{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_radius{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_smoothness{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_radius_edge_sqd{};
        bool m_is_inverted{};
    };

    template<size_t N, nt::any_of<f32, f64> Value, bool IsSmooth>
    class DrawEllipse {
    public:
        static constexpr bool IS_SMOOTH = IsSmooth;
        static constexpr size_t SIZE = N;
        using value_type = Value;
        using vector_type = Vec<value_type, N>;
        using value_or_empty_type = std::conditional_t<IS_SMOOTH, value_type, Empty>;

        constexpr DrawEllipse() = default;

        constexpr DrawEllipse(
            vector_type center,
            vector_type radius,
            value_type cvalue,
            bool is_inverted
        ) requires (not IS_SMOOTH):
            m_center(center),
            m_radius(noa::max(radius, static_cast<value_type>(1e-6))),
            m_cvalue(cvalue),
            m_is_inverted(is_inverted) {}

        constexpr DrawEllipse(
            vector_type center,
            vector_type radius,
            value_type cvalue,
            value_type smoothness,
            bool is_inverted
        ) requires IS_SMOOTH:
            m_center(center),
            m_radius(noa::max(radius, static_cast<value_type>(1e-6))),
            m_cvalue(cvalue),
            m_smoothness(smoothness),
            m_is_inverted(is_inverted)
        {
            m_radius *= m_radius;
        }

            [[nodiscard]] constexpr auto center() const -> const vector_type& { return m_center; }

        // TODO for 2d we could also accept the rotation angle and add it to phi.
        template<nt::drawable_transform<N, value_type> Xform = Empty>
        [[nodiscard]] constexpr auto draw_at(
            const vector_type& coordinates,
            const Xform& inverse_transform = Xform{}
        ) const -> value_type {
            return ng::draw_at(*this, coordinates, inverse_transform);
        }

        [[nodiscard]] constexpr auto draw_at_centered(vector_type centered_coordinates) const -> value_type {
            if constexpr (IS_SMOOTH) {
                value_type irho, erho;
                if constexpr (N == 2) {
                    // Current spherical coordinate:
                    irho = cartesian2rho(centered_coordinates);
                    const value_type iphi = cartesian2phi<false>(centered_coordinates);

                    // Radius of the ellipse at (iphi, itheta):
                    const value_type cos2phi = pow(cos(iphi), value_type{2});
                    const value_type sin2phi = pow(sin(iphi), value_type{2});
                    erho = value_type{1} / sqrt(cos2phi / m_radius[1] + sin2phi / m_radius[0]);
                } else {
                    irho = cartesian2rho(centered_coordinates);
                    const value_type iphi = cartesian2phi<false>(centered_coordinates);
                    const value_type itheta = cartesian2theta(centered_coordinates);

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
                centered_coordinates /= m_radius;
                const value_type rho = dot(centered_coordinates, centered_coordinates);
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
        bool m_is_inverted{};
    };

    template<size_t N, nt::any_of<f32, f64> Value, bool IsSmooth>
    class DrawRectangle {
    public:
        static constexpr bool IS_SMOOTH = IsSmooth;
        static constexpr size_t SIZE = N;
        using value_type = Value;
        using vector_type = Vec<value_type, N>;
        using value_or_empty_type = std::conditional_t<IS_SMOOTH, value_type, Empty>;
        using vector_or_empty_type = std::conditional_t<IS_SMOOTH, vector_type, Empty>;

        constexpr DrawRectangle() = default;

        constexpr DrawRectangle(
            vector_type center,
            vector_type radius,
            value_type cvalue,
            bool is_inverted
        ) requires (not IS_SMOOTH):
            m_center(center),
            m_radius(radius),
            m_cvalue(cvalue),
            m_is_inverted(is_inverted) {}

        constexpr DrawRectangle(
            vector_type center,
            vector_type radius,
            value_type cvalue,
            value_type smoothness,
            bool is_inverted
        ) requires IS_SMOOTH:
            m_center(center),
            m_radius(radius),
            m_cvalue(cvalue),
            m_radius_edge(radius + smoothness),
            m_smoothness(smoothness),
            m_is_inverted(is_inverted) {}

        [[nodiscard]] constexpr auto center() const -> const vector_type& { return m_center; }

        // TODO for 2d we could also accept the rotation angle and add it to phi.
        template<nt::drawable_transform<N, value_type> Xform = Empty>
        [[nodiscard]] constexpr auto draw_at(
            const vector_type& coordinates,
            const Xform& inverse_transform = Xform{}
        ) const -> value_type {
            return ng::draw_at(*this, coordinates, inverse_transform);
        }

        [[nodiscard]] constexpr auto draw_at_centered(vector_type centered_coordinates) const -> value_type {
            centered_coordinates = abs(centered_coordinates);

            if constexpr (IS_SMOOTH) {
                if (any(m_radius_edge < centered_coordinates))
                    return static_cast<value_type>(m_is_inverted) * m_cvalue;
                if (all(centered_coordinates <= m_radius))
                    return static_cast<value_type>(not m_is_inverted) * m_cvalue;

                value_type mask_value{1};
                for (size_t i{}; i < N; ++i) {
                    if (m_radius[i] < centered_coordinates[i] and centered_coordinates[i] <= m_radius_edge[i]) {
                        constexpr auto PI = Constant<value_type>::PI;
                        const auto distance = (centered_coordinates[i] - m_radius[i]) / m_smoothness;
                        mask_value *= (value_type{1} + cos(PI * distance)) * value_type{0.5};
                    }
                }
                if (m_is_inverted)
                    return (value_type{1} - mask_value) * m_cvalue;
                return mask_value * m_cvalue;
            } else {
                if (all(centered_coordinates <= m_radius))
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
        bool m_is_inverted{};
    };

    template<nt::any_of<f32, f64> Value, bool IsSmooth>
    class DrawCylinder {
    public:
        static constexpr bool IS_SMOOTH = IsSmooth;
        static constexpr size_t SIZE = 3;
        using value_type = Value;
        using value_or_empty_type = std::conditional_t<IS_SMOOTH, value_type, Empty>;
        using vector3_type = Vec3<value_type>;
        using vector2_type = Vec2<value_type>;

        constexpr DrawCylinder(
            vector3_type center,
            vector2_type radius_length,
            value_type cvalue,
            bool is_inverted
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
            value_type smoothness,
            bool is_inverted
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

        [[nodiscard]] constexpr auto center() const -> const vector3_type& { return m_center; }

        template<nt::drawable_transform<3, value_type> Xform = Empty>
        [[nodiscard]] constexpr auto draw_at(
            const vector3_type& coordinates,
            const Xform& inverse_transform = Xform{}
        ) const -> value_type {
            return ng::draw_at(*this, coordinates, inverse_transform);
        }

        [[nodiscard]] constexpr auto draw_at_centered(const vector3_type& centered_coordinates) const -> value_type {
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
        bool m_is_inverted{};
    };

    template<size_t N>
    requires (N == 2 or N == 3)
    struct Sphere {
        static constexpr size_t SIZE = N;

        /// (D)HW center of the sphere/circle.
        Vec<f64, N> center;

        /// Radius of the sphere/circle.
        f64 radius;

        /// Size, in pixels, of the raised cosine smooth edge.
        /// This is the number of pixels that will be used to compute the (cvalue,0] taper range.
        f64 smoothness{0};

        /// Value of the pixels inside the object.
        /// Pixels outside the object are set to 0.
        f64 cvalue{1};

        /// Whether the object should be inverted, i.e. elements inside the object are set to 0,
        /// and elements outside the object are set to cvalue.
        bool invert{false};

    public:
        /// Returns the drawing operator.
        template<nt::any_of<f32, f64> T = f64>
        constexpr auto draw() const noexcept {
            return DrawSphere<N, T, true>(
                center.template as<T>(),
                static_cast<T>(radius),
                static_cast<T>(cvalue),
                static_cast<T>(smoothness),
                invert
            );
        }

        /// Returns the drawing operator, enforcing zero smoothness.
        template<nt::any_of<f32, f64> T = f64>
        constexpr auto draw_binary() const noexcept {
            return DrawSphere<N, T, false>(
                center.template as<T>(),
                static_cast<T>(radius),
                static_cast<T>(cvalue),
                invert
            );
        }
    };

    template<size_t N>
    requires (N == 2 or N == 3)
    struct Ellipse {
        static constexpr size_t SIZE = N;

        /// (D)HW center of the ellipse.
        Vec<f64, N> center;

        /// (D)HW radius of the ellipse.
        Vec<f64, N> radius;

        /// Size, in pixels, of the raised cosine smooth edge.
        /// This is the number of pixels that will be used to compute the (cvalue,0] taper range.
        f64 smoothness{0};

        /// Value of the pixels inside the object.
        /// Pixels outside the object are set to 0.
        f64 cvalue{1};

        /// Whether the object should be inverted, i.e. elements inside the object are set to 0,
        /// and elements outside the object are set to cvalue.
        bool invert{false};

    public:
        /// Returns the drawing operator.
        template<nt::any_of<f32, f64> T = f64>
        constexpr auto draw() const noexcept {
            return DrawEllipse<N, T, true>(
                center.template as<T>(),
                radius.template as<T>(),
                static_cast<T>(cvalue),
                static_cast<T>(smoothness),
                invert
            );
        }

        /// Returns the drawing operator, enforcing zero smoothness.
        template<nt::any_of<f32, f64> T = f64>
        constexpr auto draw_binary() const noexcept {
            return DrawEllipse<N, T, false>(
                center.template as<T>(),
                radius.template as<T>(),
                static_cast<T>(cvalue),
                invert
            );
        }
    };

    template<size_t N>
    requires (N == 2 or N == 3)
    struct Rectangle {
        static constexpr size_t SIZE = N;

        /// (D)HW center of the rectangle.
        Vec<f64, N> center{};

        /// (D)HW radius of the rectangle.
        Vec<f64, N> radius;

        /// Size, in pixels, of the raised cosine smooth edge.
        /// This is the number of pixels that will be used to compute the (cvalue,0] taper range.
        f64 smoothness{0};

        /// Value of the pixels inside the object.
        /// Pixels outside the object are set to 0.
        f64 cvalue{1};

        /// Whether the object should be inverted, i.e. elements inside the object are set to 0,
        /// and elements outside the object are set to cvalue.
        bool invert{false};

    public:
        /// Returns the drawing operator.
        template<nt::any_of<f32, f64> T = f64>
        constexpr auto draw() const noexcept {
            return DrawRectangle<N, T, true>(
                center.template as<T>(),
                radius.template as<T>(),
                static_cast<T>(cvalue),
                static_cast<T>(smoothness),
                invert
            );
        }

        /// Returns the drawing operator, enforcing zero smoothness.
        template<nt::any_of<f32, f64> T = f64>
        constexpr auto draw_binary() const noexcept {
            return DrawRectangle<N, T, false>(
                center.template as<T>(),
                radius.template as<T>(),
                static_cast<T>(cvalue),
                invert
            );
        }
    };

    struct Cylinder {
        static constexpr size_t SIZE = 3;

        /// DHW center of the cylinder.
        Vec3<f64> center{};

        /// Radius of the cylinder.
        f64 radius;

        /// Length of the cylinder.
        f64 length;

        /// Size, in pixels, of the raised cosine smooth edge.
        /// This is the number of pixels that will be used to compute the (cvalue,0] taper range.
        f64 smoothness{0};

        /// Value of the pixels inside the object.
        /// Pixels outside the object are set to 0.
        f64 cvalue{1};

        /// Whether the object should be inverted, i.e. elements inside the object are set to 0,
        /// and elements outside the object are set to cvalue.
        bool invert{false};

    public:
        /// Returns the drawing operator.
        template<nt::any_of<f32, f64> T = f64>
        constexpr auto draw() const noexcept {
            return DrawCylinder<T, true>(
                center.template as<T>(),
                Vec2<T>::from_values(radius, length),
                static_cast<T>(cvalue),
                static_cast<T>(smoothness),
                invert
            );
        }

        /// Returns the drawing operator, enforcing zero smoothness.
        template<nt::any_of<f32, f64> T = f64>
        constexpr auto draw_binary() const noexcept {
            return DrawCylinder<T, false>(
                center.template as<T>(),
                Vec2<T>::from_values(radius, length),
                static_cast<T>(cvalue),
                invert
            );
        }
    };
}
