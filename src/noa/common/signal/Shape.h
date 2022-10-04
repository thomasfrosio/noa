#pragma once

#include "noa/common/geometry/Polar.h"
#include "noa/common/math/Constant.h"
#include "noa/common/math/Generic.h"
#include "noa/common/types/Float2.h"
#include "noa/common/types/Float3.h"

namespace noa::signal {
    /// Line defined by a center and radius.
    /// \tparam INVERT  Whether the line should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<typename out_t, bool INVERT = false, typename in_t = float>
    class Line {
    public:
        constexpr NOA_IHD Line(in_t center, in_t radius, [[maybe_unused]] in_t = in_t{0}) noexcept
                : m_center(center), m_radius(radius) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(in_t coord) const noexcept {
            coord -= m_center;
            const auto dst = math::abs(coord);

            if (dst > m_radius)
                return static_cast<out_t>(INVERT);
            else
                return static_cast<out_t>(!INVERT);
        }

    private:
        in_t m_center;
        in_t m_radius;
    };

    /// Smooth line, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the line should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<typename out_t, bool INVERT = false, typename in_t = float>
    class LineSmooth {
    public:
        constexpr NOA_IHD LineSmooth(in_t center, in_t radius, in_t edge_size) noexcept
                : m_center(center), m_radius(radius), m_edge_size(edge_size) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(in_t coord) const noexcept {
            coord -= m_center;
            const auto dst = math::abs(coord);

            constexpr auto PI = math::Constants<in_t>::PI;
            if (dst > m_radius + m_edge_size) {
                return static_cast<out_t>(INVERT);
            } else if (dst <= m_radius) {
                return static_cast<out_t>(!INVERT);
            } else {
                if constexpr (INVERT)
                    return static_cast<out_t>((in_t{1} - math::cos(PI * (dst - m_radius) / m_edge_size)) * in_t{0.5});
                else
                    return static_cast<out_t>((in_t{1} + math::cos(PI * (dst - m_radius) / m_edge_size)) * in_t{0.5});
            }
        }

    private:
        in_t m_center;
        in_t m_radius;
        in_t m_edge_size;
    };
}

namespace noa::signal {
    /// Sphere defined by a center and radius.
    /// \tparam INVERT  Whether the sphere should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<int N, typename out_t, bool INVERT = false, typename in_t = float>
    class Sphere {
    public:
        using in_vec_t = std::conditional_t<N == 2, Float2<in_t>, Float3<in_t>>;

        constexpr NOA_IHD Sphere(in_vec_t center, in_t radius, [[maybe_unused]] in_t = in_t{0}) noexcept
                : m_center(center), m_radius_sqd(radius * radius) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(in_vec_t coords) const noexcept {
            coords -= m_center;
            const auto dst_sqd = math::dot(coords, coords);

            if (dst_sqd > m_radius_sqd)
                return static_cast<out_t>(INVERT);
            else
                return static_cast<out_t>(!INVERT);
        }

    private:
        in_vec_t m_center;
        in_t m_radius_sqd;
    };

    /// Smooth sphere, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the sphere should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<int N, typename out_t, bool INVERT = false, typename in_t = float>
    class SphereSmooth {
    public:
        using in_vec_t = std::conditional_t<N == 2, Float2<in_t>, Float3<in_t>>;

        constexpr NOA_IHD SphereSmooth(in_vec_t center, in_t radius, in_t edge_size) noexcept
                : m_center(center), m_radius(radius), m_radius_sqd(radius * radius), m_edge_size(edge_size),
                  m_radius_edge_sqd((radius + edge_size) * (radius + edge_size)) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(in_vec_t coords) const noexcept {
            coords -= m_center;
            const auto dst_sqd = math::dot(coords, coords);

            constexpr auto PI = math::Constants<in_t>::PI;
            if (dst_sqd > m_radius_edge_sqd) {
                return static_cast<out_t>(INVERT);
            } else if (dst_sqd <= m_radius_sqd) {
                return static_cast<out_t>(!INVERT);
            } else {
                const auto dst = math::sqrt(dst_sqd);
                if constexpr (INVERT)
                    return static_cast<out_t>((in_t{1} - math::cos(PI * (dst - m_radius) / m_edge_size)) * in_t{0.5});
                else
                    return static_cast<out_t>((in_t{1} + math::cos(PI * (dst - m_radius) / m_edge_size)) * in_t{0.5});
            }
        }

    private:
        in_vec_t m_center;
        in_t m_radius;
        in_t m_radius_sqd;
        in_t m_edge_size;
        in_t m_radius_edge_sqd;
    };
}

namespace noa::signal {
    /// Cylinder defined by a center and radius.
    /// \tparam INVERT  Whether the cylinder should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<typename out_t, bool INVERT = false, typename in_t = float>
    class Cylinder {
    public:
        using in_vec_t = Float3<in_t>;

        constexpr NOA_IHD Cylinder(in_vec_t center, in_t radius, in_t length, [[maybe_unused]] in_t = in_t{0}) noexcept
                : m_center(center), m_radius_sqd(radius * radius), m_length(length) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(in_vec_t coords) const noexcept {
            const auto dst_z = math::abs(coords[0] - m_center[0]);

            using vec2_t = Float2<in_t>;
            vec2_t tmp{coords[1], coords[2]};
            tmp -= vec2_t{m_center[1], m_center[2]};
            const in_t dst_yx_sqd = math::dot(tmp, tmp);

            if (dst_z > m_length || dst_yx_sqd > m_radius_sqd)
                return static_cast<out_t>(INVERT);
            else
                return static_cast<out_t>(!INVERT);
        }

    private:
        in_vec_t m_center;
        in_t m_radius_sqd;
        in_t m_length;
    };

    /// Smooth cylinder, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the cylinder should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<typename out_t, bool INVERT = false, typename in_t = float>
    class CylinderSmooth {
    public:
        using in_vec_t = Float3<in_t>;

        constexpr NOA_IHD CylinderSmooth(in_vec_t center, in_t radius, in_t length, in_t edge_size) noexcept
                : m_center(center), m_radius(radius), m_radius_sqd(radius * radius),
                  m_radius_edge_sqd((radius + edge_size) * (radius + edge_size)),
                  m_edge_size(edge_size), m_length(length), m_length_edge(length + edge_size) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(in_vec_t coords) const noexcept {
            const auto dst_z = math::abs(coords[0] - m_center[0]);

            using vec2_t = Float2<in_t>;
            vec2_t tmp{coords[1], coords[2]};
            tmp -= vec2_t{m_center[1], m_center[2]};
            const in_t dst_yx_sqd = math::dot(tmp, tmp);

            if (dst_z > m_length_edge || dst_yx_sqd > m_radius_edge_sqd) {
                return static_cast<out_t>(INVERT);
            } else {
                constexpr auto PI = math::Constants<in_t>::PI;
                in_t mask;
                if (dst_yx_sqd <= m_radius_sqd) {
                    mask = in_t{1};
                } else {
                    const auto dst_yx = math::sqrt(dst_yx_sqd);
                    mask = (in_t{1} + math::cos(PI * (dst_yx - m_radius) / m_edge_size)) * in_t{0.5};
                }

                if (dst_z > m_length)
                    mask *= (in_t{1} + math::cos(PI * (dst_z - m_length) / m_edge_size)) * in_t{0.5};

                if constexpr (INVERT)
                    return static_cast<out_t>(in_t{1} - mask);
                else
                    return static_cast<out_t>(mask);
            }
        }

    private:
        in_vec_t m_center;
        in_t m_radius;
        in_t m_radius_sqd;
        in_t m_radius_edge_sqd;

        in_t m_edge_size;
        in_t m_length;
        in_t m_length_edge;

    };
}

namespace noa::signal {
    /// Rectangle defined by a center and radius.
    /// \tparam INVERT  Whether the rectangle should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<int N, typename out_t, bool INVERT = false, typename in_t = float>
    class Rectangle {
    public:
        using in_vec_t = std::conditional_t<N == 2, Float2<in_t>, Float3<in_t>>;

        constexpr NOA_IHD Rectangle(in_vec_t center, in_vec_t radius, [[maybe_unused]] in_t = in_t{0}) noexcept
                : m_center(center), m_radius(radius) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(in_vec_t coords) const noexcept {
            coords -= m_center;
            coords = math::abs(coords);

            if (all(coords <= m_radius))
                return static_cast<out_t>(!INVERT);
            else
                return static_cast<out_t>(INVERT);
        }

    private:
        in_vec_t m_center;
        in_vec_t m_radius;
    };

    /// Smooth rectangle, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the rectangle should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<int N, typename out_t, bool INVERT = false, typename in_t = float>
    class RectangleSmooth {
    public:
        using in_vec_t = std::conditional_t<N == 2, Float2<in_t>, Float3<in_t>>;

        constexpr NOA_IHD RectangleSmooth(in_vec_t center, in_vec_t radius, in_t edge_size) noexcept
                : m_center(center), m_radius(radius), m_radius_edge(radius + edge_size), m_edge_size(edge_size) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(in_vec_t coords) const noexcept {
            coords -= m_center;
            coords = math::abs(coords);

            constexpr auto PI = math::Constants<in_t>::PI;
            if (any(m_radius_edge < coords)) {
                return static_cast<out_t>(INVERT);
            } else if (all(coords <= m_radius)) {
                return static_cast<out_t>(!INVERT);
            } else {
                in_t mask_value{1};
                if (m_radius[0] < coords[0] && coords[0] <= m_radius_edge[0])
                    mask_value *= (in_t{1} + math::cos(PI * (coords[0] - m_radius[0]) / m_edge_size)) * in_t{0.5};
                if (m_radius[1] < coords[1] && coords[1] <= m_radius_edge[1])
                    mask_value *= (in_t{1} + math::cos(PI * (coords[1] - m_radius[1]) / m_edge_size)) * in_t{0.5};
                if (m_radius[2] < coords[2] && coords[2] <= m_radius_edge[2])
                    mask_value *= (in_t{1} + math::cos(PI * (coords[2] - m_radius[2]) / m_edge_size)) * in_t{0.5};
                if constexpr (INVERT)
                    return static_cast<out_t>(in_t{1} - mask_value);
                else
                    return static_cast<out_t>(mask_value);
            }
        }

    private:
        in_vec_t m_center;
        in_vec_t m_radius;
        in_vec_t m_radius_edge;
        in_t m_edge_size;
    };
}

namespace noa::signal {
    /// Ellipse defined by a center and radius.
    /// \tparam INVERT  Whether the ellipse should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<int N, typename out_t, bool INVERT = false, typename in_t = float>
    class Ellipse {
    public:
        using in_vec_t = std::conditional_t<N == 2, Float2<in_t>, Float3<in_t>>;

        constexpr NOA_IHD Ellipse(in_vec_t center, in_vec_t radius, [[maybe_unused]] in_t = in_t{0}) noexcept
                : m_center(center), m_radius(radius) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(in_vec_t coords) const noexcept {
            coords -= m_center;
            coords /= m_radius;
            const in_t rho = math::dot(coords, coords);

            if constexpr (INVERT)
                return static_cast<out_t>(rho > 1);
            else
                return static_cast<out_t>(rho <= 1);
        }

    private:
        in_vec_t m_center;
        in_vec_t m_radius;
    };

    /// Smooth ellipse, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the ellipse should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<int N, typename out_t, bool INVERT = false, typename in_t = float>
    class EllipseSmooth {
    public:
        using in_vec_t = std::conditional_t<N == 2, Float2<in_t>, Float3<in_t>>;

        constexpr NOA_IHD EllipseSmooth(in_vec_t center, in_vec_t radius, in_t edge_size) noexcept
                : m_center(center), m_radius(radius * radius), m_edge_size(edge_size) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(in_vec_t coords) const noexcept {
            coords -= m_center;

            in_t irho, erho;
            if constexpr (N == 2) {
                // Current spherical coordinate:
                irho = geometry::cartesian2rho(coords);
                const in_t iphi = geometry::cartesian2phi<false>(coords);

                // Radius of the ellipse at (iphi, itheta):
                const in_t cos2phi = math::pow(math::cos(iphi), in_t{2});
                const in_t sin2phi = math::pow(math::sin(iphi), in_t{2});
                erho = in_t{1} / math::sqrt(cos2phi / m_radius[1] + sin2phi / m_radius[0]);

            } else {
                irho = geometry::cartesian2rho(coords);
                const in_t iphi = geometry::cartesian2phi<false>(coords);
                const in_t itheta = geometry::cartesian2theta(coords);

                const in_t cos2phi = math::pow(math::cos(iphi), in_t{2});
                const in_t sin2phi = math::pow(math::sin(iphi), in_t{2});
                const in_t cos2theta = math::pow(math::cos(itheta), in_t{2});
                const in_t sin2theta = math::pow(math::sin(itheta), in_t{2});
                erho = in_t{1} / math::sqrt(cos2phi * sin2theta / m_radius[2] +
                                            sin2phi * sin2theta / m_radius[1] +
                                            cos2theta / m_radius[0]);
            }

            constexpr in_t PI = math::Constants<in_t>::PI;
            if (irho > erho + m_edge_size) {
                return static_cast<out_t>(!INVERT);
            } else if (irho <= erho) {
                return static_cast<out_t>(INVERT);
            } else {
                if constexpr (INVERT)
                    return static_cast<out_t>((in_t{1} - math::cos(PI * (irho - erho) / m_edge_size)) * in_t{0.5});
                else
                    return static_cast<out_t>((in_t{1} + math::cos(PI * (irho - erho) / m_edge_size)) * in_t{0.5});
            }
        }

    private:
        in_vec_t m_center;
        in_vec_t m_radius;
        in_t m_edge_size{};
    };
}
