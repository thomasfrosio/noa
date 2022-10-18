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
        using value_type = out_t;
        using compute_type = in_t;

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
        using vec_t = std::conditional_t<N == 2, Float2<in_t>, Float3<in_t>>;
        using rot_t = std::conditional_t<N == 2, Mat22<in_t>, Mat33<in_t>>;
        using value_type = out_t;
        using compute_type = in_t;

        constexpr NOA_IHD Sphere(vec_t center, in_t radius, [[maybe_unused]] in_t = in_t{0}) noexcept
                : m_center(center), m_radius_sqd(radius * radius) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec_t coords) const noexcept {
            return get(coords);
        }

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec_t coords, rot_t inv_transform) const noexcept {
            return get<true>(coords, inv_transform);
        }

    private:
        template<bool TRANSFORM = false>
        [[nodiscard]] constexpr NOA_IHD out_t
        get(vec_t coords, [[maybe_unused]] rot_t inv_transform = {}) const noexcept {
            coords -= m_center;
            if constexpr (TRANSFORM)
                coords *= inv_transform * coords;
            const auto dst_sqd = math::dot(coords, coords);

            if (dst_sqd > m_radius_sqd)
                return static_cast<out_t>(INVERT);
            else
                return static_cast<out_t>(!INVERT);
        }

    private:
        vec_t m_center;
        in_t m_radius_sqd;
    };

    /// Smooth sphere, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the sphere should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<int N, typename out_t, bool INVERT = false, typename in_t = float>
    class SphereSmooth {
    public:
        using vec_t = std::conditional_t<N == 2, Float2<in_t>, Float3<in_t>>;
        using rot_t = std::conditional_t<N == 2, Mat22<in_t>, Mat33<in_t>>;
        using value_type = out_t;
        using compute_type = in_t;

        constexpr NOA_IHD SphereSmooth(vec_t center, in_t radius, in_t edge_size) noexcept
                : m_center(center), m_radius(radius), m_radius_sqd(radius * radius), m_edge_size(edge_size),
                  m_radius_edge_sqd((radius + edge_size) * (radius + edge_size)) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec_t coords) const noexcept {
            return get(coords);
        }

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec_t coords, rot_t inv_transform) const noexcept {
            return get<true>(coords, inv_transform);
        }

    private:
        template<bool TRANSFORM = false>
        [[nodiscard]] constexpr NOA_IHD out_t
        get(vec_t coords, [[maybe_unused]] rot_t inv_transform = {}) const noexcept {
            coords -= m_center;
            if constexpr (TRANSFORM)
                coords *= inv_transform * coords;

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
        vec_t m_center;
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
        using vec3_t = Float3<in_t>;
        using vec2_t = Float2<in_t>;
        using mat33_t = Mat33<in_t>;
        using value_type = out_t;
        using compute_type = in_t;

        constexpr NOA_IHD Cylinder(vec3_t center, in_t radius, in_t length, [[maybe_unused]] in_t = in_t{0}) noexcept
                : m_center(center), m_radius_sqd(radius * radius), m_length(length) {}

        constexpr NOA_IHD Cylinder(vec3_t center, vec2_t length_radius, [[maybe_unused]] in_t = in_t{0}) noexcept
                : Cylinder(center, length_radius[1], length_radius[0]) {}

        constexpr NOA_IHD Cylinder(vec3_t center, vec3_t length_radius, [[maybe_unused]] in_t = in_t{0}) noexcept
                : Cylinder(center, length_radius[1], length_radius[0]) {
            NOA_ASSERT(length_radius[1] == length_radius[2]);
        }

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec3_t coords) const noexcept {
            return get(coords);
        }

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec3_t coords, mat33_t inv_transform) const noexcept {
            return get<true>(coords, inv_transform);
        }

    private:
        template<bool TRANSFORM = false>
        [[nodiscard]] constexpr NOA_IHD out_t
        get(vec3_t coords, [[maybe_unused]] mat33_t inv_transform = {}) const noexcept {
            coords -= m_center;
            if constexpr (TRANSFORM)
                coords *= inv_transform * coords;

            const auto dst_z = math::abs(coords[0]);

            vec2_t tmp{coords[1], coords[2]};
            const in_t dst_yx_sqd = math::dot(tmp, tmp);

            if (dst_z > m_length || dst_yx_sqd > m_radius_sqd)
                return static_cast<out_t>(INVERT);
            else
                return static_cast<out_t>(!INVERT);
        }

    private:
        vec3_t m_center;
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
        using vec3_t = Float3<in_t>;
        using vec2_t = Float2<in_t>;
        using mat33_t = Mat33<in_t>;
        using compute_type = in_t;
        using value_type = out_t;

        constexpr NOA_IHD CylinderSmooth(vec3_t center, in_t radius, in_t length, in_t edge_size) noexcept
                : m_center(center), m_radius(radius), m_radius_sqd(radius * radius),
                  m_radius_edge_sqd((radius + edge_size) * (radius + edge_size)),
                  m_edge_size(edge_size), m_length(length), m_length_edge(length + edge_size) {}

        constexpr NOA_IHD CylinderSmooth(vec3_t center, vec2_t length_radius, in_t edge_size) noexcept
                : CylinderSmooth(center, length_radius[1], length_radius[0], edge_size) {}

        constexpr NOA_IHD CylinderSmooth(vec3_t center, vec3_t length_radius, in_t edge_size) noexcept
                : CylinderSmooth(center, length_radius[1], length_radius[0], edge_size) {
            NOA_ASSERT(length_radius[1] == length_radius[2]);
        }

        [[nodiscard]] constexpr NOA_IHD value_type operator()(vec3_t coords) const noexcept {
            return get(coords);
        }

        [[nodiscard]] constexpr NOA_IHD value_type operator()(vec3_t coords, mat33_t inv_transform) const noexcept {
            return get<true>(coords, inv_transform);
        }

    private:
        template<bool TRANSFORM = false>
        [[nodiscard]] constexpr NOA_IHD value_type
        get(vec3_t coords, [[maybe_unused]] mat33_t inv_transform = {}) const noexcept {
            coords -= m_center;
            if constexpr (TRANSFORM)
                coords *= inv_transform * coords;

            const auto dst_z = math::abs(coords[0]);

            vec2_t tmp{coords[1], coords[2]};
            const in_t dst_yx_sqd = math::dot(tmp, tmp);

            if (dst_z > m_length_edge || dst_yx_sqd > m_radius_edge_sqd) {
                return static_cast<value_type>(INVERT);
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
        vec3_t m_center;
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
        using vec_t = std::conditional_t<N == 2, Float2<in_t>, Float3<in_t>>;
        using rot_t = std::conditional_t<N == 2, Mat22<in_t>, Mat33<in_t>>;
        using value_type = out_t;
        using compute_type = in_t;

        constexpr NOA_IHD Rectangle(vec_t center, vec_t radius, [[maybe_unused]] in_t = in_t{0}) noexcept
                : m_center(center), m_radius(radius) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec_t coords) const noexcept {
            return get(coords);
        }

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec_t coords, rot_t inv_transform) const noexcept {
            return get<true>(coords, inv_transform);
        }

    private:
        template<bool TRANSFORM = false>
        [[nodiscard]] constexpr NOA_IHD out_t
        get(vec_t coords, [[maybe_unused]] rot_t inv_transform = {}) const noexcept {
            coords -= m_center;
            if constexpr (TRANSFORM)
                coords *= inv_transform * coords;
            coords = math::abs(coords);

            if (all(coords <= m_radius))
                return static_cast<out_t>(!INVERT);
            else
                return static_cast<out_t>(INVERT);
        }

    private:
        vec_t m_center;
        vec_t m_radius;
    };

    /// Smooth rectangle, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the rectangle should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<int N, typename out_t, bool INVERT = false, typename in_t = float>
    class RectangleSmooth {
    public:
        using vec_t = std::conditional_t<N == 2, Float2<in_t>, Float3<in_t>>;
        using rot_t = std::conditional_t<N == 2, Mat22<in_t>, Mat33<in_t>>;
        using value_type = out_t;
        using compute_type = in_t;

        constexpr NOA_IHD RectangleSmooth(vec_t center, vec_t radius, in_t edge_size) noexcept
                : m_center(center), m_radius(radius), m_radius_edge(radius + edge_size), m_edge_size(edge_size) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec_t coords) const noexcept {
            return get(coords);
        }

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec_t coords, rot_t inv_transform) const noexcept {
            return get<true>(coords, inv_transform);
        }

    private:
        template<bool TRANSFORM = false>
        [[nodiscard]] constexpr NOA_IHD out_t
        get(vec_t coords, [[maybe_unused]] rot_t inv_transform = {}) const noexcept {
            coords -= m_center;
            if constexpr (TRANSFORM)
                coords = inv_transform * coords;
            coords = math::abs(coords);

            constexpr auto PI = math::Constants<in_t>::PI;
            if (any(m_radius_edge < coords)) {
                return static_cast<out_t>(INVERT);
            } else if (all(coords <= m_radius)) {
                return static_cast<out_t>(!INVERT);
            } else {
                in_t mask_value{1};
                for (dim_t i = 0; i < N; ++i) {
                    if (m_radius[i] < coords[i] && coords[i] <= m_radius_edge[i])
                        mask_value *= (in_t{1} + math::cos(PI * (coords[i] - m_radius[i]) / m_edge_size)) * in_t{0.5};
                }
                if constexpr (INVERT)
                    return static_cast<out_t>(in_t{1} - mask_value);
                else
                    return static_cast<out_t>(mask_value);
            }
        }

    private:
        vec_t m_center;
        vec_t m_radius;
        vec_t m_radius_edge;
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
        using vec_t = std::conditional_t<N == 2, Float2<in_t>, Float3<in_t>>;
        using rot_t = std::conditional_t<N == 2, Mat22<in_t>, Mat33<in_t>>;
        using value_type = out_t;
        using compute_type = in_t;

        constexpr NOA_IHD Ellipse(vec_t center, vec_t radius, [[maybe_unused]] in_t = in_t{0}) noexcept
                : m_center(center), m_radius(radius) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec_t coords) const noexcept {
            return get(coords);
        }

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec_t coords, rot_t inv_transform) const noexcept {
            return get<true>(coords, inv_transform);
        }

    private:
        template<bool TRANSFORM = false>
        [[nodiscard]] constexpr NOA_IHD out_t
        get(vec_t coords, [[maybe_unused]] rot_t inv_transform = {}) const noexcept {
            coords -= m_center;
            if constexpr (TRANSFORM)
                coords = inv_transform * coords;
            coords /= m_radius;
            const in_t rho = math::dot(coords, coords);

            if constexpr (INVERT)
                return static_cast<out_t>(rho > 1);
            else
                return static_cast<out_t>(rho <= 1);
        }

    private:
        vec_t m_center;
        vec_t m_radius;
    };

    /// Smooth ellipse, with a cosine-edge, defined by a center and radius.
    /// \tparam INVERT  Whether the ellipse should be filled with zeros instead of ones.
    /// \tparam in_t    Input, storage and computation type.
    /// \tparam out_t   Output type.
    template<int N, typename out_t, bool INVERT = false, typename in_t = float>
    class EllipseSmooth {
    public:
        using vec_t = std::conditional_t<N == 2, Float2<in_t>, Float3<in_t>>;
        using rot_t = std::conditional_t<N == 2, Mat22<in_t>, Mat33<in_t>>;
        using value_type = out_t;
        using compute_type = in_t;

        constexpr NOA_IHD EllipseSmooth(vec_t center, vec_t radius, in_t edge_size) noexcept
                : m_center(center), m_radius(radius * radius), m_edge_size(edge_size) {}

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec_t coords) const noexcept {
            return get(coords);
        }

        [[nodiscard]] constexpr NOA_IHD out_t operator()(vec_t coords, rot_t inv_transform) const noexcept {
            return get<true>(coords, inv_transform);
        }

    private:
        template<bool TRANSFORM = false>
        [[nodiscard]] constexpr NOA_IHD out_t
        get(vec_t coords, [[maybe_unused]] rot_t inv_transform = {}) const noexcept {
            coords -= m_center;
            if constexpr (TRANSFORM)
                coords = inv_transform * coords; // for 2D we could accept the rotation angle and add it to phi.

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
                return static_cast<out_t>(INVERT);
            } else if (irho <= erho) {
                return static_cast<out_t>(!INVERT);
            } else {
                if constexpr (INVERT)
                    return static_cast<out_t>((in_t{1} - math::cos(PI * (irho - erho) / m_edge_size)) * in_t{0.5});
                else
                    return static_cast<out_t>((in_t{1} + math::cos(PI * (irho - erho) / m_edge_size)) * in_t{0.5});
            }
        }

    private:
        vec_t m_center;
        vec_t m_radius;
        in_t m_edge_size{};
    };
}

// For signal/fft/shape:
namespace noa::signal::fft::details {
    // To compute the shape, we need centered coordinates, so FFTshift if the input isn't centered.
    template<noa::fft::Remap REMAP, typename T>
    NOA_FHD Int3<T> gid2CenteredIndexes(const Int3<T>& gid, const Int4<T>& shape) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & noa::fft::Layout::SRC_CENTERED;
        return {IS_SRC_CENTERED ? gid[0] : math::FFTShift(gid[0], shape[1]),
                IS_SRC_CENTERED ? gid[1] : math::FFTShift(gid[1], shape[2]),
                IS_SRC_CENTERED ? gid[2] : math::FFTShift(gid[2], shape[3])};
    }

    template<noa::fft::Remap REMAP, typename T>
    NOA_FHD Int2<T> gid2CenteredIndexes(const Int2<T>& gid, const Int3<T>& shape) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & noa::fft::Layout::SRC_CENTERED;
        return {IS_SRC_CENTERED ? gid[0] : math::FFTShift(gid[0], shape[1]),
                IS_SRC_CENTERED ? gid[1] : math::FFTShift(gid[1], shape[2])};
    }

    // For the output, we need to compare with the input. If there's no remap, then the indexes
    // match and we can use the gid. Otherwise, FFTshift for F2FC, or iFFTshift for FC2F.
    template<noa::fft::Remap REMAP, typename T>
    NOA_FHD Int3<T> gid2OutputIndexes(const Int3<T>& gid, const Int4<T>& shape) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & noa::fft::Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & noa::fft::Layout::DST_CENTERED;
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED) {
            return gid;
        } else if constexpr (IS_SRC_CENTERED && !IS_DST_CENTERED) { // FC2F
            return {math::iFFTShift(gid[0], shape[1]),
                    math::iFFTShift(gid[1], shape[2]),
                    math::iFFTShift(gid[2], shape[3])};
        } else { // F2FC
            return {math::FFTShift(gid[0], shape[1]),
                    math::FFTShift(gid[1], shape[2]),
                    math::FFTShift(gid[2], shape[3])};
        }
    }

    template<noa::fft::Remap REMAP, typename T>
    NOA_FHD Int2<T> gid2OutputIndexes(const Int2<T>& gid, const Int3<T>& shape) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & noa::fft::Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & noa::fft::Layout::DST_CENTERED;
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED) {
            return gid;
        } else if constexpr (IS_SRC_CENTERED && !IS_DST_CENTERED) { // FC2F
            return {math::iFFTShift(gid[0], shape[1]),
                    math::iFFTShift(gid[1], shape[2])};
        } else { // F2FC
            return {math::FFTShift(gid[0], shape[1]),
                    math::FFTShift(gid[1], shape[2])};
        }
    }
}
