#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/utils/Atomic.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::signal {
    /// 4d iwise operator to compute an isotropic FSC.
    /// * A lerp is used to add frequencies in its two neighbour shells, instead of rounding to the nearest shell.
    /// * The frequencies are normalized, so rectangular volumes can be passed.
    /// * The number of shells is fixed by the input shape: min(shape) // 2 + 1
    template<Remap REMAP,
             nt::real Coord,
             nt::sinteger Index,
             nt::readable_nd<4> Input,
             nt::atomic_addable_nd<2> Output>
    class FSCIsotropic {
    public:
        static_assert(not REMAP.has_layout_change());
        static constexpr bool IS_CENTERED = REMAP.is_xx2xc();
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();

        using index_type = Index;
        using coord_type = Coord;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;
        using shape_nd_type = Shape<index_type, 3 - IS_RFFT>;

        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::complex<input_value_type> and nt::real<output_value_type>);

    public:
        FSCIsotropic(
                const input_type& lhs,
                const input_type& rhs,
                const shape3_type& shape,
                const output_type& numerator_and_output,
                const output_type& denominator_lhs,
                const output_type& denominator_rhs
        ) : m_lhs(lhs), m_rhs(rhs),
            m_numerator_and_output(numerator_and_output),
            m_denominator_lhs(denominator_lhs),
            m_denominator_rhs(denominator_rhs),
            m_norm(coord_type{1} / coord3_type::from_vec(shape.vec)),
            m_scale(static_cast<coord_type>(min(shape))),
            m_max_shell_index(min(shape) / 2)
        {
            if constexpr (IS_RFFT)
                m_shape = shape.pop_back();
            else
                m_shape = shape;
        }

        // Initial reduction.
        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const {
            // Compute the normalized frequency.
            auto frequency = noa::fft::index2frequency<IS_CENTERED, IS_RFFT>(Vec{z, y, x}, m_shape);
            auto fftfreq = coord3_type::from_vec(frequency) * m_norm;

            // Shortcut for everything past Nyquist.
            const auto radius_sqd = dot(fftfreq, fftfreq);
            if (radius_sqd > coord_type{0.25})
                return;

            const auto radius = sqrt(radius_sqd) * m_scale;
            const auto fraction = floor(radius);

            // Compute lerp weights.
            const auto shell_low = static_cast<index_type>(floor(radius));
            const auto shell_high = min(m_max_shell_index, shell_low + 1); // clamp for oob
            const auto fraction_high = static_cast<input_real_type>(radius - fraction);
            const auto fraction_low = 1 - fraction_high;

            // Preliminary shell values.
            const auto lhs = m_lhs(batch, z, y, x);
            const auto rhs = m_rhs(batch, z, y, x);
            const auto numerator = dot(lhs, rhs); // sqrt(abs(lhs) * abs(rhs))
            const auto denominator_lhs = abs_squared(lhs);
            const auto denominator_rhs = abs_squared(rhs);

            // Atomic save.
            // TODO In CUDA, we could do the atomic reduction in shared memory to reduce global memory transfers.
            using oreal_t = output_value_type;
            ng::atomic_add(m_numerator_and_output, static_cast<oreal_t>(numerator * fraction_low), batch, shell_low);
            ng::atomic_add(m_numerator_and_output, static_cast<oreal_t>(numerator * fraction_high), batch, shell_high);
            ng::atomic_add(m_denominator_lhs, static_cast<oreal_t>(denominator_lhs * fraction_low), batch, shell_low);
            ng::atomic_add(m_denominator_lhs, static_cast<oreal_t>(denominator_lhs * fraction_high), batch, shell_high);
            ng::atomic_add(m_denominator_rhs, static_cast<oreal_t>(denominator_rhs * fraction_low), batch, shell_low);
            ng::atomic_add(m_denominator_rhs, static_cast<oreal_t>(denominator_rhs * fraction_high), batch, shell_high);
        }

    private:
        input_type m_lhs;
        input_type m_rhs;
        output_type m_numerator_and_output;
        output_type m_denominator_lhs;
        output_type m_denominator_rhs;

        coord3_type m_norm;
        coord_type m_scale;
        shape_nd_type m_shape;
        index_type m_max_shell_index;
    };

    /// Anisotropic/Conical FSC implementation.
    /// * Implementation is same as isotropic FSC expect that there's an additional step where
    ///   we compute the cone mask. Multiple cones can be defined and the innermost loop go through
    ///   the cones.
    /// * Cones are described by their orientation (a 3d vector) and the cone aperture.
    ///   The aperture is fixed for every batch and the angular distance from the cone is
    ///   used to compute the cone mask.
    template<Remap REMAP,
             nt::real Coord,
             nt::sinteger Index,
             nt::readable_nd<4> Input,
             nt::atomic_addable_nd<3> Output,
             nt::readable_pointer_like Direction>
    class FSCAnisotropic {
    public:
        static_assert(not REMAP.has_layout_change());
        static constexpr bool IS_CENTERED = REMAP.is_xx2xc();
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();

        using index_type = Index;
        using coord_type = Coord;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;
        using shape_nd_type = Shape<index_type, 3 - IS_RFFT>;

        using input_type = Input;
        using output_type = Output;
        using direction_type = Direction;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using direction_value_type = nt::value_type_t<direction_type>;
        static_assert(nt::complex<input_value_type> and
                      nt::real<output_value_type> and
                      nt::vec_real_size<direction_value_type, 3>);

    public:
        FSCAnisotropic(
                const input_type& lhs,
                const input_type& rhs, const shape3_type& shape,
                const output_type& numerator_and_output,
                const output_type& denominator_lhs,
                const output_type& denominator_rhs,
                const direction_type& normalized_cone_directions,
                index_type cone_count,
                coord_type cone_aperture
        ) : m_lhs(lhs), m_rhs(rhs),
            m_numerator_and_output(numerator_and_output),
            m_denominator_lhs(denominator_lhs),
            m_denominator_rhs(denominator_rhs),
            m_normalized_cone_directions(normalized_cone_directions),
            m_norm(coord_type{1} / coord3_type::from_vec(shape.vec)),
            m_scale(static_cast<coord_type>(min(shape))),
            m_cos_cone_aperture(cos(cone_aperture)),
            m_max_shell_index(min(shape) / 2),
            m_cone_count(cone_count)
        {
            if constexpr (IS_RFFT)
                m_shape = shape.pop_back();
            else
                m_shape = shape;
        }

        // Initial reduction.
        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const {
            // Compute the normalized frequency.
            auto frequency = noa::fft::index2frequency<IS_CENTERED, IS_RFFT>(Vec{z, y, x}, m_shape);
            auto fftfreq = coord3_type::from_vec(frequency) * m_norm;

            // Shortcut for everything past Nyquist.
            const auto radius_sqd = dot(fftfreq, fftfreq);
            if (radius_sqd > coord_type{0.25})
                return;

            const auto radius = sqrt(radius_sqd) * m_scale;
            const auto fraction = floor(radius);

            // Compute lerp weights.
            const auto shell_low = static_cast<index_type>(floor(radius));
            const auto shell_high = min(m_max_shell_index, shell_low + 1); // clamp for oob
            const auto fraction_high = static_cast<input_real_type>(radius - fraction);
            const auto fraction_low = 1 - fraction_high;

            // Preliminary shell values.
            const auto lhs = m_lhs(batch, z, y, x);
            const auto rhs = m_rhs(batch, z, y, x);
            const auto numerator = dot(lhs, rhs); // sqrt(abs(lhs) * abs(rhs))
            const auto denominator_lhs = abs_squared(lhs);
            const auto denominator_rhs = abs_squared(rhs);

            const auto normalized_direction = coord3_type::from_values(z, y, x) / radius;
            for (index_type cone{}; cone < m_cone_count; ++cone) {
                // angular_difference = acos(dot(a,b))
                // We could compute the angular difference between the current frequency and the cone direction.
                // However, we only want to know if the frequency is inside or outside the cone, so skip the arccos
                // and check abs(cos(angle)) > cos(cone_aperture).

                // TODO In CUDA, try constant memory for the directions, or the less appropriate shared memory.
                const auto normalized_direction_cone = m_normalized_cone_directions[cone].template as<coord_type>();
                const auto cos_angle_difference = dot(normalized_direction, normalized_direction_cone);
                if (abs(cos_angle_difference) > m_cos_cone_aperture)
                    continue;

                // Atomic save.
                // TODO In CUDA, we could do the atomic reduction in shared memory to reduce global memory transfers.
                using oreal_t = output_value_type;
                ng::atomic_add(m_numerator_and_output, static_cast<oreal_t>(numerator * fraction_low), batch, cone, shell_low);
                ng::atomic_add(m_numerator_and_output, static_cast<oreal_t>(numerator * fraction_high), batch, cone, shell_high);
                ng::atomic_add(m_denominator_lhs, static_cast<oreal_t>(denominator_lhs * fraction_low), batch, cone, shell_low);
                ng::atomic_add(m_denominator_lhs, static_cast<oreal_t>(denominator_lhs * fraction_high), batch, cone, shell_high);
                ng::atomic_add(m_denominator_rhs, static_cast<oreal_t>(denominator_rhs * fraction_low), batch, cone, shell_low);
                ng::atomic_add(m_denominator_rhs, static_cast<oreal_t>(denominator_rhs * fraction_high), batch, cone, shell_high);
            }
        }

    private:
        input_type m_lhs;
        input_type m_rhs;
        output_type m_numerator_and_output;
        output_type m_denominator_lhs;
        output_type m_denominator_rhs;
        direction_type m_normalized_cone_directions;

        coord3_type m_norm;
        coord_type m_scale;
        coord_type m_cos_cone_aperture;
        shape_nd_type m_shape;
        index_type m_max_shell_index;
        index_type m_cone_count;
    };

    struct FSCNormalization {
        using allow_vectorization = bool;

        template<typename T>
        NOA_HD void operator()(T lhs, T rhs, T& output) const {
            constexpr auto EPSILON = static_cast<T>(1e-6);
            output /= max(EPSILON, sqrt(lhs * rhs));
        }
    };
}
