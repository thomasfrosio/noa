#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/utils/Atomic.hpp"

namespace noa::algorithm::signal {
    // Isotropic FSC implementation.
    // * A lerp is used to add frequencies in its two neighbour shells, instead of rounding to the nearest shell.
    // * The frequencies are normalized, so rectangular volumes can be passed.
    // * The number of shells is fixed by the shape: min(shape) // 2 + 1
    template<noa::fft::Remap REMAP, typename Coord, typename Index, typename Offset, typename Real>
    class IsotropicFSC {
    public:
        static_assert(REMAP == noa::fft::H2H || REMAP == noa::fft::HC2HC);
        static_assert(noa::traits::is_sint_v<Index>);
        static_assert(noa::traits::is_int_v<Offset>);
        static_assert(noa::traits::is_real_v<Coord>);
        static_assert(noa::traits::is_real_v<Real>);

        using index_type = Index;
        using offset_type = Offset;
        using coord_type = Coord;
        using real_type = Real;
        using complex_type = Complex<Real>;
        using index3_type = Vec3<index_type>;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;
        using input_accessor_type = AccessorRestrict<const complex_type, 4, offset_type>;
        using output_accessor_type = AccessorRestrictContiguous<real_type, 2, offset_type>;

    public:
        IsotropicFSC(const input_accessor_type& lhs,
                     const input_accessor_type& rhs, const shape3_type& shape,
                     const output_accessor_type& numerator_and_output,
                     const output_accessor_type& denominator_lhs,
                     const output_accessor_type& denominator_rhs)
                : m_lhs(lhs), m_rhs(rhs),
                  m_numerator_and_output(numerator_and_output),
                  m_denominator_lhs(denominator_lhs),
                  m_denominator_rhs(denominator_rhs),
                  m_shape(shape) {
            m_half_shape = static_cast<coord3_type>(m_shape.vec() / 2 * 2 + index3_type(m_shape == 1));
            m_max_shell = noa::math::min(m_shape) / 2 + 1;
        }

        // Initial reduction.
        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const {
            // Compute the normalized frequency.
            const coord3_type frequency = index2frequency_(z, y, x);

            // Shortcut for everything past Nyquist.
            const auto radius_sqd = noa::math::dot(frequency, frequency);
            if (radius_sqd > coord_type{0.25})
                return;

            const auto radius = noa::math::sqrt(radius_sqd);
            const auto fraction = noa::math::floor(radius);

            // Compute lerp weights.
            const auto shell_low = static_cast<index_type>(noa::math::floor(radius));
            const auto shell_high = noa::math::min(m_max_shell, shell_low + 1); // clamp for oob
            const auto fraction_high = static_cast<real_type>(radius - fraction);
            const auto fraction_low = 1 - fraction_high;

            // Preliminary shell values.
            const auto lhs = m_lhs(batch, z, y, x);
            const auto rhs = m_rhs(batch, z, y, x);
            const auto numerator = noa::math::dot(lhs, rhs); // sqrt(abs(lhs) * abs(rhs))
            const auto denominator_lhs = noa::math::abs_squared(lhs);
            const auto denominator_rhs = noa::math::abs_squared(rhs);

            // Atomic save.
            // TODO In CUDA, we could do the atomic reduction in shared memory to reduce global memory transfers.
            noa::details::atomic_add(m_numerator_and_output, numerator * fraction_low, batch, shell_low);
            noa::details::atomic_add(m_numerator_and_output, numerator * fraction_high, batch, shell_high);
            noa::details::atomic_add(m_denominator_lhs, denominator_lhs * fraction_low, batch, shell_low);
            noa::details::atomic_add(m_denominator_lhs, denominator_lhs * fraction_high, batch, shell_high);
            noa::details::atomic_add(m_denominator_rhs, denominator_rhs * fraction_low, batch, shell_low);
            noa::details::atomic_add(m_denominator_rhs, denominator_rhs * fraction_high, batch, shell_high);
        }

        // Post-processing.
        // TODO This could be a ewise-trinary operator.
        NOA_HD void operator()(index_type batch, index_type shell) const {
            const auto denominator = noa::math::sqrt(
                    m_denominator_lhs[batch][shell] *
                    m_denominator_rhs[batch][shell]);

            constexpr auto EPSILON = static_cast<Real>(1e-6);
            m_numerator_and_output[batch][shell] /= noa::math::max(EPSILON, denominator);
        }

    private:
        constexpr NOA_FHD coord3_type index2frequency_(index_type z, index_type y, index_type x) const noexcept {
            index3_type index{z, y, x};
            if constexpr (REMAP == noa::fft::H2H) {
                for (index_type i = 0; i < 3; ++i)
                    if (index[i] >= (m_shape[i] + 1) / 2)
                        index[i] -= m_shape[i];
            } else {
                index -= m_shape.vec() / 2;
            }
            return coord3_type(index) / m_half_shape;
        }

    private:
        input_accessor_type m_lhs;
        input_accessor_type m_rhs;
        output_accessor_type m_numerator_and_output;
        output_accessor_type m_denominator_lhs;
        output_accessor_type m_denominator_rhs;

        coord3_type m_half_shape;
        shape3_type m_shape;
        index_type m_max_shell;
    };

    // Anisotropic/Conical FSC implementation.
    // * Implementation is same as isotropic FSC expect that there's an additional step where
    //   we compute the cone mask. Multiple cones can be defined and the innermost loop go through
    //   the cones.
    // * Cones are described by their orientation (3D vector) and the cone aperture.
    //   The aperture is fixed for every batch and the angular distance from the cone is
    //   used to compute the cone mask.
    template<noa::fft::Remap REMAP, typename Coord, typename Index, typename Offset, typename Real>
    class AnisotropicFSC {
    public:
        static_assert(REMAP == noa::fft::H2H || REMAP == noa::fft::HC2HC);
        static_assert(noa::traits::is_sint_v<Index>);
        static_assert(noa::traits::is_int_v<Offset>);
        static_assert(noa::traits::is_real_v<Coord>);
        static_assert(noa::traits::is_real_v<Real>);

        using index_type = Index;
        using offset_type = Offset;
        using coord_type = Coord;
        using real_type = Real;
        using complex_type = noa::Complex<Real>;
        using index3_type = Vec3<index_type>;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;
        using input_accessor_type = AccessorRestrict<const complex_type, 4, offset_type>;
        using output_accessor_type = AccessorRestrictContiguous<real_type, 3, offset_type>;
        using direction_accessor_type = AccessorRestrictContiguous<const coord3_type, 1, offset_type>;

    public:
        AnisotropicFSC(const input_accessor_type& lhs,
                       const input_accessor_type& rhs, const shape3_type& shape,
                       const output_accessor_type& numerator_and_output,
                       const output_accessor_type& denominator_lhs,
                       const output_accessor_type& denominator_rhs,
                       const direction_accessor_type& normalized_cone_directions,
                       index_type cone_count,
                       coord_type cone_aperture)
                : m_lhs(lhs), m_rhs(rhs),
                  m_numerator_and_output(numerator_and_output),
                  m_denominator_lhs(denominator_lhs),
                  m_denominator_rhs(denominator_rhs),
                  m_normalized_cone_directions(normalized_cone_directions),
                  m_shape(shape),
                  m_cone_count(cone_count) {

            m_half_shape = static_cast<coord3_type>(m_shape.vec() / 2 * 2 + index3_type(m_shape == 1));
            m_max_shell = noa::math::min(m_shape) / 2 + 1;
            m_cos_cone_aperture = noa::math::cos(cone_aperture);
        }

        // Initial reduction.
        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const {
            // Compute the normalized frequency.
            const coord3_type frequency = index2frequency_(z, y, x);

            // Shortcut for everything past Nyquist.
            const auto radius_sqd = noa::math::dot(frequency, frequency);
            if (radius_sqd > coord_type{0.25})
                return;

            const auto radius = noa::math::sqrt(radius_sqd);
            const auto fraction = noa::math::floor(radius);

            // Compute lerp weights.
            const auto shell_low = static_cast<index_type>(noa::math::floor(radius));
            const auto shell_high = noa::math::min(m_max_shell, shell_low + 1); // clamp for oob
            const auto fraction_high = static_cast<real_type>(radius - fraction);
            const auto fraction_low = 1 - fraction_high;

            // Preliminary shell values.
            const auto lhs = m_lhs(batch, z, y, x);
            const auto rhs = m_rhs(batch, z, y, x);
            const auto numerator = noa::math::dot(lhs, rhs); // sqrt(abs(lhs) * abs(rhs))
            const auto denominator_lhs = noa::math::abs_squared(lhs);
            const auto denominator_rhs = noa::math::abs_squared(rhs);

            const auto normalized_direction = coord3_type{z, y, x} / radius;
            for (index_type cone = 0; cone < m_cone_count; ++cone) {

                // angular_difference = acos(dot(a,b))
                // We could compute the angular difference between the current frequency and the cone direction.
                // However, we only want to know if the frequency is inside or outside the cone, so skip the arccos
                // and check abs(cos(angle)) > cos(cone_aperture).

                // TODO In CUDA, try constant memory for the directions, or the less appropriate shared memory.
                const auto normalized_direction_cone = m_normalized_cone_directions[cone];
                const auto cos_angle_difference = noa::math::dot(normalized_direction, normalized_direction_cone);
                if (noa::math::abs(cos_angle_difference) > m_cos_cone_aperture)
                    continue;

                // Atomic save.
                // TODO In CUDA, we could do the atomic reduction in shared memory to reduce global memory transfers.
                noa::details::atomic_add(m_numerator_and_output, numerator * fraction_low, batch, cone, shell_low);
                noa::details::atomic_add(m_numerator_and_output, numerator * fraction_high, batch, cone, shell_high);
                noa::details::atomic_add(m_denominator_lhs, denominator_lhs * fraction_low, batch, cone, shell_low);
                noa::details::atomic_add(m_denominator_lhs, denominator_lhs * fraction_high, batch, cone, shell_high);
                noa::details::atomic_add(m_denominator_rhs, denominator_rhs * fraction_low, batch, cone, shell_low);
                noa::details::atomic_add(m_denominator_rhs, denominator_rhs * fraction_high, batch, cone, shell_high);
            }
        }

        // Post-processing.
        // TODO This could be a trinary operator.
        NOA_HD void operator()(index_type batch, index_type cone, index_type shell) const {
            const auto denominator = noa::math::sqrt(
                    m_denominator_lhs(batch, cone, shell) *
                    m_denominator_rhs(batch, cone, shell));

            constexpr auto EPSILON = static_cast<Real>(1e-6); // FIXME
            m_numerator_and_output(batch, cone, shell) /= noa::math::max(EPSILON, denominator);
        }

    private:
        constexpr NOA_FHD coord3_type index2frequency_(index_type z, index_type y, index_type x) const noexcept {
            index3_type index{z, y, x};
            if constexpr (REMAP == noa::fft::H2H) {
                for (index_type i = 0; i < 3; ++i)
                    if (index[i] >= (m_shape[i] + 1) / 2)
                        index[i] -= m_shape[i];
            } else {
                index -= m_shape.vec() / 2;
            }
            return coord3_type(index) / m_half_shape;
        }

    private:
        input_accessor_type m_lhs;
        input_accessor_type m_rhs;
        output_accessor_type m_numerator_and_output;
        output_accessor_type m_denominator_lhs;
        output_accessor_type m_denominator_rhs;
        direction_accessor_type m_normalized_cone_directions;

        coord3_type m_half_shape;
        coord_type m_cos_cone_aperture;
        shape3_type m_shape;
        index_type m_max_shell;
        index_type m_cone_count;
    };


    template<noa::fft::Remap REMAP, typename Coord = float, typename Index, typename Offset, typename Real>
    auto isotropic_fsc(const noa::Complex<Real>* lhs, const Strides4<Offset>& lhs_strides,
                       const noa::Complex<Real>* rhs, const Strides4<Offset>& rhs_strides,
                       Real* numerator,
                       Real* denominator_lhs,
                       Real* denominator_rhs,
                       Shape3<Index> shape, Index shell_count) {

        const auto shell_strides = Strides2<i64>{shell_count, 1};
        const auto lhs_accessor = AccessorRestrict<const Complex<Real>, 4, Offset>(lhs, lhs_strides);
        const auto rhs_accessor = AccessorRestrict<const Complex<Real>, 4, Offset>(rhs, rhs_strides);
        const auto numerator_accessor = AccessorRestrictContiguous<Real, 2, Offset>(numerator, shell_strides);
        const auto denominator_lhs_accessor = AccessorRestrictContiguous<Real, 2, Offset>(denominator_lhs, shell_strides);
        const auto denominator_rhs_accessor = AccessorRestrictContiguous<Real, 2, Offset>(denominator_rhs, shell_strides);

        return IsotropicFSC<REMAP, Coord, Index, Offset, Real>(
                lhs_accessor, rhs_accessor, shape,
                numerator_accessor, denominator_lhs_accessor, denominator_rhs_accessor);
    }

    template<noa::fft::Remap REMAP, typename Coord, typename Index, typename Offset, typename Real>
    auto anisotropic_fsc(const noa::Complex<Real>* lhs, const Strides4<Offset>& lhs_strides,
                         const noa::Complex<Real>* rhs, const Strides4<Offset>& rhs_strides,
                         Real* numerator,
                         Real* denominator_lhs,
                         Real* denominator_rhs,
                         const Shape3<Index>& shape, Index shell_count,
                         const Vec3<Coord>* normalized_cone_directions,
                         Index cone_count, Coord cone_aperture) {

        const auto fsc_strides = Strides3<i64>{cone_count * shell_count, shell_count, 1};
        const auto lhs_accessor = AccessorRestrict<const Complex<Real>, 4, Offset>(lhs, lhs_strides);
        const auto rhs_accessor = AccessorRestrict<const Complex<Real>, 4, Offset>(rhs, rhs_strides);
        const auto numerator_accessor = AccessorRestrictContiguous<Real, 3, Offset>(numerator, fsc_strides);
        const auto denominator_lhs_accessor = AccessorRestrictContiguous<Real, 3, Offset>(denominator_lhs, fsc_strides);
        const auto denominator_rhs_accessor = AccessorRestrictContiguous<Real, 3, Offset>(denominator_rhs, fsc_strides);
        const auto cone_directions_accessor = AccessorRestrictContiguous<const Vec3<Coord>, 1, Offset>(normalized_cone_directions);

        return AnisotropicFSC<REMAP, Coord, Index, Offset, Real>(
                lhs_accessor, rhs_accessor, shape,
                numerator_accessor, denominator_lhs_accessor, denominator_rhs_accessor,
                cone_directions_accessor, cone_count, cone_aperture);
    }
}
