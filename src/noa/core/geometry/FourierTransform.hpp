#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/geometry/Quaternion.hpp"
#include "noa/core/signal/PhaseShift.hpp"

// FIXME    For even sizes, there's an asymmetry due to the fact that there's only one Nyquist. After fftshift,
//          this extra frequency is on the left side of the axis. However, since we work with non-redundant
//          transforms, there's a slight error that can be introduced. For the elements close (+-1 element) to x=0
//          and close (+-1 element) to y=z=0.5, if during the rotation x becomes negative and we have to flip it, the
//          interpolator will incorrectly weight towards 0 the output value. This is simply due to the fact that on
//          the right side of the axis, there's no Nyquist (the axis stops and n+1 element is OOB, i.e. =0).
//          Anyway, the fix should be done in the interpolator, not here. Maybe create a InterpolatorRFFT?

// FIXME Add proper support for FFT layouts, including non-centered inputs and more interpolation methods!

namespace noa::geometry {
    using Remap = noa::fft::Remap;

    /// 3d or 4d iwise operator to compute linear transformations (one rotation/scaling followed by one 2d translation)
    /// of 2d or 3d array(s) by directly manipulating their 2d or 3d non-redundant centered FFT(s).
    /// \details While the Transform operator supports complex arrays, it does not support FFT layouts.
    ///          This class adds support for some of these.
    /// \tparam N               2d or 3d.
    /// \tparam REMAP           HC2H or HC2HC.
    /// \tparam Index           Signed integer.
    /// \tparam Rotate          N == 2: Mat22<coord_type> or a pointer of that type.
    ///                         N == 3: Mat33<coord_type>, Quaternion<coord_type>
    ///                                 or a pointer of either one of these types.
    /// \tparam PostShift       Empty, or Vec<coord_type, N> or a pointer of that type.
    /// \tparam Interpolator    N-d interpolator.
    /// \tparam OutputAccessor  (N + 1)-d accessor.
    template<size_t N, Remap REMAP, typename Index,
             typename Rotate, typename PostShift,
             typename Interpolator, typename OutputAccessor>
    requires (nt::is_accessor_pure_nd<OutputAccessor, N + 1>::value and
              nt::is_interpolator_nd<Interpolator, N>::value and
              (REMAP == Remap::HC2H or REMAP == Remap::HC2HC))
    class FourierTransform {
    public:
        static constexpr bool is_output_centered = static_cast<uint8_t>(REMAP) & noa::fft::Layout::DST_CENTERED;

        using index_type = Index;
        using shape_type = Shape<index_type, N - 1>;
        using interpolator_type = Interpolator;
        using output_accessor_type = OutputAccessor;
        using input_value_type = interpolator_type::mutable_value_type;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = output_accessor_type::value_type;
        static_assert(nt::are_complex_v<input_value_type, output_value_type> or
                      nt::are_real_v<input_value_type, output_value_type>);

        // Rotation:
        using rotation_single_type = std::remove_const_t<std::remove_pointer_t<Rotate>>;
        using coord_type = nt::value_type_t<rotation_single_type>;
        static_assert((N == 2 and std::is_same_v<rotation_single_type, Mat22<coord_type>>) or
                      (N == 3 and nt::is_any_v<rotation_single_type, Mat33<coord_type>, Quaternion<coord_type>>));
        static constexpr bool has_multiple_rotations = std::is_pointer_v<Rotate>;
        using rotation_type = std::conditional_t<has_multiple_rotations, const rotation_single_type*, rotation_single_type>;
        using vec_type = Vec<coord_type, N>;
        using center_type = Vec<coord_type, N - 1>;

        // PostShift:
        using postshift_single_type = std::remove_const_t<std::remove_pointer_t<PostShift>>;
        static constexpr bool has_postshifts = not std::is_empty_v<postshift_single_type>;
        static constexpr bool has_multiple_postshifts = std::is_pointer_v<PostShift>;
        static_assert(std::is_same_v<vec_type, postshift_single_type>);
        using postshift_type = std::conditional_t<has_postshifts, std::conditional_t<has_multiple_postshifts, const vec_type*, vec_type>, Empty>;

    public:
        FourierTransform(
                const interpolator_type& input,
                const output_accessor_type& output,
                const Shape4<index_type>& shape,
                const rotation_type& inverse_rotation,
                const postshift_type& post_forward_shift,
                coord_type cutoff
        ) : m_input(input),
            m_output(output),
            m_inverse_rotation(inverse_rotation),
            m_post_forward_shift(post_forward_shift)
        {
            const auto shape_nd = Shape<index_type, N>(shape.data() + 4 - N);
            m_f_shape = vec_type::from_vec(shape_nd.vec);
            m_shape = shape_nd.pop_back();
            m_center = center_type::from_vec(m_shape.vec / 2);

            m_cutoff_fftfreq_sqd = clamp(cutoff, coord_type{0}, coord_type{0.5});
            m_cutoff_fftfreq_sqd *= m_cutoff_fftfreq_sqd;
        }

        NOA_HD constexpr void operator()(index_type batch, index_type y, index_type x) const requires (N == 2) {
            // Compute the frequency corresponding to the gid and inverse transform.
            const index_type v = noa::fft::index2frequency<is_output_centered>(y, m_shape[0]);
            const auto fftfreq = vec_type::from_vec(v, x) / m_f_shape; // x == u
            if (dot(fftfreq, fftfreq) > m_cutoff_fftfreq_sqd) {
                m_output(batch, y, x) = 0;
                return;
            }
            m_output(batch, y, x) = static_cast<output_value_type>(interpolate_(batch, fftfreq));
        }

        NOA_HD constexpr void operator()(index_type batch, index_type z, index_type y, index_type x) const requires (N == 3) {
            const index_type w = noa::fft::index2frequency<is_output_centered>(z, m_shape[0]);
            const index_type v = noa::fft::index2frequency<is_output_centered>(y, m_shape[1]);
            const auto fftfreq = vec_type::from_vec(w, v, x) / m_f_shape; // x == u
            if (dot(fftfreq, fftfreq) > m_cutoff_fftfreq_sqd) {
                m_output(batch, z, y, x) = 0;
                return;
            }
            m_output(batch, z, y, x) = static_cast<output_value_type>(interpolate_(batch, fftfreq));
        }

    private:
        NOA_HD constexpr input_value_type interpolate_(index_type batch, const vec_type& fftfreq) {
            // Rotate the spectrum.
            vec_type rotated_fftfreq;
            if constexpr (has_multiple_rotations) {
                if constexpr (nt::is_quaternion_v<rotation_single_type>) {
                    rotated_fftfreq = m_inverse_rotation[batch].rotate(fftfreq);
                } else {
                    rotated_fftfreq = m_inverse_rotation[batch] * fftfreq;
                }
            } else {
                if constexpr (nt::is_quaternion_v<rotation_single_type>) {
                    rotated_fftfreq = m_inverse_rotation.rotate(fftfreq);
                } else {
                    rotated_fftfreq = m_inverse_rotation * fftfreq;
                }
            }

            // Non-redundant transform, so flip to the valid Hermitian pair, if necessary.
            input_real_type conj = 1;
            if (rotated_fftfreq[N - 1] < coord_type{0}) {
                rotated_fftfreq = -rotated_fftfreq;
                if constexpr (nt::is_complex_v<input_real_type>)
                    conj = -1;
            }

            // Convert back to indices.
            vec_type rotated_frequencies = rotated_fftfreq * m_f_shape;
            rotated_frequencies[0] += m_center[0];
            if constexpr (N == 3)
                rotated_frequencies[1] += m_center[1];

            // Interpolate.
            auto value = m_input(rotated_frequencies, batch);
            if constexpr (nt::is_complex_v<input_real_type>)
                value.imag *= conj;
            else
                (void) conj;

            // Phase-shift the interpolated value.
            // It is a post-shift, so we need to use the original fftfreq (the ones in the output reference frame)
            if constexpr (nt::is_complex_v<input_value_type> and has_postshifts) {
                if constexpr (has_multiple_postshifts) {
                    value *= noa::signal::phase_shift<input_value_type>(m_post_forward_shift[batch], fftfreq);
                } else {
                    value *= noa::signal::phase_shift<input_value_type>(m_post_forward_shift, fftfreq);
                }
            }
            return value;
        }

    private:
        interpolator_type m_input;
        output_accessor_type m_output;
        rotation_type m_inverse_rotation;
        shape_type m_shape;
        vec_type m_f_shape;
        center_type m_center;
        coord_type m_cutoff_fftfreq_sqd;
        NOA_NO_UNIQUE_ADDRESS postshift_type m_post_forward_shift;
    };
}
