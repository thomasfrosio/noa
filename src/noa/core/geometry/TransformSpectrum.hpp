#pragma once

#include "noa/core/types/Shape.hpp"
#include "noa/core/Interpolation.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/geometry/Transform.hpp"

// FIXME    For even sizes, there's an asymmetry due to the fact that there's only one Nyquist. After fftshift,
//          this extra frequency is on the left side of the axis. However, since we work with non-redundant
//          transforms, there's a slight error that can be introduced. For the elements close (+-1 element) to x=0
//          and close (+-1 element) to y=z=0.5, if during the rotation x becomes negative and we have to flip it, the
//          interpolator will incorrectly weight towards 0 the output value. This is simply due to the fact that on
//          the right side of the axis, there's no Nyquist (the axis stops and n+1 element is OOB, i.e. =0).
//          Anyway, the fix should be done in the interpolator, not here. Maybe create a InterpolatorRFFT?

namespace noa::geometry::guts {
    /// 3d or 4d iwise operator to compute linear transformations (one rotation/scaling followed by one translation)
    /// of 2d or 3d array(s) by directly manipulating their 2d or 3d Fourier transforms(s).
    /// \details While the Transform operator supports complex arrays, it does not support FFT layouts.
    ///          This class adds support for some of these.
    /// \tparam N           2 or 3.
    /// \tparam REMAP       Every output layout is supported. Input layouts are controlled by \p input.
    /// \tparam Index       Signed integer.
    /// \tparam Rotate      BatchedParameter of Mat<T, N, N> or, iff N=3, Quaternion<T>.
    /// \tparam PostShift   BatchedParameter of Empty or Vec<T, N>.
    /// \tparam Input       N-d spectrum interpolator.
    /// \tparam Output      (N + 1)-d writable.
    template<size_t N, Remap REMAP,
             nt::integer Index,
             nt::batched_parameter Rotate,
             nt::batched_parameter PostShift,
             nt::interpolator_spectrum_nd<N> Input,
             nt::writable_nd<N + 1> Output>
    class TransformSpectrum {
    public:
        static constexpr bool IS_DST_CENTERED = REMAP.is_xx2xc();
        static constexpr bool IS_DST_RFFT = REMAP.is_xx2hx();
        using index_type = Index;
        using shape_type = Shape<index_type, N - IS_DST_RFFT>;
        using indices_type = Vec<index_type, N>;

        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::are_power_spectrum_value_types_v<input_value_type, output_value_type>);

        // Rotation:
        using rotation_parameter_type = Rotate;
        using rotation_type = nt::value_type_t<rotation_parameter_type>;
        using coord_type = nt::value_type_t<rotation_type>;
        using vec_type = Vec<coord_type, N>;
        static_assert(nt::mat_of_shape<rotation_type, N, N> or (N == 3 and nt::quaternion<rotation_type>));

        // PostShift:
        using postshift_parameter_type = Rotate;
        using postshift_type = nt::value_type_t<PostShift>;
        static_assert(nt::empty<postshift_type> or
                      (nt::vec_of_type<postshift_type, coord_type> and
                       nt::vec_of_size<postshift_type, N>));

    public:
        TransformSpectrum(
                const input_type& input,
                const output_type& output,
                const Shape<index_type, N>& output_shape,
                const rotation_parameter_type& inverse_rotation,
                const postshift_parameter_type& post_forward_shift,
                coord_type cutoff
        ) : m_input(input),
            m_output(output),
            m_inverse_rotation(inverse_rotation),
            m_post_forward_shift(post_forward_shift)
        {
            m_shape = output_shape.pop_back();
            m_f_shape = vec_type::from_vec(output_shape.vec);

            m_cutoff_fftfreq_sqd = clamp(cutoff, 0, 0.5);
            m_cutoff_fftfreq_sqd *= m_cutoff_fftfreq_sqd;
        }

        template<nt::same_as<index_type>... I> requires (sizeof...(I) == N)
        NOA_HD constexpr void operator()(index_type batch, I... indices) const {
            // Given the output indices, compute the corresponding fftfreq.
            const auto frequency = noa::fft::index2frequency<IS_DST_CENTERED, IS_DST_RFFT>(
                    indices_type{indices...}, m_shape);
            const auto fftfreq = vec_type::from_vec(frequency) / m_f_shape;

            // Shortcut for the output cutoff.
            if (dot(fftfreq, fftfreq) > m_cutoff_fftfreq_sqd) {
                m_output(batch, indices...) = 0;
                return;
            }

            vec_type rotated_fftfreq = transform_vector(m_inverse_rotation[batch], fftfreq, batch);
            auto value = m_input.interpolate_spectrum_at(rotated_fftfreq * m_f_shape, batch);

            // Phase-shift the interpolated value.
            // It is a post-shift, so we need to use the original fftfreq (the ones in the output reference frame)
            if constexpr (nt::complex<input_value_type> and not nt::empty<postshift_type>)
                value *= noa::fft::phase_shift<input_value_type>(m_post_forward_shift[batch], fftfreq);

            m_output(batch, indices...) = cast_or_abs_squared<output_value_type>(value);
        }

    private:
        input_type m_input;
        output_type m_output;
        rotation_parameter_type m_inverse_rotation;
        shape_type m_shape;
        vec_type m_f_shape;
        coord_type m_cutoff_fftfreq_sqd;
        NOA_NO_UNIQUE_ADDRESS postshift_parameter_type m_post_forward_shift;
    };
}
