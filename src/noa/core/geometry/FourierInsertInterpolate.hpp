#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/geometry/FourierUtilities.hpp"
#include "noa/core/types/Shape.hpp"

namespace noa::geometry::guts {
    template<Remap REMAP,
             nt::sinteger Index,
             nt::batched_parameter Scale,
             nt::batched_parameter Rotate,
             typename EWSCurvature,
             nt::interpolator_spectrum_nd<2> InputSlice,
             nt::interpolator_spectrum_nd_or_empty<2> InputSliceWeight,
             nt::writable_nd<3> OutputVolume,
             nt::writable_nd_or_empty<3> OutputVolumeWeight>
    class FourierInsertInterpolate {
        static constexpr bool IS_VOLUME_CENTERED = REMAP.is_xx2xc();
        static constexpr bool IS_VOLUME_RFFT = REMAP.is_xx2hx();

        using index_type = Index;
        using scale_type = Scale;
        using rotate_type = Rotate;
        using ews_type = EWSCurvature;
        using coord_type = nt::value_type_twice_t<rotate_type>;
        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;
        using shape_nd_type = Shape<index_type, 3 - IS_VOLUME_RFFT>;

        using input_type = InputSlice;
        using input_weight_type = InputSliceWeight;
        using output_type = OutputVolume;
        using output_weight_type = OutputVolumeWeight;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static constexpr bool has_input_weights = not nt::empty<input_weight_type>;
        static constexpr bool has_output_weights = not nt::empty<output_weight_type>;
        using output_weight_value_type = nt::value_type_t<output_weight_type>;
        using input_weight_value_type = std::conditional_t<
            has_input_weights, nt::mutable_value_type_t<input_weight_type>, output_weight_type>;

        static_assert(guts::fourier_projection_transform_types<scale_type, rotate_type, ews_type> and
                      guts::fourier_projection_types<input_type, output_type> and
                      guts::fourier_projection_weight_types<input_weight_type, output_weight_type>);

    public:
        FourierInsertInterpolate(
            const input_type& input_slices,
            const input_weight_type& input_weights,
            const Shape4<index_type>& input_slice_shape,
            const output_type& output_volume,
            const output_weight_type& output_weights,
            const Shape4<index_type>& output_volume_shape,
            const scale_type& fwd_scaling,
            const rotate_type& inv_rotation,
            coord_type fftfreq_sinc,
            coord_type fftfreq_blackman,
            coord_type fftfreq_cutoff,
            const Shape4<index_type>& target_shape,
            const ews_type& ews_radius
        ) :
            m_input_slices(input_slices),
            m_output_volume(output_volume),
            m_inv_rotation(inv_rotation),
            m_slice_count(input_slice_shape[0]),
            m_input_weights(input_weights),
            m_output_weights(output_weights),
            m_fwd_scaling(fwd_scaling)
        {
            const auto slice_shape_2d = input_slice_shape.filter(2, 3);
            m_f_slice_shape = coord2_type::from_vec(slice_shape_2d.vec);

            const auto grid_shape = output_volume_shape.pop_front();
            const auto l_target_shape = any(target_shape == 0) ? grid_shape : target_shape.pop_front();
            m_grid_shape = grid_shape.template pop_back<IS_VOLUME_RFFT>();
            m_f_target_shape = coord3_type::from_vec(l_target_shape.vec);

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (not nt::empty<ews_type>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_type{};

            m_fftfreq_cutoff_sqd = max(fftfreq_cutoff, coord_type{});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;

            // Clamp the windowed-sinc to ensure it's at least one pixel thick.
            const auto max_output_size = static_cast<coord_type>(min(l_target_shape));
            m_fftfreq_sinc = max(fftfreq_sinc, 1 / max_output_size);
            m_fftfreq_blackman = max(fftfreq_blackman, 1 / max_output_size);
        }

        // For every voxel of the grid.
        NOA_HD void operator()(index_type oz, index_type oy, index_type ox) const noexcept {
            const auto frequency = noa::fft::index2frequency<IS_VOLUME_CENTERED, IS_VOLUME_RFFT>(
                Vec{oz, oy, ox}, m_grid_shape);
            const auto fftfreq = coord3_type::from_vec(frequency) / m_f_target_shape;
            if (dot(fftfreq, fftfreq) > m_fftfreq_cutoff_sqd)
                return;

            input_value_type value{};
            input_weight_value_type weights{};

            for (index_type i{}; i < m_slice_count; ++i) {
                const auto [fftfreq_z, fftfreq_2d] = guts::fourier_grid2slice(
                    fftfreq, m_fwd_scaling, m_inv_rotation, i, m_ews_diam_inv);

                input_value_type i_value{};
                input_weight_value_type i_weights{};
                if (abs(fftfreq_z) <= m_fftfreq_blackman) { // the slice affects the voxel
                    const auto window = guts::windowed_sinc(fftfreq_z, m_fftfreq_sinc, m_fftfreq_blackman);
                    const auto frequency_2d = fftfreq_2d * m_f_slice_shape;

                    i_value = m_input_slices.interpolate_spectrum_at(frequency_2d, i) *
                              static_cast<input_real_type>(window);

                    if constexpr (has_output_weights) {
                        if constexpr (has_input_weights) {
                            i_weights = m_input_weights.interpolate_spectrum_at(frequency_2d, i) *
                                        static_cast<input_weight_value_type>(window);
                        } else {
                            i_weights = static_cast<input_weight_value_type>(window); // input_weight=1
                        }
                    }
                }
                value += i_value;
                if constexpr (has_output_weights)
                    weights += i_weights;
            }

            // The transformation preserves the hermitian symmetry, so there's nothing else to do.
            m_output_volume(oz, oy, ox) += cast_or_abs_squared<output_value_type>(value);
            if constexpr (has_output_weights)
                m_output_weights(oz, oy, ox) += cast_or_abs_squared<output_weight_value_type>(weights);
        }

    private:
        input_type m_input_slices;
        output_type m_output_volume;

        rotate_type m_inv_rotation;
        shape_nd_type m_grid_shape;
        coord3_type m_f_target_shape;
        coord2_type m_f_slice_shape;
        index_type m_slice_count;

        coord_type m_fftfreq_cutoff_sqd;
        coord_type m_fftfreq_sinc;
        coord_type m_fftfreq_blackman;

        NOA_NO_UNIQUE_ADDRESS input_weight_type m_input_weights;
        NOA_NO_UNIQUE_ADDRESS output_weight_type m_output_weights;
        NOA_NO_UNIQUE_ADDRESS scale_type m_fwd_scaling;
        NOA_NO_UNIQUE_ADDRESS ews_type m_ews_diam_inv{};
    };
}
