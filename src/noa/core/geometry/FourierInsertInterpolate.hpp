#pragma once

#include "noa/core/fft/RemapInterface.hpp"
#include "noa/core/geometry/FourierUtilities.hpp"
#include "noa/core/geometry/Interpolator.hpp"

namespace noa::geometry {
    template<noa::fft::Remap REMAP,
             typename Index,                    // i32|i64
             typename Scale,                    // Accessor(Value)<Mat22> or Empty
             typename Rotate,                   // Accessor(Value)<Mat33|Quaternion>
             typename EWSCurvature,             // Vec2 or Empty
             typename InputSliceInterpolator,   // Interpolator2d<f32|f64|c32|c64>
             typename InputWeightInterpolator,  // Interpolator2d<f32|f64> or Empty
             typename OutputVolumeAccessor,     // Accessor<f32|f64|c32|c64>
             typename OutputWeightAccessor>     // Accessor<f32|f64> or Empty
    class FourierInsertInterpolate {
        static constexpr auto remap = noa::fft::RemapInterface(REMAP);
        static_assert(remap.is_hc2xx() and remap.is_xx2hx());
        static constexpr bool is_volume_centered = remap.is_xx2xc();

        using index_type = Index;
        using scale_type = Scale;
        using rotate_type = Rotate;
        using ews_type = EWSCurvature;
        using input_type = InputSliceInterpolator;
        using input_weight_type = InputWeightInterpolator;
        using output_type = OutputVolumeAccessor;
        using output_weight_type = OutputWeightAccessor;

        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using output_weight_value_type = nt::value_type_t<output_weight_type>;
        using coord_type = nt::value_type_twice_t<rotate_type>;
        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;

        static constexpr bool has_input_weights = not std::is_empty_v<input_weight_type>;
        static constexpr bool has_output_weights = not std::is_empty_v<output_weight_type>;
        static_assert(nt::is_interpolator_nd_v<input_type, 2> and
                      nt::is_accessor_nd_v<output_type, 3> and
                      (nt::is_interpolator_nd_v<input_weight_type, 2> or not has_input_weights) and
                      (nt::is_accessor_nd_v<output_weight_type, 3> or not has_output_weights));

        static_assert(nt::is_any_v<ews_type, Empty, coord_type, Vec2<coord_type>> and
                      guts::is_valid_fourier_scaling_v<scale_type, coord_type> and
                      guts::is_valid_fourier_rotate_v<rotate_type> and
                      guts::is_valid_fourier_value_type_v<input_type, output_type> and
                      guts::is_valid_fourier_weight_type_v<input_weight_type, output_weight_type>);

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
        )
                : m_input_slices(input_slices),
                  m_output_volume(output_volume),
                  m_inv_rotation(inv_rotation),
                  m_grid_shape(output_volume_shape.pop_front()),
                  m_slice_count(input_slice_shape[0]),
                  m_input_weights(input_weights),
                  m_output_weights(output_weights),
                  m_fwd_scaling(fwd_scaling)
        {
            const auto slice_shape_2d = input_slice_shape.filter(2, 3);
            m_f_slice_shape = coord2_type::from_vec(slice_shape_2d.vec);
            m_f_slice_y_center = static_cast<coord_type>(slice_shape_2d[0] / 2);

            const auto l_target_shape = any(target_shape == 0) ? m_grid_shape : target_shape.pop_front();
            m_f_target_shape = coord3_type::from_vec(l_target_shape.vec);

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (not std::is_empty_v<ews_type>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_type{};

            m_fftfreq_cutoff_sqd = max(fftfreq_cutoff, coord_type{0});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;

            // Clamp the windowed-sinc to ensure it's at least one pixel thick.
            const auto max_output_size = static_cast<coord_type>(min(l_target_shape));
            m_fftfreq_sinc = max(fftfreq_sinc, 1 / max_output_size);
            m_fftfreq_blackman = max(fftfreq_blackman, 1 / max_output_size);
        }

        // For every voxel of the grid.
        NOA_HD void operator()(index_type z, index_type y, index_type u) const noexcept { // x == u
            const index_type w = noa::fft::index2frequency<is_volume_centered>(z, m_grid_shape[0]);
            const index_type v = noa::fft::index2frequency<is_volume_centered>(y, m_grid_shape[1]);
            const auto orig_freq = coord3_type::from_values(w, v, u) / m_f_target_shape;
            if (dot(orig_freq, orig_freq) > m_fftfreq_cutoff_sqd)
                return;

            input_value_type value{};
            output_weight_value_type weights{};
            for (index_type i = 0; i < m_slice_count; ++i) {
                const auto [freq_z, freq_2d] = guts::fourier_grid2slice(
                        orig_freq, m_fwd_scaling, m_inv_rotation, i, m_ews_diam_inv);

                // If voxel is not affected by the slice, skip.
                input_value_type i_value{};
                output_weight_value_type i_weights{};
                if (abs(freq_z) <= m_fftfreq_blackman) {
                    const auto window = guts::windowed_sinc(
                            freq_z, m_fftfreq_sinc, m_fftfreq_blackman);

                    i_value = guts::interpolate_slice_value(
                            freq_2d, m_f_slice_shape, m_f_slice_y_center, m_input_slices, i);
                    i_value *= static_cast<nt::value_type_t<input_value_type>>(window);

                    if constexpr (has_output_weights) {
                        if constexpr (has_input_weights) {
                            i_weights = static_cast<output_weight_value_type>(guts::interpolate_slice_value(
                                    freq_2d, m_f_slice_shape, m_f_slice_y_center, m_input_weights, i));
                            i_weights *= static_cast<output_weight_value_type>(window);
                        } else {
                            i_weights = static_cast<output_weight_value_type>(window); // defaults weight to 1
                        }
                    }
                }
                value += i_value;
                if constexpr (has_output_weights)
                    weights += i_weights;
            }

            // The transformation preserves the hermitian symmetry, so there's nothing else to do.
            m_output_volume(z, y, u) += guts::cast_or_power_spectrum<output_value_type>(value);
            if constexpr (has_output_weights)
                m_output_weights(z, y, u) += weights;
        }

    private:
        input_type m_input_slices;
        output_type m_output_volume;

        rotate_type m_inv_rotation;
        shape3_type m_grid_shape;
        index_type m_slice_count;
        coord3_type m_f_target_shape;
        coord2_type m_f_slice_shape;
        coord_type m_f_slice_y_center;
        coord_type m_fftfreq_cutoff_sqd;
        coord_type m_fftfreq_sinc;
        coord_type m_fftfreq_blackman;

        NOA_NO_UNIQUE_ADDRESS input_weight_type m_input_weights;
        NOA_NO_UNIQUE_ADDRESS output_weight_type m_output_weights;
        NOA_NO_UNIQUE_ADDRESS scale_type m_fwd_scaling;
        NOA_NO_UNIQUE_ADDRESS ews_type m_ews_diam_inv{};
    };
}
