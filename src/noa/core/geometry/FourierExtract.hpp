#pragma once

#include "noa/core/fft/RemapInterface.hpp"
#include "noa/core/geometry/FourierUtilities.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/utils/Atomic.hpp"

namespace noa::geometry {
    template<noa::fft::Remap REMAP,
             typename Index,                    // i32|i64
             typename Scale,                    // Accessor(Value)<Mat22> or Empty
             typename Rotate,                   // Accessor(Value)<Mat33|Quaternion>
             typename EWSCurvature,             // Vec2 or Empty
             typename InputVolumeInterpolator,  // Interpolator3d<f32|f64|c32|c64>
             typename InputWeightInterpolator,  // Interpolator3d<f32|f64> or Empty
             typename OutputSliceAccessor,      // Accessor<f32|f64|c32|c64>
             typename OutputWeightAccessor>     // Accessor<f32|f64> or Empty
    class FourierExtract {
        static constexpr auto remap = noa::fft::RemapInterface(REMAP);
        static_assert(remap.is_hc2xx() and remap.is_xx2hx());
        static constexpr bool are_slices_centered = remap.is_xc2xx();

        using index_type = Index;
        using scale_type = Scale;
        using rotate_type = Rotate;
        using ews_type = EWSCurvature;
        using input_type = InputVolumeInterpolator;
        using input_weight_type = InputWeightInterpolator;
        using output_type = OutputSliceAccessor;
        using output_weight_type = OutputWeightAccessor;

        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_weight_value_type = nt::mutable_value_type_t<input_weight_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using output_real_type = nt::value_type_t<output_value_type>;
        using output_weight_value_type = nt::value_type_t<output_weight_type>;
        using coord_type = nt::value_type_twice_t<rotate_type>;
        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;

        static constexpr bool has_weights = not std::is_empty_v<input_weight_type>;
        static_assert(std::is_empty_v<input_weight_type> == std::is_empty_v<output_weight_type>);
        static_assert(nt::is_interpolator_nd_v<input_type, 3> and
                      nt::is_accessor_nd_v<output_type, 3> and
                      (nt::is_interpolator_nd_v<input_weight_type, 3> or not has_weights) and
                      (nt::is_accessor_nd_v<output_weight_type, 3> or not has_weights));

        static_assert(nt::is_any_v<ews_type, Empty, coord_type, Vec2<coord_type>> and
                      guts::is_valid_fourier_scaling_v<scale_type, coord_type> and
                      guts::is_valid_fourier_rotate_v<rotate_type> and
                      guts::is_valid_fourier_value_type_v<input_type, output_type> and
                      guts::is_valid_fourier_weight_type_v<input_weight_type, output_weight_type>);

    public:
        FourierExtract(
                const input_type& input_volume,
                const input_weight_type& input_weights,
                const Shape4<index_type>& input_volume_shape,
                const output_type& output_slices,
                const output_weight_type& output_weights,
                const Shape4<index_type>& output_slice_shape,
                const scale_type& inv_scaling,
                const rotate_type& fwd_rotation,
                coord_type fftfreq_sinc,
                coord_type fftfreq_blackman,
                coord_type fftfreq_cutoff,
                const Shape4<index_type>& target_shape,
                const ews_type& ews_radius
        )
                : m_input_volume(input_volume),
                  m_output_slices(output_slices),
                  m_fwd_rotation(fwd_rotation),
                  m_input_weights(input_weights),
                  m_output_weights(output_weights),
                  m_inv_scaling(inv_scaling)
        {
            const auto slice_shape_2d = output_slice_shape.filter(2, 3);
            m_slice_size_y = slice_shape_2d[0];
            m_f_slice_shape = coord2_type::from_vec(slice_shape_2d.vec);

            // Use the grid shape as backup.
            const auto grid_shape_3d = input_volume_shape.pop_front();
            const auto target_shape_3d = any(target_shape == 0) ? grid_shape_3d : target_shape.pop_front();
            m_f_target_shape = coord3_type::from_vec(target_shape_3d.vec);
            m_f_grid_zy_center = coord2_type::from_vec((grid_shape_3d.filter(0, 1) / 2).vec); // grid ZY center

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (not std::is_empty_v<ews_type>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_type{};

            m_fftfreq_cutoff_sqd = max(fftfreq_cutoff, coord_type{0});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;

            // This is along the w of the grid.
            m_fftfreq_sinc = max(fftfreq_sinc, 1 / m_f_target_shape[0]);
            m_fftfreq_blackman = max(fftfreq_blackman, 1 / m_f_target_shape[0]);
            tie(m_blackman_size, m_w_window_sum) = guts::z_window_spec<index_type>(
                    m_fftfreq_sinc, m_fftfreq_blackman, m_f_target_shape[0]);
        }

        [[nodiscard]] constexpr index_type windowed_sinc_size() const noexcept { return m_blackman_size; }

        // For every pixel of every slice to extract.
        NOA_HD void operator()(index_type batch, index_type y, index_type u) const noexcept {
            const coord3_type freq_3d = compute_fftfreq_in_volume_(batch, y, u);

            input_value_type value{};
            input_weight_value_type weight{};
            if (dot(freq_3d, freq_3d) <= m_fftfreq_cutoff_sqd) {
                value = guts::interpolate_grid_value(
                        freq_3d, m_f_target_shape, m_f_grid_zy_center, m_input_volume);
                if constexpr (has_weights) {
                    weight = guts::interpolate_grid_value(
                            freq_3d, m_f_target_shape, m_f_grid_zy_center, m_input_weights);
                }
            }

            m_output_slices(batch, y, u) = guts::cast_or_power_spectrum<output_value_type>(value);
            if constexpr (has_weights)
                m_output_weights(batch, y, u) = static_cast<output_weight_value_type>(weight);
        }

        // For every pixel of every slice to extract.
        // w is the index within the windowed-sinc convolution along the z of the grid.
        NOA_HD void operator()(index_type batch, index_type w, index_type y, index_type u) const noexcept {
            coord3_type freq_3d = compute_fftfreq_in_volume_(batch, y, u);

            // Additional z component, within the grid coordinate system.
            const auto fftfreq_z_offset = guts::w_index_to_fftfreq_offset(w, m_blackman_size, m_f_target_shape[0]);
            freq_3d[0] += fftfreq_z_offset;

            if (dot(freq_3d, freq_3d) > m_fftfreq_cutoff_sqd)
                return;

            // W-window.
            const auto convolution_weight =
                    guts::windowed_sinc(fftfreq_z_offset, m_fftfreq_sinc, m_fftfreq_blackman) /
                    m_w_window_sum; // convolution sum is 1

            const auto value = guts::interpolate_grid_value(
                    freq_3d, m_f_target_shape, m_f_grid_zy_center, m_input_volume);
            ng::atomic_add(
                    m_output_slices,
                    guts::cast_or_power_spectrum<output_value_type>(value) *
                    static_cast<output_real_type>(convolution_weight),
                    batch, y, u);

            if constexpr (has_weights) {
                const auto weight = guts::interpolate_grid_value(
                        freq_3d, m_f_target_shape, m_f_grid_zy_center, m_input_weights);
                ng::atomic_add(
                        m_output_weights,
                        static_cast<output_weight_value_type>(weight) *
                        static_cast<output_weight_value_type>(convolution_weight),
                        batch, y, u);
            }
        }

    private:
        // The indexes give us the fftfreq in the coordinate system of the slice to extract.
        // This function transforms this fftfreq to the coordinate system of the volume.
        NOA_HD coord3_type compute_fftfreq_in_volume_(
                index_type batch, index_type y, index_type u
        ) const noexcept {
            const index_type v = noa::fft::index2frequency<are_slices_centered>(y, m_slice_size_y);
            const auto fftfreq_2d = coord2_type::from_vec(v, u) / m_f_slice_shape;
            return guts::fourier_slice2grid(
                    fftfreq_2d, m_inv_scaling, m_fwd_rotation, batch, m_ews_diam_inv);
        }

    private:
        input_type m_input_volume;
        output_type m_output_slices;

        rotate_type m_fwd_rotation;
        coord3_type m_f_target_shape;
        coord2_type m_f_slice_shape;
        coord2_type m_f_grid_zy_center;
        index_type m_slice_size_y;

        coord_type m_fftfreq_cutoff_sqd;
        coord_type m_fftfreq_sinc;
        coord_type m_fftfreq_blackman;
        index_type m_blackman_size;
        coord_type m_w_window_sum;

        NOA_NO_UNIQUE_ADDRESS input_weight_type m_input_weights;
        NOA_NO_UNIQUE_ADDRESS output_weight_type m_output_weights;
        NOA_NO_UNIQUE_ADDRESS scale_type m_inv_scaling;
        NOA_NO_UNIQUE_ADDRESS ews_type m_ews_diam_inv{};
    };
}
