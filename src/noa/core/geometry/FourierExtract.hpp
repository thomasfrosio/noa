#pragma once

#include "noa/core/Remap.hpp"
#include "noa/core/geometry/FourierUtilities.hpp"
#include "noa/core/utils/Atomic.hpp"

namespace noa::geometry::guts {
    template<Remap REMAP,
             nt::sinteger Index,
             nt::batched_parameter Scale,
             nt::batched_parameter Rotate,
             typename EWSCurvature,
             nt::interpolator_spectrum_nd<3> InputVolume,
             nt::interpolator_spectrum_nd_or_empty<3> InputWeight,
             nt::writable_nd<3> OutputSlice,
             nt::writable_nd_or_empty<3> OutputWeight>
    class FourierExtract {
        static constexpr bool ARE_SLICES_CENTERED = REMAP.is_xx2xc();
        static constexpr bool ARE_SLICES_RFFT = REMAP.is_xx2hx();

        using index_type = Index;
        using shape_nd_type = Shape<index_type, 2 - ARE_SLICES_RFFT>;

        using input_type = InputVolume;
        using input_weight_type = InputWeight;
        using output_type = OutputSlice;
        using output_weight_type = OutputWeight;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using output_real_type = nt::value_type_t<output_value_type>;
        using output_weight_value_type = nt::value_type_t<output_weight_type>;

        using scale_type = Scale;
        using rotate_type = Rotate;
        using ews_type = EWSCurvature;
        using coord_type = nt::value_type_t<rotate_type>;
        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;

        static_assert(guts::fourier_projection_transform_types<scale_type, rotate_type, ews_type> and
                      guts::fourier_projection_types<input_type, output_type> and
                      guts::fourier_projection_weight_types<input_weight_type, output_weight_type>);

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
            m_slice_shape = slice_shape_2d.template pop_back<ARE_SLICES_RFFT>();
            m_f_slice_shape = coord2_type::from_vec(slice_shape_2d.vec);

            // Use the grid shape as backup.
            const auto grid_shape_3d = input_volume_shape.pop_front();
            const auto target_shape_3d = any(target_shape == 0) ? grid_shape_3d : target_shape.pop_front();
            m_f_target_shape = coord3_type::from_vec(target_shape_3d.vec);

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (not nt::empty<ews_type>)
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
        NOA_HD constexpr void operator()(index_type batch, index_type oy, index_type ou) const {
            const coord3_type fftfreq_3d = compute_fftfreq_in_volume_(batch, oy, ou);

            output_value_type value{};
            output_weight_value_type weight{};

            if (dot(fftfreq_3d, fftfreq_3d) <= m_fftfreq_cutoff_sqd) {
                const auto frequency_3d = fftfreq_3d * m_f_target_shape;
                value = cast_or_abs_squared<output_value_type>(
                        m_input_volume.interpolate_spectrum_at(frequency_3d));

                // Passing no input weights is technically allowed, but does nothing other than returning ones.
                if constexpr (not nt::empty<output_weight_type>) {
                    if constexpr (not nt::empty<input_weight_type>) {
                        weight = static_cast<output_weight_value_type>(
                                m_input_weights.interpolate_spectrum_at(frequency_3d));
                    } else {
                        weight = 1;
                    }
                }
            }

            m_output_slices(batch, oy, ou) = value;
            if constexpr (not nt::empty<output_weight_type>)
                m_output_weights(batch, oy, ou) = weight;
        }

        // For every pixel of every slice to extract.
        // w is the index within the windowed-sinc convolution along the z of the grid.
        NOA_HD constexpr void operator()(index_type batch, index_type ow, index_type oy, index_type ox) const {
            coord3_type fftfreq_3d = compute_fftfreq_in_volume_(batch, oy, ox);

            // Additional z component, within the grid coordinate system.
            const auto fftfreq_z_offset = guts::w_index_to_fftfreq_offset(ow, m_blackman_size, m_f_target_shape[0]);
            fftfreq_3d[0] += fftfreq_z_offset;

            if (dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd)
                return;

            const auto frequency_3d = fftfreq_3d * m_f_target_shape;
            const auto convolution_weight =
                    guts::windowed_sinc(fftfreq_z_offset, m_fftfreq_sinc, m_fftfreq_blackman) /
                    m_w_window_sum; // convolution sum is 1

            const auto value = m_input_volume.interpolate_spectrum_at(frequency_3d);
            ng::atomic_add(
                    m_output_slices,
                    cast_or_abs_squared<output_value_type>(value) *
                    static_cast<output_real_type>(convolution_weight),
                    batch, oy, ox);

            if constexpr (not nt::empty<output_weight_type>) {
                output_weight_value_type weight{1};
                if constexpr (not nt::empty<input_weight_type>) {
                    weight = static_cast<output_weight_value_type>(
                            m_input_weights.interpolate_spectrum_at(frequency_3d));
                }
                ng::atomic_add(
                        m_output_weights,
                        weight * static_cast<output_weight_value_type>(convolution_weight),
                        batch, oy, ox);
            }
        }

    private:
        // The indexes give us the fftfreq in the coordinate system of the slice to extract.
        // This function transforms this fftfreq to the coordinate system of the volume.
        NOA_HD coord3_type compute_fftfreq_in_volume_(index_type batch, index_type oy, index_type ox) const {
            const auto frequency_2d = noa::fft::index2frequency<ARE_SLICES_CENTERED, ARE_SLICES_RFFT>(
                    Vec{oy, ox}, m_slice_shape);
            const auto fftfreq_2d = coord2_type::from_vec(frequency_2d) / m_f_slice_shape;
            return guts::fourier_slice2grid(
                    fftfreq_2d, m_inv_scaling, m_fwd_rotation, batch, m_ews_diam_inv);
        }

    private:
        input_type m_input_volume;
        output_type m_output_slices;

        rotate_type m_fwd_rotation;
        coord3_type m_f_target_shape;
        coord2_type m_f_slice_shape;
        shape_nd_type m_slice_shape;

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
