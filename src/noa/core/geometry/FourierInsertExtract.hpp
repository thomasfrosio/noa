#pragma once

#include "noa/core/fft/RemapInterface.hpp"
#include "noa/core/geometry/FourierUtilities.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/utils/Atomic.hpp"

namespace noa::geometry {
    /// Index-wise (3d or 4d) operator for extracting central-slices from a virtual volume made of (other) central-slices.
    /// \details There are two operator():
    ///     - The 3d operator, which should be called for every pixel of every slice to extract.
    ///     - The 4d operator, which has an additional dimension for an output windowed-sinc. Indeed, the extracted
    ///       slices can be convolved with a windowed-sinc along the z-axis of the volume (note that the convolution
    ///       is reduced to a simple weighted-sum), effectively applying a (smooth) rectangular mask along the z-axis
    ///       and centered on the ifft of the virtual volume.
    ///
    /// \note If the input slice|weight is complex and the corresponding output is real, the power-spectrum is saved.
    /// \note The weights are optional and can be real or complex (although in most cases they are real).
    ///       Creating one operator for the values and one for the weights is equivalent, but projecting the values
    ///       and weights in the same operator is often more efficient.
    template<noa::fft::Remap REMAP,
             typename Index,
             typename InputScale,
             typename InputRotate,
             typename OutputScale,
             typename OutputRotate,
             typename EWSCurvature,
             typename InputSliceInterpolator,
             typename InputWeightInterpolator,
             typename OutputSliceAccessor,
             typename OutputWeightAccessor>
    class FourierInsertExtract {
        static constexpr auto remap = noa::fft::RemapInterface(REMAP);
        static_assert(remap.is_hc2xx() and remap.is_xx2hx());
        static constexpr bool are_output_slices_centered = remap.is_xx2xc();

        using index_type = Index;
        using input_scale_type = InputScale;
        using input_rotate_type = InputRotate;
        using output_scale_type = OutputScale;
        using output_rotate_type = OutputRotate;
        using ews_type = EWSCurvature;
        using input_type = InputSliceInterpolator;
        using input_weight_type = InputWeightInterpolator;
        using output_type = OutputSliceAccessor;
        using output_weight_type = OutputWeightAccessor;

        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using input_weight_value_type = nt::mutable_value_type_t<input_weight_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using output_real_type = nt::value_type_t<output_value_type>;
        using output_weight_value_type = nt::value_type_t<output_weight_type>;
        using coord_type = nt::value_type_twice_t<input_rotate_type>;
        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;

        static constexpr bool has_input_weights = not std::is_empty_v<input_weight_type>;
        static constexpr bool has_output_weights = not std::is_empty_v<output_weight_type>;
        static_assert(nt::is_interpolator_nd_v<input_type, 2> and
                      nt::is_accessor_nd_v<output_type, 3> and
                      (nt::is_interpolator_nd_v<input_weight_type, 2> or not has_input_weights) and
                      (nt::is_accessor_nd_v<output_weight_type, 3> or not has_output_weights));

        static_assert(nt::is_any_v<ews_type, Empty, coord_type, Vec2<coord_type>> and
                      guts::is_valid_fourier_scaling_v<input_scale_type, coord_type> and
                      guts::is_valid_fourier_rotate_v<input_rotate_type> and
                      guts::is_valid_fourier_scaling_v<output_scale_type, coord_type> and
                      guts::is_valid_fourier_rotate_v<output_rotate_type> and
                      guts::is_valid_fourier_value_type_v<input_type, output_type> and
                      guts::is_valid_fourier_weight_type_v<input_weight_value_type, output_weight_value_type>);

    public:
        FourierInsertExtract(
                const input_type& input_slices,
                const input_weight_type& input_weights,
                const Shape4<index_type>& input_shape,
                const output_type& output_slices,
                const output_weight_type& output_weights,
                const Shape4<index_type>& output_shape,
                const input_scale_type& insert_fwd_scaling,
                const input_rotate_type& insert_inv_rotation,
                const output_scale_type& extract_inv_scaling,
                const output_rotate_type& extract_fwd_rotation,
                coord_type insert_fftfreq_sinc,
                coord_type insert_fftfreq_blackman,
                coord_type extract_fftfreq_sinc,
                coord_type extract_fftfreq_blackman,
                coord_type fftfreq_cutoff,
                bool add_to_output, bool correct_multiplicity,
                const ews_type& ews_radius
        ) :
                m_input_slices(input_slices),
                m_output_slices(output_slices),
                m_insert_inv_rotation(insert_inv_rotation),
                m_extract_fwd_rotation(extract_fwd_rotation),
                m_input_count(input_shape[0]),
                m_add_to_output(add_to_output),
                m_correct_multiplicity(correct_multiplicity),
                m_input_weights(input_weights),
                m_output_weights(output_weights),
                m_insert_fwd_scaling(insert_fwd_scaling),
                m_extract_inv_scaling(extract_inv_scaling)
        {
            const auto l_input_shape = input_shape.filter(2, 3);
            const auto l_output_shape = output_shape.filter(2, 3);

            m_f_input_shape = coord2_type::from_vec(l_input_shape.vec);
            m_f_output_shape = coord2_type::from_vec(l_output_shape.vec);
            m_output_slice_size_y = l_output_shape[0];
            m_f_input_center_y = static_cast<coord_type>(l_input_shape[0] / 2); // slice Y center

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (not std::is_empty_v<ews_type>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_type{};

            m_fftfreq_cutoff_sqd = max(fftfreq_cutoff, coord_type{0});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;

            // Of course, we have no z here, but the smallest axis is a good fallback.
            m_volume_z = static_cast<coord_type>(min(l_output_shape));
            m_insert_fftfreq_sinc = max(insert_fftfreq_sinc, 1 / m_volume_z);
            m_insert_fftfreq_blackman = max(insert_fftfreq_blackman, 1 / m_volume_z);
            m_extract_fftfreq_sinc = max(extract_fftfreq_sinc, 1 / m_volume_z);
            m_extract_fftfreq_blackman = max(extract_fftfreq_blackman, 1 / m_volume_z);
            tie(m_extract_blackman_size, m_extract_window_total_weight) = guts::z_window_spec<index_type>(
                    m_extract_fftfreq_sinc, m_extract_fftfreq_blackman, m_volume_z);
        }

        // Whether the operator is 4d. Otherwise, it is 3d.
        [[nodiscard]] constexpr auto is_iwise_4d() const noexcept -> bool {
            return m_extract_blackman_size > 1;
        }

        // Returns the size of the output (depth) window.
        [[nodiscard]] constexpr auto output_window_size() const noexcept -> index_type {
            return m_extract_blackman_size;
        }

        // Should be called for every pixel of every slice to extract.
        NOA_HD void operator()(index_type batch, index_type y, index_type u) const noexcept {
            const coord3_type fftfreq_3d = compute_fftfreq_in_volume_(batch, y, u);

            if (dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd) {
                if (not m_add_to_output)
                    m_output_slices(batch, y, u) = output_value_type{};
                return;
            }

            const auto value_and_weight = sample_virtual_volume_(fftfreq_3d, m_correct_multiplicity);
            if (m_add_to_output) {
                m_output_slices(batch, y, u) += value_and_weight.first;
                if constexpr (has_output_weights)
                    m_output_weights(batch, y, u) += static_cast<output_weight_value_type>(value_and_weight.second);
            } else {
                m_output_slices(batch, y, u) = value_and_weight.first;
                if constexpr (has_output_weights)
                    m_output_weights(batch, y, u) = static_cast<output_weight_value_type>(value_and_weight.second);
            }
        }

        // Should be called for every pixel of every slice to extract and for every element in the z-windowed-sinc.
        // Of course, this can be much more expensive to extract a slice. Also, the output-slice is not set, so
        // the caller may have to fill it with zeros first, depending on add_to_output.
        NOA_HD void operator()(index_type batch, index_type w, index_type y, index_type u) const noexcept {
            coord3_type fftfreq_3d = compute_fftfreq_in_volume_(batch, y, u);

            // Get and add the volume z-offset for the z-windowed-sinc.
            const auto fftfreq_z_offset = guts::w_index_to_fftfreq_offset(
                    w, m_extract_blackman_size, m_volume_z);
            fftfreq_3d[0] += fftfreq_z_offset;

            if (dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd)
                return;

            // The multiplicity/weights cannot be corrected on-the-fly in this case because
            // the final multiplicity/weight is unknown at this point!
            const auto value_and_weight = sample_virtual_volume_(fftfreq_3d, false);

            // z-windowed sinc.
            const auto convolution_weight =
                    guts::windowed_sinc(fftfreq_z_offset, m_extract_fftfreq_sinc, m_extract_fftfreq_blackman) /
                    m_extract_window_total_weight; // convolution sum is 1

            // Add the contribution for this z-offset. The z-convolution is essentially a simple weighted mean.
            ng::atomic_add(
                    m_output_slices,
                    value_and_weight.first * static_cast<output_real_type>(convolution_weight),
                    batch, y, u);
            if constexpr (has_output_weights) {
                ng::atomic_add(
                        m_output_weights,
                        static_cast<output_weight_value_type>(value_and_weight.second) *
                        static_cast<output_weight_value_type>(convolution_weight),
                        batch, y, u);
            }
        }

    private:
        // The indexes give us the fftfreq in the coordinate system of the slice to extract.
        // This function transforms this fftfreq to the coordinate system of the virtual volume.
        NOA_HD coord3_type compute_fftfreq_in_volume_(
                index_type batch, index_type y, index_type u
        ) const noexcept {
            const index_type v = noa::fft::index2frequency<are_output_slices_centered>(y, m_output_slice_size_y);
            auto fftfreq_2d = coord2_type::from_values(v, u) / m_f_output_shape;
            return guts::fourier_slice2grid(
                    fftfreq_2d, m_extract_inv_scaling, m_extract_fwd_rotation, batch, m_ews_diam_inv);
        }

        NOA_HD auto sample_virtual_volume_(const coord3_type& fftfreq_3d, bool correct_multiplicity) const noexcept {
            using weight_t = std::conditional_t<has_input_weights, input_weight_value_type, coord_type>;
            input_value_type value{};
            weight_t weight{};

            // For every slice to insert...
            for (index_type i = 0; i < m_input_count; ++i) {
                // 1. Project the 3d frequency onto that input-slice.
                //    fftfreq_z is along the normal of that input-slice.
                const auto [fftfreq_z, fftfreq_yx] = guts::fourier_grid2slice(
                        fftfreq_3d, m_insert_fwd_scaling, m_insert_inv_rotation, i, m_ews_diam_inv);

                // 2. Sample the input value and weight at this 3d frequency.
                input_value_type i_value{};
                weight_t i_weight{};

                // Compute only if this slice affects the voxel.
                // If we fall exactly at the blackman cutoff, the value is 0, so exclude the equality case too.
                if (abs(fftfreq_z) < m_insert_fftfreq_blackman) {
                    const auto windowed_sinc = guts::windowed_sinc(
                            fftfreq_z, m_insert_fftfreq_sinc, m_insert_fftfreq_blackman);

                    i_value = guts::interpolate_slice_value(
                            fftfreq_yx, m_f_input_shape, m_f_input_center_y, m_input_slices, i);
                    i_value *= static_cast<input_real_type>(windowed_sinc);

                    if constexpr (has_input_weights) {
                        i_weight = guts::interpolate_slice_value(
                                fftfreq_yx, m_f_input_shape, m_f_input_center_y, m_input_weights, i);
                        i_weight *= static_cast<input_weight_value_type>(windowed_sinc);
                    } else {
                        i_weight = static_cast<weight_t>(windowed_sinc); // defaults weight to 1
                    }
                }

                // Add the contribution of this slice to that frequency.
                value += i_value;
                weight += i_weight;
            }

            // 3. Correct for the multiplicity (assuming this is all the signal at that frequency).
            // Note that if the total weight is less than 1, we need to leave it down-weighted.
            if (correct_multiplicity) {
                const auto final_weight = max(weight_t{1}, weight);
                value /= static_cast<input_real_type>(final_weight);
            }

            return Pair{guts::cast_or_power_spectrum<output_value_type>(value), weight};
        }

    private:
        input_type m_input_slices;
        output_type m_output_slices;

        input_rotate_type m_insert_inv_rotation;
        output_rotate_type m_extract_fwd_rotation;
        coord2_type m_f_output_shape;
        coord2_type m_f_input_shape;
        coord_type m_f_input_center_y;
        index_type m_input_count;
        index_type m_output_slice_size_y;

        coord_type m_volume_z;
        coord_type m_fftfreq_cutoff_sqd;
        coord_type m_insert_fftfreq_sinc;
        coord_type m_insert_fftfreq_blackman;
        coord_type m_extract_fftfreq_sinc;
        coord_type m_extract_fftfreq_blackman;
        index_type m_extract_blackman_size;
        coord_type m_extract_window_total_weight;

        bool m_add_to_output;
        bool m_correct_multiplicity;

        NOA_NO_UNIQUE_ADDRESS input_weight_type m_input_weights;
        NOA_NO_UNIQUE_ADDRESS output_weight_type m_output_weights;
        NOA_NO_UNIQUE_ADDRESS input_scale_type m_insert_fwd_scaling;
        NOA_NO_UNIQUE_ADDRESS output_scale_type m_extract_inv_scaling;
        NOA_NO_UNIQUE_ADDRESS ews_type m_ews_diam_inv{};
    };
}
