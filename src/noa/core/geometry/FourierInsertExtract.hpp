#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/geometry/FourierUtilities.hpp"
#include "noa/core/utils/Atomic.hpp"

namespace noa::geometry::guts {
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
    template<Remap REMAP,
             nt::sinteger Index,
             nt::batched_parameter InputScale,
             nt::batched_parameter InputRotate,
             nt::batched_parameter OutputScale,
             nt::batched_parameter OutputRotate,
             typename EWSCurvature,
             nt::interpolator_spectrum_nd<2> InputSlice,
             nt::interpolator_spectrum_nd_or_empty<2> InputSliceWeight,
             nt::writable_nd<3> OutputSlice,
             nt::writable_nd_or_empty<3> OutputSliceWeight>
    class FourierInsertExtract {
        static constexpr bool ARE_OUTPUT_SLICES_CENTERED = REMAP.is_xx2xc();
        static constexpr bool ARE_OUTPUT_SLICES_RFFT = REMAP.is_xx2hx();

        using index_type = Index;
        using shape_nd_type = Shape<index_type, 2 - ARE_OUTPUT_SLICES_RFFT>;

        // Transformations:
        using input_scale_type = InputScale;
        using input_rotate_type = InputRotate;
        using output_scale_type = OutputScale;
        using output_rotate_type = OutputRotate;
        using ews_type = EWSCurvature;
        using coord_type = nt::value_type_twice_t<input_rotate_type>;
        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;

        // Input/Output value types:
        using input_type = InputSlice;
        using input_weight_type = InputSliceWeight;
        using output_type = OutputSlice;
        using output_weight_type = OutputSliceWeight;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_real_type = nt::value_type_t<output_value_type>;
        using output_weight_value_type = nt::value_type_t<output_weight_type>;
        static constexpr bool has_input_weights = not nt::empty<input_weight_type>;
        static constexpr bool has_output_weights = not nt::empty<output_weight_type>;

        static_assert(guts::fourier_projection_transform_types<input_scale_type, input_rotate_type, ews_type> and
                      guts::fourier_projection_transform_types<output_scale_type, output_rotate_type, ews_type> and
                      guts::fourier_projection_types<input_type, output_type> and
                      guts::fourier_projection_weight_types<input_weight_type, output_weight_type>);

        // Optional operator requires atomic_add.
        static constexpr bool are_outputs_atomic =
            nt::atomic_addable_nd<output_type, 3> and
            nt::atomic_addable_nd_or_empty<output_weight_type, 3>;

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
            m_input_weights(input_weights),
            m_output_weights(output_weights),
            m_insert_fwd_scaling(insert_fwd_scaling),
            m_extract_inv_scaling(extract_inv_scaling),
            m_add_to_output(add_to_output),
            m_correct_multiplicity(correct_multiplicity)
        {
            const auto l_input_shape = input_shape.filter(2, 3);
            const auto l_output_shape = output_shape.filter(2, 3);

            m_f_input_shape = coord2_type::from_vec(l_input_shape.vec);
            m_f_output_shape = coord2_type::from_vec(l_output_shape.vec);
            m_output_shape = l_output_shape.template pop_back<ARE_OUTPUT_SLICES_RFFT>();

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (not nt::empty<ews_type>)
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
        NOA_HD constexpr void operator()(index_type batch, index_type y, index_type x) const {
            const coord3_type fftfreq_3d = compute_fftfreq_in_volume_(batch, y, x);

            if (dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd) {
                if (not m_add_to_output) {
                    m_output_slices(batch, y, x) = output_value_type{};
                    if constexpr (has_output_weights)
                        m_output_weights(batch, y, x) = output_weight_value_type{};
                }
                return;
            }

            const auto value_and_weight = sample_virtual_volume_(fftfreq_3d, m_correct_multiplicity);

            auto& output = m_output_slices(batch, y, x);
            if (m_add_to_output)
                output += cast_or_abs_squared<output_value_type>(value_and_weight.first);
            else
                output = cast_or_abs_squared<output_value_type>(value_and_weight.first);

            if constexpr (has_output_weights) {
                auto& weight = m_output_weights(batch, y, x);
                if (m_add_to_output)
                    weight += static_cast<output_weight_value_type>(value_and_weight.second);
                else
                    weight = static_cast<output_weight_value_type>(value_and_weight.second);
            }
        }

        // Should be called for every pixel of every slice to extract and for every element in the z-windowed-sinc.
        // Of course, this makes the extraction much more expensive. Also, the output-slice could be unset, so
        // the caller may have to fill it with zeros first, depending on add_to_output.
        NOA_HD constexpr void operator()(
            index_type batch, index_type w, index_type y, index_type x
        ) const requires are_outputs_atomic {
            coord3_type fftfreq_3d = compute_fftfreq_in_volume_(batch, y, x);

            // Get and add the volume z-offset for the z-windowed-sinc.
            const auto fftfreq_z_offset = guts::w_index_to_fftfreq_offset(w, m_extract_blackman_size, m_volume_z);
            fftfreq_3d[0] += fftfreq_z_offset;

            if (dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd)
                return;

            // The weights cannot be corrected on-the-fly in this case because
            // the final weight is unknown at this point!
            const auto value_and_weight = sample_virtual_volume_(fftfreq_3d, false);

            // z-windowed sinc.
            const auto convolution_weight =
                guts::windowed_sinc(fftfreq_z_offset, m_extract_fftfreq_sinc, m_extract_fftfreq_blackman) /
                m_extract_window_total_weight;

            // Add the contribution for this z-offset. The z-convolution is essentially a simple weighted mean.
            ng::atomic_add(
                m_output_slices,
                cast_or_abs_squared<output_value_type>(value_and_weight.first) *
                static_cast<output_real_type>(convolution_weight),
                batch, y, x);
            if constexpr (has_output_weights) {
                ng::atomic_add(
                    m_output_weights,
                    static_cast<output_weight_value_type>(value_and_weight.second) *
                    static_cast<output_weight_value_type>(convolution_weight),
                    batch, y, x);
            }
        }

    private:
        // The indexes give us the fftfreq in the coordinate system of the slice to extract.
        // This function transforms this fftfreq to the coordinate system of the virtual volume.
        NOA_HD coord3_type compute_fftfreq_in_volume_(index_type batch, index_type y, index_type x) const noexcept {
            const auto frequency_2d = noa::fft::index2frequency<ARE_OUTPUT_SLICES_CENTERED, ARE_OUTPUT_SLICES_RFFT>(
                Vec{y, x}, m_output_shape);
            const auto fftfreq_2d = coord2_type::from_vec(frequency_2d) / m_f_output_shape;
            return guts::fourier_slice2grid(
                fftfreq_2d, m_extract_inv_scaling, m_extract_fwd_rotation, batch, m_ews_diam_inv);
        }

        NOA_HD auto sample_virtual_volume_(const coord3_type& fftfreq_3d, bool correct_multiplicity) const noexcept {
            using input_weight_value_type =
                std::conditional_t<has_input_weights, nt::mutable_value_type_t<input_weight_type>,
                std::conditional_t<has_output_weights, output_weight_value_type, input_real_type>>;

            input_value_type value{};
            input_weight_value_type weight{};

            // For every slice to insert...
            for (index_type i{}; i < m_input_count; ++i) {
                // Project the 3d frequency onto that input-slice.
                // fftfreq_z is along the normal of that input-slice.
                const auto [fftfreq_z, fftfreq_yx] = guts::fourier_grid2slice(
                    fftfreq_3d, m_insert_fwd_scaling, m_insert_inv_rotation, i, m_ews_diam_inv);

                // Add the contribution of this slice to that frequency.
                // Compute only if this slice affects the voxel.
                // If we fall exactly at the blackman cutoff, the value is 0, so exclude the equality case too.
                if (abs(fftfreq_z) < m_insert_fftfreq_blackman) {
                    const auto windowed_sinc = guts::windowed_sinc(
                        fftfreq_z, m_insert_fftfreq_sinc, m_insert_fftfreq_blackman);

                    const auto frequency_yx = fftfreq_yx * m_f_input_shape;
                    value += m_input_slices.interpolate_spectrum_at(frequency_yx, i) *
                             static_cast<input_real_type>(windowed_sinc);

                    if constexpr (has_input_weights) {
                        weight += m_input_weights.interpolate_spectrum_at(frequency_yx, i) *
                                  static_cast<input_weight_value_type>(windowed_sinc);
                    } else {
                        weight += static_cast<input_weight_value_type>(windowed_sinc); // input_weight=1
                    }
                }
            }

            // Correct for the multiplicity (assuming this is all the signal at that frequency).
            if (correct_multiplicity) {
                const auto final_weight = max(input_weight_value_type{1}, weight);
                value /= static_cast<input_real_type>(final_weight);
            }

            return Pair{value, weight};
        }

    private:
        input_type m_input_slices;
        output_type m_output_slices;

        input_rotate_type m_insert_inv_rotation;
        output_rotate_type m_extract_fwd_rotation;
        coord2_type m_f_output_shape;
        coord2_type m_f_input_shape;
        shape_nd_type m_output_shape;
        index_type m_input_count;

        coord_type m_volume_z;
        coord_type m_fftfreq_cutoff_sqd;
        coord_type m_insert_fftfreq_sinc;
        coord_type m_insert_fftfreq_blackman;
        coord_type m_extract_fftfreq_sinc;
        coord_type m_extract_fftfreq_blackman;
        index_type m_extract_blackman_size;
        coord_type m_extract_window_total_weight;

        NOA_NO_UNIQUE_ADDRESS input_weight_type m_input_weights;
        NOA_NO_UNIQUE_ADDRESS output_weight_type m_output_weights;
        NOA_NO_UNIQUE_ADDRESS input_scale_type m_insert_fwd_scaling;
        NOA_NO_UNIQUE_ADDRESS output_scale_type m_extract_inv_scaling;
        NOA_NO_UNIQUE_ADDRESS ews_type m_ews_diam_inv{};

        bool m_add_to_output;
        bool m_correct_multiplicity;
    };
}
