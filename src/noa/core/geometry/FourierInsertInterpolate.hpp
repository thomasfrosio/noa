#pragma once

#include "noa/core/geometry/FourierUtilities.hpp"
#include "noa/core/geometry/Interpolator.hpp"

namespace noa::geometry {
    /// Type parser/checker for FourierInsertInterpolateOp.
    /// \tparam REMAP                               Remap operator. H(C)2H(C).
    /// \tparam Index                               Input index type. A signed integer.
    /// \tparam Coord                               Coordinate type. f32 or f64.
    /// \tparam ScaleOrEmpty                        Empty, Mat22 or an accessor of Mat22.
    /// \tparam Rotate                              Empty, Mat33|Quaternion or an accessor of Mat33|Quaternion.
    /// \tparam EWSOrEmpty                          Empty, Coord or Vec2<Coord>.
    /// \tparam InputSliceInterpolator              A 2d interpolator (BHW) of any real or complex type.
    /// \tparam InputWeightInterpolatorOrEmpty      Empty, or a 2d interpolator (BHW) of any real or complex type.
    /// \tparam OutputVolumeAccessor                A 3d accessor (DHW) of any real or complex type.
    /// \tparam OutputWeightAccessorOrEmpty         Empty, or a 3d accessor (DHW) of any real or complex type.
    template<noa::fft::Remap REMAP,
             typename Index, typename Coord,
             typename ScaleOrEmpty, typename Rotate,
             typename EWSOrEmpty,
             typename InputSliceInterpolator,
             typename InputWeightInterpolatorOrEmpty,
             typename OutputVolumeAccessor,
             typename OutputWeightAccessorOrEmpty>
    class FourierInsertInterpolateConcept {
    public:
        static constexpr bool IS_VALID_REMAP =
                REMAP == noa::fft::Remap::HC2H or
                REMAP == noa::fft::Remap::HC2HC;

        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_OUTPUT_VOLUME_CENTERED = u8_REMAP & noa::fft::Layout::DST_CENTERED;

        static constexpr bool IS_VALID_COORD_INDEX =
                nt::is_sint_v<Index> and nt::is_any_v<Coord, f32, f64>;

        static constexpr bool HAS_INPUT_WEIGHT = not std::is_empty_v<InputWeightInterpolatorOrEmpty>;
        static constexpr bool HAS_OUTPUT_WEIGHT = not std::is_empty_v<OutputWeightAccessorOrEmpty>;
        static constexpr bool HAS_WEIGHTS = HAS_INPUT_WEIGHT and HAS_OUTPUT_WEIGHT;

        using input_value_type = nt::mutable_value_type_t<InputSliceInterpolator>;
        using output_value_type = nt::value_type_t<OutputVolumeAccessor>;
        using input_weight_type = nt::mutable_value_type_t<InputWeightInterpolatorOrEmpty>;
        using output_weight_type = nt::value_type_t<OutputWeightAccessorOrEmpty>;

        static constexpr bool IS_VALID_INPUT_SLICE =
                nt::is_interpolator_nd_v<InputSliceInterpolator, 2> and
                nt::is_real_or_complex_v<input_value_type>;

        static constexpr bool IS_VALID_INPUT_WEIGHT =
                not HAS_WEIGHTS or
                (nt::is_interpolator_nd_v<InputWeightInterpolatorOrEmpty, 2> and
                 nt::is_real_or_complex_v<input_weight_type>);

        static constexpr bool IS_VALID_OUTPUT_VOLUME =
                nt::is_accessor_nd_v<OutputVolumeAccessor, 3> and
                not std::is_const_v<output_value_type> and
                (nt::are_complex_v<output_value_type, input_value_type> or
                 (nt::is_real_v<output_value_type> and nt::is_real_or_complex_v<input_value_type>));

        static constexpr bool IS_VALID_OUTPUT_WEIGHT =
                not HAS_WEIGHTS or
                (nt::is_accessor_nd_v<OutputWeightAccessorOrEmpty, 3> and
                 not std::is_const_v<output_weight_type> and
                 (nt::are_complex_v<output_weight_type, input_weight_type> or
                  (nt::is_real_v<output_weight_type> and nt::is_real_or_complex_v<input_weight_type>)));

        static constexpr bool IS_VALID_SCALE =
                std::is_empty_v<ScaleOrEmpty> or nt::is_mat22_v<ScaleOrEmpty> or
                ((nt::is_accessor_nd_v<ScaleOrEmpty, 1> or std::is_pointer_v<ScaleOrEmpty>) and
                 nt::is_mat22_v<std::remove_pointer_t<nt::value_type_t<ScaleOrEmpty>>>);

        static constexpr bool IS_VALID_INPUT_ROTATION =
                nt::is_mat33_v<Rotate> or nt::is_quaternion_v<Rotate> or
                ((nt::is_accessor_nd_v<Rotate, 1> or std::is_pointer_v<Rotate>) and
                 (nt::is_mat33_v<std::remove_pointer_t<nt::value_type_t<Rotate>>> or
                  nt::is_quaternion_v<std::remove_pointer_t<nt::value_type_t<Rotate>>>));

        static constexpr bool IS_VALID_EWS =
                std::is_empty_v<EWSOrEmpty> or
                nt::is_any_v<EWSOrEmpty, Coord, Vec2<Coord>>;
    };

    /// Direct Fourier insertion, but this time looping through the grid.
    /// In practice, it allows to give an explicit "thickness" to the central slices.
    /// It also "looks" better (more symmetric; better/smoother aliasing) than rasterization, but it's much slower.
    /// One limitation is that it requires the input slices to be centered.
    template<noa::fft::Remap REMAP,
             typename Index, typename Coord,
             typename ScaleOrEmpty, typename Rotate,
             typename EWSOrEmpty,
             typename InputSliceInterpolator,
             typename InputWeightInterpolatorOrEmpty,
             typename OutputVolumeAccessor,
             typename OutputWeightAccessorOrEmpty>
    class FourierInsertInterpolateOp {
        using concept_type = FourierInsertInterpolateConcept<
                REMAP, Index, Coord, ScaleOrEmpty, Rotate, EWSOrEmpty,
                InputSliceInterpolator, InputWeightInterpolatorOrEmpty,
                OutputVolumeAccessor, OutputWeightAccessorOrEmpty>;

        static_assert(concept_type::IS_VALID_REMAP);
        static_assert(concept_type::IS_VALID_COORD_INDEX);
        static_assert(concept_type::IS_VALID_INPUT_SLICE);
        static_assert(concept_type::IS_VALID_INPUT_WEIGHT);
        static_assert(concept_type::IS_VALID_OUTPUT_VOLUME);
        static_assert(concept_type::IS_VALID_OUTPUT_WEIGHT);
        static_assert(concept_type::HAS_INPUT_WEIGHT == concept_type::HAS_OUTPUT_WEIGHT);
        static_assert(concept_type::IS_VALID_SCALE);
        static_assert(concept_type::IS_VALID_INPUT_ROTATION);
        static_assert(concept_type::IS_VALID_EWS);

        using index_type = Index;
        using coord_type = Coord;
        using scale_or_empty_type = ScaleOrEmpty;
        using rotate_type = Rotate;
        using ews_or_empty_type = EWSOrEmpty;
        using input_slice_interpolator_type = InputSliceInterpolator;
        using output_volume_accessor_type = OutputVolumeAccessor;
        using input_weight_interpolator_or_empty_type = InputWeightInterpolatorOrEmpty;
        using output_weight_accessor_or_empty_type = OutputWeightAccessorOrEmpty;

        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;
        using input_value_type = typename concept_type::input_value_type;
        using output_value_type = typename concept_type::output_value_type;
        using input_weight_type = typename concept_type::input_weight_type;
        using output_weight_type = typename concept_type::output_weight_type;
        using input_value_real_type = nt::value_type_t<input_value_type>;
        using input_weight_real_type = nt::value_type_t<input_weight_type>;

        static constexpr bool IS_OUTPUT_VOLUME_CENTERED = concept_type::IS_OUTPUT_VOLUME_CENTERED;
        static constexpr bool HAS_WEIGHTS = concept_type::HAS_WEIGHTS;

    public:
        FourierInsertInterpolateOp(
                const input_slice_interpolator_type& input_slices,
                const input_weight_interpolator_or_empty_type& input_weights,
                const Shape4<index_type>& input_slice_shape,
                const output_volume_accessor_type& output_volume,
                const output_weight_accessor_or_empty_type& output_weights,
                const Shape4<index_type>& output_volume_shape,
                const scale_or_empty_type& fwd_scaling,
                const rotate_type& inv_rotation,
                coord_type fftfreq_sinc,
                coord_type fftfreq_blackman,
                coord_type fftfreq_cutoff,
                const Shape4<index_type>& target_shape,
                const ews_or_empty_type& ews_radius
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
            if constexpr (not std::is_empty_v<ews_or_empty_type>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_type{};

            m_fftfreq_cutoff_sqd = max(fftfreq_cutoff, coord_type{0});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;

            // Clamp the windowed-sinc to ensure it's at least one pixel thick.
            const auto max_output_size = static_cast<coord_type>(min(l_target_shape));
            m_fftfreq_sinc = max(fftfreq_sinc, 1 / max_output_size);
            m_fftfreq_blackman = max(fftfreq_blackman, 1 / max_output_size);
        }

        // For every voxel of the grid.
        NOA_HD void operator()(index_type z, index_type y, index_type u) const noexcept { // x == u
            const index_type w = noa::fft::index2frequency<IS_OUTPUT_VOLUME_CENTERED>(z, m_grid_shape[0]);
            const index_type v = noa::fft::index2frequency<IS_OUTPUT_VOLUME_CENTERED>(y, m_grid_shape[1]);
            const auto orig_freq = coord3_type::from_values(w, v, u) / m_f_target_shape;
            if (dot(orig_freq, orig_freq) > m_fftfreq_cutoff_sqd)
                return;

            input_value_type value{0};
            input_weight_type weights{};
            for (index_type i = 0; i < m_slice_count; ++i) {
                const auto [freq_z, freq_2d] = details::transform_grid2slice(
                        orig_freq, m_fwd_scaling, m_inv_rotation, i, m_ews_diam_inv);

                // If voxel is not affected by the slice, skip.
                input_value_type i_value{0};
                input_weight_type i_weights{};
                if (abs(freq_z) <= m_fftfreq_blackman) {
                    const auto window = details::windowed_sinc(
                            freq_z, m_fftfreq_sinc, m_fftfreq_blackman);

                    i_value = details::interpolate_slice_value(
                            freq_2d, m_f_slice_shape, m_f_slice_y_center, m_input_slices, i);
                    i_value *= static_cast<input_value_real_type>(window);

                    if constexpr (HAS_WEIGHTS) {
                        i_weights = details::interpolate_slice_value(
                                freq_2d, m_f_slice_shape, m_f_slice_y_center, m_input_weights, i);
                        i_weights *= static_cast<input_weight_real_type>(window);
                    }
                }
                value += i_value;
                if constexpr (HAS_WEIGHTS)
                    weights += i_weights;
            }

            // The transformation preserves the hermitian symmetry, so there's nothing else to do.
            m_output_volume(z, y, u) += details::cast_or_power_spectrum<output_value_type>(value);
            if constexpr (HAS_WEIGHTS)
                m_output_weights(z, y, u) += details::cast_or_power_spectrum<output_weight_type>(weights);
        }

    private:
        input_slice_interpolator_type m_input_slices;
        output_volume_accessor_type m_output_volume;

        rotate_type m_inv_rotation;
        shape3_type m_grid_shape;
        index_type m_slice_count;
        coord3_type m_f_target_shape;
        coord2_type m_f_slice_shape;
        coord_type m_f_slice_y_center;
        coord_type m_fftfreq_cutoff_sqd;
        coord_type m_fftfreq_sinc;
        coord_type m_fftfreq_blackman;

        NOA_NO_UNIQUE_ADDRESS input_weight_interpolator_or_empty_type m_input_weights;
        NOA_NO_UNIQUE_ADDRESS output_weight_accessor_or_empty_type m_output_weights;
        NOA_NO_UNIQUE_ADDRESS scale_or_empty_type m_fwd_scaling;
        NOA_NO_UNIQUE_ADDRESS ews_or_empty_type m_ews_diam_inv{};
    };
}
