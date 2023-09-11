#pragma once

#include "noa/algorithms/geometry/FourierUtilities.hpp"
#include "noa/core/traits/Interpolator.hpp"

namespace noa::geometry {
    /// Type parser/checker for FourierInsertInterpolateOp.
    /// \tparam REMAP                               Remap operator. H(C)2H(C).
    /// \tparam Index                               Input index type. A signed integer.
    /// \tparam Coord                               Coordinate type. f32 or f64.
    /// \tparam ScaleOrEmpty                        Empty, Mat22 or an accessor of Mat22.
    /// \tparam Rotate                              Empty, Mat33|Quaternion or an accessor of Mat33|Quaternion.
    /// \tparam EWSOrEmpty                          Empty, Coord or Vec2<Coord>.
    /// \tparam InputVolumeInterpolator             A 3d interpolator (DHW) of any real or complex type.
    /// \tparam InputWeightInterpolatorOrEmpty      Empty, or a 3d interpolator (DHW) of any real or complex type.
    /// \tparam OutputSliceAccessor                 A 3d accessor (BHW) of any real or complex type.
    /// \tparam OutputWeightAccessorOrEmpty         Empty, or a 3d accessor (BHW) of any real or complex type.
    template<noa::fft::Remap REMAP,
            typename Index, typename Coord,
            typename ScaleOrEmpty, typename Rotate,
            typename EWSOrEmpty,
            typename InputVolumeInterpolator,
            typename InputWeightInterpolatorOrEmpty,
            typename OutputSliceAccessor,
            typename OutputWeightAccessorOrEmpty>
    class FourierExtractConcept {
    public:
        static constexpr bool IS_VALID_REMAP =
                REMAP == noa::fft::Remap::HC2H ||
                REMAP == noa::fft::Remap::HC2HC;

        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_OUTPUT_SLICE_CENTERED = u8_REMAP & noa::fft::Layout::DST_CENTERED;

        static constexpr bool IS_VALID_COORD_INDEX =
                nt::is_sint_v<Index> && nt::is_any_v<Coord, f32, f64>;

        static constexpr bool HAS_INPUT_WEIGHT = !std::is_empty_v<InputWeightInterpolatorOrEmpty>;
        static constexpr bool HAS_OUTPUT_WEIGHT = !std::is_empty_v<OutputWeightAccessorOrEmpty>;
        static constexpr bool HAS_WEIGHTS = HAS_INPUT_WEIGHT && HAS_OUTPUT_WEIGHT;

        using input_value_type = nt::mutable_value_type_t<InputVolumeInterpolator>;
        using output_value_type = nt::value_type_t<OutputSliceAccessor>;
        using input_weight_type = nt::mutable_value_type_t<InputWeightInterpolatorOrEmpty>;
        using output_weight_type = nt::value_type_t<OutputWeightAccessorOrEmpty>;

        static constexpr bool IS_VALID_INPUT_VOLUME =
                nt::is_interpolator_3d_v<InputVolumeInterpolator> &&
                nt::is_real_or_complex_v<input_value_type>;

        static constexpr bool IS_VALID_INPUT_WEIGHT =
                !HAS_WEIGHTS ||
                (nt::is_interpolator_3d_v<InputWeightInterpolatorOrEmpty> &&
                 nt::is_real_or_complex_v<input_weight_type>);

        static constexpr bool IS_VALID_OUTPUT_SLICE =
                nt::is_accessor_nd_v<OutputSliceAccessor, 3> &&
                !std::is_const_v<output_value_type> &&
                (nt::are_complex_v<output_value_type, input_value_type> ||
                 (nt::is_real_v<output_value_type> && nt::is_real_or_complex_v<input_value_type>));

        static constexpr bool IS_VALID_OUTPUT_WEIGHT =
                !HAS_WEIGHTS ||
                (nt::is_accessor_nd_v<OutputWeightAccessorOrEmpty, 3> &&
                 !std::is_const_v<output_weight_type> &&
                 (nt::are_complex_v<output_weight_type, input_weight_type> ||
                  (nt::is_real_v<output_weight_type> && nt::is_real_or_complex_v<input_weight_type>)));

        static constexpr bool IS_VALID_SCALE =
                std::is_empty_v<ScaleOrEmpty> || nt::is_mat22_v<ScaleOrEmpty> ||
                ((nt::is_accessor_1d_v<ScaleOrEmpty> || std::is_pointer_v<ScaleOrEmpty>) &&
                  nt::is_mat22_v<std::remove_pointer_t<nt::value_type_t<ScaleOrEmpty>>>);

        static constexpr bool IS_VALID_INPUT_ROTATION =
                nt::is_mat33_v<Rotate> || nt::is_quaternion_v<Rotate> ||
                ((nt::is_accessor_1d_v<Rotate> || std::is_pointer_v<Rotate>) &&
                 (nt::is_mat33_v<std::remove_pointer_t<nt::value_type_t<Rotate>>> ||
                  nt::is_quaternion_v<std::remove_pointer_t<nt::value_type_t<Rotate>>>));

        static constexpr bool IS_VALID_EWS =
                std::is_empty_v<EWSOrEmpty> ||
                nt::is_any_v<EWSOrEmpty, Coord, Vec2<Coord>>;
    };

    // The exact same transformation as insertion with gridding is applied here,
    // but instead of inserting the transformed slices into the grid,
    // the transformed slices are extracted from the grid.
    template<noa::fft::Remap REMAP,
             typename Index, typename Coord,
             typename ScaleOrEmpty, typename Rotate,
             typename EWSOrEmpty,
             typename InputVolumeInterpolator,
             typename InputWeightInterpolatorOrEmpty,
             typename OutputSliceAccessor,
             typename OutputWeightAccessorOrEmpty>
    class FourierExtractOp {
        using concept_type = FourierExtractConcept<
                REMAP, Index, Coord, ScaleOrEmpty, Rotate, EWSOrEmpty,
                InputVolumeInterpolator, InputWeightInterpolatorOrEmpty,
                OutputSliceAccessor, OutputWeightAccessorOrEmpty>;

        static_assert(concept_type::IS_VALID_REMAP);
        static_assert(concept_type::IS_VALID_COORD_INDEX);
        static_assert(concept_type::IS_VALID_INPUT_VOLUME);
        static_assert(concept_type::IS_VALID_INPUT_WEIGHT);
        static_assert(concept_type::IS_VALID_OUTPUT_SLICE);
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
        using input_volume_interpolator_type = InputVolumeInterpolator;
        using output_slice_accessor_type = OutputSliceAccessor;
        using input_weight_interpolator_or_empty_type = InputWeightInterpolatorOrEmpty;
        using output_weight_accessor_or_empty_type = OutputWeightAccessorOrEmpty;

        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;
        using input_value_type = typename concept_type::input_value_type;
        using output_value_type = typename concept_type::output_value_type;
        using input_weight_type = typename concept_type::input_weight_type;
        using output_weight_type = typename concept_type::output_weight_type;
        using output_value_real_type = nt::value_type_t<output_value_type>;
        using output_weight_real_type = nt::value_type_t<output_weight_type>;

        static constexpr bool IS_OUTPUT_SLICE_CENTERED = concept_type::IS_OUTPUT_SLICE_CENTERED;
        static constexpr bool HAS_WEIGHTS = concept_type::HAS_WEIGHTS;

    public:
        FourierExtractOp(
                const input_volume_interpolator_type& input_volume,
                const input_weight_interpolator_or_empty_type& input_weights,
                const Shape4<index_type>& input_volume_shape,
                const output_slice_accessor_type& output_slices,
                const output_weight_accessor_or_empty_type& output_weights,
                const Shape4<index_type>& output_slice_shape,
                const scale_or_empty_type& inv_scaling,
                const rotate_type& fwd_rotation,
                coord_type fftfreq_sinc,
                coord_type fftfreq_blackman,
                coord_type fftfreq_cutoff,
                const Shape4<index_type>& target_shape,
                const ews_or_empty_type& ews_radius
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
            m_f_slice_shape = coord2_type(slice_shape_2d.vec());

            // Use the grid shape as backup.
            const auto grid_shape_3d = input_volume_shape.pop_front();
            const auto target_shape_3d = noa::any(target_shape == 0) ? grid_shape_3d : target_shape.pop_front();
            m_f_target_shape = coord3_type(target_shape_3d.vec());
            m_f_grid_zy_center = coord2_type((grid_shape_3d.filter(0, 1) / 2).vec()); // grid ZY center

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_type>)
                m_ews_diam_inv = noa::any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_type{};

            m_fftfreq_cutoff_sqd = noa::math::max(fftfreq_cutoff, coord_type{0});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;

            // This is along the z of the grid.
            m_fftfreq_sinc = noa::math::max(fftfreq_sinc, 1 / m_f_target_shape[0]);
            m_fftfreq_blackman = noa::math::max(fftfreq_blackman, 1 / m_f_target_shape[0]);
            std::tie(m_blackman_size, m_z_window_sum) = details::z_window_spec<index_type>(
                    m_fftfreq_sinc, m_fftfreq_blackman, m_f_target_shape[0]);
        }

        [[nodiscard]] constexpr index_type windowed_sinc_size() const noexcept { return m_blackman_size; }

        // For every pixel of every slice to extract.
        NOA_HD void operator()(index_type batch, index_type y, index_type u) const noexcept {
            const coord3_type freq_3d = compute_fftfreq_in_volume_(batch, y, u);

            input_value_type value{0};
            input_weight_type weight{};
            if (noa::math::dot(freq_3d, freq_3d) <= m_fftfreq_cutoff_sqd) {
                value = details::interpolate_grid_value(
                        freq_3d, m_f_target_shape, m_f_grid_zy_center, m_input_volume);
                if constexpr (HAS_WEIGHTS) {
                    weight = details::interpolate_grid_value(
                            freq_3d, m_f_target_shape, m_f_grid_zy_center, m_input_weights);
                }
            }

            m_output_slices(batch, y, u) = details::cast_or_power_spectrum<output_value_type>(value);
            if constexpr (HAS_WEIGHTS)
                m_output_weights(batch, y, u) = details::cast_or_power_spectrum<output_weight_type>(weight);
        }

        // For every pixel of every slice to extract.
        // w is the index within the windowed-sinc convolution along the z of the grid.
        NOA_HD void operator()(index_type batch, index_type w, index_type y, index_type u) const noexcept {
            coord3_type freq_3d = compute_fftfreq_in_volume_(batch, y, u);

            // Additional z component, within the grid coordinate system.
            const auto fftfreq_z_offset = details::w_index_to_fftfreq_offset(w, m_blackman_size, m_f_target_shape[0]);
            freq_3d[0] += fftfreq_z_offset;

            if (noa::math::dot(freq_3d, freq_3d) > m_fftfreq_cutoff_sqd)
                return;

            // Z-window.
            const auto convolution_weight =
                    details::windowed_sinc(fftfreq_z_offset, m_fftfreq_sinc, m_fftfreq_blackman) /
                    m_z_window_sum; // convolution sum is 1

            const auto value = details::interpolate_grid_value(
                    freq_3d, m_f_target_shape, m_f_grid_zy_center, m_input_volume);
            noa::details::atomic_add(
                    m_output_slices,
                    details::cast_or_power_spectrum<output_value_type>(value) *
                    static_cast<output_value_real_type>(convolution_weight),
                    batch, y, u);

            if constexpr (HAS_WEIGHTS) {
                const auto weight = details::interpolate_grid_value(
                        freq_3d, m_f_target_shape, m_f_grid_zy_center, m_input_weights);
                noa::details::atomic_add(
                        m_output_weights,
                        details::cast_or_power_spectrum<output_weight_type>(weight) *
                        static_cast<output_weight_real_type>(convolution_weight),
                        batch, y, u);
            }
        }

    private:
        // The indexes give us the fftfreq in the coordinate system of the slice to extract.
        // This function transforms this fftfreq to the coordinate system of the volume.
        NOA_HD coord3_type compute_fftfreq_in_volume_(
                index_type batch, index_type y, index_type u
        ) const noexcept {
            const index_type v = noa::fft::index2frequency<IS_OUTPUT_SLICE_CENTERED>(y, m_slice_size_y);
            const auto fftfreq_2d = coord2_type{v, u} / m_f_slice_shape;
            return details::transform_slice2grid(
                    fftfreq_2d, m_inv_scaling, m_fwd_rotation, batch, m_ews_diam_inv);
        }

    private:
        input_volume_interpolator_type m_input_volume;
        output_slice_accessor_type m_output_slices;

        rotate_type m_fwd_rotation;
        coord3_type m_f_target_shape;
        coord2_type m_f_slice_shape;
        coord2_type m_f_grid_zy_center;
        index_type m_slice_size_y;

        coord_type m_fftfreq_cutoff_sqd;
        coord_type m_fftfreq_sinc;
        coord_type m_fftfreq_blackman;
        index_type m_blackman_size;
        coord_type m_z_window_sum;

        NOA_NO_UNIQUE_ADDRESS input_weight_interpolator_or_empty_type m_input_weights;
        NOA_NO_UNIQUE_ADDRESS output_weight_accessor_or_empty_type m_output_weights;
        NOA_NO_UNIQUE_ADDRESS scale_or_empty_type m_inv_scaling;
        NOA_NO_UNIQUE_ADDRESS ews_or_empty_type m_ews_diam_inv{};
    };

    template<noa::fft::Remap REMAP,
             typename Index, typename Coord,
             typename ScaleOrEmpty, typename Rotate,
             typename EWSOrEmpty,
             typename InputVolumeInterpolator,
             typename InputWeightInterpolatorOrEmpty,
             typename OutputSliceAccessor,
             typename OutputWeightAccessorOrEmpty>
    auto fourier_extract_op(
            const InputVolumeInterpolator& input_volume,
            const InputWeightInterpolatorOrEmpty& input_weights,
            const Shape4<Index>& input_volume_shape,
            const OutputSliceAccessor& output_slices,
            const OutputWeightAccessorOrEmpty& output_weights,
            const Shape4<Index>& output_slice_shape,
            const ScaleOrEmpty& inv_scaling,
            const Rotate& fwd_rotation,
            Coord fftfreq_z_sinc,
            Coord fftfreq_z_blackman,
            Coord fftfreq_cutoff,
            const Shape4<Index>& target_shape,
            const EWSOrEmpty& ews_radius
    ) {
        using op_t = FourierExtractOp<
                REMAP, Index, Coord, ScaleOrEmpty, Rotate, EWSOrEmpty,
                InputVolumeInterpolator, InputWeightInterpolatorOrEmpty,
                OutputSliceAccessor, OutputWeightAccessorOrEmpty>;;
        return op_t(
                input_volume, input_weights, input_volume_shape,
                output_slices, output_weights, output_slice_shape,
                inv_scaling, fwd_rotation,
                fftfreq_z_sinc, fftfreq_z_blackman, fftfreq_cutoff,
                target_shape, ews_radius);
    }
}
