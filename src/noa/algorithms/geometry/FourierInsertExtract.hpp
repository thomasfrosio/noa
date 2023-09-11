#pragma once

#include "noa/algorithms/geometry/FourierUtilities.hpp"
#include "noa/core/traits/Interpolator.hpp"

namespace noa::geometry {
    /// Type parser/checker for FourierInsertExtractOp.
    /// \tparam REMAP                           Remap operator. HC2H or HC2HC.
    /// \tparam Index                           Input index type. A signed integer.
    /// \tparam Coord                           Coordinate type. f32 or f64.
    /// \tparam InputScaleOrEmpty               Empty, Mat22 or an 1d-accessor|pointer of Mat22.
    /// \tparam InputRotate                     Empty, Mat33|Quaternion or an 1d-accessor|pointer of Mat33|Quaternion.
    /// \tparam OutputScaleOrEmpty              Empty, Mat22 or an 1d-accessor|pointer of Mat22.
    /// \tparam OutputRotate                    Empty, Mat33|Quaternion or an 1d-accessor|pointer of Mat33|Quaternion.
    /// \tparam EWSOrEmpty                      Empty, Coord or Vec2<Coord>.
    /// \tparam InputSliceInterpolator          A 2d interpolator (BHW) of any real or complex type.
    /// \tparam InputWeightInterpolatorOrEmpty  Empty, or a 2d interpolator (BHW) of any real or complex type.
    /// \tparam OutputSliceAccessor             A 3d accessor (BHW) of any real or complex type.
    /// \tparam OutputWeightAccessorOrEmpty     Empty, or a 3d accessor (BHW) of any real or complex type.
    template<noa::fft::Remap REMAP,
             typename Index, typename Coord,
             typename InputScaleOrEmpty, typename InputRotate,
             typename OutputScaleOrEmpty, typename OutputRotate,
             typename EWSOrEmpty,
             typename InputSliceInterpolator,
             typename InputWeightInterpolatorOrEmpty,
             typename OutputSliceAccessor,
             typename OutputWeightAccessorOrEmpty>
    class FourierInsertExtractConcept {
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

        using input_value_type = nt::mutable_value_type_t<InputSliceInterpolator>;
        using output_value_type = nt::value_type_t<OutputSliceAccessor>;
        using input_weight_type = nt::mutable_value_type_t<InputWeightInterpolatorOrEmpty>;
        using output_weight_type = nt::value_type_t<OutputWeightAccessorOrEmpty>;

        static constexpr bool IS_VALID_INPUT_SLICE =
                nt::is_interpolator_2d_v<InputSliceInterpolator> &&
                nt::is_real_or_complex_v<input_value_type>;

        static constexpr bool IS_VALID_INPUT_WEIGHT =
                !HAS_WEIGHTS ||
                (nt::is_interpolator_2d_v<InputWeightInterpolatorOrEmpty> &&
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

        static constexpr bool IS_VALID_INPUT_SCALE =
                std::is_empty_v<InputScaleOrEmpty> || nt::is_mat22_v<InputScaleOrEmpty> ||
                ((nt::is_accessor_1d_v<InputScaleOrEmpty> || std::is_pointer_v<InputScaleOrEmpty>) &&
                 nt::is_mat22_v<std::remove_pointer_t<nt::value_type_t<InputScaleOrEmpty>>>);

        static constexpr bool IS_VALID_OUTPUT_SCALE =
                std::is_empty_v<OutputScaleOrEmpty> || nt::is_mat22_v<OutputScaleOrEmpty> ||
                ((nt::is_accessor_1d_v<OutputScaleOrEmpty> || std::is_pointer_v<OutputScaleOrEmpty>) &&
                 nt::is_mat22_v<std::remove_pointer_t<nt::value_type_t<OutputScaleOrEmpty>>>);

        // For CUDA, we need to reinterpret a __constant__ array to the input rotation type,
        // so generate the correct type here.
        static constexpr bool IS_INPUT_ROTATION_A_VALUE_ =
                nt::is_mat33_v<InputRotate> || nt::is_quaternion_v<InputRotate>;
        using input_rotate_array_type = std::conditional_t<
                IS_INPUT_ROTATION_A_VALUE_,
                const nt::remove_ref_cv_t<InputRotate>*,
                const std::remove_pointer_t<nt::value_type_t<InputRotate>>*>;

        static constexpr bool IS_VALID_INPUT_ROTATION =
                IS_INPUT_ROTATION_A_VALUE_ ||
                ((nt::is_accessor_1d_v<InputRotate> || std::is_pointer_v<InputRotate>) &&
                 (nt::is_mat33_v<std::remove_pointer_t<nt::value_type_t<InputRotate>>> ||
                  nt::is_quaternion_v<std::remove_pointer_t<nt::value_type_t<InputRotate>>>));

        static constexpr bool IS_VALID_OUTPUT_ROTATION =
                (nt::is_mat33_v<OutputRotate> || nt::is_quaternion_v<OutputRotate>) ||
                ((nt::is_accessor_1d_v<OutputRotate> || std::is_pointer_v<OutputRotate>) &&
                 (nt::is_mat33_v<std::remove_pointer_t<nt::value_type_t<OutputRotate>>> ||
                  nt::is_quaternion_v<std::remove_pointer_t<nt::value_type_t<OutputRotate>>>));

        static constexpr bool IS_VALID_EWS =
                std::is_empty_v<EWSOrEmpty> ||
                nt::is_any_v<EWSOrEmpty, Coord, Vec2<Coord>>;
    };

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
             typename Index, typename Coord,
             typename InputScaleOrEmpty, typename InputRotate,
             typename OutputScaleOrEmpty, typename OutputRotate,
             typename EWSOrEmpty,
             typename InputSliceInterpolator,
             typename InputWeightInterpolatorOrEmpty,
             typename OutputSliceAccessor,
             typename OutputWeightAccessorOrEmpty>
    class FourierInsertExtractOp {
        // Type parse and checks.
        using concept_type = FourierInsertExtractConcept<
                REMAP, Index, Coord,
                InputScaleOrEmpty, InputRotate, OutputScaleOrEmpty, OutputRotate, EWSOrEmpty,
                InputSliceInterpolator, InputWeightInterpolatorOrEmpty,
                OutputSliceAccessor, OutputWeightAccessorOrEmpty>;

        static_assert(concept_type::IS_VALID_REMAP);
        static_assert(concept_type::IS_VALID_COORD_INDEX);
        static_assert(concept_type::IS_VALID_INPUT_SLICE);
        static_assert(concept_type::IS_VALID_OUTPUT_SLICE);
        static_assert(concept_type::HAS_INPUT_WEIGHT == concept_type::HAS_OUTPUT_WEIGHT);
        static_assert(concept_type::IS_VALID_INPUT_WEIGHT);
        static_assert(concept_type::IS_VALID_OUTPUT_WEIGHT);
        static_assert(concept_type::IS_VALID_INPUT_SCALE);
        static_assert(concept_type::IS_VALID_OUTPUT_SCALE);
        static_assert(concept_type::IS_VALID_INPUT_ROTATION);
        static_assert(concept_type::IS_VALID_OUTPUT_ROTATION);
        static_assert(concept_type::IS_VALID_EWS);

        using index_type = Index;
        using coord_type = Coord;
        using input_scale_or_empty_type = InputScaleOrEmpty;
        using input_rotate_type = InputRotate;
        using output_scale_or_empty_type = OutputScaleOrEmpty;
        using output_rotate_type = OutputRotate;
        using ews_or_empty_type = EWSOrEmpty;
        using input_slice_interpolator_type = InputSliceInterpolator;
        using output_slice_accessor_type = OutputSliceAccessor;
        using input_weight_interpolator_or_empty_type = InputWeightInterpolatorOrEmpty;
        using output_weight_accessor_or_empty_type = OutputWeightAccessorOrEmpty;

        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;
        using input_value_type = typename concept_type::input_value_type;
        using output_value_type = typename concept_type::output_value_type;
        using input_weight_type = typename concept_type::input_weight_type;
        using output_weight_type = typename concept_type::output_weight_type;
        using input_value_real_type = nt::value_type_t<input_value_type>;
        using input_weight_real_type = nt::value_type_t<input_weight_type>;
        using output_value_real_type = nt::value_type_t<output_value_type>;
        using output_weight_real_type = nt::value_type_t<output_weight_type>;

        static constexpr bool IS_OUTPUT_SLICE_CENTERED = concept_type::IS_OUTPUT_SLICE_CENTERED;
        static constexpr bool HAS_WEIGHTS = concept_type::HAS_WEIGHTS;

    public:
        FourierInsertExtractOp(
                const input_slice_interpolator_type& input_slices,
                const input_weight_interpolator_or_empty_type& input_weights,
                const Shape4<index_type>& input_shape,
                const output_slice_accessor_type& output_slices,
                const output_weight_accessor_or_empty_type& output_weights,
                const Shape4<index_type>& output_shape,
                const input_scale_or_empty_type& insert_fwd_scaling,
                const input_rotate_type& insert_inv_rotation,
                const output_scale_or_empty_type& extract_inv_scaling,
                const output_rotate_type& extract_fwd_rotation,
                coord_type insert_fftfreq_sinc,
                coord_type insert_fftfreq_blackman,
                coord_type extract_fftfreq_sinc,
                coord_type extract_fftfreq_blackman,
                coord_type fftfreq_cutoff,
                bool add_to_output, bool correct_multiplicity,
                const ews_or_empty_type& ews_radius
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

            m_f_input_shape = coord2_type(l_input_shape.vec());
            m_f_output_shape = coord2_type(l_output_shape.vec());
            m_output_slice_size_y = l_output_shape[0];
            m_f_input_center_y = static_cast<coord_type>(l_input_shape[0] / 2); // slice Y center

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_type>)
                m_ews_diam_inv = noa::any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_type{};

            m_fftfreq_cutoff_sqd = noa::math::max(fftfreq_cutoff, coord_type{0});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;

            // Of course, we have no z here, but the smallest axis is a good fallback.
            m_volume_z = static_cast<coord_type>(noa::math::min(l_output_shape));
            m_insert_fftfreq_sinc = noa::math::max(insert_fftfreq_sinc, 1 / m_volume_z);
            m_insert_fftfreq_blackman = noa::math::max(insert_fftfreq_blackman, 1 / m_volume_z);
            m_extract_fftfreq_sinc = noa::math::max(extract_fftfreq_sinc, 1 / m_volume_z);
            m_extract_fftfreq_blackman = noa::math::max(extract_fftfreq_blackman, 1 / m_volume_z);
            std::tie(m_extract_blackman_size, m_extract_window_total_weight) = details::z_window_spec<index_type>(
                    m_extract_fftfreq_sinc, m_extract_fftfreq_blackman, m_volume_z);
        }

        /// Whether the operator is 4d. Otherwise it is 3d.
        [[nodiscard]] constexpr auto is_iwise_4d() const noexcept -> bool {
            return m_extract_blackman_size > 1;
        }

        // Returns the size of the output (depth) window.
        [[nodiscard]] constexpr auto output_window_size() const noexcept -> index_type {
            return m_extract_blackman_size;
        }

        /// Should be called for every pixel of every slice to extract.
        NOA_HD void operator()(index_type batch, index_type y, index_type u) const noexcept {
            const coord3_type fftfreq_3d = compute_fftfreq_in_volume_(batch, y, u);

            if (noa::math::dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd) {
                if (!m_add_to_output)
                    m_output_slices(batch, y, u) = output_value_type{0};
                return;
            }

            const auto value_and_weight = sample_virtual_volume_(fftfreq_3d);
            if (m_add_to_output) {
                m_output_slices(batch, y, u) += value_and_weight.first;
                if constexpr (HAS_WEIGHTS)
                    m_output_weights(batch, y, u) += value_and_weight.second;
            } else {
                m_output_slices(batch, y, u) = value_and_weight.first;
                if constexpr (HAS_WEIGHTS)
                    m_output_weights(batch, y, u) = value_and_weight.second;
            }
        }

        /// Should be called for every pixel of every slice to extract and for every element in the z-windowed-sinc.
        /// Of course, this can be much more expensive to extract a slice. Also, the output-slice is not set, so
        /// the caller may have to fill it with zeros first, depending on add_to_output.
        NOA_HD void operator()(index_type batch, index_type w, index_type y, index_type u) const noexcept {
            coord3_type fftfreq_3d = compute_fftfreq_in_volume_(batch, y, u);

            // Get and add the volume z-offset for the z-windowed-sinc.
            const auto fftfreq_z_offset = details::w_index_to_fftfreq_offset(
                    w, m_extract_blackman_size, m_volume_z);
            fftfreq_3d[0] += fftfreq_z_offset;

            if (noa::math::dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd)
                return;

            const auto value_and_weight = sample_virtual_volume_(fftfreq_3d);

            // Z-window.
            const auto convolution_weight =
                    details::windowed_sinc(fftfreq_z_offset, m_extract_fftfreq_sinc, m_extract_fftfreq_blackman) /
                    m_extract_window_total_weight; // convolution sum is 1

            // Add the contribution for this z-offset. The z-convolution is essentially a simple weighted mean.
            noa::details::atomic_add(
                    m_output_slices,
                    value_and_weight.first * static_cast<output_value_real_type>(convolution_weight),
                    batch, y, u);
            if constexpr (HAS_WEIGHTS) {
                noa::details::atomic_add(
                        m_output_weights,
                        value_and_weight.second * static_cast<output_weight_real_type>(convolution_weight),
                        batch, y, u);
            }
        }

    private:
        // The indexes give us the fftfreq in the coordinate system of the slice to extract.
        // This function transforms this fftfreq to the coordinate system of the virtual volume.
        NOA_HD coord3_type compute_fftfreq_in_volume_(
                index_type batch, index_type y, index_type u
        ) const noexcept {
            const index_type v = noa::fft::index2frequency<IS_OUTPUT_SLICE_CENTERED>(y, m_output_slice_size_y);
            auto fftfreq_2d = coord2_type{v, u} / m_f_output_shape;
            return details::transform_slice2grid(
                    fftfreq_2d, m_extract_inv_scaling, m_extract_fwd_rotation,
                    batch, m_ews_diam_inv);
        }

        // Sample the signal at this 3d frequency within the virtual volume.
        // We do this for both the slice values and weights (if any).
        NOA_HD auto sample_virtual_volume_(const coord3_type& fftfreq_3d) const noexcept {
            // Select the highest precision between the input values and weights.
            using multiplicity_t = std::conditional_t<
                    (sizeof(input_value_real_type) > sizeof(input_weight_real_type)),
                    input_value_real_type, input_weight_real_type>;

            output_value_type value{0};
            output_weight_type weight{};
            multiplicity_t multiplicity{0};

//            #if defined(__CUDA_ARCH__)
//            using input_rotate_array_type = typename concept_type::input_rotate_array_type;
//            const auto insert_inv_rotation = reinterpret_cast<input_rotate_array_type>(
//                    CUDA_FOURIER_INSERT_EXTRACT_CONSTANT_ARRAY_NAME);
//            #else
//            const auto& insert_inv_rotation = m_insert_inv_rotation;
//            #endif

            // For every slice to insert...
            for (index_type i = 0; i < m_input_count; ++i) {
                // 1. Project the 3d frequency onto that input-slice.
                //    fftfreq_z is along the normal of that input-slice.
                const auto [fftfreq_z, fftfreq_yx] = details::transform_grid2slice(
                        fftfreq_3d, m_insert_fwd_scaling, m_insert_inv_rotation, i, m_ews_diam_inv);

                // 2. Sample the input-slice (and input-weight) at fftfreq_yx and update the multiplicity
                //    using a windowed-sinc at fftfreq_z.
                input_value_type i_value{0};
                input_weight_type i_weight{};
                multiplicity_t i_multiplicity{0};

                // Compute only if voxel is affected by this slice.
                // If we fall exactly at the blackman cutoff, the value is 0, so exclude the equality case too.
                if (noa::math::abs(fftfreq_z) < m_insert_fftfreq_blackman) {
                    i_multiplicity = static_cast<multiplicity_t>(details::windowed_sinc(
                            fftfreq_z, m_insert_fftfreq_sinc, m_insert_fftfreq_blackman));

                    i_value = details::interpolate_slice_value(
                            fftfreq_yx, m_f_input_shape, m_f_input_center_y, m_input_slices, i);
                    i_value *= i_multiplicity;

                    if constexpr (HAS_WEIGHTS) {
                        i_weight = details::interpolate_slice_value(
                                fftfreq_yx, m_f_input_shape, m_f_input_center_y, m_input_weights, i);
                        i_weight *= i_multiplicity;
                    }
                }
                multiplicity += i_multiplicity;
                value += i_value;
                if constexpr (HAS_WEIGHTS)
                    weight += i_weight;
            }

            // 3. Correct for the multiplicity.
            // If the weight is greater than one, it means multiple slices contributed to that frequency, so divide
            // by the total weight (resulting in a weighted mean). If the total weight is less than 1, we need to
            // leave it down-weighted, so don't divide by the total weight.
            if (m_correct_multiplicity) {
                multiplicity = noa::math::max(multiplicity_t{1}, multiplicity);
                value /= static_cast<input_value_real_type>(multiplicity);
                if constexpr (HAS_WEIGHTS)
                    weight /= static_cast<input_weight_real_type>(multiplicity);
            }

            return Pair{details::cast_or_power_spectrum<output_value_type>(value),
                        details::cast_or_power_spectrum<output_weight_type>(weight)};
        }

    private:
        input_slice_interpolator_type m_input_slices;
        output_slice_accessor_type m_output_slices;

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

        NOA_NO_UNIQUE_ADDRESS input_weight_interpolator_or_empty_type m_input_weights;
        NOA_NO_UNIQUE_ADDRESS output_weight_accessor_or_empty_type m_output_weights;
        NOA_NO_UNIQUE_ADDRESS input_scale_or_empty_type m_insert_fwd_scaling;
        NOA_NO_UNIQUE_ADDRESS output_scale_or_empty_type m_extract_inv_scaling;
        NOA_NO_UNIQUE_ADDRESS ews_or_empty_type m_ews_diam_inv{};
    };

    /// Factory function for FourierInsertExtractOp.
    template<noa::fft::Remap REMAP,
             typename Index, typename Coord,
             typename InputScaleOrEmpty, typename InputRotate,
             typename OutputScaleOrEmpty, typename OutputRotate,
             typename EWSOrEmpty,
             typename InputSliceInterpolator,
             typename InputWeightInterpolatorOrEmpty,
             typename OutputSliceAccessor,
             typename OutputWeightAccessorOrEmpty>
    auto fourier_insert_and_extract_op(
            const InputSliceInterpolator& input_slices,
            const InputWeightInterpolatorOrEmpty& input_weights,
            const Shape4<Index>& input_shape,
            const OutputSliceAccessor& output_slices,
            const OutputWeightAccessorOrEmpty& output_weights,
            const Shape4<Index>& output_shape,
            const InputScaleOrEmpty& insert_fwd_scaling,
            const InputRotate& insert_inv_rotation,
            const OutputScaleOrEmpty& extract_inv_scaling,
            const OutputRotate& extract_fwd_rotation,
            Coord insert_fftfreq_sinc,
            Coord insert_fftfreq_blackman,
            Coord extract_fftfreq_sinc,
            Coord extract_fftfreq_blackman,
            Coord fftfreq_cutoff,
            bool add_to_output, bool correct_multiplicity,
            EWSOrEmpty ews_radius
    ) {
        using op_t = FourierInsertExtractOp<
                REMAP, Index, Coord, InputScaleOrEmpty, InputRotate, OutputScaleOrEmpty, OutputRotate, EWSOrEmpty,
                InputSliceInterpolator, InputWeightInterpolatorOrEmpty,
                OutputSliceAccessor, OutputWeightAccessorOrEmpty>;
        return op_t(
                input_slices, input_weights, input_shape,
                output_slices, output_weights, output_shape,
                insert_fwd_scaling, insert_inv_rotation,
                extract_inv_scaling, extract_fwd_rotation,
                insert_fftfreq_sinc, insert_fftfreq_blackman,
                extract_fftfreq_sinc, extract_fftfreq_blackman,
                fftfreq_cutoff, add_to_output, correct_multiplicity, ews_radius
        );
    }
}
