#pragma once

#include "noa/core/geometry/FourierUtilities.hpp"

namespace noa::geometry {
    /// Type parser/checker for FourierInsertRasterize.
    /// \tparam REMAP                               Remap operator. H(C)2H(C).
    /// \tparam Index                               Input index type. A signed integer.
    /// \tparam Coord                               Coordinate type. f32 or f64.
    /// \tparam ScaleOrEmpty                        Empty, Mat22 or an accessor of Mat22.
    /// \tparam Rotate                              Empty, Mat33|Quaternion or an accessor of Mat33|Quaternion.
    /// \tparam EWSOrEmpty                          Empty, Coord or Vec2<Coord>.
    /// \tparam InputSliceAccessorOrValue           A value or 3d accessor (BHW) of any real or complex type.
    /// \tparam InputWeightAccessorOrValueOrEmpty   Empty, or a value or 3d accessor (BHW) of any real or complex type.
    /// \tparam OutputVolumeAccessor                A 3d accessor (DHW) of any real or complex type.
    /// \tparam OutputWeightAccessorOrEmpty         Empty, or a 3d accessor (DHW) of any real or complex type.
    template<noa::fft::Remap REMAP,
             typename Index, typename Coord,
             typename ScaleOrEmpty, typename Rotate,
             typename EWSOrEmpty,
             typename InputSliceAccessorOrValue,
             typename InputWeightAccessorOrValueOrEmpty,
             typename OutputVolumeAccessor,
             typename OutputWeightAccessorOrEmpty>
    class FourierInsertRasterizeConcept {
    public:
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SLICE_CENTERED = u8_REMAP & noa::fft::Layout::SRC_CENTERED;
        static constexpr bool IS_VOLUME_CENTERED = u8_REMAP & noa::fft::Layout::DST_CENTERED;
        static constexpr bool IS_VALID_REMAP =
                (u8_REMAP & noa::fft::Layout::SRC_HALF) and
                (u8_REMAP & noa::fft::Layout::DST_HALF);

        static constexpr bool IS_VALID_COORD_INDEX =
                nt::is_sint_v<Index> and nt::is_any_v<Coord, f32, f64>;

        static constexpr bool HAS_INPUT_WEIGHT = not std::is_empty_v<InputWeightAccessorOrValueOrEmpty>;
        static constexpr bool HAS_OUTPUT_WEIGHT = not std::is_empty_v<OutputWeightAccessorOrEmpty>;
        static constexpr bool HAS_WEIGHTS = HAS_INPUT_WEIGHT and HAS_OUTPUT_WEIGHT;

        static constexpr bool IS_INPUT_SLICE_A_VALUE = nt::is_real_or_complex_v<InputSliceAccessorOrValue>;
        static constexpr bool IS_INPUT_WEIGHT_A_VALUE = nt::is_real_or_complex_v<InputWeightAccessorOrValueOrEmpty>;

        using input_value_type = std::conditional_t<
                IS_INPUT_SLICE_A_VALUE,
                nt::remove_ref_cv_t<InputSliceAccessorOrValue>,
                nt::mutable_value_type_t<InputSliceAccessorOrValue>>;
        using input_weight_type = std::conditional_t<
                IS_INPUT_WEIGHT_A_VALUE,
                nt::remove_ref_cv_t<InputWeightAccessorOrValueOrEmpty>,
                nt::mutable_value_type_t<InputWeightAccessorOrValueOrEmpty>>;
        using output_value_type = nt::value_type_t<OutputVolumeAccessor>;
        using output_weight_type = nt::value_type_t<OutputWeightAccessorOrEmpty>;

        static constexpr bool IS_VALID_INPUT_SLICE =
                (IS_INPUT_SLICE_A_VALUE or nt::is_accessor_nd_v<InputSliceAccessorOrValue, 3>) and
                nt::is_real_or_complex_v<input_value_type>;

        static constexpr bool IS_VALID_INPUT_WEIGHT =
                not HAS_WEIGHTS or
                ((IS_INPUT_WEIGHT_A_VALUE or nt::is_accessor_nd_v<InputWeightAccessorOrValueOrEmpty, 3>) and
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
                (nt::is_mat33_v<Rotate> or nt::is_quaternion_v<Rotate>) or
                ((nt::is_accessor_nd_v<Rotate, 1> or std::is_pointer_v<Rotate>) and
                 (nt::is_mat33_v<std::remove_pointer_t<nt::value_type_t<Rotate>>> or
                  nt::is_quaternion_v<std::remove_pointer_t<nt::value_type_t<Rotate>>>));

        static constexpr bool IS_VALID_EWS =
                std::is_empty_v<EWSOrEmpty> or
                nt::is_any_v<EWSOrEmpty, Coord, Vec2<Coord>>;
    };

    /// Direct Fourier insertion, using rasterization.
    template<noa::fft::Remap REMAP,
             typename Index, typename Coord,
             typename ScaleOrEmpty, typename Rotate,
             typename EWSOrEmpty,
             typename InputSliceAccessorOrValue,
             typename InputWeightAccessorOrValueOrEmpty,
             typename OutputVolumeAccessor,
             typename OutputWeightAccessorOrEmpty>
    class FourierInsertRasterize {
        using concept_type = FourierInsertRasterizeConcept<
                REMAP, Index, Coord, ScaleOrEmpty, Rotate, EWSOrEmpty,
                InputSliceAccessorOrValue, InputWeightAccessorOrValueOrEmpty,
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
        using input_slice_accessor_or_value_type = InputSliceAccessorOrValue;
        using output_volume_accessor_type = OutputVolumeAccessor;
        using input_weight_accessor_or_value_or_empty_type = InputWeightAccessorOrValueOrEmpty;
        using output_weight_accessor_or_empty_type = OutputWeightAccessorOrEmpty;

        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;
        using input_value_type = concept_type::input_value_type;
        using output_value_type = concept_type::output_value_type;
        using input_weight_type = concept_type::input_weight_type;
        using output_weight_type = concept_type::output_weight_type;
        using input_value_real_type = nt::value_type_t<input_value_type>;
        using input_weight_real_type = nt::value_type_t<input_weight_type>;
        using output_value_real_type = nt::value_type_t<output_value_type>;
        using output_weight_real_type = nt::value_type_t<output_weight_type>;

        static constexpr bool IS_INPUT_SLICE_A_VALUE = concept_type::IS_INPUT_SLICE_A_VALUE;
        static constexpr bool IS_INPUT_WEIGHT_A_VALUE = concept_type::IS_INPUT_WEIGHT_A_VALUE;
        static constexpr bool IS_SLICE_CENTERED = concept_type::IS_SLICE_CENTERED;
        static constexpr bool IS_VOLUME_CENTERED = concept_type::IS_VOLUME_CENTERED;
        static constexpr bool HAS_WEIGHTS = concept_type::HAS_WEIGHTS;

    public:
        FourierInsertRasterize(
                const input_slice_accessor_or_value_type& input_slices,
                const input_weight_accessor_or_value_or_empty_type& input_weights,
                const Shape4<index_type>& input_slice_shape,
                const output_volume_accessor_type& output_volume,
                const output_weight_accessor_or_empty_type& output_weights,
                const Shape4<index_type>& output_volume_shape,
                const scale_or_empty_type& inv_scaling,
                const rotate_type& fwd_rotation,
                coord_type fftfreq_cutoff,
                const Shape4<index_type>& target_shape,
                const ews_or_empty_type& ews_radius
        ) : m_input_slices(input_slices),
            m_output_volume(output_volume),
            m_fwd_rotation(fwd_rotation),
            m_grid_shape(output_volume_shape.pop_front()),
            m_input_weights(input_weights),
            m_output_weights(output_weights),
            m_inv_scaling(inv_scaling)
        {
            const auto slice_shape_2d = input_slice_shape.filter(2, 3);
            m_slice_size_y = slice_shape_2d[0];
            m_f_slice_shape = coord2_type::from_vec(slice_shape_2d.vec);

            // Use the grid shape as backup.
            const auto target_shape_3d = any(target_shape == 0) ? m_grid_shape : target_shape.pop_front();
            m_f_target_shape = coord3_type::from_vec(target_shape_3d.vec);

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (not std::is_empty_v<ews_or_empty_type>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_type{};

            m_fftfreq_cutoff_sqd = max(fftfreq_cutoff, coord_type{0});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;
        }

        // For every pixel of every central slice to insert.
        NOA_HD void operator()(index_type batch, index_type y, index_type u) const noexcept { // x == u
            // We compute the forward transformation and use normalized frequencies.
            // The oversampling is implicitly handled when scaling back to the target shape.
            const index_type v = noa::fft::index2frequency<IS_SLICE_CENTERED>(y, m_slice_size_y);
            const auto frequency_2d = coord2_type::from_values(v, u) / m_f_slice_shape;
            coord3_type frequency_3d = details::transform_slice2grid(
                    frequency_2d, m_inv_scaling, m_fwd_rotation, batch, m_ews_diam_inv);

            // The frequency rate won't change from that point, so check for the cutoff.
            if (dot(frequency_3d, frequency_3d) > m_fftfreq_cutoff_sqd)
                return;

            // Handle the non-redundancy in x.
            coord_type conjugate = 1;
            if (frequency_3d[2] < 0) {
                frequency_3d = -frequency_3d;
                if constexpr (nt::is_complex_v<input_value_type> or
                              nt::is_complex_v<input_weight_type>) {
                    conjugate = -1;
                }
            }

            // Scale back to the target shape.
            frequency_3d *= m_f_target_shape;

            // At this point, we know we are going to use the input, so load everything.
            Pair<output_value_type, output_weight_type> value_and_weight{
                    get_input_value_(batch, y, u, conjugate),
                    get_input_weight_(batch, y, u, conjugate),
            };
            rasterize_on_3d_grid_(value_and_weight, frequency_3d);
        }

    private:
        NOA_HD constexpr auto get_input_value_(
                index_type batch, index_type y, index_type u, coord_type conjugate
        ) const noexcept {
            input_value_type value;
            if constexpr (IS_INPUT_SLICE_A_VALUE)
                value = m_input_slices;
            else
                value = m_input_slices(batch, y, u);

            if constexpr (nt::is_complex_v<input_value_type> and nt::is_real_v<output_value_type>) {
                return static_cast<output_value_type>(abs_squared(value));
            } else {
                if constexpr (nt::is_complex_v<input_value_type>)
                    value.imag *= static_cast<input_value_real_type>(conjugate);
                return static_cast<output_value_type>(value);
            }
        }

        NOA_HD constexpr auto get_input_weight_(
                index_type batch, index_type y, index_type u, coord_type conjugate
        ) const noexcept {
            if constexpr (HAS_WEIGHTS) {
                input_weight_type weight;
                if constexpr (IS_INPUT_WEIGHT_A_VALUE)
                    weight = m_input_weights;
                else
                    weight = m_input_weights(batch, y, u);

                if constexpr (nt::is_complex_v<input_weight_type> and nt::is_real_v<output_weight_type>) {
                    return static_cast<output_weight_type>(abs_squared(weight));
                } else {
                    if constexpr (nt::is_complex_v<input_weight_type>)
                        weight.imag *= static_cast<input_weight_real_type>(conjugate);
                    return static_cast<output_weight_type>(weight);
                }
            } else {
                return input_weight_type{};
            }
        }

        // The gridding/rasterization kernel is a trilinear pulse.
        // The total weight within the 2x2x2 cube is 1.
        NOA_HD static constexpr void set_rasterization_weights_(
                const Vec3<index_type>& base0,
                const Vec3<coord_type>& freq,
                coord_type o_weights[2][2][2]
        ) noexcept {
            // So if the coordinate is centered in the bottom left corner of the cube (base0),
            // i.e., its decimal is 0, the corresponding fraction for this element should be 1.
            Vec3<coord_type> fraction[2];
            fraction[1] = freq - base0.template as<coord_type>();
            fraction[0] = 1.f - fraction[1];
            for (index_type w = 0; w < 2; ++w)
                for (index_type v = 0; v < 2; ++v)
                    for (index_type u = 0; u < 2; ++u)
                        o_weights[w][v][u] = fraction[w][0] * fraction[v][1] * fraction[u][2];
        }

        // Spread the value within a 2x2x2 centered on a particular frequency using linear interpolation.
        // This is called gridding, but is also referred as rasterization with antialiasing.
        // "frequency" is the frequency, in samples, centered on DC, with negative frequencies on the left.
        NOA_HD void rasterize_on_3d_grid_(
                Pair<output_value_type, output_weight_type> value_and_weight,
                const Vec3<Coord>& frequency // in samples
        ) const noexcept {
            const auto base0 = floor(frequency).template as<index_type>();

            Coord kernel[2][2][2]; // 2x2x2 trilinear weights
            set_rasterization_weights_(base0, frequency, kernel);

            using namespace ::noa::fft;
            for (index_type w = 0; w < 2; ++w) {
                for (index_type v = 0; v < 2; ++v) {
                    for (index_type u = 0; u < 2; ++u) {
                        const index_type idx_w = frequency2index<IS_VOLUME_CENTERED>(base0[0] + w, m_grid_shape[0]);
                        const index_type idx_v = frequency2index<IS_VOLUME_CENTERED>(base0[1] + v, m_grid_shape[1]);
                        const index_type idx_u = base0[2] + u;

                        if (idx_w >= 0 and idx_w < m_grid_shape[0] and
                            idx_v >= 0 and idx_v < m_grid_shape[1] and
                            idx_u >= 0 and idx_u < m_grid_shape[2]) {
                            const auto fraction = kernel[w][v][u];
                            ng::atomic_add(
                                    m_output_volume,
                                    value_and_weight.first * static_cast<output_value_real_type>(fraction),
                                    idx_w, idx_v, idx_u);
                            if constexpr (HAS_WEIGHTS) {
                                ng::atomic_add(
                                        m_output_weights,
                                        value_and_weight.second * static_cast<output_weight_real_type>(fraction),
                                        idx_w, idx_v, idx_u);
                            }
                        }
                    }
                }
            }

            // The gridding doesn't preserve the hermitian symmetry, so enforce it on the redundant X==0 ZY plane.
            // So if a side of this plane was modified, add the conjugate at (x=0, -y, -z) with the same fraction.
            if (base0[2] == 0) {
                if constexpr (nt::is_complex_v<output_value_type>)
                    value_and_weight.first.imag = -value_and_weight.first.imag;
                if constexpr (HAS_WEIGHTS and nt::is_complex_v<output_weight_type>)
                    value_and_weight.second.imag = -value_and_weight.second.imag;

                for (index_type w = 0; w < 2; ++w) {
                    for (index_type v = 0; v < 2; ++v) {
                        const index_type idx_w = frequency2index<IS_VOLUME_CENTERED>(-(base0[0] + w), m_grid_shape[0]);
                        const index_type idx_v = frequency2index<IS_VOLUME_CENTERED>(-(base0[1] + v), m_grid_shape[1]);

                        if (idx_w >= 0 and idx_w < m_grid_shape[0] and
                            idx_v >= 0 and idx_v < m_grid_shape[1]) {
                            const auto fraction = kernel[w][v][0];
                            ng::atomic_add(
                                    m_output_volume,
                                    value_and_weight.first * static_cast<output_value_real_type>(fraction),
                                    idx_w, idx_v, index_type{0});
                            if constexpr (HAS_WEIGHTS) {
                                ng::atomic_add(
                                        m_output_weights,
                                        value_and_weight.second * static_cast<output_weight_real_type>(fraction),
                                        idx_w, idx_v, index_type{0});
                            }
                        }
                    }
                }
            }
        }

    private:
        input_slice_accessor_or_value_type m_input_slices;
        output_volume_accessor_type m_output_volume;

        rotate_type m_fwd_rotation;
        shape3_type m_grid_shape;
        index_type m_slice_size_y;
        coord3_type m_f_target_shape;
        coord2_type m_f_slice_shape;
        coord_type m_fftfreq_cutoff_sqd;

        NOA_NO_UNIQUE_ADDRESS input_weight_accessor_or_value_or_empty_type m_input_weights;
        NOA_NO_UNIQUE_ADDRESS output_weight_accessor_or_empty_type m_output_weights;
        NOA_NO_UNIQUE_ADDRESS scale_or_empty_type m_inv_scaling;
        NOA_NO_UNIQUE_ADDRESS ews_or_empty_type m_ews_diam_inv{};
    };
}
