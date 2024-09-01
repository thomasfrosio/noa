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
             nt::readable_nd<3> InputSlice,
             nt::readable_nd_or_empty<3> InputWeight,
             nt::atomic_addable_nd<3> OutputVolume,
             nt::atomic_addable_nd_or_empty<3> OutputWeight>
    class FourierInsertRasterize {
        static_assert(REMAP.is_hx2hx());
        static constexpr bool ARE_SLICES_CENTERED = REMAP.is_xc2xx();
        static constexpr bool IS_VOLUME_CENTERED = REMAP.is_xx2xc();

        using index_type = Index;
        using scale_type = Scale;
        using rotate_type = Rotate;
        using ews_type = EWSCurvature;
        using coord_type = nt::value_type_twice_t<rotate_type>;
        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;

        using input_type = InputSlice;
        using output_type = OutputVolume;
        using input_weight_type = InputWeight;
        using output_weight_type = OutputWeight;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using output_real_type = nt::value_type_t<output_value_type>;
        using output_weight_value_type = nt::value_type_t<output_weight_type>;

        static_assert(guts::fourier_projection_transform_types<scale_type, rotate_type, ews_type> and
                      guts::fourier_projection_types<input_type, output_type> and
                      guts::fourier_projection_weight_types<input_weight_type, output_weight_type>);

    public:
        constexpr FourierInsertRasterize(
                const input_type& input_slices,
                const input_weight_type& input_weights,
                const Shape4<index_type>& input_slice_shape,
                const output_type& output_volume,
                const output_weight_type& output_weights,
                const Shape4<index_type>& output_volume_shape,
                const scale_type& inv_scaling,
                const rotate_type& fwd_rotation,
                coord_type fftfreq_cutoff,
                const Shape4<index_type>& target_shape,
                const ews_type& ews_radius
        ) :
                m_input_slices(input_slices),
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
            if constexpr (not nt::empty<ews_type>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_type{};

            m_fftfreq_cutoff_sqd = max(fftfreq_cutoff, coord_type{});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;
        }

        // For every pixel of every central slice to insert.
        NOA_HD void operator()(index_type batch, index_type y, index_type u) const { // x == u
            // We compute the forward transformation and use normalized frequencies.
            // The oversampling is implicitly handled when scaling back to the target shape.
            const index_type v = noa::fft::index2frequency<ARE_SLICES_CENTERED>(y, m_slice_size_y);
            const auto fftfreq_2d = coord2_type::from_values(v, u) / m_f_slice_shape;
            coord3_type fftfreq_3d = guts::fourier_slice2grid(
                    fftfreq_2d, m_inv_scaling, m_fwd_rotation, batch, m_ews_diam_inv);

            // The frequency rate won't change from that point, so check for the cutoff.
            if (dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd)
                return;

            // Handle the non-redundancy in x.
            coord_type conjugate = 1;
            if (fftfreq_3d[2] < 0) {
                fftfreq_3d = -fftfreq_3d;
                if constexpr (nt::complex<input_value_type>)
                    conjugate = -1;
            }

            // Scale back to the target shape.
            const auto frequency_3d = fftfreq_3d * m_f_target_shape;

            // At this point, we know we are going to use the input, so load everything.
            Pair value_and_weight{
                    get_input_value_(conjugate, batch, y, u),
                    get_input_weight_(batch, y, u),
            };
            rasterize_on_3d_grid_(value_and_weight, frequency_3d);
        }

    private:
        NOA_HD constexpr auto get_input_value_(coord_type conjugate, auto... input_indices) const {
            auto value = m_input_slices(input_indices...);
            if constexpr (nt::complex<input_value_type, output_value_type>) {
                return static_cast<output_value_type>(value * conjugate);
            } else {
                return cast_or_abs_squared<output_value_type>(value);
            }
        }

        NOA_HD constexpr auto get_input_weight_(auto... input_indices) const {
            if constexpr (nt::empty<output_weight_type>)
                return output_weight_value_type{}; // no weights
            else if constexpr (nt::empty<input_weight_type>)
                return output_weight_value_type{1}; // default weights
            else
                return static_cast<output_weight_value_type>(m_input_weights(input_indices...));
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
            for (index_type w{}; w < 2; ++w)
                for (index_type v{}; v < 2; ++v)
                    for (index_type u{}; u < 2; ++u)
                        o_weights[w][v][u] = fraction[w][0] * fraction[v][1] * fraction[u][2];
        }

        // Spread the value within a 2x2x2 centered on a particular frequency using linear interpolation.
        // This is called gridding, but is also referred as rasterization with antialiasing.
        // "frequency" is the frequency, in samples, centered on DC, with negative frequencies on the left.
        NOA_HD void rasterize_on_3d_grid_(
                Pair<output_value_type, output_weight_value_type> value_and_weight,
                const Vec3<coord_type>& frequency // in samples
        ) const noexcept {
            const auto base0 = floor(frequency).template as<index_type>();

            coord_type kernel[2][2][2]; // 2x2x2 trilinear weights
            set_rasterization_weights_(base0, frequency, kernel);

            using namespace ::noa::fft;
            constexpr bool has_weights = not nt::empty<output_weight_type>;

            for (index_type w{}; w < 2; ++w) {
                for (index_type v{}; v < 2; ++v) {
                    for (index_type u{}; u < 2; ++u) {
                        const index_type idx_w = frequency2index<IS_VOLUME_CENTERED>(base0[0] + w, m_grid_shape[0]);
                        const index_type idx_v = frequency2index<IS_VOLUME_CENTERED>(base0[1] + v, m_grid_shape[1]);
                        const index_type idx_u = base0[2] + u;

                        if (idx_w >= 0 and idx_w < m_grid_shape[0] and
                            idx_v >= 0 and idx_v < m_grid_shape[1] and
                            idx_u >= 0 and idx_u < m_grid_shape[2]) {
                            const auto fraction = kernel[w][v][u];
                            ng::atomic_add(
                                    m_output_volume,
                                    value_and_weight.first * static_cast<output_real_type>(fraction),
                                    idx_w, idx_v, idx_u);
                            if constexpr (has_weights) {
                                ng::atomic_add(
                                        m_output_weights,
                                        value_and_weight.second * static_cast<output_weight_value_type>(fraction),
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

                for (index_type w{}; w < 2; ++w) {
                    for (index_type v{}; v < 2; ++v) {
                        const index_type idx_w = frequency2index<IS_VOLUME_CENTERED>(-(base0[0] + w), m_grid_shape[0]);
                        const index_type idx_v = frequency2index<IS_VOLUME_CENTERED>(-(base0[1] + v), m_grid_shape[1]);

                        if (idx_w >= 0 and idx_w < m_grid_shape[0] and
                            idx_v >= 0 and idx_v < m_grid_shape[1]) {
                            const auto fraction = kernel[w][v][0];
                            ng::atomic_add(
                                    m_output_volume,
                                    value_and_weight.first * static_cast<output_real_type>(fraction),
                                    idx_w, idx_v, index_type{});
                            if constexpr (has_weights) {
                                ng::atomic_add(
                                        m_output_weights,
                                        value_and_weight.second * static_cast<output_weight_value_type>(fraction),
                                        idx_w, idx_v, index_type{});
                            }
                        }
                    }
                }
            }
        }

    private:
        input_type m_input_slices;
        output_type m_output_volume;

        rotate_type m_fwd_rotation;
        shape3_type m_grid_shape;
        index_type m_slice_size_y;
        coord3_type m_f_target_shape;
        coord2_type m_f_slice_shape;
        coord_type m_fftfreq_cutoff_sqd;

        NOA_NO_UNIQUE_ADDRESS input_weight_type m_input_weights;
        NOA_NO_UNIQUE_ADDRESS output_weight_type m_output_weights;
        NOA_NO_UNIQUE_ADDRESS scale_type m_inv_scaling;
        NOA_NO_UNIQUE_ADDRESS ews_type m_ews_diam_inv{};
    };
}
