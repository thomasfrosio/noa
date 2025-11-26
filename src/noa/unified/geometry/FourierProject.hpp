#pragma once

#include <valarray>

#include "noa/core/Enums.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/geometry/Euler.hpp"
#include "noa/core/geometry/Quaternion.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/utils/Atomic.hpp"

#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/Interpolation.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Texture.hpp"
#include "noa/unified/Utilities.hpp"

namespace noa::geometry::details {
    template<typename ScaleBatched, typename RotateBatched, typename Ews,
             typename ScaleValue = nt::mutable_value_type_t<ScaleBatched>,
             typename RotateValue = nt::mutable_value_type_t<RotateBatched>,
             typename Coord = nt::value_type_t<RotateValue>>
    concept fourier_projection_transform_types =
        nt::real<Coord> and
        nt::any_of<ScaleValue, Empty, Mat22<Coord>> and
        nt::any_of<RotateValue, Mat33<Coord>, Quaternion<Coord>> and
        nt::any_of<Ews, Empty, Coord, Vec2<Coord>>;

    template<typename Input, typename Output>
    concept fourier_projection_types = nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>;

    template<typename Input, typename Output,
             typename InputValue = nt::value_type_t<Input>,
             typename OutputValue = nt::value_type_t<Output>>
    concept fourier_projection_weight_types =
        nt::real<InputValue, OutputValue> or
        (nt::empty<Input> and nt::real<OutputValue>) or
        (nt::real<InputValue> and nt::empty<OutputValue>) or
        (nt::empty<Input> and nt::empty<Output>);

    // Transforms a 3d fftfreq representing the slice, to its 3d fftfreq in the grid.
    // This is a forward transformation of the frequency, but because it is in Fourier-space,
    // the real-space scaling is inverted.
    template<nt::real Coord,
             nt::batched_parameter ScaleOrEmpty,
             nt::batched_parameter Rotate,
             nt::integer Integer,
             typename EWSOrEmpty>
    NOA_IHD constexpr auto fourier_slice2grid(
        Vec2<Coord> fftfreq,
        const ScaleOrEmpty& inv_scaling,
        const Rotate& fwd_rotation,
        Integer batch,
        EWSOrEmpty inv_ews_diameter
    ) -> Vec3<Coord> {
        // If we apply the EWS curvature, the scaling factors should be corrected
        // before applying the curvature, and therefore before applying the rotation.
        // That way, we use the correct frequencies to compute the EWS, e.g., resulting
        // in a spherical EWS even under anisotropic magnification.
        fftfreq = transform_vector(inv_scaling[batch], fftfreq);

        // TODO We use the Small Angle Approximation to compute the EWS curvature,
        //      so the frequency (u,v) is unchanged. Look at the cisTEM implementation
        //      to remove this approximation? RELION shows that even for low voltages
        //      and large boxes, it is probably not worth it though.
        Vec3<Coord> fftfreq_3d{0, fftfreq[0], fftfreq[1]};
        if constexpr (not nt::empty<EWSOrEmpty>)
            fftfreq_3d[0] = sum(inv_ews_diameter * fftfreq * fftfreq);

        return transform_vector(fwd_rotation[batch], fftfreq_3d);
    }

    // Same as above, but in the other direction.
    template<nt::real Coord,
             nt::batched_parameter ScaleOrEmpty,
             nt::batched_parameter Rotate,
             nt::integer Integer,
             typename EWSOrEmpty>
    NOA_IHD constexpr auto fourier_grid2slice(
        Vec3<Coord> frequency,
        const ScaleOrEmpty& fwd_scaling_matrices,
        const Rotate& inv_rotation,
        Integer batch,
        EWSOrEmpty inv_ews_diameter
    ) -> Pair<Coord, Vec2<Coord>> {
        frequency = transform_vector(inv_rotation[batch], frequency);

        Vec2<Coord> freq_2d{frequency[1], frequency[2]};
        Coord freq_z = frequency[0];
        if constexpr (not nt::empty<EWSOrEmpty>)
            freq_z -= sum(inv_ews_diameter * freq_2d * freq_2d);

        // Same reason as for the forward transformation.
        // Here the grid is correct, so rotate the EWS, then compute
        // the curvature and only then we can scale the slice.
        freq_2d = transform_vector(fwd_scaling_matrices[batch], freq_2d);
        return {freq_z, freq_2d};
    }

    // Windowed-sinc. This function assumes fftfreq <= fftfreq_blackman,
    // above that the blackman window will start again.
    template<typename Coord>
    NOA_FHD Coord windowed_sinc(Coord fftfreq, Coord fftfreq_sinc, Coord fftfreq_blackman) {
        // https://www.desmos.com/calculator/tu5b8aqg2e
        constexpr Coord PI = Constant<Coord>::PI;
        fftfreq *= PI;
        const auto sinc = noa::sinc(fftfreq / fftfreq_sinc);
        const auto blackman_cutoff = fftfreq / fftfreq_blackman;
        const auto blackman = static_cast<Coord>(0.42) +
                              static_cast<Coord>(0.5) * cos(blackman_cutoff) +
                              static_cast<Coord>(0.08) * cos(2 * blackman_cutoff);
        return sinc * blackman;
    }

    // This is only used for the Fourier extraction step.
    // The window is always an odd-numbered size.
    template<nt::integer Int, nt::real Coord>
    Int blackman_window_size(Coord fftfreq_blackman, Coord spectrum_size) {
        // Given a blackman window in range [0, fftfreq_blackman] and a spectrum logical-size
        // (the z size in our case), what is the size of the blackman window, in elements.
        // For instance:
        //  spectrum_size=10, fftfreq_blackman=0.23
        //  rfftfreq=[0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        //  rfftfreq_samples=4.6->5, window_size=11
        //  computed_window=[-0.25,-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2,0.25]
        auto rfftfreq_samples = static_cast<f64>(spectrum_size) * static_cast<f64>(fftfreq_blackman);
        if (allclose(rfftfreq_samples, 1.)) {
            // Due to floating-point precision errors, the default value (1/spectrum_size)
            // may be slightly greater than 1. In this case, we really mean 1.
            rfftfreq_samples = round(rfftfreq_samples);
        } else {
            rfftfreq_samples = ceil(rfftfreq_samples); // include last fraction
        }
        const auto rfftfreq_samples_int = max(Int{1}, static_cast<Int>(rfftfreq_samples));
        auto window_size = 2 * (rfftfreq_samples_int) + 1;

        // Truncate the edges because at these indexes, the window is 0, so there's no need to compute it.
        // So using the same example, computed_window=[-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2]
        return window_size - 2;
    }

    // This is only used for the Fourier extraction step.
    // Given the iwise-index w and the blackman window size, return the fftfreq offset. For instance,
    // window_size=9: w=[0..8] -> [-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2]
    template<nt::integer Int, nt::real Coord>
    constexpr NOA_FHD Coord w_index_to_fftfreq_offset(Int w, Int window_size, Coord spectrum_size) {
        return static_cast<Coord>(w - window_size / 2) / spectrum_size;
    }

    // This is only used for the Fourier extraction step.
    // Compute the sum of the z-window, so that it can be directly applied to the extracted values,
    // thereby correcting for the multiplicity on the fly.
    template<nt::integer Int, nt::real Real>
    Pair<Int, Real> z_window_spec(Real fftfreq_sinc, Real fftfreq_blackman, Real spectrum_size) {
        auto window_size = blackman_window_size<Int>(fftfreq_blackman, spectrum_size);
        Real sum{};
        for (Int i{}; i < window_size; ++i) {
            const auto fftfreq = w_index_to_fftfreq_offset(i, window_size, spectrum_size);
            sum += windowed_sinc(fftfreq, fftfreq_sinc, fftfreq_blackman);
        }
        return {window_size, sum};
    }

    template<nf::Layout REMAP,
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
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using output_real_type = nt::value_type_t<output_value_type>;
        using output_weight_value_type = nt::value_type_t<output_weight_type>;

        static_assert(details::fourier_projection_transform_types<scale_type, rotate_type, ews_type> and
                      details::fourier_projection_types<input_type, output_type> and
                      details::fourier_projection_weight_types<input_weight_type, output_weight_type>);

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
            const index_type v = nf::index2frequency<ARE_SLICES_CENTERED>(y, m_slice_size_y);
            const auto fftfreq_2d = coord2_type::from_values(v, u) / m_f_slice_shape;
            coord3_type fftfreq_3d = details::fourier_slice2grid(
                fftfreq_2d, m_inv_scaling, m_fwd_rotation, batch, m_ews_diam_inv);

            // The frequency rate won't change from that point, so check for the cutoff.
            if (dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd)
                return;

            // Handle the non-redundancy in x.
            input_real_type conjugate = 1;
            if (fftfreq_3d[2] < 0) {
                fftfreq_3d = -fftfreq_3d;
                if constexpr (nt::complex<input_value_type>)
                    conjugate = -1;
            }

            // Scale back to the target shape.
            const auto frequency_3d = fftfreq_3d * m_f_target_shape;

            rasterize_on_3d_grid_(
                get_input_value_(conjugate, batch, y, u),
                get_input_weight_(batch, y, u),
                frequency_3d
            );
        }

    private:
        NOA_HD constexpr auto get_input_value_(input_real_type conjugate, auto... input_indices) const {
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
            fraction[0] = 1 - fraction[1];
            for (index_type w{}; w < 2; ++w)
                for (index_type v{}; v < 2; ++v)
                    for (index_type u{}; u < 2; ++u)
                        o_weights[w][v][u] = fraction[w][0] * fraction[v][1] * fraction[u][2];
        }

        // Spread the value within a 2x2x2 centered on a particular frequency using linear interpolation.
        // This is called gridding, but is also referred as rasterization with antialiasing.
        // "frequency" is the frequency, in samples, centered on DC, with negative frequencies on the left.
        NOA_HD void rasterize_on_3d_grid_(
            output_value_type value,
            const output_weight_value_type& weight,
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
                            nd::atomic_add(
                                m_output_volume,
                                value * static_cast<output_real_type>(fraction),
                                idx_w, idx_v, idx_u);
                            if constexpr (has_weights) {
                                nd::atomic_add(
                                    m_output_weights,
                                    weight * static_cast<output_weight_value_type>(fraction),
                                    idx_w, idx_v, idx_u);
                            }
                        }
                    }
                }
            }

            // The gridding doesn't preserve the hermitian symmetry, so enforce it on the redundant X==0 ZY plane.
            // So if a side of this plane was modified, add the conjugate at (x=0, -y, -z) with the same fraction.
            if (base0[2] == 0) {
                if constexpr (nt::complex<output_value_type>)
                    value.imag = -value.imag;

                for (index_type w{}; w < 2; ++w) {
                    for (index_type v{}; v < 2; ++v) {
                        const index_type idx_w = frequency2index<IS_VOLUME_CENTERED>(-(base0[0] + w), m_grid_shape[0]);
                        const index_type idx_v = frequency2index<IS_VOLUME_CENTERED>(-(base0[1] + v), m_grid_shape[1]);

                        if (idx_w >= 0 and idx_w < m_grid_shape[0] and
                            idx_v >= 0 and idx_v < m_grid_shape[1]) {
                            const auto fraction = kernel[w][v][0];
                            nd::atomic_add(
                                m_output_volume,
                                value * static_cast<output_real_type>(fraction),
                                idx_w, idx_v, index_type{});
                            if constexpr (has_weights) {
                                nd::atomic_add(
                                    m_output_weights,
                                    weight * static_cast<output_weight_value_type>(fraction),
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

    template<nf::Layout REMAP,
             nt::sinteger Index,
             nt::batched_parameter Scale,
             nt::batched_parameter Rotate,
             typename EWSCurvature,
             nt::interpolator_spectrum_nd<2> InputSlice,
             nt::interpolator_spectrum_nd_or_empty<2> InputSliceWeight,
             nt::writable_nd<3> OutputVolume,
             nt::writable_nd_or_empty<3> OutputVolumeWeight>
    class FourierInsertInterpolate {
        static constexpr bool IS_VOLUME_CENTERED = REMAP.is_xx2xc();
        static constexpr bool IS_VOLUME_RFFT = REMAP.is_xx2hx();

        using index_type = Index;
        using scale_type = Scale;
        using rotate_type = Rotate;
        using ews_type = EWSCurvature;
        using coord_type = nt::value_type_twice_t<rotate_type>;
        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;
        using shape_nd_type = Shape<index_type, 3 - IS_VOLUME_RFFT>;

        using input_type = InputSlice;
        using input_weight_type = InputSliceWeight;
        using output_type = OutputVolume;
        using output_weight_type = OutputVolumeWeight;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static constexpr bool has_input_weights = not nt::empty<input_weight_type>;
        static constexpr bool has_output_weights = not nt::empty<output_weight_type>;
        using output_weight_value_type = nt::value_type_t<output_weight_type>;
        using input_weight_value_type = std::conditional_t<
            has_input_weights, nt::mutable_value_type_t<input_weight_type>, output_weight_value_type>;

        static_assert(details::fourier_projection_transform_types<scale_type, rotate_type, ews_type> and
                      details::fourier_projection_types<input_type, output_type> and
                      details::fourier_projection_weight_types<input_weight_type, output_weight_type>);

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
        ) :
            m_input_slices(input_slices),
            m_output_volume(output_volume),
            m_inv_rotation(inv_rotation),
            m_slice_count(input_slice_shape[0]),
            m_input_weights(input_weights),
            m_output_weights(output_weights),
            m_fwd_scaling(fwd_scaling)
        {
            const auto slice_shape_2d = input_slice_shape.filter(2, 3);
            m_f_slice_shape = coord2_type::from_vec(slice_shape_2d.vec);

            const auto grid_shape = output_volume_shape.pop_front();
            const auto l_target_shape = any(target_shape == 0) ? grid_shape : target_shape.pop_front();
            m_grid_shape = grid_shape.template pop_back<IS_VOLUME_RFFT>();
            m_f_target_shape = coord3_type::from_vec(l_target_shape.vec);

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (not nt::empty<ews_type>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_type{};

            m_fftfreq_cutoff_sqd = max(fftfreq_cutoff, coord_type{});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;

            // Clamp the windowed-sinc to ensure it's at least one pixel thick.
            const auto max_output_size = static_cast<coord_type>(min(l_target_shape));
            m_fftfreq_sinc = max(fftfreq_sinc, 1 / max_output_size);
            m_fftfreq_blackman = max(fftfreq_blackman, 1 / max_output_size);
        }

        // For every voxel of the grid.
        NOA_HD void operator()(index_type oz, index_type oy, index_type ox) const noexcept {
            const auto frequency = nf::index2frequency<IS_VOLUME_CENTERED, IS_VOLUME_RFFT>(
                Vec{oz, oy, ox}, m_grid_shape);
            const auto fftfreq = coord3_type::from_vec(frequency) / m_f_target_shape;
            if (dot(fftfreq, fftfreq) > m_fftfreq_cutoff_sqd)
                return;

            input_value_type value{};
            input_weight_value_type weights{};

            for (index_type i{}; i < m_slice_count; ++i) {
                const auto [fftfreq_z, fftfreq_2d] = details::fourier_grid2slice(
                    fftfreq, m_fwd_scaling, m_inv_rotation, i, m_ews_diam_inv);

                input_value_type i_value{};
                input_weight_value_type i_weights{};
                if (abs(fftfreq_z) <= m_fftfreq_blackman) { // the slice affects the voxel
                    const auto window = details::windowed_sinc(fftfreq_z, m_fftfreq_sinc, m_fftfreq_blackman);
                    const auto frequency_2d = fftfreq_2d * m_f_slice_shape;

                    i_value = m_input_slices.interpolate_spectrum_at(frequency_2d, i) *
                              static_cast<input_real_type>(window);

                    if constexpr (has_output_weights) {
                        if constexpr (has_input_weights) {
                            i_weights = m_input_weights.interpolate_spectrum_at(frequency_2d, i) *
                                        static_cast<input_weight_value_type>(window);
                        } else {
                            i_weights = static_cast<input_weight_value_type>(window); // input_weight=1
                        }
                    }
                }
                value += i_value;
                if constexpr (has_output_weights)
                    weights += i_weights;
            }

            // The transformation preserves the hermitian symmetry, so there's nothing else to do.
            m_output_volume(oz, oy, ox) += cast_or_abs_squared<output_value_type>(value);
            if constexpr (has_output_weights)
                m_output_weights(oz, oy, ox) += cast_or_abs_squared<output_weight_value_type>(weights);
        }

    private:
        input_type m_input_slices;
        output_type m_output_volume;

        rotate_type m_inv_rotation;
        shape_nd_type m_grid_shape;
        coord3_type m_f_target_shape;
        coord2_type m_f_slice_shape;
        index_type m_slice_count;

        coord_type m_fftfreq_cutoff_sqd;
        coord_type m_fftfreq_sinc;
        coord_type m_fftfreq_blackman;

        NOA_NO_UNIQUE_ADDRESS input_weight_type m_input_weights;
        NOA_NO_UNIQUE_ADDRESS output_weight_type m_output_weights;
        NOA_NO_UNIQUE_ADDRESS scale_type m_fwd_scaling;
        NOA_NO_UNIQUE_ADDRESS ews_type m_ews_diam_inv{};
    };

    template<nf::Layout REMAP,
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

        using batched_scale_type = Scale;
        using batched_rotate_type = Rotate;
        using ews_type = EWSCurvature;
        using coord_type = nt::value_type_twice_t<batched_rotate_type>;
        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;

        static_assert(details::fourier_projection_transform_types<batched_scale_type, batched_rotate_type, ews_type> and
                      details::fourier_projection_types<input_type, output_type> and
                      details::fourier_projection_weight_types<input_weight_type, output_weight_type>);

        // Optional operator requires atomic_add.
        static constexpr bool are_outputs_atomic =
            nt::atomic_addable_nd<output_type, 3> and
            nt::atomic_addable_nd_or_empty<output_weight_type, 3>;

    public:
        FourierExtract(
            const input_type& input_volume,
            const input_weight_type& input_weights,
            const Shape4<index_type>& input_volume_shape,
            const output_type& output_slices,
            const output_weight_type& output_weights,
            const Shape4<index_type>& output_slice_shape,
            const batched_scale_type& inv_scaling,
            const batched_rotate_type& fwd_rotation,
            coord_type fftfreq_sinc,
            coord_type fftfreq_blackman,
            coord_type fftfreq_cutoff,
            const Shape4<index_type>& target_shape,
            const ews_type& ews_radius
        ) :
            m_input_volume(input_volume),
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
            tie(m_blackman_size, m_w_window_sum) = details::z_window_spec<index_type>(
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
        NOA_HD constexpr void operator()(
            index_type batch, index_type ow, index_type oy, index_type ox
        ) const requires are_outputs_atomic {
            coord3_type fftfreq_3d = compute_fftfreq_in_volume_(batch, oy, ox);

            // Additional z component, within the grid coordinate system.
            const auto fftfreq_z_offset = details::w_index_to_fftfreq_offset(ow, m_blackman_size, m_f_target_shape[0]);
            fftfreq_3d[0] += fftfreq_z_offset;

            if (dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd)
                return;

            const auto frequency_3d = fftfreq_3d * m_f_target_shape;
            const auto convolution_weight =
                details::windowed_sinc(fftfreq_z_offset, m_fftfreq_sinc, m_fftfreq_blackman) / m_w_window_sum;

            const auto value = m_input_volume.interpolate_spectrum_at(frequency_3d);
            nd::atomic_add(
                m_output_slices,
                cast_or_abs_squared<output_value_type>(value) *
                static_cast<output_real_type>(convolution_weight),
                batch, oy, ox);

            if constexpr (not nt::empty<output_weight_type>) {
                output_weight_value_type weight;
                if constexpr (not nt::empty<input_weight_type>) {
                    weight = static_cast<output_weight_value_type>(
                        m_input_weights.interpolate_spectrum_at(frequency_3d));
                } else {
                    weight = 1;
                }
                nd::atomic_add(
                    m_output_weights,
                    weight * static_cast<output_weight_value_type>(convolution_weight),
                    batch, oy, ox);
            }
        }

    private:
        // The indexes give us the fftfreq in the coordinate system of the slice to extract.
        // This function transforms this fftfreq to the coordinate system of the volume.
        NOA_HD coord3_type compute_fftfreq_in_volume_(index_type batch, index_type oy, index_type ox) const {
            const auto frequency_2d = nf::index2frequency<ARE_SLICES_CENTERED, ARE_SLICES_RFFT>(
                Vec{oy, ox}, m_slice_shape);
            const auto fftfreq_2d = coord2_type::from_vec(frequency_2d) / m_f_slice_shape;
            return details::fourier_slice2grid(
                fftfreq_2d, m_inv_scaling, m_fwd_rotation, batch, m_ews_diam_inv);
        }

    private:
        input_type m_input_volume;
        output_type m_output_slices;

        batched_rotate_type m_fwd_rotation;
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
        NOA_NO_UNIQUE_ADDRESS batched_scale_type m_inv_scaling;
        NOA_NO_UNIQUE_ADDRESS ews_type m_ews_diam_inv{};
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
    template<nf::Layout REMAP,
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

        static_assert(details::fourier_projection_transform_types<input_scale_type, input_rotate_type, ews_type> and
                      details::fourier_projection_transform_types<output_scale_type, output_rotate_type, ews_type> and
                      details::fourier_projection_types<input_type, output_type> and
                      details::fourier_projection_weight_types<input_weight_type, output_weight_type>);

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
            bool add_to_output, bool correct_weights,
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
            m_correct_weights(correct_weights)
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
            tie(m_extract_blackman_size, m_extract_window_total_weight) = details::z_window_spec<index_type>(
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

            const auto value_and_weight = sample_virtual_volume_(fftfreq_3d, m_correct_weights);

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
            const auto fftfreq_z_offset = details::w_index_to_fftfreq_offset(w, m_extract_blackman_size, m_volume_z);
            fftfreq_3d[0] += fftfreq_z_offset;

            if (dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd)
                return;

            // The weights cannot be corrected on-the-fly in this case because
            // the final weight is unknown at this point!
            const auto value_and_weight = sample_virtual_volume_(fftfreq_3d, false);

            // z-windowed sinc.
            const auto convolution_weight =
                details::windowed_sinc(fftfreq_z_offset, m_extract_fftfreq_sinc, m_extract_fftfreq_blackman) /
                m_extract_window_total_weight;

            // Add the contribution for this z-offset. The z-convolution is essentially a simple weighted mean.
            nd::atomic_add(
                m_output_slices,
                cast_or_abs_squared<output_value_type>(value_and_weight.first) *
                static_cast<output_real_type>(convolution_weight),
                batch, y, x);
            if constexpr (has_output_weights) {
                nd::atomic_add(
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
            const auto frequency_2d = nf::index2frequency<ARE_OUTPUT_SLICES_CENTERED, ARE_OUTPUT_SLICES_RFFT>(
                Vec{y, x}, m_output_shape);
            const auto fftfreq_2d = coord2_type::from_vec(frequency_2d) / m_f_output_shape;
            return details::fourier_slice2grid(
                fftfreq_2d, m_extract_inv_scaling, m_extract_fwd_rotation, batch, m_ews_diam_inv);
        }

        NOA_HD auto sample_virtual_volume_(const coord3_type& fftfreq_3d, bool correct_weights) const noexcept {
            using input_weight_value_type =
                std::conditional_t<has_input_weights, nt::mutable_value_type_t<input_weight_type>,
                std::conditional_t<has_output_weights, output_weight_value_type, input_real_type>>;

            input_value_type value{};
            input_weight_value_type weight{};

            // For every slice to insert...
            for (index_type i{}; i < m_input_count; ++i) {
                // Project the 3d frequency onto that input-slice.
                // fftfreq_z is along the normal of that input-slice.
                const auto [fftfreq_z, fftfreq_yx] = details::fourier_grid2slice(
                    fftfreq_3d, m_insert_fwd_scaling, m_insert_inv_rotation, i, m_ews_diam_inv);

                // Add the contribution of this slice to that frequency.
                // Compute only if this slice affects the voxel.
                // If we fall exactly at the blackman cutoff, the value is 0, so exclude the equality case too.
                if (abs(fftfreq_z) < m_insert_fftfreq_blackman) {
                    const auto windowed_sinc = details::windowed_sinc(
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
            if (correct_weights) {
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
        bool m_correct_weights;
    };

        /// Pre/post gridding correction, assuming linear interpolation.
    template<bool POST_CORRECTION,
             nt::real Coord,
             nt::readable_nd<4> Input,
             nt::writable_nd<4> Output>
    class GriddingCorrection {
        using coord_type = Coord;
        using coord3_type = Vec3<coord_type>;
        using input_type = Input;
        using output_type = Output;
        using output_value_type = nt::value_type_t<output_type>;
        using input_value_type = nt::value_type_t<input_type>;
        static_assert(nt::real<input_value_type, output_value_type>);

    public:
        template<typename T>
        constexpr GriddingCorrection(
            const input_type& input,
            const output_type& output,
            const Shape4<T>& shape
        ) :
            m_input(input),
            m_output(output)
        {
            const auto l_shape = shape.pop_front();
            m_f_shape = coord3_type::from_vec(l_shape.vec);
            m_half = m_f_shape / 2 * coord3_type::from_vec(l_shape != 1); // if size == 1, half should be 0
        }

        template<nt::integer T>
        NOA_HD void operator()(T batch, T j, T k, T l) const noexcept {
            auto dist = coord3_type::from_values(j, k, l);
            dist -= m_half;
            dist /= m_f_shape;

            constexpr coord_type PI = Constant<coord_type>::PI;
            const coord_type radius = sqrt(dot(dist, dist));
            const coord_type sinc = noa::sinc(PI * radius);
            const auto sinc2 = static_cast<input_value_type>(sinc * sinc); // > 0.05

            const auto value = m_input(batch, j, k, l);
            if constexpr (POST_CORRECTION) {
                m_output(batch, j, k, l) = static_cast<output_value_type>(value / sinc2);
            } else {
                m_output(batch, j, k, l) = static_cast<output_value_type>(value * sinc2);
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        coord3_type m_f_shape;
        coord3_type m_half;
    };

    template<bool AllowTexture, bool AllowValue,
             typename Input, typename Output,
             typename InputValue = nt::value_type_t<Input>,
             typename OutputValue = nt::value_type_t<Output>>
    concept fourier_projection_input_output_value =
        nt::writable_varray<Output> and
        ((nt::readable_varray<Input> and nt::spectrum_types<InputValue, OutputValue>) or
         (AllowTexture and nt::texture<Input> and nt::spectrum_types<InputValue, OutputValue>) or
         (AllowValue and nt::spectrum_types<Input, OutputValue>));

    template<bool AllowTexture, bool AllowValue, typename Input, typename Output>
    concept fourier_projection_input_output_weight =
        (nt::writable_varray_of_real<Output> or nt::empty<Output>) and
        (nt::readable_varray_of_real<Input> or nt::empty<Input> or
         (AllowTexture and nt::texture_of_real<Input>) or
         (AllowValue and nt::real<Input>));

    template<bool AllowTexture, bool AllowValue,
             typename Input, typename Output, typename InputWeight, typename OutputWeight>
    concept fourier_projection_input_output =
        fourier_projection_input_output_value<AllowTexture, AllowValue, std::decay_t<Input>, std::decay_t<Output>> and
        fourier_projection_input_output_weight<AllowTexture, AllowValue, std::decay_t<InputWeight>, std::decay_t<OutputWeight>>;

    template<typename Scale, typename Rotation,
             typename Coord = nt::mutable_value_type_twice_t<Rotation>>
    concept fourier_projection_transform =
        (nt::any_of<std::decay_t<Rotation>, Mat33<f32>, Mat33<f64>, Quaternion<f32>, Quaternion<f64>> or
         nt::varray_decay_of_almost_any<Rotation, Mat33<f32>, Mat33<f64>, Quaternion<f32>, Quaternion<f64>>) and
        (nt::any_of<std::decay_t<Scale>, Empty, Mat22<Coord>> or nt::varray_decay_of_almost_any<Scale, Mat22<Coord>>);

    enum class FourierProjectionType { INSERT_RASTERIZE, INSERT_INTERPOLATE, EXTRACT, INSERT_EXTRACT };

    template<FourierProjectionType DIRECTION,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename InputScale, typename InputRotate,
             typename OutputScale = Mat22<f64>, typename OutputRotate = Mat33<f64>>
    void fourier_projection_check_parameters(
        const Input& input, const InputWeight& input_weight, const Shape4<i64>& input_shape,
        const Output& output, const OutputWeight& output_weight, const Shape4<i64>& output_shape,
        const Shape4<i64>& target_shape,
        const InputScale& input_scaling,
        const InputRotate& input_rotation,
        const OutputScale& output_scaling = {},
        const OutputRotate& output_rotation = {}
    ) {
        check(not output.is_empty(), "Empty array detected");
        const Device output_device = output.device();
        check(vall(Equal{}, output.shape(), output_shape.rfft()),
              "The shape of the rfft output does not match the expected shape. Got output:shape={} and expected:shape={}",
              output.shape(), output_shape.rfft());

        auto check_input = [&]<typename T>(const T& array, std::string_view name) {
            if constexpr (nt::varray<T> or nt::texture<T>) {
                check(not array.is_empty(), "Empty array detected");
                if constexpr (nt::varray<Input>)
                    check(not ni::are_overlapped(array, output), "Input and output arrays should not overlap");

                if constexpr (nt::texture<Input>) {
                    check(array.border() == Border::ZERO,
                          "The texture border mode should be {}, but got {}",
                          Border::ZERO, array.border());
                }
                const Device device = array.device();
                check(device == output_device,
                      "The arrays should be on the same device, but got {}:device={} and output:device={}",
                      name, device, output_device);
                check(vall(Equal{}, array.shape(), input_shape.rfft()),
                      "The shape of the rfft does not match the expected shape. Got {}:shape={} and shape={}",
                      name, array.shape(), input_shape.rfft());
            }
        };
        check_input(input, "input");
        check_input(input_weight, "input_weight");

        if constexpr (not nt::empty<OutputWeight>) {
            check(not output_weight.is_empty(), "Empty array detected");
            check(not ni::are_overlapped(output_weight, output), "Output arrays should not overlap");
            check(output_weight.device() == output_device,
                  "The arrays should be on the same device, but got output_weight:device={} and output:device={}",
                  output_weight.device(), output_device);
            check(vall(Equal{}, output_weight.shape(), output_shape.rfft()),
                  "The shape of the rfft does not match the expected shape. Got output_weight:shape={} and shape={}",
                  output_weight.shape(), output_shape.rfft());
        }

        if constexpr (DIRECTION == FourierProjectionType::INSERT_RASTERIZE or
                      DIRECTION == FourierProjectionType::INSERT_INTERPOLATE) {
            check(input_shape[1] == 1, "2d input slices are expected, but got shape={}", input_shape);
            check(output_shape[0] == 1 and target_shape[0] <= 1,
                  "A single 3d volume is expected, but got output_shape={} and target_shape={} (optional)",
                  output_shape, target_shape);
        } else if constexpr (DIRECTION == FourierProjectionType::EXTRACT) {
            check(input_shape[0] == 1 and target_shape[0] <= 1,
                  "A single 3d volume is expected, but got input_shape={} and target_shape={} (optional)",
                  input_shape, target_shape);
            check(output_shape[1] == 1, "2d input slices are expected but got shape {}", output_shape);
        } else { // INSERT_EXTRACT
            check(input_shape[1] == 1 and output_shape[1] == 1,
                  "2d slices are expected but got shape input:shape={} and output:shape={}",
                  input_shape, output_shape);
        }

        auto check_transform = [&](const auto& transform, i64 required_size, std::string_view name) {
            check(not transform.is_empty(), "{} should not be empty", name);
            check(ni::is_contiguous_vector(transform) and transform.n_elements() == required_size,
                  "{} should be a contiguous vector with n_slices={} elements, but got {}:shape={}, {}:strides={}",
                  name, required_size, name, transform.shape(), name, transform.strides());
            check(transform.device() == output_device, "{} should be on the compute device", name);
        };

        const auto required_count = DIRECTION == FourierProjectionType::EXTRACT ? output_shape[0] : input_shape[0];
        if constexpr (nt::varray<InputScale>)
            check_transform(input_scaling, required_count, "input_scaling");
        if constexpr (nt::varray<InputRotate>)
            check_transform(input_rotation, required_count, "input_rotation");

        // Only for INSERT_EXTRACT.
        if constexpr (nt::varray<OutputScale>)
            check_transform(output_scaling, output_shape[0], "output_scaling");
        if constexpr (nt::varray<OutputRotate>)
            check_transform(output_rotation, output_shape[0], "output_rotation");
    }

    template<typename T, typename U, typename V, typename W>
    auto fourier_projection_is_i32_safe_access(
        const T& input, const U& input_weight, const V& output, const W& output_weight
    ) {
        bool is_safe_access{true};
        if constexpr (nt::varray_decay<T>)
            is_safe_access = nd::is_accessor_access_safe<i32>(input, input.shape());
        if constexpr (nt::varray_decay<U>)
            is_safe_access = is_safe_access and nd::is_accessor_access_safe<i32>(input_weight, input_weight.shape());
        if constexpr (nt::varray_decay<V>)
            is_safe_access = is_safe_access and nd::is_accessor_access_safe<i32>(output, output.shape());
        if constexpr (nt::varray_decay<W>)
            is_safe_access = is_safe_access and nd::is_accessor_access_safe<i32>(output_weight, output_weight.shape());
        return is_safe_access;
    }

    template<typename T, typename U>
    auto fourier_insert_extract_interp_mode(const T& input, const U& input_weight, Interp interp) -> Interp {
        if constexpr (nt::texture<T>) {
            interp = input.interp();
            if constexpr (nt::texture<U>) {
                check(input_weight.interp() == interp,
                      "Input textures should have the same interpolation method, "
                      "but got input:interp={} and input_weight:interp={}",
                      interp, input_weight.interp());
            }
        } else if constexpr (nt::texture<U>) {
            interp = input_weight.interp();
        }
        return interp;
    }

    template<size_t N, nf::Layout REMAP, bool IS_GPU, Interp INTERP, typename Coord, typename Index, typename T>
    auto fourier_projection_to_interpolator(const T& input, const Shape<Index, 4>& shape) {
        if constexpr (nt::varray_or_texture<T>) {
            return nd::to_interpolator_spectrum<N, REMAP, INTERP, Coord, IS_GPU>(input, shape);

        } else if constexpr (nt::empty<T>) {
            return input;

        } else { // real or complex
            using accessor_t = AccessorValue<const T, Index>;
            using interpolator_t = InterpolatorSpectrum<N, REMAP, INTERP.erase_fast(), accessor_t>;
            return interpolator_t(accessor_t(input), shape.template filter_nd<N>().pop_front());
        }
    }

    template<typename T>
    constexpr bool fourier_project_has_scale(const T& scale) {
        if constexpr (nt::mat22<T>)
            return scale != T::eye(1);
        else if constexpr (nt::varray<T> and nt::mat22<nt::value_type_t<T>>)
            return true;
        else if constexpr (nt::empty<T>)
            return false;
        else
            static_assert(nt::always_false<T>);
    }

    template<bool ENFORCE_EMPTY, typename Coord>
    constexpr auto fourier_projection_to_ews(const Vec<f64, 2>& ews) {
        if constexpr (ENFORCE_EMPTY)
            return Empty{};
        else
            return ews.as<Coord>();
    }

    // nvcc struggles with C++20 template parameters in lambda, so use a worse C++17 syntax and create this type...
    template<bool VALUE>
    struct WrapNoEwaldAndScale {
        consteval auto operator()() const -> bool { return VALUE;}
    };

    template<nf::Layout REMAP, typename Index,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename Scale, typename Rotate>
    void launch_rasterize_central_slices_3d(
        Input&& slice, InputWeight&& slice_weight, const Shape4<i64>& slice_shape,
        Output&& volume, OutputWeight&& volume_weight, const Shape4<i64>& volume_shape,
        Scale&& scaling, Rotate&& rotation, const auto& options
    ) {
        constexpr auto input_accessor_config = nd::AccessorConfig<3>{
            .enforce_const=true,
            .enforce_restrict=true,
            .allow_empty=true,
            .filter={0, 2, 3},
        };
        constexpr auto output_accessor_config = nd::AccessorConfig<3>{
            .enforce_restrict=true,
            .allow_empty=true,
            .filter={1, 2, 3},
        };
        auto slice_accessor = nd::to_accessor<input_accessor_config, Index>(slice);
        auto slice_weight_accessor = nd::to_accessor<input_accessor_config, Index>(slice_weight);
        auto volume_accessor = nd::to_accessor<output_accessor_config, Index>(volume);
        auto volume_weight_accessor = nd::to_accessor<output_accessor_config, Index>(volume_weight);

        const auto s_input_slice_shape = slice_shape.as<Index>();
        const auto s_volume_shape = volume_shape.as<Index>();
        auto batched_rotation = nd::to_batched_transform(rotation);
        using coord_t = nt::value_type_twice_t<Rotate>;

        auto launch = [&](auto no_ews_and_scale) {
            auto ews = fourier_projection_to_ews<no_ews_and_scale(), coord_t>(options.ews_radius);
            auto batched_scaling = nd::to_batched_transform<true, no_ews_and_scale()>(scaling);

            using op_t = FourierInsertRasterize<
                REMAP, Index,
                decltype(batched_scaling), decltype(batched_rotation), decltype(ews),
                decltype(slice_accessor), decltype(slice_weight_accessor),
                decltype(volume_accessor), decltype(volume_weight_accessor)>;
            auto op = op_t(
                slice_accessor, slice_weight_accessor, s_input_slice_shape,
                volume_accessor, volume_weight_accessor, s_volume_shape,
                batched_scaling, batched_rotation,
                static_cast<coord_t>(options.fftfreq_cutoff),
                options.target_shape.template as<Index>(), ews);

            iwise(s_input_slice_shape.filter(0, 2, 3).rfft(), volume.device(), op,
                  std::forward<Input>(slice), std::forward<InputWeight>(slice_weight),
                  std::forward<Output>(volume), std::forward<OutputWeight>(volume_weight),
                  std::forward<Scale>(scaling), std::forward<Rotate>(rotation));
        };

        const auto has_ews = any(options.ews_radius != 0);
        const bool has_scale = fourier_project_has_scale(scaling);
        if (has_ews or has_scale)
            return launch(WrapNoEwaldAndScale<false>{});
        return launch(WrapNoEwaldAndScale<true>{});
    }

    template<nf::Layout REMAP, typename Index, bool IS_GPU,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename Scale, typename Rotate>
    void launch_insert_central_slices_3d(
        Input&& slice, InputWeight&& slice_weight, const Shape4<i64>& slice_shape,
        Output&& volume, OutputWeight&& volume_weight, const Shape4<i64>& volume_shape,
        Scale&& scaling, Rotate&& rotation, const auto& options
    ) {
        constexpr auto accessor_config = nd::AccessorConfig<3>{
            .enforce_restrict = true,
            .allow_empty = true,
            .filter = {1, 2, 3},
        };
        auto volume_accessor = nd::to_accessor<accessor_config, Index>(volume);
        auto volume_weight_accessor = nd::to_accessor<accessor_config, Index>(volume_weight);

        const auto s_slice_shape = slice_shape.as<Index>();
        const auto s_volume_shape = volume_shape.as<Index>();
        auto batched_rotation = nd::to_batched_transform<false>(rotation);
        using coord_t = nt::value_type_twice_t<Rotate>;

        auto launch = [&](auto no_ews_and_scale, auto interp) {
            auto slice_interpolator = fourier_projection_to_interpolator
                <2, REMAP, IS_GPU, interp(), coord_t>(slice, s_slice_shape);
            auto slice_weight_interpolator = fourier_projection_to_interpolator
                <2, REMAP, IS_GPU, interp(), coord_t>(slice_weight, s_slice_shape);

            auto batched_scaling = nd::to_batched_transform<true, no_ews_and_scale()>(scaling);
            auto ews = fourier_projection_to_ews<no_ews_and_scale(), coord_t>(options.ews_radius);

            using op_t = FourierInsertInterpolate<
                REMAP, Index, decltype(batched_scaling), decltype(batched_rotation), decltype(ews),
                decltype(slice_interpolator), decltype(slice_weight_interpolator),
                decltype(volume_accessor), decltype(volume_weight_accessor)>;
            auto op = op_t(
                slice_interpolator, slice_weight_interpolator, s_slice_shape,
                volume_accessor, volume_weight_accessor, s_volume_shape,
                batched_scaling, batched_rotation,
                static_cast<coord_t>(options.windowed_sinc.fftfreq_sinc),
                static_cast<coord_t>(options.windowed_sinc.fftfreq_blackman),
                static_cast<coord_t>(options.fftfreq_cutoff),
                options.target_shape.template as<Index>(), ews);

            iwise(s_volume_shape.filter(1, 2, 3).rfft(), volume.device(), op,
                  std::forward<Input>(slice), std::forward<InputWeight>(slice_weight),
                  std::forward<Output>(volume), std::forward<OutputWeight>(volume_weight),
                  std::forward<Scale>(scaling), std::forward<Rotate>(rotation));
        };

        const auto has_ews = any(options.ews_radius != 0);
        const bool has_scale = fourier_project_has_scale(scaling);
        auto launch_scale = [&](auto interp) {
            if (has_ews or has_scale)
                return launch(WrapNoEwaldAndScale<false>{}, interp);
            return launch(WrapNoEwaldAndScale<true>{}, interp);
        };

        const Interp interp = details::fourier_insert_extract_interp_mode(slice, slice_weight, options.interp);
        switch (interp) {
            case Interp::NEAREST:            return launch_scale(nd::WrapInterp<Interp::NEAREST>{});
            case Interp::NEAREST_FAST:       return launch_scale(nd::WrapInterp<Interp::NEAREST_FAST>{});
            case Interp::LINEAR:             return launch_scale(nd::WrapInterp<Interp::LINEAR>{});
            case Interp::LINEAR_FAST:        return launch_scale(nd::WrapInterp<Interp::LINEAR_FAST>{});
            case Interp::CUBIC:              return launch_scale(nd::WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_FAST:         return launch_scale(nd::WrapInterp<Interp::CUBIC_FAST>{});
            case Interp::CUBIC_BSPLINE:      return launch_scale(nd::WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::CUBIC_BSPLINE_FAST: return launch_scale(nd::WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
            case Interp::LANCZOS4:           return launch_scale(nd::WrapInterp<Interp::LANCZOS4>{});
            case Interp::LANCZOS6:           return launch_scale(nd::WrapInterp<Interp::LANCZOS6>{});
            case Interp::LANCZOS8:           return launch_scale(nd::WrapInterp<Interp::LANCZOS8>{});
            case Interp::LANCZOS4_FAST:      return launch_scale(nd::WrapInterp<Interp::LANCZOS4_FAST>{});
            case Interp::LANCZOS6_FAST:      return launch_scale(nd::WrapInterp<Interp::LANCZOS6_FAST>{});
            case Interp::LANCZOS8_FAST:      return launch_scale(nd::WrapInterp<Interp::LANCZOS8_FAST>{});
        }
    }

    template<nf::Layout REMAP, typename Index, bool IS_GPU,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename Scale, typename Rotate>
    void launch_extract_central_slices_3d(
        Input&& volume, InputWeight&& volume_weight, const Shape4<i64>& volume_shape,
        Output&& slice, OutputWeight&& slice_weight, const Shape4<i64>& slice_shape,
        Scale&& scaling, Rotate&& rotation, const auto& options
    ) {
        constexpr auto accessor_config = nd::AccessorConfig<3>{
            .enforce_restrict = true,
            .allow_empty = true,
            .filter = {0, 2, 3},
        };
        auto slice_accessor = nd::to_accessor<accessor_config, Index>(slice);
        auto slice_weight_accessor = nd::to_accessor<accessor_config, Index>(slice_weight);
        auto batched_rotation = nd::to_batched_transform(rotation);

        const auto s_slice_shape = slice_shape.as<Index>();
        const auto s_volume_shape = volume_shape.as<Index>();
        const auto s_target_shape = options.target_shape.template as<Index>();

        auto launch = [&](auto no_ews_and_scale, auto interp) {
            using coord_t = nt::value_type_twice_t<Rotate>;
            auto volume_interpolator = fourier_projection_to_interpolator
                <3, REMAP, IS_GPU, interp(), coord_t>(volume, s_volume_shape);
            auto volume_weight_interpolator = fourier_projection_to_interpolator
                <3, REMAP, IS_GPU, interp(), coord_t>(volume_weight, s_volume_shape);

            auto batched_scale = nd::to_batched_transform<true, no_ews_and_scale()>(scaling);
            auto ews = fourier_projection_to_ews<no_ews_and_scale(), coord_t>(options.ews_radius);

            using op_t = FourierExtract<
                REMAP, Index,
                decltype(batched_scale), decltype(batched_rotation), decltype(ews),
                decltype(volume_interpolator), decltype(volume_weight_interpolator),
                decltype(slice_accessor), decltype(slice_weight_accessor)>;
            auto op = op_t(
                volume_interpolator, volume_weight_interpolator, s_volume_shape,
                slice_accessor, slice_weight_accessor, s_slice_shape,
                batched_scale, batched_rotation,
                static_cast<coord_t>(options.w_windowed_sinc.fftfreq_sinc),
                static_cast<coord_t>(options.w_windowed_sinc.fftfreq_blackman),
                static_cast<coord_t>(options.fftfreq_cutoff),
                s_target_shape, ews);

            if (op.windowed_sinc_size() > 1) {
                if constexpr (nt::empty<OutputWeight>)
                    ewise({}, wrap(slice), Zero{});
                else
                    ewise({}, wrap(slice, slice_weight), Zero{});

                const auto iwise_shape = s_slice_shape.template set<1>(op.windowed_sinc_size()).rfft();
                iwise(iwise_shape, volume.device(), op,
                      std::forward<Input>(volume), std::forward<InputWeight>(volume_weight),
                      std::forward<Output>(slice), std::forward<OutputWeight>(slice_weight),
                      std::forward<Scale>(scaling), std::forward<Rotate>(rotation));
            } else {
                const auto iwise_shape = s_slice_shape.filter(0, 2, 3).rfft();
                iwise(iwise_shape, volume.device(), op,
                      std::forward<Input>(volume), std::forward<InputWeight>(volume_weight),
                      std::forward<Output>(slice), std::forward<OutputWeight>(slice_weight),
                      std::forward<Scale>(scaling), std::forward<Rotate>(rotation));
            }
        };

        const auto has_ews = any(options.ews_radius != 0);
        const bool has_scale = fourier_project_has_scale(scaling);
        auto launch_scale = [&] (auto interp){
            if (has_ews or has_scale)
                return launch(WrapNoEwaldAndScale<false>{}, interp);
            return launch(WrapNoEwaldAndScale<true>{}, interp);
        };

        const Interp interp = details::fourier_insert_extract_interp_mode(volume, volume_weight, options.interp);
        switch (interp) {
            case Interp::NEAREST:            return launch_scale(nd::WrapInterp<Interp::NEAREST>{});
            case Interp::NEAREST_FAST:       return launch_scale(nd::WrapInterp<Interp::NEAREST_FAST>{});
            case Interp::LINEAR:             return launch_scale(nd::WrapInterp<Interp::LINEAR>{});
            case Interp::LINEAR_FAST:        return launch_scale(nd::WrapInterp<Interp::LINEAR_FAST>{});
            case Interp::CUBIC:              return launch_scale(nd::WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_FAST:         return launch_scale(nd::WrapInterp<Interp::CUBIC_FAST>{});
            case Interp::CUBIC_BSPLINE:      return launch_scale(nd::WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::CUBIC_BSPLINE_FAST: return launch_scale(nd::WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
            case Interp::LANCZOS4:           return launch_scale(nd::WrapInterp<Interp::LANCZOS4>{});
            case Interp::LANCZOS6:           return launch_scale(nd::WrapInterp<Interp::LANCZOS6>{});
            case Interp::LANCZOS8:           return launch_scale(nd::WrapInterp<Interp::LANCZOS8>{});
            case Interp::LANCZOS4_FAST:      return launch_scale(nd::WrapInterp<Interp::LANCZOS4_FAST>{});
            case Interp::LANCZOS6_FAST:      return launch_scale(nd::WrapInterp<Interp::LANCZOS6_FAST>{});
            case Interp::LANCZOS8_FAST:      return launch_scale(nd::WrapInterp<Interp::LANCZOS8_FAST>{});
        }
    }

    template<nf::Layout REMAP, typename Index, bool IS_GPU = false,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename InputScale, typename InputRotate,
             typename OutputScale, typename OutputRotate>
    void launch_insert_and_extract_central_slices_3d(
        Input&& input_slice, InputWeight&& input_weight, const Shape4<i64>& input_shape,
        Output&& output_slice, OutputWeight&& output_weight, const Shape4<i64>& output_shape,
        InputScale&& input_scaling, InputRotate&& input_rotation,
        OutputScale&& output_scaling, OutputRotate&& output_rotation,
        const auto& options
    ) {
        constexpr auto output_config = nd::AccessorConfig<3>{
            .enforce_restrict = true,
            .allow_empty = true,
            .filter = {0, 2, 3},
        };
        auto output_slice_accessor = nd::to_accessor<output_config, Index>(output_slice);
        auto output_weight_accessor = nd::to_accessor<output_config, Index>(output_weight);
        auto input_rotation_accessor = nd::to_batched_transform(input_rotation);
        auto output_rotation_accessor = nd::to_batched_transform(output_rotation);

        const auto s_input_shape = input_shape.as<Index>();
        const auto s_output_shape = output_shape.as<Index>();

        auto launch = [&](auto no_ews_and_scale, auto interp) {
            using coord_t = nt::value_type_twice_t<InputRotate>;
            auto input_interpolator = fourier_projection_to_interpolator
                <2, REMAP, IS_GPU, interp(), coord_t>(input_slice, s_input_shape);
            auto input_weight_interpolator = fourier_projection_to_interpolator
                <2, REMAP, IS_GPU, interp(), coord_t>(input_weight, s_input_shape);

            auto ews = fourier_projection_to_ews<no_ews_and_scale(), coord_t>(options.ews_radius);
            auto input_scaling_accessor = nd::to_batched_transform<true, no_ews_and_scale()>(input_scaling);
            auto output_scaling_accessor = nd::to_batched_transform<true, no_ews_and_scale()>(output_scaling);

            using op_t = FourierInsertExtract<
                REMAP, Index,
                decltype(input_scaling_accessor), decltype(input_rotation_accessor),
                decltype(output_scaling_accessor), decltype(output_rotation_accessor), decltype(ews),
                decltype(input_interpolator), decltype(input_weight_interpolator),
                decltype(output_slice_accessor), decltype(output_weight_accessor)>;
            auto op = op_t(
                input_interpolator, input_weight_interpolator, s_input_shape,
                output_slice_accessor, output_weight_accessor, s_output_shape,
                input_scaling_accessor, input_rotation_accessor,
                output_scaling_accessor, output_rotation_accessor,
                static_cast<coord_t>(options.input_windowed_sinc.fftfreq_sinc),
                static_cast<coord_t>(options.input_windowed_sinc.fftfreq_blackman),
                static_cast<coord_t>(options.w_windowed_sinc.fftfreq_sinc),
                static_cast<coord_t>(options.w_windowed_sinc.fftfreq_blackman),
                static_cast<coord_t>(options.fftfreq_cutoff),
                options.add_to_output, options.correct_weights, ews);

            if (op.is_iwise_4d()) {
                check(not options.correct_weights);
                if (not options.add_to_output) {
                    if constexpr (nt::empty<OutputWeight>)
                        ewise({}, wrap(output_slice), Zero{});
                    else
                        ewise({}, wrap(output_slice, output_weight), Zero{});
                }
                iwise(s_output_shape.template set<1>(op.output_window_size()).rfft(), output_slice.device(), op,
                      std::forward<Input>(input_slice),
                      std::forward<InputWeight>(input_weight),
                      std::forward<Output>(output_slice),
                      std::forward<OutputWeight>(output_weight),
                      std::forward<InputScale>(input_scaling),
                      std::forward<InputRotate>(input_rotation),
                      std::forward<OutputScale>(output_scaling),
                      std::forward<OutputRotate>(output_rotation));
            } else {
                iwise(s_output_shape.filter(0, 2, 3).rfft(), output_slice.device(), op,
                      std::forward<Input>(input_slice),
                      std::forward<InputWeight>(input_weight),
                      std::forward<Output>(output_slice),
                      std::forward<OutputWeight>(output_weight),
                      std::forward<InputScale>(input_scaling),
                      std::forward<InputRotate>(input_rotation),
                      std::forward<OutputScale>(output_scaling),
                      std::forward<OutputRotate>(output_rotation));
            }
        };

        const auto has_ews = any(options.ews_radius != 0);
        const bool has_scale = fourier_project_has_scale(input_scaling) or fourier_project_has_scale(output_scaling);
        auto launch_scale = [&](auto interp) {
            if (has_ews or has_scale)
                return launch(WrapNoEwaldAndScale<false>{}, interp);
            return launch(WrapNoEwaldAndScale<true>{}, interp);
        };

        const Interp interp = details::fourier_insert_extract_interp_mode(input_slice, input_weight, options.interp);
        switch (interp) {
            case Interp::NEAREST:            return launch_scale(nd::WrapInterp<Interp::NEAREST>{});
            case Interp::NEAREST_FAST:       return launch_scale(nd::WrapInterp<Interp::NEAREST_FAST>{});
            case Interp::LINEAR:             return launch_scale(nd::WrapInterp<Interp::LINEAR>{});
            case Interp::LINEAR_FAST:        return launch_scale(nd::WrapInterp<Interp::LINEAR_FAST>{});
            case Interp::CUBIC:              return launch_scale(nd::WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_FAST:         return launch_scale(nd::WrapInterp<Interp::CUBIC_FAST>{});
            case Interp::CUBIC_BSPLINE:      return launch_scale(nd::WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::CUBIC_BSPLINE_FAST: return launch_scale(nd::WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
            case Interp::LANCZOS4:           return launch_scale(nd::WrapInterp<Interp::LANCZOS4>{});
            case Interp::LANCZOS6:           return launch_scale(nd::WrapInterp<Interp::LANCZOS6>{});
            case Interp::LANCZOS8:           return launch_scale(nd::WrapInterp<Interp::LANCZOS8>{});
            case Interp::LANCZOS4_FAST:      return launch_scale(nd::WrapInterp<Interp::LANCZOS4_FAST>{});
            case Interp::LANCZOS6_FAST:      return launch_scale(nd::WrapInterp<Interp::LANCZOS6_FAST>{});
            case Interp::LANCZOS8_FAST:      return launch_scale(nd::WrapInterp<Interp::LANCZOS8_FAST>{});
        }
    }
}

namespace noa::geometry {
    struct RasterizeCentralSlicesOptions {
        /// Frequency cutoff of the output volume, in cycle/pix.
        /// Frequencies above this are left unchanged.
        f64 fftfreq_cutoff{0.5};

        /// Actual BDHW logical shape of the 3d volume.
        /// The function normalizes the slice and volume dimensions, and works with normalized frequencies.
        /// As such, if the volume is larger than the slices, the slices are implicitly stretched (over-sampling case).
        /// Similarly, if the volume is smaller than the slices, the slices are shrunk (under-sampling case).
        /// This parameter specifies the size of the volume onto which the slice frequencies should be mapped against.
        /// By default, i.e. empty target_shape or target_shape == volume_shape, the slice frequencies are mapped onto
        /// the volume frequencies, as mentioned above.
        Shape4<i64> target_shape{};

        /// HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
        /// If negative, the negative curve is computed. If {0,0}, the slices are projections.
        /// To have both left and right beams assigned to different values, the function only computes one
        /// "side" of the EWS, as specified by ews_radius. To insert the other side, one has to call the function
        /// a second time with ews_radius * -1.
        /// Note that the scaling and the rotation are kept separated from one another in order to properly compute the
        /// curve of the Ewald sphere. Indeed, the scaling is applied first to correct for magnification, so that the
        /// EWS is computed using the original frequencies (from the scattering) and is therefore spherical even
        /// under anisotropic magnification. If ews_radius is 0, the scaling factors can be merged with the rotation
        /// matrices.
        Vec2<f64> ews_radius{};
    };

    /// Inserts 2d Fourier central-slice(s) into a 3d Fourier volume, using tri-linear rasterization.
    /// \details Fourier-insertion using rasterization to insert central-slices in a volume.
    ///          This method is mostly used for cases with a lot of central-slices, where errors can be averaged-out.
    ///          Indeed, this method is not very accurate as central-slices are modeled using a simple trilinear-pulse
    ///          for rasterization.
    ///
    /// \tparam REMAP               Remapping. Should be HX2HX.
    /// \tparam Input               A varray|value of (const) f32|f64|c32|c64.
    /// \tparam InputWeight         A varray|value of (const) f32|f64, or Empty.
    /// \tparam Output              VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight        VArray of type f32|f64, or Empty.
    /// \tparam Scale               Mat22, a varray of Mat22, or Empty.
    /// \tparam Rotate              Mat33|Quaternion, or a varray of this type.
    ///                             Sets the floating-point precision of the transformation.
    ///
    /// \param[in] slice            2d-rfft central-slice(s) to insert (can be a constant value).
    /// \param[in] slice_weight     Optional weights associated with slice. Default to ones.
    /// \param slice_shape          BDHW logical shape of slice.
    /// \param[out] volume          3d-rfft volume inside which the slices are inserted.
    /// \param[out] volume_weight   Optional weights associated with volume.
    /// \param volume_shape         BDHW logical shape of volume.
    /// \param[in] inv_scaling      2x2 HW inverse scaling matrix to apply to the slices before the rotation.
    /// \param[in] fwd_rotation     3x3 DHW forward rotation-matrices or quaternions to apply to the slices.
    /// \param options              Insertion options.
    template<nf::Layout REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename Scale = Empty, typename Rotate>
    requires (details::fourier_projection_input_output<false, true, Input, Output, InputWeight, OutputWeight> and
              details::fourier_projection_transform<Scale, Rotate> and
              REMAP.is_hx2hx())
    void rasterize_central_slices_3d(
        Input&& slice,
        InputWeight&& slice_weight,
        const Shape4<i64>& slice_shape,
        Output&& volume,
        OutputWeight&& volume_weight,
        const Shape4<i64>& volume_shape,
        Scale&& inv_scaling,
        Rotate&& fwd_rotation,
        const RasterizeCentralSlicesOptions& options = {}
    ) {
        details::fourier_projection_check_parameters<details::FourierProjectionType::INSERT_RASTERIZE>(
            slice, slice_weight, slice_shape, volume, volume_weight, volume_shape,
            options.target_shape, inv_scaling, fwd_rotation);

        if (volume.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            check(details::fourier_projection_is_i32_safe_access(slice, slice_weight, volume, volume_weight),
                  "i64 indexing not instantiated for GPU devices");
            return details::launch_rasterize_central_slices_3d<REMAP, i32>(
                std::forward<Input>(slice), std::forward<InputWeight>(slice_weight), slice_shape,
                std::forward<Output>(volume), std::forward<OutputWeight>(volume_weight), volume_shape,
                std::forward<Scale>(inv_scaling), std::forward<Rotate>(fwd_rotation), options);
            #else
            panic_no_gpu_backend();
            #endif
        }

        details::launch_rasterize_central_slices_3d<REMAP, i64>(
            std::forward<Input>(slice), std::forward<InputWeight>(slice_weight), slice_shape,
            std::forward<Output>(volume), std::forward<OutputWeight>(volume_weight), volume_shape,
            std::forward<Scale>(inv_scaling), std::forward<Rotate>(fwd_rotation), options);
    }

    /// Settings for the windowed-sinc convolution of the central-slice.\n
    /// \b Fourier-insertion: Central slices are inserted in a (virtual) volume. This parameter defines
    /// the windowed-sinc that is convolved along the normal of the perfectly thin slice(s) to insert.\n
    /// \b Fourier-extraction: Central slices are extracted from a (virtual) volume. This parameter defines the
    /// windowed-sinc that is convolved, along the z of the reconstruction, with the perfectly thin slice(s) to
    /// extract. This is used to effectively apply a horizontal (smooth) rectangular mask centered on the object
    /// _before_ the forward projection. The current API doesn't allow changing the orientation of this sinc
    /// (it is always along z) since its only purpose was originally to improve projections from tomograms by
    /// masking out the noise from above and below the sample.
    struct WindowedSinc {
        /// Frequency, in cycle/pix, of the first zero of the sinc.
        /// This is clamped to ensure a minimum of 1 pixel diameter,
        /// which is usually what we want for the Fourier insertion.
        f64 fftfreq_sinc{-1};

        /// Frequency, in cycle/pix, where the blackman window stops (weight is 0 at this frequency).
        /// This parameter is usually here to control the accuracy-performance ratio, but it can also be used
        /// to control the smoothness of the corresponding real-space mask, in which case it is usually a multiple
        /// of the sinc-cutoff. The larger this multiple, the sharper the real-space step window, but the slower it
        /// is to compute the slice.
        /// This is clamped to ensure the window stops at least to the first sinc-cutoff.
        /// So if both frequencies are left to their default value (-1), a 1 pixel thick slice
        /// is generated, which is usually what we want for Fourier insertion.
        f64 fftfreq_blackman{-1};
    };

    struct InsertCentralSlicesOptions {
        /// Interpolation method.
        /// This is ignored if the input(_weights) is a texture.
        Interp interp{Interp::LINEAR};

        /// Windowed-sinc convolved along the normal of the input slice(s).
        WindowedSinc windowed_sinc{};

        /// Frequency cutoff in the output volume, in cycle/pix.
        /// The frequencies above this are left unchanged.
        f64 fftfreq_cutoff{0.5};

        /// Actual BDHW logical shape of the 3d volume.
        /// See RasterizeCentralSlicesOptions for more details.
        Shape4<i64> target_shape{};

        /// HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
        /// See RasterizeCentralSlicesOptions for more details.
        Vec2<f64> ews_radius{};
    };

    /// Fourier-insertion using 2d-interpolation to insert central-slices in the volume.
    /// \details This method is the more accurate than rasterizationm but can be slower. Here, instead of
    ///          calling every pixel in the central-slices for rasterization, every voxel in the volume is sampled
    ///          by collecting the contribution of every central-slice for each output voxel. The advantage is that
    ///          it allows to use a more accurate model for the central-slices, i.e., a windowed-sinc. Indeed, slices
    ///          are now effectively convolved with a windowed-sinc (both the sinc frequency and window size can be
    ///          controlled) along their normal before the insertion. Note that this (windowed) sinc translates to
    ///          a (smooth) rectangular mask in real-space, along the normal of the slice (an interesting property
    ///          that can be useful for some applications).
    /// \warning This function computes the inverse transformation compared to the overload above using rasterization.
    ///
    /// \tparam REMAP               Remapping. Should be HX2HX.
    /// \tparam Input               A varray|texture of (const) f32|f64|c32|c64.
    /// \tparam InputWeight         A varray|texture of (const) f32|f64, or Empty.
    /// \tparam Output              VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight        VArray of type f32|f64, or Empty.
    /// \tparam Scale               Mat22, a varray of Mat22, or Empty.
    /// \tparam Rotate              Mat33|Quaternion, or a varray of this type.
    ///                             Sets the floating-point precision of the transformation.
    ///
    /// \param[in] slice            2d-rFFT central-slice(s) to insert.
    /// \param[in] slice_weight     Optional weights associated with slice. Default to ones.
    /// \param slice_shape          BDHW logical shape of slice.
    /// \param[out] volume          3d-rFFT volume inside which the slices are to be inserted.
    /// \param[out] volume_weight   Optional weights associated with volume.
    /// \param volume_shape         BDHW logical shape of volume.
    /// \param[in] fwd_scaling      2x2 HW forward scaling matrix to apply to the slices before the rotation.
    /// \param[in] inv_rotation     3x3 DHW inverse rotation-matrices or quaternions to apply to the slices.
    /// \param options              Insertion options.
    template<nf::Layout REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename Scale = Empty, typename Rotate>
    requires (details::fourier_projection_input_output<true, true, Input, Output, InputWeight, OutputWeight> and
              details::fourier_projection_transform<Scale, Rotate> and
              REMAP.is_hx2hx())
    void insert_central_slices_3d(
        Input&& slice,
        InputWeight&& slice_weight,
        const Shape4<i64>& slice_shape,
        Output&& volume,
        OutputWeight&& volume_weight,
        const Shape4<i64>& volume_shape,
        Scale&& fwd_scaling,
        Rotate&& inv_rotation,
        const InsertCentralSlicesOptions& options = {}
    ) {
        details::fourier_projection_check_parameters<details::FourierProjectionType::INSERT_INTERPOLATE>(
            slice, slice_weight, slice_shape, volume, volume_weight, volume_shape,
            options.target_shape, fwd_scaling, inv_rotation);

        if (volume.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (
                (nt::texture_decay<Input> and nt::any_of<nt::mutable_value_type_t<Input>, f64, c64>) or
                (nt::texture_decay<InputWeight> and nt::any_of<nt::mutable_value_type_t<InputWeight>, f64, c64>)) {
                std::terminate(); // unreachable
            } else {
                check(details::fourier_projection_is_i32_safe_access(slice, slice_weight, volume, volume_weight),
                      "i64 indexing not instantiated for GPU devices");
                return details::launch_insert_central_slices_3d<REMAP, i32, true>(
                    std::forward<Input>(slice), std::forward<InputWeight>(slice_weight), slice_shape,
                    std::forward<Output>(volume), std::forward<OutputWeight>(volume_weight), volume_shape,
                    std::forward<Scale>(fwd_scaling), std::forward<Rotate>(inv_rotation), options);
            }
            #else
            panic_no_gpu_backend();
            #endif
        }

        details::launch_insert_central_slices_3d<REMAP, i64, false>(
            std::forward<Input>(slice), std::forward<InputWeight>(slice_weight), slice_shape,
            std::forward<Output>(volume), std::forward<OutputWeight>(volume_weight), volume_shape,
            std::forward<Scale>(fwd_scaling), std::forward<Rotate>(inv_rotation), options);
    }

    struct ExtractCentralSlicesOptions {
        /// Interpolation method.
        /// This is ignored if the input(_weights) is a texture.
        Interp interp{Interp::LINEAR};

        /// Windowed-sinc convolved along the w (the depth) of the Fourier volume.
        /// This is used to apply a real-space step mask along the z (the depth) of the reconstruction
        /// before the forward projection.
        WindowedSinc w_windowed_sinc{};

        /// Frequency cutoff in the volume, in cycle/pix.
        /// The frequencies above this cutoff are assumed to be zero.
        f64 fftfreq_cutoff{0.5};

        /// Actual BDHW logical shape of the 3d volume.
        /// See RasterizeCentralSlicesOptions for more details.
        Shape4<i64> target_shape{};

        /// HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
        /// See RasterizeCentralSlicesOptions for more details.
        Vec2<f64> ews_radius{};
    };

    /// Extracts 2d central-slice(s) from a volume.
    /// \details This is the reverse operation of the Fourier insertion. There are two main behaviors (both
    ///          controlled by the \p options.w_windowed_sinc parameter):
    ///          1. (default) A simple and fast slice extraction, where every pixel of the output slice(s) are sampled
    ///          from the volume using 3d-interpolation.
    ///          2. A w-windowed-sinc slice extraction. This is similar, but instead of simply extracting the slice
    ///          from the volume, it convolves the volume with a 1d windowed-sinc along the w-axis of the volume.
    ///          Note that the convolution is simplified to a simple per-slice weighted-mean along the w-axis of the
    ///          volume. This windowed-sinc convolution translates to a (smooth) centered rectangular mask along the
    ///          z-axis of the reconstruction.
    ///
    /// \tparam REMAP               Remapping. Should be HX2HX.
    /// \tparam Input               A varray|texture of (const) f32|f64|c32|c64.
    /// \tparam InputWeight         A varray|texture of (const) f32|f64, or Empty.
    /// \tparam Output              VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight        VArray of type f32|f64, or Empty.
    /// \tparam Scale               Mat22, a varray of Mat22, or Empty.
    /// \tparam Rotate              Mat33|Quaternion, or a varray of this type.
    ///                             Sets the floating-point precision of the transformation.
    ///
    /// \param[in] volume           3d-rFFT volume from which to extract the slices.
    /// \param[in] volume_weight    Optional weights associated with volume.
    /// \param volume_shape         BDHW logical shape of volume.
    /// \param[out] slice           2d-rFFT central-slice(s) to extract.
    /// \param[out] slice_weight    Optional weights associated with slice.
    /// \param slice_shape          BDHW logical shape of slice.
    /// \param[in] inv_scaling      2x2 HW inverse scaling to apply to the slices before the rotation.
    /// \param[in] fwd_rotation     3x3 DHW forward rotation-matrices or quaternions to apply to the slices.
    /// \param options              Extraction options.
    template<nf::Layout REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename Scale = Empty, typename Rotate>
    requires (details::fourier_projection_input_output<true, false, Input, Output, InputWeight, OutputWeight> and
              details::fourier_projection_transform<Scale, Rotate> and
              REMAP.is_hx2hx())
    void extract_central_slices_3d(
        Input&& volume,
        InputWeight&& volume_weight,
        const Shape4<i64>& volume_shape,
        Output&& slice,
        OutputWeight&& slice_weight,
        const Shape4<i64>& slice_shape,
        Scale&& inv_scaling,
        Rotate&& fwd_rotation,
        const ExtractCentralSlicesOptions& options = {}
    ) {
        details::fourier_projection_check_parameters<details::FourierProjectionType::EXTRACT>(
            volume, volume_weight, volume_shape, slice, slice_weight, slice_shape,
            options.target_shape, inv_scaling, fwd_rotation);

        if (volume.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (
                (nt::texture_decay<Input> and nt::any_of<nt::mutable_value_type_t<Input>, f64, c64>) or
                (nt::texture_decay<InputWeight> and nt::any_of<nt::mutable_value_type_t<InputWeight>, f64, c64>)) {
                std::terminate(); // unreachable
            } else {
                check(details::fourier_projection_is_i32_safe_access(slice, slice_weight, volume, volume_weight),
                      "i64 indexing not instantiated for GPU devices");
                return details::launch_extract_central_slices_3d<REMAP, i32, true>(
                    std::forward<Input>(volume), std::forward<InputWeight>(volume_weight), volume_shape,
                    std::forward<Output>(slice), std::forward<OutputWeight>(slice_weight), slice_shape,
                    std::forward<Scale>(inv_scaling), std::forward<Rotate>(fwd_rotation), options);
            }
            #else
            panic_no_gpu_backend();
            #endif
        }

        details::launch_extract_central_slices_3d<REMAP, i64, false>(
            std::forward<Input>(volume), std::forward<InputWeight>(volume_weight), volume_shape,
            std::forward<Output>(slice), std::forward<OutputWeight>(slice_weight), slice_shape,
            std::forward<Scale>(inv_scaling), std::forward<Rotate>(fwd_rotation), options);
    }

    struct InsertAndExtractCentralSlicesOptions {
        /// Interpolation method.
        /// This is ignored if input(_weights) is a texture.
        Interp interp{Interp::LINEAR};

        /// Windowed-sinc convolved along the normal of the input slice(s).
        WindowedSinc input_windowed_sinc{};

        /// Windowed-sinc convolved along the w (the depth) of the virtual Fourier volume.
        /// This is used to apply a real-space step mask along the z (the depth) of the reconstruction
        /// before the forward projection.
        WindowedSinc w_windowed_sinc{};

        /// Whether the contribution of the input slices should be added to the output.
        /// By default, the operator sets the output_{slice|weight}. With this option enabled, it instead adds the
        /// contribution of input_{slice|weight} to the signal already present in the output_{slice|weight}, allowing
        /// to progressively build the output_{slice|weight}.
        bool add_to_output{false};

        /// Correct for the weights, i.e. divide the output sampled values by their corresponding weight.
        /// This is the equivalent of doing `output_slice/max(1, output_weight)` after the function and assumes all
        /// the input slices are included in the same function call. It is not compatible with add_to_output=true or
        /// with a non-default w_windowed_sinc.
        /// This can be useful for cases where there's no real-space mask to apply before the forward projection and
        /// if the weights are to be applied right away and not used for anything else (in which case output_weight
        /// doesn't need to be saved and can be left empty).
        bool correct_weights{false};

        /// Frequency cutoff of the virtual 3d volume, in cycle/pix.
        /// The frequencies above this cutoff are assumed to be zero.
        f64 fftfreq_cutoff{0.5};

        /// HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
        /// See RasterizeCentralSlicesOptions for more details.
        Vec2<f64> ews_radius{};
    };

    /// Extracts 2d central-slice(s) from a virtual volume filled by other central-slices.
    /// \details This function effectively combines the insertion and extraction, but instead of actually inserting
    ///          slices into a volume, it directly inserts them in the extracted slices. In other words, it builds a
    ///          virtual volume, made of central-slices, and this virtual volume is then sampled at (exactly) the
    ///          frequency of the central-slices to extract. This has major performance benefits because it only
    ///          samples the frequency of the output slices and never allocates/reconstructs the volume. It is also
    ///          more accurate since the volume is never actually discretized (thus skipping an interpolation step).
    ///          Note that these performance benefits are expected to disappear if thousands (possibly hundreds?) of
    ///          slices are extracted. Indeed, for every output slice, the operator needs to sample the volume by
    ///          collecting the signal of every input slice using 2d-interpolation. This is as opposed to the other
    ///          extract method, where the volume is already sampled, making the extraction much cheaper (and constant
    ///          cost: it's a simple 3d-interpolation).
    ///
    /// \tparam REMAP                   Remapping. Should be HX2HX.
    /// \tparam Input                   A varray|texture|value of (const) f32|f64|c32|c64.
    /// \tparam InputWeight             A varray|texture|value of (const) f32|f64, or Empty.
    /// \tparam Output                  VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight            VArray of type f32|f64, or Empty.
    /// \tparam InputScale              Mat22 or a varray of this type, or Empty
    /// \tparam InputRotate             Mat33|Quaternion, or a varray of this type.
    /// \tparam OutputScale             Mat22 or a varray of this type, or Empty
    /// \tparam OutputRotate            Mat33|Quaternion, or a varray of this type.
    ///
    /// \param[in] input_slice          2d central-slice(s) to insert.
    /// \param[in] input_weight         Optional weights associated with input_slice. Defaults to ones.
    /// \param input_slice_shape        BDHW logical shape of input_slice.
    /// \param[in,out] output_slice     2d central-slice(s) to extract. See options.add_to_output.
    /// \param[in,out] output_weight    Optional weights associated with output_slice.
    /// \param output_slice_shape       BDHW logical shape of output_slice.
    /// \param[in] input_fwd_scaling    2x2 HW forward scaling matrices to apply to the input slices before the rotation.
    /// \param[in] input_inv_rotation   3x3 DHW inverse rotation-matrices or quaternions to apply to the input slices.
    /// \param[in] output_inv_scaling   2x2 HW inverse scaling matrix to apply to the output slices before the rotation.
    /// \param[in] output_fwd_rotation  3x3 DHW forward rotation-matrices or quaternions to apply to the output slices.
    /// \param options                  Operator options.
    template<nf::Layout REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename InputScale = Empty, typename InputRotate,
             typename OutputScale = Empty, typename OutputRotate>
    requires (details::fourier_projection_input_output<true, true, Input, Output, InputWeight, OutputWeight> and
              details::fourier_projection_transform<InputScale, InputRotate> and
              details::fourier_projection_transform<OutputScale, OutputRotate> and
              REMAP.is_hx2hx())
    void insert_and_extract_central_slices_3d(
        Input&& input_slice,
        InputWeight&& input_weight,
        const Shape4<i64>& input_slice_shape,
        Output&& output_slice,
        OutputWeight&& output_weight,
        const Shape4<i64>& output_slice_shape,
        InputScale&& input_fwd_scaling,
        InputRotate&& input_inv_rotation,
        OutputScale&& output_inv_scaling,
        OutputRotate&& output_fwd_rotation,
        const InsertAndExtractCentralSlicesOptions& options = {}
    ) {
        details::fourier_projection_check_parameters<details::FourierProjectionType::INSERT_EXTRACT>(
            input_slice, input_weight, input_slice_shape, output_slice, output_weight, output_slice_shape,
            {}, input_fwd_scaling, input_inv_rotation, output_inv_scaling, output_fwd_rotation);

        using coord_t = nt::value_type_twice_t<OutputRotate>;
        const auto volume_z = static_cast<coord_t>(min(output_slice_shape.filter(2, 3)));
        const auto fftfreq_blackman = static_cast<coord_t>(options.w_windowed_sinc.fftfreq_blackman);
        const auto w_blackman_size = details::blackman_window_size<i64>(fftfreq_blackman, volume_z);
        check(not options.correct_weights or (not options.add_to_output and w_blackman_size == 1),
              "options.correct_weights=true is not compatible with "
              "options.add_to_output=true and options.w_windowed_sinc.fftfreq_blackman={} (={} pixels)",
              options.w_windowed_sinc.fftfreq_blackman, w_blackman_size);

        if (output_slice.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (
                (nt::texture_decay<Input> and nt::any_of<nt::mutable_value_type_t<Input>, f64, c64>) or
                (nt::texture_decay<InputWeight> and nt::any_of<nt::mutable_value_type_t<InputWeight>, f64, c64>)) {
                std::terminate(); // unreachable
            } else {
                check(details::fourier_projection_is_i32_safe_access(input_slice, input_weight, output_slice, output_weight),
                      "i64 indexing not instantiated for GPU devices");
                return details::launch_insert_and_extract_central_slices_3d<REMAP, i32, true>(
                    std::forward<Input>(input_slice), std::forward<InputWeight>(input_weight), input_slice_shape,
                    std::forward<Output>(output_slice), std::forward<OutputWeight>(output_weight), output_slice_shape,
                    std::forward<InputScale>(input_fwd_scaling), std::forward<InputRotate>(input_inv_rotation),
                    std::forward<OutputScale>(output_inv_scaling), std::forward<OutputRotate>(output_fwd_rotation),
                    options);
            }
            #else
            panic_no_gpu_backend();
            #endif
        }

        details::launch_insert_and_extract_central_slices_3d<REMAP, i64>(
            std::forward<Input>(input_slice), std::forward<InputWeight>(input_weight), input_slice_shape,
            std::forward<Output>(output_slice), std::forward<OutputWeight>(output_weight), output_slice_shape,
            std::forward<InputScale>(input_fwd_scaling), std::forward<InputRotate>(input_inv_rotation),
            std::forward<OutputScale>(output_inv_scaling), std::forward<OutputRotate>(output_fwd_rotation),
            options);
    }

    /// Corrects for the interpolation kernel applied to Fourier transforms.
    /// \details When interpolating Fourier transforms, we effectively convolve the input Fourier components with
    ///          an interpolation kernel. As such, the resulting iFT of the interpolated output is the product of the
    ///          final wanted output and the iFT of the interpolation kernel. This function corrects for the effect of
    ///          the interpolation kernel in real-space.
    /// \param[in] input        Inverse Fourier transform of the 3d volume used for direct Fourier insertion.
    /// \param[out] output      Corrected output. Can be equal to \p input.
    /// \param interp           Interpolation method.
    /// \param post_correction  Whether the correction is the post- or pre-correction. Post-correction is meant to be
    ///                         applied to the interpolated output, whereas pre-correction is meant to be applied to
    ///                         the input about to be interpolated.
    template<nt::varray_decay_of_almost_any<f32, f64> Input,
             nt::varray_decay_of_any<f32, f64> Output>
    void fourier_interpolation_correction(Input&& input, Output&& output, Interp interp, bool post_correction) {
        /// TODO Add correction for other interpolation methods.
        check(interp.is_almost_any(Interp::LINEAR), "{} is currently not supported", interp);

        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using coord_t = std::conditional_t<nt::any_of<f64, input_value_t, output_value_t>, f64, f32>;
        const auto output_shape = output.shape();
        const auto input_strides = nd::broadcast_strides(input, output);
        const auto input_accessor = Accessor<const input_value_t, 4, i64>(input.get(), input_strides.template as<i64>());
        const auto output_accessor = Accessor<output_value_t, 4, i64>(output.get(), output.strides().template as<i64>());

        if (post_correction) {
            const auto op = details::GriddingCorrection<true, coord_t, decltype(input_accessor), decltype(output_accessor)>(
                input_accessor, output_accessor, output_shape);
            iwise(output_shape, output.device(), op,
                  std::forward<Input>(input), std::forward<Output>(output));
        } else {
            const auto op = details::GriddingCorrection<false, coord_t, decltype(input_accessor), decltype(output_accessor)>(
                input_accessor, output_accessor, output_shape);
            iwise(output_shape, output.device(), op,
                  std::forward<Input>(input), std::forward<Output>(output));
        }
    }
}
