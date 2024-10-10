#pragma once

#include "noa/core/types/Shape.hpp"
#include "noa/core/geometry/Quaternion.hpp"
#include "noa/core/geometry/Transform.hpp"

namespace noa::geometry::guts {
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
        rfftfreq_samples = ceil(rfftfreq_samples); // include last fraction
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
}
