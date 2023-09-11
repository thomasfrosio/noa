#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/geometry/Interpolate.hpp"
#include "noa/core/geometry/Quaternion.hpp"
#include "noa/core/utils/Atomic.hpp"

namespace noa::geometry::details {
    using Remap = ::noa::fft::Remap;

    // Transforms a 3d fftfreq representing the slice, to its 3d fftfreq in the grid.
    // This is a forward transformation of the frequency, but because it is in Fourier-space,
    // the real-space scaling is inverted.
    template<typename Coord, typename ScaleOrEmtpy, typename Rotate, typename Integer, typename EWSOrEmpty>
    constexpr NOA_IHD Vec3<Coord>
    transform_slice2grid(
            Vec2<Coord> fftfreq,
            const ScaleOrEmtpy& inv_scaling, // 2x2 matrix
            const Rotate& fwd_rotation, // 3x3
            Integer batch_index,
            EWSOrEmpty inv_ews_diameter
    ) {
        // If we apply the EWS curvature, the scaling factors should be corrected
        // before applying the curvature, and therefore before applying the rotation.
        // That way, we use the correct frequencies to compute the EWS, e.g., resulting
        // in a spherical EWS even under anisotropic magnification.
        if constexpr (std::is_pointer_v<ScaleOrEmtpy> || nt::is_accessor_1d_v<ScaleOrEmtpy>)
            fftfreq = inv_scaling[batch_index] * fftfreq;
        else if constexpr (nt::is_mat22_v<ScaleOrEmtpy>)
            fftfreq = inv_scaling * fftfreq;

        // TODO We use the Small Angle Approximation to compute the EWS curvature,
        //      so the frequency (u,v) is unchanged. Look at the cisTEM implementation
        //      to remove this approximation? RELION shows that even for low voltages
        //      and large boxes, it is probably not worth it though.
        Vec3<Coord> fftfreq_3d{0, fftfreq[0], fftfreq[1]};
        if constexpr (!std::is_empty_v<EWSOrEmpty>)
            fftfreq_3d[0] = noa::math::sum(inv_ews_diameter * fftfreq * fftfreq);

        // TODO C++20 requires{fwd_rotation[batch_index]}
        if constexpr (std::is_pointer_v<Rotate> || nt::is_accessor_1d_v<Rotate>) {
            using rotate_value_t = std::remove_pointer_t<nt::value_type_t<Rotate>>;
            if constexpr (nt::is_mat33_v<rotate_value_t>) {
                return fwd_rotation[batch_index] * fftfreq_3d;
            } else if constexpr (nt::is_quaternion_v<rotate_value_t>) {
                return fwd_rotation[batch_index].rotate(fftfreq_3d);
            } else {
                static_assert(nt::always_false_v<Rotate>);
            }
        } else if constexpr (nt::is_mat33_v<Rotate>) {
            return fwd_rotation * fftfreq_3d;
        } else if constexpr (nt::is_quaternion_v<Rotate>) {
            return fwd_rotation.rotate(fftfreq_3d);
        } else {
            static_assert(nt::always_false_v<Rotate>);
        }
    }

    // Same as above, but in the other direction.
    template<typename Coord, typename ScaleOrEmpty, typename Rotate, typename Int, typename EWSOrEmpty>
    constexpr NOA_IHD Pair<float, Vec2<Coord>>
    transform_grid2slice(
            Vec3<Coord> frequency,
            const ScaleOrEmpty& fwd_scaling_matrices,
            const Rotate& inv_rotation,
            Int index, EWSOrEmpty inv_ews_diameter
    ) {
        if constexpr (std::is_pointer_v<Rotate> || nt::is_accessor_1d_v<Rotate>) {
            using rotate_value_t = std::remove_pointer_t<nt::value_type_t<Rotate>>;
            if constexpr (nt::is_mat33_v<rotate_value_t>)
                frequency = inv_rotation[index] * frequency;
            else  if constexpr (nt::is_quaternion_v<rotate_value_t>)
                frequency = inv_rotation[index].rotate(frequency);
            else
                static_assert(nt::always_false_v<Rotate>);
        } else if constexpr (nt::is_mat33_v<Rotate>)
            frequency = inv_rotation * frequency;
        else if constexpr (nt::is_quaternion_v<Rotate>)
            frequency = inv_rotation.rotate(frequency);
        else
            static_assert(nt::always_false_v<Rotate>);

        Vec2<Coord> freq_2d{frequency[1], frequency[2]};
        Coord freq_z = frequency[0];
        if constexpr (!std::is_empty_v<EWSOrEmpty>)
            freq_z -= noa::math::sum(inv_ews_diameter * freq_2d * freq_2d);

        // Same reason as for the forward transformation.
        // Here the grid is correct, so rotate the EWS, then compute
        // the curvature and only then we can scale the slice.
        if constexpr (std::is_pointer_v<ScaleOrEmpty>)
            freq_2d = fwd_scaling_matrices[index] * freq_2d;
        else if constexpr (nt::is_mat22_v<ScaleOrEmpty>)
            freq_2d = fwd_scaling_matrices * freq_2d;

        return {freq_z, freq_2d};
    }

    // Interpolates the value at a given frequency, in cycle per pixel.
    template<typename Interpolator, typename Int, typename Coord>
    constexpr NOA_HD auto interpolate_slice_value(
            Vec2<Coord> frequency, const Vec2<Coord>& slice_shape, Coord center_y,
            const Interpolator& interpolator, Int batch
    ) {
        using value_t = typename Interpolator::value_type;
        using real_t = nt::value_type_t<value_t>;
        constexpr bool IS_COMPLEX = nt::is_complex_v<value_t>;

        // From the normalized frequency to the multidimensional index.
        real_t conj = 1;
        if (frequency[1] < 0) {
            frequency = -frequency;
            if constexpr (IS_COMPLEX)
                conj = -1;
        }
        frequency *= slice_shape;
        frequency[0] += center_y;

        value_t value = interpolator(frequency, batch);
        if constexpr (IS_COMPLEX)
            value.imag *= conj;
        else
            (void) conj;
        return value;
    }

    // Interpolates the value at a given frequency, in cycle per pixel.
    template<typename Interpolator, typename Coord>
    constexpr NOA_HD auto interpolate_grid_value(
            Vec3<Coord> frequency, const Vec3<Coord>& target_shape,
            const Vec2<Coord>& grid_center_zy, const Interpolator& interpolator
    ) {
        using value_t = typename Interpolator::value_type;
        using real_t = nt::value_type_t<value_t>;
        constexpr bool IS_COMPLEX = nt::is_complex_v<value_t>;

        real_t conj = 1;
        if (frequency[2] < 0) {
            frequency = -frequency;
            if constexpr(IS_COMPLEX)
                conj = -1;
        }
        frequency *= target_shape;
        frequency[0] += grid_center_zy[0];
        frequency[1] += grid_center_zy[1];

        value_t value = interpolator(frequency);
        if constexpr(IS_COMPLEX)
            value.imag *= conj;
        else
            (void) conj;
        return value;
    }

    template<typename Dst, typename Src>
    constexpr NOA_FHD auto cast_or_power_spectrum(Src value) {
        if constexpr (nt::is_complex_v<Src> && nt::is_real_v<Dst>) {
            return static_cast<Dst>(noa::abs_squared_t{}(value));
        } else {
            return static_cast<Dst>(value);
        }
    }

    // Windowed-sinc. This function assumes fftfreq <= fftfreq_blackman,
    // above that the blackman window will start again.
    template<typename Coord>
    NOA_FHD Coord windowed_sinc(Coord fftfreq, Coord fftfreq_sinc, Coord fftfreq_blackman) {
        // https://www.desmos.com/calculator/tu5b8aqg2e
        constexpr Coord PI = noa::math::Constant<Coord>::PI;
        fftfreq *= PI;
        const auto sinc = noa::math::sinc(fftfreq / fftfreq_sinc);
        const auto blackman_cutoff = fftfreq / fftfreq_blackman;
        const auto blackman = static_cast<Coord>(0.42) +
                              static_cast<Coord>(0.5) * noa::math::cos(blackman_cutoff) +
                              static_cast<Coord>(0.08) * noa::math::cos(2 * blackman_cutoff);
        return sinc * blackman;
    }

    // This is only used for the Fourier extraction step.
    // The window is always an odd-numbered size.
    template<typename Int, typename Coord, typename = std::enable_if_t<nt::is_real_v<Coord> && nt::is_int_v<Int>>>
    Int blackman_window_size(Coord fftfreq_blackman, Coord spectrum_size) {
        // Given a blackman window in range [0, fftfreq_blackman] and a spectrum logical-size
        // (the z size in our case), what is the size of the blackman window, in elements.
        // For instance:
        //  spectrum_size=10, fftfreq_blackman=0.23
        //  rfftfreq=[0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        //  rfftfreq_samples=4.6->5, window_size=11
        //  computed_window=[-0.25,-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2,0.25]
        auto rfftfreq_samples = static_cast<f64>(spectrum_size) * static_cast<f64>(fftfreq_blackman);
        rfftfreq_samples = noa::math::ceil(rfftfreq_samples); // include last fraction
        const auto rfftfreq_samples_int = noa::math::max(Int{1}, static_cast<Int>(rfftfreq_samples));
        auto window_size = 2 * (rfftfreq_samples_int) + 1;

        // Truncate the edges because at these indexes, the window is 0, so there's no need to compute it.
        // So using the same example, computed_window=[-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2]
        return window_size - 2;
    }

    // This is only used for the Fourier extraction step.
    // Given the iwise-index w and the blackman window size, return the fftfreq offset. For instance,
    // window_size=9: w=[0..8] -> [-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2]
    template<typename Coord, typename Int, typename = std::enable_if_t<nt::is_real_v<Coord> && nt::is_int_v<Int>>>
    constexpr NOA_FHD Coord w_index_to_fftfreq_offset(Int w, Int window_size, Coord spectrum_size) {
        return static_cast<Coord>(w - window_size / 2) / spectrum_size;
    }

    // This is only used for the Fourier extraction step.
    // Compute the sum of the z-window, so that it can be directly applied to the extracted values,
    // thereby correcting for the multiplicity on the fly.
    template<typename Int, typename Real>
    std::pair<Int, Real> z_window_spec(Real fftfreq_sinc, Real fftfreq_blackman, Real spectrum_size) {
        auto window_size = blackman_window_size<Int>(fftfreq_blackman, spectrum_size);
        Real sum{0};
        for (Int i = 0; i < window_size; ++i) {
            const auto fftfreq = w_index_to_fftfreq_offset(i, window_size, spectrum_size);
            sum += windowed_sinc(fftfreq, fftfreq_sinc, fftfreq_blackman);
        }
        return {window_size, sum};
    }
}
