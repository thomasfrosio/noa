#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/geometry/Interpolate.hpp"
#include "noa/core/geometry/Quaternion.hpp"

namespace noa::geometry::guts {
    template<typename Scale, typename Coord, typename Value = nt::value_type_t<Scale>>
    constexpr bool is_valid_fourier_scaling_v =
            std::is_empty_v<Scale> or
            (nt::is_accessor_nd_v<Scale, 1> and nt::is_any_v<Value, Mat22<Coord>>);

    template<typename Rotate, typename Matrix = nt::value_type_t<Rotate>>
    constexpr bool is_valid_fourier_rotate_v =
            nt::is_accessor_nd_v<Rotate, 1> and (nt::is_mat33_v<Matrix> or nt::is_quaternion_v<Matrix>);

    template<typename Input, typename Output,
             typename InputValue = nt::value_type_t<Input>,
             typename OutputValue = nt::value_type_t<Output>>
    constexpr bool is_valid_fourier_value_type_v =
            nt::are_complex_v<InputValue, OutputValue> or
            nt::are_real_v<InputValue, OutputValue> or
            (nt::is_complex_v<InputValue> and nt::is_real_v<OutputValue>);

    template<typename Input, typename Output,
             typename InputValue = nt::value_type_t<Input>,
             typename OutputValue = nt::value_type_t<Output>>
    constexpr bool is_valid_fourier_weight_type_v =
            nt::are_real_v<InputValue, OutputValue> or
            (std::is_empty_v<Input> and nt::is_real_v<OutputValue>) or
            (std::is_empty_v<Input> and std::is_empty_v<Output>);

    template<typename Coord, typename Rotate, typename Integer>
    NOA_FHD constexpr auto fourier_apply_rotate(
            const Vec3<Coord>& fftfreq_3d,
            const Rotate& rotation, // Accessor(Value)<Mat33|Quaternion>, Mat33|Quaternion
            Integer index
    ) -> Vec3<Coord> {
        if constexpr (nt::is_accessor_nd_v<Rotate, 1>) {
            using rotate_value_t = nt::value_type_t<Rotate>;
            if constexpr (nt::is_mat33_v<rotate_value_t>) {
                return rotation[index] * fftfreq_3d;
            } else if constexpr (nt::is_quaternion_v<rotate_value_t>) {
                return rotation[index].rotate(fftfreq_3d);
            } else {
                static_assert(nt::always_false_v<Rotate>);
            }
        } else if constexpr (nt::is_mat33_v<Rotate>) {
            return rotation * fftfreq_3d;
        } else if constexpr (nt::is_quaternion_v<Rotate>) {
            return rotation.rotate(fftfreq_3d);
        } else {
            static_assert(nt::always_false_v<Rotate>);
        }
    }

    template<typename Coord, typename ScaleOrEmpty, typename Integer>
    NOA_FHD constexpr auto fourier_apply_scaling(
            const Vec2<Coord>& fftfreq_2d,
            const ScaleOrEmpty& scaling_matrix, // Accessor<Mat22>, AccessorValue<Mat22|Empty>, Mat22|Empty
            Integer index
    ) -> Vec2<Coord> {
        if constexpr (nt::is_accessor_nd_v<ScaleOrEmpty, 1> and nt::is_mat22_v<nt::value_type_t<ScaleOrEmpty>>)
            return scaling_matrix[index] * fftfreq_2d;
        else if constexpr (nt::is_mat22_v<ScaleOrEmpty>)
            return scaling_matrix * fftfreq_2d;
        else if constexpr (std::is_empty_v<ScaleOrEmpty>)
            return fftfreq_2d;
        else
            static_assert(nt::always_false_v<ScaleOrEmpty>);
    }

    // Transforms a 3d fftfreq representing the slice, to its 3d fftfreq in the grid.
    // This is a forward transformation of the frequency, but because it is in Fourier-space,
    // the real-space scaling is inverted.
    template<typename Coord, typename ScaleOrEmpty, typename Rotate, typename Integer, typename EWSOrEmpty>
    NOA_IHD constexpr auto fourier_slice2grid(
            Vec2<Coord> fftfreq,
            const ScaleOrEmpty& inv_scaling, // Accessor<Mat22>, AccessorValue<Mat22|Empty>, Mat22|Empty
            const Rotate& fwd_rotation,      // Accessor(Value)<Mat33|Quaternion>, Mat33|Quaternion
            Integer batch_index,
            EWSOrEmpty inv_ews_diameter      // Vec2|Empty
    ) -> Vec3<Coord> {
        // If we apply the EWS curvature, the scaling factors should be corrected
        // before applying the curvature, and therefore before applying the rotation.
        // That way, we use the correct frequencies to compute the EWS, e.g., resulting
        // in a spherical EWS even under anisotropic magnification.
        fftfreq = fourier_apply_scaling(fftfreq, inv_scaling, batch_index);

        // TODO We use the Small Angle Approximation to compute the EWS curvature,
        //      so the frequency (u,v) is unchanged. Look at the cisTEM implementation
        //      to remove this approximation? RELION shows that even for low voltages
        //      and large boxes, it is probably not worth it though.
        Vec3<Coord> fftfreq_3d{0, fftfreq[0], fftfreq[1]};
        if constexpr (not std::is_empty_v<EWSOrEmpty>)
            fftfreq_3d[0] = sum(inv_ews_diameter * fftfreq * fftfreq);

        return fourier_apply_rotate(fftfreq_3d, fwd_rotation, batch_index);
    }

    // Same as above, but in the other direction.
    template<typename Coord, typename ScaleOrEmpty, typename Rotate, typename Integer, typename EWSOrEmpty>
    NOA_IHD constexpr auto fourier_grid2slice(
            Vec3<Coord> frequency,
            const ScaleOrEmpty& fwd_scaling_matrices, // Accessor<Mat22>, AccessorValue<Mat22|Empty>, Mat22|Empty
            const Rotate& inv_rotation,               // Accessor(Value)<Mat33|Quaternion>, Mat33|Quaternion
            Integer index,
            EWSOrEmpty inv_ews_diameter               // Vec2|Empty
    ) -> Pair<Coord, Vec2<Coord>> {
        frequency = fourier_apply_rotate(frequency, inv_rotation, index);

        Vec2<Coord> freq_2d{frequency[1], frequency[2]};
        Coord freq_z = frequency[0];
        if constexpr (not std::is_empty_v<EWSOrEmpty>)
            freq_z -= sum(inv_ews_diameter * freq_2d * freq_2d);

        // Same reason as for the forward transformation.
        // Here the grid is correct, so rotate the EWS, then compute
        // the curvature and only then we can scale the slice.
        freq_2d = fourier_apply_scaling(freq_2d, fwd_scaling_matrices, index);
        return {freq_z, freq_2d};
    }

    // Interpolates the value at a given frequency, in cycle per pixel.
    template<typename Interpolator, typename Int, typename Coord>
    NOA_HD constexpr auto interpolate_slice_value(
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
    NOA_HD constexpr auto interpolate_grid_value(
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
    NOA_FHD constexpr auto cast_or_power_spectrum(Src value) {
        if constexpr (nt::is_complex_v<Src> and nt::is_real_v<Dst>) {
            return static_cast<Dst>(abs_squared(value));
        } else {
            return static_cast<Dst>(value);
        }
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
    template<typename Int, typename Coord> requires (nt::is_real_v<Coord> and nt::is_int_v<Int>)
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
    template<typename Coord, typename Int> requires (nt::is_real_v<Coord> and nt::is_int_v<Int>)
    constexpr NOA_FHD Coord w_index_to_fftfreq_offset(Int w, Int window_size, Coord spectrum_size) {
        return static_cast<Coord>(w - window_size / 2) / spectrum_size;
    }

    // This is only used for the Fourier extraction step.
    // Compute the sum of the z-window, so that it can be directly applied to the extracted values,
    // thereby correcting for the multiplicity on the fly.
    template<typename Int, typename Real>
    Pair<Int, Real> z_window_spec(Real fftfreq_sinc, Real fftfreq_blackman, Real spectrum_size) {
        auto window_size = blackman_window_size<Int>(fftfreq_blackman, spectrum_size);
        Real sum{0};
        for (Int i = 0; i < window_size; ++i) {
            const auto fftfreq = w_index_to_fftfreq_offset(i, window_size, spectrum_size);
            sum += windowed_sinc(fftfreq, fftfreq_sinc, fftfreq_blackman);
        }
        return {window_size, sum};
    }
}
