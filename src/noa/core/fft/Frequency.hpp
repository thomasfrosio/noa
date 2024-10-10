#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa::fft {
    /// Returns the highest normalized frequency (in cycle/pix) that a dimension with a given size can have.
    template<std::floating_point T>
    [[nodiscard]] constexpr T highest_normalized_frequency(nt::integer auto size) noexcept {
        // even: the highest frequency is always 0.5, i.e. Nyquist. E.g. size=64, 32/64=0.5
        // odd: Nyquist cannot be reached. Eg. size=63, 31/63 = 0.49206
        const auto max_index = size / 2; // integer division
        return static_cast<T>(max_index) / static_cast<T>(size);
    }

    /// Returns the highest normalized frequency (in cycle/pix) that a dimension with a given size can have.
    template<std::floating_point T>
    [[nodiscard]] constexpr auto highest_fftfreq(nt::integer auto size) noexcept {
        return highest_normalized_frequency<T>(size);
    }

    /// Returns the fft centered index of the corresponding fft non-centered index.
    /// Should satisfy `0 <= index < size`.
    template<nt::integer T>
    [[nodiscard]] constexpr T fftshift(T index, T size) noexcept {
        // n=10: [0, 1, 2, 3, 4,-5,-4,-3,-2,-1] -> [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4]
        // n=11: [0, 1, 2, 3, 4, 5,-5,-4,-3,-2,-1] -> [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5]
        return (index < (size + 1) / 2) ? index + size / 2 : index - (size + 1) / 2; // or (index + size / 2) % size
    }
    template<nt::vec_integer T>
    [[nodiscard]] constexpr T fftshift(const T& indices, const T& sizes) noexcept {
        T shifted_indices;
        for (size_t i{}; i < T::SIZE; ++i)
            shifted_indices[i] = fftshift(indices[i], sizes[i]);
        return shifted_indices;
    }
    template<nt::integer T, size_t N, size_t A>
    [[nodiscard]] constexpr auto fftshift(
        const Vec<T, N, A>& indices,
        const Shape<T, N, A>& shape
    ) noexcept -> Vec<T, N, A> {
        return fftshift(indices, shape.vec);
    }

    /// Returns the fft non-centered index of the corresponding centered fft index.
    /// Should be within `0 <= index < size`.
    template<nt::integer T>
    [[nodiscard]] constexpr T ifftshift(T index, T size) noexcept {
        // n=10: [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4] -> [0, 1, 2, 3, 4,-5,-4,-3,-2,-1]
        // n=11: [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5] -> [0, 1, 2, 3, 4, 5,-5,-4,-3,-2,-1]
        return (index < size / 2) ? index + (size + 1) / 2 : index - size / 2; // or (index + (size + 1) / 2) % size
    }
    template<nt::vec_integer T>
    [[nodiscard]] constexpr T ifftshift(const T& indices, const T& sizes) noexcept {
        T shifted_indices;
        for (size_t i{}; i < T::SIZE; ++i)
            shifted_indices[i] = ifftshift(indices[i], sizes[i]);
        return shifted_indices;
    }
    template<nt::integer T, size_t N, size_t A>
    [[nodiscard]] constexpr auto ifftshift(
        const Vec<T, N, A>& indices,
        const Shape<T, N, A>& shape
    ) noexcept -> Vec<T, N, A> {
        return ifftshift(indices, shape.vec);
    }

    /// Returns the frequency (in samples) at a given index in range [0, size).
    /// \warning This function is only intended to be used for full dimensions!
    ///          For a rfft's half dimension (which is not supported by this function),
    ///          the index is equal to the frequency, regardless of the centering.
    template<bool IS_CENTERED, nt::sinteger T>
    [[nodiscard]] constexpr T index2frequency(T index, T size) noexcept {
        // n=5: [0, 1, 2, 3, 4] -> centered=[-2,-1, 0, 1, 2], non-centered=[0, 1, 2,-2,-1]
        // n=6: [0, 1, 2, 3, 4, 5] -> centered=[-3,-2,-1, 0, 1, 2], non-centered=[0, 1, 2,-3,-2,-1]
        if constexpr (IS_CENTERED)
            return index - size / 2;
        return index < (size + 1) / 2 ? index : index - size;
    }

    /// Returns the multidimensional frequency at a given multidimensional index, assuming ((D)H)W order.
    /// For the rfft, the index along the width (ie rightmost dimension) is the frequency, so the width
    /// value is ignored and can be omitted from the shape.
    template<bool IS_CENTERED, bool IS_RFFT,
             nt::sinteger T, size_t N0, size_t N1, size_t A0, size_t A1>
    requires (1 <= N0 and N0 <= 3 and ((N0 == N1) or (IS_RFFT and N0 - 1 == N1)))
    [[nodiscard]] constexpr auto index2frequency(
        Vec<T, N0, A0> indices,
        const Shape<T, N1, A1>& shape
    ) noexcept {
        static_for_each<N0>([&]<size_t I>(){
            if constexpr (not (IS_RFFT and I == N0 - 1)) // if width of rfft, frequency == index
                indices[I] = index2frequency<IS_CENTERED>(indices[I], shape[I]);
        });
        return indices;
    }

    /// Returns the index at a given frequency (in samples).
    /// The frequency should be in range [-size/2, (size-1)/2]
    /// \warning This function is only intended to be used for full dimensions!
    ///          For a rfft's half dimension (which is not supported by this function),
    ///          the index is equal to the frequency, regardless of the centering.
    template<bool IS_CENTERED, nt::sinteger T>
    [[nodiscard]] constexpr T frequency2index(T frequency, T size) noexcept {
        // n=5: [0, 1, 2, 3, 4] -> centered=[-2,-1, 0, 1, 2], non-centered=[0, 1, 2,-2,-1]
        // n=6: [0, 1, 2, 3, 4, 5] -> centered=[-3,-2,-1, 0, 1, 2], non-centered=[0, 1, 2,-3,-2,-1]
        if constexpr (IS_CENTERED)
            return frequency + size / 2;
        return frequency < 0 ? frequency + size : frequency;
    }

    /// Returns the multidimensional index at a given multidimensional frequency, assuming ((D)H)W order.
    /// For the rfft, the index along the width (ie rightmost dimension) is the frequency, so the width
    /// value is ignored and can be omitted from the shape.
    template<bool IS_CENTERED, bool IS_RFFT,
             nt::sinteger T, size_t N0, size_t N1, size_t A0, size_t A1>
    requires (1 <= N0 and N0 <= 3 and ((N0 == N1) or (IS_RFFT and N0 - 1 == N1)))
    [[nodiscard]] constexpr auto frequency2index(
        Vec<T, N0, A0> frequency,
        const Shape<T, N1, A1>& shape
    ) noexcept {
        static_for_each<N0>([&]<size_t I>(){
            if constexpr (not (IS_RFFT and I == N0 - 1)) // if width of rfft, frequency == index
                frequency[I] = frequency2index<IS_CENTERED>(frequency[I], shape[I]);
        });
        return frequency;
    }

    /// Returns the centered index, given the (non-)centered index.
    /// \warning This function only supports full-dimensions.
    template<bool IS_CENTERED, nt::integer T>
    constexpr T to_centered_index(T index, T shape) noexcept {
        if constexpr (IS_CENTERED)
            return index;
        return fftshift(index, shape);
    }

    /// Returns the multidimensional centered index given the multidimensional (non-)centered index, assuming ((D)H)W order.
    /// For rffts, the centering doesn't apply to the width (ie rightmost dimension) and the index is left unchanged.
    template<bool IS_CENTERED, bool IS_RFFT,
             nt::sinteger T, size_t N0, size_t N1, size_t A0, size_t A1>
    requires (1 <= N0 and N0 <= 3 and ((N0 == N1) or (IS_RFFT and N0 - 1 == N1)))
    constexpr auto to_centered_indices(
        Vec<T, N0, A0> indices,
        const Shape<T, N1, A1>& shape
    ) noexcept -> Vec<T, N0> {
        if constexpr (not IS_CENTERED) {
            static_for_each<N0>([&]<size_t I>(){
                if constexpr (not (IS_RFFT and I == N0 - 1))
                    indices[I] = fftshift(indices[I], shape[I]);
            });
        }
        return indices;
    }

    /// Remaps the input index onto the output.
    /// \warning This function only supports full-dimensions.
    template<Remap REMAP, bool FLIP_REMAP = false, nt::integer T>
    constexpr T remap_index(T index, T shape) noexcept {
        constexpr Remap remap = FLIP_REMAP ? REMAP.flip() : REMAP;
        if constexpr (remap.is_xc2xx() == remap.is_xx2xc()) {
            return index;
        } else if constexpr (remap.is_xc2xx()) { // input is centered, output isn't
            return ifftshift(index, shape);
        } else { // input is not centered, output is centered
            return fftshift(index, shape);
        }
    }

    /// Remaps the input indices onto the output.
    /// This function is limited to input/output being both rffts or both ffts.
    /// For rffts, the centering doesn't apply to the width (ie rightmost dimension) and the index is left unchanged.
    template<Remap REMAP, bool FLIP_REMAP = false,
             nt::integer T, size_t N0, size_t N1, size_t A0, size_t A1>
    requires (1 <= N0 and N0 <= 3)
    constexpr auto remap_indices(
        Vec<T, N0, A0> indices,
        const Shape<T, N1, A1>& shape
    ) noexcept -> Vec<T, N0, A0> {
        constexpr Remap ACTUAL_REMAP = FLIP_REMAP ? REMAP.flip() : REMAP;
        constexpr bool IS_RFFT = ACTUAL_REMAP.is_hx2hx();
        constexpr bool IS_INPUT_CENTERED = ACTUAL_REMAP.is_xc2xx(); // nvcc bug, needs to be outside lambda
        static_assert(IS_RFFT or ACTUAL_REMAP.is_fx2fx());
        static_assert((N0 == N1) or (IS_RFFT and N0 - 1 == N1));

        if constexpr (REMAP.is_xc2xx() != REMAP.is_xx2xc()) {
            static_for_each<N0>([&]<size_t I>(){
                if constexpr (not (IS_RFFT and I == N0 - 1)) {
                    if constexpr (IS_INPUT_CENTERED) // input is centered, output isn't
                        indices[I] = ifftshift(indices[I], shape[I]);
                    else // input is not centered, output is centered
                        indices[I] = fftshift(indices[I], shape[I]);
                }
            });
        }
        return indices;
    }

    /// Computes the phase shift at a given normalized-frequency.
    template<nt::complex T, nt::any_of<f32, f64> R, size_t N, size_t A0, size_t A1>
    requires (N == 2 or N == 3)
    [[nodiscard]] auto phase_shift(
        const Vec<R, N, A0>& shift,
        const Vec<R, N, A1>& fftfreq
    ) noexcept -> T {
        using real_t = typename T::value_type;
        const auto factor = static_cast<real_t>(-dot(2 * Constant<R>::PI * shift, fftfreq));
        T phase_shift;
        sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }
}
