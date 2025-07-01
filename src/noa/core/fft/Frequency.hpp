#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa::fft {
    /// Returns the highest normalized frequency (in cycle/pix) that a dimension with a given size can have.
    template<nt::real T>
    [[nodiscard]] constexpr auto highest_normalized_frequency(nt::integer auto size) noexcept -> T {
        // even: the highest frequency is always 0.5, i.e. Nyquist. E.g. size=64, 32/64=0.5
        // odd: Nyquist cannot be reached. Eg. size=63, 31/63 = 0.49206
        const auto max_index = size / 2; // integer division
        return static_cast<T>(max_index) / static_cast<T>(size);
    }

    /// Returns the highest normalized frequency (in cycle/pix) that a dimension with a given size can have.
    template<nt::real T>
    [[nodiscard]] constexpr auto highest_fftfreq(nt::integer auto size) noexcept {
        return highest_normalized_frequency<T>(size);
    }

    template<nt::real T, nt::integer U, size_t N> requires (N >= 1 and N <= 3)
    [[nodiscard]] constexpr auto highest_fftfreq(const Shape<U, N>& shape) noexcept {
        Vec<T, N> max_fftfreq;
        for (size_t i{}; i < N; ++i)
            max_fftfreq[i] = highest_fftfreq<T>(shape[i]);
        return max_fftfreq;
    }

    /// Returns the fft centered index of the corresponding fft non-centered index.
    /// Should satisfy `0 <= index < size`.
    template<nt::integer T>
    [[nodiscard]] constexpr auto fftshift(T index, T size) noexcept -> T {
        // n=10: [0, 1, 2, 3, 4,-5,-4,-3,-2,-1] -> [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4]
        // n=11: [0, 1, 2, 3, 4, 5,-5,-4,-3,-2,-1] -> [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5]
        return (index < (size + 1) / 2) ? index + size / 2 : index - (size + 1) / 2; // or (index + size / 2) % size
    }
    template<nt::vec_integer T>
    [[nodiscard]] constexpr auto fftshift(const T& indices, const T& sizes) noexcept -> T {
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
    [[nodiscard]] constexpr auto ifftshift(T index, T size) noexcept -> T {
        // n=10: [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4] -> [0, 1, 2, 3, 4,-5,-4,-3,-2,-1]
        // n=11: [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5] -> [0, 1, 2, 3, 4, 5,-5,-4,-3,-2,-1]
        return (index < size / 2) ? index + (size + 1) / 2 : index - size / 2; // or (index + (size + 1) / 2) % size
    }
    template<nt::vec_integer T>
    [[nodiscard]] constexpr auto ifftshift(const T& indices, const T& sizes) noexcept -> T {
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
    [[nodiscard]] constexpr auto index2frequency(T index, T size) noexcept -> T {
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
        if constexpr (N0 == 1 and not IS_RFFT) {
            indices[0] = index2frequency<IS_CENTERED>(indices[0], shape[0]);
        } else if constexpr (N0 > 1) {
            for (size_t i{}; i < N0; ++i) {
                if (not (IS_RFFT and i == N0 - 1)) // if width of rfft, frequency == index
                    indices[i] = index2frequency<IS_CENTERED>(indices[i], shape[i]);
            }
        }
        return indices;
    }

    /// Returns the index at a given frequency (in samples).
    /// The frequency should be in range [-size/2, (size-1)/2]
    /// \warning This function is only intended to be used for full dimensions!
    ///          For a rfft's half-dimension (which is not supported by this function),
    ///          the index is equal to the frequency, regardless of the centering.
    template<bool IS_CENTERED, typename T, nt::sinteger U> requires (nt::real<T> or nt::sinteger<T>)
    [[nodiscard]] constexpr auto frequency2index(T frequency, U size) noexcept -> T {
        // n=5: [0, 1, 2, 3, 4] -> centered=[-2,-1, 0, 1, 2], non-centered=[0, 1, 2,-2,-1]
        // n=6: [0, 1, 2, 3, 4, 5] -> centered=[-3,-2,-1, 0, 1, 2], non-centered=[0, 1, 2,-3,-2,-1]
        if constexpr (IS_CENTERED)
            return frequency + static_cast<T>(size / 2);
        return frequency < 0 ? frequency + static_cast<T>(size) : frequency;
    }

    /// Returns the multidimensional index at a given multidimensional frequency, assuming ((D)H)W order.
    /// For the rfft, the index along the width (ie rightmost dimension) is the frequency, so the width
    /// value is ignored and can be omitted from the shape.
    template<bool IS_CENTERED, bool IS_RFFT,
             typename T, nt::sinteger U, size_t N0, size_t N1, size_t A0, size_t A1>
    requires ((nt::real<T> or nt::sinteger<T>) and
              1 <= N0 and N0 <= 3 and
              ((N0 == N1) or (IS_RFFT and N0 - 1 == N1)))
    [[nodiscard]] constexpr auto frequency2index(
        Vec<T, N0, A0> frequency,
        const Shape<U, N1, A1>& shape
    ) noexcept {
        if constexpr (N0 == 1 and not IS_RFFT) {
            frequency[0] = frequency2index<IS_CENTERED>(frequency[0], shape[0]);
        } else if constexpr (N0 > 1) {
            for (size_t i{}; i < N0; ++i) {
                if (not (IS_RFFT and i == N0 - 1)) // if width of rfft, frequency == index
                    frequency[i] = frequency2index<IS_CENTERED>(frequency[i], shape[i]);
            }
        }
        return frequency;
    }

    /// Returns the centered index, given the (non-)centered index.
    /// \warning This function only supports full-dimensions.
    template<bool IS_CENTERED, nt::integer T>
    constexpr auto to_centered_index(T index, T shape) noexcept -> T {
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
            if constexpr (N0 == 1 and not IS_RFFT) {
                indices[0] = fftshift(indices[0], shape[0]);
            } else if constexpr (N0 > 1) {
                for (size_t i{}; i < N0; ++i) {
                    if (not (IS_RFFT and i == N0 - 1))
                        indices[i] = fftshift(indices[i], shape[i]);
                }
            }
        }
        return indices;
    }

    /// Remaps the input index onto the output.
    /// \warning This function only supports full-dimensions.
    template<Remap REMAP, bool FLIP_REMAP = false, nt::integer T>
    constexpr auto remap_index(T index, T shape) noexcept -> T {
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
        constexpr bool IS_INPUT_CENTERED = ACTUAL_REMAP.is_xc2xx();
        static_assert(IS_RFFT or ACTUAL_REMAP.is_fx2fx());
        static_assert((N0 == N1) or (IS_RFFT and N0 - 1 == N1));

        if constexpr (REMAP.is_xc2xx() != REMAP.is_xx2xc()) {
            if constexpr (N0 == 1 and not IS_RFFT) {
                if constexpr (IS_INPUT_CENTERED) // input is centered, output isn't
                    indices[0] = ifftshift(indices[0], shape[0]);
                else // input is not centered, output is centered
                    indices[0] = fftshift(indices[0], shape[0]);
            } else if constexpr (N0 > 1) {
                for (size_t i{}; i < N0; ++i) {
                    if (not (IS_RFFT and i == N0 - 1)) {
                        if constexpr (IS_INPUT_CENTERED) // input is centered, output isn't
                            indices[i] = ifftshift(indices[i], shape[i]);
                        else // input is not centered, output is centered
                            indices[i] = fftshift(indices[i], shape[i]);
                    }
                }
            }
        }
        return indices;
    }

    /// Computes the phase shift at a given normalized-frequency.
    /// \param shift    ((D)H)W shift.
    /// \param fftfreq  ((D)H)W normalized frequency.
    template<nt::complex T, nt::any_of<f32, f64> R, size_t N0, size_t A0, size_t N1, size_t A1>
    requires ((1 <= N0 and N0 <= 3) and (N0 <= N1 and N1 <= 3))
    [[nodiscard]] constexpr auto phase_shift(
        const Vec<R, N0, A0>& shift,
        const Vec<R, N1, A1>& fftfreq
    ) noexcept -> T {
        using real_t = typename T::value_type;
        real_t factor;
        if constexpr (N0 == 1)
            factor = static_cast<real_t>(-2 * Constant<R>::PI * shift[0], fftfreq[N1 - 1]);
        else if constexpr (N0 == 2)
            factor = static_cast<real_t>(-dot(2 * Constant<R>::PI * shift, fftfreq.filter(N1 - 2, N1 - 1)));
        else if constexpr (N0 == 3)
            factor = static_cast<real_t>(-dot(2 * Constant<R>::PI * shift, fftfreq));
        T phase_shift;
        sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    /// Computes the left-right frequency bounds.
    /// \example
    /// \code
    /// n=5 (full) -> frequency=[-2,-1, 0, 1, 2]      -> left=-2, right=2
    /// n=6 (full) -> frequency=[-3,-2,-1, 0, 1, 2]   -> left=-3, right=2
    /// n=5 (half) -> frequency=[0, 1, 2]             -> left=0,  right=2
    /// n=6 (half) -> frequency=[0, 1, 2, 3]          -> left=0,  right=3
    /// \endcode
    template<bool IS_RFFT, nt::sinteger T, size_t N, size_t A> requires (N >= 1 and N <= 3)
    [[nodiscard]] constexpr auto frequency_bounds(const Shape<T, N, A>& shape) noexcept {
        using bound_t = Vec<T, N, A>;
        Pair<bound_t, bound_t> bounds;
        if constexpr (IS_RFFT) {
            for (size_t i{}; i < N - 1; ++i) {
                bounds.first[i] = -shape[i] / 2;
                bounds.second[i] = (shape[i] - 1) / 2;
            }
            bounds.first[N - 1] = 0;
            bounds.second[N - 1] = shape[N - 1] / 2;
        } else {
            bounds.first = -shape / 2;
            bounds.second = (shape - 1) / 2;
        }
        return bounds;
    }

    /// Whether the (unnormalized) frequency is within bounds.
    /// TODO Finish optimization IS_RFFT and IS_FLIPPED
    template<bool IS_RFFT, bool IS_FLIPPED = false, nt::sinteger T, size_t N, size_t A, nt::pair U>
    requires (N >= 1 and N <= 3)
    [[nodiscard]] constexpr auto is_inbound(const Vec<T, N, A>& frequency, const U& bounds) noexcept {
        for (size_t i{}; i < N; ++i) {
            if (frequency[i] < bounds.first[i] or frequency[i] > bounds.second[i])
                return false;
        }
        return true;
    }
}
