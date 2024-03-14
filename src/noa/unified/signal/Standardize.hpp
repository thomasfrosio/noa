#pragma once

#include "noa/core/fft/Frequency.hpp"
#include "noa/core/fft/RemapInterface.hpp"
#include "noa/core/signal/StandardizeIFFT.hpp"
#include "noa/unified/fft/Transform.hpp"
#include "noa/unified/ReduceAxesEwise.hpp"
#include "noa/unified/Factory.hpp"

namespace noa::signal {
    /// Standardizes (mean=0, stddev=1) a real-space signal, by modifying its Fourier coefficients.
    /// \tparam REMAP       Remapping operator. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] input    Input (r)FFT.
    /// \param[out] output  Output (r)FFT. Can be equal to \p input.
    ///                     The c2r transform of \p output has its mean set to 0 and its stddev set to 1.
    /// \param shape        BDHW logical shape of \p input and \p output.
    /// \param norm         Normalization mode of \p input.
    template<noa::fft::RemapInterface REMAP, typename Input, typename Output>
    requires (nt::is_varray_of_almost_any_v<Input, c32, c64> &&
              nt::is_varray_of_any_v<Output, c32, c64> &&
              nt::are_almost_same_value_type_v<Input, Output> &&
              (REMAP.remap == noa::fft::Remap::H2H or
               REMAP.remap == noa::fft::Remap::HC2HC or
               REMAP.remap == noa::fft::Remap::F2F or
               REMAP.remap == noa::fft::Remap::FC2FC))
    void standardize_ifft(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            noa::fft::Norm norm = noa::fft::NORM_DEFAULT
    ) {
        using Remap = noa::fft::Remap;
        using Norm = noa::fft::Norm;

        constexpr bool is_full = REMAP.remap == Remap::F2F or REMAP.remap == Remap::FC2FC;
        constexpr bool is_centered = REMAP.remap == Remap::FC2FC or REMAP.remap == Remap::HC2HC;
        const auto actual_shape = is_full ? shape : shape.rfft();

        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(all(input.shape() == actual_shape) and all(output.shape() == actual_shape),
              "The input {} and output {} {}redundant FFTs don't match the expected shape {}",
              input.shape(), output.shape(), is_full ? "" : "non-", actual_shape);
        check(input.device() == output.device(),
              "The input and output arrays must be on the same device, but got input:{}, output:{}",
              input.device(), output.device());

        using real_t = nt::value_type_t<Output>;
        using complex_t = Complex<real_t>;
        const auto count = static_cast<f64>(shape.elements());
        const auto scale = norm == Norm::FORWARD ? 1 : norm == Norm::ORTHO ? sqrt(count) : count;
        const auto options = ArrayOption{input.device(), Allocator(MemoryResource::DEFAULT_ASYNC)};

        auto dc_position = ni::Subregion{
                ni::FullExtent{},
                is_centered ? noa::fft::fftshift(i64{0}, shape[1]) : 0,
                is_centered ? noa::fft::fftshift(i64{0}, shape[2]) : 0,
                is_centered and is_full ? noa::fft::fftshift(i64{0}, shape[3]) : 0,
        };

        if constexpr (REMAP.remap == Remap::F2F or REMAP.remap == Remap::FC2FC) {
            // Compute the energy of the input (excluding the dc).
            auto dcomponents = input.view().subregion(dc_position);
            auto energies = Array<real_t>(dcomponents.shape(), options);
            reduce_axes_ewise(input.view(), real_t{0}, wrap(energies.view(), dcomponents), guts::FFTSpectrumEnergy{scale});

            // Standardize.
            ewise(wrap(input, std::move(energies)), output.view(), Multiply{});
            fill(output.subregion(dc_position), 0);

        } else if constexpr (REMAP.remap == Remap::H2H or REMAP.remap == Remap::HC2HC) {
            const bool is_even = noa::is_even(shape[3]);
            const auto original = ni::SubregionIndexer(input.shape(), input.strides());

            auto energies = zeros<real_t>({shape[0], 1, 1, 1}, options);
            auto energies_0 = energies.view().subregion(ni::FullExtent{}, 0, 0, 0);
            auto energies_1 = energies.view().subregion(ni::FullExtent{}, 1, 0, 0);
            auto energies_2 = energies.view().subregion(ni::FullExtent{}, 2, 0, 0);

            // Reduce unique chunk:
            auto subregion = original.extract_subregion(ni::Ellipsis{}, ni::Slice{1, original.shape[3] - is_even});
            reduce_axes_ewise(input.view().subregion(subregion), real_t{0}, energies_0, guts::rFFTSpectrumEnergy{});

            // Reduce common column/plane containing the DC:
            subregion = original.extract_subregion(ni::Ellipsis{}, 0);
            reduce_axes_ewise(input.view().subregion(subregion), real_t{0}, energies_1, guts::rFFTSpectrumEnergy{});

            if (is_even) {
                // Reduce common column/plane containing the real Nyquist:
                subregion = original.extract_subregion(ni::Ellipsis{}, -1);
                reduce_axes_ewise(input.view().subregion(subregion), real_t{0}, energies_2, guts::rFFTSpectrumEnergy{});
            }

            // Standardize.
            ewise(wrap(input.view().subregion(dc_position), energies_1, energies_2), energies_0,
                  [scale]NOA_HD(complex_t dc, real_t energy_1, real_t energy_2, real_t& energy_0) {
                      energy_0 = scale / sqrt(2 * energy_0 + energy_1 - abs_squared(dc) + energy_2);
                  }); // if batch=1, this is one single value...
            ewise(wrap(input, energies.subregion(ni::FullExtent{}, 0, 0, 0)), output.view(), Multiply{});
            fill(output.subregion(dc_position), 0);

        } else {
            static_assert(nt::always_false_v<Output>);
        }
    }
}
