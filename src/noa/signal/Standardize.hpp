#pragma once

#include "noa/runtime/core/Complex.hpp"
#include "noa/runtime/core/Math.hpp"
#include "noa/runtime/ReduceAxesEwise.hpp"
#include "noa/runtime/Factory.hpp"

#include "noa/fft/core/Frequency.hpp"
#include "noa/fft/core/Layout.hpp"
#include "noa/fft/Transform.hpp"

namespace noa::signal::details {
    struct FFTSpectrumEnergy {
        using enable_vectorization = bool;
        using remove_default_final = bool;

        f64 scale;

        template<nt::complex C, nt::real R>
        static constexpr void init(const C& input, R& sum) {
            sum += static_cast<nt::value_type_t<C>>(abs_squared(input));
        }

        template<typename R>
        static constexpr void join(const R& isum, R& sum) {
            sum += isum;
        }

        template<typename R, typename F, typename C>
        constexpr void final(const R& sum, F& energy, const C& dc) const {
            energy = 1 / (sqrt(sum - abs_squared(dc)) / static_cast<R>(scale)); // remove the dc=0
        }
    };

    struct rFFTSpectrumEnergy {
        using enable_vectorization = bool;

        template<nt::complex C, nt::real R>
        static constexpr void init(const C& input, R& sum) {
            sum += static_cast<nt::value_type_t<C>>(abs_squared(input));
        }

        template<typename R>
        static constexpr void join(const R& isum, R& sum) {
            sum += isum;
        }
    };

    template<typename C, typename R>
    struct CombineSpectrumEnergies {
        using enable_vectorization = bool;
        R scale;

        constexpr void operator()(C dc, R energy_1, R energy_2, R& energy_0) const {
            energy_0 = scale / sqrt(2 * energy_0 + energy_1 - abs_squared(dc) + energy_2);
        }
    };

    struct SpectrumAccurateEnergy {
        using enable_vectorization = bool;
        using remove_default_final = bool;

        static constexpr void init(const auto& input, f64& sum, f64& error) {
            kahan_sum(static_cast<f64>(abs_squared(input)), sum, error);
        }

        static constexpr void join(const f64& local_sum, const f64& local_error, f64& global_sum, f64& global_error) {
            global_sum += local_sum;
            global_error += local_error;
        }

        template<typename F>
        static constexpr void final(const f64& global_sum, const f64& global_error, F& final) {
            final = static_cast<F>(sqrt(global_sum + global_error));
        }
    };
}

namespace noa::signal {
    /// Standardizes (mean=0, stddev=1) a real-space signal, by modifying its Fourier coefficients.
    /// \tparam REMAP       Remapping operator. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] input    Input (r)FFT.
    /// \param[out] output  Output (r)FFT. Can be equal to \p input.
    ///                     The c2r transform of \p output has its mean set to 0 and its stddev set to 1.
    /// \param shape        BDHW logical shape of \p input and \p output.
    /// \param norm         Normalization mode of \p input.
    template<nf::Layout REMAP,
             nt::readable_varray_decay_of_complex Input,
             nt::writable_varray_decay_of_complex Output>
    requires (REMAP.is_any(nf::Layout::H2H, nf::Layout::HC2HC, nf::Layout::F2F, nf::Layout::FC2FC))
    void standardize_ifft(
        const Input& input,
        const Output& output,
        const Shape4& shape,
        nf::Norm norm = nf::NORM_DEFAULT
    ) {
        constexpr bool is_full = REMAP == nf::Layout::F2F or REMAP == nf::Layout::FC2FC;
        constexpr bool is_centered = REMAP == nf::Layout::FC2FC or REMAP == nf::Layout::HC2HC;
        const auto actual_shape = is_full ? shape : shape.rfft();

        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(input.shape() == actual_shape and output.shape() == actual_shape,
              "The input {} and output {} {}redundant FFTs don't match the expected shape {}",
              input.shape(), output.shape(), is_full ? "" : "non-", actual_shape);
        check(input.device() == output.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), output.device());

        using real_t = nt::value_type_twice_t<Output>;
        using complex_t = Complex<real_t>;
        const auto count = static_cast<f64>(shape.pop_front().n_elements());
        const auto scale = norm == nf::Norm::FORWARD ? 1 : norm == nf::Norm::ORTHO ? sqrt(count) : count;
        const auto options = ArrayOption{input.device(), Allocator::DEFAULT_ASYNC};

        auto dc_position = ni::make_subregion<4>(
            ni::Full{},
            is_centered ? nf::fftshift(isize{}, shape[1]) : 0,
            is_centered ? nf::fftshift(isize{}, shape[2]) : 0,
            is_centered and is_full ? nf::fftshift(isize{}, shape[3]) : 0);

        if constexpr (REMAP == nf::Layout::F2F or REMAP == nf::Layout::FC2FC) {
            // Compute the energy of the input (excluding the dc).
            auto dc_components = input.view().subregion(dc_position);
            auto energies = Array<real_t>(dc_components.shape(), options);
            reduce_axes_ewise(
                input.view(), real_t{}, wrap(energies.view(), dc_components),
                details::FFTSpectrumEnergy{scale});

            // Standardize.
            ewise(wrap(input, std::move(energies)), output.view(), Multiply{});
            fill(output.subregion(dc_position), {});

        } else if constexpr (REMAP == nf::Layout::H2H or REMAP == nf::Layout::HC2HC) {
            const bool is_even = noa::is_even(shape[3]);

            auto energies = zeros<real_t>({shape[0], 3, 1, 1}, options);
            auto energies_0 = energies.view().subregion(ni::Full{}, 0, 0, 0);
            auto energies_1 = energies.view().subregion(ni::Full{}, 1, 0, 0);
            auto energies_2 = energies.view().subregion(ni::Full{}, 2, 0, 0);

            // Reduce unique chunk:
            reduce_axes_ewise(input.view().subregion(ni::Ellipsis{}, ni::Slice{1, input.shape()[3] - is_even}),
                              real_t{}, energies_0, details::rFFTSpectrumEnergy{});

            // Reduce common column/plane containing the DC:
            reduce_axes_ewise(input.view().subregion(ni::Ellipsis{}, 0),
                              real_t{}, energies_1, details::rFFTSpectrumEnergy{});

            if (is_even) {
                // Reduce common column/plane containing the real Nyquist:
                reduce_axes_ewise(input.view().subregion(ni::Ellipsis{}, -1),
                                  real_t{}, energies_2, details::rFFTSpectrumEnergy{});
            }

            // Standardize.
            ewise(wrap(input.view().subregion(dc_position), energies_1, energies_2), energies_0,
                  details::CombineSpectrumEnergies<complex_t, real_t>{static_cast<real_t>(scale)}); // if batch=1, this is one single value...
            ewise(wrap(input, energies.subregion(ni::Full{}, 0, 0, 0)), output.view(), Multiply{});
            fill(output.subregion(dc_position), {});

        } else {
            static_assert(nt::always_false<>);
        }
    }
}
