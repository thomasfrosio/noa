#pragma once

#include "noa/base/Complex.hpp"
#include "noa/base/Math.hpp"
#include "noa/runtime/ReduceAxesEwise.hpp"
#include "noa/runtime/Factory.hpp"

#include "noa/fft/core/Frequency.hpp"
#include "noa/fft/core/Layout.hpp"
#include "noa/fft/Transform.hpp"

namespace noa::signal::details {
    struct FFTSpectrumEnergy {
        using enable_vectorization = bool;
        using remove_default_post = bool;

        f64 scale;

        template<nt::complex C, nt::real R>
        constexpr void operator()(const C& input, R& sum) {
            sum += static_cast<nt::value_type_t<C>>(abs_squared(input));
        }

        template<typename R>
        static constexpr void join(const R& isum, R& sum) {
            sum += isum;
        }

        template<typename R, typename F, typename C>
        constexpr void post(const R& sum, F& energy, const C& dc) const {
            energy = 1 / (sqrt(sum - abs_squared(dc)) / static_cast<R>(scale)); // remove the dc=0
        }
    };

    struct rFFTSpectrumEnergy {
        using enable_vectorization = bool;

        template<nt::complex C, nt::real R>
        constexpr void operator()(const C& input, R& sum) {
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
        using remove_default_post = bool;

        constexpr void operator()(const auto& input, f64& sum, f64& error) {
            kahan_sum(static_cast<f64>(abs_squared(input)), sum, error);
        }

        static constexpr void join(const f64& local_sum, const f64& local_error, f64& global_sum, f64& global_error) {
            global_sum += local_sum;
            global_error += local_error;
        }

        template<typename F>
        static constexpr void post(const f64& global_sum, const f64& global_error, F& final) {
            final = static_cast<F>(sqrt(global_sum + global_error));
        }
    };
}

namespace noa::signal {
    struct StandardizeIFFTOptions {
        /// The rank of the transform.
        /// See ranked_shape for more details.
        i32 rank = -1;

        /// Normalization mode.
        nf::Norm norm = nf::NORM_DEFAULT;
    };

    /// Standardizes (mean=0, stddev=1) a real-space signal, by modifying its Fourier coefficients.
    /// \tparam LAYOUT:
    ///     Layout of the transform.
    /// \param[in] input:
    ///     Input (r)FFT.
    ///     Should be reshapeable to 4D.
    /// \param[out] output:
    ///     Output (r)FFT. Should be reshapeable to 4D. Can be equal to the input.
    ///     The c2r transform of the output has its mean set to 0 and its stddev set to 1.
    /// \param shape:
    ///     Logical shape of the input and output.
    /// \param options:
    ///     FFT options.
    template<nf::Layout LAYOUT, usize N,
             nt::readable_array_decay_of_complex Input,
             nt::writable_array_decay_of_complex Output>
    requires (not LAYOUT.has_layout_change())
    void standardize_ifft(
        const Input& input,
        const Output& output,
        const Shape<isize, N>& shape,
        const StandardizeIFFTOptions& options = {}
    ) {
        constexpr bool is_full = LAYOUT == nf::Layout::F2F or LAYOUT == nf::Layout::FC2FC;
        constexpr bool is_centered = LAYOUT == nf::Layout::FC2FC or LAYOUT == nf::Layout::HC2HC;
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
        const auto scale = options.norm == nf::Norm::FORWARD ? 1 : options.norm == nf::Norm::ORTHO ? sqrt(count) : count;
        const auto array_options = ArrayOption{input.device(), Allocator::DEFAULT_ASYNC};

        // Reshape to BDHW.
        auto input_4d = std::forward<Input>(input).template as_nd<4>();
        auto output_4d = std::forward<Output>(output).template as_nd<4>();
        auto shape_4d = input_4d.shape();
        shape_4d = ranked_shape(shape_4d, options.rank);
        input_4d = std::move(input_4d).reshape(shape_4d.set<3>(input_4d.shape()[3]));
        output_4d = std::move(output_4d).reshape(shape_4d.set<3>(output_4d.shape()[3]));

        auto dc_position = make_subregion<4>(
            Full{},
            is_centered ? nf::fftshift(isize{}, shape_4d[1]) : 0,
            is_centered ? nf::fftshift(isize{}, shape_4d[2]) : 0,
            is_centered and is_full ? nf::fftshift(isize{}, shape_4d[3]) : 0);

        if constexpr (LAYOUT == nf::Layout::F2F or LAYOUT == nf::Layout::FC2FC) {
            // Compute the energy of the input (excluding the dc).
            auto dc_components = input_4d.view().subregion(dc_position);
            auto energies = Array<real_t, 4>(dc_components.shape(), array_options);
            reduce_axes_ewise(
                input_4d.view(), real_t{}, wrap(energies.view(), dc_components),
                details::FFTSpectrumEnergy{scale});

            // Standardize.
            ewise(wrap(std::move(input_4d), std::move(energies)), output_4d.view(), Multiply{});
            fill(std::move(output_4d).subregion(dc_position), {});

        } else if constexpr (LAYOUT == nf::Layout::H2H or LAYOUT == nf::Layout::HC2HC) {
            const bool is_even = noa::is_even(shape_4d[3]);

            auto energies = zeros<real_t, 4>({shape_4d[0], 3, 1, 1}, array_options);
            auto energies_0 = energies.view().subregion(Full{}, 0, 0, 0);
            auto energies_1 = energies.view().subregion(Full{}, 1, 0, 0);
            auto energies_2 = energies.view().subregion(Full{}, 2, 0, 0);

            // Reduce unique chunk:
            reduce_axes_ewise(input_4d.view().subregion(Ellipsis{}, Slice{1, input_4d.shape()[3] - is_even}),
                              real_t{}, energies_0, details::rFFTSpectrumEnergy{});

            // Reduce common column/plane containing the DC:
            reduce_axes_ewise(input_4d.view().subregion(Ellipsis{}, 0),
                              real_t{}, energies_1, details::rFFTSpectrumEnergy{});

            if (is_even) {
                // Reduce common column/plane containing the real Nyquist:
                reduce_axes_ewise(input_4d.view().subregion(Ellipsis{}, -1),
                                  real_t{}, energies_2, details::rFFTSpectrumEnergy{});
            }

            // Standardize.
            ewise(wrap(input_4d.view().subregion(dc_position), energies_1, energies_2), energies_0,
                  details::CombineSpectrumEnergies<complex_t, real_t>{static_cast<real_t>(scale)}); // if batch=1, this is one single value...
            ewise(wrap(std::move(input_4d), energies.subregion(Full{}, 0, 0, 0)), output_4d.view(), Multiply{});
            fill(std::move(output_4d).subregion(dc_position), {});

        } else {
            static_assert(nt::always_false<real_t>);
        }
    }
}
