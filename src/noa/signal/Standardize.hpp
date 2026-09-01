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
            R spectrum_energy = max(R{}, sum - abs_squared(dc));
            if (spectrum_energy > 0) {
                energy = static_cast<R>(scale) / sqrt(spectrum_energy);
            } else {
                energy = 0;
            }
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
        usize rank{};

        /// Normalization mode.
        nf::Norm norm = nf::NORM_DEFAULT;
    };

    /// Standardizes (mean=0, stddev=1) a real-space signal, by modifying its Fourier coefficients.
    /// \tparam LAYOUT:
    ///     Layout of the transform.
    /// \param[in] input:
    ///     Input (r)FFT.
    ///     The rank is set by options.rank and the spectrum should be reshapeable to 4D.
    /// \param[out] output:
    ///     Output (r)FFT. Can be equal to the input.
    ///     The rank is set by options.rank and the spectrum should be reshapeable to 4D.
    ///     The c2r transform of the output has its mean set to 0 and its stddev set to 1.
    /// \param shape:
    ///     Logical shape of the input and output.
    /// \param options:
    ///     FFT options.
    template<nf::Layout LAYOUT,
             nt::readable_array_decay_of_complex Input,
             nt::writable_array_decay_of_complex Output,
             usize N>
        requires (not LAYOUT.has_layout_change() and
                  nt::array_size_v<Input> == N and
                  nt::array_size_v<Output> == N)
    void standardize_ifft(
        const Input& input,
        const Output& output,
        const Shape<isize, N>& shape,
        const StandardizeIFFTOptions& options = {}
    ) {
        constexpr bool IS_FULL = LAYOUT.is_fx2fx();
        constexpr bool IS_CENTERED = LAYOUT.is_xc2xc();
        const auto actual_shape = IS_FULL ? shape : shape.rfft();

        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(input.shape() == actual_shape and output.shape() == actual_shape,
              "The input:shape={} and output:shape{} don't match the expected shape {} given rfft={}",
              input.shape(), output.shape(), actual_shape, not IS_FULL);
        check(input.device() == output.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), output.device());

        using real_t = nt::value_type_twice_t<Output>;
        using complex_t = Complex<real_t>;

        // Reshape to BDHW.
        // TODO Add support for compile-time rank, removing the need for collapsible batch axes.
        const auto rank = shape.rank_checked(options.rank);
        const auto shape_4d = shape.template as_nd<4>().ranked(rank);
        const auto input_4d = NOA_FWD(input).template as_nd<4>().reshape(shape_4d.template set<3>(input.shape()[N - 1]));
        const auto output_4d = NOA_FWD(output).template as_nd<4>().reshape(shape_4d.template set<3>(output.shape()[N - 1]));

        const auto array_options = ArrayOption{.device = input.device(), .allocator = Allocator::DEFAULT_ASYNC};
        const auto n_elements = static_cast<f64>(shape_4d.pop_front().n_elements());
        const auto scale =
            options.norm == nf::Norm::FORWARD ? 1 :
            options.norm == nf::Norm::ORTHO ? std::sqrt(n_elements) :
            n_elements;

        const auto dc_positions = Subregion(
            Full{},
            IS_CENTERED ? nf::fftshift(isize{}, shape_4d[1]) : 0,
            IS_CENTERED ? nf::fftshift(isize{}, shape_4d[2]) : 0,
            IS_CENTERED and IS_FULL ? nf::fftshift(isize{}, shape_4d[3]) : 0);

        if constexpr (IS_FULL) {
            // Compute the energy of the input (excluding the DC).
            auto dc_components = input_4d.view().subregion(dc_positions);
            auto energies = Array<real_t, 4>(dc_components.shape(), array_options);
            noa::reduce_axes_ewise(
                input_4d.view(), real_t{}, noa::wrap(energies.view(), dc_components),
                details::FFTSpectrumEnergy{scale});

            // Standardize.
            noa::ewise(noa::wrap(std::move(input_4d), std::move(energies)), output_4d.view(), Multiply{});
            noa::fill(std::move(output_4d).subregion(dc_positions), {});
        } else {
            const bool is_even = noa::is_even(shape_4d[3]);
            auto energies = noa::zeros<real_t, 2>({3, shape_4d[0]}, array_options);
            auto energies_0 = energies.subregion(0).reshape(Shape4{shape_4d[0], 1, 1, 1});
            auto energies_1 = energies.view().subregion(1).reshape(Shape4{shape_4d[0], 1, 1, 1});
            auto energies_2 = energies.view().subregion(2).reshape(Shape4{shape_4d[0], 1, 1, 1});

            // Reduce unique chunk:
            noa::reduce_axes_ewise(
                input_4d.view().subregion(Ellipsis{}, Slice{.start = 1, .end = input_4d.shape()[3] - is_even}),
                real_t{}, energies_0.view(), details::rFFTSpectrumEnergy{}
            );

            // Reduce common column/plane containing the DC:
            if (shape_4d[1] == 1 and shape_4d[2] == 1) { // 1d case
                noa::ewise(
                    input_4d.view().subregion(Ellipsis{}, 0),
                    energies_1, details::rFFTSpectrumEnergy{}
                );
            } else {
                if constexpr (N >= 2) {
                    noa::reduce_axes_ewise(
                        input_4d.view().subregion(Ellipsis{}, 0),
                        real_t{}, energies_1, details::rFFTSpectrumEnergy{}
                    );
                } else {
                    panic();
                }
            }

            if (is_even) {
                // Reduce common column/plane containing the real Nyquist:
                if (shape_4d[1] == 1 and shape_4d[2] == 1) { // 1d case
                    noa::ewise(
                        input_4d.view().subregion(Ellipsis{}, -1),
                        energies_2, details::rFFTSpectrumEnergy{}
                    );
                } else {
                    if constexpr (N >= 2) {
                        noa::reduce_axes_ewise(
                            input_4d.view().subregion(Ellipsis{}, -1),
                            real_t{}, energies_2, details::rFFTSpectrumEnergy{}
                        );
                    } else {
                        panic();
                    }
                }
            }

            // Standardize.
            noa::ewise(
                noa::wrap(input_4d.view().subregion(dc_positions), energies_1, energies_2), energies_0.view(),
                details::CombineSpectrumEnergies<complex_t, real_t>{static_cast<real_t>(scale)}
            );
            noa::ewise(
                noa::wrap(std::move(input_4d), energies_0),
                output_4d.view(), Multiply{}
            );
            noa::fill(std::move(output_4d).subregion(dc_positions), {});
        }
    }

    template<nf::Layout LAYOUT, typename Input, typename Output, usize N>
    void standardize_ifft1(
        const Input& input,
        const Output& output,
        const Shape<isize, N>& shape,
        StandardizeIFFTOptions options = {}
    ) {
        options.rank = 1;
        standardize_ifft<LAYOUT>(input, output, shape, options);
    }
    template<nf::Layout LAYOUT, typename Input, typename Output, usize N>
    void standardize_ifft2(
        const Input& input,
        const Output& output,
        const Shape<isize, N>& shape,
        StandardizeIFFTOptions options = {}
    ) {
        options.rank = 2;
        standardize_ifft<LAYOUT>(input, output, shape, options);
    }
    template<nf::Layout LAYOUT, typename Input, typename Output, usize N>
    void standardize_ifft3(
        const Input& input,
        const Output& output,
        const Shape<isize, N>& shape,
        StandardizeIFFTOptions options = {}
    ) {
        options.rank = 3;
        standardize_ifft<LAYOUT>(input, output, shape, options);
    }
}
