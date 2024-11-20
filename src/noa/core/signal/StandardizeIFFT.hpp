#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/math/Generic.hpp"

namespace noa::signal::guts {
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
