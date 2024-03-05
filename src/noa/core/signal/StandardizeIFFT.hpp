#pragma once

#include "noa/core/Types.hpp"

namespace noa::signal::guts {
    struct FFTSpectrumEnergy {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;

        f64 scale;

        template<typename C, typename R>
        requires (nt::is_complex_v<C> and nt::is_real_v<R>)
        NOA_FHD constexpr void init(const C& input, R& sum) const noexcept {
            sum += static_cast<nt::value_type_t<C>>(abs_squared(input));
        }

        template<typename R>
        NOA_FHD constexpr void join(const R& isum, R& sum) const noexcept {
            sum += isum;
        }

        template<typename R, typename F, typename C>
        NOA_FHD constexpr void final(const R& sum, F& energy, const C& dc) {
            energy = 1 / (sqrt(sum - abs_squared_t(dc)) / static_cast<R>(scale)); // remove the dc=0
        }
    };

    struct rFFTSpectrumEnergy {
        using allow_vectorization = bool;

        template<typename C, typename R>
        requires (nt::is_complex_v<C> and nt::is_real_v<R>)
        NOA_FHD constexpr void init(const C& input, R& sum) const noexcept {
            sum += static_cast<nt::value_type_t<C>>(abs_squared(input));
        }

        template<typename R>
        NOA_FHD constexpr void join(const R& isum, R& sum) const noexcept {
            sum += isum;
        }
    };

    struct SpectrumAccurateEnergy {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;

        NOA_FHD constexpr void init(const auto& input, f64& sum, f64& error) const noexcept {
            kahan_sum(static_cast<f64>(abs_squared(input)), sum, error);
        }

        NOA_FHD constexpr void join(
                const f64& local_sum, const f64& local_error, f64& global_sum, f64& global_error
        ) const noexcept {
            global_sum += local_sum;
            global_error += local_error;
        }

        template<typename F>
        NOA_FHD constexpr void final(const f64& global_sum, const f64& global_error, F& final) {
            final = static_cast<F>(sqrt(global_sum + global_error));
        }
    };
}
