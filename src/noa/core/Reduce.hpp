#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Pair.hpp"
#include "noa/core/math/Comparison.hpp"
#include "noa/core/math/Generic.hpp"

// Reduction operators
namespace noa {
    struct ReduceSum {
        using enable_vectorization = bool;

        template<typename T>
        static constexpr void init(const auto& value, T& sum) {
            sum += static_cast<T>(value);
        }
        template<typename T>
        static constexpr void init(const auto& lhs, const auto& rhs, T& sum) { // dot
            sum += static_cast<T>(lhs * rhs);
        }
        template<typename T>
        static constexpr void join(const T& ireduced, T& reduced) {
            reduced += ireduced;
        }
    };

    template<nt::scalar S>
    struct ReduceMean {
        using enable_vectorization = bool;
        using remove_default_final = bool;
        S size;

        template<typename T>
        static constexpr void init(const auto& value, T& sum) {
            sum += static_cast<T>(value);
        }
        template<typename T>
        static constexpr void join(const T& ireduced, T& reduced) {
            reduced += ireduced;
        }
        template<typename T, typename U>
        constexpr void final(const T& sum, U& mean) const {
            mean = static_cast<U>(sum / size);
        }
    };

    struct ReduceL2Norm {
        using enable_vectorization = bool;
        using remove_default_final = bool;

        template<typename T>
        static constexpr void init(const auto& value, T& sum) {
            sum += static_cast<T>(abs_squared(value));
        }
        template<typename T>
        static constexpr void join(const T& ireduced, T& reduced) {
            reduced += ireduced;
        }
        template<typename T>
        static constexpr void final(const auto& sum, T& norm) {
            norm = static_cast<T>(sqrt(sum));
        }
    };

    struct ReduceMin {
        using enable_vectorization = bool;

        template<typename T>
        constexpr void operator()(const T& value, T& min) const {
            min = noa::min(value, min);
        }
    };

    struct ReduceMax {
        using enable_vectorization = bool;

        template<typename T>
        constexpr void operator()(const T& value, T& min) const {
            min = noa::max(value, min);
        }
    };

    struct ReduceMinMax {
        using enable_vectorization = bool;

        template<typename T>
        static constexpr void init(const T& value, T& min, T& max) {
            min = noa::min(value, min);
            max = noa::max(value, max);
        }
        template<typename T>
        static constexpr void join(const T& imin, const T& imax, T& min, T& max) {
            min = noa::min(imin, min);
            max = noa::max(imax, max);
        }
    };

    struct ReduceMinMaxSum {
        using enable_vectorization = bool;

        template<typename T, typename U>
        static constexpr void init(const T& value, T& min, T& max, U& sum) {
            min = noa::min(value, min);
            max = noa::max(value, max);
            sum += static_cast<U>(value);
        }
        template<typename T, typename U>
        static constexpr void join(const T& imin, const T& imax, const U& isum, T& min, T& max, U& sum) {
            min = noa::min(imin, min);
            max = noa::max(imax, max);
            sum += isum;
        }
    };

    template<nt::scalar R>
    struct ReduceVariance {
        using enable_vectorization = bool;
        using remove_default_final = bool;
        R size{};

        template<typename T, typename U>
        static constexpr void init(const T& value, const U& mean, R& reduced) {
            if constexpr (nt::complex<T>) {
                const auto distance = abs(static_cast<U>(value) - mean);
                reduced += static_cast<R>(distance * distance);
            } else {
                const auto distance = static_cast<U>(value) - mean;
                reduced += static_cast<R>(distance * distance);
            }
        }
        static constexpr void join(const R& ireduced, R& reduced) {
            reduced += ireduced;
        }
        template<typename T>
        constexpr void final(const R& reduced, T& variance) const {
            variance = static_cast<T>(reduced / size);
        }
    };

    template<nt::scalar R>
    struct ReduceStddev {
        using enable_vectorization = bool;
        using remove_default_final = bool;
        R size{};

        template<typename T, typename U>
        static constexpr void init(const T& value, const U& mean, R& reduced) {
            if constexpr (nt::complex<T>) {
                const auto distance = abs(static_cast<U>(value) - mean);
                reduced += static_cast<R>(distance * distance);
            } else {
                const auto distance = static_cast<U>(value) - mean;
                reduced += static_cast<R>(distance * distance);
            }
        }
        static constexpr void join(const R& ireduced, R& reduced) {
            reduced += ireduced;
        }
        template<typename T>
        constexpr void final(const R& reduced, T& stddev) const {
            auto variance = reduced / size;
            stddev = static_cast<T>(sqrt(variance));
        }
    };

    template<nt::scalar T>
    struct ReduceRMSD {
        using enable_vectorization = bool;
        using remove_default_final = bool;
        T size;

        template<typename I>
        static constexpr void init(const I& lhs, const I& rhs, T& sum) {
            auto diff = static_cast<T>(lhs) - static_cast<T>(rhs);
            sum += diff * diff;
        }
        static constexpr void join(const T& isum, T& sum) {
            sum += isum;
        }
        template<typename F>
        constexpr void final(const T& sum, F& rmsd) const {
            rmsd = static_cast<F>(sqrt(sum / size));
        }
    };

    struct ReduceAllEqual {
        using enable_vectorization = bool;

        template<typename T>
        static constexpr void init(const auto& lhs, const auto& rhs, T& reduced) {
            reduced = static_cast<T>(lhs == rhs);
        }
        template<typename T>
        static constexpr void join(const T& ireduced, T& reduced) {
            if (not ireduced)
                reduced = false;
        }
    };

    /// Accurate sum reduction operator for (complex) floating-points using Kahan summation, with Neumaier variation.
    template<nt::real_or_complex T>
    struct ReduceAccurateSum {
        using enable_vectorization = bool;
        using remove_default_final = bool;
        using reduced_type = std::conditional_t<nt::real<T>, f64, c64>;
        using pair_type = Pair<reduced_type, reduced_type>;

        static constexpr void init(const auto& input, pair_type& sum) {
            auto value = static_cast<reduced_type>(input);
            kahan_sum(value, sum.first, sum.second);
        }
        static constexpr void init(const auto& lhs, const auto& rhs, pair_type& sum) { // dot
            auto value = static_cast<reduced_type>(lhs * rhs);
            kahan_sum(value, sum.first, sum.second);
        }
        static constexpr void join(const pair_type& local_sum, pair_type& global_sum) {
            global_sum.first += local_sum.first;
            global_sum.second += local_sum.second;
        }
        template<typename F>
        static constexpr void final(const pair_type& global_sum, F& final) {
            final = static_cast<F>(global_sum.first + global_sum.second);
        }
    };

    template<nt::real_or_complex T>
    struct ReduceAccurateMean {
        using enable_vectorization = bool;
        using remove_default_final = bool;
        using reduced_type = std::conditional_t<nt::real<T>, f64, c64>;
        using pair_type = Pair<reduced_type, reduced_type>;
        using mean_type = nt::value_type_t<reduced_type>;
        mean_type size;

        static constexpr void init(const auto& input, pair_type& sum) {
            auto value = static_cast<reduced_type>(input);
            kahan_sum(value, sum.first, sum.second);
        }
        static constexpr void join(const pair_type& local_sum, pair_type& global_sum) {
            global_sum.first += local_sum.first;
            global_sum.second += local_sum.second;
        }
        template<typename F>
        constexpr void final(const pair_type& global_sum, F& final) const {
            final = static_cast<F>((global_sum.first + global_sum.second) / size);
        }
    };

    struct ReduceAccurateL2Norm {
        using enable_vectorization = bool;
        using remove_default_final = bool;
        using pair_type = Pair<f64, f64>;

        static constexpr void init(const auto& input, pair_type& sum) {
            kahan_sum(static_cast<f64>(abs_squared(input)), sum.first, sum.second);
        }
        static constexpr void join(const pair_type& local_sum, pair_type& global_sum) {
            global_sum.first += local_sum.first;
            global_sum.second += local_sum.second;
        }
        template<typename F>
        static constexpr void final(const pair_type& global_sum, F& final) {
            final = static_cast<F>(sqrt(global_sum.first + global_sum.second));
        }
    };
}

namespace noa {
    template<typename Accessor, typename Reduced, bool SaveValue, typename Reducer>
    struct ReduceArg {
        using accessor_type = Accessor;
        using reduced_type = Reduced;
        using value_type = reduced_type::first_type;
        using offset_type = reduced_type::second_type;

        accessor_type accessor;

    public:
        constexpr void init(const auto& indices, reduced_type& reduced) const {
            // TODO Add option for per batch offsets?
            reduced_type current{
                cast_or_abs_squared<value_type>(accessor(indices)),
                static_cast<offset_type>(accessor.offset_at(indices))
            };
            static_cast<const Reducer&>(*this).join(current, reduced);
        }

        template<typename T>
        static constexpr void final(const reduced_type& reduced, T& output) {
            if constexpr (SaveValue)
                output = static_cast<T>(reduced.first);
            else
                output = static_cast<T>(reduced.second);
        }

        template<typename T, typename U>
        static constexpr void final(const reduced_type& reduced, T& value, U& offset) {
            value = static_cast<T>(reduced.first);
            offset = static_cast<U>(reduced.second);
        }
    };

    template<typename Accessor, typename Reduced, bool SaveValue = true>
    struct ReduceFirstMin : ReduceArg<Accessor, Reduced, SaveValue, ReduceFirstMin<Accessor, Reduced, SaveValue>> {
        using remove_default_final = bool;
        using enable_vectorization = bool;
        using reduced_type = ReduceArg<Accessor, Reduced, SaveValue, ReduceFirstMin>::reduced_type;

        static constexpr void join(const reduced_type& current, reduced_type& reduced) {
            if (current.first < reduced.first or (current.first == reduced.first and current.second < reduced.second))
                reduced = current;
        }
    };

    template<typename Accessor, typename Reduced, bool SaveValue = true>
    struct ReduceFirstMax : ReduceArg<Accessor, Reduced, SaveValue, ReduceFirstMax<Accessor, Reduced, SaveValue>> {
        using remove_default_final = bool;
        using enable_vectorization = bool;
        using reduced_type = ReduceArg<Accessor, Reduced, SaveValue, ReduceFirstMax>::reduced_type;

        static constexpr void join(const reduced_type& current, reduced_type& reduced) {
            if (current.first > reduced.first or (reduced.first == current.first and current.second < reduced.second))
                reduced = current;
        }
    };

    template<typename Accessor, typename Reduced, bool SaveValue = true>
    struct ReduceLastMin : ReduceArg<Accessor, Reduced, SaveValue, ReduceLastMin<Accessor, Reduced, SaveValue>> {
        using remove_default_final = bool;
        using enable_vectorization = bool;
        using reduced_type = ReduceArg<Accessor, Reduced, SaveValue, ReduceLastMin>::reduced_type;

        static constexpr void join(const reduced_type& current, reduced_type& reduced) {
            if (current.first < reduced.first or (current.first == reduced.first and current.second > reduced.second))
                reduced = current;
        }
    };

    template<typename Accessor, typename Reduced, bool SaveValue = true>
    struct ReduceLastMax : ReduceArg<Accessor, Reduced, SaveValue, ReduceLastMax<Accessor, Reduced, SaveValue>> {
        using remove_default_final = bool;
        using enable_vectorization = bool;
        using reduced_type = ReduceArg<Accessor, Reduced, SaveValue, ReduceLastMax>::reduced_type;

        static constexpr void join(const reduced_type& current, reduced_type& reduced) {
            if (current.first > reduced.first or (reduced.first == current.first and current.second > reduced.second))
                reduced = current;
        }
    };
}
