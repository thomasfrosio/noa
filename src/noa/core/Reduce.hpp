#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Pair.hpp"
#include "noa/core/math/Comparison.hpp"
#include "noa/core/math/Generic.hpp"

// Reduction operators
namespace noa {
    struct ReduceSum {
        using allow_vectorization = bool;

        template<typename T>
        NOA_FHD constexpr void init(const auto& value, T& sum) const {
            sum += static_cast<T>(value);
        }
        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            reduced += ireduced;
        }
    };

    template<typename S> requires nt::is_scalar_v<S>
    struct ReduceMean {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        S size;

        template<typename T>
        NOA_FHD constexpr void init(const auto& value, T& sum) const {
            sum += static_cast<T>(value);
        }
        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            reduced += ireduced;
        }
        template<typename T, typename U>
        NOA_FHD constexpr void final(const T& sum, U& mean) const {
            mean = static_cast<U>(sum / size);
        }
    };

    struct ReduceL2Norm {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;

        template<typename T>
        NOA_FHD constexpr void init(const auto& value, T& sum) const {
            sum += static_cast<T>(abs_squared(value));
        }
        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            reduced += ireduced;
        }
        template<typename T>
        NOA_FHD constexpr void final(const auto& sum, T& norm) const {
            norm = static_cast<T>(sqrt(sum));
        }
    };

    struct ReduceMin {
        using allow_vectorization = bool;

        template<typename T>
        NOA_FHD constexpr void operator()(const T& value, T& min) const {
            min = noa::min(value, min);
        }
    };

    struct ReduceMax {
        using allow_vectorization = bool;

        template<typename T>
        NOA_FHD constexpr void operator()(const T& value, T& min) const {
            min = noa::min(value, min);
        }
    };

    struct ReduceMinMax {
        using allow_vectorization = bool;

        template<typename T>
        NOA_FHD constexpr void init(const T& value, T& min, T& max) const {
            min = noa::min(value, min);
            max = noa::max(value, max);
        }
        template<typename T>
        NOA_FHD constexpr void join(const T& imin, const T& imax, T& min, T& max) const {
            min = noa::min(imin, min);
            max = noa::max(imax, max);
        }
    };

    struct ReduceMinMaxSum {
        using allow_vectorization = bool;

        template<typename T, typename U>
        NOA_FHD constexpr void init(const T& value, T& min, T& max, U& sum) const {
            min = noa::min(value, min);
            max = noa::max(value, max);
            sum += static_cast<U>(value);
        }
        template<typename T, typename U>
        NOA_FHD constexpr void join(const T& imin, const T& imax, const U& isum, T& min, T& max, U& sum) const {
            min = noa::min(imin, min);
            max = noa::max(imax, max);
            sum += isum;
        }
    };

    template<typename R>
    struct ReduceVariance {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        R size{};

        template<typename T, typename U>
        NOA_FHD constexpr void init(const T& value, const U& mean, R& reduced) const {
            if constexpr (nt::is_complex_v<T>) {
                const auto distance = abs(static_cast<U>(value) - mean);
                reduced += static_cast<R>(distance * distance);
            } else {
                const auto distance = static_cast<U>(value) - mean;
                reduced += static_cast<R>(distance * distance);
            }
        }
        NOA_FHD constexpr void join(const R& ireduced, R& reduced) const {
            reduced += ireduced;
        }
        template<typename T>
        NOA_FHD constexpr void final(const R& reduced, T& variance) const {
            variance = static_cast<T>(reduced / size);
        }
    };

    template<typename R>
    struct ReduceStddev {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        R size{};

        template<typename T, typename U>
        NOA_FHD constexpr void init(const T& value, const U& mean, R& reduced) const {
            if constexpr (nt::is_complex_v<T>) {
                const auto distance = abs(static_cast<U>(value) - mean);
                reduced += static_cast<R>(distance * distance);
            } else {
                const auto distance = static_cast<U>(value) - mean;
                reduced += static_cast<R>(distance * distance);
            }
        }
        NOA_FHD constexpr void join(const R& ireduced, R& reduced) const {
            reduced += ireduced;
        }
        template<typename T>
        NOA_FHD constexpr void final(const R& reduced, T& stddev) const {
            auto variance = reduced / size;
            stddev = static_cast<T>(sqrt(variance));
        }
    };

    template<typename T>
    struct ReduceRMSD {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        T size;

        template<typename I>
        NOA_FHD constexpr void init(const I& lhs, const I& rhs, T& sum) const {
            auto diff = static_cast<T>(lhs) - static_cast<T>(rhs);
            sum += diff * diff;
        }
        NOA_FHD constexpr void join(const T& isum, T& sum) const {
            sum += isum;
        }
        template<typename F>
        NOA_FHD constexpr void final(const T& sum, F& rmsd) const {
            rmsd = static_cast<F>(sqrt(sum / size));
        }
    };

    struct ReduceAllEqual {
        using allow_vectorization = bool;

        template<typename T>
        NOA_FHD constexpr void init(const auto& lhs, const auto& rhs, T& reduced) const {
            reduced = static_cast<T>(lhs == rhs);
        }
        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            if (not ireduced)
                reduced = false;
        }
    };

    /// Accurate sum reduction operator for (complex) floating-points using Kahan summation, with Neumaier variation.
    template<typename T>
    struct ReduceAccurateSum {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        using reduced_type = std::conditional_t<nt::is_real_v<T>, f64, c64>;
        using pair_type = Pair<reduced_type, reduced_type>;

        NOA_FHD constexpr void init(const auto& input, pair_type& sum) const {
            auto value = static_cast<reduced_type>(input);
            kahan_sum(value, sum.first, sum.second);
        }
        NOA_FHD constexpr void join(const pair_type& local_sum, pair_type& global_sum) const {
            global_sum.first += local_sum.first;
            global_sum.second += local_sum.second;
        }
        template<typename F>
        NOA_FHD constexpr void final(const pair_type& global_sum, F& final) const {
            final = static_cast<F>(global_sum.first + global_sum.second);
        }
    };

    template<typename T>
    struct ReduceAccurateMean {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        using reduced_type = std::conditional_t<nt::is_real_v<T>, f64, c64>;
        using pair_type = Pair<reduced_type, reduced_type>;
        using mean_type = nt::value_type_t<reduced_type>;
        mean_type size;

        NOA_FHD constexpr void init(const auto& input, pair_type& sum) const {
            auto value = static_cast<reduced_type>(input);
            kahan_sum(value, sum.first, sum.second);
        }
        NOA_FHD constexpr void join(const pair_type& local_sum, pair_type& global_sum) const {
            global_sum.first += local_sum.first;
            global_sum.second += local_sum.second;
        }
        template<typename F>
        NOA_FHD constexpr void final(const pair_type& global_sum, F& final) const {
            final = static_cast<F>((global_sum.first + global_sum.second) / size);
        }
    };

    struct ReduceAccurateL2Norm {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        using pair_type = Pair<f64, f64>;

        NOA_FHD constexpr void init(const auto& input, pair_type& sum) const {
            kahan_sum(static_cast<f64>(abs_squared(input)), sum.first, sum.second);
        }
        NOA_FHD constexpr void join(const pair_type& local_sum, pair_type& global_sum) const {
            global_sum.first += local_sum.first;
            global_sum.second += local_sum.second;
        }
        template<typename F>
        NOA_FHD constexpr void final(const pair_type& global_sum, F& final) const {
            final = static_cast<F>(sqrt(global_sum.first + global_sum.second));
        }
    };
}

namespace noa {
    template<typename Accessor, typename Offset, bool SaveValue, typename Reducer>
    struct ReduceArg {
        using accessor_type = Accessor;
        using value_type = accessor_type::mutable_value_type;
        using offset_type = Offset;
        using reduced_type = Pair<value_type, offset_type>;
        accessor_type accessor;

    public:
        constexpr void init(const auto& indices, reduced_type& reduced) const {
            // TODO Add option for per batch offsets?
            reduced_type current{accessor(indices), static_cast<offset_type>(accessor.offset_at(indices))};
            static_cast<const Reducer&>(*this).join(current, reduced);
        }

        template<typename T>
        constexpr void final(const reduced_type& reduced, T& output) const {
            if constexpr (SaveValue)
                output = static_cast<T>(reduced.first);
            else
                output = static_cast<T>(reduced.second);
        }

        template<typename T, typename U>
        constexpr void final(const reduced_type& reduced, T& value, U& offset) const {
            value = static_cast<T>(reduced.first);
            offset = static_cast<U>(reduced.second);
        }
    };

    template<typename Accessor, typename Offset = i64, bool SaveValue = false>
    struct ReduceFirstMin : ReduceArg<Accessor, Offset, SaveValue, ReduceFirstMin<Accessor, Offset, SaveValue>> {
        using base_type = ReduceArg<Accessor, Offset, SaveValue, ReduceFirstMin<Accessor, Offset, SaveValue>>;
        using remove_defaulted_final = bool;
        using allow_vectorization = bool;
        using accessor_type = base_type::accessor_type;
        using reduced_type = base_type::reduced_type;

        constexpr void join(const reduced_type& current, reduced_type& reduced) const {
            if (current.first < reduced.first or (current.first == reduced.first and current.second < reduced.second))
                reduced = current;
        }
    };

    template<typename Accessor, typename Offset = i64, bool SaveValue = false>
    struct ReduceFirstMax : ReduceArg<Accessor, Offset, SaveValue, ReduceFirstMax<Accessor, Offset, SaveValue>> {
        using base_type = ReduceArg<Accessor, Offset, SaveValue, ReduceFirstMax<Accessor, Offset, SaveValue>>;
        using remove_defaulted_final = bool;
        using allow_vectorization = bool;
        using accessor_type = base_type::accessor_type;
        using reduced_type = base_type::reduced_type;

        constexpr void join(const reduced_type& current, reduced_type& reduced) const {
            if (current.first > reduced.first or (reduced.first == current.first and current.second < reduced.second))
                reduced = current;
        }
    };

    template<typename Accessor, typename Offset = i64, bool SaveValue = false>
    struct ReduceLastMin : ReduceArg<Accessor, Offset, SaveValue, ReduceFirstMax<Accessor, Offset, SaveValue>> {
        using base_type = ReduceArg<Accessor, Offset, SaveValue, ReduceFirstMax<Accessor, Offset, SaveValue>>;
        using remove_defaulted_final = bool;
        using allow_vectorization = bool;
        using accessor_type = base_type::accessor_type;
        using reduced_type = base_type::reduced_type;

        constexpr void join(const reduced_type& current, reduced_type& reduced) const {
            if (current.first < reduced.first or (current.first == reduced.first and current.second > reduced.second))
                reduced = current;
        }
    };

    template<typename Accessor, typename Offset = i64, bool SaveValue = false>
    struct ReduceLastMax : ReduceArg<Accessor, Offset, SaveValue, ReduceFirstMax<Accessor, Offset, SaveValue>> {
        using base_type = ReduceArg<Accessor, Offset, SaveValue, ReduceFirstMax<Accessor, Offset, SaveValue>>;
        using remove_defaulted_final = bool;
        using allow_vectorization = bool;
        using accessor_type = base_type::accessor_type;
        using reduced_type = base_type::reduced_type;

        constexpr void join(const reduced_type& current, reduced_type& reduced) const {
            if (current.first > reduced.first or (reduced.first == current.first and current.second > reduced.second))
                reduced = current;
        }
    };
}
