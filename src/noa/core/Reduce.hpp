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
    struct ReduceMean : ReduceSum {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        S size;

        template<typename T, typename U>
        NOA_FHD constexpr void final(const T& sum, U& mean) const {
            mean = static_cast<U>(sum / size);
        }
    };

    struct ReduceL2Norm : ReduceSum {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;

        template<typename T>
        NOA_FHD constexpr void init(const auto& value, T& sum) const {
            sum += static_cast<T>(abs_squared(value));
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
        NOA_FHD constexpr void init(const T& value, const U& mean, R& reduced) const noexcept {
            if constexpr (nt::is_complex_v<T>) {
                const auto distance = abs(static_cast<U>(value) - mean);
                reduced += static_cast<R>(distance * distance);
            } else {
                const auto distance = static_cast<U>(value) - mean;
                reduced += static_cast<R>(distance * distance);
            }
        }
        NOA_FHD constexpr void join(const R& ireduced, R& reduced) const noexcept {
            reduced += ireduced;
        }
        template<typename T>
        NOA_FHD constexpr void final(const R& reduced, T& variance) const noexcept {
            variance = static_cast<T>(reduced / size);
        }
    };

    template<typename R>
    struct ReduceStddev : ReduceVariance<R> {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;

        template<typename T>
        NOA_FHD constexpr void final(const R& reduced, T& stddev) const noexcept {
            auto variance = reduced / this->size;
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

        NOA_FHD constexpr void init(const auto& input, reduced_type& sum, reduced_type& error) const noexcept {
            auto value = static_cast<reduced_type>(input);
            kahan_sum(value, sum, error);
        }

        NOA_FHD constexpr void join(
                const reduced_type& local_sum, const reduced_type& local_error,
                reduced_type& global_sum, reduced_type& global_error
        ) const noexcept {
            global_sum += local_sum;
            global_error += local_error;
        }

        template<typename F>
        NOA_FHD constexpr void final(const reduced_type& global_sum, const reduced_type& global_error, F& final) {
            final = static_cast<F>(global_sum + global_error);
        }
    };

    template<typename T>
    struct ReduceAccurateMean : ReduceAccurateSum<T> {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        using reduced_type = ReduceAccurateSum<T>::reduced_type;
        using mean_type = nt::value_type_t<reduced_type>;
        mean_type mean;

        template<typename F>
        NOA_FHD constexpr void final(const reduced_type& global_sum, const reduced_type& global_error, F& final) {
            final = static_cast<F>((global_sum + global_error) / mean);
        }
    };

    struct ReduceAccurateL2Norm {
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

namespace noa {
    template<typename Accessor, typename Offset, bool SaveValue>
    struct ReduceArg {
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using offset_type = Offset;
        using reduced_type = Pair<value_type, offset_type>;
        accessor_type accessor;

    public:
        constexpr void init(const auto& indices, reduced_type& reduced) const noexcept {
            reduced.first = accessor(indices);
            reduced.second = static_cast<offset_type>(accessor.offset_at(indices));
        }

        template<typename T>
        constexpr void final(const reduced_type& reduced, T& output) const noexcept {
            if constexpr (SaveValue)
                output = static_cast<T>(reduced.first);
            else
                output = static_cast<T>(reduced.second);
        }

        template<typename T, typename U>
        constexpr void final(const reduced_type& reduced, T& value, U& offset) const noexcept {
            value = static_cast<T>(reduced.first);
            offset = static_cast<U>(reduced.second);
        }
    };

    template<typename Accessor, typename Offset = i64, bool SaveValue = false>
    struct ReduceFirstMin : ReduceArg<Accessor, Offset, SaveValue> {
        using base_type = ReduceArg<Accessor, Offset, SaveValue>;
        using remove_defaulted_final = bool;
        using accessor_type = base_type::accessor_type;
        using reduced_type = base_type::reduced_type;

        constexpr void join(const reduced_type& current, reduced_type& reduced) const noexcept {
            if (current.first < reduced.first or (current.first == reduced.first and current.second < reduced.second))
                reduced = current;
        }
    };

    template<typename Accessor, typename Offset = i64, bool SaveValue = false>
    struct ReduceFirstMax : ReduceArg<Accessor, Offset, SaveValue> {
        using base_type = ReduceArg<Accessor, Offset, SaveValue>;
        using remove_defaulted_final = bool;
        using accessor_type = base_type::accessor_type;
        using reduced_type = base_type::reduced_type;

        constexpr void join(const reduced_type& current, reduced_type& reduced) const noexcept {
            if (current.first > reduced.first or (reduced.first == current.first and current.second < reduced.second))
                reduced = current;
        }
    };

    template<typename Accessor, typename Offset = i64, bool SaveValue = false>
    struct ReduceLastMin : ReduceArg<Accessor, Offset, SaveValue> {
        using base_type = ReduceArg<Accessor, Offset, SaveValue>;
        using remove_defaulted_final = bool;
        using accessor_type = base_type::accessor_type;
        using reduced_type = base_type::reduced_type;

        constexpr void join(const reduced_type& current, reduced_type& reduced) const noexcept {
            if (current.first < reduced.first or (current.first == reduced.first and current.second > reduced.second))
                reduced = current;
        }
    };

    template<typename Accessor, typename Offset = i64, bool SaveValue = false>
    struct ReduceLastMax : ReduceArg<Accessor, Offset, SaveValue> {
        using base_type = ReduceArg<Accessor, Offset, SaveValue>;
        using remove_defaulted_final = bool;
        using accessor_type = base_type::accessor_type;
        using reduced_type = base_type::reduced_type;

        constexpr void join(const reduced_type& current, reduced_type& reduced) const noexcept {
            if (current.first > reduced.first or (reduced.first == current.first and current.second > reduced.second))
                reduced = current;
        }
    };
}
