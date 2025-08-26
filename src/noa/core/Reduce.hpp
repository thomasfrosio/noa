#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/math/Comparison.hpp"
#include "noa/core/math/Generic.hpp"

// Reduction operators
namespace noa {
    struct ReduceSum {
        using enable_vectorization = bool;
        using remove_default_final = bool;

        template<typename I, typename T>
        static constexpr void init(const I& value, T& sum) {
            sum += static_cast<T>(value);
        }
        template<typename I, typename J, typename T>
        static constexpr void init(const I& lhs, const J& rhs, T& sum) { // dot
            sum += static_cast<T>(lhs * rhs);
        }
        template<typename T>
        static constexpr void join(const T& ireduced, T& reduced) {
            reduced += ireduced;
        }
        template<typename T, typename O>
        static constexpr void final(const T& reduced, O& sum) {
            sum = static_cast<O>(reduced);
        }
    };

    struct ReduceSumKahan {
        using enable_vectorization = bool;
        using remove_default_final = bool;

        template<nt::real_or_complex I, typename T>
        static constexpr void init(const I& input, Vec<T, 2>& sum) {
            auto value = static_cast<T>(input);
            kahan_sum(value, sum[0], sum[1]);
        }
        template<nt::real_or_complex I, nt::real_or_complex J, typename T>
        static constexpr void init(const I& lhs, const J& rhs, Vec<T, 2>& sum) { // dot
            auto value = static_cast<T>(lhs * rhs);
            kahan_sum(value, sum[0], sum[1]);
        }
        template<typename T>
        static constexpr void join(const Vec<T, 2>& isum, Vec<T, 2>& sum) {
            sum += isum;
        }
        template<typename T, typename F>
        static constexpr void final(const Vec<T, 2>& sum, F& final) {
            final = static_cast<F>(sum[0] + sum[1]);
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
        static constexpr void join(const T& isum, T& sum) {
            sum += isum;
        }
        template<typename T, typename U>
        constexpr void final(const T& sum, U& mean) const {
            using tmp_t = std::conditional_t<nt::integer<T>, f64, T>;
            mean = static_cast<U>(static_cast<tmp_t>(sum) / size);
        }
    };

    template<nt::real_or_complex S>
    struct ReduceMeanKahan {
        using enable_vectorization = bool;
        using remove_default_final = bool;
        S size;

        template<nt::real_or_complex I, typename T>
        static constexpr void init(const I& input, Vec<T, 2>& sum) {
            auto value = static_cast<T>(input);
            kahan_sum(value, sum[0], sum[1]);
        }
        template<typename T>
        static constexpr void join(const Vec<T, 2>& isum, Vec<T, 2>& sum) {
            sum += isum;
        }
        template<typename T, typename F>
        constexpr void final(const Vec<T, 2>& sum, F& final) const {
            final = static_cast<F>((sum[0] + sum[1]) / size);
        }
    };

    struct ReduceL2Norm {
        using enable_vectorization = bool;
        using remove_default_final = bool;

        template<typename I, typename T>
        static constexpr void init(const I& value, T& sum) {
            sum += static_cast<T>(abs_squared(value));
        }
        template<typename T>
        static constexpr void join(const T& isum, T& sum) {
            sum += isum;
        }
        template<typename T, typename F>
        static constexpr void final(const T& sum, F& norm) {
            using tmp_t = std::conditional_t<nt::integer<T>, f64, T>;
            norm = static_cast<F>(sqrt(static_cast<tmp_t>(sum)));
        }
    };

    struct ReduceL2NormKahan {
        using enable_vectorization = bool;
        using remove_default_final = bool;

        template<typename I, nt::real T>
        static constexpr void init(const I& input, Vec<T, 2>& sum) {
            kahan_sum(static_cast<T>(abs_squared(input)), sum[0], sum[1]);
        }
        template<nt::real T>
        static constexpr void join(const Vec<T, 2>& isum, Vec<T, 2>& sum) {
            sum += isum;
        }
        template<nt::real T, typename F>
        static constexpr void final(const Vec<T, 2>& global_sum, F& final) {
            final = static_cast<F>(sqrt(global_sum[0] + global_sum[1]));
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
            const auto distance_sqd = abs_squared(static_cast<U>(value) - mean);
            reduced += static_cast<R>(distance_sqd);
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
            const auto distance_sqd = abs_squared(static_cast<U>(value) - mean);
            reduced += static_cast<R>(distance_sqd);
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

    template<nt::real S, bool STDDEV = false>
    struct ReduceMeanVariance {
        using enable_vectorization = bool;
        using remove_default_final = bool;
        S size{};
        S ddof{};

        template<typename I, typename T, typename U>
        static constexpr void init(const I& value, T& sum, U& sum_sqd) {
            const auto v = static_cast<T>(value);
            sum += v;
            sum_sqd += abs_squared(v);
        }
        template<typename T, typename U>
        static constexpr void join(const T& isum, const U& isum_sqd, T& sum, U& sum_sqd) {
            sum += isum;
            sum_sqd += isum_sqd;
        }
        template<typename T, typename U, typename V, typename W>
        constexpr void final(const T& sum, const U& sum_sqd, V& mean, W& variance) const {
            using t0 = std::conditional_t<nt::integer<T>, S, T>;
            using t1 = std::conditional_t<nt::integer<U>, S, U>;
            if constexpr (not nt::empty<V>)
                mean = static_cast<V>(static_cast<t0>(sum) / size);

            auto tmp = abs_squared(sum) / size;
            U variance_ = (static_cast<t1>(sum_sqd) - tmp) / (size - ddof);
            if constexpr (STDDEV)
                variance_ = sqrt(variance_);
            variance = static_cast<W>(variance_);
        }
        template<typename T, typename U, typename V>
        constexpr void final(const T& sum, const U& sum_sqd, V& variance) const {
            auto empty = Empty{};
            final(sum, sum_sqd, empty, variance);
        }
    };

    template<nt::real S, bool STDDEV = false>
    struct ReduceMeanVarianceKahan {
        using enable_vectorization = bool;
        using remove_default_final = bool;
        S size{};
        S ddof{};

        template<nt::real_or_complex I, typename T, typename U>
        constexpr void init(const I& value, Vec<T, 2>& sum, Vec<U, 2>& sum_sqd) {
            const auto x = static_cast<T>(value);
            noa::kahan_sum(x, sum[0], sum[1]);
            noa::kahan_sum(abs_squared(x), sum_sqd[0], sum_sqd[1]);
        }
        template<typename T, typename U>
        static constexpr void join(
            const Vec<T, 2>& isum,
            const Vec<U, 2>& isum_sqd,
            Vec<T, 2>& sum,
            Vec<U, 2>& sum_sqd
        ) {
            sum += isum;
            sum_sqd += isum_sqd;
        }
        template<typename T, typename U, typename V, typename W>
        constexpr void final(const Vec<T, 2>& sum, const Vec<U, 2>& sum_sqd, V& mean, W& variance) const {
            auto sum_ = sum[0] + sum[1];
            if constexpr (not nt::empty<V>)
                mean = static_cast<V>(sum_ / size);

            auto sum_sqd_ = sum_sqd[0] + sum_sqd[1];
            auto tmp = abs_squared(sum_) / size;
            U variance_ = (sum_sqd_ - tmp) / (size - ddof);
            if constexpr (STDDEV)
                variance_ = sqrt(variance_);
            variance = static_cast<W>(variance_);
        }
        template<typename T, typename U, typename V>
        constexpr void final(const Vec<T, 2>& sum, const Vec<U, 2>& sum_sqd, V& variance) const {
            auto empty = Empty{};
            final(sum, sum_sqd, empty, variance);
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
                output = cast_or_abs_squared<T>(reduced.first);
            else
                output = static_cast<T>(reduced.second);
        }

        template<typename T, typename U>
        static constexpr void final(const reduced_type& reduced, T& value, U& offset) {
            value = cast_or_abs_squared<T>(reduced.first);
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
