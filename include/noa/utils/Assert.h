/**
 * @file Assert.h
 * @brief Various assertions.
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */
#pragma once

#include "Core.h"
#include "Traits.h"


/// Group of asserts.
namespace Noa::Assert {
    /**
     * @brief               Check that scalar(s) are within a given range, i.e min <= x <= max.
     *
     * @tparam [in] T       A scalar (is_arith_v) or a sequence (is_sequence_of_arith_v).
     * @tparam [in] U       A scalar (is_arith_v) or a sequence (is_sequence_of_arith_v).
     *                      If sequence, the ith element of a_value will be compared with
     *                      the ith element of a_min and a_max.
     * @param [in] a_value  Value(s) to assert.
     * @param [in] a_min    Value(s) to use as min.
     * @param [in] a_max    Value(s) to use as max.
     * @return              Whether or not the scalar(s) is within the range.
     *
     * @throw Noa::Error    If a_min and a_max are sequences, they must have the same size than a_value.
     *
     * @example
     * @code
     * float a{3.45f};
     * std::array<int, 2> b{0.5, 0.7};
     * std::vector<int> min{0, 0};
     * std::vector<int> max{0.5, 2};
     *
     * ::Noa::Assert::isWithin(a, 0f, 10f) // returns true
     * ::Noa::Assert::isWithin(b, 0, 0.5) // returns false
     * ::Noa::Assert::isWithin(b, min, max) // returns true
     * @endcode
     */
    template<typename T, typename U>
    inline bool isWithin(T&& a_value, U&& a_min, U&& a_max) {
        static_assert((Traits::is_arith_v<T> && Traits::is_arith_v<U>) ||
                      (Traits::is_sequence_of_arith_v<T> && Traits::is_arith_v<U>) ||
                      (Traits::is_sequence_of_arith_v<T> && Traits::is_sequence_of_arith_v<U>));

        if constexpr(Traits::is_arith_v<T> && Traits::is_arith_v<U>) {
            return (a_min <= a_value || a_max >= a_value);

        } else if constexpr(Traits::is_sequence_of_arith_v<T> && Traits::is_arith_v<U>) {
            for (auto& value : a_value) {
                if (a_min > value || a_max < value)
                    return false;
            }
        } else {
            if (a_value.size() != a_min.size() != a_max.size()) {
                NOA_ERROR("comparing sequences with different sizes, "
                          "got {} values for {} min and {} max",
                          a_value.size(), a_min.size(), a_max.size());
            }
            for (unsigned int i{0}; i < a_value.size(); ++i) {
                if (a_min[i] > a_value[i] || a_max[i] < a_value[i])
                    return false;
            }
        }
        return true;
    }


    /**
     * @brief               Check that scalar(s) are greater than the limit(s), i.e x > limit.
     *
     * @tparam [in] T       Same as Noa::Assert::isWithin.
     * @tparam [in] U       Same as Noa::Assert::isWithin.
     * @param [in] a_value  Value(s) to assert.
     * @param [in] a_limit  Value(s) to use as limit.
     * @return              Whether or not the scalar(s) are greater than the limit(s).
     *
     * @throw Noa::Error    If a_limit is a sequence, it must have the same size than a_value.
     */
    template<typename T, typename U>
    inline bool isGreaterThan(T&& a_value, U&& a_limit) {
        static_assert((Traits::is_arith_v<T> && Traits::is_arith_v<U>) ||
                      (Traits::is_sequence_of_arith_v<T> && Traits::is_arith_v<U>) ||
                      (Traits::is_sequence_of_arith_v<T> && Traits::is_sequence_of_arith_v<U>));

        if constexpr(Traits::is_arith_v<T> && Traits::is_arith_v<U>) {
            return (a_limit < a_value);

        } else if constexpr(Traits::is_sequence_of_arith_v<T> && Traits::is_arith_v<U>) {
            for (auto& value : a_value) {
                if (a_limit >= value)
                    return false;
            }
        } else {
            if (a_value.size() != a_limit.size()) {
                NOA_ERROR("comparing sequences with different sizes, got {} values for {} limits",
                          a_value.size(), a_limit.size());
            }
            for (unsigned int i{0}; i < a_value.size(); ++i) {
                if (a_limit[i] >= a_value[i])
                    return false;
            }
        }
        return true;
    }


    /**
     * @brief               Check that scalar(s) are greater or equal than the limit(s), i.e x >= limit.
     *
     * @tparam [in] T       Same as Noa::Assert::isWithin.
     * @tparam [in] U       Same as Noa::Assert::isWithin.
     * @param [in] a_value  Value(s) to assert.
     * @param [in] a_limit  Value(s) to use as limit.
     * @return              Whether or not the scalar(s) are greater or equal than the limit(s).
     *
     * @throw Noa::Error    If a_limit is a sequence, it must have the same size than a_value.
     */
    template<typename T, typename U>
    inline bool isGreaterOrEqualThan(T&& a_value, U&& a_limit) {
        static_assert((Traits::is_arith_v<T> && Traits::is_arith_v<U>) ||
                      (Traits::is_sequence_of_arith_v<T> && Traits::is_arith_v<U>) ||
                      (Traits::is_sequence_of_arith_v<T> && Traits::is_sequence_of_arith_v<U>));

        if constexpr(Traits::is_arith_v<T> && Traits::is_arith_v<U>) {
            return (a_limit <= a_value);

        } else if constexpr(Traits::is_sequence_of_arith_v<T> && Traits::is_arith_v<U>) {
            for (auto& value : a_value) {
                if (a_limit > value)
                    return false;
            }
        } else {
            if (a_value.size() != a_limit.size()) {
                NOA_ERROR("comparing sequences with different sizes, got {} values for {} limits",
                          a_value.size(), a_limit.size());
            }
            for (unsigned int i{0}; i < a_value.size(); ++i) {
                if (a_limit[i] > a_value[i])
                    return false;
            }
        }
        return true;
    }


    /**
     * @brief               Check that scalar(s) are lower than the limit(s), i.e x < limit.
     *
     * @tparam [in] T       Same as Noa::Assert::isWithin.
     * @tparam [in] U       Same as Noa::Assert::isWithin.
     * @param [in] a_value  Value(s) to assert.
     * @param [in] a_limit  Value(s) to use as limit.
     * @return              Whether or not the scalar(s) are lower than the limit(s).
     *
     * @throw Noa::Error    If a_limit is a sequence, it must have the same size than a_value.
     */
    template<typename T, typename U>
    inline bool isLowerThan(T&& a_value, U&& a_limit) {
        static_assert((Traits::is_arith_v<T> && Traits::is_arith_v<U>) ||
                      (Traits::is_sequence_of_arith_v<T> && Traits::is_arith_v<U>) ||
                      (Traits::is_sequence_of_arith_v<T> && Traits::is_sequence_of_arith_v<U>));

        if constexpr(Traits::is_arith_v<T> && Traits::is_arith_v<U>) {
            return (a_limit > a_value);

        } else if constexpr(Traits::is_sequence_of_arith_v<T> && Traits::is_arith_v<U>) {
            for (auto& value : a_value) {
                if (a_limit <= value)
                    return false;
            }
        } else {
            if (a_value.size() != a_limit.size()) {
                NOA_ERROR("comparing sequences with different sizes, got {} values for {} limits",
                          a_value.size(), a_limit.size());
            }
            for (unsigned int i{0}; i < a_value.size(); ++i) {
                if (a_limit[i] <= a_value[i])
                    return false;
            }
        }
        return true;
    }


    /**
     * @brief               Check that scalar(s) are lower or equal than the limit(s), i.e x <= limit.
     *
     * @tparam [in] T       Same as Noa::Assert::isWithin.
     * @tparam [in] U       Same as Noa::Assert::isWithin.
     * @param [in] a_value  Value(s) to assert.
     * @param [in] a_limit  Value(s) to use as limit.
     * @return              Whether or not the scalar(s) are lower or equal than the limit(s).
     *
     * @throw Noa::Error    If a_limit is a sequence, it must have the same size than a_value.
     */
    template<typename T, typename U>
    inline bool isLowerOrEqualThan(T&& a_value, U&& a_limit) {
        static_assert((Traits::is_arith_v<T> && Traits::is_arith_v<U>) ||
                      (Traits::is_sequence_of_arith_v<T> && Traits::is_arith_v<U>) ||
                      (Traits::is_sequence_of_arith_v<T> && Traits::is_sequence_of_arith_v<U>));

        if constexpr(Traits::is_arith_v<T> && Traits::is_arith_v<U>) {
            return (a_limit >= a_value);

        } else if constexpr(Traits::is_sequence_of_arith_v<T> && Traits::is_arith_v<U>) {
            for (auto& value : a_value) {
                if (a_limit < value)
                    return false;
            }
        } else {
            if (a_value.size() != a_limit.size()) {
                NOA_ERROR("comparing sequences with different sizes, got {} values for {} limits",
                          a_value.size(), a_limit.size());
            }
            for (unsigned int i{0}; i < a_value.size(); ++i) {
                if (a_limit[i] < a_value[i])
                    return false;
            }
        }
        return true;
    }
}
