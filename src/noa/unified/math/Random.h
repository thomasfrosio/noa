#pragma once

#include "noa/unified/Array.h"

namespace noa::math::details {
    using namespace ::noa::traits;
    template<typename T, typename U>
    constexpr bool is_valid_random_v
            = is_any_v<T, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t,
                       half_t, float, double, chalf_t, cfloat_t, cdouble_t> && traits::is_scalar_v<U>;
}

namespace noa::math {
    /// Randomizes an array with uniform random values.
    /// \tparam T       Any data type.
    /// \tparam U       Any scalar type.
    /// \param output   Array to randomize.
    /// \param min, max Minimum and maximum value of the uniform range.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::uniform_t, const Array<T>& output, U min, U max);

    /// Randomizes an array with normal random values.
    /// \tparam T           Any data type.
    /// \tparam U           Any scalar type.
    /// \param output       Array to randomize.
    /// \param mean, stddev Mean and standard-deviation of the normal range.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::normal_t, const Array<T>& output, U mean = U{0}, U stddev = U{1});

    /// Randomizes an array with log-normal random values.
    /// \tparam T           Any data type.
    /// \tparam U           Any scalar type.
    /// \param output       Array to randomize.
    /// \param mean, stddev Mean and standard-deviation of the log-normal range.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::log_normal_t, const Array<T>& output, U mean = U{0}, U stddev = U{1});

    /// Randomizes an array with poisson random values.
    /// \tparam T       Any data type.
    /// \param output   Array to randomize.
    /// \param lambda   Mean value of the poisson range.
    template<typename T, typename = std::enable_if_t<details::is_valid_random_v<T, traits::value_type_t<T>>>>
    void randomize(noa::math::poisson_t, const Array<T>& output, float lambda);
}

namespace noa::math {
    /// Randomizes an array with uniform random values.
    /// \tparam T       Any data type.
    /// \tparam U       Any scalar type.
    /// \param shape    Rightmost shape of the array.
    /// \param min, max Minimum and maximum value of the uniform range.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    Array<T> random(noa::math::uniform_t, size4_t shape, U min, U max, ArrayOption option = {});

    /// Randomizes an array with normal random values.
    /// \tparam T           Any data type.
    /// \tparam U           Any scalar type.
    /// \param shape        Rightmost shape of the array.
    /// \param mean, stddev Mean and standard-deviation of the normal range.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    Array<T> random(noa::math::normal_t, size4_t shape, U mean = U{0}, U stddev = U{1}, ArrayOption option = {});

    /// Randomizes an array with log-normal random values.
    /// \tparam T           Any data type.
    /// \tparam U           Any scalar type.
    /// \param shape        Rightmost shape of the array.
    /// \param mean, stddev Mean and standard-deviation of the log-normal range.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    Array<T> random(noa::math::log_normal_t, size4_t shape, U mean = U{0}, U stddev = U{1}, ArrayOption option = {});

    /// Randomizes an array with poisson random values.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param lambda   Mean value of the poisson range.
    template<typename T, typename = std::enable_if_t<details::is_valid_random_v<T, traits::value_type_t<T>>>>
    Array<T> random(noa::math::poisson_t, size4_t shape, float lambda, ArrayOption option = {});

    /// Randomizes an array with uniform random values.
    /// \tparam T       Any data type.
    /// \tparam U       Any scalar type.
    /// \param elements Number of elements to generate.
    /// \param min, max Minimum and maximum value of the uniform range.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    Array<T> random(noa::math::uniform_t, size_t elements, U min, U max, ArrayOption option = {});

    /// Randomizes an array with normal random values.
    /// \tparam T           Any data type.
    /// \tparam U           Any scalar type.
    /// \param elements     Number of elements to generate.
    /// \param mean, stddev Mean and standard-deviation of the normal range.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    Array<T> random(noa::math::normal_t, size_t elements, U mean = U{0}, U stddev = U{1}, ArrayOption option = {});

    /// Randomizes an array with log-normal random values.
    /// \tparam T           Any data type.
    /// \tparam U           Any scalar type.
    /// \param elements     Number of elements to generate.
    /// \param mean, stddev Mean and standard-deviation of the log-normal range.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    Array<T> random(noa::math::log_normal_t, size_t elements, U mean = U{0}, U stddev = U{1}, ArrayOption option = {});

    /// Randomizes an array with poisson random values.
    /// \tparam T       Any data type.
    /// \param elements Number of elements to generate.
    /// \param lambda   Mean value of the poisson range.
    template<typename T, typename = std::enable_if_t<details::is_valid_random_v<T, traits::value_type_t<T>>>>
    Array<T> random(noa::math::poisson_t, size_t elements, float lambda, ArrayOption option = {});
}

#define NOA_UNIFIED_RANDOM_
#include "noa/unified/math/Random.inl"
#undef NOA_UNIFIED_RANDOM_
