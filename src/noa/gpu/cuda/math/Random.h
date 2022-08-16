#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::math::details {
    using namespace ::noa::traits;
    template<typename T, typename U>
    constexpr bool is_valid_random_v
            = is_any_v<T, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, half_t, float, double, chalf_t, cfloat_t, cdouble_t> &&
              (std::is_same_v<value_type_t<T>, U> || std::is_same_v<T, U> ||
               (std::is_same_v<T, half_t> && is_float_v<U>) || (std::is_same_v<T, chalf_t> && is_float_v<value_type_t<U>>));
}

namespace noa::cuda::math {
    // Randomizes an array with uniform random values.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::uniform_t, const shared_t<T[]>& output, size_t elements, U min, U max, Stream& stream);

    // Randomizes an array with normal random values.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::normal_t, const shared_t<T[]>& output, size_t elements, U mean, U stddev, Stream& stream);

    // Randomizes an array with log-normal random values.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::log_normal_t, const shared_t<T[]>& output, size_t elements, U mean, U stddev, Stream& stream);

    // Randomizes an array with poisson random values.
    template<typename T, typename = std::enable_if_t<details::is_valid_random_v<T, traits::value_type_t<T>>>>
    void randomize(noa::math::poisson_t, const shared_t<T[]>& output, size_t elements, float lambda, Stream& stream);

    // Randomizes an array with uniform random values.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::uniform_t, const shared_t<T[]>& output, size4_t strides, size4_t shape,
                   U min, U max, Stream& stream);

    // Randomizes an array with normal random values.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::normal_t, const shared_t<T[]>& output, size4_t strides, size4_t shape,
                   U mean, U stddev, Stream& stream);

    // Randomizes an array with log-normal random values.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::log_normal_t, const shared_t<T[]>& output, size4_t strides, size4_t shape,
                   U mean, U stddev, Stream& stream);

    // Randomizes an array with poisson random values.
    template<typename T, typename = std::enable_if_t<details::is_valid_random_v<T, traits::value_type_t<T>>>>
    void randomize(noa::math::poisson_t, const shared_t<T[]>& output, size4_t strides, size4_t shape,
                   float lambda, Stream& stream);
}
