#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::math::details {
    using namespace ::noa::traits;
    template<typename T, typename U>
    constexpr bool is_valid_random_v
            = is_any_v<T, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, half_t, float, double, chalf_t, cfloat_t, cdouble_t> &&
              (std::is_same_v<value_type_t<T>, U> || std::is_same_v<T, U> ||
              (std::is_same_v<T, half_t> && is_float_v<U>) || (std::is_same_v<T, chalf_t> && is_float_v<value_type_t<U>>));
}

namespace noa::cpu::math {
    /// Randomizes an array with uniform random values.
    /// \tparam T               Any data type.
    /// \tparam U               If \p T is real, \p U should be equal to \p T.
    ///                         If \p T is complex, \p U should be equal to \p T or its corresponding real type.
    ///                         If \p T is half_t, \p U can also be float.
    ///                         If \p T is chalf_t, \p U can also be float or cfloat_t.
    /// \param output           On the \b host. Array to randomize.
    /// \param elements         Number of elements in \p output to randomize.
    /// \param min, max         Minimum and maximum value of the uniform range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::uniform_t, const shared_t<T[]>& output, size_t elements, U min, U max, Stream& stream);

    /// Randomizes an array with normal random values.
    /// \tparam T               Any data type.
    /// \tparam U               If \p T is real, \p U should be equal to \p T.
    ///                         If \p T is complex, \p U should be equal to \p T or its corresponding real type.
    ///                         If \p T is half_t, \p U can also be float.
    ///                         If \p T is chalf_t, \p U can also be float or cfloat_t.
    /// \param output           On the \b host. Array to randomize.
    /// \param elements         Number of elements in \p output to randomize.
    /// \param mean, stddev     Mean and standard-deviation of the normal range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::normal_t, const shared_t<T[]>& output, size_t elements, U mean, U stddev, Stream& stream);

    /// Randomizes an array with log-normal random values.
    /// \tparam T               Any data type.
    /// \tparam U               If \p T is real, \p U should be equal to \p T.
    ///                         If \p T is complex, \p U should be equal to \p T or its corresponding real type.
    ///                         If \p T is half_t, \p U can also be float.
    ///                         If \p T is chalf_t, \p U can also be float or cfloat_t.
    /// \param output           On the \b host. Array to randomize.
    /// \param elements         Number of elements in \p output to randomize.
    /// \param mean, stddev     Mean and standard-deviation of the log-normal range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::log_normal_t, const shared_t<T[]>& output, size_t elements, U mean, U stddev, Stream& stream);

    /// Randomizes an array with poisson random values.
    /// \tparam T               Any data type.
    /// \param output           On the \b host. Array to randomize.
    /// \param elements         Number of elements in \p output to randomize.
    /// \param lambda           Mean value of the poisson range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<details::is_valid_random_v<T, traits::value_type_t<T>>>>
    void randomize(noa::math::poisson_t, const shared_t<T[]>& output, size_t elements, float lambda, Stream& stream);

    /// Randomizes an array with uniform random values.
    /// \tparam T               Any data type.
    /// \tparam U               If \p T is real, \p U should be equal to \p T.
    ///                         If \p T is complex, \p U should be equal to \p T or its corresponding real type.
    ///                         If \p T is half_t, \p U can also be float.
    ///                         If \p T is chalf_t, \p U can also be float or cfloat_t.
    /// \param output           On the \b host. Array to randomize.
    /// \param strides          Strides of \p output.
    /// \param shape            Shape of \p output.
    /// \param min, max         Minimum and maximum value of the uniform range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::uniform_t, const shared_t<T[]>& output, size4_t strides, size4_t shape,
                   U min, U max, Stream& stream);

    /// Randomizes an array with normal random values.
    /// \tparam T               Any data type.
    /// \tparam U               If \p T is real, \p U should be equal to \p T.
    ///                         If \p T is complex, \p U should be equal to \p T or its corresponding real type.
    ///                         If \p T is half_t, \p U can also be float.
    ///                         If \p T is chalf_t, \p U can also be float or cfloat_t.
    /// \param output           On the \b host. Array to randomize.
    /// \param strides          Strides of \p output.
    /// \param shape            Shape of \p output.
    /// \param mean, stddev     Mean and standard-deviation of the normal range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::normal_t, const shared_t<T[]>& output, size4_t strides, size4_t shape,
                   U mean, U stddev, Stream& stream);

    /// Randomizes an array with log-normal random values.
    /// \tparam T               Any data type.
    /// \tparam U               If \p T is real, \p U should be equal to \p T.
    ///                         If \p T is complex, \p U should be equal to \p T or its corresponding real type.
    ///                         If \p T is half_t, \p U can also be float.
    ///                         If \p T is chalf_t, \p U can also be float or cfloat_t.
    /// \param output           On the \b host. Array to randomize.
    /// \param strides          Strides of \p output.
    /// \param shape            Shape of \p output.
    /// \param mean, stddev     Mean and standard-deviation of the log-normal range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::log_normal_t, const shared_t<T[]>& output, size4_t strides, size4_t shape,
                   U mean, U stddev, Stream& stream);

    /// Randomizes an array with poisson random values.
    /// \tparam T               Any data type.
    /// \param output           On the \b host. Array to randomize.
    /// \param strides          Strides of \p output.
    /// \param shape            Shape of \p output.
    /// \param lambda           Mean value of the poisson range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<details::is_valid_random_v<T, traits::value_type_t<T>>>>
    void randomize(noa::math::poisson_t, const shared_t<T[]>& output, size4_t strides, size4_t shape,
                   float lambda, Stream& stream);
}
