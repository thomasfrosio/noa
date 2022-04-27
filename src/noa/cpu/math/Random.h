#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::math {
    /// Randomizes an array with uniform random values.
    /// \tparam T               Any data type.
    /// \tparam U               If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                         If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                         Otherwise, \p U should be equal to \p T.
    /// \param output           On the \b host. Array to randomize.
    /// \param elements         Number of elements in \p output to randomize.
    /// \param min, max         Minimum and maximum value of the uniform range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U>
    void randomize(noa::math::uniform_t, const shared_t<T[]>& output, size_t elements, U min, U max, Stream& stream);

    /// Randomizes an array with normal random values.
    /// \tparam T               Any data type.
    /// \tparam U               If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                         If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                         Otherwise, \p U should be equal to \p T.
    /// \param output           On the \b host. Array to randomize.
    /// \param elements         Number of elements in \p output to randomize.
    /// \param mean, stddev     Mean and standard-deviation of the normal range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U>
    void randomize(noa::math::normal_t, const shared_t<T[]>& output, size_t elements, U mean, U stddev, Stream& stream);

    /// Randomizes an array with log-normal random values.
    /// \tparam T               Any data type.
    /// \tparam U               If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                         If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                         Otherwise, \p U should be equal to \p T.
    /// \param output           On the \b host. Array to randomize.
    /// \param elements         Number of elements in \p output to randomize.
    /// \param mean, stddev     Mean and standard-deviation of the log-normal range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U>
    void randomize(noa::math::log_normal_t, const shared_t<T[]>& output, size_t elements, U mean, U stddev, Stream& stream);

    /// Randomizes an array with poisson random values.
    /// \tparam T               Any data type.
    /// \param output           On the \b host. Array to randomize.
    /// \param elements         Number of elements in \p output to randomize.
    /// \param lambda           Mean value of the poisson range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    void randomize(noa::math::poisson_t, const shared_t<T[]>& output, size_t elements, float lambda, Stream& stream);

    /// Randomizes an array with uniform random values.
    /// \tparam T               Any data type.
    /// \tparam U               If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                         If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                         Otherwise, \p U should be equal to \p T.
    /// \param output           On the \b host. Array to randomize.
    /// \param stride           Rightmost stride of \p output.
    /// \param shape            Rightmost shape of \p output.
    /// \param min, max         Minimum and maximum value of the uniform range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U>
    void randomize(noa::math::uniform_t, const shared_t<T[]>& output, size4_t stride, size4_t shape,
                   U min, U max, Stream& stream);

    /// Randomizes an array with normal random values.
    /// \tparam T               Any data type.
    /// \tparam U               If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                         If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                         Otherwise, \p U should be equal to \p T.
    /// \param output           On the \b host. Array to randomize.
    /// \param stride           Rightmost stride of \p output.
    /// \param shape            Rightmost shape of \p output.
    /// \param mean, stddev     Mean and standard-deviation of the normal range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U>
    void randomize(noa::math::normal_t, const shared_t<T[]>& output, size4_t stride, size4_t shape,
                   U mean, U stddev, Stream& stream);

    /// Randomizes an array with log-normal random values.
    /// \tparam T               Any data type.
    /// \tparam U               If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                         If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                         Otherwise, \p U should be equal to \p T.
    /// \param output           On the \b host. Array to randomize.
    /// \param stride           Rightmost stride of \p output.
    /// \param shape            Rightmost shape of \p output.
    /// \param mean, stddev     Mean and standard-deviation of the log-normal range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U>
    void randomize(noa::math::log_normal_t, const shared_t<T[]>& output, size4_t stride, size4_t shape,
                   U mean, U stddev, Stream& stream);

    /// Randomizes an array with poisson random values.
    /// \tparam T               Any data type.
    /// \param output           On the \b host. Array to randomize.
    /// \param stride           Rightmost stride of \p output.
    /// \param shape            Rightmost shape of \p output.
    /// \param lambda           Mean value of the poisson range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    void randomize(noa::math::poisson_t, const shared_t<T[]>& output, size4_t stride, size4_t shape,
                   float lambda, Stream& stream);
}
