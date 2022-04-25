#pragma once

#include "noa/cpu/math/Reduce.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Reduce.h"
#endif

#include "noa/unified/Array.h"

namespace noa::math {
    /// Returns the minimum value of the input array.
    /// \tparam T           (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] array    Array to reduce.
    template<typename T>
    [[nodiscard]] T min(const Array<T>& array) {
        NOA_PROFILE_FUNCTION();
        const Device device{array.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::min(array.share(), array.stride(), array.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::min(array.share(), array.stride(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the maximum value of the input array.
    /// \tparam T           (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] array    Array to reduce.
    template<typename T>
    [[nodiscard]] T max(const Array<T>& array) {
        NOA_PROFILE_FUNCTION();
        const Device device{array.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::max(array.share(), array.stride(), array.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::max(array.share(), array.stride(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the sum of the input array(s).
    /// \tparam T           (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] array    Array to reduce.
    template<typename T>
    [[nodiscard]] T sum(const Array<T>& array) {
        NOA_PROFILE_FUNCTION();
        const Device device{array.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::sum(array.share(), array.stride(), array.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::sum(array.share(), array.stride(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the mean of the input array.
    /// \tparam T           (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] array    Array to reduce.
    template<typename T>
    [[nodiscard]] T mean(const Array<T>& array) {
        NOA_PROFILE_FUNCTION();
        const Device device{array.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::mean(array.share(), array.stride(), array.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::mean(array.share(), array.stride(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the variance of the input array.
    /// \tparam DDOF        Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    ///                     If \p T is complex, return the corresponding real type.
    /// \param[in] array    Array to reduce.
    template<int DDOF = 0, typename T, typename U = noa::traits::value_type_t<T>>
    [[nodiscard]] U var(const Array<T>& array) {
        NOA_PROFILE_FUNCTION();
        const Device device{array.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::var<DDOF>(array.share(), array.stride(), array.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::var<DDOF>(array.share(), array.stride(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the standard-deviation of the input array.
    /// \tparam DDOF        Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    ///                     If \p T is complex, return the corresponding real type.
    /// \param[in] array    Array to reduce.
    template<int DDOF = 0, typename T, typename U = noa::traits::value_type_t<T>>
    [[nodiscard]] U std(const Array<T>& array) {
        NOA_PROFILE_FUNCTION();
        const Device device{array.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::std<DDOF>(array.share(), array.stride(), array.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::std<DDOF>(array.share(), array.stride(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the sum, mean, variance and stddev of the input array.
    /// \tparam DDOF        Delta Degree Of Freedom used to calculate the variance and standard deviation.
    ///                     In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    ///                     If \p T is complex, return the corresponding real type.
    /// \param[in] array    Array to reduce.
    template<int DDOF = 0, typename T, typename U = noa::traits::value_type_t<T>>
    [[nodiscard]] std::tuple<T, T, U, U> statistics(const Array<T>& array) {
        NOA_PROFILE_FUNCTION();
        const Device device{array.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::statistics<DDOF>(array.share(), array.stride(), array.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::statistics<DDOF>(array.share(), array.stride(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

// -- Reduce along particular axes -- //
namespace noa::math {
    /// Reduces an array along some dimensions by taking the minimum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam T           (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of minimum values.
    template<typename T>
    void min(const Array<T>& input, const Array<T>& output) {
        const Device device{output.device()};
        NOA_CHECK(input.get() != output.get(), "The input and output arrays should not overlap");
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::min(input.share(), input.stride(), input.shape(),
                           output.share(), output.stride(), output.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::min(input.share(), input.stride(), input.shape(),
                            output.share(), output.stride(), output.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the maximum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam T           (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of maximum values.
    template<typename T>
    void max(const Array<T>& input, const Array<T>& output) {
        const Device device{output.device()};
        NOA_CHECK(input.get() != output.get(), "The input and output arrays should not overlap");
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::max(input.share(), input.stride(), input.shape(),
                           output.share(), output.stride(), output.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::max(input.share(), input.stride(), input.shape(),
                            output.share(), output.stride(), output.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the sum.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam T           (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced sums.
    template<typename T>
    void sum(const Array<T>& input, const Array<T>& output) {
        const Device device{output.device()};
        NOA_CHECK(input.get() != output.get(), "The input and output arrays should not overlap");
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::sum(input.share(), input.stride(), input.shape(),
                           output.share(), output.stride(), output.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::sum(input.share(), input.stride(), input.shape(),
                            output.share(), output.stride(), output.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the mean.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam T           (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced means.
    template<typename T>
    void mean(const Array<T>& input, const Array<T>& output) {
        const Device device{output.device()};
        NOA_CHECK(input.get() != output.get(), "The input and output arrays should not overlap");
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::mean(input.share(), input.stride(), input.shape(),
                            output.share(), output.stride(), output.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::mean(input.share(), input.stride(), input.shape(),
                             output.share(), output.stride(), output.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the variance.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam DDOF        Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \tparam U           If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced variances.
    /// \note For complex types, the absolute value is taken before squaring, so the result is always real and positive.
    template<int DDOF = 0, typename T, typename U>
    void var(const Array<T>& input, const Array<T>& output) {
        const Device device{output.device()};
        NOA_CHECK(input.get() != output.get(), "The input and output arrays should not overlap");
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::var<DDOF>(input.share(), input.stride(), input.shape(),
                                 output.share(), output.stride(), output.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::var<DDOF>(input.share(), input.stride(), input.shape(),
                                  output.share(), output.stride(), output.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the standard-deviation.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam DDOF        Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \tparam U           If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced variances.
    /// \note For complex types, the absolute value is taken before squaring, so the result is always real and positive.
    template<int DDOF = 0, typename T, typename U>
    void std(const Array<T>& input, const Array<T>& output) {
        const Device device{output.device()};
        NOA_CHECK(input.get() != output.get(), "The input and output arrays should not overlap");
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::std<DDOF>(input.share(), input.stride(), input.shape(),
                                 output.share(), output.stride(), output.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::std<DDOF>(input.share(), input.stride(), input.shape(),
                                  output.share(), output.stride(), output.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
