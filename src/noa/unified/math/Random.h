#pragma once

#include "noa/cpu/math/Random.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Random.h"
#endif

#include "noa/unified/Array.h"

namespace noa::math::details {
    template<typename T>
    using supported_type = std::conditional_t<
            noa::traits::is_almost_same_v<noa::traits::value_type_t<T>, half_t>, double, noa::traits::value_type_t<T>>;
}

namespace noa::math {
    /// Randomizes an array with uniform random values.
    /// \tparam T       Any data type.
    /// \tparam U       If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                 If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                 Otherwise, \p U should be equal to \p T.
    /// \param output   Array to randomize.
    /// \param min, max Minimum and maximum value of the uniform range.
    template<typename T, typename U>
    void randomize(noa::math::uniform_t, const Array<T>& output, U min, U max) {
        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::randomize(noa::math::uniform_t{}, output.share(), output.stride(), output.shape(),
                                 static_cast<details::supported_type<T>>(min),
                                 static_cast<details::supported_type<T>>(max), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::randomize(noa::math::uniform_t{}, output.share(),
                                  output.stride(), output.shape(),
                                  static_cast<details::supported_type<T>>(min),
                                  static_cast<details::supported_type<T>>(max), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Randomizes an array with normal random values.
    /// \tparam T           Any data type.
    /// \tparam U           If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                     If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                     Otherwise, \p U should be equal to \p T.
    /// \param output       Array to randomize.
    /// \param mean, stddev Mean and standard-deviation of the normal range.
    template<typename T, typename U>
    void randomize(noa::math::normal_t, const Array<T>& output, U mean = U{0}, U stddev = U{1}) {
        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::randomize(noa::math::normal_t{}, output.share(),
                                 output.stride(), output.shape(),
                                 static_cast<details::supported_type<T>>(mean),
                                 static_cast<details::supported_type<T>>(stddev), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::randomize(noa::math::normal_t{}, output.share(),
                                  output.stride(), output.shape(),
                                  static_cast<details::supported_type<T>>(mean),
                                  static_cast<details::supported_type<T>>(stddev), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Randomizes an array with log-normal random values.
    /// \tparam T           Any data type.
    /// \tparam U           If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                     If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                     Otherwise, \p U should be equal to \p T.
    /// \param output       Array to randomize.
    /// \param mean, stddev Mean and standard-deviation of the log-normal range.
    template<typename T, typename U>
    void randomize(noa::math::log_normal_t, const Array<T>& output, U mean = U{0}, U stddev = U{1}) {
        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::randomize(noa::math::log_normal_t{}, output.share(),
                                 output.stride(), output.shape(),
                                 static_cast<details::supported_type<T>>(mean),
                                 static_cast<details::supported_type<T>>(stddev), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::randomize(noa::math::log_normal_t{}, output.share(),
                                  output.stride(), output.shape(),
                                  static_cast<details::supported_type<T>>(mean),
                                  static_cast<details::supported_type<T>>(stddev), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Randomizes an array with poisson random values.
    /// \tparam T       Any data type.
    /// \param output   Array to randomize.
    /// \param lambda   Mean value of the poisson range.
    template<typename T>
    void randomize(noa::math::poisson_t, const Array<T>& output, float lambda) {
        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::randomize(noa::math::poisson_t{}, output.share(),
                                 output.stride(), output.shape(),
                                 lambda, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::randomize(noa::math::poisson_t{}, output.share(),
                                  output.stride(), output.shape(),
                                  lambda, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Randomizes an array with uniform random values.
    /// \tparam T       Any data type.
    /// \tparam U       If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                 If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                 Otherwise, \p U should be equal to \p T.
    /// \param shape    Rightmost shape of the array.
    /// \param min, max Minimum and maximum value of the uniform range.
    template<typename T, typename U>
    Array<T> random(noa::math::uniform_t, size4_t shape, U min, U max, ArrayOption option = {}) {
        Array<T> out{shape, option};
        randomize(noa::math::uniform_t{}, out, min, max);
        return out;
    }

    /// Randomizes an array with normal random values.
    /// \tparam T           Any data type.
    /// \tparam U           If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                     If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                     Otherwise, \p U should be equal to \p T.
    /// \param shape        Rightmost shape of the array.
    /// \param mean, stddev Mean and standard-deviation of the normal range.
    template<typename T, typename U>
    Array<T> random(noa::math::normal_t, size4_t shape, U mean = U{0}, U stddev = U{1}, ArrayOption option = {}) {
        Array<T> out{shape, option};
        randomize(noa::math::normal_t{}, out, mean, stddev);
        return out;
    }

    /// Randomizes an array with log-normal random values.
    /// \tparam T           Any data type.
    /// \tparam U           If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                     If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                     Otherwise, \p U should be equal to \p T.
    /// \param shape        Rightmost shape of the array.
    /// \param mean, stddev Mean and standard-deviation of the log-normal range.
    template<typename T, typename U>
    Array<T> random(noa::math::log_normal_t, size4_t shape, U mean = U{0}, U stddev = U{1}, ArrayOption option = {}) {
        Array<T> out{shape, option};
        randomize(noa::math::log_normal_t{}, out, mean, stddev);
        return out;
    }

    /// Randomizes an array with poisson random values.
    /// \tparam T       Any data type.
    /// \param shape    Rightmost shape of the array.
    /// \param lambda   Mean value of the poisson range.
    template<typename T>
    Array<T> random(noa::math::poisson_t, size4_t shape, float lambda, ArrayOption option = {}) {
        Array<T> out{shape, option};
        randomize(noa::math::poisson_t{}, out, lambda);
        return out;
    }

    /// Randomizes an array with uniform random values.
    /// \tparam T       Any data type.
    /// \tparam U       If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                 If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                 Otherwise, \p U should be equal to \p T.
    /// \param elements Number of elements to generate.
    /// \param min, max Minimum and maximum value of the uniform range.
    template<typename T, typename U>
    Array<T> random(noa::math::uniform_t, size_t elements, U min, U max, ArrayOption option = {}) {
        Array<T> out{elements, option};
        randomize(noa::math::uniform_t{}, out, min, max);
        return out;
    }

    /// Randomizes an array with normal random values.
    /// \tparam T           Any data type.
    /// \tparam U           If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                     If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                     Otherwise, \p U should be equal to \p T.
    /// \param elements     Number of elements to generate.
    /// \param mean, stddev Mean and standard-deviation of the normal range.
    template<typename T, typename U>
    Array<T> random(noa::math::normal_t, size_t elements, U mean = U{0}, U stddev = U{1}, ArrayOption option = {}) {
        Array<T> out{elements, option};
        randomize(noa::math::normal_t{}, out, mean, stddev);
        return out;
    }

    /// Randomizes an array with log-normal random values.
    /// \tparam T           Any data type.
    /// \tparam U           If \p T is half_t or chalf_t, \p U should be half_t, float or double.
    ///                     If \p T is cfloat_t or cdouble_t, \p U should be float or double, respectively.
    ///                     Otherwise, \p U should be equal to \p T.
    /// \param elements     Number of elements to generate.
    /// \param mean, stddev Mean and standard-deviation of the log-normal range.
    template<typename T, typename U>
    Array<T> random(noa::math::log_normal_t, size_t elements, U mean = U{0}, U stddev = U{1}, ArrayOption option = {}) {
        Array<T> out{elements, option};
        randomize(noa::math::log_normal_t{}, out, mean, stddev);
        return out;
    }

    /// Randomizes an array with poisson random values.
    /// \tparam T       Any data type.
    /// \param elements Number of elements to generate.
    /// \param lambda   Mean value of the poisson range.
    template<typename T>
    Array<T> random(noa::math::poisson_t, size_t elements, float lambda, ArrayOption option = {}) {
        Array<T> out{elements, option};
        randomize(noa::math::poisson_t{}, out, lambda);
        return out;
    }
}
