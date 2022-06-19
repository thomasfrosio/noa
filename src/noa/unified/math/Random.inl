#pragma once

#ifndef NOA_UNIFIED_RANDOM_
#error "This is a private header"
#endif

#include "noa/cpu/math/Random.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Random.h"
#endif

namespace noa::math::details {
    template<typename T, typename U>
    auto castToSupportedType_(U value) {
        if constexpr (traits::is_complex_v<T> && traits::is_complex_v<U>) {
            using supported_t = std::conditional_t<traits::is_almost_same_v<T, chalf_t>, cfloat_t, T>;
            return static_cast<supported_t>(value);
        } else {
            using real_t = traits::value_type_t<T>;
            using supported_t = std::conditional_t<traits::is_almost_same_v<real_t, half_t>, float, real_t>;
            return static_cast<supported_t>(value);
        }
    }
}

namespace noa::math {
    template<typename T, typename U, typename>
    void randomize(noa::math::uniform_t, const Array<T>& output, U min, U max) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::randomize(noa::math::uniform_t{}, output.share(), output.stride(), output.shape(),
                                 details::castToSupportedType_<T>(min),
                                 details::castToSupportedType_<T>(max), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::randomize(noa::math::uniform_t{}, output.share(), output.stride(), output.shape(),
                                  details::castToSupportedType_<T>(min),
                                  details::castToSupportedType_<T>(max), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename U, typename>
    void randomize(noa::math::normal_t, const Array<T>& output, U mean, U stddev) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::randomize(noa::math::normal_t{}, output.share(),
                                 output.stride(), output.shape(),
                                 details::castToSupportedType_<T>(mean),
                                 details::castToSupportedType_<T>(stddev), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::randomize(noa::math::normal_t{}, output.share(),
                                  output.stride(), output.shape(),
                                  details::castToSupportedType_<T>(mean),
                                  details::castToSupportedType_<T>(stddev), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename U, typename>
    void randomize(noa::math::log_normal_t, const Array<T>& output, U mean, U stddev) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::randomize(noa::math::log_normal_t{}, output.share(),
                                 output.stride(), output.shape(),
                                 details::castToSupportedType_<T>(mean),
                                 details::castToSupportedType_<T>(stddev), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::randomize(noa::math::log_normal_t{}, output.share(),
                                  output.stride(), output.shape(),
                                  details::castToSupportedType_<T>(mean),
                                  details::castToSupportedType_<T>(stddev), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void randomize(noa::math::poisson_t, const Array<T>& output, float lambda) {
        const Device device = output.device();
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
}

namespace noa::math {
    template<typename T, typename U, typename>
    Array<T> random(noa::math::uniform_t, size4_t shape, U min, U max, ArrayOption option) {
        Array<T> out{shape, option};
        randomize(noa::math::uniform_t{}, out, min, max);
        return out;
    }

    template<typename T, typename U, typename>
    Array<T> random(noa::math::normal_t, size4_t shape, U mean, U stddev, ArrayOption option) {
        Array<T> out{shape, option};
        randomize(noa::math::normal_t{}, out, mean, stddev);
        return out;
    }

    template<typename T, typename U, typename>
    Array<T> random(noa::math::log_normal_t, size4_t shape, U mean, U stddev, ArrayOption option) {
        Array<T> out{shape, option};
        randomize(noa::math::log_normal_t{}, out, mean, stddev);
        return out;
    }

    template<typename T, typename>
    Array<T> random(noa::math::poisson_t, size4_t shape, float lambda, ArrayOption option) {
        Array<T> out{shape, option};
        randomize(noa::math::poisson_t{}, out, lambda);
        return out;
    }

    template<typename T, typename U, typename>
    Array<T> random(noa::math::uniform_t, size_t elements, U min, U max, ArrayOption option) {
        Array<T> out{elements, option};
        randomize(noa::math::uniform_t{}, out, min, max);
        return out;
    }

    template<typename T, typename U, typename>
    Array<T> random(noa::math::normal_t, size_t elements, U mean, U stddev, ArrayOption option) {
        Array<T> out{elements, option};
        randomize(noa::math::normal_t{}, out, mean, stddev);
        return out;
    }

    template<typename T, typename U, typename>
    Array<T> random(noa::math::log_normal_t, size_t elements, U mean, U stddev, ArrayOption option) {
        Array<T> out{elements, option};
        randomize(noa::math::log_normal_t{}, out, mean, stddev);
        return out;
    }

    template<typename T, typename>
    Array<T> random(noa::math::poisson_t, size_t elements, float lambda, ArrayOption option) {
        Array<T> out{elements, option};
        randomize(noa::math::poisson_t{}, out, lambda);
        return out;
    }
}
