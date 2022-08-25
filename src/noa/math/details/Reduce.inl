#pragma once

#ifndef NOA_UNIFIED_REDUCE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/math/Reduce.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Reduce.h"
#endif

namespace noa::math {
    template<typename T, typename>
    T min(const Array<T>& array) {
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::min(array.share(), array.strides(), array.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::min(array.share(), array.strides(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    T max(const Array<T>& array) {
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::max(array.share(), array.strides(), array.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::max(array.share(), array.strides(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    T sum(const Array<T>& array) {
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::sum(array.share(), array.strides(), array.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::sum(array.share(), array.strides(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    T mean(const Array<T>& array) {
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::mean(array.share(), array.strides(), array.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::mean(array.share(), array.strides(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    auto var(const Array<T>& array, int ddof) {
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::var(array.share(), array.strides(), array.shape(), ddof, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::var(array.share(), array.strides(), array.shape(), ddof, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    auto std(const Array<T>& array, int ddof) {
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::std(array.share(), array.strides(), array.shape(), ddof, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::std(array.share(), array.strides(), array.shape(), ddof, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    auto statistics(const Array<T>& array, int ddof) {
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::statistics(array.share(), array.strides(), array.shape(), ddof, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::statistics(array.share(), array.strides(), array.shape(), ddof, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

// -- Reduce along particular axes -- //
namespace noa::math {
    template<typename T, typename>
    void min(const Array<T>& input, const Array<T>& output) {
        const Device device = output.device();
        NOA_CHECK(input.get() != output.get(), "The input and output arrays should not overlap");

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{} and output:{}",
                      input.device(), device);
            cpu::math::min(input.share(), input.strides(), input.shape(),
                           output.share(), output.strides(), output.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            NOA_CHECK(device == input.device() || all(size3_t{output.shape().get() + 1} == 1),
                      "The input and output arrays must be on the same device, but got input:{} and output:{}",
                      input.device(), device);
            cuda::math::min(input.share(), input.strides(), input.shape(),
                            output.share(), output.strides(), output.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void max(const Array<T>& input, const Array<T>& output) {
        const Device device = output.device();
        NOA_CHECK(input.get() != output.get(), "The input and output arrays should not overlap");

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{} and output:{}",
                      input.device(), device);
            cpu::math::max(input.share(), input.strides(), input.shape(),
                           output.share(), output.strides(), output.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            NOA_CHECK(device == input.device() || all(size3_t{output.shape().get() + 1} == 1),
                      "The input and output arrays must be on the same device, but got input:{} and output:{}",
                      input.device(), device);
            cuda::math::max(input.share(), input.strides(), input.shape(),
                            output.share(), output.strides(), output.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void sum(const Array<T>& input, const Array<T>& output) {
        const Device device = output.device();
        NOA_CHECK(input.get() != output.get(), "The input and output arrays should not overlap");

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{} and output:{}",
                      input.device(), device);
            cpu::math::sum(input.share(), input.strides(), input.shape(),
                           output.share(), output.strides(), output.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            NOA_CHECK(device == input.device() || all(size3_t{output.shape().get() + 1} == 1),
                      "The input and output arrays must be on the same device, but got input:{} and output:{}",
                      input.device(), device);
            cuda::math::sum(input.share(), input.strides(), input.shape(),
                            output.share(), output.strides(), output.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void mean(const Array<T>& input, const Array<T>& output) {
        const Device device = output.device();
        NOA_CHECK(input.get() != output.get(), "The input and output arrays should not overlap");

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{} and output:{}",
                      input.device(), device);
            cpu::math::mean(input.share(), input.strides(), input.shape(),
                            output.share(), output.strides(), output.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            NOA_CHECK(device == input.device() || all(size3_t{output.shape().get() + 1} == 1),
                      "The input and output arrays must be on the same device, but got input:{} and output:{}",
                      input.device(), device);
            cuda::math::mean(input.share(), input.strides(), input.shape(),
                             output.share(), output.strides(), output.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename U, typename>
    void var(const Array<T>& input, const Array<U>& output, int ddof) {
        const Device device = output.device();
        NOA_CHECK(input.get() != output.get(), "The input and output arrays should not overlap");

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{} and output:{}",
                      input.device(), device);
            cpu::math::var(input.share(), input.strides(), input.shape(),
                           output.share(), output.strides(), output.shape(), ddof, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            NOA_CHECK(device == input.device() || all(size3_t{output.shape().get() + 1} == 1),
                      "The input and output arrays must be on the same device, but got input:{} and output:{}",
                      input.device(), device);
            cuda::math::var(input.share(), input.strides(), input.shape(),
                            output.share(), output.strides(), output.shape(), ddof, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename U, typename>
    void std(const Array<T>& input, const Array<U>& output, int ddof) {
        const Device device = output.device();
        NOA_CHECK(input.get() != output.get(), "The input and output arrays should not overlap");

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{} and output:{}",
                      input.device(), device);
            cpu::math::std(input.share(), input.strides(), input.shape(),
                           output.share(), output.strides(), output.shape(), ddof, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            NOA_CHECK(device == input.device() || all(size3_t{output.shape().get() + 1} == 1),
                      "The input and output arrays must be on the same device, but got input:{} and output:{}",
                      input.device(), device);
            cuda::math::std(input.share(), input.strides(), input.shape(),
                            output.share(), output.strides(), output.shape(), ddof, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
