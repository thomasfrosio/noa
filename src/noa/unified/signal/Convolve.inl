#pragma once

#ifndef NOA_UNIFIED_CONVOLVE_
#error "This is a private header"
#endif

#include "noa/cpu/signal/Convolve.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/Convolve.h"
namespace noa::filter::details {
    constexpr size_t CUDA_FILTER_MAX_BYTES = 1032;
}
#endif

namespace noa::filter {
    template<typename T, typename U, typename>
    void convolve(const Array<T>& input, const Array<T>& output, const Array<U>& filter) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const size4_t filter_shape = filter.shape();
        size_t ndim = output.shape().ndim();
        ndim = ndim == 4 ? 3 : ndim;
        NOA_CHECK(all(filter.contiguous()),
                  "The input filter must be contiguous, but got shape {} and stride {}",
                  filter_shape, filter.stride());
        NOA_CHECK(filter_shape.ndim() <= ndim && all(filter_shape % 2 == 1),
                  "Given a {}N (batched) output, the input filter should be {}D at most and each dimension should have "
                  "an odd number of elements, but got filter shape {}", ndim, ndim, filter_shape);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == filter.device(),
                      "The input arrays must be on the same device, but got input:{}, filter:{}",
                      input.device(), filter.device());
            cpu::signal::convolve(input.share(), input_stride,
                                  output.share(), output.stride(), output.shape(),
                                  filter.share(), size3_t{filter_shape.get() + 1}, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
                NOA_CHECK(filter_shape.elements() * sizeof(T) <= details::CUDA_FILTER_MAX_BYTES,
                          "In the CUDA backend, the filter size is limited to {} bytes, but got {} of {} type",
                          details::CUDA_FILTER_MAX_BYTES, filter_shape, string::human<U>());
                cuda::signal::convolve(input.share(), input_stride,
                                       output.share(), output.stride(), output.shape(),
                                       filter.share(), size3_t{filter_shape.get() + 1}, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename U, typename>
    void convolve(const Array<T>& input, const Array<T>& output,
                  const Array<U>& filter1, const Array<U>& filter2, const Array<U>& filter3,
                  const Array<T>& tmp) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        const std::array<const Array<U>&, 3> filters = {filter1, filter2, filter3};
        for (auto i: irange<size_t>(3)) {
            const auto& filter = filters[i];
            if (filter.empty())
                continue;
            NOA_CHECK(all(filter.contiguous()),
                      "The input filters must be contiguous, but got filter{} with shape {} and stride {}",
                      i + 1, filter.shape(), filter.stride());
            NOA_CHECK(filter.ndim() == 1 && filter.shape()[3] % 2 == 1,
                      "The input filters should be row vectors with an odd number of elements, "
                      "but got filter{} with a shape of {}", i + 1, filter.shape());
            NOA_CHECK(device.gpu() || filter.device().cpu(),
                      "The input filter must be on the same device as the input, but got input:{}, filter{}:{}",
                      input.device(), i + 1, filter.device());
        }

        if (!tmp.empty()) {
            NOA_CHECK(all(tmp.shape() == output.shape()) && !any(tmp.stride() == 0),
                      "The temporary array should be able to hold an array of shape {}, but got shape {} and stride {}",
                      output.shape(), tmp.shape(), tmp.stride());
            NOA_CHECK(device == tmp.device(),
                      "The temporary array must be on the same device as the output, but got tmp:{}, output:{}",
                      tmp.device(), device);
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            if (tmp.empty()) {
                cpu::signal::convolve(input.share(), input_stride,
                                      output.share(), output.stride(), output.shape(),
                                      filter1.share(), filter1.shape()[3],
                                      filter2.share(), filter2.shape()[3],
                                      filter3.share(), filter3.shape()[3],
                                      stream.cpu());
            } else {
                cpu::signal::convolve(input.share(), input_stride,
                                      output.share(), output.stride(), output.shape(),
                                      filter1.share(), filter1.shape()[3],
                                      filter2.share(), filter2.shape()[3],
                                      filter3.share(), filter3.shape()[3],
                                      tmp.share(), tmp.stride(), stream.cpu());
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
                NOA_CHECK(filter1.shape()[3] * sizeof(T) <= details::CUDA_FILTER_MAX_BYTES &&
                          filter2.shape()[3] * sizeof(T) <= details::CUDA_FILTER_MAX_BYTES &&
                          filter3.shape()[3] * sizeof(T) <= details::CUDA_FILTER_MAX_BYTES,
                          "In the CUDA backend, separable filters have a size limited to {} bytes",
                          details::CUDA_FILTER_MAX_BYTES);
            if (tmp.empty()) {
                cuda::signal::convolve(input.share(), input_stride,
                                       output.share(), output.stride(), output.shape(),
                                       filter1.share(), filter1.shape()[3],
                                       filter2.share(), filter2.shape()[3],
                                       filter3.share(), filter3.shape()[3],
                                       stream.cuda());
            } else {
                cuda::signal::convolve(input.share(), input_stride,
                                       output.share(), output.stride(), output.shape(),
                                       filter1.share(), filter1.shape()[3],
                                       filter2.share(), filter2.shape()[3],
                                       filter3.share(), filter3.shape()[3],
                                       tmp.share(), tmp.stride(), stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
