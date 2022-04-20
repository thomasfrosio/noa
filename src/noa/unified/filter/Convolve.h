#pragma once

#include "noa/cpu/filter/Convolve.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/filter/Convolve.h"
#endif

#include "noa/unified/Array.h"

namespace noa::filter {
    /// 1D convolution.
    /// \tparam T           half_t, float, double.
    /// \tparam U           Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input    Array to convolve.
    /// \param[out] output  Convolved array. Should not overlap with \p input.
    /// \param[in] filter   1D, 2D or 3D filter.
    ///                     Dimensions should have an odd number of elements.
    ///                     Dimensions don't have to have the same size.
    ///
    /// \note If the output is on the GPU:\n
    ///         - \p U should be equal to \p T.\n
    ///         - \p filter can be on any device, including the CPU.\n
    ///         - \p filter size on each dimension is limited to 129 (1D), 17 (2D) and 5 (3) elements.
    ///         - This function modifies the GPU state via the usage of constant memory. As such,
    ///           there should be no concurrent calls from different streams associated to the same GPU.
    template<typename T, typename U>
    void convolve(const Array<T>& input, const Array<T>& output, const Array<U>& filter) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const size4_t filter_shape = filter.shape();
        NOA_CHECK(all(filter.contiguous()),
                  "The input filter must be contiguous, but got shape {} and stride {}",
                  filter_shape, filter.stride());
        NOA_CHECK(filter_shape.ndim() <= 3 && all(filter_shape % 2 == 1),
                  "The input filter should be 1D, 2D or 3D and each dimension should have an odd number of elements, "
                  "but got shape {}", filter_shape);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == filter.device(),
                      "The input arrays must be on the same device, but got input:{}, filter:{}",
                      input.device(), filter.device());
            cpu::filter::convolve(input.share(), input_stride,
                                  output.share(), output.stride(), output.shape(),
                                  filter.share(), size3_t{filter_shape.get() + 1}, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::filter::convolve(input.share(), input_stride,
                                   output.share(), output.stride(), output.shape(),
                                   filter.share(), size3_t{filter_shape.get() + 1}, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Separable convolutions. \p input is convolved with \p filter0, then \p filter1, then \p filter2.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input        Input array to convolve.
    /// \param[out] output      Output convolved array. Should not overlap with \p input.
    /// \param[in] filter1      Filter applied along the third-most dimension. Can be empty.
    ///                         Should be 1D filter with an odd number of elements, from 1 to 129.
    /// \param[in] filter2      Filter applied along the second-most dimension. Can be empty.
    ///                         Should be 1D filter with an odd number of elements, from 1 to 129.
    /// \param[in] filter3      Filter applied along the innermost dimension. Can be empty.
    ///                         Should be 1D filter with an odd number of elements, from 1 to 129.
    /// \note If a filter is empty, the convolution in the corresponding dimension is not applied and it goes
    ///       directly to the next filter, if any. Filters can be equal to each other.
    template<typename T, typename U>
    void convolve(const Array<T>& input, const Array<T>& output,
                  const Array<U>& filter1, const Array<U>& filter2, const Array<U>& filter3) {
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

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::filter::convolve(input.share(), input_stride,
                                  output.share(), output.stride(), output.shape(),
                                  filter1.share(), filter1.shape()[3],
                                  filter2.share(), filter2.shape()[3],
                                  filter3.share(), filter3.shape()[3],
                                  stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::filter::convolve(input.share(), input_stride,
                                   output.share(), output.stride(), output.shape(),
                                   filter1.share(), filter1.shape()[3],
                                   filter2.share(), filter2.shape()[3],
                                   filter3.share(), filter3.shape()[3],
                                   stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
