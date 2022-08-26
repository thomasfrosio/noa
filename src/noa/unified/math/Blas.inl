#pragma once

#ifndef NOA_UNIFIED_BLAS_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/math/Blas.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Blas.h"
#endif

namespace noa::math {
    template<typename T, typename>
    T dot(const Array<T>& lhs, const Array<T>& rhs) {
        NOA_CHECK(lhs.shape().ndim() <= 2 && rhs.shape().ndim() <= 2,
                  "The inputs should be unbatched row or column vectors, but got shape lhs:{} and rhs:{}",
                  lhs.shape(), rhs.shape());

        const Device device = rhs.device();
        NOA_CHECK(device == lhs.device(),
                  "The input arrays must be on the same device, but got lhs:{} and rhs:{}",
                  lhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::dot(lhs.share(), lhs.strides(), lhs.shape(),
                           rhs.share(), rhs.strides(), rhs.shape(),
                           stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::dot(lhs.share(), lhs.strides(), lhs.shape(),
                            rhs.share(), rhs.strides(), rhs.shape(),
                            stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void dot(const Array<T>& lhs, const Array<T>& rhs, const Array<T>& output) {
        NOA_CHECK(indexing::isVector(lhs.shape(), true) && indexing::isVector(rhs.shape(), true) &&
                  lhs.shape()[1] == 1 && rhs.shape()[1] == 1,
                  "The input should be (batched) column or row vectors, but got lhs:{} and rhs:{}",
                  lhs.shape(), rhs.shape());
        NOA_CHECK(lhs.shape()[2] * lhs.shape()[3] == rhs.shape()[2] * rhs.shape()[3],
                  "The input vectors don't have the same number of elements. Got lhs:{} and rhs:{}",
                  lhs.shape()[2] * lhs.shape()[3], rhs.shape()[2] * rhs.shape()[3]);

        NOA_CHECK(indexing::isVector(output.shape()) && output.contiguous(),
                  "The output should be a contiguous vector, but got shape {} and stride {}",
                  output.shape(), output.strides());

        const size_t batches = output.shape().elements();
        size4_t lhs_stride = lhs.strides();
        if (!indexing::broadcast(lhs.shape()[0], lhs_stride[0], batches))
            NOA_THROW("Cannot broadcast a size of {} into a size of {}", lhs.shape()[0], batches);
        size4_t rhs_stride = rhs.strides();
        if (!indexing::broadcast(rhs.shape()[0], rhs_stride[0], batches))
            NOA_THROW("Cannot broadcast a size of {} into a size of {}", rhs.shape()[0], batches);

        const Device device = lhs.device();
        NOA_CHECK(device == rhs.device(),
                  "The input arrays must be on the same device, but got lhs:{} and rhs:{}",
                  device, rhs.device());

        if (device != output.device())
            Stream::current(output.device()).synchronize();

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(output.dereferenceable(), "The output should be accessible to the CPU");
            cpu::math::dot(lhs.share(), lhs_stride, lhs.shape(),
                           rhs.share(), rhs_stride, rhs.shape(),
                           output.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::dot(lhs.share(), lhs_stride, lhs.shape(),
                            rhs.share(), rhs_stride, rhs.shape(),
                            output.share(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void matmul(const Array<T>& lhs, const Array<T>& rhs, const Array<T>& output) {
        matmul(lhs, rhs, output, T{1}, T{0});
    }

    template<typename T, typename>
    void matmul(const Array<T>& lhs, const Array<T>& rhs, const Array<T>& output,
                T alpha, T beta, bool lhs_transpose, bool rhs_transpose) {
        [[maybe_unused]] const bool is_col = indexing::isColMajor(lhs.strides());
        NOA_CHECK(is_col == indexing::isColMajor(rhs.strides()) &&
                  is_col == indexing::isColMajor(output.strides()),
                  "All matrices should either be row-major or column-major");

        [[maybe_unused]] const int innermost = 3 - is_col;
        [[maybe_unused]] const int secondmost = 2 + is_col;
        NOA_CHECK(lhs.strides()[innermost] == 1 && lhs.strides()[secondmost] >= lhs.shape()[innermost],
                  "The innermost dimension of the left-hand side should be contiguous and "
                  "the second-most dimension cannot be broadcast, but got shape:{} and stride:{}",
                  lhs.shape(), lhs.strides());
        NOA_CHECK(rhs.strides()[innermost] == 1 && rhs.strides()[secondmost] >= rhs.shape()[innermost],
                  "The innermost dimension of the right-hand side should be contiguous and "
                  "the second-most dimension cannot be broadcast, but got shape:{} and stride:{}",
                  rhs.shape(), rhs.strides());
        NOA_CHECK(output.strides()[innermost] == 1 && output.strides()[secondmost] >= output.shape()[innermost],
                  "The innermost dimension of the output should be contiguous and "
                  "the second-most dimension cannot be broadcast, but got shape:{} and stride:{}",
                  output.shape(), output.strides());

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
                  lhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::matmul(lhs.share(), lhs.strides(), lhs.shape(),
                              rhs.share(), rhs.strides(), rhs.shape(),
                              alpha, beta, lhs_transpose, rhs_transpose,
                              output.share(), output.strides(), output.shape(),
                              stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::matmul(lhs.share(), lhs.strides(), lhs.shape(),
                               rhs.share(), rhs.strides(), rhs.shape(),
                               alpha, beta, lhs_transpose, rhs_transpose,
                               output.share(), output.strides(), output.shape(),
                               stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}