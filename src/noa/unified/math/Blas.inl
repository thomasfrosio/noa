#pragma once

#ifndef NOA_UNIFIED_BLAS_
#error "This is a private header"
#endif

#include "noa/cpu/math/Blas.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Blas.h"
#endif

namespace noa::math {
    /// Returns the vector-vector dot product.
    /// \tparam T       (u)int32_t, (u)int64_t, float, double, cfloat_t and cdouble_t.
    /// \param[in] lhs  Unbatched row or column vector.
    /// \param[in] rhs  Unbatched row or column vector.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
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
            cpu::math::dot(lhs.share(), lhs.stride(), lhs.shape(),
                           rhs.share(), rhs.stride(), rhs.shape(),
                           stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::dot(lhs.share(), lhs.stride(), lhs.shape(),
                            rhs.share(), rhs.stride(), rhs.shape(),
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

        NOA_CHECK(indexing::isVector(output.shape()) && all(output.contiguous()),
                  "The output should be a contiguous vector, but got shape {} and stride {}",
                  output.shape(), output.stride());

        const size_t batches = output.shape().elements();
        size4_t lhs_stride = lhs.stride(), rhs_stride = rhs.stride();
        if (!indexing::broadcast(lhs.shape()[0], lhs_stride[0], batches))
            NOA_THROW("Cannot broadcast a size of {} into a size of {}", lhs.shape()[0], batches);
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
            NOA_CHECK(output.dereferencable(), "The output should be accessible to the CPU");
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
        NOA_CHECK(lhs.stride()[3] == 1 && lhs.stride()[2] >= lhs.shape()[3],
                  "The innermost dimension of the left-hand side should be contiguous and "
                  "the second-most dimension cannot be broadcast, but got shape:{} and stride:{}",
                  lhs.shape(), lhs.stride());
        NOA_CHECK(rhs.stride()[3] == 1 && rhs.stride()[2] >= rhs.shape()[3],
                  "The innermost dimension of the right-hand side should be contiguous and "
                  "the second-most dimension cannot be broadcast, but got shape:{} and stride:{}",
                  rhs.shape(), rhs.stride());
        NOA_CHECK(output.stride()[3] == 1 && output.stride()[2] >= output.shape()[3],
                  "The innermost dimension of the output should be contiguous and "
                  "the second-most dimension cannot be broadcast, but got shape:{} and stride:{}",
                  output.shape(), output.stride());

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
                  lhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::matmul(lhs.share(), lhs.stride(), lhs.shape(),
                              rhs.share(), rhs.stride(), rhs.shape(),
                              alpha, beta, lhs_transpose, rhs_transpose,
                              output.share(), output.stride(), output.shape(),
                              stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::matmul(lhs.share(), lhs.stride(), lhs.shape(),
                               rhs.share(), rhs.stride(), rhs.shape(),
                               alpha, beta, lhs_transpose, rhs_transpose,
                               output.share(), output.stride(), output.shape(),
                               stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
