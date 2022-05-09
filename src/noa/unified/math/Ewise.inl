#pragma once

#ifndef NOA_UNIFIED_EWISE_
#error "This is a private header"
#endif

#include "noa/cpu/math/Ewise.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Ewise.h"
#endif

// -- Unary operators -- //
namespace noa::math {
    template<typename T, typename U, typename UnaryOp>
    void ewise(const Array<T>& input, const Array<U>& output, UnaryOp unary_op) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device{output.device()};
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(input.share(), input_stride,
                             output.share(), output.stride(), output.shape(),
                             unary_op, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_unary_v<T, U, UnaryOp>) {
                cuda::math::ewise(input.share(), input_stride,
                                  output.share(), output.stride(), output.shape(),
                                  unary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

// -- Binary operators -- //
namespace noa::math {
    template<typename T, typename U, typename V, typename BinaryOp, typename>
    void ewise(const Array<T>& lhs, U rhs, const Array<V>& output, BinaryOp binary_op) {
        size4_t lhs_stride = lhs.stride();
        if (!indexing::broadcast(lhs.shape(), lhs_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }

        const Device device{output.device()};
        NOA_CHECK(device == lhs.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  lhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs.share(), lhs_stride, rhs,
                             output.share(), output.stride(), output.shape(),
                             binary_op, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_binary_v<T, U, V, BinaryOp>) {
                cuda::math::ewise(lhs.share(), lhs_stride, rhs,
                                  output.share(), output.stride(), output.shape(),
                                  binary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename U, typename V, typename BinaryOp, typename>
    void ewise(T lhs, const Array<U>& rhs, const Array<V>& output, BinaryOp binary_op) {
        size4_t rhs_stride = rhs.stride();
        if (!indexing::broadcast(rhs.shape(), rhs_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device{output.device()};
        NOA_CHECK(device == rhs.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs, rhs.share(), rhs_stride,
                             output.share(), output.stride(), output.shape(),
                             binary_op, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_binary_v<T, U, V, BinaryOp>) {
                cuda::math::ewise(lhs, rhs.share(), rhs_stride,
                                  output.share(), output.stride(), output.shape(),
                                  binary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const Array<T>& lhs, const Array<U>& rhs, const Array<V>& output, BinaryOp binary_op) {
        size4_t lhs_stride = lhs.stride();
        if (!indexing::broadcast(lhs.shape(), lhs_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }
        size4_t rhs_stride = rhs.stride();
        if (!indexing::broadcast(rhs.shape(), rhs_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device{output.device()};
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
                  lhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs.share(), lhs_stride, rhs.share(), rhs_stride,
                             output.share(), output.stride(), output.shape(),
                             binary_op, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_binary_v<T, U, V, BinaryOp>) {
                cuda::math::ewise(lhs.share(), lhs_stride, rhs.share(), rhs_stride,
                                  output.share(), output.stride(), output.shape(),
                                  binary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

// -- Trinary operators -- //
namespace noa::math {
    template<typename T, typename U, typename V, typename W, typename TrinaryOp, typename>
    void ewise(const Array<T>& lhs, U mhs, V rhs, const Array<W>& output, TrinaryOp trinary_op) {
        size4_t lhs_stride = lhs.stride();
        if (!indexing::broadcast(lhs.shape(), lhs_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }

        const Device device{output.device()};
        NOA_CHECK(device == lhs.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  lhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs.share(), lhs_stride, mhs, rhs,
                             output.share(), output.stride(), output.shape(),
                             trinary_op, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_trinary_v<T, U, V, W, TrinaryOp>) {
                cuda::math::ewise(lhs.share(), lhs_stride, mhs, rhs,
                                  output.share(), output.stride(), output.shape(),
                                  trinary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename U, typename V, typename W, typename TrinaryOp>
    void ewise(const Array<T>& lhs, const Array<U>& mhs, const Array<V>& rhs,
               const Array<W>& output, TrinaryOp trinary_op) {
        size4_t lhs_stride = lhs.stride();
        if (!indexing::broadcast(lhs.shape(), lhs_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }
        size4_t mhs_stride = mhs.stride();
        if (!indexing::broadcast(mhs.shape(), mhs_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      mhs.shape(), output.shape());
        }
        size4_t rhs_stride = rhs.stride();
        if (!indexing::broadcast(rhs.shape(), rhs_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device{output.device()};
        NOA_CHECK(device == lhs.device() && device == mhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, mhs:{}, rhs:{} and output:{}",
                  lhs.device(), mhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs.share(), lhs_stride, mhs.share(), mhs_stride, rhs.share(), rhs_stride,
                             output.share(), output.stride(), output.shape(),
                             trinary_op, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_trinary_v<T, U, V, W, TrinaryOp>) {
                cuda::math::ewise(lhs.share(), lhs_stride, mhs.share(), mhs_stride, rhs.share(), rhs_stride,
                                  output.share(), output.stride(), output.shape(),
                                  trinary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
