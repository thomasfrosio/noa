#pragma once

#ifndef NOA_UNIFIED_EWISE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/math/Ewise.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Ewise.h"
#endif

// -- Unary operators -- //
namespace noa::math {
    template<typename In, typename Out, typename UnaryOp>
    void ewise(const Array<In>& input, const Array<Out>& output, UnaryOp&& unary_op) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");

        dim4_t input_strides = input.strides();
        if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(input.share(), input_strides,
                             output.share(), output.strides(), output.shape(),
                             std::forward<UnaryOp>(unary_op), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_unary_v<In, Out, UnaryOp>) {
                cuda::math::ewise(input.share(), input_strides,
                                  output.share(), output.strides(), output.shape(),
                                  unary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

// -- Binary operators -- //
namespace noa::math {
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp, typename>
    void ewise(const Array<Lhs>& lhs, Rhs rhs, const Array<Out>& output, BinaryOp&& binary_op) {
        NOA_CHECK(!lhs.empty() && !output.empty(), "Empty array detected");

        dim4_t lhs_strides = lhs.strides();
        if (!indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  lhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs.share(), lhs_strides, rhs,
                             output.share(), output.strides(), output.shape(),
                             std::forward<BinaryOp>(binary_op), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_binary_v<Lhs, Rhs, Out, BinaryOp>) {
                cuda::math::ewise(lhs.share(), lhs_strides, rhs,
                                  output.share(), output.strides(), output.shape(),
                                  binary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp, typename>
    void ewise(Lhs lhs, const Array<Rhs>& rhs, const Array<Out>& output, BinaryOp&& binary_op) {
        NOA_CHECK(!rhs.empty() && !output.empty(), "Empty array detected");

        dim4_t rhs_strides = rhs.strides();
        if (!indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == rhs.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs, rhs.share(), rhs_strides,
                             output.share(), output.strides(), output.shape(),
                             std::forward<BinaryOp>(binary_op), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_binary_v<Lhs, Rhs, Out, BinaryOp>) {
                cuda::math::ewise(lhs, rhs.share(), rhs_strides,
                                  output.share(), output.strides(), output.shape(),
                                  binary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp>
    void ewise(const Array<Lhs>& lhs, const Array<Rhs>& rhs, const Array<Out>& output, BinaryOp&& binary_op) {
        NOA_CHECK(!lhs.empty() && !rhs.empty() && !output.empty(), "Empty array detected");

        dim4_t lhs_strides = lhs.strides();
        if (!indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }
        dim4_t rhs_strides = rhs.strides();
        if (!indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
                  lhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs.share(), lhs_strides, rhs.share(), rhs_strides,
                             output.share(), output.strides(), output.shape(),
                             std::forward<BinaryOp>(binary_op), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_binary_v<Lhs, Rhs, Out, BinaryOp>) {
                cuda::math::ewise(lhs.share(), lhs_strides, rhs.share(), rhs_strides,
                                  output.share(), output.strides(), output.shape(),
                                  binary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

// -- Trinary operators -- //
namespace noa::math {
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise(const Array<Lhs>& lhs, Mhs mhs, Rhs rhs, const Array<Out>& output, TrinaryOp&& trinary_op) {
        NOA_CHECK(!lhs.empty() && !output.empty(), "Empty array detected");

        dim4_t lhs_strides = lhs.strides();
        if (!indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  lhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs.share(), lhs_strides, mhs, rhs,
                             output.share(), output.strides(), output.shape(),
                             std::forward<TrinaryOp>(trinary_op), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>) {
                cuda::math::ewise(lhs.share(), lhs_strides, mhs, rhs,
                                  output.share(), output.strides(), output.shape(),
                                  trinary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise(const Array<Lhs>& lhs, const Array<Mhs>& mhs, Rhs rhs,
               const Array<Out>& output, TrinaryOp&& trinary_op) {
        NOA_CHECK(!lhs.empty() && !mhs.empty() && !output.empty(), "Empty array detected");

        dim4_t lhs_strides = lhs.strides();
        if (!indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }
        dim4_t mhs_strides = mhs.strides();
        if (!indexing::broadcast(mhs.shape(), mhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      mhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == mhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, mhs:{} and output:{}",
                  lhs.device(), mhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs.share(), lhs_strides, mhs.share(), mhs_strides, rhs,
                             output.share(), output.strides(), output.shape(),
                             std::forward<TrinaryOp>(trinary_op), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>) {
                cuda::math::ewise(lhs.share(), lhs_strides, mhs.share(), mhs_strides, rhs,
                                  output.share(), output.strides(), output.shape(),
                                  trinary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise(const Array<Lhs>& lhs, Mhs mhs, const Array<Rhs>& rhs,
               const Array<Out>& output, TrinaryOp&& trinary_op) {
        NOA_CHECK(!lhs.empty() && !rhs.empty() && !output.empty(), "Empty array detected");

        dim4_t lhs_strides = lhs.strides();
        if (!indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }
        dim4_t rhs_strides = rhs.strides();
        if (!indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
                  lhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs.share(), lhs_strides, mhs, rhs.share(), rhs_strides,
                             output.share(), output.strides(), output.shape(),
                             std::forward<TrinaryOp>(trinary_op), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>) {
                cuda::math::ewise(lhs.share(), lhs_strides, mhs, rhs.share(), rhs_strides,
                                  output.share(), output.strides(), output.shape(),
                                  trinary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise(const Array<Lhs>& lhs, const Array<Mhs>& mhs, const Array<Rhs>& rhs,
               const Array<Out>& output, TrinaryOp&& trinary_op) {
        NOA_CHECK(!lhs.empty() && !mhs.empty() && !rhs.empty() && !output.empty(), "Empty array detected");

        dim4_t lhs_strides = lhs.strides();
        if (!indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }
        dim4_t mhs_strides = mhs.strides();
        if (!indexing::broadcast(mhs.shape(), mhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      mhs.shape(), output.shape());
        }
        dim4_t rhs_strides = rhs.strides();
        if (!indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == mhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, mhs:{}, rhs:{} and output:{}",
                  lhs.device(), mhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs.share(), lhs_strides, mhs.share(), mhs_strides, rhs.share(), rhs_strides,
                             output.share(), output.strides(), output.shape(),
                             std::forward<TrinaryOp>(trinary_op), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>) {
                cuda::math::ewise(lhs.share(), lhs_strides, mhs.share(), mhs_strides, rhs.share(), rhs_strides,
                                  output.share(), output.strides(), output.shape(),
                                  trinary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
