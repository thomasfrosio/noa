#pragma once

#include "noa/cpu/math/Ewise.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Ewise.h"
#endif

#include "noa/unified/Array.h"

// -- Unary operators -- //
namespace noa::math {
    /// Element-wise transformation using an unary operator()(\p T) -> \p U
    /// \param[in] input    Input to transform.
    /// \param[out] output  Transformed array.
    /// \note On the GPU, supported operators and types are limited to the following list:
    ///     Integers:
    ///       - (copy|square|abs|negate|one_minus|nonzero|logical_not)_t(A) -> A
    ///       - (copy|square|nonzero|logical_not)_t(B) -> B
    ///       - (nonzero|logical_not)_t(A|B) -> bool
    ///     Floating-points:
    ///       - (copy|square|abs|negate|one_minus|inverse|sqrt|rsqrt|exp|log|cos|sin)_t(C) -> C
    ///       - (round|rint|ceil|floor|trunc)_t(C) -> C
    ///     Complex:
    ///       - (square|one_minus|inverse|normalize|conj)_t(D) -> D
    ///       - (abs|abs_squared|real|imag)_t(D) -> C
    ///     Where:
    ///         A = int16_t, int32_t, or int64_t
    ///         B = uint16_t, uint32_t, or uint64_t
    ///         C = half_t, float, or double
    ///         D = chalf_t, cfloat_t, or cdouble_t
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
    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param binary_op    Binary operator. The output is explicitly casted to \p V.
    /// \note On the GPU, supported operators and types are limited to the following list:
    ///     Integers:
    ///       - (plus|minus|multiply|divide|divide_safe|dist2|min|max)_t(A,A) -> A
    ///       - (equal|not_equal|less|less_equal|greater|greater_equal|modulo_t|logical_and|logical_or)_t(A,A) -> A
    ///       - (equal|not_equal|less|less_equal|greater|greater_equal|logical_and|logical_not)_t(A,A) -> bool
    ///     Floating-points:
    ///       - (plus|minus|multiply|divide|divide_safe|dist2|min|max)_t(B,B) -> B
    ///       - (equal|not_equal|less|less_equal|greater|greater_equal|pow)_t(B,B) -> B
    ///       - (equal|not_equal|less|less_equal|greater|greater_equal)_t(B,B) -> bool
    ///     Complex:
    ///       - (plus|minus|multiply|divide|divide_safe|dist2|multiply_conj)_t(C,C) -> C
    ///       - (plus|minus|multiply|divide|divide_safe|dist2)_t(C,B) -> C
    ///       - (plus|minus|multiply|divide|divide_safe|dist2)_t(B,C) -> C
    ///     Where:
    ///         A = int16_t, int32_t, int64_t, uint16_t, uint32_t, or uint64_t
    ///         B = half_t, float, or double
    ///         C = chalf_t, cfloat_t, or cdouble_t
    template<typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U>>>
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

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param binary_op    Binary operator. The output is explicitly casted to \p V.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<T>>>
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

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param binary_op    Binary operator. The output is explicitly casted to \p V.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
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
    /// Element-wise transformation using a trinary operator()(\p T, \p U, \p U) -> \p V
    /// \param[in] lhs      On the \b device. Left-hand side argument.
    /// \param mhs          Middle-hand side argument.
    /// \param rhs          Right-hand side argument.
    /// \param[out] output  On the \b device. Transformed array.
    /// \param trinary_op   Trinary operation function object that will be applied.
    /// \note Supported operators and types are limited to the following list:
    ///     Integers:
    ///       - (within|within_equal|clamp)_t(A,A,A) -> A or bool
    ///       - fma_t(A,A,A) -> A
    ///     Floating-points:
    ///       - (within|within_equal|clamp)_t(B,B,B) -> B or bool
    ///       - fma_t(B,B,B) -> B
    ///     Complex:
    ///       - fma_t(C,C,C) -> C
    ///       - fma_t(C,B,B) -> C
    ///     Where:
    ///         A = (u)int16_t, (u)int32_t, (u)int64_t
    ///         B = half_t, float, double
    ///         C = chalf_t, cfloat_t, cdouble_t
    template<typename T, typename U, typename V, typename W, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U> && noa::traits::is_data_v<V>>>
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

    /// Element-wise transformation using a trinary operator()(\p T, \p U, \p V) -> \p W
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] mhs      Middle-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param trinary_op   Trinary operation function object that will be applied.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
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
