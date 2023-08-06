#pragma once

#include "noa/cpu/Ewise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Ewise.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa {
    /// Element-wise transformation using a unary \c operator()(input)->output.
    /// \param[in] input    Input array to transform.
    /// \param[out] output  Transformed array.
    /// \param unary_op     Unary operator. The output is explicitly cast to the \p output value type.
    /// \note On the GPU, supported operators and types are limited to the following list.
    ///       Other type combinations can be added (see noa/gpu/cuda/Ewise.hpp).
    ///     \b Integers:\n
    ///       - \c (copy|square|abs|abs_squared|negate|one_minus|nonzero|logical_not)_t(A)->A\n
    ///       - \c (copy|square|nonzero|logical_not)_t(B)->B\n
    ///       - \c (nonzero|logical_not)_t(A|B)->bool\n
    ///     \b Floating-points:\n
    ///       - \c (copy|square|abs|abs_squared|negate|one_minus|inverse)_t(C)->C\n
    ///       - \c (sqrt|rsqrt|exp|log|cos|sin|one_log|abs_one_log)_t(C)->C\n
    ///       - \c (round|rint|ceil|floor|trunc)_t(C)->C\n
    ///     \b Complex:\n
    ///       - \c (square|one_minus|inverse|normalize|conj)_t(D)->D\n
    ///       - \c (abs|abs_squared|abs_one_log|real|imag)_t(D)->C\n
    ///     \b Where:\n
    ///         \c A = \c i8,i16,i32,i64\n
    ///         \c B = \c u8,u16,u32,u64\n
    ///         \c C = \c f16,f32,f64\n
    ///         \c D = \c c16,c32,c64\n
    template<typename Input, typename Output, typename UnaryOp,
             typename = std::enable_if_t<noa::traits::are_varray_v<Input, Output>>>
    void ewise_unary(const Input& input, const Output& output, UnaryOp&& unary_op) {
        using input_value_t = noa::traits::value_type_t<Input>;
        using output_value_t = noa::traits::value_type_t<Output>;
        using mutable_input_value_t = std::remove_const_t<input_value_t>;
        static_assert(noa::traits::is_detected_convertible_v<
                output_value_t, noa::traits::has_unary_operator, UnaryOp, input_value_t>);

        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");

        auto input_strides = input.strides();
        if (!noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, op = std::forward<UnaryOp>(unary_op)]() {
                cpu::ewise_unary(input.get(), input_strides,
                                 output.get(), output.strides(), output.shape(),
                                 op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::details::is_valid_ewise_unary_v<
                    mutable_input_value_t, output_value_t,
                    noa::traits::remove_ref_cv_t<UnaryOp>>) {
                auto& cuda_stream = stream.cuda();
                cuda::ewise_unary(input.get(), input_strides,
                                  output.get(), output.strides(), output.shape(),
                                  unary_op, cuda_stream);
                cuda_stream.enqueue_attach(input.share(), output.share());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::ewise_unary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Element-wise transformation using a unary \c operator()(input)->output.
    /// The output is allocated and returned. By default, the output value type is deduced from the operator.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Output = void, typename Input, typename UnaryOp,
             typename = std::enable_if_t<noa::traits::is_varray_v<Input> &&
                                         (std::is_void_v<Output> || noa::traits::is_numeric_v<Output>)>>
    [[nodiscard]] auto ewise_unary(const Input& input, UnaryOp&& unary_op) {
        using input_value_t = noa::traits::value_type_t<Input>;
        using return_value_t = std::conditional_t<
                std::is_void_v<Output>, std::invoke_result_t<UnaryOp, input_value_t>, Output>;
        Array<return_value_t> output(input.shape(), input.options());
        ewise_unary(input, output, std::forward<UnaryOp>(unary_op));
        return output;
    }
}

namespace noa {
    /// Element-wise transformation using a binary \c operator()(lhs,rhs)->output.
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param binary_op    Binary operator. The output is explicitly cast to the \p output value type.
    /// \note On the GPU, supported operators and types are limited to the following list.
    ///       Other type combinations can be added (see noa/gpu/cuda/Ewise.hpp).
    ///     \b Integers:\n
    ///       - \c (plus|minus|multiply|divide|divide_safe|dist2|min|max)_t(A,A)->A\n
    ///       - \c (equal|not_equal|less|less_equal|greater|greater_equal|modulo|logical_and|logical_or)_t(A,A)->A\n
    ///       - \c (equal|not_equal|less|less_equal|greater|greater_equal|logical_and|logical_not)_t(A,A)->bool\n
    ///     \b Floating-points:\n
    ///       - \c (plus|minus|multiply|divide|divide_safe|dist2|min|max)_t(B,B)->B\n
    ///       - \c (equal|not_equal|less|less_equal|greater|greater_equal|pow)_t(B,B)->B\n
    ///       - \c (equal|not_equal|less|less_equal|greater|greater_equal)_t(B,B)->bool\n
    ///     \b Complex:\n
    ///       - \c (plus|minus|multiply|divide|divide_safe|dist2|multiply_conj)_t(C,C)->C\n
    ///       - \c (plus|minus|multiply|divide|divide_safe|dist2)_t(C,B)->C\n
    ///       - \c (plus|minus|multiply|divide|divide_safe|dist2)_t(B,C)->C\n
    ///     \b Where:\n
    ///         \c A = \c i8,i16,i32,i64,u8,u16,u32,u64\n
    ///         \c B = \c f16,f32,f64\n
    ///         \c C = \c c16,c32,c64\n
    template<typename Lhs, typename Rhs, typename Output, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::are_varray_v<Lhs, Rhs, Output>>>
    void ewise_binary(const Lhs& lhs, const Rhs& rhs, const Output& output, BinaryOp&& binary_op) {
        using lhs_value_t = noa::traits::value_type_t<Lhs>;
        using rhs_value_t = noa::traits::value_type_t<Rhs>;
        using output_value_t = noa::traits::value_type_t<Output>;
        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
        static_assert(noa::traits::is_detected_convertible_v<
                output_value_t, noa::traits::has_binary_operator, BinaryOp, lhs_value_t, rhs_value_t>);

        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty() && !output.is_empty(), "Empty array detected");

        auto lhs_strides = lhs.strides();
        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }
        auto rhs_strides = rhs.strides();
        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
                  lhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, op = std::forward<BinaryOp>(binary_op)]() {
                cpu::ewise_binary(lhs.get(), lhs_strides,
                                  rhs.get(), rhs_strides,
                                  output.get(), output.strides(), output.shape(),
                                  op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::details::is_valid_ewise_binary_v<
                    mutable_lhs_value_t, mutable_rhs_value_t, output_value_t,
                    noa::traits::remove_ref_cv_t<BinaryOp>>) {
                auto& cuda_stream = stream.cuda();
                cuda::ewise_binary(lhs.get(), lhs_strides,
                                   rhs.get(), rhs_strides,
                                   output.get(), output.strides(), output.shape(),
                                   binary_op, cuda_stream);
                cuda_stream.enqueue_attach(lhs.share(), rhs.share(), output.share());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::ewise_binary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Element-wise transformation using a binary \c operator()(lhs,rhs)->output.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Rhs, typename Output, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::are_varray_v<Lhs, Output> &&
                                         noa::traits::is_numeric_v<Rhs>>>
    void ewise_binary(const Lhs& lhs, Rhs rhs, const Output& output, BinaryOp&& binary_op) {
        using lhs_value_t = noa::traits::value_type_t<Lhs>;
        using rhs_value_t = noa::traits::value_type_t<Rhs>;
        using output_value_t = noa::traits::value_type_t<Output>;
        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
        static_assert(noa::traits::is_detected_convertible_v<
                output_value_t, noa::traits::has_binary_operator, BinaryOp, lhs_value_t, rhs_value_t>);

        NOA_CHECK(!lhs.is_empty() && !output.is_empty(), "Empty array detected");

        auto lhs_strides = lhs.strides();
        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, and output:{}",
                  lhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, op = std::forward<BinaryOp>(binary_op)]() {
                cpu::ewise_binary(lhs.get(), lhs_strides,
                                  rhs,
                                  output.get(), output.strides(), output.shape(),
                                  op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::details::is_valid_ewise_binary_v<
                    mutable_lhs_value_t, mutable_rhs_value_t, output_value_t,
                    noa::traits::remove_ref_cv_t<BinaryOp>>) {
                auto& cuda_stream = stream.cuda();
                cuda::ewise_binary(lhs.get(), lhs_strides,
                                   rhs,
                                   output.get(), output.strides(), output.shape(),
                                   binary_op, cuda_stream);
                cuda_stream.enqueue_attach(lhs.share(), output.share());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::ewise_binary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Element-wise transformation using a binary \c operator()(lhs,rhs)->output.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Rhs, typename Output, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::are_varray_v<Rhs, Output> &&
                                         noa::traits::is_numeric_v<Lhs>>>
    void ewise_binary(Lhs lhs, const Rhs& rhs, const Output& output, BinaryOp&& binary_op) {
        using lhs_value_t = noa::traits::value_type_t<Lhs>;
        using rhs_value_t = noa::traits::value_type_t<Rhs>;
        using output_value_t = noa::traits::value_type_t<Output>;
        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
        static_assert(noa::traits::is_detected_convertible_v<
                output_value_t, noa::traits::has_binary_operator, BinaryOp, lhs_value_t, rhs_value_t>);

        NOA_CHECK(!rhs.is_empty() && !output.is_empty(), "Empty array detected");

        auto rhs_strides = rhs.strides();
        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == rhs.device(),
                  "The input and output arrays must be on the same device, but got rhs:{} and output:{}",
                  rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, op = std::forward<BinaryOp>(binary_op)]() {
                cpu::ewise_binary(lhs,
                                  rhs.get(), rhs_strides,
                                  output.get(), output.strides(), output.shape(),
                                  op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::details::is_valid_ewise_binary_v<
                    mutable_lhs_value_t, mutable_rhs_value_t, output_value_t,
                    noa::traits::remove_ref_cv_t<BinaryOp>>) {
                auto& cuda_stream = stream.cuda();
                cuda::ewise_binary(lhs,
                                   rhs.get(), rhs_strides,
                                   output.get(), output.strides(), output.shape(),
                                   binary_op, cuda_stream);
                cuda_stream.enqueue_attach(rhs.share(), output.share());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::ewise_binary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
    /// The output is allocated and returned. By default, the output value type is deduced from the operator.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Output = void, typename Lhs, typename Rhs, typename BinaryOp,
             typename = std::enable_if_t<
                     (std::is_void_v<Output> || noa::traits::is_numeric_v<Output>) &&
                     (noa::traits::is_varray_v<Lhs> || noa::traits::is_numeric_v<Lhs>) &&
                     (noa::traits::is_varray_v<Rhs> || noa::traits::is_numeric_v<Rhs>)>>
    [[nodiscard]] auto ewise_binary(const Lhs& lhs, const Rhs& rhs, BinaryOp&& binary_op) {
        using lhs_value_t = noa::traits::value_type_t<Lhs>;
        using rhs_value_t = noa::traits::value_type_t<Rhs>;
        using return_value_t = std::conditional_t<
                std::is_void_v<Output>, std::invoke_result_t<BinaryOp, lhs_value_t, rhs_value_t>, Output>;
        Shape4<i64> shape;
        ArrayOption options;
        if constexpr (noa::traits::is_varray_v<Lhs>) {
            shape = lhs.shape();
            options = lhs.options();
        } else if constexpr (noa::traits::is_varray_v<Rhs>) {
            shape = rhs.shape();
            options = rhs.options();
        } else {
            static_assert(noa::traits::always_false_v<BinaryOp>);
        }
        Array<return_value_t> output(shape, options);
        ewise_binary(lhs, rhs, output, binary_op);
        return output;
    }
}

namespace noa {
    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] mhs      Middle-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param trinary_op   Trinary operator. The output is explicitly cast to the \p output value type.
    /// \note On the GPU, supported operators and types are limited to the following list.
    ///       Other type combinations can be added (see noa/gpu/cuda/Ewise.hpp).
    ///     \b Integers and Floating-points:\n
    ///       - \c (within|within_equal)_t(A,A,A)->A|bool\n
    ///       - \c clamp_t(A,A,A)->A\n
    ///       - \c (plus|plus_minus|plus_multiply|plus_divide)_t(A,A,A)->A\n
    ///       - \c (minus|minus_plus|minus_multiply|minus_divide)_t(A,A,A)->A\n
    ///       - \c (multiply|multiply_plus|multiply_minus|multiply_divide)_t(A,A,A)->A\n
    ///       - \c (divide|divide_plus|divide_minus|divide_multiply|divide_epsilon)_t(A,A,A)->A\n
    ///     \b Complex:\n
    ///       - \c (plus|plus_minus|plus_multiply|plus_divide)_t(B,B,B)->C\n
    ///       - \c (minus|minus_plus|minus_multiply|minus_divide)_t(B,B,B)->C\n
    ///       - \c (multiply|multiply_plus|multiply_minus|multiply_divide)_t(B,B,B)->C\n
    ///       - \c (divide|divide_plus|divide_minus|divide_multiply|divide_epsilon)_t(B,B,B)->C\n
    ///     \b Where:\n
    ///         \c A = \c i8,i16,i32,i64,u8,u16,u32,u64,f16,f32,f64\n
    ///         \c B = \c f16,f32,f64,c16,c32,c64\n
    ///         \c C = \c c16,c32,c64\n
    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::are_varray_v<Lhs, Mhs, Rhs, Output>>>
    void ewise_trinary(const Lhs& lhs, const Mhs& mhs, const Rhs& rhs, const Output& output, TrinaryOp&& trinary_op) {
        using lhs_value_t = noa::traits::value_type_t<Lhs>;
        using mhs_value_t = noa::traits::value_type_t<Mhs>;
        using rhs_value_t = noa::traits::value_type_t<Rhs>;
        using output_value_t = noa::traits::value_type_t<Output>;
        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
        static_assert(noa::traits::is_detected_convertible_v<
                output_value_t, noa::traits::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);

        NOA_CHECK(!lhs.is_empty() && !mhs.is_empty() && !rhs.is_empty() && !output.is_empty(), "Empty array detected");

        auto lhs_strides = lhs.strides();
        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }
        auto mhs_strides = mhs.strides();
        if (!noa::indexing::broadcast(mhs.shape(), mhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      mhs.shape(), output.shape());
        }
        auto rhs_strides = rhs.strides();
        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == mhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, mhs:{}, rhs:{} and output:{}",
                  lhs.device(), mhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
                cpu::ewise_trinary(lhs.get(), lhs_strides,
                                   mhs.get(), mhs_strides,
                                   rhs.get(), rhs_strides,
                                   output.get(), output.strides(), output.shape(),
                                   op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::details::is_valid_ewise_trinary_v<
                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
                    noa::traits::remove_ref_cv_t<TrinaryOp>>) {
                auto& cuda_stream = stream.cuda();
                cuda::ewise_trinary(lhs.get(), lhs_strides,
                                    mhs.get(), mhs_strides,
                                    rhs.get(), rhs_strides,
                                    output.get(), output.strides(), output.shape(),
                                    trinary_op, cuda_stream);
                cuda_stream.enqueue_attach(lhs.share(), mhs.share(), rhs.share(), output.share());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::are_varray_v<Lhs, Mhs, Output> &&
                                         noa::traits::is_numeric_v<Rhs>>>
    void ewise_trinary(const Lhs& lhs, const Mhs& mhs, Rhs rhs, const Output& output, TrinaryOp&& trinary_op) {
        using lhs_value_t = noa::traits::value_type_t<Lhs>;
        using mhs_value_t = noa::traits::value_type_t<Mhs>;
        using rhs_value_t = noa::traits::value_type_t<Rhs>;
        using output_value_t = noa::traits::value_type_t<Output>;
        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
        static_assert(noa::traits::is_detected_convertible_v<
                output_value_t, noa::traits::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);

        NOA_CHECK(!lhs.is_empty() && !mhs.is_empty() && !output.is_empty(), "Empty array detected");

        auto lhs_strides = lhs.strides();
        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }
        auto mhs_strides = mhs.strides();
        if (!noa::indexing::broadcast(mhs.shape(), mhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      mhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == mhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, mhs:{} and output:{}",
                  lhs.device(), mhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
                cpu::ewise_trinary(lhs.get(), lhs_strides,
                                   mhs.get(), mhs_strides,
                                   rhs,
                                   output.get(), output.strides(), output.shape(),
                                   op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::details::is_valid_ewise_trinary_v<
                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
                    noa::traits::remove_ref_cv_t<TrinaryOp>>) {
                auto& cuda_stream = stream.cuda();
                cuda::ewise_trinary(lhs.get(), lhs_strides,
                                    mhs.get(), mhs_strides,
                                    rhs,
                                    output.get(), output.strides(), output.shape(),
                                    trinary_op, cuda_stream);
                cuda_stream.enqueue_attach(lhs.share(), mhs.share(), output.share());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::are_varray_v<Lhs, Rhs, Output> &&
                                         noa::traits::is_numeric_v<Mhs>>>
    void ewise_trinary(const Lhs& lhs, Mhs mhs, const Rhs& rhs, const Output& output, TrinaryOp&& trinary_op) {
        using lhs_value_t = noa::traits::value_type_t<Lhs>;
        using mhs_value_t = noa::traits::value_type_t<Mhs>;
        using rhs_value_t = noa::traits::value_type_t<Rhs>;
        using output_value_t = noa::traits::value_type_t<Output>;
        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
        static_assert(noa::traits::is_detected_convertible_v<
                output_value_t, noa::traits::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);

        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty() && !output.is_empty(), "Empty array detected");

        auto lhs_strides = lhs.strides();
        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }
        auto rhs_strides = rhs.strides();
        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
                  lhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
                cpu::ewise_trinary(lhs.get(), lhs_strides,
                                   mhs,
                                   rhs.get(), rhs_strides,
                                   output.get(), output.strides(), output.shape(),
                                   op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::details::is_valid_ewise_trinary_v<
                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
                    noa::traits::remove_ref_cv_t<TrinaryOp>>) {
                auto& cuda_stream = stream.cuda();
                cuda::ewise_trinary(lhs.get(), lhs_strides,
                                    mhs,
                                    rhs.get(), rhs_strides,
                                    output.get(), output.strides(), output.shape(),
                                    trinary_op, cuda_stream);
                cuda_stream.enqueue_attach(lhs.share(), rhs.share(), output.share());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::are_varray_v<Mhs, Rhs, Output> &&
                                         noa::traits::is_numeric_v<Lhs>>>
    void ewise_trinary(Lhs lhs, const Mhs& mhs, const Rhs& rhs, const Output& output, TrinaryOp&& trinary_op) {
        using lhs_value_t = noa::traits::value_type_t<Lhs>;
        using mhs_value_t = noa::traits::value_type_t<Mhs>;
        using rhs_value_t = noa::traits::value_type_t<Rhs>;
        using output_value_t = noa::traits::value_type_t<Output>;
        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
        static_assert(noa::traits::is_detected_convertible_v<
                output_value_t, noa::traits::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);

        NOA_CHECK(!mhs.is_empty() && !rhs.is_empty() && !output.is_empty(), "Empty array detected");

        auto mhs_strides = mhs.strides();
        if (!noa::indexing::broadcast(mhs.shape(), mhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      mhs.shape(), output.shape());
        }
        auto rhs_strides = rhs.strides();
        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == mhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got mhs:{}, rhs:{} and output:{}",
                  mhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
                cpu::ewise_trinary(lhs,
                                   mhs.get(), mhs_strides,
                                   rhs.get(), rhs_strides,
                                   output.get(), output.strides(), output.shape(),
                                   op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::details::is_valid_ewise_trinary_v<
                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
                    noa::traits::remove_ref_cv_t<TrinaryOp>>) {
                auto& cuda_stream = stream.cuda();
                cuda::ewise_trinary(lhs,
                                    mhs.get(), mhs_strides,
                                    rhs.get(), rhs_strides,
                                    output.get(), output.strides(), output.shape(),
                                    trinary_op, cuda_stream);
                cuda_stream.enqueue_attach(mhs.share(), rhs.share(), output.share());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::are_varray_v<Lhs, Output> &&
                                         noa::traits::are_numeric_v<Mhs, Rhs>>>
    void ewise_trinary(const Lhs& lhs, Mhs mhs, Rhs rhs, const Output& output, TrinaryOp&& trinary_op) {
        using lhs_value_t = noa::traits::value_type_t<Lhs>;
        using mhs_value_t = noa::traits::value_type_t<Mhs>;
        using rhs_value_t = noa::traits::value_type_t<Rhs>;
        using output_value_t = noa::traits::value_type_t<Output>;
        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
        static_assert(noa::traits::is_detected_convertible_v<
                output_value_t, noa::traits::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);

        NOA_CHECK(!lhs.is_empty() && !output.is_empty(), "Empty array detected");

        auto lhs_strides = lhs.strides();
        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{} and output:{}",
                  lhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
                cpu::ewise_trinary(lhs.get(), lhs_strides,
                                   mhs,
                                   rhs,
                                   output.get(), output.strides(), output.shape(),
                                   op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::details::is_valid_ewise_trinary_v<
                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
                    noa::traits::remove_ref_cv_t<TrinaryOp>>) {
                auto& cuda_stream = stream.cuda();
                cuda::ewise_trinary(lhs.get(), lhs_strides,
                                    mhs,
                                    rhs,
                                    output.get(), output.strides(), output.shape(),
                                    trinary_op, cuda_stream);
                cuda_stream.enqueue_attach(lhs.share(), output.share());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::are_varray_v<Mhs, Output> &&
                                         noa::traits::are_numeric_v<Lhs, Rhs>>>
    void ewise_trinary(Lhs lhs, const Mhs& mhs, Rhs rhs, const Output& output, TrinaryOp&& trinary_op) {
        using lhs_value_t = noa::traits::value_type_t<Lhs>;
        using mhs_value_t = noa::traits::value_type_t<Mhs>;
        using rhs_value_t = noa::traits::value_type_t<Rhs>;
        using output_value_t = noa::traits::value_type_t<Output>;
        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
        static_assert(noa::traits::is_detected_convertible_v<
                output_value_t, noa::traits::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);

        NOA_CHECK(!mhs.is_empty() && !output.is_empty(), "Empty array detected");

        auto mhs_strides = mhs.strides();
        if (!noa::indexing::broadcast(mhs.shape(), mhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      mhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == mhs.device(),
                  "The input and output arrays must be on the same device, but got mhs:{} and output:{}",
                  mhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
                cpu::ewise_trinary(lhs,
                                   mhs.get(), mhs_strides,
                                   rhs,
                                   output.get(), output.strides(), output.shape(),
                                   op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::details::is_valid_ewise_trinary_v<
                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
                    noa::traits::remove_ref_cv_t<TrinaryOp>>) {
                auto& cuda_stream = stream.cuda();
                cuda::ewise_trinary(lhs,
                                    mhs.get(), mhs_strides,
                                    rhs,
                                    output.get(), output.strides(), output.shape(),
                                    trinary_op, cuda_stream);
                cuda_stream.enqueue_attach(mhs.share(), output.share());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::are_varray_v<Rhs, Output> &&
                                         noa::traits::are_numeric_v<Lhs, Mhs>>>
    void ewise_trinary(Lhs lhs, Mhs mhs, const Rhs& rhs, const Output& output, TrinaryOp&& trinary_op) {
        using lhs_value_t = noa::traits::value_type_t<Lhs>;
        using mhs_value_t = noa::traits::value_type_t<Mhs>;
        using rhs_value_t = noa::traits::value_type_t<Rhs>;
        using output_value_t = noa::traits::value_type_t<Output>;
        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
        static_assert(noa::traits::is_detected_convertible_v<
                output_value_t, noa::traits::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);

        NOA_CHECK(!rhs.is_empty() && !output.is_empty(), "Empty array detected");

        auto rhs_strides = rhs.strides();
        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == rhs.device(),
                  "The input and output arrays must be on the same device, but got rhs:{} and output:{}",
                  rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
                cpu::ewise_trinary(lhs,
                                   mhs,
                                   rhs.get(), rhs_strides,
                                   output.get(), output.strides(), output.shape(),
                                   op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::details::is_valid_ewise_trinary_v<
                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
                    noa::traits::remove_ref_cv_t<TrinaryOp>>) {
                auto& cuda_stream = stream.cuda();
                cuda::ewise_trinary(lhs,
                                    mhs,
                                    rhs.get(), rhs_strides,
                                    output.get(), output.strides(), output.shape(),
                                    trinary_op, cuda_stream);
                cuda_stream.enqueue_attach(rhs.share(), output.share());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
    /// The output is allocated and returned. By default, the output value type is deduced from the operator.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Output = void, typename Lhs, typename Mhs, typename Rhs, typename TrinaryOp,
             typename = std::enable_if_t<
                     (std::is_void_v<Output> || noa::traits::is_numeric_v<Output>) &&
                     (noa::traits::is_varray_v<Lhs> || noa::traits::is_numeric_v<Lhs>) &&
                     (noa::traits::is_varray_v<Mhs> || noa::traits::is_numeric_v<Mhs>) &&
                     (noa::traits::is_varray_v<Rhs> || noa::traits::is_numeric_v<Rhs>)>>
    [[nodiscard]] auto ewise_trinary(const Lhs& lhs, const Mhs& mhs, const Rhs& rhs, TrinaryOp&& trinary_op) {
        using lhs_value_t = noa::traits::value_type_t<Lhs>;
        using mhs_value_t = noa::traits::value_type_t<Mhs>;
        using rhs_value_t = noa::traits::value_type_t<Rhs>;
        using return_value_t = std::conditional_t<
                std::is_void_v<Output>, std::invoke_result_t<TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>, Output>;
        Shape4<i64> shape;
        ArrayOption options;
        if constexpr (noa::traits::is_varray_v<Lhs>) {
            shape = lhs.shape();
            options = lhs.options();
        } else if constexpr (noa::traits::is_varray_v<Mhs>) {
            shape = mhs.shape();
            options = mhs.options();
        } else if constexpr (noa::traits::is_varray_v<Rhs>) {
            shape = rhs.shape();
            options = rhs.options();
        } else {
            static_assert(noa::traits::always_false_v<TrinaryOp>);
        }
        Array<return_value_t> output(shape, options);
        ewise_trinary(lhs, mhs, rhs, output, trinary_op);
        return output;
    }
}
