#pragma once

#include "noa/unified/Traits.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Indexing.hpp"

#include "noa/cpu/Ewise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Ewise.hpp"
#endif

#include <optional>

namespace noa {
    template<typename T>
    class Array;

//    /// Element-wise transformation using a unary \c operator()(input)->output.
//    /// \param[in] input    Input array to transform.
//    /// \param[out] output  Transformed array.
//    /// \param unary_op     Unary operator. The output is explicitly cast to the \p output value type.
//    template<typename Input, typename Output, typename UnaryOp,
//             typename nt::enable_if_bool_t<nt::are_varray_v<Input, Output>> = true>
//    void ewise(const Input& input, const Output& output, UnaryOp&& unary_op) {
//        using input_value_t = nt::value_type_t<Input>;
//        using output_value_t = nt::value_type_t<Output>;
//        using mutable_input_value_t = std::remove_const_t<input_value_t>;
//        static_assert(nt::is_detected_convertible_v<
//                output_value_t, nt::has_unary_operator, UnaryOp, input_value_t>);
//
//        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
//
//        auto input_strides = input.strides();
//        if (!noa::broadcast(input.shape(), input_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      input.shape(), output.shape());
//        }
//
//        const Device device = output.device();
//        NOA_CHECK(device == input.device(),
//                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
//                  input.device(), device);
//
//        //
//        //        // Rearrange to rightmost order.
//        //        shape = noa::indexing::effective_shape(shape, output_strides);
//        //        const auto order = noa::indexing::order(output_strides, shape);
//        //        if (noa::any(order != Vec4<Index>{0, 1, 2, 3})) {
//        //            shape = noa::indexing::reorder(shape, order);
//        //            lhs_strides = noa::indexing::reorder(lhs_strides, order);
//        //            mhs_strides = noa::indexing::reorder(mhs_strides, order);
//        //            rhs_strides = noa::indexing::reorder(rhs_strides, order);
//        //            output_strides = noa::indexing::reorder(output_strides, order);
//        //        }
//        //
//        //        const Index elements = shape.elements();
//
//        Stream& stream = Stream::current(device);
//        if (device.is_cpu()) {
//            auto& cpu_stream = stream.cpu();
//            const auto threads = cpu_stream.thread_limit();
//            cpu_stream.enqueue([=, op = std::forward<UnaryOp>(unary_op)]() {
//                cpu::ewise_unary(input.get(), input_strides,
//                                 output.get(), output.strides(), output.shape(),
//                                 op, threads);
//            });
//        } else {
//            #ifdef NOA_ENABLE_CUDA
//            if constexpr (cuda::details::is_valid_ewise_unary_v<
//                    mutable_input_value_t, output_value_t,
//                    nt::remove_ref_cv_t<UnaryOp>>) {
//                auto& cuda_stream = stream.cuda();
//                cuda::ewise_unary(input.get(), input_strides,
//                                  output.get(), output.strides(), output.shape(),
//                                  unary_op, cuda_stream);
//                cuda_stream.enqueue_attach(input, output);
//            } else {
//                NOA_THROW("These types of operands are not supported by the CUDA backend. "
//                          "See documentation or noa::cuda::ewise_unary(...) for more details");
//            }
//            #else
//            NOA_THROW("No GPU backend detected");
//            #endif
//        }
//    }
//
//    /// Element-wise transformation using a unary \c operator()(input)->output.
//    /// The output is allocated and returned. By default, the output value type is deduced from the operator.
//    /// \note On the GPU, the same operators and types are supported as in the overload above.
//    template<typename Output = void, typename Input, typename UnaryOp,
//             typename = std::enable_if_t<nt::is_varray_v<Input> &&
//                                         (std::is_void_v<Output> || nt::is_numeric_v<Output>)>>
//    [[nodiscard]] auto ewise_unary(const Input& input, UnaryOp&& unary_op) {
//        using input_value_t = nt::value_type_t<Input>;
//        using return_value_t = std::conditional_t<
//                std::is_void_v<Output>, std::invoke_result_t<UnaryOp, input_value_t>, Output>;
//        Array<return_value_t> output(input.shape(), input.options());
//        ewise_unary(input, output, std::forward<UnaryOp>(unary_op));
//        return output;
//    }
}

namespace noa {
    /// Generic element-wise transformation.
    /// \tparam Inputs
    /// \tparam Outputs
    /// \tparam UnaryOp
    /// \param[in] inputs   Input varray(s) to transform.
    /// \param[out] outputs Transformed varray(s).
    /// \param ewise_op     Element-wise operator.
    ///                     If there are multiple output values, they should be returned in a Tuple.
    ///                     If there's a single output value, wrapping it in a Tuple is optional.
    ///                     The output values are explicitly casted to the \p outputs value types.
    ///
    /// \example Simple cosine map, in-place:
    /// \code a = cos(a) ->
    /// noa::ewise(a, a, noa::cos_t{});
    /// noa::ewise(a, Tuple<>{}, [](f64& a) { return noa::cos(a); }); // taking a lvalue reference
    /// noa::ewise(wrap(a), wrap(a), noa::cos_t{}); // wrap is short for noa::forward_as_tuple
    /// \endcode
    ///
    /// \example More complex example showing variadic ability:
    /// \code c = cos(a) * b + 1, d = c + a ->
    /// noa::ewise(wrap(a, b), wrap(c, d), [](f64 ai, f64 bi) {
    ///     auto ci = noa::cos(ai) * bi + 1;
    ///     auto di = ci + ai;
    ///     return Tuple{ci, di};
    /// });
    /// \endcode
    /// The constant value 1 can be passed as an input runtime value too,
    /// but note that the GPU backends will generate different code because
    /// the value 1 is lost and "one" can be any f64 value. The CPU backend
    /// may be able to optimize this.
    /// \code
    /// noa::ewise(wrap(a, b, 1.), wrap(c, d), [](f64 ai, f64 bi, f64 one) {
    ///     auto ci = noa::cos(ai) * bi + one;
    ///     auto di = ci + ai;
    ///     return Tuple{ci, di};
    /// });
    /// \endcode
    ///
    /// \note To be supported by the CUDA backend, the source of \p ewise_op needs to be added to the sources
    ///       available to the runtime compiler. This can be set up as part of the build, but sources can also
    ///       be added at runtime using the noa::Session::add_cuda_sources(...) functions.
    ///
    /// \note This function is internally relying on the iwise functions. One advantage of this function
    ///       is that it can analyse the inputs and outputs to deduce the most efficient way to traverse
    ///       these arrays, by reordering the dimensions, collapsing contiguous dimensions together (up to 1d),
    ///       and can trigger the vectorization for the 1d case by checking for data contiguity and aliasing.
    template<typename... Inputs, typename... Outputs, typename UnaryOp,
             typename nt::enable_if_bool_t<nt::are_varray_v<Outputs...>> = true> // FIXME at least one varray between outputs and inputs
    void ewise(
            const Tuple<const Inputs&...>& inputs,
            const Tuple<const Outputs&...>& outputs,
            UnaryOp&& ewise_op
    ) {
        // Check operator is valid
        // Check output shapes are the same and order, do we allow stride=0 on the outputs?

        constexpr size_t N_INPUTS = sizeof...(Inputs);
        constexpr size_t N_OUTPUTS = sizeof...(Outputs);
        if constexpr (N_INPUTS == 0 && N_OUTPUTS == 0)
            return;

        Tuple input_accessors = make_tuple_of_accessors(inputs);
        Tuple output_accessors = make_tuple_of_accessors(outputs);

        Shape4<i64> shape;
        Device device;
        if constexpr (N_OUTPUTS >= 1) {
            constexpr bool all_varrays = inputs.all([&](const auto& v) {
                if constexpr (nt::is_varray_v<decltype(v)>)
                    return true;
                return false;
            });
            noa::check(all_varrays, "The outputs should be arrays"); // FIXME this should be a compile time check

            noa::check(ni::are_all_same_device(output_accessors), // FIXME
                       "The output arrays should have the same shape, but got shapes: {}",
                       forward_as_tuple_of_shapes(output_accessors));
            noa::check(are_all_same_shape(outputs),
                       "The output arrays should have the same shape, but got shapes: {}",
                       forward_as_tuple_of_shapes(outputs));
            noa::check(are_all_same_order(outputs),
                       "The output arrays should have the same stride order, but got strides={}",
                       forward_as_tuple_of_strides(outputs));

            // FIXME do we allow strides of 0 on the outputs?

            shape = outputs[Tag<0>{}].shape();
            device = outputs[Tag<0>{}].device();

            // Get order of the output and reorder.
            auto order = ni::order(outputs[Tag<0>{}].strides(), shape);
            if (noa::any(order != Vec4<i64>{0, 1, 2, 3})) {
                const auto reorder_strides = [&order](auto& accessor) { accessor.reorder(order); };
                input_accessors.for_each(reorder_strides);
                output_accessors.for_each(reorder_strides);
            }

            // Automatic broadcasting of the inputs.
            input_accessors.for_each([&shape](auto& input) {
                if (!ni::broadcast(input.shape(), input.strides(), shape)) {
                    noa::panic("Cannot broadcast an array of shape {} into an array of shape {}",
                               input.shape(), shape);
                }
            });

        } else if constexpr (N_INPUTS >= 1) {
            Vec4<i64> order;
            bool found = inputs.any([&](const auto& v) {
                if constexpr (nt::is_varray_v<decltype(v)>) {
                    shape = v.shape();
                    device = v.device();
                    order = ni::order(v.strides(), shape);
                    return true;
                }
                return false;
            });
            noa::check(found, "There should be at least one input array");

            // No automatic broadcasting of the inputs.
            noa::check(ni::are_all_same_shape(inputs),
                       "The input arrays should have the same shape, but got shapes={}",
                       ni::forward_as_tuple_of_shapes(inputs));
            noa::check(ni::are_all_same_device(inputs),
                       "The input arrays should have the same device, but got devices={}",
                       ni::forward_as_tuple_of_devices(inputs));

            // FIXME make take the order of the input with the largest effective shape as reference?
            //       ni::order moves empty dimensions to the left.
            //       I think that forcing the same order is okay, but a bit too restrictive: it effectively prevents broadcasting

            // Get order of the inputs and reorder.
            // Only reorder if all the inputs have the same order.
            const bool do_reorder =
                    noa::any(order != Vec4<i64>{0, 1, 2, 3}) &&
                    input_accessors.all([&order, &shape](const auto& input) {
                        return noa::all(order == ni::order(input.strides(), shape)); // FIXME AccessorValue
                    });
            if (do_reorder) {
                const auto reorder_strides = [&order](auto& accessor) { accessor.reorder(order); };
                input_accessors.for_each(reorder_strides);
                output_accessors.for_each(reorder_strides);
            }
        }

        // FIXME If last 3 dimensions are contiguous, we can
        const bool is_contiguous =
                noa::indexing::are_contiguous(input_strides, shape) &&
                noa::indexing::are_contiguous(output_strides, shape);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            // reduce dimensions
            // check for parallel
            // check for aliasing
        } else {
            // reduce dimensions
        }

        // Reorder inputs to the output order.

    }
}

//
//namespace noa {
//    /// Element-wise transformation using a binary \c operator()(lhs,rhs)->output.
//    /// \param[in] lhs      Left-hand side argument.
//    /// \param[in] rhs      Right-hand side argument.
//    /// \param[out] output  Transformed array.
//    /// \param binary_op    Binary operator. The output is explicitly cast to the \p output value type.
//    /// \note On the GPU, supported operators and types are limited to the following list.
//    ///       Other type combinations can be added (see noa/gpu/cuda/Ewise.hpp).
//    ///     \b Integers:\n
//    ///       - \c (plus|minus|multiply|divide|divide_safe|dist2|min|max)_t(A,A)->A\n
//    ///       - \c (equal|not_equal|less|less_equal|greater|greater_equal|modulo|logical_and|logical_or)_t(A,A)->A\n
//    ///       - \c (equal|not_equal|less|less_equal|greater|greater_equal|logical_and|logical_not)_t(A,A)->bool\n
//    ///     \b Floating-points:\n
//    ///       - \c (plus|minus|multiply|divide|divide_safe|dist2|min|max)_t(B,B)->B\n
//    ///       - \c (equal|not_equal|less|less_equal|greater|greater_equal|pow)_t(B,B)->B\n
//    ///       - \c (equal|not_equal|less|less_equal|greater|greater_equal)_t(B,B)->bool\n
//    ///     \b Complex:\n
//    ///       - \c (plus|minus|multiply|divide|divide_safe|dist2|multiply_conj)_t(C,C)->C\n
//    ///       - \c (plus|minus|multiply|divide|divide_safe|dist2)_t(C,B)->C\n
//    ///       - \c (plus|minus|multiply|divide|divide_safe|dist2)_t(B,C)->C\n
//    ///     \b Where:\n
//    ///         \c A = \c i8,i16,i32,i64,u8,u16,u32,u64\n
//    ///         \c B = \c f16,f32,f64\n
//    ///         \c C = \c c16,c32,c64\n
//    template<typename Lhs, typename Rhs, typename Output, typename BinaryOp,
//             typename = std::enable_if_t<nt::are_varray_v<Lhs, Rhs, Output>>>
//    void ewise_binary(const Lhs& lhs, const Rhs& rhs, const Output& output, BinaryOp&& binary_op) {
//        using lhs_value_t = nt::value_type_t<Lhs>;
//        using rhs_value_t = nt::value_type_t<Rhs>;
//        using output_value_t = nt::value_type_t<Output>;
//        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
//        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
//        static_assert(nt::is_detected_convertible_v<
//                output_value_t, nt::has_binary_operator, BinaryOp, lhs_value_t, rhs_value_t>);
//
//        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty() && !output.is_empty(), "Empty array detected");
//
//        auto lhs_strides = lhs.strides();
//        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      lhs.shape(), output.shape());
//        }
//        auto rhs_strides = rhs.strides();
//        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      rhs.shape(), output.shape());
//        }
//
//        const Device device = output.device();
//        NOA_CHECK(device == lhs.device() && device == rhs.device(),
//                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
//                  lhs.device(), rhs.device(), device);
//
//        Stream& stream = Stream::current(device);
//        if (device.is_cpu()) {
//            auto& cpu_stream = stream.cpu();
//            const auto threads = cpu_stream.thread_limit();
//            cpu_stream.enqueue([=, op = std::forward<BinaryOp>(binary_op)]() {
//                cpu::ewise_binary(lhs.get(), lhs_strides,
//                                  rhs.get(), rhs_strides,
//                                  output.get(), output.strides(), output.shape(),
//                                  op, threads);
//            });
//        } else {
//            #ifdef NOA_ENABLE_CUDA
//            if constexpr (cuda::details::is_valid_ewise_binary_v<
//                    mutable_lhs_value_t, mutable_rhs_value_t, output_value_t,
//                    nt::remove_ref_cv_t<BinaryOp>>) {
//                auto& cuda_stream = stream.cuda();
//                cuda::ewise_binary(lhs.get(), lhs_strides,
//                                   rhs.get(), rhs_strides,
//                                   output.get(), output.strides(), output.shape(),
//                                   binary_op, cuda_stream);
//                cuda_stream.enqueue_attach(lhs, rhs, output);
//            } else {
//                NOA_THROW("These types of operands are not supported by the CUDA backend. "
//                          "See documentation or noa::cuda::ewise_binary(...) for more details");
//            }
//            #else
//            NOA_THROW("No GPU backend detected");
//            #endif
//        }
//    }
//
//    /// Element-wise transformation using a binary \c operator()(lhs,rhs)->output.
//    /// \note On the GPU, the same operators and types are supported as in the overload above.
//    template<typename Lhs, typename Rhs, typename Output, typename BinaryOp,
//             typename = std::enable_if_t<nt::are_varray_v<Lhs, Output> &&
//                                         nt::is_numeric_v<Rhs>>>
//    void ewise_binary(const Lhs& lhs, Rhs rhs, const Output& output, BinaryOp&& binary_op) {
//        using lhs_value_t = nt::value_type_t<Lhs>;
//        using rhs_value_t = nt::value_type_t<Rhs>;
//        using output_value_t = nt::value_type_t<Output>;
//        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
//        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
//        static_assert(nt::is_detected_convertible_v<
//                output_value_t, nt::has_binary_operator, BinaryOp, lhs_value_t, rhs_value_t>);
//
//        NOA_CHECK(!lhs.is_empty() && !output.is_empty(), "Empty array detected");
//
//        auto lhs_strides = lhs.strides();
//        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      lhs.shape(), output.shape());
//        }
//
//        const Device device = output.device();
//        NOA_CHECK(device == lhs.device(),
//                  "The input and output arrays must be on the same device, but got lhs:{}, and output:{}",
//                  lhs.device(), device);
//
//        Stream& stream = Stream::current(device);
//        if (device.is_cpu()) {
//            auto& cpu_stream = stream.cpu();
//            const auto threads = cpu_stream.thread_limit();
//            cpu_stream.enqueue([=, op = std::forward<BinaryOp>(binary_op)]() {
//                cpu::ewise_binary(lhs.get(), lhs_strides,
//                                  rhs,
//                                  output.get(), output.strides(), output.shape(),
//                                  op, threads);
//            });
//        } else {
//            #ifdef NOA_ENABLE_CUDA
//            if constexpr (cuda::details::is_valid_ewise_binary_v<
//                    mutable_lhs_value_t, mutable_rhs_value_t, output_value_t,
//                    nt::remove_ref_cv_t<BinaryOp>>) {
//                auto& cuda_stream = stream.cuda();
//                cuda::ewise_binary(lhs.get(), lhs_strides,
//                                   rhs,
//                                   output.get(), output.strides(), output.shape(),
//                                   binary_op, cuda_stream);
//                cuda_stream.enqueue_attach(lhs, output);
//            } else {
//                NOA_THROW("These types of operands are not supported by the CUDA backend. "
//                          "See documentation or noa::cuda::ewise_binary(...) for more details");
//            }
//            #else
//            NOA_THROW("No GPU backend detected");
//            #endif
//        }
//    }
//
//    /// Element-wise transformation using a binary \c operator()(lhs,rhs)->output.
//    /// \note On the GPU, the same operators and types are supported as in the overload above.
//    template<typename Lhs, typename Rhs, typename Output, typename BinaryOp,
//             typename = std::enable_if_t<nt::are_varray_v<Rhs, Output> &&
//                                         nt::is_numeric_v<Lhs>>>
//    void ewise_binary(Lhs lhs, const Rhs& rhs, const Output& output, BinaryOp&& binary_op) {
//        using lhs_value_t = nt::value_type_t<Lhs>;
//        using rhs_value_t = nt::value_type_t<Rhs>;
//        using output_value_t = nt::value_type_t<Output>;
//        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
//        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
//        static_assert(nt::is_detected_convertible_v<
//                output_value_t, nt::has_binary_operator, BinaryOp, lhs_value_t, rhs_value_t>);
//
//        NOA_CHECK(!rhs.is_empty() && !output.is_empty(), "Empty array detected");
//
//        auto rhs_strides = rhs.strides();
//        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      rhs.shape(), output.shape());
//        }
//
//        const Device device = output.device();
//        NOA_CHECK(device == rhs.device(),
//                  "The input and output arrays must be on the same device, but got rhs:{} and output:{}",
//                  rhs.device(), device);
//
//        Stream& stream = Stream::current(device);
//        if (device.is_cpu()) {
//            auto& cpu_stream = stream.cpu();
//            const auto threads = cpu_stream.thread_limit();
//            cpu_stream.enqueue([=, op = std::forward<BinaryOp>(binary_op)]() {
//                cpu::ewise_binary(lhs,
//                                  rhs.get(), rhs_strides,
//                                  output.get(), output.strides(), output.shape(),
//                                  op, threads);
//            });
//        } else {
//            #ifdef NOA_ENABLE_CUDA
//            if constexpr (cuda::details::is_valid_ewise_binary_v<
//                    mutable_lhs_value_t, mutable_rhs_value_t, output_value_t,
//                    nt::remove_ref_cv_t<BinaryOp>>) {
//                auto& cuda_stream = stream.cuda();
//                cuda::ewise_binary(lhs,
//                                   rhs.get(), rhs_strides,
//                                   output.get(), output.strides(), output.shape(),
//                                   binary_op, cuda_stream);
//                cuda_stream.enqueue_attach(rhs, output);
//            } else {
//                NOA_THROW("These types of operands are not supported by the CUDA backend. "
//                          "See documentation or noa::cuda::ewise_binary(...) for more details");
//            }
//            #else
//            NOA_THROW("No GPU backend detected");
//            #endif
//        }
//    }
//
//    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
//    /// The output is allocated and returned. By default, the output value type is deduced from the operator.
//    /// \note On the GPU, the same operators and types are supported as in the overload above.
//    template<typename Output = void, typename Lhs, typename Rhs, typename BinaryOp,
//             typename = std::enable_if_t<
//                     (std::is_void_v<Output> || nt::is_numeric_v<Output>) &&
//                     (nt::is_varray_v<Lhs> || nt::is_numeric_v<Lhs>) &&
//                     (nt::is_varray_v<Rhs> || nt::is_numeric_v<Rhs>)>>
//    [[nodiscard]] auto ewise_binary(const Lhs& lhs, const Rhs& rhs, BinaryOp&& binary_op) {
//        using lhs_value_t = nt::value_type_t<Lhs>;
//        using rhs_value_t = nt::value_type_t<Rhs>;
//        using return_value_t = std::conditional_t<
//                std::is_void_v<Output>, std::invoke_result_t<BinaryOp, lhs_value_t, rhs_value_t>, Output>;
//        Shape4<i64> shape;
//        ArrayOption options;
//        if constexpr (nt::is_varray_v<Lhs>) {
//            shape = lhs.shape();
//            options = lhs.options();
//        } else if constexpr (nt::is_varray_v<Rhs>) {
//            shape = rhs.shape();
//            options = rhs.options();
//        } else {
//            static_assert(nt::always_false_v<BinaryOp>);
//        }
//        Array<return_value_t> output(shape, options);
//        ewise_binary(lhs, rhs, output, binary_op);
//        return output;
//    }
//}
//
//namespace noa {
//    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
//    /// \param[in] lhs      Left-hand side argument.
//    /// \param[in] mhs      Middle-hand side argument.
//    /// \param[in] rhs      Right-hand side argument.
//    /// \param[out] output  Transformed array.
//    /// \param trinary_op   Trinary operator. The output is explicitly cast to the \p output value type.
//    /// \note On the GPU, supported operators and types are limited to the following list.
//    ///       Other type combinations can be added (see noa/gpu/cuda/Ewise.hpp).
//    ///     \b Integers and Floating-points:\n
//    ///       - \c (within|within_equal)_t(A,A,A)->A|bool\n
//    ///       - \c clamp_t(A,A,A)->A\n
//    ///       - \c (plus|plus_minus|plus_multiply|plus_divide)_t(A,A,A)->A\n
//    ///       - \c (minus|minus_plus|minus_multiply|minus_divide)_t(A,A,A)->A\n
//    ///       - \c (multiply|multiply_plus|multiply_minus|multiply_divide)_t(A,A,A)->A\n
//    ///       - \c (divide|divide_plus|divide_minus|divide_multiply|divide_epsilon)_t(A,A,A)->A\n
//    ///     \b Complex:\n
//    ///       - \c (plus|plus_minus|plus_multiply|plus_divide)_t(B,B,B)->C\n
//    ///       - \c (minus|minus_plus|minus_multiply|minus_divide)_t(B,B,B)->C\n
//    ///       - \c (multiply|multiply_plus|multiply_minus|multiply_divide)_t(B,B,B)->C\n
//    ///       - \c (divide|divide_plus|divide_minus|divide_multiply|divide_epsilon)_t(B,B,B)->C\n
//    ///     \b Where:\n
//    ///         \c A = \c i8,i16,i32,i64,u8,u16,u32,u64,f16,f32,f64\n
//    ///         \c B = \c f16,f32,f64,c16,c32,c64\n
//    ///         \c C = \c c16,c32,c64\n
//    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
//             typename = std::enable_if_t<nt::are_varray_v<Lhs, Mhs, Rhs, Output>>>
//    void ewise_trinary(const Lhs& lhs, const Mhs& mhs, const Rhs& rhs, const Output& output, TrinaryOp&& trinary_op) {
//        using lhs_value_t = nt::value_type_t<Lhs>;
//        using mhs_value_t = nt::value_type_t<Mhs>;
//        using rhs_value_t = nt::value_type_t<Rhs>;
//        using output_value_t = nt::value_type_t<Output>;
//        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
//        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
//        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
//        static_assert(nt::is_detected_convertible_v<
//                output_value_t, nt::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);
//
//        NOA_CHECK(!lhs.is_empty() && !mhs.is_empty() && !rhs.is_empty() && !output.is_empty(), "Empty array detected");
//
//        auto lhs_strides = lhs.strides();
//        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      lhs.shape(), output.shape());
//        }
//        auto mhs_strides = mhs.strides();
//        if (!noa::indexing::broadcast(mhs.shape(), mhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      mhs.shape(), output.shape());
//        }
//        auto rhs_strides = rhs.strides();
//        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      rhs.shape(), output.shape());
//        }
//
//        const Device device = output.device();
//        NOA_CHECK(device == lhs.device() && device == mhs.device() && device == rhs.device(),
//                  "The input and output arrays must be on the same device, but got lhs:{}, mhs:{}, rhs:{} and output:{}",
//                  lhs.device(), mhs.device(), rhs.device(), device);
//
//        Stream& stream = Stream::current(device);
//        if (device.is_cpu()) {
//            auto& cpu_stream = stream.cpu();
//            const auto threads = cpu_stream.thread_limit();
//            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
//                cpu::ewise_trinary(lhs.get(), lhs_strides,
//                                   mhs.get(), mhs_strides,
//                                   rhs.get(), rhs_strides,
//                                   output.get(), output.strides(), output.shape(),
//                                   op, threads);
//            });
//        } else {
//            #ifdef NOA_ENABLE_CUDA
//            if constexpr (cuda::details::is_valid_ewise_trinary_v<
//                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
//                    nt::remove_ref_cv_t<TrinaryOp>>) {
//                auto& cuda_stream = stream.cuda();
//                cuda::ewise_trinary(lhs.get(), lhs_strides,
//                                    mhs.get(), mhs_strides,
//                                    rhs.get(), rhs_strides,
//                                    output.get(), output.strides(), output.shape(),
//                                    trinary_op, cuda_stream);
//                cuda_stream.enqueue_attach(lhs, mhs, rhs, output);
//            } else {
//                NOA_THROW("These types of operands are not supported by the CUDA backend. "
//                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
//            }
//            #else
//            NOA_THROW("No GPU backend detected");
//            #endif
//        }
//    }
//
//    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
//    /// \note On the GPU, the same operators and types are supported as in the overload above.
//    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
//             typename = std::enable_if_t<nt::are_varray_v<Lhs, Mhs, Output> &&
//                                         nt::is_numeric_v<Rhs>>>
//    void ewise_trinary(const Lhs& lhs, const Mhs& mhs, Rhs rhs, const Output& output, TrinaryOp&& trinary_op) {
//        using lhs_value_t = nt::value_type_t<Lhs>;
//        using mhs_value_t = nt::value_type_t<Mhs>;
//        using rhs_value_t = nt::value_type_t<Rhs>;
//        using output_value_t = nt::value_type_t<Output>;
//        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
//        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
//        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
//        static_assert(nt::is_detected_convertible_v<
//                output_value_t, nt::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);
//
//        NOA_CHECK(!lhs.is_empty() && !mhs.is_empty() && !output.is_empty(), "Empty array detected");
//
//        auto lhs_strides = lhs.strides();
//        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      lhs.shape(), output.shape());
//        }
//        auto mhs_strides = mhs.strides();
//        if (!noa::indexing::broadcast(mhs.shape(), mhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      mhs.shape(), output.shape());
//        }
//
//        const Device device = output.device();
//        NOA_CHECK(device == lhs.device() && device == mhs.device(),
//                  "The input and output arrays must be on the same device, but got lhs:{}, mhs:{} and output:{}",
//                  lhs.device(), mhs.device(), device);
//
//        Stream& stream = Stream::current(device);
//        if (device.is_cpu()) {
//            auto& cpu_stream = stream.cpu();
//            const auto threads = cpu_stream.thread_limit();
//            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
//                cpu::ewise_trinary(lhs.get(), lhs_strides,
//                                   mhs.get(), mhs_strides,
//                                   rhs,
//                                   output.get(), output.strides(), output.shape(),
//                                   op, threads);
//            });
//        } else {
//            #ifdef NOA_ENABLE_CUDA
//            if constexpr (cuda::details::is_valid_ewise_trinary_v<
//                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
//                    nt::remove_ref_cv_t<TrinaryOp>>) {
//                auto& cuda_stream = stream.cuda();
//                cuda::ewise_trinary(lhs.get(), lhs_strides,
//                                    mhs.get(), mhs_strides,
//                                    rhs,
//                                    output.get(), output.strides(), output.shape(),
//                                    trinary_op, cuda_stream);
//                cuda_stream.enqueue_attach(lhs, mhs, output);
//            } else {
//                NOA_THROW("These types of operands are not supported by the CUDA backend. "
//                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
//            }
//            #else
//            NOA_THROW("No GPU backend detected");
//            #endif
//        }
//    }
//
//    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
//    /// \note On the GPU, the same operators and types are supported as in the overload above.
//    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
//             typename = std::enable_if_t<nt::are_varray_v<Lhs, Rhs, Output> &&
//                                         nt::is_numeric_v<Mhs>>>
//    void ewise_trinary(const Lhs& lhs, Mhs mhs, const Rhs& rhs, const Output& output, TrinaryOp&& trinary_op) {
//        using lhs_value_t = nt::value_type_t<Lhs>;
//        using mhs_value_t = nt::value_type_t<Mhs>;
//        using rhs_value_t = nt::value_type_t<Rhs>;
//        using output_value_t = nt::value_type_t<Output>;
//        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
//        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
//        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
//        static_assert(nt::is_detected_convertible_v<
//                output_value_t, nt::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);
//
//        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty() && !output.is_empty(), "Empty array detected");
//
//        auto lhs_strides = lhs.strides();
//        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      lhs.shape(), output.shape());
//        }
//        auto rhs_strides = rhs.strides();
//        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      rhs.shape(), output.shape());
//        }
//
//        const Device device = output.device();
//        NOA_CHECK(device == lhs.device() && device == rhs.device(),
//                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
//                  lhs.device(), rhs.device(), device);
//
//        Stream& stream = Stream::current(device);
//        if (device.is_cpu()) {
//            auto& cpu_stream = stream.cpu();
//            const auto threads = cpu_stream.thread_limit();
//            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
//                cpu::ewise_trinary(lhs.get(), lhs_strides,
//                                   mhs,
//                                   rhs.get(), rhs_strides,
//                                   output.get(), output.strides(), output.shape(),
//                                   op, threads);
//            });
//        } else {
//            #ifdef NOA_ENABLE_CUDA
//            if constexpr (cuda::details::is_valid_ewise_trinary_v<
//                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
//                    nt::remove_ref_cv_t<TrinaryOp>>) {
//                auto& cuda_stream = stream.cuda();
//                cuda::ewise_trinary(lhs.get(), lhs_strides,
//                                    mhs,
//                                    rhs.get(), rhs_strides,
//                                    output.get(), output.strides(), output.shape(),
//                                    trinary_op, cuda_stream);
//                cuda_stream.enqueue_attach(lhs, rhs, output);
//            } else {
//                NOA_THROW("These types of operands are not supported by the CUDA backend. "
//                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
//            }
//            #else
//            NOA_THROW("No GPU backend detected");
//            #endif
//        }
//    }
//
//    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
//    /// \note On the GPU, the same operators and types are supported as in the overload above.
//    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
//             typename = std::enable_if_t<nt::are_varray_v<Mhs, Rhs, Output> &&
//                                         nt::is_numeric_v<Lhs>>>
//    void ewise_trinary(Lhs lhs, const Mhs& mhs, const Rhs& rhs, const Output& output, TrinaryOp&& trinary_op) {
//        using lhs_value_t = nt::value_type_t<Lhs>;
//        using mhs_value_t = nt::value_type_t<Mhs>;
//        using rhs_value_t = nt::value_type_t<Rhs>;
//        using output_value_t = nt::value_type_t<Output>;
//        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
//        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
//        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
//        static_assert(nt::is_detected_convertible_v<
//                output_value_t, nt::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);
//
//        NOA_CHECK(!mhs.is_empty() && !rhs.is_empty() && !output.is_empty(), "Empty array detected");
//
//        auto mhs_strides = mhs.strides();
//        if (!noa::indexing::broadcast(mhs.shape(), mhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      mhs.shape(), output.shape());
//        }
//        auto rhs_strides = rhs.strides();
//        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      rhs.shape(), output.shape());
//        }
//
//        const Device device = output.device();
//        NOA_CHECK(device == mhs.device() && device == rhs.device(),
//                  "The input and output arrays must be on the same device, but got mhs:{}, rhs:{} and output:{}",
//                  mhs.device(), rhs.device(), device);
//
//        Stream& stream = Stream::current(device);
//        if (device.is_cpu()) {
//            auto& cpu_stream = stream.cpu();
//            const auto threads = cpu_stream.thread_limit();
//            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
//                cpu::ewise_trinary(lhs,
//                                   mhs.get(), mhs_strides,
//                                   rhs.get(), rhs_strides,
//                                   output.get(), output.strides(), output.shape(),
//                                   op, threads);
//            });
//        } else {
//            #ifdef NOA_ENABLE_CUDA
//            if constexpr (cuda::details::is_valid_ewise_trinary_v<
//                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
//                    nt::remove_ref_cv_t<TrinaryOp>>) {
//                auto& cuda_stream = stream.cuda();
//                cuda::ewise_trinary(lhs,
//                                    mhs.get(), mhs_strides,
//                                    rhs.get(), rhs_strides,
//                                    output.get(), output.strides(), output.shape(),
//                                    trinary_op, cuda_stream);
//                cuda_stream.enqueue_attach(mhs, rhs, output);
//            } else {
//                NOA_THROW("These types of operands are not supported by the CUDA backend. "
//                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
//            }
//            #else
//            NOA_THROW("No GPU backend detected");
//            #endif
//        }
//    }
//
//    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
//    /// \note On the GPU, the same operators and types are supported as in the overload above.
//    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
//             typename = std::enable_if_t<nt::are_varray_v<Lhs, Output> &&
//                                         nt::are_numeric_v<Mhs, Rhs>>>
//    void ewise_trinary(const Lhs& lhs, Mhs mhs, Rhs rhs, const Output& output, TrinaryOp&& trinary_op) {
//        using lhs_value_t = nt::value_type_t<Lhs>;
//        using mhs_value_t = nt::value_type_t<Mhs>;
//        using rhs_value_t = nt::value_type_t<Rhs>;
//        using output_value_t = nt::value_type_t<Output>;
//        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
//        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
//        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
//        static_assert(nt::is_detected_convertible_v<
//                output_value_t, nt::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);
//
//        NOA_CHECK(!lhs.is_empty() && !output.is_empty(), "Empty array detected");
//
//        auto lhs_strides = lhs.strides();
//        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      lhs.shape(), output.shape());
//        }
//
//        const Device device = output.device();
//        NOA_CHECK(device == lhs.device(),
//                  "The input and output arrays must be on the same device, but got lhs:{} and output:{}",
//                  lhs.device(), device);
//
//        Stream& stream = Stream::current(device);
//        if (device.is_cpu()) {
//            auto& cpu_stream = stream.cpu();
//            const auto threads = cpu_stream.thread_limit();
//            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
//                cpu::ewise_trinary(lhs.get(), lhs_strides,
//                                   mhs,
//                                   rhs,
//                                   output.get(), output.strides(), output.shape(),
//                                   op, threads);
//            });
//        } else {
//            #ifdef NOA_ENABLE_CUDA
//            if constexpr (cuda::details::is_valid_ewise_trinary_v<
//                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
//                    nt::remove_ref_cv_t<TrinaryOp>>) {
//                auto& cuda_stream = stream.cuda();
//                cuda::ewise_trinary(lhs.get(), lhs_strides,
//                                    mhs,
//                                    rhs,
//                                    output.get(), output.strides(), output.shape(),
//                                    trinary_op, cuda_stream);
//                cuda_stream.enqueue_attach(lhs, output);
//            } else {
//                NOA_THROW("These types of operands are not supported by the CUDA backend. "
//                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
//            }
//            #else
//            NOA_THROW("No GPU backend detected");
//            #endif
//        }
//    }
//
//    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
//    /// \note On the GPU, the same operators and types are supported as in the overload above.
//    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
//             typename = std::enable_if_t<nt::are_varray_v<Mhs, Output> &&
//                                         nt::are_numeric_v<Lhs, Rhs>>>
//    void ewise_trinary(Lhs lhs, const Mhs& mhs, Rhs rhs, const Output& output, TrinaryOp&& trinary_op) {
//        using lhs_value_t = nt::value_type_t<Lhs>;
//        using mhs_value_t = nt::value_type_t<Mhs>;
//        using rhs_value_t = nt::value_type_t<Rhs>;
//        using output_value_t = nt::value_type_t<Output>;
//        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
//        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
//        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
//        static_assert(nt::is_detected_convertible_v<
//                output_value_t, nt::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);
//
//        NOA_CHECK(!mhs.is_empty() && !output.is_empty(), "Empty array detected");
//
//        auto mhs_strides = mhs.strides();
//        if (!noa::indexing::broadcast(mhs.shape(), mhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      mhs.shape(), output.shape());
//        }
//
//        const Device device = output.device();
//        NOA_CHECK(device == mhs.device(),
//                  "The input and output arrays must be on the same device, but got mhs:{} and output:{}",
//                  mhs.device(), device);
//
//        Stream& stream = Stream::current(device);
//        if (device.is_cpu()) {
//            auto& cpu_stream = stream.cpu();
//            const auto threads = cpu_stream.thread_limit();
//            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
//                cpu::ewise_trinary(lhs,
//                                   mhs.get(), mhs_strides,
//                                   rhs,
//                                   output.get(), output.strides(), output.shape(),
//                                   op, threads);
//            });
//        } else {
//            #ifdef NOA_ENABLE_CUDA
//            if constexpr (cuda::details::is_valid_ewise_trinary_v<
//                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
//                    nt::remove_ref_cv_t<TrinaryOp>>) {
//                auto& cuda_stream = stream.cuda();
//                cuda::ewise_trinary(lhs,
//                                    mhs.get(), mhs_strides,
//                                    rhs,
//                                    output.get(), output.strides(), output.shape(),
//                                    trinary_op, cuda_stream);
//                cuda_stream.enqueue_attach(mhs, output);
//            } else {
//                NOA_THROW("These types of operands are not supported by the CUDA backend. "
//                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
//            }
//            #else
//            NOA_THROW("No GPU backend detected");
//            #endif
//        }
//    }
//
//    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
//    /// \note On the GPU, the same operators and types are supported as in the overload above.
//    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename TrinaryOp,
//             typename = std::enable_if_t<nt::are_varray_v<Rhs, Output> &&
//                                         nt::are_numeric_v<Lhs, Mhs>>>
//    void ewise_trinary(Lhs lhs, Mhs mhs, const Rhs& rhs, const Output& output, TrinaryOp&& trinary_op) {
//        using lhs_value_t = nt::value_type_t<Lhs>;
//        using mhs_value_t = nt::value_type_t<Mhs>;
//        using rhs_value_t = nt::value_type_t<Rhs>;
//        using output_value_t = nt::value_type_t<Output>;
//        using mutable_lhs_value_t = std::remove_const_t<lhs_value_t>;
//        using mutable_mhs_value_t = std::remove_const_t<mhs_value_t>;
//        using mutable_rhs_value_t = std::remove_const_t<rhs_value_t>;
//        static_assert(nt::is_detected_convertible_v<
//                output_value_t, nt::has_trinary_operator, TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>);
//
//        NOA_CHECK(!rhs.is_empty() && !output.is_empty(), "Empty array detected");
//
//        auto rhs_strides = rhs.strides();
//        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, output.shape())) {
//            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
//                      rhs.shape(), output.shape());
//        }
//
//        const Device device = output.device();
//        NOA_CHECK(device == rhs.device(),
//                  "The input and output arrays must be on the same device, but got rhs:{} and output:{}",
//                  rhs.device(), device);
//
//        Stream& stream = Stream::current(device);
//        if (device.is_cpu()) {
//            auto& cpu_stream = stream.cpu();
//            const auto threads = cpu_stream.thread_limit();
//            cpu_stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
//                cpu::ewise_trinary(lhs,
//                                   mhs,
//                                   rhs.get(), rhs_strides,
//                                   output.get(), output.strides(), output.shape(),
//                                   op, threads);
//            });
//        } else {
//            #ifdef NOA_ENABLE_CUDA
//            if constexpr (cuda::details::is_valid_ewise_trinary_v<
//                    mutable_lhs_value_t, mutable_mhs_value_t, mutable_rhs_value_t, output_value_t,
//                    nt::remove_ref_cv_t<TrinaryOp>>) {
//                auto& cuda_stream = stream.cuda();
//                cuda::ewise_trinary(lhs,
//                                    mhs,
//                                    rhs.get(), rhs_strides,
//                                    output.get(), output.strides(), output.shape(),
//                                    trinary_op, cuda_stream);
//                cuda_stream.enqueue_attach(rhs, output);
//            } else {
//                NOA_THROW("These types of operands are not supported by the CUDA backend. "
//                          "See documentation or noa::cuda::ewise_trinary(...) for more details");
//            }
//            #else
//            NOA_THROW("No GPU backend detected");
//            #endif
//        }
//    }
//
//    /// Element-wise transformation using a trinary \c operator()(lhs,mhs,rhs)->output.
//    /// The output is allocated and returned. By default, the output value type is deduced from the operator.
//    /// \note On the GPU, the same operators and types are supported as in the overload above.
//    template<typename Output = void, typename Lhs, typename Mhs, typename Rhs, typename TrinaryOp,
//             typename = std::enable_if_t<
//                     (std::is_void_v<Output> || nt::is_numeric_v<Output>) &&
//                     (nt::is_varray_v<Lhs> || nt::is_numeric_v<Lhs>) &&
//                     (nt::is_varray_v<Mhs> || nt::is_numeric_v<Mhs>) &&
//                     (nt::is_varray_v<Rhs> || nt::is_numeric_v<Rhs>)>>
//    [[nodiscard]] auto ewise_trinary(const Lhs& lhs, const Mhs& mhs, const Rhs& rhs, TrinaryOp&& trinary_op) {
//        using lhs_value_t = nt::value_type_t<Lhs>;
//        using mhs_value_t = nt::value_type_t<Mhs>;
//        using rhs_value_t = nt::value_type_t<Rhs>;
//        using return_value_t = std::conditional_t<
//                std::is_void_v<Output>, std::invoke_result_t<TrinaryOp, lhs_value_t, mhs_value_t, rhs_value_t>, Output>;
//        Shape4<i64> shape;
//        ArrayOption options;
//        if constexpr (nt::is_varray_v<Lhs>) {
//            shape = lhs.shape();
//            options = lhs.options();
//        } else if constexpr (nt::is_varray_v<Mhs>) {
//            shape = mhs.shape();
//            options = mhs.options();
//        } else if constexpr (nt::is_varray_v<Rhs>) {
//            shape = rhs.shape();
//            options = rhs.options();
//        } else {
//            static_assert(nt::always_false_v<TrinaryOp>);
//        }
//        Array<return_value_t> output(shape, options);
//        ewise_trinary(lhs, mhs, rhs, output, trinary_op);
//        return output;
//    }
//}
