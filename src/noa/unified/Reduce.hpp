#pragma once

#include "noa/cpu/utils/ReduceUnary.hpp"
#include "noa/cpu/utils/ReduceBinary.hpp"

#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"

namespace noa::details {
    namespace nt = noa::traits;

    template<typename Input, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    constexpr bool is_valid_reduce_unary_v =
            (nt::is_detected_exact_v<Reduced, nt::has_unary_operator, PreProcessOp, Input> ||
             nt::is_detected_exact_v<Reduced, nt::has_binary_operator, PreProcessOp, Input, Index>) &&
            nt::is_detected_exact_v<Reduced, nt::has_binary_operator, ReduceOp, Reduced, Reduced> &&
            nt::is_detected_convertible_v<Output, nt::has_unary_operator, PostProcessOp, Reduced> &&
            !std::is_const_v<Reduced> && !std::is_const_v<Output> && std::is_integral_v<Index>;

    template<typename Lhs, typename Rhs, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    constexpr bool is_valid_reduce_binary_v =
            nt::is_detected_exact_v<Reduced, nt::has_unary_operator, PreProcessOp, Lhs, Rhs> &&
            nt::is_detected_exact_v<Reduced, nt::has_binary_operator, ReduceOp, Reduced, Reduced> &&
            nt::is_detected_convertible_v<Output, nt::has_unary_operator, PostProcessOp, Reduced> &&
            !std::is_const_v<Reduced> && !std::is_const_v<Output> && std::is_integral_v<Index>;
}

namespace noa {
    /// Generic element-wise (B)DHW unary reduction.
    /// \details Merges in a single step, a generic ewise_unary, a generic reduction and a ewise_unary, in this order.
    ///
    /// \tparam Index               Index type. On the GPU, 4-bytes integers can be faster.
    /// \param input                Input BDHW to reduce.
    /// \param output               Output (batched) reduced array.
    /// \param initial_reduce       Initial value to pass to the reduce operator.
    /// \param pre_process_op       Initial pre-processing element-wise operator. \c pre_process_op(Input)->Reduced.
    /// \param reduce_op            Reduction operator. \c reduce_op(Reduced,Reduced)->Reduced.
    /// \param post_process_op      Post-processing element-wise operator to run on the last reduced element(s).
    ///                             \c pre_process_op(Reduced)->Output.
    /// \param allow_swap_layout    Whether the dimensions are allowed to be swapped for best performance.
    ///                             If the batch dimension is not reduced, only the DHW dimensions can be swapped.
    /// \note This function is currently not supported on the GPU.
    template<typename Index = i64, typename Input, typename Reduced, typename Output,
             typename PreProcessOp = noa::copy_t,
             typename ReduceOp,
             typename PostProcessOp = noa::copy_t,
             typename = std::enable_if_t<
                     noa::traits::are_array_or_view_v<Input, Output> &&
                     (details::is_valid_reduce_unary_v<
                             noa::traits::value_type_t<Input>, Reduced,
                             noa::traits::value_type_t<Output>, Index,
                             PreProcessOp, ReduceOp, PostProcessOp>)>>
    constexpr void reduce_unary(
            const Input& input, const Output& output,
            Reduced initial_reduce,
            PreProcessOp&& pre_process_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op,
            bool allow_swap_layout = true
    ) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The arrays should be on the same device, but got input={} and output={}",
                  input.device(), device);

        NOA_CHECK(noa::all(output.shape().pop_front() == Shape3<i64>(1)) &&
                  (output.shape()[0] == 1 || output.shape()[0] == input.shape()[0]),
                  "The output array should have its DHW dimensions reduced and its batch should be 1 or "
                  "be the same as the input. Got shape input={} and output={}",
                  input.shape(), output.shape());
        const bool reduce_batch = output.shape()[0] == 1;

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            noa::cpu::utils::reduce_unary(
                    input.data(), input.strides(), input.shape(),
                    output.data(), output.strides().filter(0),
                    initial_reduce,
                    std::forward<PreProcessOp>(pre_process_op),
                    std::forward<ReduceOp>(reduce_op),
                    std::forward<PostProcessOp>(post_process_op),
                    cpu_stream.threads(),
                    reduce_batch, allow_swap_layout);
        } else {
            #ifdef NOA_ENABLE_CUDA
            NOA_THROW("These (combination of) types are not supported by the CUDA backend"); // TODO
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Generic element-wise BDHW unary reduction.
    /// This overload is the same as above, but returns a single value.
    template<typename Index = i64, typename Input, typename Reduced,
             typename PreProcessOp = noa::copy_t,
             typename ReduceOp,
             typename PostProcessOp = noa::copy_t,
             typename = std::enable_if_t<
                     noa::traits::are_array_or_view_v<Input, View<std::invoke_result_t<ReduceOp, Reduced, Reduced>>> &&
                     (details::is_valid_reduce_unary_v<
                             noa::traits::value_type_t<Input>, Reduced,
                             std::invoke_result_t<ReduceOp, Reduced, Reduced>, Index,
                             PreProcessOp, ReduceOp, PostProcessOp>)>>
    constexpr auto reduce_unary(
            const Input& input,
            Reduced initial_reduce,
            PreProcessOp&& pre_process_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op,
            bool allow_swap_layout = true
    ) {
        using output_value_t = std::invoke_result_t<PostProcessOp, Reduced>;
        output_value_t output_value{};
        const auto output_cpu = View(&output_value, 1);

        Device device = input.device();
        if (device.is_cpu()) {
            reduce_unary<Index>(
                    input, output_cpu, initial_reduce,
                    std::forward<PreProcessOp>(pre_process_op),
                    std::forward<ReduceOp>(reduce_op),
                    std::forward<PostProcessOp>(post_process_op),
                    allow_swap_layout);
        } else {
            Array<output_value_t> output_gpu(1, ArrayOption(device, Allocator::DEFAULT_ASYNC));
            reduce_unary<Index>(
                    input, output_gpu.view(), initial_reduce,
                    std::forward<PreProcessOp>(pre_process_op),
                    std::forward<ReduceOp>(reduce_op),
                    std::forward<PostProcessOp>(post_process_op),
                    allow_swap_layout);
            output_gpu.to(output_cpu);
        }

        return *output_cpu.eval().data();
    }

    /// Generic element-wise (B)DHW binary reduction.
    /// \details Merges in a single step, a generic ewise_binary, a generic reduction and a ewise_unary, in this order.
    ///
    /// \tparam Index               Index type. On the GPU, 4-bytes integers can be faster.
    /// \param lhs                  Lhs BDHW to reduce.
    /// \param rhs                  Rhs BDHW to reduce.
    /// \param output               Output (batched) reduced array.
    /// \param initial_reduce       Initial value to pass to the reduce operator.
    /// \param pre_process_op       Initial pre-processing element-wise operator. \c pre_process_op(Lhs, Rhs)->Reduced.
    /// \param reduce_op            Reduction operator. \c reduce_op(Reduced,Reduced)->Reduced.
    /// \param post_process_op      Post-processing element-wise operator to run on the last reduced element(s).
    ///                             \c pre_process_op(Reduced)->Output.
    /// \param allow_swap_layout    Whether the dimensions are allowed to be swapped for best performance.
    ///                             If the batch dimension is not reduced, only the DHW dimensions can be swapped.
    /// \note This function is currently not supported on the GPU.
    template<typename Index = i64, typename Lhs, typename Rhs, typename Reduced, typename Output,
             typename PreProcessOp = noa::copy_t,
             typename ReduceOp,
             typename PostProcessOp = noa::copy_t,
             typename = std::enable_if_t<
                     noa::traits::are_array_or_view_v<Lhs, Output> &&
                     details::is_valid_reduce_binary_v<
                             noa::traits::value_type_t<Lhs>,
                             noa::traits::value_type_t<Rhs>, Reduced,
                             noa::traits::value_type_t<Output>, Index,
                             PreProcessOp, ReduceOp, PostProcessOp>>>
    constexpr void reduce_binary(
            const Lhs& lhs, const Rhs& rhs, const Output& output,
            Reduced initial_reduce,
            PreProcessOp&& pre_process_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op,
            bool allow_swap_layout = true
    ) {
        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty() && !output.is_empty(), "Empty array detected");

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The arrays should be on the same device, but got lhs={}, rhs={} and output={}",
                  lhs.device(), rhs.device(), device);

        NOA_CHECK(noa::all(lhs.shape() == rhs.shape()),
                  "The lhs and rhs arrays should have the same shape, but got lhs={} and rhs={}",
                  lhs.shape(), rhs.shape());
        NOA_CHECK(noa::all(output.shape().pop_front() == Shape3<i64>(1)) &&
                  (output.shape()[0] == 1 || output.shape()[0] == lhs.shape()[0]),
                  "The output array should have its DHW dimensions reduced and its batch should be 1 or "
                  "be the same as the inputs. Got shape inputs={} and output={}",
                  lhs.shape(), output.shape());
        const bool reduce_batch = output.shape()[0] == 1;

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            noa::cpu::utils::reduce_binary(
                    lhs.data(), lhs.strides(),
                    rhs.data(), rhs.strides(), rhs.shape(),
                    output.data(), output.strides().filter(0),
                    initial_reduce,
                    std::forward<PreProcessOp>(pre_process_op),
                    std::forward<ReduceOp>(reduce_op),
                    std::forward<PostProcessOp>(post_process_op),
                    cpu_stream.threads(),
                    reduce_batch, allow_swap_layout);
        } else {
            #ifdef NOA_ENABLE_CUDA
            NOA_THROW("These (combination of) types are not supported by the CUDA backend"); // TODO
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Generic element-wise BDHW binary reduction.
    /// This overload is the same as above, but returns a single value.
    template<typename Index = i64, typename Lhs, typename Rhs, typename Reduced,
             typename PreProcessOp = noa::copy_t,
             typename ReduceOp,
             typename PostProcessOp = noa::copy_t,
             typename = std::enable_if_t<
                     noa::traits::are_array_or_view_v<Lhs, View<std::invoke_result_t<ReduceOp, Reduced, Reduced>>> &&
                     details::is_valid_reduce_binary_v<
                             noa::traits::value_type_t<Lhs>,
                             noa::traits::value_type_t<Rhs>, Reduced,
                             std::invoke_result_t<ReduceOp, Reduced, Reduced>, Index,
                             PreProcessOp, ReduceOp, PostProcessOp>>>
    constexpr auto reduce_binary(
            const Lhs& lhs, const Rhs& rhs,
            Reduced initial_reduce,
            PreProcessOp&& pre_process_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op,
            bool allow_swap_layout = true
    ) {
        using output_value_t = std::invoke_result_t<PostProcessOp, Reduced>;
        output_value_t output_value{};
        const auto output_cpu = View(&output_value, 1);

        Device device = rhs.device();
        if (device.is_cpu()) {
            reduce_binary<Index>(
                    lhs, rhs, output_cpu, initial_reduce,
                    std::forward<PreProcessOp>(pre_process_op),
                    std::forward<ReduceOp>(reduce_op),
                    std::forward<PostProcessOp>(post_process_op),
                    allow_swap_layout);
        } else {
            Array<output_value_t> output_gpu(1, ArrayOption(device, Allocator::DEFAULT_ASYNC));
            reduce_binary<Index>(
                    lhs, rhs, output_gpu.view(), initial_reduce,
                    std::forward<PreProcessOp>(pre_process_op),
                    std::forward<ReduceOp>(reduce_op),
                    std::forward<PostProcessOp>(post_process_op),
                    allow_swap_layout);
            output_gpu.to(output_cpu);
        }

        return *output_cpu.eval().data();
    }
}
