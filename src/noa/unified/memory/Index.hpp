#pragma once

#include "noa/cpu/memory/Index.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Index.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::memory {
    template<typename Value, typename Offset>
    struct Extracted {
        Array<Value> values;
        Array<Offset> offsets;
    };

    /// Extracts elements (and/or offsets) from the input array based on an unary bool operator.
    /// \tparam Value           Any numeric type.
    /// \tparam Offset          Any integral type.
    /// \param[in] input        Input array to extract from. Can be empty if \p extract_values is false.
    /// \param[lhs] lhs         Left-hand side argument.
    /// \param unary_op         Unary operation device function object that will be used as criterion to extract elements.
    ///                         Each element of \p lhs is passed through that operator and if the return value evaluates
    ///                         to true, the corresponding element in \p input is extracted.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \return 1: 1D array with the extracted values. Can be empty, depending on \p extract_values.
    ///         2: 1D array with the extracted offsets. Can be empty, depending on \p extract_offsets.
    ///         Both arrays are on the same device as \p input, or if it is empty, the same device as \p lhs.
    /// \note On the GPU:
    ///         - \p Value should be one of i32, i64, u32, u64, f16, f32, f64.
    ///         - \p Offset should be one of i32, i64, u32, u64.
    ///         - The input and lhs value type should be equal to \p Value.
    ///         - \p unary_op is limited to \c noa::logical_not_t.
    template<typename Value, typename Offset, typename Input, typename Lhs, typename UnaryOp,
            typename = std::enable_if_t<noa::traits::are_array_or_view_of_restricted_numeric_v<Input, Lhs>>>
    [[nodiscard]] Extracted<Value, Offset>
    extract_unary(const Input& input, const Lhs& lhs, UnaryOp&& unary_op,
                  bool extract_values = true, bool extract_offsets = true) {
        NOA_CHECK(!extract_values || !input.is_empty(), "The input array should not be empty");
        NOA_CHECK(input.is_empty() || noa::all(input.shape() == lhs.shape()),
                  "The input arrays should have the same shape, but got input:{} and lhs:{}",
                  input.shape(), lhs.shape());
        NOA_CHECK(input.is_empty() || input.device() == lhs.device(),
                  "The input arrays should be on the same device, but got input:{} and lhs:{}",
                  input.device(), lhs.device());

        Extracted<Value, Offset> output;
        const Device device = lhs.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            stream.synchronize();
            auto extracted = cpu::memory::extract_unary<Value, Offset>(
                    input.get(), input.strides(), lhs.get(), lhs.strides(), input.shape(),
                    std::forward<UnaryOp>(unary_op), extract_values, extract_offsets);
            output.values = Array<Value>(extracted.values, extracted.count);
            output.offsets = Array<Offset>(extracted.offsets, extracted.count);
        } else {
            #ifdef NOA_ENABLE_CUDA
            using input_t = std::remove_const_t<noa::traits::value_type_t<Input>>;
            using lhs_t = std::remove_const_t<noa::traits::value_type_t<Lhs>>;
            if constexpr (cuda::memory::details::is_valid_extract_unary_v<input_t, lhs_t, UnaryOp, Value, Offset>) {
                auto& cuda_stream = stream.cuda();
                auto extracted = cuda::memory::extract_unary<Value, Offset>(
                        input.get(), input.strides(), lhs.get(), lhs.strides(), input.shape(),
                        unary_op, extract_values, extract_offsets, cuda_stream);
                cuda_stream.enqueue_attach(input.share(), lhs.share());
                const auto option = ArrayOption(input.device(), Allocator::DEFAULT_ASYNC);
                output.values = Array<Value>(extracted.values, extracted.count, option);
                output.offsets = Array<Offset>(extracted.offsets, extracted.count, option);
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract_unary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        return output;
    }

    /// Extracts elements (and/or offsets) from the input array based on a binary bool operator.
    /// \tparam Value           Any numeric type.
    /// \tparam Offset          Any integral type.
    /// \param[in] input        Input array to extract from. Can be empty if \p extract_values is false.
    /// \param[in] lhs          Left-hand side argument.
    /// \param[in] rhs          Right-hand side argument.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of both \p lhs and \p rhs are passed through that operator and if the
    ///                         return value evaluates to true, the corresponding element in \p input is extracted.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    ///
    /// \return 1: 1D array with the extracted elements. Can be empty, depending on \p extract_values.
    ///         2: 1D array with the extracted offsets. Can be empty, depending on \p extract_offsets.
    ///         Both arrays are on the same device as \p input, or if it is empty, the same device as \p lhs and \p rhs.
    /// \note On the GPU:
    ///         - \p Value should be one of i32, i64, u32, u64, f16, f32, f64.
    ///         - \p Offset should be one of i32, i64, u32, u64.
    ///         - The input value types should be equal to \p Value.
    ///         - \p binary_op is limited to \c noa::{equal|not_equal|less|less_equal|greater|greater_equal}_t.
    template<typename Value, typename Offset, typename Input, typename Lhs, typename Rhs, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::are_array_or_view_of_restricted_numeric_v<Input, Lhs, Rhs>>>
    [[nodiscard]] Extracted<Value, Offset>
    extract_binary(const Input& input, const Lhs& lhs, const Rhs& rhs, BinaryOp&& binary_op,
                   bool extract_values = true, bool extract_offsets = true) {
        NOA_CHECK(!extract_values || !input.is_empty(), "The input array should not be empty");
        NOA_CHECK(input.is_empty() || (noa::all(input.shape() == lhs.shape()) && noa::all(input.shape() == rhs.shape())),
                  "The input arrays should have the same shape, but got input:{}, lhs:{} and rhs:{}",
                  input.shape(), lhs.shape(), rhs.shape());
        NOA_CHECK(input.is_empty() || (input.device() == lhs.device() && input.device() == rhs.device()),
                  "The input arrays should be on the same device, but got input:{}, lhs:{} and rhs:{}",
                  input.device(), lhs.device(), rhs.device());

        Extracted<Value, Offset> output;
        const Device device = lhs.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            stream.synchronize();
            auto extracted = cpu::memory::extract_binary<Value, Offset>(
                    input.get(), input.strides(),
                    lhs.get(), lhs.strides(),
                    rhs.get(), rhs.strides(), input.shape(),
                    std::forward<BinaryOp>(binary_op), extract_values, extract_offsets);
            output.values = Array<Value>(extracted.values, extracted.count);
            output.offsets = Array<Offset>(extracted.offsets, extracted.count);
        } else {
            #ifdef NOA_ENABLE_CUDA
            using input_t = std::remove_const_t<noa::traits::value_type_t<Input>>;
            using lhs_t = std::remove_const_t<noa::traits::value_type_t<Lhs>>;
            using rhs_t = std::remove_const_t<noa::traits::value_type_t<Rhs>>;
            if constexpr (cuda::memory::details::is_valid_extract_binary_v<
                    input_t, lhs_t, rhs_t, BinaryOp, Value, Offset>) {
                auto& cuda_stream = stream.cuda();
                auto extracted = cuda::memory::extract_binary<Value, Offset>(
                        input.get(), input.strides(),
                        lhs.get(), lhs.strides(),
                        rhs.get(), rhs.strides(), input.shape(),
                        binary_op, extract_values, extract_offsets, cuda_stream);
                cuda_stream.enqueue_attach(input.share(), lhs.share(), rhs.share());
                const auto option = ArrayOption(input.device(), Allocator::DEFAULT_ASYNC);
                output.values = Array<Value>(extracted.values, extracted.count, option);
                output.offsets = Array<Offset>(extracted.offsets, extracted.count, option);
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract_binary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        return output;
    }

    /// Extracts elements (and/or offsets) from the input array based on a binary bool operator.
    template<typename Value, typename Offset, typename Input, typename Lhs, typename Rhs, typename BinaryOp,
             typename = std::enable_if_t<
                     noa::traits::are_array_or_view_of_restricted_numeric_v<Input, Lhs> &&
                     noa::traits::is_numeric_v<Rhs>>>
    [[nodiscard]] Extracted<Value, Offset>
    extract_binary(const Input& input, const Lhs& lhs, Rhs rhs, BinaryOp&& binary_op,
                   bool extract_values = true, bool extract_offsets = true) {
        NOA_CHECK(!extract_values || !input.is_empty(), "The input array should not be empty");
        NOA_CHECK(input.is_empty() || noa::all(input.shape() == lhs.shape()),
                  "The input arrays should have the same shape, but got input:{} and lhs:{}",
                  input.shape(), lhs.shape());
        NOA_CHECK(input.is_empty() || input.device() == lhs.device(),
                  "The input arrays should be on the same device, but got input:{} and lhs:{}",
                  input.device(), lhs.device());

        Extracted<Value, Offset> output;
        const Device device = lhs.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            stream.synchronize();
            auto extracted = cpu::memory::extract_binary<Value, Offset>(
                    input.get(), input.strides(),
                    lhs.get(), lhs.strides(),
                    rhs, input.shape(),
                    std::forward<BinaryOp>(binary_op), extract_values, extract_offsets);
            output.values = Array<Value>(extracted.values, extracted.count);
            output.offsets = Array<Offset>(extracted.offsets, extracted.count);
        } else {
            #ifdef NOA_ENABLE_CUDA
            using input_t = std::remove_const_t<noa::traits::value_type_t<Input>>;
            using lhs_t = std::remove_const_t<noa::traits::value_type_t<Lhs>>;
            using rhs_t = std::remove_const_t<noa::traits::value_type_t<Rhs>>;
            if constexpr (cuda::memory::details::is_valid_extract_binary_v<
                    input_t, lhs_t, rhs_t, BinaryOp, Value, Offset>) {
                auto& cuda_stream = stream.cuda();
                auto extracted = cuda::memory::extract_binary<Value, Offset>(
                        input.get(), input.strides(),
                        lhs.get(), lhs.strides(),
                        rhs, input.shape(),
                        binary_op, extract_values, extract_offsets, cuda_stream);
                cuda_stream.enqueue_attach(input.share(), lhs.share());
                const auto option = ArrayOption(input.device(), Allocator::DEFAULT_ASYNC);
                output.values = Array<Value>(extracted.values, extracted.count, option);
                output.offsets = Array<Offset>(extracted.offsets, extracted.count, option);
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract_binary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        return output;
    }

    /// Extracts elements (and/or offsets) from the input array based on a binary bool operator.
    template<typename Value, typename Offset, typename Input, typename Lhs, typename Rhs, typename BinaryOp,
             typename = std::enable_if_t<
                     noa::traits::are_array_or_view_of_restricted_numeric_v<Input, Rhs> &&
                     noa::traits::is_numeric_v<Lhs>>>
    [[nodiscard]] Extracted<Value, Offset>
    extract_binary(const Input& input, Lhs lhs, const Rhs& rhs, BinaryOp&& binary_op,
                   bool extract_values = true, bool extract_offsets = true) {
        NOA_CHECK(!extract_values || !input.is_empty(), "The input array should not be empty");
        NOA_CHECK(input.is_empty() || noa::all(input.shape() == rhs.shape()),
                  "The input arrays should have the same shape, but got input:{} and rhs:{}",
                  input.shape(), rhs.shape());
        NOA_CHECK(input.is_empty() || input.device() == rhs.device(),
                  "The input arrays should be on the same device, but got input:{} and rhs:{}",
                  input.device(), rhs.device());

        Extracted<Value, Offset> output;
        const Device device = rhs.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            stream.synchronize();
            auto extracted = cpu::memory::extract_binary<Value, Offset>(
                    input.get(), input.strides(),
                    lhs,
                    rhs.get(), rhs.strides(), input.shape(),
                    std::forward<BinaryOp>(binary_op), extract_values, extract_offsets);
            output.values = Array<Value>(extracted.values, extracted.count);
            output.offsets = Array<Offset>(extracted.offsets, extracted.count);
        } else {
            #ifdef NOA_ENABLE_CUDA
            using input_t = std::remove_const_t<noa::traits::value_type_t<Input>>;
            using lhs_t = std::remove_const_t<noa::traits::value_type_t<Lhs>>;
            using rhs_t = std::remove_const_t<noa::traits::value_type_t<Rhs>>;
            if constexpr (cuda::memory::details::is_valid_extract_binary_v<
                    input_t, lhs_t, rhs_t, BinaryOp, Value, Offset>) {
                auto& cuda_stream = stream.cuda();
                auto extracted = cuda::memory::extract_binary<Value, Offset>(
                        input.get(), input.strides(),
                        lhs,
                        rhs.get(), rhs.strides(), input.shape(),
                        binary_op, extract_values, extract_offsets, cuda_stream);
                cuda_stream.enqueue_attach(input.share(), rhs.share());
                const auto option = ArrayOption(input.device(), Allocator::DEFAULT_ASYNC);
                output.values = Array<Value>(extracted.values, extracted.count, option);
                output.offsets = Array<Offset>(extracted.offsets, extracted.count, option);
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract_binary(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        return output;
    }

    /// Inserts elements into \p output.
    /// \param[in] values   Sequence of values that were extracted and need to be reinserted.
    /// \param[in] offsets  Linear offsets in \p output where the values should be inserted.
    /// \param[out] output  Output array inside which the values are going to be inserted.
    /// \note On the GPU:
    ///         - The values type should be one of i32, i64, u32, u64, f16, f32, f64.
    ///         - The offsets type should be one of i32, i64, u32, u64.
    template<typename Value, typename Offset, typename Output, typename = std::enable_if_t<
             noa::traits::are_array_or_view_of_restricted_numeric_v<Value, Offset, Output>>>
    void insert_elements(const Value& values, const Offset& offsets, const Output& output) {
        NOA_CHECK(noa::all(values.shape() == offsets.shape()) &&
                  noa::indexing::is_vector(offsets.shape()) &&
                  values.contiguous() && offsets.contiguous(),
                  "The sequence of values and offsets should be two contiguous vectors of the same size, "
                  "but got values:{} and offsets:{}", values.shape(), offsets.shape());

        const Device device = output.device();
        NOA_CHECK(values.device() == offsets.device() && values.device() == device,
                  "The input and output arrays should be on the same device, "
                  "but got values:{}, offsets:{} and output:{}",
                  values.device(), offsets.device(), device);

        const i64 elements = values.elements();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            stream.cpu().enqueue([=]() {
                cpu::memory::insert_elements(
                        values.get(), offsets.get(),
                        elements, output.share());
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            using output_t = std::remove_const_t<noa::traits::value_type_t<Output>>;
            if constexpr (cuda::memory::details::is_valid_insert_v<Value, Offset, output_t>) {
                auto& cuda_stream = stream.cuda();
                cuda::memory::insert_elements(
                        values.get(), offsets.get(),
                        elements, output.share(), cuda_stream);
                cuda_stream.enqueue_attach(values.share(), offsets.share(), output.share());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::insert_elements(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
