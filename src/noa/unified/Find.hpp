#pragma once

#include "noa/cpu/Find.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Find.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::details {
    template<typename ReduceOp, typename Value, typename Offset>
    constexpr bool is_valid_find_v =
            traits::is_any_v<Value, u32, u64, i32, i64, f16, f32, f64> &&
            traits::is_any_v<Offset, u32, u64, i32, i64> &&
            traits::is_any_v<ReduceOp, noa::first_min_t, noa::first_max_t, noa::last_min_t, noa::last_max_t>;
}

namespace noa {
    /// Returns the memory offset(s) of a particular kind of value(s).
    /// \tparam ReduceOp    Any of {first|last}_{min|max}_t.
    /// \tparam Input       Array or View of i32, i64, u32, u64, f16, f32, f64.
    /// \tparam Offset      Array or View of i32, i64, u32, u64.
    /// \param reduce_op    Search functor.
    /// \param[in] input    Input array.
    /// \param[out] offsets Contiguous vector where the memory offset(s) are saved.
    /// \param reduce_batch Whether the batch dimension should be reduced with the other DHW dimensions.
    ///                     If false, the offset of the result value in each batch is returned and these offsets
    ///                     are relative to the beginning of each batch. If true, a single value is returned
    ///                     in \p offsets.
    /// \param swap_layout  Whether the function is allowed to reorder the input for fastest search.
    ///                     If true, the search is done following the optimal strides order of \p input,
    ///                     so the first minimum value might be different if the array is e.g. column-major
    ///                     or row-major. Otherwise, the search is done in the BDHW order. Note that if
    ///                     \p reduce_batch is false, the batch dimension is cannot swapped.
    /// \note One can retrieve the multidimensional indexes from the offset using
    ///       \c noa::indexing::offset2indexes(offset,input).
    template<typename ReduceOp, typename Input, typename Offset,
             typename = std::enable_if_t<
                     noa::traits::are_array_or_view_v<Input, Offset> &&
                     details::is_valid_find_v<ReduceOp, noa::traits::value_type_t<Input>, noa::traits::value_type_t<Offset>>>>
    void find_offsets(ReduceOp reduce_op, const Input& input, const Offset& offsets,
                      bool reduce_batch = false, bool swap_layout = false) {
        NOA_CHECK(!input.is_empty() && !offsets.is_empty(), "Empty array detected");
        NOA_CHECK(is_safe_cast<noa::traits::value_type_t<Offset>>(
                  noa::indexing::at((input.shape() - 1).vec(), input.strides())),
                  "The input is too large (shape:{}) for the desired offset type {}",
                  input.shape(), noa::string::human<Offset>());

        [[maybe_unused]] const i64 required_size = reduce_batch ? 1 : input.shape()[0]; // TODO >=1 ?
        NOA_CHECK(noa::indexing::is_vector(offsets.shape()) &&
                  offsets.are_contiguous() &&
                  offsets.elements() == required_size,
                  "The output offsets should be specified as a contiguous vector of {} elements, "
                  "but got strides:{} and shape:{}", required_size, offsets.strides(), offsets.shape());

        const Device device = input.device();
        NOA_CHECK(device == offsets.device(),
                  "The input and output should be on the same device, but got input:{} and output:{}",
                  device, offsets.device());
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::find_offsets(reduce_op, input.get(), input.strides(), input.shape(),
                                  offsets.get(), reduce_batch, swap_layout, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::find_offsets(reduce_op, input.get(), input.strides(), input.shape(),
                               offsets.get(), reduce_batch, swap_layout, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), offsets.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the memory offset of a particular kind of value.
    /// \tparam ReduceOp    Any of {first|last}_{min|max}_t.
    /// \tparam Input       Array or View of i32, i64, u32, u64, f16, f32, f64.
    /// \param reduce_op    Search functor.
    /// \param[in] input    Input array.
    /// \param swap_layout  Whether the function is allowed to reorder the input for fastest search.
    ///                     If true, the search is done following the optimal strides order of \p input,
    ///                     so the first minimum value might be different if the array is e.g. column-major
    ///                     or row-major. Otherwise, the search is done in the BDHW order.
    template<typename ReduceOp, typename Input,
             typename = std::enable_if_t<
                     noa::traits::is_array_or_view_v<Input> &&
                     details::is_valid_find_v<ReduceOp, noa::traits::value_type_t<Input>, i64>>>
    [[nodiscard]] i64 find_offset(ReduceOp reduce_op, const Input& input, bool swap_layout = false) {
        NOA_CHECK(!input.is_empty(), "Empty array detected");

        const Device device = input.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.synchronize();
            return cpu::find_offset(
                    reduce_op, input.get(), input.strides(), input.shape(),
                    swap_layout, threads);
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            return cuda::find_offset(
                    reduce_op, input.get(), input.strides(), input.shape(),
                    swap_layout, cuda_stream);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
