#pragma once

#include "noa/unified/Array.hpp"
#include "noa/unified/Subregion.hpp"

namespace noa {
    /// (Deep-)Copies batches across arrays.
    /// \param[in] input            Input array to copy.
    /// \param[out] output          Output array.
    /// \param[in] batch_indices    Contiguous vector with the input batch indices to copy into \p output.
    /// \param group_copy_at_count  If the batches are not consecutive, this function will either do a per-batch
    ///                             copy or a grouped copy using extract_subregions(). This parameter sets
    ///                             the threshold when to use a grouped copy. This has no effect if the batches
    ///                             are consecutive, or not on the same device.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay_of_almost_any<i32, i64> Indices>
    requires nt::varray_decay_of_almost_same_type<Input, Output>
    void copy_batches(
        Input&& input,
        Output&& output,
        Indices&& batch_indices,
        i64 group_copy_at_count = 3
    ) {
        check(not input.is_empty() and not output.is_empty() and not batch_indices.is_empty(), "Empty array detected");
        check(all(input.shape().pop_front() == output.shape().pop_front() or input.shape().pop_front() == 1),
              "Cannot copy batches of shape {} into batches of shape {}",
              input.shape().pop_front(), output.shape().pop_front());
        check(input.device() == output.device(),
              "The input and output should be on the same device, but got input={} and output={}",
              input.device(), output.device());
        check(ni::is_contiguous_vector(batch_indices) and output.shape()[0] == batch_indices.n_elements(),
              "The indices should be specified as a contiguous vector of size {}, but got shape={} and strides={}",
              output.shape()[0], batch_indices.shape(), batch_indices.strides());
        check(batch_indices.device().is_cpu(),
              "The indices should be on the CPU, got device={}",
              batch_indices.device());

        const auto n_batches_to_copy = output.shape()[0];
        const auto batch_indices_1d = batch_indices.span_1d_contiguous();

        // If the batches to copy to the output are next to each other,
        // this becomes a slice operation. So try to identify this case:
        using index_t = nt::value_type_t<Indices>;
        index_t index{};
        for (i64 i{}; i < n_batches_to_copy; ++i, ++index) {
            const auto current_index = batch_indices_1d[i];
            check(current_index >= 0 and current_index < input.shape()[0],
                  "At least one input batch index is out of bound: 0 <= indices[{}]={} < input:batch={} is not true",
                  i, current_index, input.shape()[0]);

            // Check if it's a sequence with step 1.
            if (i == 0) {
                index = current_index;
            } else {
                if (index != current_index)
                    index = -2;
            }
        }
        if (index > 0) {
            NOA_ASSERT(batch_indices_1d[0] + n_batches_to_copy == index);
            std::forward<Input>(input)
                .subregion(ni::Slice{batch_indices_1d[0], index})
                .to(std::forward<Output>(output));
            return;
        }

        // Otherwise, if the arrays are on the same device, we can use extract_subregions (only one iwise call).
        // FIXME Benchmark to check whether this is faster than the simple
        //       one by one copy for small number of batches.
        const auto device = output.device();
        if (input.device() == device and n_batches_to_copy >= group_copy_at_count) {
            Array<Vec4<i32>> batch_origins(n_batches_to_copy);
            for (size_t i{}; i < n_batches_to_copy; ++i)
                batch_origins(0, 0, 0, i) = {safe_cast<i32>(batch_indices_1d[i]), 0, 0, 0};
            if (device.is_gpu())
                batch_origins = batch_origins.to(ArrayOption{device, Allocator::DEFAULT_ASYNC});
            return extract_subregions(
                std::forward<Input>(input), std::forward<Output>(output), batch_origins, Border::NOTHING);
        }

        // Worst case scenario, copy batches one by one across devices.
        for (i64 i{}; i < n_batches_to_copy - 1; ++i)
            input.view().subregion(batch_indices_1d[i]).to(output.view().subregion(i));
        std::forward<Input>(input)
            .subregion(batch_indices_1d[n_batches_to_copy - 1])
            .to(std::forward<Output>(output).subregion(n_batches_to_copy - 1));
    }
}
