#include "noa/unified/Array.hpp"
#include "noa/unified/memory/Subregion.hpp"

namespace noa::memory {
    /// (Deep-)Copies batches across arrays.
    /// \param[in] input            Input array to copy.
    /// \param[out] output          Output array.
    /// \param[in] batch_indexes    Contiguous vector with the input batch index(es) to copy into \p output.
    /// \param group_copy_at_count  If the batches are not consecutive, this function will either do a per-batch
    ///                             copy or a grouped copy using memory::extract_subregions. This parameter sets
    ///                             the threshold when to use a grouped copy. This has no effect if the batches
    ///                             are consecutive, or not on the same device, or for types that don't support
    ///                             memory::extract_subregions.
    template<typename Input, typename Output, typename Indexes, typename = std::enable_if_t<
             noa::traits::are_array_or_view_v<Input, Output> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             noa::traits::is_array_or_view_of_almost_any_v<Indexes, i32, i64>>>
    void copy_batches(const Input& input, const Output& output,
                      const Indexes& batch_indexes,
                      i64 group_copy_at_count = 2) {

        NOA_CHECK(!input.is_empty() && !output.is_empty() && !batch_indexes.is_empty(), "Empty array detected");
        NOA_CHECK(noa::all(input.shape().pop_front() == output.shape().pop_front() || input.shape().pop_front() == 1),
                  "Cannot copy batches of shape {} into batches of shape {}",
                  input.shape().pop_front(), output.shape().pop_front());
        NOA_CHECK(input.device() == output.device(),
                  "The input and output should be on the same device, but got input={} and output={}",
                  input.device(), output.device());
        NOA_CHECK(noa::indexing::is_contiguous_vector(batch_indexes) && output.shape()[0] == batch_indexes.elements(),
                  "The indexes should be specified as a contiguous vector of size {}, but got shape={} and strides={}",
                  output.shape()[0], batch_indexes.shape(), batch_indexes.strides());
        NOA_CHECK(batch_indexes.device().is_cpu(),
                  "The indexes should be on the CPU, got device={}",
                  batch_indexes.device());

        const auto batches_to_copy = output.shape()[0];
        const auto batch_indexes_1d = batch_indexes.accessor_contiguous_1d();

        using index_t = typename Indexes::value_type;
        index_t index{};
        for (i64 i = 0; i < batches_to_copy; ++i, ++index) {
            const auto current_index = batch_indexes_1d[i];
            NOA_CHECK(current_index >= 0 && current_index < input.shape()[0],
                      "At least one input batch index is not valid: i={}, 0 <= {} < {}",
                      i, current_index, input.shape()[0]);

            // Check if range with step 1.
            if (i == 0) {
                index = current_index;
            } else {
                if (index != current_index)
                    index = -2;
            }
        }

        // If the batches to copy into output are next to each other,
        // this becomes a slice operation. So try to identify this case:
        if (index > 0) {
            NOA_ASSERT(batch_indexes_1d[0] + batches_to_copy == index);
            const auto slice = noa::indexing::slice_t{batch_indexes_1d[0], index};
            return input.subregion(slice).to(output);
        }

        if constexpr (noa::traits::is_array_or_view_of_restricted_numeric_v<Input>) {
            // If the arrays are on the same device, use extract for
            // better performance (only one iwise call).
            // FIXME Benchmark to check whether this is faster than the simple
            //       one by one copy for small number of batches.
            const auto device = output.device();
            if (input.device() == device && batches_to_copy > group_copy_at_count) {
                Array<Vec4<i32>> batch_origins(batches_to_copy);
                for (size_t i = 0; i < batches_to_copy; ++i)
                    batch_origins(0, 0, 0, i) = {safe_cast<i32>(batch_indexes_1d[i]), 0, 0, 0};
                if (device.is_gpu())
                    batch_origins = batch_origins.to(ArrayOption(device, Allocator::DEFAULT_ASYNC));
                return extract_subregions(input, output, batch_origins, BorderMode::NOTHING);
            }
        }

        // Worst case, copy batches one by one across devices.
        for (i64 i = 0; i < batches_to_copy; ++i)
            input.subregion(batch_indexes_1d[i]).to(output.subregion(i));
    }
}
