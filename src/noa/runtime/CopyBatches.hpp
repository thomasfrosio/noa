#pragma once

#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"

namespace noa::details {
    template<
        usize N, usize BATCH_DIM,
        nt::readable_nd<N> Input,
        nt::writable_nd<N> Output,
        nt::readable_nd<1> Indices>
    struct CopyBatches {
        using input_type = std::remove_const_t<Input>;
        using output_type = std::remove_const_t<Output>;
        using indices_type = std::remove_const_t<Indices>;

        input_type input;
        output_type output;
        indices_type batch_indices;

        template<nt::integer I>
        NOA_HD constexpr void operator()(const Vec<I, N>& output_indices) const {
            auto input_indices = output_indices;
            input_indices[BATCH_DIM] = batch_indices[BATCH_DIM];
            output(output_indices) = input(input_indices);
        }
    };
}

namespace noa {
    /// Deep-copies batches of an array.
    /// \tparam BATCH_DIM
    ///     Index of the "batch" axis.
    /// \param[in] input:
    ///     (...n) Input array to copy.
    ///     All axes except the "batch" axis should match the output or be 1 (broadcast).
    /// \param[out] output:
    ///     (...n) Output array.
    /// \param[in] batch_indices
    ///     Contiguous vector with the input batch indices to copy into output.
    ///     If the input and output are on the same device, it is faster for the indices to be on the output device too.
    ///     Otherwise, it is faster for the indices to be on the CPU.
    /// \safety The batch indices are not checked, so the caller should make sure they are pointing within the input.
    template<usize BATCH_DIM = 0,
             nt::readable_array_decay Input,
             nt::writable_array_decay Output,
             nt::readable_array_decay_of_sinteger Indices>
        requires (nt::array_decay_of_almost_same_type<Input, Output> and
                  nt::array_decay_with_same_nd<Input, Output> and
                  nt::array_decay_nd<Indices, 1> and
                  nt::array_size_v<Input> > BATCH_DIM and
                  nt::array_size_v<Input> >= 2)
    void copy_batches(
        Input&& input,
        Output&& output,
        Indices&& batch_indices
    ) {
        check(nd::are_arrays_valid(input, output, batch_indices), "Empty array detected");
        check(batch_indices.is_contiguous() and output.shape()[BATCH_DIM] == batch_indices.n_elements(),
              "The indices should be specified as a contiguous vector of size {}, but got shape={} and strides={}",
              output.shape()[BATCH_DIM], batch_indices.shape(), batch_indices.strides());

        // Broadcast axes except at BATCH_DIM.
        // TODO reorder to rightmost except for BATCH_DIM
        const auto input_batch_shape = input.shape().set<BATCH_DIM>(1);
        const auto output_batch_shape = output.shape().set<BATCH_DIM>(1);
        auto input_strides = input.strides();
        check(noa::broadcast(input_batch_shape, input_strides, output_batch_shape),
              "Cannot broadcast shape={} into a shape={}", input_batch_shape, output_batch_shape);

        constexpr usize N = nt::array_size_v<Input>;
        const auto n = output.shape()[BATCH_DIM];
        if (n == 0)
            return;

        // If device is unchanged, a simple iwise is best.
        if (input.device() == output.device()) {
            auto buffer = Array<nt::value_type_t<Indices>, 1>{};
            auto buffer_ptr = batch_indices.data();
            if (batch_indices.device() != output.device()) {
                buffer = std::forward<Indices>(batch_indices)
                    .to(ArrayOption{.device = output.device(), .allocator = Allocator::ASYNC});
                buffer_ptr = buffer.data();
            }
            using input_t = AccessorRestrict<nt::const_value_type_t<Input>, N, isize>;
            using output_t = AccessorRestrict<nt::value_type_t<Output>, N, isize>;
            using indices_t = AccessorRestrictContiguous<nt::const_value_type_t<Indices>, 1, isize>;
            using op_t = nd::CopyBatches<N, BATCH_DIM, input_t, output_t, indices_t>;
            auto input_ = input_t(input.data(), input_strides);
            auto output_ = output_t(output.data(), output.strides());
            auto batch_indices_ = indices_t(buffer_ptr);
            noa::iwise(output.shape(), output.device(), op_t{
                .input = input_,
                .output = output_,
                .batch_indices = batch_indices_,
            }, std::forward<Input>(input), std::forward<Output>(output), std::move(buffer));
            return;
        }

        // Make sure the indices are ready and dereferenceable.
        auto buffer = Array<nt::value_type_t<Indices>, 1>{};
        if (not batch_indices.device().is_cpu())
            buffer = std::forward<Indices>(batch_indices).to(ArrayOption{.device = Device{}});
        else
            buffer = std::forward<Indices>(batch_indices);
        const auto batch_indices_1d = buffer.eval().span_1d();

        // If the batches to copy to the output are next to each other, this becomes a slice operation.
        // So try to identify this case by checking if the batch indices are describing a sequence with step 1.
        using index_t = nt::mutable_value_type_t<Indices>;
        bool is_sequence{true};
        index_t index = batch_indices_1d[0];
        for (isize i{1}; i < n; ++i) {
            if (index != batch_indices_1d[i]) {
                is_sequence = false;
                break;
            }
            ++index;
        }
        if (is_sequence) {
            std::forward<Input>(input)
                .subregion(Slice{batch_indices_1d[0], index})
                .to(std::forward<Output>(output));
            return;
        }

        // Worst case scenario, copy batches one by one across devices.
        for (isize i{}; i < n - 1; ++i)
            input.view().subregion(batch_indices_1d[i]).to(output.view().subregion(i));
        std::forward<Input>(input)
            .subregion(batch_indices_1d[n - 1])
            .to(std::forward<Output>(output).subregion(n - 1));
    }
}
