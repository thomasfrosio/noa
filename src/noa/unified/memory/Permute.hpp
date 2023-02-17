#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/utils/Indexing.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Handle.hpp"

#include "noa/cpu/memory/Permute.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Permute.hpp"
#endif

namespace noa::memory {
    /// Permutes the input by reordering its dimensions. The returned object points to the same data.
    template<typename Input, typename = std::enable_if_t<noa::traits::is_array_or_view_v<Input>>>
    Input permute(const Input& input, const Vec4<i64>& permutation) {
        const auto permuted_shape = noa::indexing::reorder(input.shape(), permutation);
        const auto permuted_strides = noa::indexing::reorder(input.strides(), permutation);
        return Input(input.share(), permuted_shape, permuted_strides, input.options());
    }

    /// Permutes, in memory, the axes of an array.
    /// \tparam T           Any numeric type.
    /// \param[in] input    Array to permute.
    /// \param[out] output  Permuted array. Its shape and strides should be permuted already.
    /// \param permutation  Permutation. Axes are numbered from 0 to 3.
    /// \note For in-place permutations, only 0123, 0213, 0132, and 0321 are supported. Anything else throws an error.
    /// \note The in-place 0213 permutation requires the axis 1 and 2 to have the same size.
    ///       The in-place 0132 permutation requires the axis 3 and 2 to have the same size.
    ///       The in-place 0321 permutation requires the axis 3 and 1 to have the same size.
    /// \note On the GPU, the following permutations are optimized: 0123, 0132, 0312, 0321, 0213, 0231.
    ///       Anything else calls copy(), which is slower.
    template<typename Input, typename Output,
             typename = std::enable_if_t<noa::traits::are_array_or_view_of_restricted_numeric_v<Input, Output> &&
                                         noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void permute_copy(const Input& input, const Output& output, const Vec4<i64>& permutation) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");

        // To enable broadcasting, we need to permute the input.
        auto input_strides = input.strides();
        auto input_shape = input.shape();
        for (i64 i = 0; i < 4; ++i) {
            const i64 d = permutation[i];
            if (input.shape()[d] == 1 && output.shape()[i] != 1) {
                input_strides[d] = 0; // broadcast this dimension
                input_shape[d] = output.shape()[i];
            } else if (input.shape()[d] != output.shape()[i]) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          noa::indexing::reorder(input.shape(), permutation), output.shape());
            }
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            noa::cpu::Stream& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=](){
                noa::cpu::memory::permute(
                        input.get(), input.strides(), input.shape(),
                        output.get(), output.strides(),
                        permutation, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            noa::cuda::Stream& cuda_stream = stream.cuda();
            noa::cuda::memory::permute(
                    input.get(), input.strides(), input.shape(),
                    output.get(), output.strides(),
                    permutation, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Permutes the input by performing a deep-copy. The returned Array is a new C-contiguous array.
    /// \param[in] input    Array or View to permute.
    /// \param permutation  Permutation with the axes numbered from 0 to 3.
    template<typename Input, typename = std::enable_if_t<noa::traits::is_array_or_view_of_numeric_v<Input>>>
    auto permute_copy(const Input& input, const Vec4<i64>& permutation) {
        using mutable_value_type = std::remove_const_t<noa::traits::value_type_t<Input>>;
        const auto permuted_shape = noa::indexing::reorder(input.shape(), permutation);
        auto output = Array<mutable_value_type>(permuted_shape, input.options());
        permute_copy(input, output, permutation);
        return output;
    }
}
