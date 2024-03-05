#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/utils/Adaptor.hpp"
#include "noa/unified/Traits.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Indexing.hpp"
#include "noa/unified/Utilities.hpp"

#include "noa/cpu/ReduceAxesIwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/ReduceAxesIwise.cuh"
#endif

namespace noa::guts {
    template<bool, bool, typename Index, size_t N, typename Reduced, typename Outputs, typename Op>
    constexpr void reduce_axes_iwise(Shape<Index, N>, Reduced&&, Outputs&&, Op&&);
}

namespace noa {
    /// Computes an element-wise reduction along one or multiple axes.
    /// \details The size of the dimensions of the output array(s) should match the input shape, or be 1, indicating
    ///          the dimension should be reduced. Currently, reducing more than one axis at a time is only supported
    ///          if the reduction results to having one value or one value per batch, i.e. if the DHW dimensions are
    ///          empty after reduction. As opposed to reduce_ewise, this function is asynchronous and does not perform
    ///          any synchronization. If all axes are reduced, it is otherwise equivalent to reduce_ewise.
    ///
    /// \param[in,out] inputs   Input varray or an adaptor containing at least one varray.
    ///                         If multiple varrays are entered, there should have the same shape.
    ///                         Input arrays should be on the same device as the output arrays.
    /// \param[in] reduced      Initial reduced value, or an adaptor containing the initial reduced value(s).
    /// \param[in,out] outputs  Output array, or an adaptor containing the output array(s). While the input adaptor
    ///                         can contain non-varray types, the output should only contain varrays, all of which
    ///                         should be on the same device.
    /// \param[in] op           Operator satisfying the reduce_(axes_)ewise core interface. The operator is perfectly
    ///                         forwarded to the backend (it is moved or copied to the backend compute kernel).
    ///                         Each compute (CPU or GPU) thread holds a copy of the operator.
    template<typename Index, size_t N,
             typename Reduced = guts::AdaptorUnzip<>,
             typename Outputs = guts::AdaptorUnzip<>,
             typename Operator>
    void reduce_axes_iwise(
            const Shape<Index, N>& shape,
            Reduced&& reduced,
            Outputs&& outputs,
            Operator&& op
    ) {

    }
}

namespace noa::guts {
    template<bool ZipReduced, bool ZipOutput,
             typename Index, size_t N, typename Reduced, typename Outputs, typename Op>
    constexpr void reduce_axes_iwise(
            const Shape<Index, N>& input_shape,
            Reduced&& reduced,
            Outputs&& outputs,
            Op&& reduce_operator
    ) {
        static_assert(guts::are_all_varrays<Outputs>(), "All of the outputs should be varrays");
        static_assert(std::tuple_size_v<Outputs> > 0, "There should be at least one output");

        const auto output_shape = outputs[Tag<0>{}].shape();
        const auto device = outputs[Tag<0>{}].device();
        outputs.for_each_enumerate([&]<size_t I, typename T>(T& output) {
            if constexpr (I > 0) {
                check(device == output.device(),
                      "Output arrays should be on the same device, but got device:0={} and device:{}={}",
                      device, I, output.device());
                check(all(output_shape == output.shape()),
                      "Output arrays should have the same shape, but got shape:0={} and shape:{}={}",
                      output_shape, I, output.shape());
            }
        });

        Tuple reduced_accessors = guts::to_tuple_of_accessors(std::forward<Reduced>(reduced));
        Tuple output_accessors = guts::to_tuple_of_accessors(std::forward<Outputs>(outputs));

        // Backends expect the output accessors to have N dimensions, i.e. we need to remove the leftmost axes.
        constexpr auto filter_nd = []() -> Vec<i64, N> {
            if constexpr (N == 1)
                return {0};
            else if constexpr (N == 2)
                return {0, 1};
            else if constexpr (N == 3)
                return {0, 1, 2};
            else if constexpr (N == 4)
                return {0, 1, 2, 3};
        };
        Tuple output_accessors_nd = ng::reconfig_accessors<ng::AccessorConfig<N>{.filter=filter_nd}>(output_accessors);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            auto n_threads = cpu_stream.thread_limit();
            using config = noa::cpu::ReduceAxesIwiseConfig<ZipReduced, ZipOutput>;

            if (cpu_stream.is_sync()) {
                noa::cpu::reduce_axes_iwise<config>(
                        input_shape, output_shape,
                        std::forward<Op>(reduce_operator),
                        std::move(reduced_accessors),
                        std::move(output_accessors_nd),
                        n_threads);
            } else {
                cpu_stream.enqueue(
                        [=,
                                op = std::forward<Op>(reduce_operator),
                                ir = std::move(reduced_accessors),
                                oa = std::move(output_accessors_nd),
                                oh = guts::extract_shared_handle_from_arrays(outputs)
                        ]() {
                            noa::cpu::reduce_axes_iwise<config>(
                                    input_shape, output_shape, std::move(op),
                                    std::move(ir), std::move(oa), n_threads);
                        });
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = Stream::current(device).cuda();
            using config = noa::cuda::ReduceIwiseConfig<ZipReduced, ZipOutput>;
            noa::cuda::reduce_axes_iwise<config>(
                    input_shape, output_shape,
                    std::forward<Op>(reduce_operator),
                    std::move(reduced_accessors),
                    std::move(output_accessors),
                    cuda_stream);

            // Enqueue the shared handles. See ewise() for more details.
            std::forward<Outputs>(outputs).apply([]<typename... T>(T&&... varrays) {
                cuda_stream.enqueue_attach(std::forward<T>(varrays)...);
            });
            #else
            panic("No GPU backend detected");
            #endif
        }
    }
}
#endif
