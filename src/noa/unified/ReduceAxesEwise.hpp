#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/utils/Adaptor.hpp"
#include "noa/unified/Traits.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Indexing.hpp"
#include "noa/unified/Utilities.hpp"

#include "noa/cpu/ReduceAxesEwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/ReduceAxesEwise.cuh"
#endif

namespace noa::guts {
    template<bool, bool, bool, typename Inputs, typename Reduced, typename Outputs, typename Op>
    constexpr void reduce_axes_ewise(Inputs&&, Reduced&&, Outputs&&, Op&&);
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
    template<typename Inputs = guts::AdaptorUnzip<>,
             typename Reduced = guts::AdaptorUnzip<>,
             typename Outputs = guts::AdaptorUnzip<>,
             typename Operator>
    void reduce_axes_ewise(
            Inputs&& inputs,
            Reduced&& reduced,
            Outputs&& outputs,
            Operator&& op
    ) {
        if constexpr (guts::are_adaptor_v<Inputs, Reduced, Outputs>) {
            guts::reduce_axes_ewise<Inputs::ZIP, Reduced::ZIP, Outputs::ZIP>(
                    std::forward<Inputs>(inputs).tuple,
                    std::forward<Reduced>(reduced).tuple,
                    std::forward<Outputs>(outputs).tuple,
                    std::forward<Operator>(op));
        } else if constexpr (guts::are_adaptor_v<Inputs, Reduced>) {
            guts::reduce_axes_ewise<Inputs::ZIP, Reduced::ZIP, false>(
                    std::forward<Inputs>(inputs).tuple,
                    std::forward<Reduced>(reduced).tuple,
                    forward_as_tuple(std::forward<Outputs>(outputs)),
                    std::forward<Operator>(op));
        } else if constexpr (guts::are_adaptor_v<Inputs, Outputs>) {
            guts::reduce_axes_ewise<Inputs::ZIP, false, Outputs::ZIP>(
                    std::forward<Inputs>(inputs).tuple,
                    forward_as_tuple(std::forward<Reduced>(reduced)),
                    std::forward<Outputs>(outputs).tuple,
                    std::forward<Operator>(op));
        } else if constexpr (guts::are_adaptor_v<Reduced, Outputs>) {
            guts::reduce_axes_ewise<false, Reduced::ZIP, Outputs::ZIP>(
                    forward_as_tuple(std::forward<Inputs>(inputs)),
                    std::forward<Reduced>(reduced).tuple,
                    std::forward<Outputs>(outputs).tuple,
                    std::forward<Operator>(op));
        } else if constexpr (guts::is_adaptor_v<Outputs>) {
            guts::reduce_axes_ewise<false, false, Outputs::ZIP>(
                    forward_as_tuple(std::forward<Inputs>(inputs)),
                    forward_as_tuple(std::forward<Reduced>(reduced)),
                    std::forward<Outputs>(outputs).tuple,
                    std::forward<Operator>(op));
        } else if constexpr (guts::is_adaptor_v<Reduced>) {
            guts::reduce_axes_ewise<false, Reduced::ZIP, false>(
                    forward_as_tuple(std::forward<Inputs>(inputs)),
                    std::forward<Reduced>(reduced).tuple,
                    forward_as_tuple(std::forward<Outputs>(outputs)),
                    std::forward<Operator>(op));
        } else if constexpr (guts::is_adaptor_v<Inputs>) {
            guts::reduce_axes_ewise<Inputs::ZIP, false, false>(
                    std::forward<Inputs>(inputs).tuple,
                    forward_as_tuple(std::forward<Reduced>(reduced)),
                    forward_as_tuple(std::forward<Outputs>(outputs)),
                    std::forward<Operator>(op));
        } else {
            guts::reduce_axes_ewise<false, false, false>(
                    forward_as_tuple(std::forward<Inputs>(inputs)),
                    forward_as_tuple(std::forward<Reduced>(reduced)),
                    forward_as_tuple(std::forward<Outputs>(outputs)),
                    std::forward<Operator>(op));
        }
    }
}

namespace noa::guts {
    template<bool ZipInput, bool ZipReduced, bool ZipOutput,
             typename Inputs, typename Reduced, typename Outputs, typename Op>
    constexpr void reduce_axes_ewise(
            Inputs&& inputs,
            Reduced&& reduced,
            Outputs&& outputs,
            Op&& reduce_operator
    ) {
        constexpr i64 index_of_first_varray = guts::index_of_first_varray<Inputs>();
        static_assert(index_of_first_varray >= 0, "There should be at least one input varray");
        constexpr auto index = static_cast<size_t>(index_of_first_varray);

        static_assert(guts::are_all_varrays<Outputs>(), "All of the outputs should be varrays");
        static_assert(std::tuple_size_v<Outputs> > 0, "There should be at least one output");

        Tuple input_accessors = guts::to_tuple_of_accessors(std::forward<Inputs>(inputs));
        Tuple reduced_accessors = guts::to_tuple_of_accessors(std::forward<Reduced>(reduced));
        Tuple output_accessors = guts::to_tuple_of_accessors(std::forward<Outputs>(outputs));

        const auto& first_input_array = inputs[Tag<index>{}];
        auto input_shape = first_input_array.shape();
        auto output_shape = outputs[Tag<0>{}].shape();
        const auto device = outputs[Tag<0>{}].device();

        // If reduce to one element or one element per batch, and if the inputs have the same stride order,
        // then try to reorder the input(s) to the rightmost order. Don't reorder the batch axis if it is
        // not reduced.
        const Vec4<bool> reduced_dimensions = output_shape == 1 or input_shape != output_shape;
        bool do_reorder{};
        Vec4<i64> order;
        if (all(reduced_dimensions.pop_front())) {
            auto strides = first_input_array.strides();
            if (not reduced_dimensions[0])
                strides[0] = std::numeric_limits<i64>::max(); // keep batch to leftmost
            order = ni::order(strides, input_shape);
            do_reorder = any(order != Vec4<i64>{0, 1, 2, 3});
        }

        inputs.for_each_enumerate([&]<size_t I, typename T>(T& input) {
            if constexpr (nt::is_varray_v<T>) {
                check(device == input.device(),
                      "Input arrays should be on the same device as the output(s), but got output device={} and device:{}={}",
                      device, I, input.device());
            }
            if constexpr (I > index and nt::is_varray_v<T>) {
                check(all(input_shape == input.shape()),
                      "Input arrays should have the same shape, but got shape:0={} and shape:{}={}",
                      input_shape, I, input.shape());

                // Only reorder if all the inputs have the same order.
                if (do_reorder) {
                    auto strides = input.strides();
                    if (not reduced_dimensions[0])
                        strides[0] = std::numeric_limits<i64>::max(); // keep batch to leftmost
                    do_reorder = all(order == ni::order(strides, input_shape));
                }
            }
        });

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

        // No need to reorder the output shape, since we only reorder axes that are empty in the output.
        if (do_reorder) {
            input_shape = input_shape.reorder(order);
            guts::reorder_accessors(order, input_accessors);
        }

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            auto n_threads = cpu_stream.thread_limit();
            using config = noa::cpu::ReduceAxesEwiseConfig<ZipInput, ZipReduced, ZipOutput>;

            if (cpu_stream.is_sync()) {
                noa::cpu::reduce_axes_ewise<config>(
                        input_shape, output_shape,
                        std::forward<Op>(reduce_operator),
                        std::move(input_accessors),
                        std::move(reduced_accessors),
                        std::move(output_accessors),
                        n_threads);
            } else {
                cpu_stream.enqueue(
                        [=,
                         op = std::forward<Op>(reduce_operator),
                         ia = std::move(input_accessors),
                         ir = std::move(reduced_accessors),
                         oa = std::move(output_accessors),
                         ih = guts::extract_shared_handle_from_arrays(inputs),
                         oh = guts::extract_shared_handle_from_arrays(outputs)
                        ]() {
                            noa::cpu::reduce_axes_ewise<config>(
                                    input_shape, output_shape, std::move(op),
                                    std::move(ia), std::move(ir), std::move(oa), n_threads);
                        });
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = Stream::current(device).cuda();
            using config = noa::cuda::ReduceEwiseConfig<ZipInput, ZipReduced, ZipOutput>;
            noa::cuda::reduce_axes_ewise<config>(
                    input_shape, output_shape,
                    std::forward<Op>(reduce_operator),
                    std::move(input_accessors),
                    std::move(reduced_accessors),
                    std::move(output_accessors),
                    cuda_stream);

            // Enqueue the shared handles. See ewise() for more details.
            [&]<size_t... I, size_t... O>(std::index_sequence<I...>, std::index_sequence<O...>) {
                auto ih = guts::extract_shared_handle_from_arrays(inputs);
                auto oh = guts::extract_shared_handle_from_arrays(outputs);
                cuda_stream.enqueue_attach(std::move(ih)[Tag<I>{}]..., std::move(oh)[Tag<O>{}]...);
            }(nt::index_list_t<Inputs>{}, nt::index_list_t<Outputs>{});
            #else
            panic("No GPU backend detected");
            #endif
        }
    }
}
#endif
