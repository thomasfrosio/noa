#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/utils/Adaptor.hpp"
#include "noa/unified/Traits.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Indexing.hpp"
#include "noa/unified/Utilities.hpp"

#include "noa/cpu/Ewise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Ewise.cuh"
#endif

namespace noa::guts {
    template<bool, bool, typename Inputs, typename Outputs, typename EwiseOp>
    void ewise(Inputs&&, Outputs&&, EwiseOp&&);
}

namespace noa {
    /// Generic element-wise transformation.
    /// \param[in] inputs       Input varray(s) and/or values to transform.
    /// \param[out] outputs     Output varray(s).
    /// \param ewise_operator   Operator satisfying the ewise core interface.
    ///                         The operator is perfectly forwarded to the backend, but note than more than
    ///                         one move can happen by the time the operator reaches the compute kernel.
    ///
    /// \note The advantage of this function compared to iwise is that it can analyse the inputs and outputs to deduce
    ///       the most efficient way to traverse these arrays, e.g. by reordering the dimensions, collapsing contiguous
    ///       dimensions together (up to 1d), and can trigger the vectorization for the 1d case by checking for data
    ///       contiguity and aliasing.
    ///       Note that because the core interface allows the operator to modify the inputs and allows the inputs/outputs
    ///       to alias, GPU vectorization can only be triggered if the input varrays have a const value type and if the
    ///       inputs/outputs don't alias. Note that the core interface also allows the output values to be read/updated,
    ///       which greatly simplifies some operations.
    ///
    /// \note Views or Arrays (i.e. varrays) are supported and handled separately from other types. Varrays are
    ///       parsed, wrapped into Accessors and sent to the backends' ewise core function. For asynchronous cases
    ///       (async CPU stream or GPU), the shared handle of every Array (see Array::share()) is copied and will
    ///       be released (i.e. destructed) when the core function finishes, thus ensuring that the Array's resource
    ///       stays alive during the processing. Other types (so everything that is not a varray) are moved/copied
    ///       into a wrapper (AccessorValue) which will pass a (const) lvalue reference of the moved/copied value
    ///       to the operator. No other actions are performed on these types, so it is the responsibility of the
    ///       caller to make sure these stay valid until completion.
    ///
    /// \note \p inputs and \p outputs can be left empty, but note that there should be at least one varray
    ///       (in either \p inputs and \p outputs), otherwise the function will not compile.
    ///       If no outputs are provided, all varray inputs should have the same shape. If they do have the same
    ///       stride layout, the function will reorder them to the rightmost layout for better performance.
    ///       If outputs are provided, there should all be varrays with the same shape, have the same stride order
    ///       and cannot be broadcasted (strides of zeros are not allowed). In this case, the varray inputs will
    ///       need to match the output shape or be broadcastable to the output shape. If the stride order of the
    ///       outputs is not the rightmost order, output varrays are reordered to the rightmost order to maximise
    ///       performance. Of course, this means that input varrays are reordered as well for correctness.
    ///
    /// \note To be supported by the CUDA JIT backend, the source of \p ewise_operator needs to be added to the sources
    ///       available to the JIT compiler. This can be set up as part of the build, but source code can also
    ///       be added at runtime using the noa::Session::add_cuda_sources(...) functions. TODO
    template<typename Input = guts::AdaptorUnzip<>,
             typename Output = guts::AdaptorUnzip<>,
             typename EwiseOperator> // TODO EwiseChecker
    void ewise(Input&& inputs, Output&& outputs, EwiseOperator&& ewise_operator) {
        if constexpr (guts::are_adaptor_v<Input, Output>) {
            guts::ewise<Input::ZIP, Output::ZIP>(
                    std::forward<Input>(inputs).tuple,
                    std::forward<Output>(outputs).tuple,
                    std::forward<EwiseOperator>(ewise_operator));
        } else if constexpr (guts::is_adaptor_v<Input>) {
            guts::ewise<Input::ZIP, false>(
                    std::forward<Input>(inputs).tuple,
                    forward_as_tuple(std::forward<Output>(outputs)), // wrap
                    std::forward<EwiseOperator>(ewise_operator));
        } else if constexpr (guts::is_adaptor_v<Output>) {
            guts::ewise<false, Output::ZIP>(
                    forward_as_tuple(std::forward<Input>(inputs)), // wrap
                    std::forward<Output>(outputs).tuple,
                    std::forward<EwiseOperator>(ewise_operator));
        } else {
            guts::ewise<false, false>(
                    forward_as_tuple(std::forward<Input>(inputs)), // wrap
                    forward_as_tuple(std::forward<Output>(outputs)), // wrap
                    std::forward<EwiseOperator>(ewise_operator));
        }
    }
}

namespace noa::guts {
    template<bool ZipInput, bool ZipOutput, typename Inputs, typename Outputs, typename EwiseOp>
    void ewise(Inputs&& inputs, Outputs&& outputs, EwiseOp&& ewise_op) {
        constexpr size_t N_INPUTS = std::tuple_size_v<std::decay_t<Inputs>>;
        constexpr size_t N_OUTPUTS = std::tuple_size_v<std::decay_t<Outputs>>;
        if constexpr (N_INPUTS == 0 and N_OUTPUTS == 0) {
            return; // valid, do nothing
        } else {
            // While these are forwarded, varrays are actually never moved, it simply returns the .accessor().
            // For anything other than varrays, it can be moved thus left in an unspecified state,
            // i.e. we shouldn't read from these elements again.
            Tuple input_accessors = guts::to_tuple_of_accessors(std::forward<Inputs>(inputs));
            Tuple output_accessors = guts::to_tuple_of_accessors(std::forward<Outputs>(outputs));

            Shape4<i64> shape;
            Device device;
            Vec4<i64> order;
            bool do_reorder{};

            if constexpr (N_OUTPUTS >= 1) {
                if constexpr (guts::are_all_varrays<Outputs>()) {
                    const auto& first_output = outputs[Tag<0>{}];
                    shape = first_output.shape();
                    device = first_output.device();
                    order = ni::order(first_output.strides(), shape);
                    do_reorder = any(order != Vec4<i64>{0, 1, 2, 3});

                    outputs.for_each_enumerate([&]<size_t I, typename T>(const T& output) {
                        check(not output.is_empty(), "Empty output array detected (index={})", I);
                        check(all(output.strides() > 0),
                              "Output arrays cannot be broadcasted, i.e. strides should not be 0, but got strides:{}={}",
                              I, output.strides());
                        if constexpr (I > 0) {
                            check(device == output.device(),
                                  "Output arrays should be on the same device, but got device:0={} and device:{}={}",
                                  device, I, output.device());
                            check(all(shape == output.shape()),
                                  "Output arrays should have the same shape, but got shape:0={} and shape:{}={}",
                                  shape, I, output.shape());
                            check(all(order == ni::order(output.strides(), shape)),
                                  "Output arrays should have the same stride order, but got strides:0={} and strides:{}={}",
                                  first_output.strides(), I, output.strides());
                        }
                    });

                    // Automatic broadcasting of the inputs.
                    // "inputs" is used after forward, but as mentioned above "to_tuple_of_accessors"
                    // doesn't actually move varrays and here we only read varrays.
                    input_accessors.for_each_enumerate([&inputs, &shape]<size_t I, typename T>(T& accessor) {
                        if constexpr (nt::is_varray_v<decltype(inputs[Tag<I>{}])>) {
                            static_assert(nt::is_accessor_pure_v<T>);
                            const auto& input_shape = inputs[Tag<I>{}].shape();
                            if (not ni::broadcast(input_shape, accessor.strides(), shape)) {
                                panic("Cannot broadcast an array of shape {} into an array of shape {}",
                                      input_shape, shape);
                            }
                        }
                    });
                } else {
                    static_assert(nt::always_false_v<Outputs>, "The outputs should be varrays");
                }
            } else { // N_INPUTS >= 1
                constexpr i64 index_of_first_varray = guts::index_of_first_varray<Inputs>();
                if constexpr (index_of_first_varray >= 0) {
                    constexpr auto index = static_cast<size_t>(index_of_first_varray);
                    const auto& first_input_array = inputs[Tag<index>{}];
                    shape = first_input_array.shape();
                    device = first_input_array.device();
                    order = ni::order(first_input_array.strides(), shape);
                    do_reorder = any(order != Vec4<i64>{0, 1, 2, 3});

                    inputs.for_each_enumerate([&]<size_t I, typename T>(T& input) {
                        if constexpr (nt::is_varray_v<T>)
                            check(not input.is_empty(), "Empty input array detected (index={})", I);
                        if constexpr (I > index and nt::is_varray_v<T>) {
                            check(device == input.device(),
                                  "Input arrays should be on the same device, but got device:0={} and device:{}={}",
                                  device, I, input.device());
                            check(all(shape == input.shape()),
                                  "Input arrays should have the same shape, but got shape:0={} and shape:{}={}",
                                  shape, I, input.shape());

                            // Only reorder if all the inputs have the same order.
                            if (do_reorder)
                                do_reorder = all(order == ni::order(input.strides(), shape));
                            // TODO Forcing the same order is okay, but may be a bit too restrictive since it effectively
                            //      prevents automatic broadcasting (the caller can still explicitly broadcast though).
                            //      We may instead find the input with the largest effective shape and use it as
                            //      as reference for reordering the inputs?
                        }
                    });
                } else {
                    static_assert(nt::always_false_v<Inputs>,
                                  "For cases with inputs but without outputs, there should be at least one input varray");
                }
            }

            if (do_reorder) {
                shape = shape.reorder(order);
                guts::reorder_accessors(order, input_accessors, output_accessors);
            }

            Stream& stream = Stream::current(device);
            if (device.is_cpu()) {
                auto& cpu_stream = stream.cpu();
                auto n_threads = cpu_stream.thread_limit();
                using config = noa::cpu::EwiseConfig<ZipInput, ZipOutput>;

                if (cpu_stream.is_sync()) {
                    noa::cpu::ewise<config>(
                            shape, std::forward<EwiseOp>(ewise_op),
                            std::move(input_accessors),
                            std::move(output_accessors),
                            n_threads);
                } else {
                    cpu_stream.enqueue(
                            [=,
                             op = std::forward<EwiseOp>(ewise_op),
                             ia = std::move(input_accessors),
                             oa = std::move(output_accessors),
                             ih = guts::extract_shared_handle_from_arrays(inputs),
                             oh = guts::extract_shared_handle_from_arrays(outputs)
                            ]() {
                                noa::cpu::ewise<config>(shape, std::move(op), std::move(ia), std::move(oa), n_threads);
                            });
                }
            } else {
                #ifdef NOA_ENABLE_CUDA
                auto& cuda_stream = Stream::current(device).cuda();
                using config = noa::cuda::EwiseConfig<ZipInput, ZipOutput>;
                noa::cuda::ewise<config>(
                        shape, std::forward<EwiseOp>(ewise_op),
                        std::move(input_accessors),
                        std::move(output_accessors),
                        cuda_stream);

                // Enqueue the shared handles. Doing it using a single call to enqueue_attach is slightly more efficient.
                [&]<size_t... I, size_t... O>(std::index_sequence<I...>, std::index_sequence<O...>) {
                    // "enqueue_attach" saves shared_ptr types and anything with a .share() that returns a shared_ptr,
                    // and ignores everything else. As such, we could directly pass the values of "inputs" and "outputs",
                    // but here we explicitly only want to save the shared_ptr from arrays.
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
}
#endif
