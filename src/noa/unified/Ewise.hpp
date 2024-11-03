#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/utils/Adaptor.hpp"
#include "noa/unified/Traits.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Indexing.hpp"
#include "noa/unified/Utilities.hpp"

#include "noa/cpu/Ewise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Ewise.cuh"
#endif

namespace noa {
    struct EwiseOptions {
        /// Whether to compile for the CPU compute device.
        bool generate_cpu{true};

        /// Whether to compile for the GPU compute device.
        bool generate_gpu{true};

        /// GPU kernel configuration.
        u32 gpu_block_size{128};
        u32 gpu_n_elements_per_thread{4};
        bool gpu_vectorize{true};
    };
}

namespace noa::guts {
    template<EwiseOptions, bool, bool, typename Inputs, typename Outputs, typename EwiseOp>
    void ewise(Inputs&&, Outputs&&, EwiseOp&&);
}

namespace noa {
    /// Generic element-wise transformation.
    /// \param[in] inputs   Input varray(s) and/or values to transform.
    /// \param[out] outputs Output varray(s).
    /// \param op           Operator satisfying the ewise core interface.
    ///                     The operator is perfectly forwarded to the backend, but note that more than
    ///                     one move can happen by the time the operator reaches the compute kernel.
    ///                     Each computing thread holds a copy of this operator.
    ///
    /// \note Compared to iwise, this function can analyze the inputs and outputs to deduce the most efficient way to
    ///       traverse the arrays. For instance, it can reorder dimensions, collapse contiguous dimensions together
    ///       (up to 1d), and can trigger the vectorization for the 1d case by checking for data contiguity and aliasing.
    ///       Note that because the core interface allows the operator to modify the inputs and allows the inputs/outputs
    ///       to alias, GPU vectorization can only be triggered if the input varrays have a const value type and if the
    ///       inputs/outputs don't alias. Note that the core interface also allows the output values to be read/updated,
    ///       which greatly simplifies some operations.
    ///
    /// \note Views or Arrays (i.e. varrays) are supported and handled separately from other types. Varrays are
    ///       converted and wrapper into a tuple of accessors, and sent to the backends' ewise core function. For
    ///       asynchronous cases (async CPU stream or GPU), the shared handle of every Array (see Array::share())
    ///       is copied and will be released (i.e. destructed) when the core function finishes, thus ensuring that the
    ///       Array's resource stays alive during the processing. Other types (so everything that is not a varray)
    ///       are moved/copied into a wrapper (AccessorValue). The stored value is passed by (const) lvalue reference
    ///       to the operator. No other actions are performed on these types, so it is the responsibility of the caller
    ///       to make sure these stay valid until completion.
    ///
    /// \note \p inputs and \p outputs can be left empty, but note that there should be at least one varray
    ///       (in either \p inputs and \p outputs), otherwise the function will not compile.
    ///       If no outputs are provided, all varray inputs should have the same shape. If they do have the same
    ///       stride layout, the function can reorder them to the rightmost layout for better performance.
    ///       If outputs are provided, there should all be varrays with the same shape, have the same stride order
    ///       and cannot be broadcasted (strides of zeros are not allowed). In this case, the varray inputs will
    ///       need to match the output shape or be broadcastable to the output shape. If the stride order of the
    ///       outputs is not the rightmost order, output varrays are reordered to the rightmost order to maximize
    ///       performance. Of course, this means that input varrays are reordered as well for correctness.
    template<EwiseOptions OPTIONS = EwiseOptions{},
             typename Input = ng::AdaptorUnzip<>,
             typename Output = ng::AdaptorUnzip<>,
             typename EwiseOperator> // TODO EwiseChecker
    void ewise(Input&& inputs, Output&& outputs, EwiseOperator&& op) {
        if constexpr (ng::adaptor_decay<Input, Output>) {
            ng::ewise<OPTIONS, std::decay_t<Input>::ZIP, std::decay_t<Output>::ZIP>(
                std::forward<Input>(inputs).tuple,
                std::forward<Output>(outputs).tuple,
                std::forward<EwiseOperator>(op));
        } else if constexpr (ng::adaptor_decay<Input>) {
            ng::ewise<OPTIONS, std::decay_t<Input>::ZIP, false>(
                std::forward<Input>(inputs).tuple,
                forward_as_tuple(std::forward<Output>(outputs)), // wrap
                std::forward<EwiseOperator>(op));
        } else if constexpr (ng::adaptor_decay<Output>) {
            ng::ewise<OPTIONS, false, std::decay_t<Output>::ZIP>(
                forward_as_tuple(std::forward<Input>(inputs)), // wrap
                std::forward<Output>(outputs).tuple,
                std::forward<EwiseOperator>(op));
        } else {
            ng::ewise<OPTIONS, false, false>(
                forward_as_tuple(std::forward<Input>(inputs)), // wrap
                forward_as_tuple(std::forward<Output>(outputs)), // wrap
                std::forward<EwiseOperator>(op));
        }
    }
}

namespace noa::guts {
    template<EwiseOptions OPTIONS, bool ZIP_INPUT, bool ZIP_OUTPUT, typename Inputs, typename Outputs, typename EwiseOp>
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
                    do_reorder = vany(NotEqual{}, order, Vec{0, 1, 2, 3});

                    outputs.for_each_enumerate([&]<size_t I>(const nt::varray auto& output) {
                        check(not output.is_empty(), "Empty output array detected (index={})", I);
                        check(all(output.strides() > 0),
                              "Output arrays cannot be broadcasted, i.e. strides should not be 0, but got strides:{}={}",
                              I, output.strides());
                        if constexpr (I > 0) {
                            check(device == output.device(),
                                  "Output arrays should be on the same device, but got device:0={} and device:{}={}",
                                  device, I, output.device());
                            check(vall(Equal{}, shape, output.shape()),
                                  "Output arrays should have the same shape, but got shape:0={} and shape:{}={}",
                                  shape, I, output.shape());
                            check(vall(Equal{}, order, ni::order(output.strides(), shape)),
                                  "Output arrays should have the same stride order, but got strides:0={} and strides:{}={}",
                                  first_output.strides(), I, output.strides());
                        }
                    });

                    // Automatic broadcasting of the inputs.
                    // "inputs" is used after forward, but as mentioned above "to_tuple_of_accessors"
                    // doesn't actually move varrays and here we only read varrays.
                    input_accessors.for_each_enumerate([&inputs, &shape, &device]<size_t I, typename T>(T& accessor) {
                        if constexpr (nt::varray_decay<decltype(inputs[Tag<I>{}])>) {
                            static_assert(nt::accessor_pure<T>);

                            const auto& input = inputs[Tag<I>{}];
                            check(not input.is_empty(), "Empty input array detected (index={})", I);
                            check(device == input.device(),
                                  "Input arrays should be on the output device, but got device={} and input:{}:device={}",
                                  device, I, input.device());

                            const auto& input_shape = input.shape();
                            if (not ni::broadcast(input_shape, accessor.strides(), shape)) {
                                panic("Cannot broadcast an array of shape {} into an array of shape {}",
                                      input_shape, shape);
                            }
                        } else {
                            static_assert(nt::accessor_value<T>);
                        }
                    });
                } else {
                    static_assert(nt::always_false<>, "The outputs should be varrays");
                }
            } else { // N_INPUTS >= 1
                constexpr i64 index_of_first_varray = guts::index_of_first_varray<Inputs>();
                if constexpr (index_of_first_varray >= 0) {
                    constexpr auto INDEX = static_cast<size_t>(index_of_first_varray);
                    const auto& first_input_array = inputs[Tag<INDEX>{}];
                    shape = first_input_array.shape();
                    device = first_input_array.device();
                    order = ni::order(first_input_array.strides(), shape);
                    do_reorder = vany(NotEqual{}, order, Vec{0, 1, 2, 3});

                    inputs.for_each_enumerate([&]<size_t I, typename T>(T& input) {
                        if constexpr (nt::varray<T>)
                            check(not input.is_empty(), "Empty input array detected (index={})", I);
                        if constexpr (I > INDEX and nt::varray<T>) {
                            check(device == input.device(),
                                  "Input arrays should be on the same device, but got device:0={} and device:{}={}",
                                  device, I, input.device());
                            check(vall(Equal{}, shape, input.shape()),
                                  "Input arrays should have the same shape, but got shape:0={} and shape:{}={}",
                                  shape, I, input.shape());

                            // Only reorder if all the inputs have the same order.
                            if (do_reorder)
                                do_reorder = vall(Equal{}, order, ni::order(input.strides(), shape));
                            // TODO Forcing the same order is okay, but may be a bit too restrictive since it effectively
                            //      prevents automatic broadcasting (the caller can still explicitly broadcast though).
                            //      We may instead find the input with the largest effective shape and use it as
                            //      as reference for reordering the inputs?
                        }
                    });
                } else {
                    static_assert(nt::always_false<>, "For cases with inputs but without outputs, there should be at least one input varray");
                }
            }

            if (do_reorder) {
                shape = shape.reorder(order);
                guts::reorder_accessors(order, input_accessors, output_accessors);
            }

            Stream& stream = Stream::current(device);
            if constexpr (OPTIONS.generate_cpu) {
                if (device.is_cpu()) {
                    auto& cpu_stream = stream.cpu();
                    auto n_threads = cpu_stream.thread_limit();
                    using config = noa::cpu::EwiseConfig<ZIP_INPUT, ZIP_OUTPUT>;

                    if (cpu_stream.is_sync()) {
                        noa::cpu::ewise<config>(
                            shape, std::forward<EwiseOp>(ewise_op),
                            std::move(input_accessors),
                            std::move(output_accessors),
                            n_threads);
                    } else {
                        cpu_stream.enqueue([=,
                            op = std::forward<EwiseOp>(ewise_op),
                            ia = std::move(input_accessors),
                            oa = std::move(output_accessors),
                            ih = guts::extract_shared_handle_from_arrays(std::forward<Inputs>(inputs)),
                            oh = guts::extract_shared_handle_from_arrays(std::forward<Outputs>(outputs))
                        ] {
                            noa::cpu::ewise<config>(shape, std::move(op), std::move(ia), std::move(oa), n_threads);
                        });
                    }
                    return;
                }
            }
            if constexpr (OPTIONS.generate_gpu) {
                if (device.is_gpu()) {
                    #ifdef NOA_ENABLE_CUDA
                    auto& cuda_stream = Stream::current(device).cuda();
                    using config = noa::cuda::EwiseConfig<
                        ZIP_INPUT, ZIP_OUTPUT,
                        OPTIONS.gpu_block_size,
                        OPTIONS.gpu_n_elements_per_thread,
                        OPTIONS.gpu_vectorize>;
                    noa::cuda::ewise<config>(
                        shape, std::forward<EwiseOp>(ewise_op),
                        std::move(input_accessors),
                        std::move(output_accessors),
                        cuda_stream);

                    // Enqueue the shared handles.
                    // Doing it using a single call to enqueue_attach is slightly more efficient.
                    [&]<size_t... I, size_t... O>(std::index_sequence<I...>, std::index_sequence<O...>) {
                        // "enqueue_attach" saves shared_ptr types and anything with a .share() that returns
                        // a shared_ptr, and ignores everything else. As such, we could directly pass the values
                        // of "inputs" and "outputs", but here we explicitly only want to save the shared_ptr
                        // from arrays.
                        auto ih = guts::extract_shared_handle_from_arrays(std::forward<Inputs>(inputs));
                        auto oh = guts::extract_shared_handle_from_arrays(std::forward<Outputs>(outputs));
                        cuda_stream.enqueue_attach(std::move(ih)[Tag<I>{}]..., std::move(oh)[Tag<O>{}]...);

                        // Work-around to remove spurious "set but unused variable" warning (g++11).
                        if constexpr (sizeof...(I) == 0) (void) ih;
                        if constexpr (sizeof...(O) == 0) (void) oh;
                    }(nt::index_list_t<Inputs>{}, nt::index_list_t<Outputs>{});
                    return;
                    #else
                    panic_no_gpu_backend();
                    #endif
                }
            }
        }
    }
}
#endif
