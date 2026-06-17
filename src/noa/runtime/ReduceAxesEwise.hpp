#pragma once

#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/core/Utils.hpp"
#include "noa/runtime/Traits.hpp"
#include "noa/runtime/Stream.hpp"
#include "noa/runtime/Utils.hpp"

#include "noa/runtime/cpu/ReduceAxesEwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/runtime/cuda/ReduceAxesEwise.cuh"
#endif

#include "noa/runtime/ReduceEwise.hpp"
#include "noa/runtime/ReduceAxesIwise.hpp"

namespace noa::details {
    template<ReduceEwiseOptions, bool, bool, bool, typename Inputs, typename Reduced, typename Outputs, typename Op>
    constexpr void reduce_axes_ewise(Inputs&&, Reduced&&, Outputs&&, Op&&);
}

namespace noa {
    /// Computes an element-wise reduction along one or multiple axes.
    /// \param[in,out] inputs:
    ///     Input varray or an adaptor containing at least one varray.
    ///     If multiple varrays are entered, they should have the same shape.
    ///     Input arrays should be on the same device as the output arrays.
    /// \param[in,out] outputs:
    ///     Output array, or an adaptor containing the output array(s). While the input adaptor can contain
    ///     non-varray types, the output should only contain varrays, all of which should be on the same device.
    ///     As opposed to reduce_axes_ewise, this function is asynchronous and does not perform any synchronization.
    ///
    /// \param[in] reduced: Same as reduce-ewise.
    /// \param[in] op:      Same as reduce-ewise.
    template<ReduceEwiseOptions OPTIONS = ReduceEwiseOptions{},
             typename Inputs = nd::AdaptorUnzip<>,
             typename Reduced = nd::AdaptorUnzip<>,
             typename Outputs = nd::AdaptorUnzip<>,
             typename Operator>
    void reduce_axes_ewise(
        Inputs&& inputs,
        Reduced&& reduced,
        Outputs&& outputs,
        Operator&& op
    ) {
        if constexpr (nd::adaptor_decay<Inputs, Reduced, Outputs>) {
            nd::reduce_axes_ewise<OPTIONS, std::decay_t<Inputs>::ZIP, std::decay_t<Reduced>::ZIP, std::decay_t<Outputs>::ZIP>(
                std::forward<Inputs>(inputs).tuple,
                std::forward<Reduced>(reduced).tuple,
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op));
        } else if constexpr (nd::adaptor_decay<Inputs, Reduced>) {
            nd::reduce_axes_ewise<OPTIONS, std::decay_t<Inputs>::ZIP, std::decay_t<Reduced>::ZIP, false>(
                std::forward<Inputs>(inputs).tuple,
                std::forward<Reduced>(reduced).tuple,
                noa::forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op));
        } else if constexpr (nd::adaptor_decay<Inputs, Outputs>) {
            nd::reduce_axes_ewise<OPTIONS, std::decay_t<Inputs>::ZIP, false, std::decay_t<Outputs>::ZIP>(
                std::forward<Inputs>(inputs).tuple,
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op));
        } else if constexpr (nd::adaptor_decay<Reduced, Outputs>) {
            nd::reduce_axes_ewise<OPTIONS, false, std::decay_t<Reduced>::ZIP, std::decay_t<Outputs>::ZIP>(
                noa::forward_as_tuple(std::forward<Inputs>(inputs)),
                std::forward<Reduced>(reduced).tuple,
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op));
        } else if constexpr (nd::adaptor_decay<Outputs>) {
            nd::reduce_axes_ewise<OPTIONS, false, false, std::decay_t<Outputs>::ZIP>(
                noa::forward_as_tuple(std::forward<Inputs>(inputs)),
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op));
        } else if constexpr (nd::adaptor_decay<Reduced>) {
            nd::reduce_axes_ewise<OPTIONS, false, std::decay_t<Reduced>::ZIP, false>(
                noa::forward_as_tuple(std::forward<Inputs>(inputs)),
                std::forward<Reduced>(reduced).tuple,
                noa::forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op));
        } else if constexpr (nd::adaptor_decay<Inputs>) {
            nd::reduce_axes_ewise<OPTIONS, std::decay_t<Inputs>::ZIP, false, false>(
                std::forward<Inputs>(inputs).tuple,
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                noa::forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op));
        } else {
            nd::reduce_axes_ewise<OPTIONS, false, false, false>(
                noa::forward_as_tuple(std::forward<Inputs>(inputs)),
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                noa::forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op));
        }
    }
}

namespace noa::details {
    template<ReduceEwiseOptions OPTIONS, bool ZIP_INPUTS, bool ZIP_REDUCED, bool ZIP_OUTPUTS,
             typename Inputs, typename Reduced, typename Outputs, typename Op>
    constexpr void reduce_axes_ewise(
        Inputs&& inputs,
        Reduced&& reduced,
        Outputs&& outputs,
        Op&& reduce_operator
    ) {
        constexpr auto INDEX_OF_FIRST_ARRAY = [] {
            constexpr isize INDEX = nd::index_of_first_array<Inputs>();
            static_assert(INDEX >= 0, "There should be at least one input varray");
            return static_cast<usize>(INDEX);
        }();

        static_assert(nd::are_all_arrays<Outputs>(), "All of the outputs should be arrays");
        static_assert(std::tuple_size_v<Outputs> > 0, "There should be at least one output");

        constexpr usize NDIM = nd::maximum_nd_axes_of_arrays<Inputs, Outputs>();
        Tuple input_accessors = nd::to_tuple_of_accessors_nd<NDIM>(std::forward<Inputs>(inputs));
        Tuple reduced_accessors = nd::to_tuple_of_accessor_values(std::forward<Reduced>(reduced));
        Tuple output_accessors = nd::to_tuple_of_accessors_nd<NDIM>(std::forward<Outputs>(outputs));

        const auto& first_input_array = inputs[Tag<INDEX_OF_FIRST_ARRAY>{}];
        auto input_shape = first_input_array.shape().template extend_front_to<NDIM>(1);
        auto output_shape = outputs[Tag<0>{}].shape().template extend_front_to<NDIM>(1);
        const auto device = outputs[Tag<0>{}].device();

        inputs.for_each_enumerate([&]<usize I, typename T>(T& input) {
            if constexpr (nt::array<T>) {
                check(device == input.device(),
                      "Input arrays should be on the same device as the output(s), but got output device={} and input:{}:device={}",
                      device, I, input.device());
            }
            if constexpr (I > INDEX_OF_FIRST_ARRAY and nt::array<T>) {
                const auto shape = input.shape().template extend_front_to<NDIM>(1);
                check(input_shape == shape,
                      "Input arrays should have the same shape, but got input:0:shape={} and input:{}:shape={}",
                      input_shape, I, shape);
            }
        });

        outputs.for_each_enumerate([&]<usize I, typename T>(T& output) {
            if constexpr (I > 0) {
                check(device == output.device(),
                      "Output arrays should be on the same device, but got output:0:device={} and output:{}:device={}",
                      device, I, output.device());
                const auto shape = output.shape().template extend_front_to<NDIM>(1);
                check(output_shape == shape,
                      "Output arrays should have the same shape, but got output:0:shape={} and output:{}:shape={}",
                      output_shape, I, shape);
            }
        });

        // Preprocessing.
        auto axes_to_reduce = input_shape.cmp_ne(output_shape);
        check((axes_to_reduce and output_shape.cmp_ne(1)) == false,
              "The output shape should match the input shape, or be 1, indicating the dimension should be reduced to one element. Got input:shape={}, output:shape={}",
              input_shape, output_shape);
        check(axes_to_reduce.any_eq(true),
              "No reduction to compute. Got input:shape={}, output:shape={}. Use ewise instead.",
              input_shape, output_shape);

        nd::optimize_reduce_axes_ewise_layout(input_shape, output_shape, input_accessors, output_accessors);

        // Check the reduction is supported.
        axes_to_reduce = input_shape.cmp_ne(output_shape);
        i32 n_axes_reduced{0};
        bool reduction{false};
        bool batch_reduction{true};
        for (usize i{}; i < NDIM; ++i) {
            if (input_shape[i] > 1)
                continue;
            if (axes_to_reduce[i]) {
                n_axes_reduced++;
                reduction = true;
            } else if (reduction) {
                batch_reduction = false;
            }
        }
        check(n_axes_reduced == 1 or batch_reduction,
              "Reducing more than one axis at a time (after collapsing contiguous dimensions together) is currently limited to a reduction of all axes except the leftmost axis. Got collapsed_input_shape={}, collapsed_output_shape={}, axes_to_reduce={}",
              input_shape, output_shape, axes_to_reduce);

        Stream& stream = Stream::current(device);
        if constexpr (OPTIONS.generate_cpu) {
            if (device.is_cpu()) {
                auto& cpu_stream = stream.cpu();
                auto n_threads = cpu_stream.thread_limit();
                using config = noa::cpu::ReduceAxesEwiseConfig<ZIP_INPUTS, ZIP_REDUCED, ZIP_OUTPUTS>;

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
                            ih = nd::extract_shared_handle_from_arrays(std::forward<Inputs>(inputs)),
                            oh = nd::extract_shared_handle_from_arrays(std::forward<Outputs>(outputs))
                        ] {
                            noa::cpu::reduce_axes_ewise<config>(
                                input_shape, output_shape, std::move(op),
                                std::move(ia), std::move(ir), std::move(oa), n_threads);
                        });
                }
                return;
            }
        }
        if constexpr (OPTIONS.generate_gpu) {
            if (device.is_gpu()) {
                #ifdef NOA_ENABLE_CUDA
                auto& cuda_stream = Stream::current(device).cuda();
                using config = noa::cuda::ReduceEwiseConfig<
                    ZIP_INPUTS, ZIP_REDUCED, ZIP_OUTPUTS,
                    OPTIONS.gpu_n_elements_per_thread,
                    OPTIONS.gpu_block_size,
                    OPTIONS.gpu_max_grid_size,
                    OPTIONS.gpu_enable_vectorization>;
                noa::cuda::reduce_axes_ewise<config>(
                    input_shape, output_shape,
                    std::forward<Op>(reduce_operator),
                    std::move(input_accessors),
                    std::move(reduced_accessors),
                    std::move(output_accessors),
                    cuda_stream);

                // Enqueue the shared handles. See ewise() for more details.
                [&]<usize... I, usize... O>(std::index_sequence<I...>, std::index_sequence<O...>) {
                    auto ih = nd::extract_shared_handle_from_arrays(std::forward<Inputs>(inputs));
                    auto oh = nd::extract_shared_handle_from_arrays(std::forward<Outputs>(outputs));
                    cuda_stream.enqueue_attach(std::move(ih)[Tag<I>{}]..., std::move(oh)[Tag<O>{}]...);
                    // Work-around to remove spurious warning of set but unused variable (g++11).
                    if constexpr (sizeof...(I) == 0) (void) ih;
                    if constexpr (sizeof...(O) == 0) (void) oh;
                }(nt::index_list_t<Inputs>{}, nt::index_list_t<Outputs>{});
                #else
                panic_no_gpu_backend();
                #endif
            }
        }
    }
}
