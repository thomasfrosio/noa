#pragma once

#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/Traits.hpp"
#include "noa/runtime/Stream.hpp"
#include "noa/runtime/Utils.hpp"

#include "noa/runtime/cpu/ReduceEwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Copy.cuh"
#include "noa/runtime/cuda/ReduceEwise.cuh"
#endif

namespace noa {
    struct ReduceEwiseOptions {
        /// Whether to compile for the CPU compute device.
        bool generate_cpu{true};

        /// Whether to compile for the GPU compute device.
        bool generate_gpu{true};

        /// Whether the implementation can use vectorized load/store instructions.
        ///  See the corresponding option in EwiseOptions for more details.
        bool gpu_enable_vectorization{true};

        /// Size of the block. The shape of the block is implementation defined and depends on the inputs.
        /// See ReduceIwiseOptions::gpu_block_shape for more details.
        u32 gpu_block_size{512};

        /// The maximum number of blocks to launch.
        /// See the corresponding option in ReduceIwiseOptions for more details.
        u32 gpu_max_grid_size{4096};

        /// Sets the number of elements done by each thread.
        /// See the corresponding option in EwiseOptions for more details.
        u32 gpu_n_elements_per_thread{8};
    };
}

namespace noa::details {
    template<ReduceEwiseOptions, bool, bool, bool, typename Inputs, typename Reduced, typename Outputs, typename Op>
    constexpr void reduce_ewise(Inputs&&, Reduced&&, Outputs&&, Op&&);
}

namespace noa {
    /// Dispatches an element-wise reduction operator.
    /// \param[in,out] inputs:
    ///     Input varray or an adaptor containing at least one varray.
    ///     If multiple varrays are entered, they should have the same shape and device.
    /// \param[in] reduced:
    ///     Same as reduce-iwise.
    /// \param[in,out] outputs:
    ///     Output value, or an adaptor containing (a reference of) the output value(s).
    ///     When this function returns, the output values will have been updated.
    ///
    /// \param[in] op:
    ///     Operator satisfying the reduce-iwise interface described below.
    ///     The operator is forwarded to the backend and ultimately copied to each compute thread.
    ///     The implementation calls the operator in the following manner:
    ///
    /// ->  op.init(...): Same as reduce-iwise.
    ///
    /// ->  op(handle, inputs&..., reduced&...) or
    ///     op(inputs&..., reduced&...):
    ///         Main reduction step, called once per input element, according to the provided input and output adaptors.
    ///         The reduced values are initialized (using \p reduced) or already joined, and should be updated during
    ///         this step. An empty adaptor can be passed to \p reduced, in which case no reduced values will be passed
    ///         (an empty fuse adaptor will pass an empty tuple).
    ///
    /// ->  op.deinit(...): Same as reduce-iwise.
    ///
    /// ->  op.join(reduced&..., reduced&...) or
    ///     op(reduced&..., reduced&...):
    ///         Same as reduce-iwise, except that it can default to operator(), which is useful for reductions
    ///         where the input type(s) are the same as the reduced type(s).
    ///
    /// ->  op.post(...): Same as reduce-iwise.
    template<ReduceEwiseOptions OPTIONS = ReduceEwiseOptions{},
             typename Inputs = nd::AdaptorUnzip<>,
             typename Reduced = nd::AdaptorUnzip<>,
             typename Outputs = nd::AdaptorUnzip<>,
             typename Operator>
    void reduce_ewise(
        Inputs&& inputs,
        Reduced&& reduced,
        Outputs&& outputs,
        Operator&& op
    ) {
        static_assert(nd::adaptor_decay<Outputs> or std::is_lvalue_reference_v<Outputs>,
                      "Output values should be lvalue references");

        if constexpr (nd::adaptor_decay<Inputs, Reduced, Outputs>) {
            nd::reduce_ewise<OPTIONS, std::decay_t<Inputs>::ZIP, std::decay_t<Reduced>::ZIP, std::decay_t<Outputs>::ZIP>(
                std::forward<Inputs>(inputs).tuple,
                std::forward<Reduced>(reduced).tuple,
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op));
        } else if constexpr (nd::adaptor_decay<Inputs, Reduced>) {
            nd::reduce_ewise<OPTIONS, std::decay_t<Inputs>::ZIP, std::decay_t<Reduced>::ZIP, false>(
                std::forward<Inputs>(inputs).tuple,
                std::forward<Reduced>(reduced).tuple,
                noa::forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op));
        } else if constexpr (nd::adaptor_decay<Inputs, Outputs>) {
            nd::reduce_ewise<OPTIONS, std::decay_t<Inputs>::ZIP, false, std::decay_t<Outputs>::ZIP>(
                std::forward<Inputs>(inputs).tuple,
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op));
        } else if constexpr (nd::adaptor_decay<Reduced, Outputs>) {
            nd::reduce_ewise<OPTIONS, false, std::decay_t<Reduced>::ZIP, std::decay_t<Outputs>::ZIP>(
                noa::forward_as_tuple(std::forward<Inputs>(inputs)),
                std::forward<Reduced>(reduced).tuple,
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op));
        } else if constexpr (nd::adaptor_decay<Outputs>) {
            nd::reduce_ewise<OPTIONS, false, false, std::decay_t<Outputs>::ZIP>(
                noa::forward_as_tuple(std::forward<Inputs>(inputs)),
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op));
        } else if constexpr (nd::adaptor_decay<Reduced>) {
            nd::reduce_ewise<OPTIONS, false, std::decay_t<Reduced>::ZIP, false>(
                noa::forward_as_tuple(std::forward<Inputs>(inputs)),
                std::forward<Reduced>(reduced).tuple,
                noa::forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op));
        } else if constexpr (nd::adaptor_decay<Inputs>) {
            nd::reduce_ewise<OPTIONS, std::decay_t<Inputs>::ZIP, false, false>(
                std::forward<Inputs>(inputs).tuple,
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                noa::forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op));
        } else {
            nd::reduce_ewise<OPTIONS, false, false, false>(
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
    constexpr void reduce_ewise(
        Inputs&& inputs,
        Reduced&& reduced,
        Outputs&& outputs,
        Op&& op
    ) {
        constexpr auto INDEX_OF_FIRST_ARRAY = [] {
            constexpr isize INDEX = nd::index_of_first_array<Inputs>();
            static_assert(INDEX >= 0, "There should be at least one input varray");
            return static_cast<usize>(INDEX);
        }();

        static_assert(nd::are_all_not_arrays_and_mutable<Reduced, Outputs>(), "The initial reduction and output values should be mutable and not arrays");
        constexpr auto NDIM = nd::maximum_nd_axes_of_arrays<Inputs>();
        Tuple input_accessors = nd::to_tuple_of_accessors_nd<NDIM>(std::forward<Inputs>(inputs));
        Tuple reduced_accessors = nd::to_tuple_of_accessor_values(std::forward<Reduced>(reduced));

        const auto& first_input_array = inputs[Tag<INDEX_OF_FIRST_ARRAY>{}];
        auto shape = first_input_array.shape().template extend_front_to<NDIM>(1);
        const auto device = first_input_array.device();

        inputs.for_each_enumerate([&]<usize I, typename T>(T& input) {
            if constexpr (I > INDEX_OF_FIRST_ARRAY and nt::array<T>) {
                check(device == input.device(),
                      "Input arrays should be on the same device, but got input:0:device={} and input:{}:device={}",
                      device, I, input.device());
                const auto input_shape = input.shape().template extend_front_to<NDIM>(1);
                check(shape == input_shape,
                      "Input arrays should have the same shape, but got input:0:shape={} and input:{}:shape={}",
                      shape, I, input_shape);
            }
        });

        nd::optimize_reduce_ewise_layout(shape, input_accessors);

        Stream& stream = Stream::current(device);

        if constexpr (OPTIONS.generate_cpu) {
            if (device.is_cpu()) {
                auto& cpu_stream = stream.cpu();
                using config = noa::cpu::ReduceEwiseConfig<ZIP_INPUTS, ZIP_REDUCED, ZIP_OUTPUTS>;

                Tuple output_accessors = outputs.map([]<typename U>(U& v) {
                    return AccessorRestrictContiguous<U, 1, i32>(&v);
                });
                cpu_stream.synchronize();
                noa::cpu::reduce_ewise<config>(
                    shape, std::forward<Op>(op),
                    std::move(input_accessors),
                    std::move(reduced_accessors),
                    output_accessors,
                    cpu_stream.thread_limit());
                return;
            }
        }

        if constexpr (OPTIONS.generate_gpu) {
            if (device.is_gpu()) {
                #ifdef NOA_ENABLE_CUDA
                auto& cuda_stream = stream.cuda();

                // Create the accessors as placeholders for device pointers.
                auto output_accessors = outputs.map([]<typename T>(T&) {
                    return AccessorRestrictContiguous<T, 1, i32>();
                });

                constexpr bool use_device_memory =
                    nt::enable_vectorization_v<Op> and
                    nd::are_all_value_types_trivially_copyable<decltype(output_accessors)>();

                // Allocate and initialize the output values for the device.
                [[maybe_unused]] auto buffers = output_accessors.map_enumerate([&]<usize J, typename A>(A& accessor) {
                    using value_t = typename A::value_type;
                    if constexpr (use_device_memory) {
                        auto buffer = noa::cuda::AllocatorDevice::allocate_async<value_t>(1, cuda_stream);
                        accessor.reset_pointer(buffer.get());
                        return buffer;
                    } else {
                        // We use managed memory to do the copy on the host, allowing us to support non-trivially
                        // copyable types (which cannot be safely copied between unregistered host and device memory).
                        auto buffer = noa::cuda::AllocatorManaged::allocate<value_t>(1, cuda_stream);
                        accessor.reset_pointer(buffer.get());

                        // In case the final() function is defined, the core interface requires the output values
                        // to be correctly initialized so that the operator can read from them.
                        // This requirement is removed by the "enable_vectorization" flag.
                        if constexpr (not nt::enable_vectorization_v<Op>)
                            accessor[0] = outputs[Tag<J>{}]; // TODO else prefetch to device?
                        return buffer;
                    }
                });

                // Compute the reduction.
                using config = noa::cuda::ReduceEwiseConfig<
                    ZIP_INPUTS, ZIP_REDUCED, ZIP_OUTPUTS,
                    OPTIONS.gpu_n_elements_per_thread,
                    OPTIONS.gpu_block_size,
                    OPTIONS.gpu_max_grid_size,
                    OPTIONS.gpu_enable_vectorization>;
                noa::cuda::reduce_ewise<config>(
                    shape, std::forward<Op>(op),
                    std::move(input_accessors),
                    std::move(reduced_accessors),
                    output_accessors,
                    cuda_stream);
                if constexpr (not use_device_memory)
                    cuda_stream.synchronize();

                // Copy the results back to the output values.
                output_accessors.for_each_enumerate([&]<usize J>(auto& accessor) {
                    if constexpr (use_device_memory) {
                        auto& output = outputs[Tag<J>{}];
                        noa::cuda::copy(accessor.get(), &output, cuda_stream);
                    } else {
                        outputs[Tag<J>{}] = accessor[0];
                    }
                });
                if constexpr (use_device_memory)
                    cuda_stream.synchronize();
                #else
                panic_no_gpu_backend();
                #endif
            }
        }
    }
}
