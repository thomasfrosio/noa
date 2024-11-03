#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/utils/Adaptor.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/unified/Traits.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Utilities.hpp"

#include "noa/cpu/ReduceEwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Allocators.hpp"
#include "noa/gpu/cuda/Copy.cuh"
#include "noa/gpu/cuda/ReduceEwise.cuh"
#endif

namespace noa {
    struct ReduceEwiseOptions {
        /// Whether to compile for the CPU compute device.
        bool generate_cpu{true};

        /// Whether to compile for the GPU compute device.
        bool generate_gpu{true};

        /// GPU kernel configurations.
        u32 gpu_n_elements_per_thread{8};
        u32 gpu_block_size{512};
        u32 gpu_max_grid_size{4096};
        bool gpu_enable_vectorization{true};
    };
}

namespace noa::guts {
    template<ReduceEwiseOptions, bool, bool, bool, typename Inputs, typename Reduced, typename Outputs, typename Op>
    constexpr void reduce_ewise(Inputs&&, Reduced&&, Outputs&&, Op&&);
}

namespace noa {
    /// Dispatches an element-wise reduction operator.
    /// \param[in,out] inputs   Input varray or an adaptor containing at least one varray.
    ///                         If multiple varrays are entered, they should have the same shape and device.
    /// \param[in] reduced      Initial value for the reduction, or an adaptor containing these value(s).
    /// \param[in,out] outputs  Output value, or an adaptor containing (a reference of) the output value(s).
    ///                         When this function returns, the output values will have been updated.
    /// \param[in] op           Operator satisfying the reduce_ewise core interface. The operator is perfectly
    ///                         forwarded to the backend (it is moved or copied to the backend compute kernel).
    template<ReduceEwiseOptions OPTIONS = ReduceEwiseOptions{},
             typename Inputs = ng::AdaptorUnzip<>,
             typename Reduced = ng::AdaptorUnzip<>,
             typename Outputs = ng::AdaptorUnzip<>,
             typename Operator>
    void reduce_ewise(
            Inputs&& inputs,
            Reduced&& reduced,
            Outputs&& outputs,
            Operator&& op
    ) {
        static_assert(ng::adaptor_decay<Outputs> or std::is_lvalue_reference_v<Outputs>,
                      "Output values should be lvalue references");

        if constexpr (ng::adaptor_decay<Inputs, Reduced, Outputs>) {
            ng::reduce_ewise<OPTIONS, std::decay_t<Inputs>::ZIP, std::decay_t<Reduced>::ZIP, std::decay_t<Outputs>::ZIP>(
                std::forward<Inputs>(inputs).tuple,
                std::forward<Reduced>(reduced).tuple,
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op));
        } else if constexpr (ng::adaptor_decay<Inputs, Reduced>) {
            ng::reduce_ewise<OPTIONS, std::decay_t<Inputs>::ZIP, std::decay_t<Reduced>::ZIP, false>(
                std::forward<Inputs>(inputs).tuple,
                std::forward<Reduced>(reduced).tuple,
                forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op));
        } else if constexpr (ng::adaptor_decay<Inputs, Outputs>) {
            ng::reduce_ewise<OPTIONS, std::decay_t<Inputs>::ZIP, false, std::decay_t<Outputs>::ZIP>(
                std::forward<Inputs>(inputs).tuple,
                forward_as_tuple(std::forward<Reduced>(reduced)),
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op));
        } else if constexpr (ng::adaptor_decay<Reduced, Outputs>) {
            ng::reduce_ewise<OPTIONS, false, std::decay_t<Reduced>::ZIP, std::decay_t<Outputs>::ZIP>(
                forward_as_tuple(std::forward<Inputs>(inputs)),
                std::forward<Reduced>(reduced).tuple,
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op));
        } else if constexpr (ng::adaptor_decay<Outputs>) {
            ng::reduce_ewise<OPTIONS, false, false, std::decay_t<Outputs>::ZIP>(
                forward_as_tuple(std::forward<Inputs>(inputs)),
                forward_as_tuple(std::forward<Reduced>(reduced)),
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op));
        } else if constexpr (ng::adaptor_decay<Reduced>) {
            ng::reduce_ewise<OPTIONS, false, std::decay_t<Reduced>::ZIP, false>(
                forward_as_tuple(std::forward<Inputs>(inputs)),
                std::forward<Reduced>(reduced).tuple,
                forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op));
        } else if constexpr (ng::adaptor_decay<Inputs>) {
            ng::reduce_ewise<OPTIONS, std::decay_t<Inputs>::ZIP, false, false>(
                std::forward<Inputs>(inputs).tuple,
                forward_as_tuple(std::forward<Reduced>(reduced)),
                forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op));
        } else {
            ng::reduce_ewise<OPTIONS, false, false, false>(
                forward_as_tuple(std::forward<Inputs>(inputs)),
                forward_as_tuple(std::forward<Reduced>(reduced)),
                forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op));
        }
    }
}

namespace noa::guts {
    template<ReduceEwiseOptions OPTIONS, bool ZIP_INPUTS, bool ZIP_REDUCED, bool ZIP_OUTPUTS,
             typename Inputs, typename Reduced, typename Outputs, typename Op>
    constexpr void reduce_ewise(
            Inputs&& inputs,
            Reduced&& reduced,
            Outputs&& outputs,
            Op&& op
    ) {
        constexpr i64 index_of_first_varray = ng::index_of_first_varray<Inputs>();
        static_assert(index_of_first_varray >= 0, "There should be at least one input varray");
        constexpr auto index = static_cast<size_t>(index_of_first_varray);

        Tuple input_accessors = ng::to_tuple_of_accessors(std::forward<Inputs>(inputs));
        Tuple reduced_accessors = ng::to_tuple_of_accessors(std::forward<Reduced>(reduced));

        const auto& first_input_array = inputs[Tag<index>{}];
        auto shape = first_input_array.shape();
        const auto device = first_input_array.device();
        const auto order = ni::order(first_input_array.strides(), shape);
        bool do_reorder = vany(NotEqual{}, order, Vec4<i64>{0, 1, 2, 3});

        inputs.for_each_enumerate([&]<size_t I, typename T>(T& input) {
            if constexpr (I > index and nt::varray<T>) {
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

        if (do_reorder) {
            shape = shape.reorder(order);
            ng::reorder_accessors(order, input_accessors);
        }

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
                    nt::has_allow_vectorization_v<Op> and
                    ng::are_all_value_types_trivially_copyable<decltype(output_accessors)>();

                // Allocate and initialize the output values for the device.
                [[maybe_unused]] auto buffers = output_accessors.map_enumerate([&]<size_t J, typename A>(A& accessor) {
                    using value_t = typename A::value_type;
                    if constexpr (use_device_memory) {
                        auto buffer = noa::cuda::AllocatorDevice<value_t>::allocate_async(1, cuda_stream);
                        accessor.reset_pointer(buffer.get());
                        return buffer;
                    } else {
                        // We use managed memory to do the copy on the host, allowing us to support non-trivially
                        // copyable types (which cannot be safely copied between unregistered host and device memory).
                        auto buffer = noa::cuda::AllocatorManaged<value_t>::allocate(1, cuda_stream);
                        accessor.reset_pointer(buffer.get());

                        // In case the final() function is defined, the core interface requires the output values
                        // to be correctly initialized so that the operator can read from them.
                        // This requirement is removed by the "allow_vectorization" flag.
                        if constexpr (not nt::has_allow_vectorization_v<Op>)
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
                output_accessors.for_each_enumerate([&]<size_t J>(auto& accessor) {
                    if constexpr (use_device_memory) {
                        auto& output = outputs[Tag<J>{}];
                        noa::cuda::copy(accessor.get(), &output, cuda_stream);
                    } else {
                        outputs[Tag<J>{}] = accessor[0];
                    }
                });
                if constexpr (use_device_memory)
                    cuda_stream.synchronize();
                return;
#else
                panic_no_gpu_backend();
#endif
            }
        }
    }
}
#endif
