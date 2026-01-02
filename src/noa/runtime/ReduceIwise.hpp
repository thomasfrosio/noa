#pragma once

#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/Traits.hpp"
#include "noa/runtime/Stream.hpp"
#include "noa/runtime/Utilities.hpp"

#include "noa/runtime/cpu/ReduceIwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Copy.cuh"
#include "noa/runtime/cuda/ReduceIwise.cuh"
#endif

namespace noa::details {
    template<bool, bool, typename Op, typename Reduced, typename Output, typename I, usize N>
    constexpr void reduce_iwise(const Shape<I, N>&, Device, Op&&, Reduced&&, Output&&);
}

namespace noa {
    /// Dispatches an index-wise reduction operator across N-dimensional (parallel) for-loops.
    /// \param shape            Shape of the 1-, 2-, 3- or 4-dimensional loop.
    /// \param device           Device on which to dispatch the reduction. When this function returns, the current
    ///                         stream of that device is synchronized.
    /// \param[in] reduced      Initial reduction value, or an adaptor (see wrap() and fuse()) containing the initial
    ///                         reduction value(s).
    /// \param[in,out] outputs  Output value, an adaptor (see wrap() and fuse()) containing (a reference of) the output
    ///                         value(s), or an empty adaptor. When this function returns, the output values will have
    ///                         been updated.
    /// \param[in] op           Operator satisfying the reduce_iwise core interface. The operator is perfectly
    ///                         forwarded to the backend (it is moved or copied to the backend compute kernel).
    template<typename Reduced = nd::AdaptorUnzip<>,
             typename Outputs = nd::AdaptorUnzip<>,
             typename Operator, typename Index, usize N>
    requires (N <= 4)
    void reduce_iwise(
        const Shape<Index, N>& shape,
        Device device,
        Reduced&& reduced,
        Outputs&& outputs,
        Operator&& op
    ) {
        static_assert(nd::adaptor_decay<Outputs> or std::is_lvalue_reference_v<Outputs>,
                      "Output value(s) should be reference(s)");

        if constexpr (nd::adaptor_decay<Reduced, Outputs>) {
            nd::reduce_iwise<std::decay_t<Reduced>::ZIP, std::decay_t<Outputs>::ZIP>(
                shape, device, std::forward<Operator>(op),
                std::forward<Reduced>(reduced).tuple,
                std::forward<Outputs>(outputs).tuple);

        } else if constexpr (nd::adaptor_decay<Reduced>) {
            nd::reduce_iwise<std::decay_t<Reduced>::ZIP, false>(
                shape, device, std::forward<Operator>(op),
                std::forward<Reduced>(reduced).tuple,
                noa::forward_as_tuple(std::forward<Outputs>(outputs)));

        } else if constexpr (nd::adaptor_decay<Outputs>) {
            nd::reduce_iwise<false, std::decay_t<Outputs>::ZIP>(
                shape, device, std::forward<Operator>(op),
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                std::forward<Outputs>(outputs).tuple);
        } else {
            nd::reduce_iwise<false, false>(
                shape, device, std::forward<Operator>(op),
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                noa::forward_as_tuple(std::forward<Outputs>(outputs)));
        }
    }
}

namespace noa::details {
    template<bool ZIP_REDUCED, bool ZIP_OUTPUT,
             typename Op, typename Reduced, typename Output, typename I, usize N>
    constexpr void reduce_iwise(
        const Shape<I, N>& shape,
        Device device,
        Op&& op,
        Reduced&& reduced,
        Output&& outputs
    ) {
        []<typename... T>(nt::TypeList<T...>){
            static_assert(((not nt::varray<std::remove_reference_t<T>>) and ...),
                          "The initial reduced or the output values should not be varrays");
            static_assert(((not std::is_const_v<std::remove_reference_t<T>>) and ...),
                          "The initial reduced or the output values should not be const qualified");
        }(nt::type_list_t<Reduced>{} + nt::type_list_t<Output>{});

        Tuple reduced_accessors = details::to_tuple_of_accessors(std::forward<Reduced>(reduced));

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            Tuple output_accessors = outputs.map([]<typename U>(U& v) {
                return AccessorRestrictContiguous<U, 1, I>(&v);
            });
            cpu_stream.synchronize();
            using config_t = noa::cpu::ReduceIwiseConfig<ZIP_REDUCED, ZIP_OUTPUT>;
            noa::cpu::reduce_iwise<config_t>(
                shape, std::forward<Op>(op),
                std::move(reduced_accessors), output_accessors,
                cpu_stream.thread_limit());

        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();

            // Create the accessors as placeholders for device pointers.
            auto output_accessors = outputs.map([]<typename T>(T&) {
                return AccessorRestrictContiguous<T, 1, I>();
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
                    // We use managed memory to do the copy on the host, allowing us to support non-trivially copyable
                    // types (such types cannot be safely copied between unregistered host and device memory).
                    auto buffer = noa::cuda::AllocatorManaged::allocate<value_t>(1, cuda_stream);
                    accessor.reset_pointer(buffer.get());

                    // In the case of a defined final() operator member function, the core interface requires
                    // the output values to be correctly initialized so that the operator can read from them.
                    // This is turned off using the "enable_vectorization" flag.
                    if constexpr (not nt::enable_vectorization_v<Op>)
                        accessor[0] = outputs[Tag<J>{}]; // TODO else prefetch to device?
                    return buffer;
                }
            });

            // Compute the reduction.
            using config = noa::cuda::ReduceIwiseConfig<ZIP_REDUCED, ZIP_OUTPUT>;
            noa::cuda::reduce_iwise<config>(
                shape, std::forward<Op>(op),
                std::move(reduced_accessors),
                output_accessors, cuda_stream);
            if constexpr (not use_device_memory)
                cuda_stream.synchronize();

            // Copy the results back to the output values.
            output_accessors.for_each_enumerate([&]<usize J, typename A>(A& accessor) {
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
