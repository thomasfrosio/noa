#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/utils/Adaptor.hpp"
#include "noa/unified/Traits.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Indexing.hpp"
#include "noa/unified/Utilities.hpp"

#include "noa/cpu/ReduceIwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/AllocatorManaged.hpp"
#include "noa/gpu/cuda/Copy.cuh"
#include "noa/gpu/cuda/ReduceIwise.cuh"
#endif

namespace noa::guts {
    template<bool, bool, typename Op, typename Reduced, typename Output, typename I, size_t N>
    constexpr void reduce_iwise(const Shape<I, N>&, Device, Op&&, Reduced&&, Output&&);
}

namespace noa {
    /// Dispatches a index-wise reduction operator across N-dimensional (parallel) for-loops.
    /// \param shape            Shape of the N-dimensional loop. Between 1d and 4d.
    /// \param device           Device on which to dispatch the reduction. When this function returns,
    ///                         the current stream of that device is guaranteed to be synchronized.
    /// \param[in] op           Operator satisfying the reduce_iwise core interface. The operator is perfectly
    ///                         forwarded to the backend (it is moved or copied to the backend compute kernel).
    /// \param[in] reduced      Initial reduced value, or an adaptor (see wrap() and zip()) containing
    ///                         the initial reduced value(s).
    /// \param[in,out] outputs  Output value, or an adaptor (see wrap() and zip()) containing (reference of)
    ///                         the output value(s). When this function returns, the output values will have
    ///                         been updated.
    template<typename Reduced = guts::AdaptorUnzip<>,
             typename Output = guts::AdaptorUnzip<>,
             typename Operator, typename Index, size_t N>
    requires (N <= 4)
    void reduce_iwise(
            const Shape<Index, N>& shape,
            Device device,
            Operator&& op,
            Reduced&& reduced,
            Output&& outputs
    ) {
        if constexpr (guts::are_adaptor_v<Reduced, Output>) {
            guts::reduce_iwise<Reduced::ZIP, Output::ZIP>(
                    shape, device, std::forward<Operator>(op),
                    std::forward<Reduced>(reduced).tuple,
                    std::forward<Output>(outputs).tuple);

        } else if constexpr (guts::is_adaptor_v<Reduced>) {
            guts::reduce_iwise<Reduced::ZIP, false>(
                    shape, device, std::forward<Operator>(op),
                    std::forward<Reduced>(reduced).tuple,
                    forward_as_tuple(std::forward<Output>(outputs))); // wrap

        } else if constexpr (guts::is_adaptor_v<Output>) {
            guts::reduce_iwise<false, Output::ZIP>(
                    shape, device, std::forward<Operator>(op),
                    forward_as_tuple(std::forward<Reduced>(reduced)), // wrap
                    std::forward<Output>(outputs).tuple);
        } else {
            guts::reduce_iwise<false, false>(
                    shape, device, std::forward<Operator>(op),
                    forward_as_tuple(std::forward<Reduced>(reduced)), // wrap
                    forward_as_tuple(std::forward<Output>(outputs))); // wrap
        }
    }
}

namespace noa::guts {
    template<bool ZipReduced, bool ZipOutput,
             typename Op, typename Reduced, typename Output, typename I, size_t N>
    constexpr void reduce_iwise(
            const Shape<I, N>& shape,
            Device device,
            Op&& op,
            Reduced&& reduced,
            Output&& outputs
    ) {
        []<typename... T>(nt::TypeList<T...>){
            static_assert(nt::bool_and<(not nt::is_varray_v<std::remove_reference_t<T>>)...>::value,
                          "The initial reduced or the output values should not be varrays");
            static_assert(nt::bool_and<(not std::is_const_v<std::remove_reference_t<T>>)...>::value,
                          "The initial reduced or the output values should not be const qualified");
        }(nt::type_list_t<Reduced>{} + nt::type_list_t<Output>{});

        Tuple reduced_accessors = guts::to_tuple_of_accessors(std::forward<Reduced>(reduced));

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            Tuple output_accessors = outputs.map([]<typename U>(U& v) {
                return AccessorRestrictContiguous<U, 1, I>(&v);
            });
            cpu_stream.synchronize();
            using config_t = noa::cpu::ReduceIwiseConfig<ZipReduced, ZipOutput>;
            noa::cpu::reduce_iwise<config_t>(
                    shape, std::forward<Op>(op),
                    reduced_accessors, output_accessors,
                    cpu_stream.thread_limit());

        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();

            // Create the accessors as placeholders for device pointers.
            auto output_accessors = outputs.map([]<typename T>(T&) {
                return AccessorRestrictContiguous<T, 1, I>();
            });

            // Allocate and initialize the output values for the device.
            [[maybe_unused]] auto buffers = output_accessors.map_enumerate([&]<size_t J, typename A>(A& accessor) {
                // We use managed memory to do the copy on the host, allowing us to support non-trivially copyable
                // types (such types cannot be safely copied between unregistered host and device memory).
                auto buffer = noa::cuda::AllocatorManaged<typename A::value_type>::allocate(1, cuda_stream);
                accessor.reset_pointer(buffer.get());

                // In the case of a defined final() operator member function, the core interface requires
                // the output values to be correctly initialized so that the operator can read from them.
                accessor[0] = outputs[Tag<J>{}];
                return buffer;
            });

            // Compute the reduction.
            using config = noa::cuda::ReduceIwiseConfig<ZipReduced, ZipOutput>;
            noa::cuda::reduce_iwise<config>(shape, std::forward<Op>(op), reduced_accessors, output_accessors, cuda_stream);
            cuda_stream.synchronize();

            // Copy the results back to the output values.
            output_accessors.for_each_enumerate([&]<size_t J>(auto& accessor) {
                outputs[Tag<J>{}] = accessor[0];
            });

            #else
            panic("No GPU backend detected");
            #endif
        }
    }
}
#endif
