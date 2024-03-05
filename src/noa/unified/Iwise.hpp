#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/types/Shape.hpp"
#include "noa/unified/Device.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Utilities.hpp"

#include "noa/cpu/Iwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Iwise.cuh"
#endif

namespace noa {
    /// Index-wise core function; dispatches an index-wise operator across N-dimensional (parallel) for-loops.
    /// \tparam I           Integral type of the multidimensional indices.
    /// \param shape        Shape of the N-dimensional loop. Between 1d and 4d.
    /// \param device       Device on which to dispatch the operator.
    ///                     The function is enqueued to the current stream of that device.
    /// \param op           Operator satisfying the iwise core interface.
    ///                     The operator is perfectly forwarded to the backend.
    /// \param attachments  Resources to attach to the function call. These are usually Arrays that hold the
    ///                     resources used by the operator, but others attachments can be passed to (see note below).
    ///
    /// \note Attachments are resources that should be kept alive (at least) until the stream is done computing
    ///       the iwise loop. A resource is anything that is convertible to `std::shared_ptr<const void>` or a type
    ///       that has a .share() member function returning a `std::shared_ptr`. Passing anything else in
    ///       \p attachments is also valid, but will be ignored! In practice, these shared_ptr are copied|moved and
    ///       sent to the stream alongside the operator, thereby incrementing their reference count. When the stream
    ///       is done executing the loop, the shared_ptr will be deleted. The CPU backend deletes the shared_ptr
    ///       immediately, but note that other backends (e.g. CUDA) may not destroy these shared_ptr right away and
    ///       instead delay the destruction to the next synchronization or enqueueing call.
    template<typename Op, typename I, size_t N, typename... Ts>
    void iwise(const Shape<I, N>& shape, const Device& device, Op&& op, Ts&&... attachments) {
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            // TODO For now, use the default config, which is meant to trigger the parallel loop
            //      only for large shapes. We could add a way for the user to change that default?
            auto& cpu_stream = stream.cpu();
            const auto n_threads =  cpu_stream.thread_limit();
            if constexpr (sizeof...(Ts) == 0) {
                cpu_stream.enqueue(
                        noa::cpu::iwise<noa::cpu::IwiseConfig<>, N, I, Op>,
                        shape, std::forward<Op>(op),
                        n_threads);
            } else {
                if (cpu_stream.is_sync()) {
                    noa::cpu::iwise(shape, std::forward<Op>(op), n_threads);
                } else {
                    cpu_stream.enqueue(
                            [shape, n_threads,
                             op_ = std::forward<Op>(op),
                             h = guts::extract_shared_handle(forward_as_tuple(std::forward<Ts>(attachments)...))
                            ]() {
                                noa::cpu::iwise(shape, std::move(op_), n_threads);
                            });
                }
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
            // TODO Add option to set the number of bytes of dynamic shared memory.
            auto& cuda_stream = Stream::current(device).cuda();
            noa::cuda::iwise(shape, std::forward<Op>(op), cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Ts>(attachments)...);
            #else
            panic("No GPU backend detected");
            #endif
        }
    }
}
#endif
