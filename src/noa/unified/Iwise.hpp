#pragma once

#include "noa/core/types/Shape.hpp"
#include "noa/unified/Device.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Utilities.hpp"

#include "noa/cpu/Iwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Iwise.cuh"
#endif

namespace noa {
    struct GpuThreadBlock {};

    struct IwiseOptions {
        /// Whether the CPU code path should be generated.
        bool generate_cpu{true};

        /// Whether the GPU code path should be generated.
        bool generate_gpu{true};

        /// Distribute the nd-shape to this exact number of CPU threads.
        /// - A value of 0 means to not enforce the number of threads and to instead let the CPU backend decide the
        ///   best number of threads to launch according to the option cpu_number_of_indices_per_threads (see below)
        ///   and Stream::thread_limit.
        /// - A positive value bypasses this logic and requires the CPU backend to launch that many threads,
        ///   regardless of the nd-shape, cpu_number_of_indices_per_threads or Stream::thread_limit.
        ///   This also means that cpu_launch_n_threads=1 turns off the multithreading completely and
        ///   only the serial loop is generated.
        ///   Note that if this value cannot be known at compile-time, the same behavior can be achieved by setting
        ///   cpu_number_of_indices_per_threads=1 and the Stream::set_thread_limit(n). The only difference is that
        ///   in this case the optimizer may not be able to identify which of the parallel or serial nd-loops is
        ///   needed, thus both implementations will be likely generated even if only one is needed.
        i64 cpu_launch_n_threads{0};

        /// Each CPU thread is assigned to work on at least this number of indices.
        /// As such, the parallel version of the nd-loop is only called if there are more elements than this value.
        i64 cpu_number_of_indices_per_threads{1'048'576}; // 2^20
    };

    /// Index-wise core function; dispatches an index-wise operator across N-dimensional (parallel) for-loops.
    /// \tparam I           Integral type of the multidimensional indices.
    /// \param shape        Shape of the N-dimensional loop. Between 1d and 4d.
    /// \param device       Device on which to dispatch the operator.
    ///                     The function is enqueued to the current stream of that device.
    /// \param op           Operator satisfying the iwise core interface.
    ///                     The operator is perfectly forwarded to the backend, but
    ///                     each computing thread ends up holding a copy of the operator.
    /// \param attachments  Resources to attach to the function call. These are usually Arrays that hold the
    ///                     resources used by the operator, but other attachments can be passed too (see note below).
    ///
    /// \note Attachments are resources that should be kept alive (at least) until the stream is done computing
    ///       the iwise loop. A resource is anything that is convertible to `std::shared_ptr<const void>` or a type
    ///       that has a .share() member function returning a `std::shared_ptr`. Passing anything else in
    ///       \p attachments is also valid, but will be ignored! In practice, these shared_ptr are copied|moved and
    ///       sent to the stream alongside the operator, thereby incrementing their reference count. When the stream
    ///       is done executing the loop, the shared_ptr will be deleted. The CPU backend deletes the shared_ptr
    ///       immediately, but note that other backends (e.g. CUDA) may not destroy these shared_ptr right away and
    ///       instead delay the destruction to the next synchronization or enqueueing call.
    template<IwiseOptions OPTIONS = IwiseOptions{}, typename Op, typename I, size_t N, typename... Ts>
    void iwise(const Shape<I, N>& shape, const Device& device, Op&& op, Ts&&... attachments) {
        Stream& stream = Stream::current(device);
        if constexpr (OPTIONS.generate_cpu) {
            if (device.is_cpu()) {
                auto& cpu_stream = stream.cpu();

                constexpr bool LAUNCH_EXACT = OPTIONS.cpu_launch_n_threads > 0;
                constexpr i64 N_ELEMENTS_PER_THREAD = LAUNCH_EXACT ? 1 : OPTIONS.cpu_number_of_indices_per_threads;
                const auto n_threads = LAUNCH_EXACT ? OPTIONS.cpu_launch_n_threads : cpu_stream.thread_limit();
                using config_t = noa::cpu::IwiseConfig<N_ELEMENTS_PER_THREAD>;

                if constexpr (sizeof...(Ts) == 0) {
                    cpu_stream.enqueue(noa::cpu::iwise<config_t, N, I, Op>, shape, std::forward<Op>(op), n_threads);
                } else {
                    if (cpu_stream.is_sync()) {
                        noa::cpu::iwise<config_t>(shape, std::forward<Op>(op), n_threads);
                    } else {
                        cpu_stream.enqueue(
                            [shape, n_threads,
                                op_ = std::forward<Op>(op),
                                h = guts::extract_shared_handle(forward_as_tuple(std::forward<Ts>(attachments)...))
                            ] {
                                noa::cpu::iwise<config_t>(shape, std::move(op_), n_threads);
                            });
                    }
                }
                return;
            }
        }
        if constexpr (OPTIONS.generate_gpu) {
            if (device.is_gpu()) {
                #ifdef NOA_ENABLE_CUDA
                // TODO Add option to set the number of bytes of dynamic shared memory.
                auto& cuda_stream = Stream::current(device).cuda();
                noa::cuda::iwise(shape, std::forward<Op>(op), cuda_stream);
                cuda_stream.enqueue_attach(std::forward<Ts>(attachments)...);
                return;
                #else
                panic_no_gpu_backend();
                #endif
            }
        }
    }
}
