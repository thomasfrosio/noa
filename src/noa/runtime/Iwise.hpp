#pragma once

#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/Device.hpp"
#include "noa/runtime/Stream.hpp"
#include "noa/runtime/Utils.hpp"

#include "noa/runtime/cpu/Iwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/runtime/cuda/Iwise.cuh"
#endif

namespace noa {
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
        i32 cpu_launch_n_threads{0};

        /// Each CPU thread is assigned to work on at least this number of indices.
        /// As such, the parallel version of the nd-loop is only called if there are more elements than this value.
        isize cpu_number_of_indices_per_threads{1'048'576}; // 2^20

        /// GPU thread block size.
        /// 1d shapes map to a 1d block of gpu_block_size threads, and higher dimensions use a 2d block of
        /// (height=WARP_SIZE/gpu_block_size, width=WARP_SIZE) threads.
        /// TODO Would a gpu_block_shape be more useful?
        u32 gpu_block_size{512};

        /// Sets the number of iterations along the width done by each thread.
        /// Increasing this value decreases the parallelism (the number of blocks launched),
        /// but may still be beneficial for operators doing little work per iteration.
        /// TODO Make this more flexible and allow to increase it for depth and height too?
        u32 gpu_number_of_indices_per_threads{1};

        /// Allocate the specified number of bytes for the per-block scratch (shared memory in CUDA).
        /// The maximum alignment is std::max_align_t (16 in CUDA), which should be fine for most cases.
        /// The scratch is available from the operators via the compute handle during init/call/deinit.
        /// TODO The CUDA backend has this as a runtime option. This should be a runtime option too.
        usize gpu_scratch_size{0};
    };

    /// Index-wise core function; dispatches an index-wise operator across N-dimensional (parallel) for-loops.
    /// \param shape:
    ///     Shape of the N-dimensional loop. Between 1d and 4d.
    ///     The index type is the integer type of the multidimensional indices.
    /// \param device:
    ///     Device on which to dispatch the operator.
    ///     The function is enqueued to the current stream of that device.
    ///
    /// \param[in] op:
    ///     Operator satisfying the iwise interface described below.
    ///     The operator is forwarded to the backend and ultimately copied to each compute thread.
    ///     The implementation calls the operator in the following manner:
    ///
    /// -> op.init(handle) or
    ///    op.init():
    ///     Defaulted to no-op. If defined, each thread calls it. Since operators are per-thread,
    ///     this can be used to perform some initialization of the operator or of the per-block scratch.
    ///     The default no-op can be removed by defining the type Op::remove_default_init.
    ///
    /// -> op(handle, Vec{indices...}),
    ///    op(handle, indices...),
    ///    op(Vec{indices...}) or
    ///    op(indices...):
    ///     Called once per nd-index.
    ///
    /// -> op.deinit(handle) or
    ///    op.deinit():
    ///     Defaulted to no-op. If defined, each thread calls it. It is the mirror operation of op.init()
    ///     and is often used to save some state or save the per-block scratch to a permanent location.
    ///     The default no-op can be removed by defining the type Op::remove_default_deinit.
    ///
    /// \param[in] attachments:
    ///     Resources that should be kept alive (at least) until the stream is done computing the iwise loop.
    ///     A resource is anything that is convertible to `std::shared_ptr<const void>` or a type that has a .share()
    ///     member function returning a `std::shared_ptr`. Passing anything else is also valid, but will be ignored!
    ///     In practice, these shared_ptr are copied|moved and sent to the stream alongside the operator, thereby
    ///     incrementing their reference count. When the stream is done executing the loop, the shared_ptr will be
    ///     deleted. The CPU backend deletes the shared_ptr immediately, but note that other backends (e.g. CUDA)
    ///     may not destroy these shared_ptr right away and instead delay the destruction to the next synchronization
    ///     or enqueueing call.
    template<IwiseOptions OPTIONS = IwiseOptions{}, typename Op, typename I, usize N, typename... Ts>
    void iwise(const Shape<I, N>& shape, const Device& device, Op&& op, Ts&&... attachments) {
        Stream& stream = Stream::current(device);

        if constexpr (OPTIONS.generate_cpu) {
            if (device.is_cpu()) {
                auto& cpu_stream = stream.cpu();

                namespace nc = noa::cpu;
                constexpr bool LAUNCH_EXACT = OPTIONS.cpu_launch_n_threads > 0;
                constexpr isize N_ELEMENTS_PER_THREAD = LAUNCH_EXACT ? 1 : OPTIONS.cpu_number_of_indices_per_threads;
                const auto n_threads = LAUNCH_EXACT ? OPTIONS.cpu_launch_n_threads : cpu_stream.thread_limit();
                using config_t = nc::IwiseConfig<N_ELEMENTS_PER_THREAD>;

                if constexpr (sizeof...(Ts) == 0) {
                    cpu_stream.enqueue(nc::iwise<config_t, N, I, Op>, shape, std::forward<Op>(op), n_threads);
                } else {
                    if (cpu_stream.is_sync()) {
                        nc::iwise<config_t>(shape, std::forward<Op>(op), n_threads);
                    } else {
                        cpu_stream.enqueue(
                            [shape, n_threads,
                                op_ = std::forward<Op>(op),
                                h = nd::extract_shared_handle(noa::forward_as_tuple(std::forward<Ts>(attachments)...))
                            ] {
                                nc::iwise<config_t>(shape, std::move(op_), n_threads);
                            });
                    }
                }
                return;
            }
        }

        if constexpr (OPTIONS.generate_gpu) {
            if (device.is_gpu()) {
                #ifdef NOA_ENABLE_CUDA
                namespace nc = noa::cuda;
                constexpr auto BLOCK_SIZE = OPTIONS.gpu_block_size;
                constexpr auto WARP_SIZE = nc::Constant::WARP_SIZE;
                static_assert(is_multiple_of(BLOCK_SIZE, WARP_SIZE));
                using Config = std::conditional_t<
                    N == 1,
                    nc::StaticBlock<BLOCK_SIZE, 1, 1, OPTIONS.gpu_number_of_indices_per_threads>,
                    nc::StaticBlock<WARP_SIZE, BLOCK_SIZE / WARP_SIZE, 1, OPTIONS.gpu_number_of_indices_per_threads>>;

                auto& cuda_stream = stream.cuda();
                nc::iwise<N, Config>(shape, std::forward<Op>(op), cuda_stream, OPTIONS.gpu_scratch_size);
                cuda_stream.enqueue_attach(std::forward<Ts>(attachments)...);
                #else
                panic_no_gpu_backend();
                #endif
            }
        }
    }
}
