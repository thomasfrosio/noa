#pragma once

#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/Traits.hpp"
#include "noa/runtime/Stream.hpp"
#include "noa/runtime/Utils.hpp"

#include "noa/runtime/cpu/ReduceIwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Copy.cuh"
#include "noa/runtime/cuda/ReduceIwise.cuh"
#endif

namespace noa {
    struct ReduceIwiseOptions {
        /// Whether to compile for the CPU compute device.
        bool generate_cpu{true};

        /// Whether to compile for the GPU compute device.
        bool generate_gpu{true};

        /// Whether the implementation can use two kernels for maximum performance.
        /// For large generic reductions with non-empty reduced types, the most efficient implementation on GPU
        /// consists in launching two kernels. The blocks of the first kernel will call init->call->deinit->join,
        /// and the single block of the second kernel will call join->post. In other words, the blocks may not go
        /// through the full reduction. Turning off this option is a good way to guarantee that the same block will go
        /// through all phases of the reduction, which may be important for some operators. Note, however, that with
        /// non-empty reduced types, gpu_allow_two_kernels=false is enforcing one block per output element, so there may
        /// be a performance penalty for large reductions (increasing .gpu_block_size might help then). To help
        /// with these edge cases, the GPU compute handle exposes which kernel is currently being executed.
        /// Note that even if there are no reduced values, gpu_allow_two_kernels=true may still result in launching
        /// two kernels just to call the "post" step.
        bool gpu_allow_two_kernels{true};

        /// Shape of the block.
        /// The width of the block always maps to the width of the nd-shape. The height of the block can map any
        /// (or all) of the other dimensions (so, BDH) depending on the nd-shape and type of reductions. As a
        /// consequence, 1d-shapes require 1d-blocks (except if gpu_optimize_block_shape=true, in which case 2d blocks
        /// are reshaped to 1d). The block size needs to be a multiple of gpu::WARP_SIZE. For the two-kernel reductions,
        /// the second kernel has a 1d block of gpu_block_shape.n_elements() threads.
        /// Increasing the block size might increase performance for some large reductions, but it also increases
        /// the size threshold above which the two-kernel reduction is used, so predicting the performance impact
        /// is challenging. This value often goes hand in hand with .gpu_scratch_size.
        Shape<u32, 2> gpu_block_shape{Shape<u32, 2>{1, 512}};

        /// Whether the block can be reshaped (keeping the overall block size) for better performance.
        bool gpu_optimize_block_shape{true};

        /// Sets the minimum number of iterations done by each thread, for each block dimension.
        /// This is only relevant for the two-kernel reductions, i.e., for large reductions. Indeed, single kernel
        /// reductions assign one block per output element, and the block loops through the reduced dimension(s).
        /// Increasing this value decreases the parallelism (the number of blocks launched), but may still be
        /// beneficial for operators doing little work per iteration.
        Vec<u32, 2> gpu_number_of_indices_per_threads{Vec<u32, 2>{1, 1}};

        /// The maximum number of blocks to launch per output element.
        /// This is only relevant for the two-kernel reductions, i.e., for large reductions, as it limits the size of
        /// the temporary buffer that needs to be reduced by the second kernel (with one block per reduced axis).
        /// Reductions without reduced values are an exception as this temporary buffer is elided and the second kernel
        /// is a single thread kernel calling op::post. In this case, it may be good to increase the grid size to
        /// increase the parallelism of the first kernel, but even then the default value is usually fine.
        u32 gpu_max_grid_size{4096};

        /// Allocate the specified number of bytes for the per-block scratch (shared memory in CUDA).
        /// The maximum alignment is std::max_align_t (16 in CUDA), which should be fine for most cases.
        /// The scratch is available from the reduction operators via the compute handle during init/call/deinit
        /// and is destructed after that (join/post).
        usize gpu_scratch_size{0};
    };
}

namespace noa::details {
    template<ReduceIwiseOptions OPTIONS, bool, bool, typename Op, typename Reduced, typename Output, typename I, usize N>
    constexpr void reduce_iwise(const Shape<I, N>&, Device, Op&&, Reduced&&, Output&&);
}

namespace noa {
    /// Dispatches an index-wise reduction operator across N-dimensional (parallel) for-loops.
    /// \param shape:
    ///     Shape of the 1-, 2-, 3- or 4-dimensional loop.
    ///     The index type of the shape defines the index type for the indices.
    /// \param device:
    ///     Device on which to dispatch the reduction.
    ///     When this function returns, the current stream of that device is synchronized.
    /// \param[in] reduced:
    ///     Initial reduction value, an adaptor (see wrap() and fuse()) containing the initial reduction value(s),
    ///     or an empty adaptor. Reductions without reduced values is intended for cases where the operator
    ///     should fully handle the reduction (e.g. to compute partial reductions like with histograms) usually
    ///     by directly manipulating the per-block scratch space (see ReduceIwiseOptions::gpu_scratch_size).
    /// \param[in,out] outputs:
    ///     Output value, an adaptor (see wrap() and fuse()) containing (a reference of) the output value(s),
    ///     or an empty adaptor. When this function returns, the output values will have been updated.
    ///
    /// \param[in] op:
    ///     Operator satisfying the reduce-iwise interface described below.
    ///     The operator is forwarded to the backend and ultimately copied to each compute thread.
    ///     The implementation calls the operator in the following manner:
    ///
    /// ->  op.init(handle, Vec{output-indices...}),
    ///     op.init(handle, output-indices...),
    ///     op.init(handle),
    ///     op.init(Vec{output-indices...}),
    ///     op.init(output-indices...) or
    ///     op.init():
    ///         Defaulted to no-op. If defined, each thread calls it when the reduction starts. Since operators are
    ///         per-thread, this can be used to perform some initialization of the operator or of the per-block scratch,
    ///         similar to iwise/ewise operators. The default no-op can be removed by defining the type Op::remove_default_init.
    ///         Note that for reduce_iwise, output-indices is always zero, but they become more useful for the other
    ///         reductions: reduce_axes_iwise, reduce_ewise, or reduce_axes_ewise.
    ///
    /// ->  op(handle, Vec{input-indices...}, reduced&...),
    ///     op(handle, input-indices..., reduced&...),
    ///     op(Vec{input-indices...}, reduced&...) or
    ///     op(input-indices..., reduced&...):
    ///         Main reduction step, called once per nd-index. The reduced values are initialized (using \p reduced) or
    ///         already joined, and should be updated during this step. An empty adaptor can be passed to \p reduced,
    ///         in which case no reduced values will be passed (an empty fuse adaptor will pass an empty tuple).
    ///
    /// ->  op.deinit(handle, Vec{output-indices...}),
    ///     op.deinit(handle, output-indices...),
    ///     op.deinit(handle),
    ///     op.deinit(Vec{output-indices...}),
    ///     op.deinit(output-indices...) or
    ///     op.deinit():
    ///         Defaulted to no-op. If defined, each thread calls it when its (partial) reduction ends.
    ///         It is the mirror operation of op.init(...) and is often used to save some state or save the per-block
    ///         scratch to a permanent location, similar to iwise- or ewise-operators. Importantly, the scratch becomes
    ///         unavailable after this step. The default no-op can be removed by defining the type Op::remove_default_deinit.
    ///
    /// ->  op.join(const reduced&..., reduced&...):
    ///         To join the reduced values together. This is necessary for multithreaded/GPU reductions. Depending on the
    ///         implementation, this may or may not be called. Similarly to the previous step, if an empty adaptor is
    ///         passed to \p reduced, no reduced values (or two empty tuples) will be passed. In this case, join doesn't
    ///         need to be defined and is defaulted to a no-op.
    ///
    /// ->  op.post(const reduced&..., outputs&..., Vec{output-indices...}),
    ///     op.post(const reduced&..., outputs&..., output-indices...) or
    ///     op.post(const reduced&..., outputs&...):
    ///         Defaulted to copy. If defined, it is called once per output element. This is called after the reduction
    ///         and is meant as a post-processing step to transform the reduced value(s) into output value(s). As always,
    ///         an empty adaptor to \p reduced or \p output can be passed, in which case no reduced/output values
    ///         (or empty tuples) will be passed. If not defined, this defaults to a copy. As such, the following
    ///         expression must be valid: ((outputs = static_cast<Outputs>(reduced)), ...).
    ///         This default copy can be removed by defining the type Op::remove_default_post.
    template<ReduceIwiseOptions OPTIONS = ReduceIwiseOptions{},
             typename Reduced = nd::AdaptorUnzip<>,
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
                      "Output values should be references");

        if constexpr (nd::adaptor_decay<Reduced, Outputs>) {
            nd::reduce_iwise<OPTIONS, std::decay_t<Reduced>::ZIP, std::decay_t<Outputs>::ZIP>(
                shape, device, std::forward<Operator>(op),
                std::forward<Reduced>(reduced).tuple,
                std::forward<Outputs>(outputs).tuple);

        } else if constexpr (nd::adaptor_decay<Reduced>) {
            nd::reduce_iwise<OPTIONS, std::decay_t<Reduced>::ZIP, false>(
                shape, device, std::forward<Operator>(op),
                std::forward<Reduced>(reduced).tuple,
                noa::forward_as_tuple(std::forward<Outputs>(outputs)));

        } else if constexpr (nd::adaptor_decay<Outputs>) {
            nd::reduce_iwise<OPTIONS, false, std::decay_t<Outputs>::ZIP>(
                shape, device, std::forward<Operator>(op),
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                std::forward<Outputs>(outputs).tuple);
        } else {
            nd::reduce_iwise<OPTIONS, false, false>(
                shape, device, std::forward<Operator>(op),
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                noa::forward_as_tuple(std::forward<Outputs>(outputs)));
        }
    }
}

namespace noa::details {
    template<ReduceIwiseOptions OPTIONS, bool ZIP_REDUCED, bool ZIP_OUTPUT,
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

        if constexpr (OPTIONS.generate_cpu) {
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
                    cpu_stream.thread_limit()
                );
                return;
            }
        }

        if constexpr (OPTIONS.generate_gpu) {
            if (device.is_gpu()) {
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
                using config = noa::cuda::ReduceIwiseConfig<
                    ZIP_REDUCED, ZIP_OUTPUT,
                    OPTIONS.gpu_block_shape[1],
                    OPTIONS.gpu_block_shape[0],
                    OPTIONS.gpu_number_of_indices_per_threads[1],
                    OPTIONS.gpu_number_of_indices_per_threads[0],
                    OPTIONS.gpu_max_grid_size,
                    OPTIONS.gpu_optimize_block_shape,
                    OPTIONS.gpu_allow_two_kernels
                >;
                noa::cuda::reduce_iwise<config>(
                    shape, std::forward<Op>(op),
                    std::move(reduced_accessors),
                    output_accessors, cuda_stream,
                    OPTIONS.gpu_scratch_size
                );

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
}
