#pragma once

#include "noa/base/Bounds.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/Stream.hpp"
#include "noa/runtime/Utils.hpp"

#include "noa/runtime/cpu/ReduceAxesIwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/runtime/cuda/ReduceAxesIwise.cuh"
#endif

#include "noa/runtime/ReduceIwise.hpp"

namespace noa::inline types {
    template<usize N>
    struct ReduceAxes {
        Vec<bool, N> axes;

        constexpr auto reduce_axis(nt::integer auto i) -> ReduceAxes& { axes[i] = true; return *this; }
        constexpr auto keep_axis(nt::integer auto i) -> ReduceAxes& { axes[i] = false; return *this; }

        constexpr auto operator[](nt::integer auto i) const -> const bool& { return axes[i]; }
        constexpr auto operator[](nt::integer auto i) -> bool& { return axes[i]; }

        static constexpr auto all() -> ReduceAxes {
            return {.axes = Vec<bool, N>::from_value(true)};
        }

        static constexpr auto all_but(nt::integer auto i) -> ReduceAxes {
            auto out = all();
            out[i] = false;
            return out;
        }

        static constexpr auto from_vec(const Vec<bool, N>& reduce_axes) -> ReduceAxes {
            return {.axes = reduce_axes};
        }

        template<typename T>
        static constexpr auto from_shape(const Shape<T, N>& output_shape) -> ReduceAxes {
            return from_vec(output_shape.cmp_eq(1));
        }

        template<typename T>
        constexpr auto to_reduced_shape(const Shape<T, N>& input_shape) {
            Shape<T, N> output_shape;
            for (usize i{}; i < N; ++i)
                output_shape[i] = axes[i] ? 1 : input_shape[i];
            return output_shape;
        }
    };
}
namespace noa::traits {
    namespace details {
        template<typename>
        struct IsReduceAxes : std::false_type {};
        template<usize N>
        struct IsReduceAxes<ReduceAxes<N>> : std::true_type {};
    }
    template<typename T>
    concept reduce_axes = details::IsReduceAxes<std::decay_t<T>>::value;
}

namespace noa::details {
    template<ReduceIwiseOptions, bool, bool, bool, typename Index, usize N, typename Reduced, typename Outputs, typename Op, typename... Ts>
    constexpr void reduce_axes_iwise(const Shape<Index, N>&, Device, Reduced&&, Outputs&&, Op&&, ReduceAxes<N>, Ts&&...);
}

namespace noa {
    /// Computes an index-wise reduction along one or multiple axes.
    /// \param shape: Shape of the 1-to-6-dimensional loop.
    /// \param device:
    ///     Device on which to dispatch the reduction. Should match the outputs.
    ///     As opposed to reduce_iwise, this function is asynchronous and does not perform any synchronization.
    ///
    /// \param[in,out] outputs:
    ///     Output array, an adaptor containing the output array(s) (all of which should be on the same device),
    ///     or an empty adaptor. The arrays should have the same number of axes as the input shape. The size of each
    ///     output axis should match the input shape, or be 1, indicating the axis should be reduced.
    ///     There should be at least one axis being reduced. Currently, reducing more than one axis at a time is
    ///     only supported if all the axes are reduced, or all the axes except the leftmost axis.
    ///     If all axes are reduced, it is equivalent to reduce_iwise.
    ///
    /// \param[in] reduced:     Same as reduce-iwise.
    /// \param[in] op:          Same as reduce-iwise.
    /// \param[in] attachments: Same as iwise.
    template<ReduceIwiseOptions OPTIONS = ReduceIwiseOptions{},
             typename Index, usize N,
             typename Reduced = nd::AdaptorUnzip<>,
             typename Outputs = nd::AdaptorUnzip<>,
             typename Operator, typename... Ts>
        requires (not nt::reduce_axes<Outputs>)
    void reduce_axes_iwise(
        const Shape<Index, N>& shape,
        Device device,
        Reduced&& reduced,
        Outputs&& outputs,
        Operator&& op,
        Ts&&... attachments
    ) {
        if constexpr (nd::adaptor_decay<Reduced, Outputs>) {
            nd::reduce_axes_iwise<OPTIONS, std::decay_t<Reduced>::ZIP, std::decay_t<Outputs>::ZIP, false>(
                shape, device,
                std::forward<Reduced>(reduced).tuple,
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op), ReduceAxes<N>{},
                std::forward<Ts>(attachments)...);
        } else if constexpr (nd::adaptor_decay<Reduced>) {
            nd::reduce_axes_iwise<OPTIONS, std::decay_t<Reduced>::ZIP, false, false>(
                shape, device,
                std::forward<Reduced>(reduced).tuple,
                noa::forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op), ReduceAxes<N>{},
                std::forward<Ts>(attachments)...);
        } else if constexpr (nd::adaptor_decay<Outputs>) {
            nd::reduce_axes_iwise<OPTIONS, false, std::decay_t<Outputs>::ZIP, false>(
                shape, device,
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                std::forward<Outputs>(outputs).tuple,
                std::forward<Operator>(op), ReduceAxes<N>{},
                std::forward<Ts>(attachments)...);
        } else {
            nd::reduce_axes_iwise<OPTIONS, false, false, false>(
                shape, device,
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                noa::forward_as_tuple(std::forward<Outputs>(outputs)),
                std::forward<Operator>(op), ReduceAxes<N>{},
                std::forward<Ts>(attachments)...);
        }
    }

    /// Computes an index-wise reduction along one or multiple axes.
    /// \details This overload does not have output array(s), and the axes to reduce are specified explicitly.
    template<ReduceIwiseOptions OPTIONS = ReduceIwiseOptions{},
             typename Index, usize N,
             typename Reduced = nd::AdaptorUnzip<>,
             typename Operator, typename... Ts>
    void reduce_axes_iwise(
        const Shape<Index, N>& shape,
        Device device,
        Reduced&& reduced,
        ReduceAxes<N> reduce_axes,
        Operator&& op,
        Ts&&... attachments
    ) {
        if constexpr (nd::adaptor_decay<Reduced>) {
            nd::reduce_axes_iwise<OPTIONS, std::decay_t<Reduced>::ZIP, false, true>(
                shape, device,
                std::forward<Reduced>(reduced).tuple,
                Tuple{},
                std::forward<Operator>(op), reduce_axes,
                std::forward<Ts>(attachments)...);
        } else {
            nd::reduce_axes_iwise<OPTIONS, false, false, true>(
                shape, device,
                noa::forward_as_tuple(std::forward<Reduced>(reduced)),
                Tuple{},
                std::forward<Operator>(op), reduce_axes,
                std::forward<Ts>(attachments)...);
        }
    }
}

namespace noa::details {
    template<ReduceIwiseOptions OPTIONS, bool ZIP_REDUCED, bool ZIP_OUTPUT, bool ALLOW_NO_OUTPUTS,
             typename Index, usize N, typename Reduced, typename Outputs, typename Op, typename... Ts>
    constexpr void reduce_axes_iwise(
        const Shape<Index, N>& input_shape,
        Device device,
        Reduced&& reduced,
        Outputs&& outputs,
        Op&& reduce_operator,
        ReduceAxes<N> reduce_axes,
        Ts&&... attachments
    ) {
        Shape<Index, N> output_shape;
        if constexpr (ALLOW_NO_OUTPUTS) {
            output_shape = reduce_axes.to_reduced_shape(input_shape);
        } else {
            static_assert(nd::are_all_arrays_nd<Outputs, N>(), "All of the outputs should be arrays with the same number of axes as the input shape");
            static_assert(std::tuple_size_v<Outputs> > 0, "There should be at least one output");

            output_shape = outputs[Tag<0>{}].shape().template as_safe<Index>();
            outputs.for_each_enumerate([&]<usize I, typename T>(T& output) {
                check(device == output.device(),
                      "Output arrays should be on device={}, but got output:{}:device={}",
                      device, I, output.device());
                if constexpr (I > 0) {
                    check(output_shape == output.shape(),
                          "Output arrays should have the same shape, but got output:0:shape={} and output:{}:shape={}",
                          output_shape, I, output.shape());
                }
            });

            const auto axes_to_reduce = input_shape.cmp_ne(output_shape);
            check((axes_to_reduce and output_shape.cmp_ne(1)) == false,
                  "Dimensions should match the input shape, or be 1, indicating the dimension should be reduced to one element. "
                  "Got shape input:shape={}, output:shape={}", input_shape, output_shape);
            check(axes_to_reduce.any_eq(true),
                  "No reduction to compute. Got shape input:shape={}, output:shape={}. Use iwise instead.",
                  input_shape, output_shape);
        }

        Tuple reduced_accessors = nd::to_tuple_of_accessor_values(std::forward<Reduced>(reduced));
        Tuple output_accessors = nd::to_tuple_of_accessors(outputs);

        Stream& stream = Stream::current(device);
        if constexpr (OPTIONS.generate_cpu) {
            if (device.is_cpu()) {
                auto& cpu_stream = stream.cpu();
                auto n_threads = cpu_stream.thread_limit();
                using config = noa::cpu::ReduceAxesIwiseConfig<ZIP_REDUCED, ZIP_OUTPUT>;

                if (cpu_stream.is_sync()) {
                    noa::cpu::reduce_axes_iwise<config>(
                        input_shape, output_shape,
                        std::forward<Op>(reduce_operator),
                        std::move(reduced_accessors),
                        output_accessors,
                        n_threads);
                } else {
                    cpu_stream.enqueue(
                        [=,
                            op = std::forward<Op>(reduce_operator),
                            ra = std::move(reduced_accessors),
                            oh = nd::extract_shared_handle_from_arrays(std::forward<Outputs>(outputs)),
                            ah = nd::extract_shared_handle(noa::forward_as_tuple(std::forward<Ts>(attachments)...))
                        ] {
                            noa::cpu::reduce_axes_iwise<config>(
                                input_shape, output_shape, std::move(op),
                                std::move(ra), output_accessors, n_threads);
                        });
                }
                return;
            }
        }
        if constexpr (OPTIONS.generate_gpu) {
            if (device.is_gpu()) {
                #ifdef NOA_ENABLE_CUDA
                auto& cuda_stream = Stream::current(device).cuda();
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
                noa::cuda::reduce_axes_iwise<config>(
                    input_shape, output_shape,
                    std::forward<Op>(reduce_operator),
                    std::move(reduced_accessors),
                    output_accessors,
                    cuda_stream,
                    OPTIONS.gpu_scratch_size
                );

                // Enqueue the shared handles. See ewise() for more details.
                [&]<usize... O>(std::index_sequence<O...>) {
                    auto oh = nd::extract_shared_handle_from_arrays(std::forward<Outputs>(outputs));
                    cuda_stream.enqueue_attach(std::move(oh)[Tag<O>{}]..., std::forward<Ts>(attachments)...);

                    // Work-around to remove spurious warning of set but unused variable (g++11).
                    if constexpr (sizeof...(O) == 0) (void) oh;
                }(nt::index_list_t<Outputs>{});
                #else
                panic_no_gpu_backend();
                #endif
            }
        }
    }
}
