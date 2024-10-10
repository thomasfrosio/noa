#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/utils/Adaptor.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Utilities.hpp"

#include "noa/cpu/ReduceAxesIwise.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/ReduceAxesIwise.cuh"
#endif

namespace noa::inline types {
    struct ReduceAxes {
        bool batch{};
        bool depth{};
        bool height{};
        bool width{};

        ReduceAxes& reduce_batch(bool flag) { batch = flag; return *this; }
        ReduceAxes& reduce_depth(bool flag) { depth = flag; return *this; }
        ReduceAxes& reduce_height(bool flag) { height = flag; return *this; }
        ReduceAxes& reduce_width(bool flag) { width = flag; return *this; }

        constexpr auto operator[](nt::integer auto i) const -> const bool& {
            switch (i) {
                case 0: return batch;
                case 1: return depth;
                case 2: return height;
                case 3: return width;
                default: panic("invalid index");
            }
        }
        constexpr auto operator[](nt::integer auto i) -> bool& {
            switch (i) {
                case 0: return batch;
                case 1: return depth;
                case 2: return height;
                case 3: return width;
                default: panic("invalid index");
            }
        }

        static constexpr ReduceAxes all() {
            return {true, true, true, true};
        }

        template<size_t N> requires (1 <= N and N <= 4)
        static constexpr ReduceAxes from_vec(const Vec<bool, N>& reduce_axes) {
            ReduceAxes axes{};
            for (size_t i{}; i < N; ++i)
                axes[4 - N + i] = reduce_axes[i];
            return axes;
        }

        template<size_t N>
        static constexpr ReduceAxes from_shape(const Shape<i64, N>& output_shape) {
            return from_vec(output_shape == 1);
        }

        template<typename T, size_t N>
        constexpr auto to_reduced_shape(const Shape<T, N>& input_shape) {
            Shape<T, N> output_shape;
            for (size_t i{}; i < N; ++i)
                output_shape[i] = (*this)[4 - N + i] ? 1 : input_shape[i];
            return output_shape;
        }
    };
}

namespace noa::guts {
    template<bool, bool, bool, typename Index, size_t N, typename Reduced, typename Outputs, typename Op, typename... Ts>
    constexpr void reduce_axes_iwise(const Shape<Index, N>&, Device, Reduced&&, Outputs&&, Op&&, ReduceAxes, Ts&&...);
}

namespace noa {
    /// Computes an index-wise reduction along one or multiple axes.
    /// \details The output axes are mapped from right to left, i.e.:
    ///          N=4 -> BDHW
    ///          N=3 -> DHW  (B   is  ignored)
    ///          N=2 -> HW   (BD  are ignored)
    ///          N=1 -> W    (BDH are ignored)
    ///
    /// \details The size of each output axis should match the input shape, or be 1, indicating the axis
    ///          should be reduced. There should be at least one axis being reduced. Currently, reducing more than
    ///          one axis at a time is only supported if the reduction results to having one value or one value per
    ///          leftmost dimension. Or in other words, if the DHW axes for N=4, or the HW axes for N==3, are empty
    ///          after reduction. As opposed to reduce_iwise, this function is asynchronous and does not perform any
    ///          synchronization. If all axes are reduced, it is otherwise equivalent to reduce_iwise.
    ///
    /// \param shape            Shape of the 1-, 2-, 3- or 4-dimensional loop.
    /// \param device           Device on which to dispatch the reduction. Should match the outputs.
    /// \param[in] reduced      Initial reduction value, or an adaptor containing the initial reduction value(s).
    /// \param[in,out] outputs  Output array, an adaptor containing the output array(s) (all of which should be on the
    ///                         same device), or an empty adaptor.
    /// \param[in] op           Operator satisfying the reduce_(axes_)ewise core interface. The operator is perfectly
    ///                         forwarded to the backend (it is moved or copied to the backend compute kernel).
    ///                         Each compute (CPU or GPU) thread holds a copy of the operator.
    /// \param[in] attachments  Resources to attach to the function call. These are usually Arrays that hold the
    ///                         resources used by the operator, but other attachments can be passed too (see iwise).
    ///                         Note that the output arrays are already attached, so this should only be used to attach
    ///                         additional resources.
    template<typename Index, size_t N,
             typename Reduced = ng::AdaptorUnzip<>,
             typename Outputs = ng::AdaptorUnzip<>,
             typename Operator, typename... Ts>
    void reduce_axes_iwise(
            const Shape<Index, N>& shape,
            Device device,
            Reduced&& reduced,
            Outputs&& outputs,
            Operator&& op,
            Ts&&... attachments
    ) {
        if constexpr (ng::adaptor<Reduced, Outputs>) {
            ng::reduce_axes_iwise<Reduced::ZIP, Outputs::ZIP, false>(
                    shape, device,
                    std::forward<Reduced>(reduced).tuple,
                    std::forward<Outputs>(outputs).tuple,
                    std::forward<Operator>(op), {},
                    std::forward<Ts>(attachments)...);
        } else if constexpr (ng::adaptor<Reduced>) {
            ng::reduce_axes_iwise<Reduced::ZIP, false, false>(
                    shape, device,
                    std::forward<Reduced>(reduced).tuple,
                    forward_as_tuple(std::forward<Outputs>(outputs)),
                    std::forward<Operator>(op), {},
                    std::forward<Ts>(attachments)...);
        } else if constexpr (ng::adaptor<Outputs>) {
            ng::reduce_axes_iwise<false, Outputs::ZIP, false>(
                    shape, device,
                    forward_as_tuple(std::forward<Reduced>(reduced)),
                    std::forward<Outputs>(outputs).tuple,
                    std::forward<Operator>(op), {},
                    std::forward<Ts>(attachments)...);
        } else {
            ng::reduce_axes_iwise<false, false, false>(
                    shape, device,
                    forward_as_tuple(std::forward<Reduced>(reduced)),
                    forward_as_tuple(std::forward<Outputs>(outputs)),
                    std::forward<Operator>(op), {},
                    std::forward<Ts>(attachments)...);
        }
    }

    /// Computes an index-wise reduction along one or multiple axes.
    /// \details This overload does not have output array(s), and the axes to reduce are specified explicitly.
    template<typename Index, size_t N,
             typename Reduced = ng::AdaptorUnzip<>,
             typename Operator, typename... Ts>
    void reduce_axes_iwise(
            const Shape<Index, N>& shape,
            Device device,
            Reduced&& reduced,
            ReduceAxes reduce_axes,
            Operator&& op,
            Ts&&... attachments
    ) {
        if constexpr (ng::adaptor<Reduced>) {
            ng::reduce_axes_iwise<Reduced::ZIP, false, true>(
                    shape, device,
                    std::forward<Reduced>(reduced).tuple,
                    Tuple{},
                    std::forward<Operator>(op), {},
                    std::forward<Ts>(attachments)...);
        } else {
            ng::reduce_axes_iwise<false, false, true>(
                    shape, device,
                    forward_as_tuple(std::forward<Reduced>(reduced)),
                    Tuple{},
                    std::forward<Operator>(op), reduce_axes,
                    std::forward<Ts>(attachments)...);
        }
    }
}

namespace noa::guts {
    template<bool ZIP_REDUCED, bool ZIP_OUTPUT, bool ALLOW_NO_OUTPUTS,
             typename Index, size_t N, typename Reduced, typename Outputs, typename Op, typename... Ts>
    constexpr void reduce_axes_iwise(
            const Shape<Index, N>& input_shape,
            Device device,
            Reduced&& reduced,
            Outputs&& outputs,
            Op&& reduce_operator,
            ReduceAxes reduce_axes,
            Ts&&... attachments
    ) {
        Shape<Index, N> output_shape;

        if constexpr (ALLOW_NO_OUTPUTS) {
            output_shape = reduce_axes.to_reduced_shape(input_shape);
        } else {
            static_assert(ng::are_all_varrays<Outputs>(), "All of the outputs should be varrays");
            static_assert(std::tuple_size_v<Outputs> > 0, "There should be at least one output");

            output_shape = outputs[Tag<0>{}].shape().template as_safe<Index>();
            outputs.for_each_enumerate([&]<size_t I, typename T>(T& output) {
                check(device == output.device(),
                      "Output arrays should be on device={}, but got output:{}:device={}",
                      device, I, output.device());
                if constexpr (I > 0) {
                    check(vall(Equal{}, output_shape, output.shape()),
                          "Output arrays should have the same shape, but got output:0:shape={} and output:{}:shape={}",
                          output_shape, I, output.shape());
                }
            });
        }

        Tuple reduced_accessors = ng::to_tuple_of_accessors(std::forward<Reduced>(reduced));
        Tuple output_accessors = ng::to_tuple_of_accessors(outputs);

        // Backends expect the output accessors to have N dimensions, i.e. we need to remove the leftmost axes.
        constexpr auto filter_nd = []() -> Vec<size_t, N> {
            if constexpr (N == 1)
                return {3};
            else if constexpr (N == 2)
                return {2, 3};
            else if constexpr (N == 3)
                return {1, 2, 3};
            else if constexpr (N == 4)
                return {0, 1, 2, 3};
            else
                static_assert(nt::always_false<>);
        }();
        Tuple output_accessors_nd = ng::reconfig_accessors<ng::AccessorConfig<N>{.filter=filter_nd}>(output_accessors);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            auto n_threads = cpu_stream.thread_limit();
            using config = noa::cpu::ReduceAxesIwiseConfig<ZIP_REDUCED, ZIP_OUTPUT>;

            if (cpu_stream.is_sync()) {
                noa::cpu::reduce_axes_iwise<config>(
                        input_shape, output_shape,
                        std::forward<Op>(reduce_operator),
                        std::move(reduced_accessors),
                        output_accessors_nd,
                        n_threads);
            } else {
                cpu_stream.enqueue(
                        [=,
                         op = std::forward<Op>(reduce_operator),
                         ra = std::move(reduced_accessors),
                         oh = ng::extract_shared_handle_from_arrays(std::forward<Outputs>(outputs)),
                         ah = ng::extract_shared_handle(forward_as_tuple(std::forward<Ts>(attachments)...))
                        ] {
                            noa::cpu::reduce_axes_iwise<config>(
                                    input_shape, output_shape, std::move(op),
                                    std::move(ra), output_accessors_nd, n_threads);
                        });
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = Stream::current(device).cuda();
            using config = noa::cuda::ReduceIwiseConfig<ZIP_REDUCED, ZIP_OUTPUT>;
            noa::cuda::reduce_axes_iwise<config>(
                    input_shape, output_shape,
                    std::forward<Op>(reduce_operator),
                    std::move(reduced_accessors),
                    output_accessors_nd,
                    cuda_stream);

            // Enqueue the shared handles. See ewise() for more details.
            [&]<size_t... O>(std::index_sequence<O...>) {
                auto oh = ng::extract_shared_handle_from_arrays(std::forward<Outputs>(outputs));
                cuda_stream.enqueue_attach(std::move(oh)[Tag<O>{}]..., std::forward<Ts>(attachments)...);

                // Work-around to remove spurious warning of set but unused variable (g++11).
                if constexpr (sizeof...(O) == 0) (void) oh;
            }(nt::index_list_t<Outputs>{});
            #else
            panic("No GPU backend detected");
            #endif
        }
    }
}
#endif
