#pragma once

#include "noa/runtime/Array.hpp"
#include "noa/signal/core/Utils.hpp"

// TODO permute rightmost, preserving groups (B..)/(((D)H)W)

namespace noa::signal::details {
    template<usize B, usize R, Border BORDER,
             i32 WINDOW_WIDTH, // 0 means runtime
             i32 BLOCK_SIZE_X, // 1 for CPU
             i32 BLOCK_SIZE_Y, // 1 for CPU
             nt::sinteger Index,
             nt::scalar Value,
             nt::readable_nd<B + R> Input,
             nt::writable_nd<B + R> Output,
             nt::writable_nd_or_empty<1> Scratch>
    struct MedianFilter {
        static constexpr bool HAS_STATIC_WINDOW_WIDTH = WINDOW_WIDTH > 0;
        static constexpr usize N = B + R;
        using window_type = std::conditional_t<HAS_STATIC_WINDOW_WIDTH, Empty, i32>;
        using height_type = std::conditional_t<(N > 1 and R == 1), Index, Empty>;

        using input_value_type = nt::mutable_value_type_t<Input>;
        using output_value_type = nt::value_type_t<Output>;
        using scratch_value_type = Value;

        constexpr MedianFilter(
            const Input& input,
            const Output& output,
            Scratch scratch,
            const Shape<Index, R>& shape,
            i32 window
        ) : m_input(input),
            m_output(output),
            m_scratch(scratch),
            m_shape(shape)
        {
            if constexpr (not HAS_STATIC_WINDOW_WIDTH) {
                m_window_width = window;
                m_window_size = window;
                if constexpr (R >= 2)
                    m_window_size *= window;
                if constexpr (R == 3)
                    m_window_size *= window;
                m_window_size_half = m_window_size / 2;
            }
        }

        constexpr MedianFilter(
            const Input& input,
            const Output& output,
            const Shape<Index, R>& shape,
            Index height
        ) requires HAS_STATIC_WINDOW_WIDTH :
            m_input(input),
            m_output(output),
            m_shape(shape)
        {
            if constexpr (not nt::empty<height_type>)
                m_height = height;
        }

        constexpr auto window_width() const -> i32 {
            if constexpr (HAS_STATIC_WINDOW_WIDTH)
                return WINDOW_WIDTH;
            else
                return m_window_width;
        }

        constexpr auto halo() const -> Index {
            return static_cast<Index>(window_width() / 2); // assumes odd
        }

        static constexpr auto window_size() -> i32 requires HAS_STATIC_WINDOW_WIDTH {
            auto s = WINDOW_WIDTH;
            if constexpr (R >= 2)
                s *= WINDOW_WIDTH;
            if constexpr (R == 3)
                s *= WINDOW_WIDTH;
            return s;
        }

        constexpr auto window_size() const -> i32 requires (not HAS_STATIC_WINDOW_WIDTH) {
            return m_window_size;
        }

        constexpr auto window_size_half() const -> i32 {
            if constexpr (HAS_STATIC_WINDOW_WIDTH)
                return window_size() / 2;
            else
                return m_window_size_half;
        }

        constexpr auto scratch_width() const -> i32 {
            if constexpr (BLOCK_SIZE_X == 1) {
                return window_width();
            } else {
                return BLOCK_SIZE_X + window_width() - 1;
            }
        }

        constexpr auto scratch_height() const -> i32 {
            if constexpr (BLOCK_SIZE_Y == 1) {
                return window_width();
            } else {
                return BLOCK_SIZE_Y + window_width() - 1;
            }
        }

        template<typename T, typename U>
        NOA_HD constexpr auto load_scratch(
            const T& input_r,
            const Vec<Index, R>& gid,
            const Vec<i32, R>& tid,
            const U& scratch_r
        ) const {
            if constexpr (R == 1) {
                Index gi = gid[R - 1] - halo();
                for (i32 li{tid[R - 1]}; li < scratch_width(); li += BLOCK_SIZE_X, gi += BLOCK_SIZE_X) {
                    if constexpr (BORDER == Border::REFLECT) {
                        const auto gir = index_at<BORDER>(m_shape[R - 1], gi);
                        scratch_r[li] = static_cast<scratch_value_type>(input_r[gir]);
                    } else {
                        scratch_r[li] =
                            gi >= 0 and gi < m_shape[R - 1] ?
                            static_cast<scratch_value_type>(input_r[gi]) : scratch_value_type{};
                    }
                }
            } else if constexpr (R == 2) {
                Index gi = gid[R - 2] - halo();
                for (i32 li{tid[R - 2]}; li < scratch_height(); li += BLOCK_SIZE_Y, gi += BLOCK_SIZE_Y) {
                    Index gj = gid[R - 1] - halo();
                    for (i32 lj{tid[R - 1]}; lj < scratch_width(); lj += BLOCK_SIZE_X, gj += BLOCK_SIZE_X) {
                        if constexpr (BORDER == Border::REFLECT) {
                            const auto gir = index_at<BORDER>(m_shape[R - 2], gi);
                            const auto gjr = index_at<BORDER>(m_shape[R - 1], gj);
                            scratch_r[li * scratch_width() + lj] = static_cast<scratch_value_type>(input_r(gir, gjr));
                        } else {
                            scratch_r[li * scratch_width() + lj] =
                                gi >= 0 and gi < m_shape[R - 2] and
                                gj >= 0 and gj < m_shape[R - 1] ?
                                static_cast<scratch_value_type>(input_r(gi, gj)) : scratch_value_type{};
                        }
                    }
                }
            } else if constexpr (R == 3) {
                Index gi = gid[R - 3] - halo();
                for (i32 li{tid[R - 3]}; li < window_width(); li += 1, gi += 1) {
                    Index gj = gid[R - 2] - halo();
                    for (i32 lj{tid[R - 2]}; lj < scratch_height(); lj += BLOCK_SIZE_Y, gj += BLOCK_SIZE_Y) {
                        Index gk = gid[R - 1] - halo();
                        for (i32 lk{tid[R - 1]}; lk < scratch_width(); lk += BLOCK_SIZE_X, gk += BLOCK_SIZE_X) {
                            if constexpr (BORDER == Border::REFLECT) {
                                const auto gir = index_at<BORDER>(m_shape[R - 3], gi);
                                const auto gjr = index_at<BORDER>(m_shape[R - 2], gj);
                                const auto gkr = index_at<BORDER>(m_shape[R - 1], gk);
                                scratch_r[(li * scratch_height() + lj) * scratch_width() + lk] =
                                    static_cast<scratch_value_type>(input_r(gir, gjr, gkr));
                            } else {
                                scratch_r[(li * scratch_height() + lj) * scratch_width() + lk] =
                                    gi >= 0 and gi < m_shape[R - 3] and
                                    gj >= 0 and gj < m_shape[R - 2] and
                                    gk >= 0 and gk < m_shape[R - 1] ?
                                    static_cast<scratch_value_type>(input_r(gi, gj, gk)) : scratch_value_type{};
                            }
                        }
                    }
                }
            }
        }

        // Ensures a < b. If not, swap.
        template<typename T>
        NOA_FD constexpr void sort_swap(T& a, T& b) const {
            if (b < a) {
                // TODO Use XOR swap?
                T tmp = a;
                a = b;
                b = tmp;
            }
        }

        // Sets the array v, so that the min is at v[0] and max is at v[length-1].
        NOA_FD constexpr void sort_edges(scratch_value_type* v, i32 length) const {
            for (i32 i = 0; i < length / 2; i++) // ensure min in first half, max in second half
                sort_swap(v[i], v[length - 1 - i]);
            for (i32 i = 1; i < (length + 1) / 2; i++) // move min in first half to first pos
                sort_swap(v[0], v[i]);
            for (i32 i = length - 2; i >= length / 2; i--) // move max in second half to last pos
                sort_swap(v[i], v[length - 1]);
        }

        NOA_FD constexpr auto scratch_median(scratch_value_type* scratch, const Vec<i32, 2>& tid) const {
            // Exchange sort.
            // The window is divided into two halves: the first "activated" half, where the sorting happens,
            // and the second half, which is the pool of contestants that will be progressively added.
            constexpr i32 ACTIVE = window_size() / 2 + 2; // the final size should be at least 3.

            // Load active half.
            // Indexes are known at compile time, so hopefully registers are used, otherwise spill to local memory.
            scratch_value_type v[static_cast<usize>(ACTIVE)];
            if constexpr (R == 1) {
                for (i32 x = 0; x < ACTIVE; ++x)
                    v[x] = scratch[tid[1] + x];
            } else if constexpr (R == 2) {
                for (i32 count = 0, y = 0; y < WINDOW_WIDTH; ++y)
                    for (i32 x = 0; count < ACTIVE and x < WINDOW_WIDTH; ++x, ++count)
                        v[count] = scratch[(tid[0] + y) * scratch_width() + tid[1] + x];
            } else {
                // Load active window from shared memory into this 1D array.
                for (i32 count = 0, z = 0; z < WINDOW_WIDTH; ++z)
                    for (i32 y = 0; count < ACTIVE and y < WINDOW_WIDTH; ++y)
                        for (i32 x = 0; count < ACTIVE and x < WINDOW_WIDTH; ++x, ++count)
                            v[count] = scratch[(z * scratch_height() + tid[0] + y) * scratch_width() + tid[1] + x];
            }

            sort_edges(v, ACTIVE); // ensure min at 0, max at ACTIVE - 1

            // We can replace the min by a new contestant and then reorder.
            // We also don't have to compare the max at ACTIVE - 1, we just need to stack the new max at the end.
            // As such, as we add new contestants, we right-truncate the active half to ignore the previous max.
            i32 length = ACTIVE;
            for (i32 k = ACTIVE; k < window_size(); ++k) {
                if constexpr (R == 1) {
                    v[0] = scratch[tid[1] + k]; // replace min by new contestant
                } else if constexpr (R == 2) {
                    const auto coords = noa::offset2index(k, WINDOW_WIDTH);
                    v[0] = scratch[(coords[0] + tid[0]) * scratch_width() + coords[1] + tid[1]];
                } else {
                    const auto coords = noa::offset2index(k, WINDOW_WIDTH, WINDOW_WIDTH);
                    v[0] = scratch[(coords[0] * scratch_height() + coords[1] + tid[0]) * scratch_width() + coords[2] + tid[1]];
                }

                --length; // ignore the previous max at the end
                sort_edges(v, length); // min at 0, max at length - 1
            }

            // Since we know the min and max are at v[0] and v[length-1] respectively, to find the median,
            // we simply need to (exchange) sort the elements from v[1] to v[length-2] included.
            // The median will then be at length/2.
            for (i32 k = 1; k < length - 1; k++)
                sort_edges(v + k, length - k);
            return static_cast<output_value_type>(v[length / 2]);
        }

        NOA_FD void init(nt::compute_handle_gpu auto const&) const {}

        NOA_FD void operator()(nt::compute_handle_gpu auto const& handle, const Vec<Index, B + R>& gid) const requires (R == 1) {
            // If N > 1, we use a 2D block and may be OOB.
            bool is_height_inbound;
            if constexpr (N == 1)
                is_height_inbound = true;
            else
                is_height_inbound = gid[N - 2] < m_height;

            if (is_height_inbound) {
                const auto tid = handle.thread().template id<i32, 2>();
                auto* scratch = handle.block().template scratch_pointer<scratch_value_type>();
                if constexpr (N > 1)
                    scratch += tid[0] * scratch_width();

                const auto [batches, indices] = gid.template split<B>();
                load_scratch(m_input[batches], indices, handle.thread().template id<i32, R>(), scratch);
                handle.block().synchronize();

                if (gid[N - 1] < m_shape[R - 1])
                    m_output[gid] = scratch_median(scratch, tid);
            }
        }

        NOA_FD void operator()(nt::compute_handle_gpu auto const& handle, const Vec<Index, B + R>& gid) const requires (R >= 2) {
            const auto tid = handle.thread().template id<i32, 2>();
            auto* scratch = handle.block().template scratch_pointer<scratch_value_type>();
            if constexpr (R == 1 and N > 1)
                scratch += tid[0] * scratch_width();

            const auto [batches, indices] = gid.template split<B>();
            load_scratch(m_input[batches], indices, handle.thread().template id<i32, R>(), scratch);
            handle.block().synchronize();

            if (gid[N - 2] < m_shape[R - 2] and gid[N - 1] < m_shape[R - 1])
                m_output[gid] = scratch_median(scratch, tid);
        }

    public:
        using remove_default_init = bool;
        void init(nt::compute_handle_cpu auto& handle) noexcept {
            // Before starting the loop, offset to the thread workspace.
            m_scratch = Scratch(m_scratch.get() + handle.block().lid() * window_size());
        }

        void operator()(nt::compute_handle_cpu auto const& handle, const Vec<Index, B + R>& gid) const {
            const auto [batches, indices] = gid.template split<B>();
            const auto input_r = m_input[batches];
            const auto tid = handle.thread().template id<i32, R>();

            load_scratch(input_r, indices, tid, m_scratch);

            // Sort the elements in the scratch to get the median.
            std::nth_element(m_scratch.get(), m_scratch.get() + window_size_half(), m_scratch.get() + window_size());
            m_output[gid] = static_cast<output_value_type>(m_scratch[window_size_half()]);
        }

    private:
        Input m_input;
        Output m_output;
        Scratch m_scratch{};
        Shape<Index, R> m_shape;
        NOA_NO_UNIQUE_ADDRESS window_type m_window_width;
        NOA_NO_UNIQUE_ADDRESS window_type m_window_size;
        NOA_NO_UNIQUE_ADDRESS window_type m_window_size_half;
        NOA_NO_UNIQUE_ADDRESS height_type m_height;
    };

    template<CompileOptions OPTIONS, usize R, Border BORDER, typename Input, typename Index, usize N, typename Output>
    void launch_median_filter(Input&& input, const Strides<Index, N>& input_strides,  Output&& output, i32 window_width) {
        using input_value_t = nt::const_value_type_t<Input>;
        using output_value_t = nt::mutable_value_type_t<Output>;
        constexpr usize B = N - R;
        using input_t = AccessorRestrict<input_value_t, N, Index>;
        using output_t = AccessorRestrict<output_value_t, N, Index>;
        const auto iaccessor = input_t(input.data(), input_strides);
        const auto oaccessor = output_t(output.data(), output.strides());
        const auto shape = output.shape();
        const auto shape_r = shape.template pop_front<B>();
        const auto device = output.device();

        constexpr bool SMALL_TYPE = sizeof(input_value_t) < 4;
        using scratch_value_t =
            std::conditional_t<SMALL_TYPE and nt::sinteger<input_value_t>, i32,
            std::conditional_t<SMALL_TYPE and nt::uinteger<input_value_t>, u32,
            std::conditional_t<SMALL_TYPE and nt::real<input_value_t>, f32,
            nt::mutable_value_type_t<Input>>>>;

        if constexpr (OPTIONS.generate_cpu) {
            if (device.is_cpu()) {
                // TODO For window 3 and 5, use fixed array on the stack of the operator().
                // TODO Is this scratch cache friendly when shared between multiple threads?
                using scratch_t = AccessorRestrictContiguous<scratch_value_t, 1, i32>;
                const i32 max_nb_threads = Stream::current(device).thread_limit();
                const i32 window_size = window_width * (R >= 2 ? window_width : 1) * (R == 3 ? window_width : 1);
                auto scratch = noa::Array<scratch_value_t, 1>(window_size * max_nb_threads);
                auto saccessor = scratch_t(scratch.data());
                constexpr auto COMPILE_OPTIONS = noa::IwiseCompileOptions{.generate_gpu = false};

                #define NOA_LAUNCH_MEDIAN_FILTER_(n)                                                    \
                    noa::iwise<COMPILE_OPTIONS>(shape, device, MedianFilter<                            \
                        B, R, BORDER, n, 1, 1, Index, scratch_value_t, input_t, output_t, scratch_t>(   \
                            iaccessor, oaccessor, saccessor, shape_r, window_width), {},                \
                            NOA_FWD(input), NOA_FWD(output), std::move(scratch));

                switch (window_width) {
                    case 3: return NOA_LAUNCH_MEDIAN_FILTER_(3);
                    case 5: return NOA_LAUNCH_MEDIAN_FILTER_(5);
                    default: return NOA_LAUNCH_MEDIAN_FILTER_(0);
                }
                #undef NOA_LAUNCH_MEDIAN_FILTER_
            }
        } else {
            #ifdef NOA_ENABLE_GPU
            check(device.is_gpu());
            #else
            static_assert(nt::always_false<Input>, "CPU-only builds must generate the CPU codepath");
            #endif
        }

        if constexpr (OPTIONS.generate_gpu) {
            #ifdef NOA_ENABLE_GPU
            // If N > 1, use a 2D block. For N=1 (thus R=1), use a 1D block.
            constexpr u32 WARP_SIZE = noa::gpu::Constant::WARP_SIZE;
            constexpr u32 BLOCK_SIZE_Y = N == 1 ? 1 : WARP_SIZE / 2;
            constexpr u32 BLOCK_SIZE_X = N == 1 ? WARP_SIZE * 4 : WARP_SIZE / 2;
            constexpr auto COMPILE_OPTIONS = noa::IwiseCompileOptions{
                .generate_cpu = false,
                .gpu_block_shape = Shape<u32, 3>{1, BLOCK_SIZE_Y, BLOCK_SIZE_X},
                .gpu_optimize_block_shape = false,
                .gpu_number_of_indices_per_thread = Vec<u32, 3>{1, 1, 1},
                .gpu_enable_block_bound_check = false,
            };

            const auto padding = static_cast<u32>(window_width) - 1;
            const auto iwise_options = noa::IwiseOptions{
                .gpu_scratch_size =
                    static_cast<u32>(sizeof(scratch_value_t)) *
                    (BLOCK_SIZE_X + padding) *
                    (BLOCK_SIZE_Y + (R >= 2 ? padding : 0)) *
                    (R == 3 ? static_cast<u32>(window_width) : 1),
            };

            auto iwise_shape = shape;
            if constexpr (N > 1)
                iwise_shape[N - 2] = next_multiple_of(iwise_shape[N - 2], BLOCK_SIZE_Y);
            iwise_shape[N - 1] = next_multiple_of(iwise_shape[N - 1], BLOCK_SIZE_X);

            Index height{};
            if constexpr (N > 1)
                height = shape[N - 2];

            #define NOA_LAUNCH_MEDIAN_FILTER_(n)                                                        \
                noa::iwise<COMPILE_OPTIONS>(iwise_shape, device, MedianFilter<                          \
                    B, R, BORDER, n, static_cast<i32>(BLOCK_SIZE_X), static_cast<i32>(BLOCK_SIZE_Y),    \
                    Index, scratch_value_t, input_t, output_t, Empty>(                                  \
                        iaccessor, oaccessor, shape_r, height), iwise_options, NOA_FWD(input), NOA_FWD(output))

            switch (window_width) {
                case 3: return NOA_LAUNCH_MEDIAN_FILTER_(3);
                case 5: return NOA_LAUNCH_MEDIAN_FILTER_(5);
                case 7: if constexpr (R <= 2) { return NOA_LAUNCH_MEDIAN_FILTER_(7); }
                case 9: if constexpr (R <= 2) { return NOA_LAUNCH_MEDIAN_FILTER_(9); }
                case 11: if constexpr (R <= 2) { return NOA_LAUNCH_MEDIAN_FILTER_(11); }
                case 13: if constexpr (R == 1) { return NOA_LAUNCH_MEDIAN_FILTER_(13); }
                case 15: if constexpr (R == 1) { return NOA_LAUNCH_MEDIAN_FILTER_(15); }
                case 17: if constexpr (R == 1) { return NOA_LAUNCH_MEDIAN_FILTER_(17); }
                case 19: if constexpr (R == 1) { return NOA_LAUNCH_MEDIAN_FILTER_(19); }
                case 21: if constexpr (R == 1) { return NOA_LAUNCH_MEDIAN_FILTER_(21); }
            }
            #undef NOA_LAUNCH_MEDIAN_FILTER_
            panic();
            #endif
        }
    }

    template<usize RANK, typename Input, typename Output>
    concept median_filterable =
        nt::readable_array_decay_of_scalar<Input> and nt::writable_array_decay_of_scalar<Output> and
        nt::array_size_v<Input> == nt::array_size_v<Output> and nt::array_size_v<Output> >= RANK;
}

namespace noa::signal {
    struct MedianFilterOptions {
        /// Number of elements to consider for the computation of the median. Only odd numbers are supported.
        /// RANK=1: Along the width axis. On the GPU, this is limited to 21.
        /// RANK=2: Along the height and width axes. On the GPU, this is limited to 11.
        /// RANK=3: Along the depth, height and width axes. On the GPU, this is limited to 5.
        i32 window_size;

        /// Border mode used for the "implicit padding".
        /// Either Border::ZERO or Border::REFLECT.
        Border border{Border::REFLECT};
    };

    /// Computes the median filter using a 1D window along the width.
    /// \param[in] input:
    ///     RANK=1: ((Bi..,)W)
    ///     RANK=2: ((Bi..,)H,W)
    ///     RANK=3: ((Bi..,)D,H,W)
    ///     Input array of integers or floating-points to filter.
    ///     Batch axes are broadcast to the output batches.
    /// \param[out] output:
    ///     RANK=1: ((Bo..,)W)
    ///     RANK=2: ((Bo..,)H,W)
    ///     RANK=3: ((Bo..,)D,H,W)
    ///     Output convolved array of integers or floating-points.
    ///     Should not overlap with input.
    /// \param options:
    ///     Filter options.
    template<usize RANK, CompileOptions OPTIONS = CompileOptions{}, typename Input, typename Output>
        requires details::median_filterable<RANK, Input, Output>
    void median_filter(Input&& input, Output&& output, const MedianFilterOptions& options) {
        if (options.window_size <= 1) {
            NOA_FWD(input).to(output);
            return;
        }

        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not are_overlapped(input, output), "The input and output array should not overlap");

        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize B = N - RANK;
        auto [input_shape_b, input_shape_r] = input.shape().template split<B>();
        auto [output_shape_b, output_shape_r] = output.shape().template split<B>();

        // Broadcast input batches.
        auto input_strides = input.strides();
        if constexpr (B >= 1) {
            for (usize i{}; i < B; ++i)
                if (input_shape_b[i] == 1)
                    input_strides[i] = 0;
        }

        // Check input and output shapes are compatible.
        if constexpr (B >= 1) {
            for (usize i{}; i < B; ++i)
                input_shape_b[i] = input_shape_b[i] == 1 ? output_shape_b[i] : input_shape_b[i];
            check(input_shape_b == output_shape_b,
                  "The batch axes are not compatible, input:batches={}, output:batches={}",
                  input_shape_b, output_shape_b);
        }
        check(input_shape_r == output_shape_r,
              "The input and output ranked axes should match, but got input:ranks={}, output:ranks={}",
              input_shape_r, output_shape_r);

        check(is_odd(options.window_size), "Only odd windows are currently supported");

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:{}, output:{}",
              input.device(), device);

        if constexpr (OPTIONS.borders[Border::ZERO]) {
            if (options.border == Border::ZERO) {
                details::launch_median_filter<OPTIONS, RANK, Border::ZERO>(
                    NOA_FWD(input), input_strides, NOA_FWD(output), options.window_size);
                return;
            }
        }
        if constexpr (OPTIONS.borders[Border::REFLECT]) {
            if (options.border == Border::REFLECT) {
                details::launch_median_filter<OPTIONS, RANK, Border::REFLECT>(
                    NOA_FWD(input), input_strides, NOA_FWD(output), options.window_size);
            }
            return;
        }
        panic("The border mode should be {} or {}, but got {}", Border::ZERO, Border::REFLECT, options.border);
    }

    template<CompileOptions OPTIONS = CompileOptions{}, typename Input, typename Output>
    void median_filter_1d(Input&& input, Output&& output, const MedianFilterOptions& options) {
        median_filter<1, OPTIONS>(NOA_FWD(input), NOA_FWD(output), options);
    }
    template<CompileOptions OPTIONS = CompileOptions{}, typename Input, typename Output>
    void median_filter_2d(Input&& input, Output&& output, const MedianFilterOptions& options) {
        median_filter<2, OPTIONS>(NOA_FWD(input), NOA_FWD(output), options);
    }
    template<CompileOptions OPTIONS = CompileOptions{}, typename Input, typename Output>
    void median_filter_3d(Input&& input, Output&& output, const MedianFilterOptions& options) {
        median_filter<3, OPTIONS>(NOA_FWD(input), NOA_FWD(output), options);
    }
}
