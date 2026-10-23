#pragma once

#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"

namespace noa::signal {
    struct ConvolveCompileOptions {
        /// Border modes to generate code for.
        BorderSet borders{BorderSet::all()};

        /// Whether CPU code should be generated.
        bool generate_cpu{true};

        /// Whether GPU code should be generated.
        bool generate_gpu{true};
    };
}

namespace noa::signal::details {
    template<usize B, usize R, bool ALONG_X, Border BORDER,
             u32 BLOCK_SIZE_X, // ignored for CPU implementation (assumed 1)
             u32 BLOCK_SIZE_Y, // ignored for CPU implementation (assumed 1)
             nt::integer Index,
             nt::readable_nd<B + R> InputAccessor,
             nt::writable_nd<B + R> OutputAccessor,
             nt::readable_nd<B + 1, 1> FilterAccessor>
    class ConvolutionSeparable {
    public:
        static constexpr usize N = B + R;
        static constexpr bool ALONG_Y = not ALONG_X;
        using input_value_type = nt::mutable_value_type_t<InputAccessor>;
        using output_value_type = nt::value_type_t<OutputAccessor>;
        using filter_value_type = nt::mutable_value_type_t<FilterAccessor>;
        using shape_type = Shape<Index, 1 + (N != 1)>;

    public:
        ConvolutionSeparable(
            const InputAccessor& input,
            const OutputAccessor& output,
            const FilterAccessor& filter,
            const shape_type& shape_hw,
            Index filter_size
        ) : m_input(input), m_output(output), m_filter(filter), m_shape_hw(shape_hw),
            m_halo(filter_size / 2),
            m_filter_size(static_cast<i32>(filter_size)) {}

        NOA_FHD constexpr auto height() const -> Index requires (N > 1) {
            return m_shape_hw[0];
        }

        NOA_FHD constexpr auto width() const -> Index {
            if constexpr (N == 1)
                return m_shape_hw[0];
            else
                return m_shape_hw[1];
        }

        NOA_FHD constexpr auto convolution_axis_size() const -> Index {
            if constexpr (ALONG_Y)
                return height();
            else
                return width();
        }

        NOA_FHD constexpr auto filter(const Vec<Index, N>& gid) const {
            if constexpr (FilterAccessor::SIZE == 1) {
                return m_filter;
            } else {
                return m_filter[gid.template pop_back<R>()]; // 1d
            }
        }

        NOA_FD constexpr void operator()(nt::compute_handle_gpu auto const& handle, const Vec<Index, N>& gid) const requires ALONG_X {
            const auto tid = handle.thread().template id<i32, 2>();
            auto* shared = handle.block().template scratch_pointer<filter_value_type>();

            bool is_valid_row;
            if constexpr (N == 1)
                is_valid_row = true;
            else
                is_valid_row = gid[N - 2] < height();

            // Filter along x.
            if (is_valid_row) {
                const auto input_x = m_input[gid.pop_back()];
                const i32 padding = m_filter_size - 1;
                const i32 shared_len = BLOCK_SIZE_X + padding;
                auto row = shared + tid[0] * shared_len; // focus on current row

                // Load shared memory. Loop to take into account padding.
                i32 lx = tid[1];
                Index gx = gid[N - 1];
                for (; lx < shared_len; lx += BLOCK_SIZE_X, gx += BLOCK_SIZE_X) {
                    const Index ix = gx - m_halo;
                    filter_value_type value{};
                    if constexpr (BORDER == Border::ZERO) {
                        if (ix >= 0 and ix < width())
                            value = static_cast<filter_value_type>(input_x[ix]);
                    } else {
                        const auto ix_reflected = index_at<Border::REFLECT>(width(), ix);
                        value = static_cast<filter_value_type>(input_x[ix_reflected]);
                    }
                    row[lx] = value;
                }
                handle.block().synchronize();

                if (gid[N - 1] < width()) {
                    auto filter_1d = filter(gid);
                    filter_value_type result{};
                    for (i32 idx = 0; idx < m_filter_size; ++idx)
                        result += row[tid[1] + idx] * m_filter[idx];
                    m_output[gid] = static_cast<output_value_type>(result);
                }
            }
        }

        constexpr void operator()(nt::compute_handle_gpu auto const& handle, const Vec<Index, N>& gid) const requires ALONG_Y {
            const auto tid = handle.thread().template id<i32, 2>();
            auto* shared = handle.block().template scratch_pointer<filter_value_type>();

            // Filter along y.
            if (gid[N - 1] < width()) {
                const auto input_yx = m_input[gid.template pop_back<2>()];
                const i32 padding = m_filter_size - 1;
                const i32 shared_len_y = static_cast<i32>(BLOCK_SIZE_Y) + padding;

                i32 ly = tid[0];
                Index gy = gid[N - 2];
                for (; ly < shared_len_y; ly += BLOCK_SIZE_Y, gy += BLOCK_SIZE_Y) {
                    const Index iy = gy - m_halo;

                    filter_value_type value{};
                    if constexpr (BORDER == Border::ZERO) {
                        if (iy >= 0 and iy < height())
                            value = static_cast<filter_value_type>(input_yx(iy, gid[N - 1]));
                    } else {
                        const auto iy_reflected = index_at<Border::REFLECT>(height(), iy);
                        value = static_cast<filter_value_type>(input_yx(iy_reflected, gid[N - 1]));
                    }
                    shared[ly * BLOCK_SIZE_X + tid[1]] = value;
                }
                handle.block().synchronize();

                if (gid[N - 2] < height()) {
                    auto filter_1d = filter(gid);
                    filter_value_type result{};
                    for (i32 idx = 0; idx < m_filter_size; ++idx)
                        result += shared[(tid[0] + idx) * BLOCK_SIZE_X + tid[1]] * filter_1d[idx];
                    m_output[gid] = static_cast<output_value_type>(result);
                }
            }
        }

        constexpr void operator()(nt::compute_handle_cpu auto const&, const Vec<Index, N>& gid) const {
            filter_value_type conv{};
            auto input_indices = gid;
            auto size = convolution_axis_size();
            auto filter_1d = filter(gid);

            constexpr Index I = ALONG_X ? N - 1 : N - 2;
            for (Index i{}; i < static_cast<Index>(m_filter_size); ++i) {
                input_indices[I] = gid[I] - m_halo + i;
                if constexpr (BORDER == Border::ZERO) {
                    if (input_indices[I] >= 0 and input_indices[I] < size)
                        conv += static_cast<filter_value_type>(m_input[input_indices]) * filter_1d[i];
                } else {
                    input_indices[I] = index_at<BORDER>(size, input_indices[I]);
                    conv += static_cast<filter_value_type>(m_input[input_indices]) * filter_1d[i];
                }
            }
            m_output[gid] = static_cast<output_value_type>(conv);
        }

    private:
        InputAccessor m_input;
        OutputAccessor m_output;
        FilterAccessor m_filter;
        shape_type m_shape_hw;
        Index m_halo;
        i32 m_filter_size;
    };

    template<usize R, ConvolveCompileOptions OPTIONS, Border BORDER, usize I,
             typename Input, typename Output, typename Index, usize N, typename Filter,
             typename... Buffers>
    void launch_convolve_separable_axis(
        Input input, Output output, Shape<Index, N> shape, Filter&& filter,
        Buffers&&... buffers
    ) {
        using filter_value_t = nt::const_value_type_t<Filter>;
        using filter_t = AccessorRestrictContiguous<filter_value_t, nt::array_size_v<Filter>, i32>;
        const auto faccessor = filter_t(filter.data(), filter.strides().template as<i32>());
        const auto filter_size = filter.shape().back();
        const auto device = filter.device();

        if constexpr (N >= 3 and I == N - 3) {
            // Move the depth to the height, so that we only convolve along N-1 or N-2.
            shape = shape.swap(N - 3, N - 2);
            input.strides() = input.strides().swap(N - 3, N - 2);
            output.strides() = output.strides().swap(N - 3, N - 2);
        }

        Shape<Index, N == 1 ? 1 : 2> shape_hw;
        if constexpr (N >= 2)
            shape_hw = shape.filter(N - 2, N - 1);
        else
            shape_hw = shape.filter(N - 1);

        constexpr bool ALONG_X = I == N - 1;

        if constexpr (OPTIONS.generate_cpu) {
            if (device.is_cpu()) {
                auto op = ConvolutionSeparable<N - R, R, ALONG_X, BORDER, 1, 1, Index, Input, Output, filter_t>(
                    input, output, faccessor, shape_hw, filter_size);
                constexpr auto IWISE_COMPILE_OPTIONS = IwiseCompileOptions{.generate_gpu = false};
                noa::iwise<IWISE_COMPILE_OPTIONS>(
                    shape, device, op, {}, NOA_FWD(filter), NOA_FWD(buffers)...);
                return;
            }
        } else {
            #ifdef NOA_ENABLE_GPU
            check(device.is_gpu());
            #else
            static_assert(nt::always_false<Input>, "CPU-only builds must generate the CPU codepath");
            #endif
        }

        #ifdef NOA_ENABLE_GPU
        if constexpr (OPTIONS.generate_gpu) {
            constexpr u32 WARP_SIZE = noa::gpu::Constant::WARP_SIZE;
            constexpr u32 BLOCK_SIZE_Y = N == 1 ? 1             : WARP_SIZE / 2;
            constexpr u32 BLOCK_SIZE_X = N == 1 ? WARP_SIZE * 4 : WARP_SIZE / 2;
            constexpr auto IWISE_COMPILE_OPTIONS = IwiseCompileOptions{
                .generate_cpu = false,
                .gpu_block_shape = Shape<u32, 3>{1, BLOCK_SIZE_Y, BLOCK_SIZE_X},
                .gpu_optimize_block_shape = false,
                .gpu_number_of_indices_per_thread = Vec<u32, 3>::from_value(1),
                .gpu_enable_block_bound_check = false,
            };
            const auto iwise_options = noa::IwiseOptions{
                .gpu_scratch_size = I == N - 1 ?
                    (BLOCK_SIZE_X + static_cast<u32>(filter_size - 1)) * BLOCK_SIZE_Y * sizeof(filter_value_t) :
                    (BLOCK_SIZE_Y + static_cast<u32>(filter_size - 1)) * BLOCK_SIZE_X * sizeof(filter_value_t),
            };

            auto op = ConvolutionSeparable<N - R, R, ALONG_X, BORDER, BLOCK_SIZE_X, BLOCK_SIZE_Y, Index, Input, Output, filter_t>(
                    input, output, faccessor, shape_hw, filter_size);

            // Use iwise as a generic kernel launch, launch the grid of blocks and
            // do the bound checking explicitly.
            shape[N - 2] = next_multiple_of(shape[N - 2], WARP_SIZE);
            shape[N - 1] = next_multiple_of(shape[N - 1], WARP_SIZE);
            noa::iwise<IWISE_COMPILE_OPTIONS>(
                shape, device, op, iwise_options, NOA_FWD(filter), NOA_FWD(buffers)...);
        }
        #endif
    }

    template<usize R, ConvolveCompileOptions OPTIONS, Border BORDER,
             typename Input, typename Index, usize N, typename Output,
             typename FilterDepth, typename FilterHeight, typename FilterWidth,
             typename Buffer>
    void launch_convolve_separable(
        Input&& input,
        const Strides<Index, N>& input_strides,
        Output&& output,
        FilterDepth&& filter_depth,
        FilterHeight&& filter_height,
        FilterWidth&& filter_width,
        Buffer&& buffer
    ) {
        if (filter_depth.n_elements() <= 0 or R <= 2)
            filter_depth = nullptr;
        if (filter_height.n_elements() <= 0 or R <= 2)
            filter_height = nullptr;
        if (filter_width.n_elements() <= 0)
            filter_width = nullptr;

        bool has_filter_depth = not filter_depth.is_empty();
        bool has_filter_height = not filter_height.is_empty();
        bool has_filter_width = not filter_width.is_empty();

        // Allocate temp buffer if necessary.
        i32 count = 0;
        if (has_filter_depth)
            count += 1;
        if (has_filter_height)
            count += 1;
        if (has_filter_width)
            count += 1;

        using value_t = nt::mutable_value_type_t<FilterWidth>;
        ArrayView<value_t, N> tmp;
        Array<value_t, N> tmp_buffer;
        if (not buffer.is_empty()) {
            tmp = buffer.view();
        } else if (count > 1) {
            tmp_buffer = Array<value_t, N>(output.shape(), {
                .device = output.device(),
                .allocator = Allocator::ASYNC,
            });
            tmp = tmp_buffer.view();
        }

        using index_t = nt::index_type_t<Output>;
        using input_t = AccessorRestrict<nt::const_value_type_t<Input>, N, index_t>;
        using output_t = AccessorRestrict<nt::value_type_t<Output>, N, index_t>;
        const auto iaccessor = input_t(input.data(), input_strides);
        const auto oaccessor = output_t(output.data(), output.strides());
        const auto taccessor = output_t(tmp.data(), tmp.strides());
        const auto shape = output.shape();

        constexpr auto D = N - 3;
        constexpr auto H = N - 2;
        constexpr auto W = N - 1;
        if constexpr (R == 3) {
            if (has_filter_depth and has_filter_height and has_filter_width) {
                launch_convolve_separable_axis<R, OPTIONS, BORDER, D>(
                    iaccessor, oaccessor, shape, NOA_FWD(filter_depth));
                launch_convolve_separable_axis<R, OPTIONS, BORDER, H>(
                    oaccessor, taccessor, shape, NOA_FWD(filter_height));
                launch_convolve_separable_axis<R, OPTIONS, BORDER, W>(
                    taccessor, oaccessor, shape, NOA_FWD(filter_width),
                    NOA_FWD(input), NOA_FWD(output), NOA_FWD(buffer), std::move(tmp_buffer));
                return;
            }
            if (has_filter_depth and has_filter_height) {
                launch_convolve_separable_axis<R, OPTIONS, BORDER, D>(
                    iaccessor, taccessor, shape, NOA_FWD(filter_depth));
                launch_convolve_separable_axis<R, OPTIONS, BORDER, H>(
                    taccessor, oaccessor, shape, NOA_FWD(filter_height),
                    NOA_FWD(input), NOA_FWD(output), NOA_FWD(buffer), std::move(tmp_buffer));
                return;
            }
            if (has_filter_depth and has_filter_width) {
                launch_convolve_separable_axis<R, OPTIONS, BORDER, D>(
                    iaccessor, taccessor, shape, NOA_FWD(filter_depth));
                launch_convolve_separable_axis<R, OPTIONS, BORDER, W>(
                    taccessor, oaccessor, shape, NOA_FWD(filter_width),
                    NOA_FWD(input), NOA_FWD(output), NOA_FWD(buffer), std::move(tmp_buffer));
                return;
            }
            if (has_filter_depth) {
                return launch_convolve_separable_axis<R, OPTIONS, BORDER, D>(
                    iaccessor, oaccessor, shape, NOA_FWD(filter_depth),
                    NOA_FWD(input), NOA_FWD(output), NOA_FWD(buffer), std::move(tmp_buffer));
            }
        }

        if constexpr (R >= 2) {
            if (has_filter_height and has_filter_width) {
                launch_convolve_separable_axis<R, OPTIONS, BORDER, H>(
                    iaccessor, taccessor, shape, NOA_FWD(filter_height));
                launch_convolve_separable_axis<R, OPTIONS, BORDER, W>(
                    taccessor, oaccessor, shape, NOA_FWD(filter_width),
                    NOA_FWD(input), NOA_FWD(output), NOA_FWD(buffer), std::move(tmp_buffer));
                return;
            }
            if (has_filter_height) {
                return launch_convolve_separable_axis<R, OPTIONS, BORDER, H>(
                    iaccessor, oaccessor, shape, NOA_FWD(filter_height),
                    NOA_FWD(input), NOA_FWD(output), NOA_FWD(buffer), std::move(tmp_buffer));
            }
        }

        if (has_filter_width)
            return launch_convolve_separable_axis<R, OPTIONS, BORDER, W>(
                iaccessor, oaccessor, shape, NOA_FWD(filter_width),
                NOA_FWD(input), NOA_FWD(output), NOA_FWD(buffer), std::move(tmp_buffer));
    }

    template<typename T, typename I, usize B>
    auto check_separable_filter(
        const T& filter, const Shape<I, B>& output_shape_b, Device device,
        std::source_location location = std::source_location::current()
    ) {
        if (filter.is_empty())
            return;
        check_at_location(
            location, filter.strides().back() == 1 and is_odd(filter.shape().back()),
            "The input filters should be contiguous vectors with an odd number of elements, but got a filter:shape={} and filter:strides={}",
            filter.shape(), filter.strides());
        check_at_location(
            location, filter.device() == device,
            "The input filters must be on the same device as the compute device, but got output:device={}, filter:device={}",
            device, filter.device());
        if constexpr (nt::array_size_v<T> > 1) {
            check(
                location, filter.shape().pop_back() == output_shape_b,
                "The batch axes are not compatible, filter:batches={}, output:batches={}",
                filter.shape().pop_back(), output_shape_b);
        }
    }

    template<
        usize RANK,
        typename Input, typename Output, typename FilterDepth, typename FilterHeight, typename FilterWidth, typename Buffer,
        typename InputValue = nt::value_type_t<Input>,
        typename OutputValue = nt::value_type_t<Output>,
        typename FilterDepthValue = nt::value_type_t<FilterDepth>,
        typename FilterHeightValue = nt::value_type_t<FilterHeight>,
        typename FilterWidthValue = nt::value_type_t<FilterWidth>,
        typename BufferValue = nt::value_type_t<Buffer>,
        usize B = nt::array_size_v<Input> - RANK,
        usize FILTER_DEPTH_N = nt::array_size_v<FilterDepth>,
        usize FILTER_HEIGHT_N = nt::array_size_v<FilterHeight>,
        usize FILTER_WIDTH_N = nt::array_size_v<FilterWidth>>
    concept convolve_separable_able =
        nt::array_decay_with_same_nd<Input, Output, Buffer> and nt::array_size_v<Input> >= RANK and
        nt::array_decay<FilterDepth, FilterHeight, FilterWidth> and
        (FILTER_DEPTH_N == 1 or FILTER_DEPTH_N == B) and
        (FILTER_HEIGHT_N == 1 or FILTER_HEIGHT_N == B) and
        (FILTER_WIDTH_N == 1 or FILTER_WIDTH_N == B) and
        nt::same_as<OutputValue, BufferValue> and not std::is_const_v<OutputValue> and
        nt::scalar<InputValue, OutputValue, FilterDepthValue, FilterHeightValue, FilterWidthValue>;
}

namespace noa::signal::details {
    template<usize B, usize R, Border BORDER,
             usize BLOCK_SIZE_X, // ignored for CPU implementation (assumed 1)
             usize BLOCK_SIZE_Y, // ignored for CPU implementation (assumed 1)
             usize FILTER_LENGTH, // 0 is runtime
             nt::readable_nd<B + R> Input,
             nt::writable_nd<B + R> Output,
             nt::readable_nd<B + 1, 1> Filter>
    class Convolution {
    public:
        static constexpr usize N = B + R;
        using shape_type = Shape<isize, R>;
        using input_value_type = nt::mutable_value_type_t<Input>;
        using output_value_type = nt::value_type_t<Output>;
        using filter_value_type = nt::mutable_value_type_t<Filter>;

        static constexpr bool STATIC_FILTER_LENGTH = FILTER_LENGTH > 0;
        static constexpr bool HAS_SCRATCH = BLOCK_SIZE_X > 1;
        using filter_shape_type = std::conditional_t<STATIC_FILTER_LENGTH, Empty, Shape<i32, R>>;
        using halo_shape_type = filter_shape_type;
        using scratch_shape_type = std::conditional_t<HAS_SCRATCH, Shape<i32, R>, Empty>;

    public:
        Convolution(
            const Input& input,
            const Output& output,
            const Filter& filter,
            const shape_type& shape,
            const Shape<i32, R>& filter_shape
        ) : m_input(input), m_output(output), m_filter(filter), m_shape(shape)
        {
            if constexpr (not STATIC_FILTER_LENGTH) {
                m_filter_shape = filter_shape;
                m_halo = filter_shape / 2;
            }
            if constexpr (HAS_SCRATCH) {
                if constexpr (R == 3)
                    m_scratch_shape[0] = filter_shape[0];
                m_scratch_shape[R - 2] = static_cast<i32>(BLOCK_SIZE_Y) + filter_shape[R - 2] - 1;
                m_scratch_shape[R - 1] = static_cast<i32>(BLOCK_SIZE_X) + filter_shape[R - 1] - 1;
            }
        }

        constexpr auto halo(usize i) const -> isize {
            if constexpr (STATIC_FILTER_LENGTH) {
                return static_cast<isize>(FILTER_LENGTH / 2);
            } else {
                return static_cast<isize>(m_halo[i]);
            }
        }

        template<usize I>
        constexpr auto scratch_size() const -> i32 requires HAS_SCRATCH {
            if constexpr (STATIC_FILTER_LENGTH) {
                if constexpr (I == 0 and R == 3) {
                    return static_cast<i32>(FILTER_LENGTH);
                } else if constexpr ((I == 0 and R == 2) or (I == 1 and R == 3)) {
                    return static_cast<i32>(BLOCK_SIZE_Y + FILTER_LENGTH - 1);
                } else if constexpr ((I == 1 and R == 2) or (I == 2 and R == 3)) {
                    return static_cast<i32>(BLOCK_SIZE_X + FILTER_LENGTH - 1);
                } else {
                    static_assert(nt::always_false<Output>);
                }
            } else {
                return m_scratch_shape[I];
            }
        }

        template<typename I>
        constexpr auto filter_shape(usize i) const {
            if constexpr (STATIC_FILTER_LENGTH) {
                return static_cast<I>(FILTER_LENGTH);
            } else {
                return static_cast<I>(m_filter_shape[i]);
            }
        }

        constexpr auto filter(const Vec<isize, B + R>& gid) const {
            if constexpr (Filter::SIZE == 1) {
                return m_filter;
            } else {
                return m_filter[gid.template pop_back<R>()];
            }
        }

        constexpr void operator()(nt::compute_handle_gpu auto& handle, const Vec<isize, B + R>& gid) const requires (R == 2) {
            const auto tid = handle.thread().template id<i32, 2>();
            const auto input_2d = m_input[gid.template pop_back<2>()];
            auto* shared = handle.block().template scratch_pointer<filter_value_type>();

            // Load the input convolution window into the scratch.
            // Because of the halo, we need to loop and handle OOB cases.
            i32 ly, lx;
            isize gy, gx;
            for (ly = tid[0], gy = gid[N - 2]; ly < scratch_size<0>(); ly += BLOCK_SIZE_Y, gy += BLOCK_SIZE_Y) {
                for (lx = tid[1], gx = gid[N - 1]; lx < scratch_size<1>(); lx += BLOCK_SIZE_X, gx += BLOCK_SIZE_X) {
                    const isize iy = gy - halo(0);
                    const isize ix = gx - halo(1);

                    filter_value_type value{};
                    if constexpr (BORDER == Border::ZERO) {
                        if (iy >= 0 and iy < m_shape[0] and ix >= 0 and ix < m_shape[1])
                            value = static_cast<filter_value_type>(input_2d(iy, ix));
                    } else {
                        const auto iy_reflected = index_at<Border::REFLECT>(m_shape[0], iy);
                        const auto ix_reflected = index_at<Border::REFLECT>(m_shape[1], ix);
                        value = static_cast<filter_value_type>(input_2d(iy_reflected, ix_reflected));
                    }
                    shared[ly * scratch_size<1>() + lx] = value;
                }
            }
            handle.block().synchronize();

            // Convolve.
            if (gid[N - 2] < m_shape[0] and gid[N - 1] < m_shape[1]) {
                auto filter_1d = filter(gid);
                filter_value_type result{};
                for (i32 y = 0; y < filter_shape<i32>(0); ++y)
                    for (i32 x = 0; x < filter_shape<i32>(1); ++x)
                        result += shared[(tid[0] + y) * scratch_size<1>() + tid[1] + x] * filter_1d[y * filter_shape<i32>(1) + x];
                m_output[gid] = static_cast<output_value_type>(result);
            }
        }

        constexpr void operator()(nt::compute_handle_gpu auto& handle, const Vec<isize, B + R>& gid) const requires (R == 3) {
            const auto tid = handle.thread().template id<i32, 2>();
            const auto input_3d = m_input[gid.template pop_back<3>()];
            auto* shared = handle.block().template scratch_pointer<filter_value_type>();

            // Load the input convolution window into the scratch.
            // Because of the halo, we need to loop and handle OOB cases.
            i32 lz, ly, lx;
            isize gz, gy, gx;
            for (lz = 0, gz = gid[N - 3]; lz < scratch_size<0>(); ++lz, ++gz) {
                for (ly = tid[0], gy = gid[N - 2]; ly < scratch_size<1>(); ly += BLOCK_SIZE_Y, gy += BLOCK_SIZE_Y) {
                    for (lx = tid[1], gx = gid[N - 1]; lx < scratch_size<2>(); lx += BLOCK_SIZE_X, gx += BLOCK_SIZE_X) {
                        const isize iz = gz - halo(0);
                        const isize iy = gy - halo(1);
                        const isize ix = gx - halo(2);

                        filter_value_type value{};
                        if constexpr (BORDER == Border::ZERO) {
                            if (is_inbound(m_shape, iz, iy, ix))
                                value = static_cast<filter_value_type>(input_3d(iz, iy, ix));
                        } else {
                            const auto iz_reflected = noa::index_at<Border::REFLECT>(m_shape[0], iz);
                            const auto iy_reflected = noa::index_at<Border::REFLECT>(m_shape[1], iy);
                            const auto ix_reflected = noa::index_at<Border::REFLECT>(m_shape[2], ix);
                            value = static_cast<filter_value_type>(input_3d(iz_reflected, iy_reflected, ix_reflected));
                        }
                        shared[(lz * scratch_size<1>() + ly) * scratch_size<2>() + lx] = value;
                    }
                }
            }
            handle.block().synchronize();

            // Convolve.
            if (gid[N - 2] < m_shape[1] and gid[N - 1] < m_shape[2]) {
                auto filter_1d = filter(gid);
                filter_value_type result{};
                for (i32 z = 0; z < filter_shape<i32>(0); ++z)
                    for (i32 y = 0; y < filter_shape<i32>(1); ++y)
                        for (i32 x = 0; x < filter_shape<i32>(2); ++x)
                            result += shared[(z * scratch_size<1>() + tid[0] + y) * scratch_size<2>() + tid[1] + x] *
                                filter_1d[(z * filter_shape<i32>(1) + y) * filter_shape<i32>(2) + x];
                m_output[gid] = static_cast<output_value_type>(result);
            }
        }

        constexpr void operator()(nt::compute_handle_cpu auto&, const Vec<isize, B + R>& gid) const {
            const auto [batches, indices] = gid.template split<B>();
            auto input_r = m_input[batches];
            auto filter_1d = filter(gid);

            filter_value_type conv{};
            if constexpr (R == 2) {
                if constexpr (BORDER == Border::ZERO) {
                    for (isize wk{}; wk < filter_shape<isize>(0); ++wk) {
                        const isize ik = indices[0] - halo(0) + wk;
                        if (ik < 0 or ik >= m_shape[0])
                            continue;
                        const auto filter_l = filter_1d.data() + filter_shape<isize>(1) * wk;
                        for (isize wl{}; wl < filter_shape<isize>(1); ++wl) {
                            const isize il = indices[1] - halo(1) + wl;
                            if (il >= 0 and il < m_shape[1])
                                conv += static_cast<filter_value_type>(input_r(ik, il)) * filter_l[wl];
                        }
                    }
                } else {
                    for (isize wk{}; wk < filter_shape<isize>(0); ++wk) {
                        const isize ik = index_at<BORDER>(m_shape[0], indices[0] - halo(0) + wk);
                        const auto filter_l = filter_1d.data() + filter_shape<isize>(1) * wk;
                        for (isize wl{}; wl < filter_shape<isize>(1); ++wl) {
                            const isize il = index_at<BORDER>(m_shape[1], indices[1] - halo(1) + wl);
                            conv += static_cast<filter_value_type>(input_r(ik, il)) * filter_l[wl];
                        }
                    }
                }
            } else if constexpr (R == 3) {
                if constexpr (BORDER == Border::ZERO) {
                    for (isize wj{}; wj < filter_shape<isize>(0); ++wj) {
                        const isize ij = indices[0] - halo(0) + wj;
                        if (ij < 0 or ij >= m_shape[0])
                            continue;
                        const auto filter_k = filter_1d.data() + filter_shape<isize>(1) * filter_shape<isize>(2) * wj;
                        for (isize wk{}; wk < filter_shape<isize>(1); ++wk) {
                            const isize ik = indices[1] - halo(1) + wk;
                            if (ik < 0 or ik >= m_shape[1])
                                continue;
                            const auto filter_l = filter_k + filter_shape<isize>(2) * wk;
                            for (isize wl{}; wl < filter_shape<isize>(2); ++wl) {
                                const isize il = indices[2] - halo(2) + wl;
                                if (il >= 0 and il < m_shape[2])
                                    conv += static_cast<filter_value_type>(input_r(ij, ik, il)) * filter_l[wl];
                            }
                        }
                    }
                } else {
                    for (isize wj{}; wj < filter_shape<isize>(0); ++wj) {
                        const isize ij = index_at<BORDER>(m_shape[0], indices[0] - halo(0) + wj);
                        const auto filter_k = filter_1d.data() + filter_shape<isize>(1) * filter_shape<isize>(2) * wj;
                        for (isize wk{}; wk < filter_shape<isize>(1); ++wk) {
                            const isize ik = index_at<BORDER>(m_shape[1], indices[1] - halo(1) + wk);
                            const auto filter_l = filter_k + filter_shape<isize>(2) * wk;
                            for (isize wl{}; wl < filter_shape<isize>(2); ++wl) {
                                const isize il = index_at<BORDER>(m_shape[2], indices[2] - halo(2) + wl);
                                conv += static_cast<filter_value_type>(input_r(ij, ik, il)) * filter_l[wl];
                            }
                        }
                    }
                }
            } else {
                static_assert(nt::always_false<output_value_type>);
            }
            m_output[gid] = static_cast<output_value_type>(conv);
        }

    private:
        Input m_input;
        Output m_output;
        Filter m_filter;
        shape_type m_shape;
        NOA_NO_UNIQUE_ADDRESS filter_shape_type m_filter_shape{};
        NOA_NO_UNIQUE_ADDRESS halo_shape_type m_halo{};
        NOA_NO_UNIQUE_ADDRESS scratch_shape_type m_scratch_shape{};
    };

    template<usize R, ConvolveCompileOptions OPTIONS, Border BORDER,
             typename Input, typename Index, usize N, typename Output, typename Filter>
    void launch_convolve(Input&& input, const Strides<Index, N>& input_strides, Output&& output, Filter&& filter) {
        constexpr usize B = N - R;

        using input_t = AccessorRestrict<nt::const_value_type_t<Input>, N, Index>;
        using output_t = AccessorRestrict<nt::value_type_t<Output>, N, Index>;
        const auto iaccessor = input_t(input.data(), input_strides);
        const auto oaccessor = output_t(output.data(), output.strides());
        const auto shape = output.shape();

        constexpr auto FILTER_N = nt::array_size_v<Filter>;
        constexpr auto FILTER_B = FILTER_N - R;
        constexpr auto FILTER_BATCH_AXES = Vec<usize, FILTER_B>::arange();

        // If there's only one dimension to convolve, use the separable convolution kernels.
        const auto filter_shape = filter.shape().template pop_front<FILTER_B>().template as<i32>();
        const auto n_dimensions_to_convolve = sum(filter_shape.cmp_gt(1));
        if (n_dimensions_to_convolve == 1) {
            if constexpr (R == 3) {
                if (filter_shape[R - 3] > 1) {
                    constexpr auto FILTER_AXES = FILTER_BATCH_AXES.push_back(FILTER_N - 3);
                    details::launch_convolve_separable_axis<R, OPTIONS, BORDER, N - 3>(
                        iaccessor, oaccessor, shape, NOA_FWD(filter).filter(FILTER_AXES),
                        NOA_FWD(input), NOA_FWD(output));
                    return;
                }
            }
            if constexpr (R >= 2) {
                if (filter_shape[R - 2] > 1) {
                    constexpr auto FILTER_AXES = FILTER_BATCH_AXES.push_back(FILTER_N - 2);
                    details::launch_convolve_separable_axis<R, OPTIONS, BORDER, N - 2>(
                        iaccessor, oaccessor, shape, NOA_FWD(filter).filter(FILTER_AXES),
                        NOA_FWD(input), NOA_FWD(output));
                    return;
                }
            }
            constexpr auto FILTER_AXES = FILTER_BATCH_AXES.push_back(FILTER_N - 1);
            details::launch_convolve_separable_axis<R, OPTIONS, BORDER, N - 1>(
                iaccessor, oaccessor, shape, NOA_FWD(filter).filter(FILTER_AXES),
                NOA_FWD(input), NOA_FWD(output));
            return;
        }

        if constexpr (R >= 2) {
            // Assume the filter is contiguous. Since we may know the rank stride as compile time,
            // only store the equivalent of a pointer (+ the batch strides, if any).
            using filter_value_t = nt::const_value_type_t<Filter>;
            using filter_t = AccessorRestrictContiguous<filter_value_t, FILTER_B + 1, i32>;
            const auto faccessor = filter_t(filter.data(), filter.strides().template pop_back<R>().template as<i32>());

            const auto shape_rd = output.shape().template pop_front<B>();
            if constexpr (OPTIONS.generate_cpu) {
                if (output.device().is_cpu()) {
                    constexpr auto COMPILE_OPTIONS = noa::IwiseCompileOptions{.generate_gpu = false};
                    if (filter_shape == 3) {
                        auto op = details::Convolution<B, R, BORDER, 1, 1, 3, input_t, output_t, filter_t>{
                            iaccessor, oaccessor, faccessor, shape_rd, filter_shape};
                        noa::iwise<COMPILE_OPTIONS>(
                            output.shape(), output.device(), op, {},
                            NOA_FWD(input), NOA_FWD(output), NOA_FWD(filter)
                        );
                    } else if (filter_shape == 5) {
                        auto op = details::Convolution<B, R, BORDER, 1, 1, 5, input_t, output_t, filter_t>{
                            iaccessor, oaccessor, faccessor, shape_rd, filter_shape};
                        noa::iwise<COMPILE_OPTIONS>(
                            output.shape(), output.device(), op, {},
                            NOA_FWD(input), NOA_FWD(output), NOA_FWD(filter)
                        );
                    } else {
                        auto op = details::Convolution<B, R, BORDER, 1, 1, 0, input_t, output_t, filter_t>{
                            iaccessor, oaccessor, faccessor, shape_rd, filter_shape};
                        noa::iwise<COMPILE_OPTIONS>(
                            output.shape(), output.device(), op, {},
                            NOA_FWD(input), NOA_FWD(output), NOA_FWD(filter)
                        );
                    }
                    return;
                }
            } else {
                #ifdef NOA_ENABLE_GPU
                check(output.device().is_gpu());
                #else
                static_assert(nt::always_false<Input>, "CPU-only builds must generate the CPU codepath");
                #endif
            }

            #ifdef NOA_ENABLE_GPU
            if constexpr (OPTIONS.generate_gpu) {
                constexpr u32 WARP_SIZE = noa::gpu::Constant::WARP_SIZE;
                constexpr auto BLOCK_SIZE = WARP_SIZE / 2;
                constexpr auto COMPILE_OPTIONS = noa::IwiseCompileOptions{
                    .generate_cpu = false,
                    .gpu_block_shape = Shape<u32, 3>{1, BLOCK_SIZE, BLOCK_SIZE},
                    .gpu_optimize_block_shape = false,
                    .gpu_number_of_indices_per_thread = Vec<u32, 3>{1, 1, 1},
                    .gpu_enable_block_bound_check = false, // kernels handle it
                };

                const auto iwise_options = noa::IwiseOptions{
                    .gpu_scratch_size =
                        static_cast<u32>(sizeof(filter_value_t)) *
                        (BLOCK_SIZE + static_cast<u32>(filter_shape[R - 1]) - 1) *
                        (BLOCK_SIZE + static_cast<u32>(filter_shape[R - 2]) - 1) *
                        (R == 2 ? 1 : static_cast<u32>(filter_shape[R - 3])),
                };

                // Use iwise as a generic kernel launch, launch the grid of blocks and
                // do the bound checking explicitly.
                auto iwise_shape = output.shape().template as<Index>();
                iwise_shape[N - 2] = next_multiple_of(iwise_shape[N - 2], WARP_SIZE);
                iwise_shape[N - 1] = next_multiple_of(iwise_shape[N - 1], WARP_SIZE);

                if (filter_shape == 3) {
                    auto op = details::Convolution<B, R, BORDER, BLOCK_SIZE, BLOCK_SIZE, 3, input_t, output_t, filter_t>{
                        iaccessor, oaccessor, faccessor, shape_rd, filter_shape
                    };
                    noa::iwise<COMPILE_OPTIONS>(
                        iwise_shape, output.device(), op, iwise_options,
                        NOA_FWD(input), NOA_FWD(output), NOA_FWD(filter)
                    );
                } else if (filter_shape == 5) {
                    auto op = details::Convolution<B, R, BORDER, BLOCK_SIZE, BLOCK_SIZE, 5, input_t, output_t, filter_t>{
                        iaccessor, oaccessor, faccessor, shape_rd, filter_shape
                    };
                    noa::iwise<COMPILE_OPTIONS>(
                        iwise_shape, output.device(), op, iwise_options,
                        NOA_FWD(input), NOA_FWD(output), NOA_FWD(filter)
                    );
                } else {
                    auto op = details::Convolution<B, R, BORDER, BLOCK_SIZE, BLOCK_SIZE, 0, input_t, output_t, filter_t>{
                        iaccessor, oaccessor, faccessor, shape_rd, filter_shape
                    };
                    noa::iwise<COMPILE_OPTIONS>(
                        iwise_shape, output.device(), op, iwise_options,
                        NOA_FWD(input), NOA_FWD(output), NOA_FWD(filter)
                    );
                }
            }
            #endif
        }
    }

    template<usize R, typename Input, typename Output, typename Filter,
             usize N = nt::array_size_v<Output>,
             usize FILTER_N = nt::array_size_v<Filter>,
             usize FILTER_B = FILTER_N - R>
    concept convolveable =
        nt::array_decay_of_scalar<Input, Output, Filter> and
        not std::is_const_v<nt::value_type_t<Output>> and
        nt::array_size_v<Input> == N and
        (R == 1 or R == 2 or R == 3) and
        FILTER_N >= R and (FILTER_B == N - R or FILTER_B == 0);
}

namespace noa::signal {
    struct ConvolveOptions {
        /// Border mode used for the convolution.
        /// Only Border::ZERO and Border::REFLECT are currently supported.
        Border border{Border::ZERO};

        // TODO Add kernel stride.
    };

    /// Separable convolutions.
    /// \tparam RANK:
    ///     Rank of the input and output arrays.
    /// \param[in] input:
    ///     RANK=1: ((Bi..,)W)
    ///     RANK=2: ((Bi..,)H,W)
    ///     RANK=3: ((Bi..,)D,H,W)
    ///     Input array of integers or floating-points to convolve.
    ///     Batch axes are broadcast to the output batches.
    /// \param[out] output:
    ///     RANK=1: ((Bo..,)W)
    ///     RANK=2: ((Bo..,)H,W)
    ///     RANK=3: ((Bo..,)D,H,W)
    ///     Output convolved array of integers or floating-points.
    ///     Should not overlap with input.
    /// \param[in] filter_depth, filter_height, filter_width:
    ///     filter_depth:  (D) or ((Bo..,)D)
    ///     filter_height: (H) or ((Bo..,)H)
    ///     filter_width:  (W) or ((Bo..,)W)
    ///     1D filters with an odd number of elements to apply along the D, H, and W axis, respectively.
    ///     Filters can be empty, in which case, the convolution in the corresponding dimension is not applied.
    ///     Each filter can be unbatched, or batched like the output array.
    ///     The input is convolved with:
    ///         1. the filter_depth (if RANK == 3),
    ///         2. then filter_height (if RANK >= 2),
    ///         3. then filter_width.
    ///     Filters can be equal to each other, can have different integer or floating-point types.
    ///     The input values are cast to the value types of the filter before convolution,
    ///     and the result of the convolution is cast to the output type.
    /// \param[out] buffer:
    ///     Temporary array.
    ///     If only one dimension is filtered, no temporary buffer is needed and this is ignored.
    ///     Otherwise, it should be an array of the same shape as the output, or be an empty array,
    ///     in which case a temporary array will be allocated internally.
    /// \param options:
    ///     Convolution options.
    template<usize RANK = 3, ConvolveCompileOptions OPTIONS = ConvolveCompileOptions{},
             typename Input, typename Output,
             typename FilterDepth = ArrayView<nt::const_value_type_t<Input>, 1>,
             typename FilterHeight = ArrayView<nt::const_value_type_t<Input>, 1>,
             typename FilterWidth = ArrayView<nt::const_value_type_t<Input>, 1>,
             typename Buffer = ArrayView<nt::value_type_t<Output>, nt::array_size_v<Output>>>
        requires details::convolve_separable_able<RANK, Input, Output, FilterDepth, FilterHeight, FilterWidth, Buffer>
    void convolve_separable(
        Input&& input,
        Output&& output,
        FilterDepth&& filter_depth,
        FilterHeight&& filter_height,
        FilterWidth&& filter_width,
        Buffer&& buffer = Buffer{},
        const ConvolveOptions& options = {}
    ) {
        check(nd::are_arrays_valid(input, output), "Empty array detected");
        check(not are_overlapped(input, output), "The input and output array should not overlap");

        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize B = N - RANK;
        auto [input_shape_b, input_shape_r] = input.shape().template split<B>();
        auto [output_shape_b, output_shape_r] = output.shape().template split<B>();

        // Broadcast input batches.
        auto input_strides = input.strides();
        for (usize i{}; i < B; ++i)
            if (input_shape_b[i] == 1)
                input_strides[i] = 0;

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

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        details::check_separable_filter(filter_depth, output_shape_b, device);
        details::check_separable_filter(filter_height, output_shape_b, device);
        details::check_separable_filter(filter_width, output_shape_b, device);

        if (not buffer.is_empty()) {
            check(buffer.shape() == output.shape() and nd::are_elements_unique(buffer.strides(), output.shape()),
                  "The temporary array should be able to hold an array of shape {}, but got buffer:shape={} and buffer:strides={}",
                  output.shape(), buffer.shape(), buffer.strides());
            check(device == buffer.device(),
                  "The temporary array must be on the same device as the output, but got buffer:device={}, output:device={}",
                  buffer.device(), device);
        }

        check(OPTIONS.borders[options.border], "Border mode is not generated");

        if constexpr (OPTIONS.borders[Border::ZERO]) {
            if (options.border == Border::ZERO) {
                details::launch_convolve_separable<RANK, OPTIONS, Border::ZERO>(
                    NOA_FWD(input), input_strides, NOA_FWD(output),
                    NOA_FWD(filter_depth), NOA_FWD(filter_height), NOA_FWD(filter_width),
                    NOA_FWD(buffer));
                return;
            }
        }
        if constexpr (OPTIONS.borders[Border::REFLECT]) {
            if (options.border == Border::REFLECT) {
                details::launch_convolve_separable<RANK, OPTIONS, Border::REFLECT>(
                    NOA_FWD(input), input_strides, NOA_FWD(output),
                    NOA_FWD(filter_depth), NOA_FWD(filter_height), NOA_FWD(filter_width),
                    NOA_FWD(buffer));
            }
            return;
        }
        panic("The border mode should be {} or {}, but got {}", Border::ZERO, Border::REFLECT, options.border);
    }

    template<
        ConvolveCompileOptions OPTIONS = ConvolveCompileOptions{},
        typename Input, typename Output,
        typename FilterWidth = ArrayView<nt::const_value_type_t<Input>, 1>,
        typename Buffer = ArrayView<nt::value_type_t<Output>, nt::array_size_v<Output>>>
    void convolve_separable_1d(
        Input&& input,
        Output&& output,
        FilterWidth&& filter_width,
        Buffer&& buffer = Buffer{},
        const ConvolveOptions& options = {}
    ) {
        convolve_separable<1, OPTIONS>(NOA_FWD(input), NOA_FWD(output), {}, {}, NOA_FWD(filter_width), NOA_FWD(buffer), options);
    }
    template<
        ConvolveCompileOptions OPTIONS = ConvolveCompileOptions{},
        typename Input, typename Output,
        typename FilterHeight = ArrayView<nt::const_value_type_t<Input>, 1>,
        typename FilterWidth = ArrayView<nt::const_value_type_t<Input>, 1>,
        typename Buffer = ArrayView<nt::value_type_t<Output>, nt::array_size_v<Output>>>
    void convolve_separable_2d(
        Input&& input,
        Output&& output,
        FilterHeight&& filter_height,
        FilterWidth&& filter_width,
        Buffer&& buffer = Buffer{},
        const ConvolveOptions& options = {}
    ) {
        convolve_separable<2, OPTIONS>(NOA_FWD(input), NOA_FWD(output), {}, NOA_FWD(filter_height), NOA_FWD(filter_width), NOA_FWD(buffer), options);
    }
    template<
        ConvolveCompileOptions OPTIONS = ConvolveCompileOptions{},
        typename Input, typename Output,
        typename FilterDepth = ArrayView<nt::const_value_type_t<Input>, 1>,
        typename FilterHeight = ArrayView<nt::const_value_type_t<Input>, 1>,
        typename FilterWidth = ArrayView<nt::const_value_type_t<Input>, 1>,
        typename Buffer = ArrayView<nt::value_type_t<Output>, nt::array_size_v<Output>>>
    void convolve_separable_3d(
        Input&& input,
        Output&& output,
        FilterDepth&& filter_depth,
        FilterHeight&& filter_height,
        FilterWidth&& filter_width,
        Buffer&& buffer = Buffer{},
        const ConvolveOptions& options = {}
    ) {
        convolve_separable<3, OPTIONS>(NOA_FWD(input), NOA_FWD(output), NOA_FWD(filter_depth), NOA_FWD(filter_height), NOA_FWD(filter_width), NOA_FWD(buffer), options);
    }

    /// 1d, 2d or 3d convolution.
    /// \param[in] input:
    ///     RANK=1: ((Bi..,)W)
    ///     RANK=2: ((Bi..,)H,W)
    ///     RANK=3: ((Bi..,)D,H,W)
    ///     Input array of integers or floating-points to convolve.
    ///     Batch axes are broadcast to the output batches.
    /// \param[out] output:
    ///     RANK=1: ((Bo..,)W)
    ///     RANK=2: ((Bo..,)H,W)
    ///     RANK=3: ((Bo..,)D,H,W)
    ///     Output convolved array of integers or floating-points.
    ///     Should not overlap with input.
    /// \param[in] filter:
    ///     RANK=1: (Wf)       or ((Bo..,)Wf)
    ///     RANK=2: (Hf,Wf)    or ((Bo..,)Hf,Wf)
    ///     RANK=3: (Df,Hf,Wf) or ((Bo..,)Df,Hf,Wf)
    ///     Batched or unbatched filter(s) of integers or floating-points.
    ///     Ranked axes should have an odd number of elements and be contiguous.
    ///     The input values are cast to the value types of the filter before convolution,
    ///     and the result of the convolution is cast to the output type.
    /// \param options:
    ///     Convolution options.
    template<usize RANK, ConvolveCompileOptions OPTIONS = ConvolveCompileOptions{},
             typename Input, typename Output, typename Filter>
        requires details::convolveable<RANK, Input, Output, Filter>
    void convolve(Input&& input, Output&& output, Filter&& filter, const ConvolveOptions& options = {}) {
        if (filter.shape() == 1)
            return ewise(noa::wrap(NOA_FWD(input), filter.first()), NOA_FWD(output), noa::Multiply{});

        check(nd::are_arrays_valid(input, output, filter), "Empty array detected");
        check(not are_overlapped(input, output), "The input and output array should not overlap");

        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize B = N - RANK;
        auto [input_shape_b, input_shape_r] = input.shape().template split<B>();
        auto [output_shape_b, output_shape_r] = output.shape().template split<B>();

        // Broadcast input batches.
        auto input_strides = input.strides();
        for (usize i{}; i < B; ++i)
            if (input_shape_b[i] == 1)
                input_strides[i] = 0;

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

        constexpr usize FILTER_B = nt::array_size_v<Filter> - RANK;
        auto [filter_shape_b, filter_shape_r] = filter.shape().template split<FILTER_B>();
        if constexpr (FILTER_B >= 1) {
            check(filter_shape_b == output_shape_b,
                  "The batch axes are not compatible, filter:batches={}, output:batches={}",
                  filter_shape_b, output_shape_b);
        }
        check(filter.contiguity().template pop_front<FILTER_B>() == true and (filter_shape_r % 2) == 1,
              "The ranked axes of the filter should be contiguous and with an odd number of elements, but got a filter:shape={} and filter:strides={}",
              filter.shape(), filter.strides());

        const Device device = output.device();
        check(device == input.device() and device == filter.device(),
              "The input and output arrays must be on the same device, but got input:device={}, filter:device={}, output:device={}",
              input.device(), filter.device(), device);

        if constexpr (OPTIONS.borders[Border::ZERO]) {
            if (options.border == Border::ZERO) {
                details::launch_convolve<RANK, OPTIONS, Border::ZERO>(
                    NOA_FWD(input), input_strides, NOA_FWD(output), NOA_FWD(filter));
                return;
            }
        }
        if constexpr (OPTIONS.borders[Border::REFLECT]) {
            if (options.border == Border::REFLECT) {
                details::launch_convolve<RANK, OPTIONS, Border::REFLECT>(
                    NOA_FWD(input), input_strides, NOA_FWD(output), NOA_FWD(filter));
            }
            return;
        }
        panic("The border mode should be {} or {}, but got {}", Border::ZERO, Border::REFLECT, options.border);
    }

    template<ConvolveCompileOptions OPTIONS = ConvolveCompileOptions{},
             typename Input, typename Output, typename Filter>
    void convolve_1d(Input&& input, Output&& output, Filter&& filter, const ConvolveOptions& options = {}) {
        convolve<1, OPTIONS>(NOA_FWD(input), NOA_FWD(output), NOA_FWD(filter), options);
    }
    template<ConvolveCompileOptions OPTIONS = ConvolveCompileOptions{},
             typename Input, typename Output, typename Filter>
    void convolve_2d(Input&& input, Output&& output, Filter&& filter, const ConvolveOptions& options = {}) {
        convolve<2, OPTIONS>(NOA_FWD(input), NOA_FWD(output), NOA_FWD(filter), options);
    }
    template<ConvolveCompileOptions OPTIONS = ConvolveCompileOptions{},
             typename Input, typename Output, typename Filter>
    void convolve_3d(Input&& input, Output&& output, Filter&& filter, const ConvolveOptions& options = {}) {
        convolve<3, OPTIONS>(NOA_FWD(input), NOA_FWD(output), NOA_FWD(filter), options);
    }
}
