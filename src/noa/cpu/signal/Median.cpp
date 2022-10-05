#include <algorithm> // std::nth_element
#include <omp.h>

#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/signal/Median.h"

namespace {
    using namespace noa;

    // If idx is out of bound, apply mirror (d c b | a b c d | c b a).
    // This requires shape >= window/2+1.
    int64_t getMirrorIdx_(int64_t idx, int64_t shape) {
        if (idx < 0)
            idx *= -1;
        else if (idx >= shape)
            idx = 2 * (shape - 1) - idx;
        return idx;
    }

    template<typename T, BorderMode MODE>
    void medfilt1_(AccessorRestrict<const T, 4, dim_t> input,
                   AccessorRestrict<T, 4, dim_t> output,
                   dim4_t shape, dim_t window, dim_t threads) {
        const long4_t l_shape(shape);
        const auto l_window = static_cast<int64_t>(window);
        const dim_t WINDOW_HALF = window / 2;
        const int64_t HALO = l_window / 2;

        // If half precision, do the sort in single-precision.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer(window * threads);
        Comp* __restrict buffer_ptr = buffer.get();

        #pragma omp parallel for default(none) num_threads(threads) collapse(4) \
        shared(input, output, l_shape, l_window, buffer_ptr, WINDOW_HALF, HALO)

        for (int64_t i = 0; i < l_shape[0]; ++i) {
            for (int64_t j = 0; j < l_shape[1]; ++j) {
                for (int64_t k = 0; k < l_shape[2]; ++k) {
                    for (int64_t l = 0; l < l_shape[3]; ++l) {

                        #if NOA_ENABLE_OPENMP
                        Comp* __restrict tmp = buffer_ptr + omp_get_thread_num() * l_window;
                        #else
                        Comp* __restrict tmp = buffer_ptr;
                        #endif

                        // Gather the window.
                        if constexpr (MODE == BORDER_REFLECT) {
                            for (int64_t wl = 0; wl < l_window; ++wl) {
                                const int64_t il = getMirrorIdx_(l - HALO + wl, l_shape[3]);
                                tmp[wl] = static_cast<Comp>(input(i, j, k, il));
                            }
                        } else { // BORDER_ZERO
                            for (int64_t wl = 0; wl < l_window; ++wl) {
                                const int64_t il = l - HALO + wl;
                                if (il < 0 || il >= l_shape[3])
                                    tmp[wl] = static_cast<Comp>(0);
                                else
                                    tmp[wl] = static_cast<Comp>(input(i, j, k, il));
                            }
                        }

                        // Sort the elements in the window to get the median.
                        std::nth_element(tmp, tmp + WINDOW_HALF, tmp + l_window);
                        output(i, j, k, l) = static_cast<T>(tmp[WINDOW_HALF]);
                    }
                }
            }
        }
    }

    template<typename T, BorderMode MODE>
    void medfilt2_(AccessorRestrict<const T, 4, dim_t> input,
                   AccessorRestrict<T, 4, dim_t> output,
                   dim4_t shape, dim_t window, dim_t threads) {
        const long4_t l_shape(shape);
        const auto l_window = static_cast<int64_t>(window);
        const dim_t WINDOW_SIZE = window * window;
        const dim_t WINDOW_HALF = WINDOW_SIZE / 2;
        const int64_t HALO = l_window / 2;

        // If half precision, do the sort in single-precision.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer(WINDOW_SIZE * threads);
        Comp* __restrict buffer_ptr = buffer.get();

        #pragma omp parallel for default(none) num_threads(threads) collapse(4) \
        shared(input, output, l_shape, l_window, buffer_ptr, WINDOW_SIZE, WINDOW_HALF, HALO)

        for (int64_t i = 0; i < l_shape[0]; ++i) {
            for (int64_t j = 0; j < l_shape[1]; ++j) {
                for (int64_t k = 0; k < l_shape[2]; ++k) {
                    for (int64_t l = 0; l < l_shape[3]; ++l) {

                        #if NOA_ENABLE_OPENMP
                        Comp* __restrict tmp = buffer_ptr + static_cast<dim_t>(omp_get_thread_num()) * WINDOW_SIZE;
                        #else
                        Comp* __restrict tmp = buffer_ptr;
                        #endif

                        if constexpr (MODE == BORDER_REFLECT) {
                            for (int64_t wk = 0; wk < l_window; ++wk) {
                                int64_t ik = getMirrorIdx_(k - HALO + wk, l_shape[2]);
                                for (int64_t wl = 0; wl < l_window; ++wl) {
                                    int64_t il = getMirrorIdx_(l - HALO + wl, l_shape[3]);
                                    tmp[wk * l_window + wl] = static_cast<Comp>(input(i, j, ik, il));
                                }
                            }
                        } else { // BORDER_ZERO
                            for (int64_t wk = 0; wk < l_window; ++wk) {
                                int64_t ik = k - HALO + wk;
                                for (int64_t wl = 0; wl < l_window; ++wl) {
                                    int64_t il = l - HALO + wl;
                                    if (ik < 0 || ik >= l_shape[2] || il < 0 || il >= l_shape[3]) {
                                        tmp[wk * l_window + wl] = static_cast<Comp>(0);
                                    } else {
                                        tmp[wk * l_window + wl] = static_cast<Comp>(input(i, j, ik, il));
                                    }
                                }
                            }
                        }

                        // Sort the elements in the window to get the median.
                        std::nth_element(tmp, tmp + WINDOW_HALF, tmp + WINDOW_SIZE);
                        output(i, j, k, l) = static_cast<T>(tmp[WINDOW_HALF]);
                    }
                }
            }
        }
    }

    template<typename T, BorderMode MODE>
    void medfilt3_(AccessorRestrict<const T, 4, dim_t> input,
                   AccessorRestrict<T, 4, dim_t> output,
                   dim4_t shape, dim_t window, dim_t threads) {
        const long4_t l_shape(shape);
        const auto l_window = static_cast<int64_t>(window);
        const dim_t WINDOW = window;
        const dim_t WINDOW_SIZE = WINDOW * WINDOW * WINDOW;
        const dim_t WINDOW_HALF = WINDOW_SIZE / 2;
        const int64_t HALO = l_window / 2;

        // If half precision, do the sort in single-precision.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer(WINDOW_SIZE * threads);
        Comp* __restrict buffer_ptr = buffer.get();

        #pragma omp parallel for default(none) num_threads(threads) collapse(4) \
        shared(input, output, l_shape, l_window, buffer_ptr, WINDOW_SIZE, WINDOW_HALF, HALO)

        for (int64_t i = 0; i < l_shape[0]; ++i) {
            for (int64_t j = 0; j < l_shape[1]; ++j) {
                for (int64_t k = 0; k < l_shape[2]; ++k) {
                    for (int64_t l = 0; l < l_shape[3]; ++l) {

                        #if NOA_ENABLE_OPENMP
                        Comp* __restrict tmp = buffer_ptr + static_cast<dim_t>(omp_get_thread_num()) * WINDOW_SIZE;
                        #else
                        Comp* __restrict tmp = buffer_ptr;
                        #endif

                        if constexpr (MODE == BORDER_REFLECT) {
                            for (int64_t wj = 0; wj < l_window; ++wj) {
                                const int64_t ij = getMirrorIdx_(j - HALO + wj, l_shape[1]);
                                for (int64_t wk = 0; wk < l_window; ++wk) {
                                    const int64_t ik = getMirrorIdx_(k - HALO + wk, l_shape[2]);
                                    for (int64_t wl = 0; wl < l_window; ++wl) {
                                        const int64_t il = getMirrorIdx_(l - HALO + wl, l_shape[3]);
                                        tmp[(wj * l_window + wk) * l_window + wl] =
                                                static_cast<Comp>(input(i, ij, ik, il));
                                    }
                                }
                            }
                        } else { // BORDER_ZERO
                            for (int64_t wj = 0; wj < l_window; ++wj) {
                                const int64_t ij = j - HALO + wj;
                                for (int64_t wk = 0; wk < l_window; ++wk) {
                                    const int64_t ik = k - HALO + wk;
                                    for (int64_t wl = 0; wl < l_window; ++wl) {
                                        const int64_t il = l - HALO + wl;
                                        const int64_t idx = (wj * l_window + wk) * l_window + wl;
                                        if (ij < 0 || ij >= l_shape[1] ||
                                            ik < 0 || ik >= l_shape[2] ||
                                            il < 0 || il >= l_shape[3]) {
                                            tmp[idx] = static_cast<Comp>(0);
                                        } else {
                                            tmp[idx] = static_cast<Comp>(input(i, ij, ik, il));
                                        }
                                    }
                                }
                            }
                        }

                        // Sort the elements in the window to get the median.
                        std::nth_element(tmp, tmp + WINDOW_HALF, tmp + WINDOW_SIZE);
                        output(i, j, k, l) = static_cast<T>(tmp[WINDOW_HALF]);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::signal {
    template<typename T, typename>
    void median1(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides,
                 dim4_t shape, BorderMode border_mode, dim_t window_size, Stream& stream) {
        NOA_ASSERT(input != output && all(shape > 0));
        if (window_size == 1)
            return memory::copy(input, input_strides, output, output_strides, shape, stream);

        const dim_t threads = stream.threads();
        stream.enqueue([=](){
            NOA_ASSERT(window_size % 2);

            switch (border_mode) {
                case BORDER_REFLECT:
                    NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                    return medfilt1_<T, BORDER_REFLECT>(
                            {input.get(), input_strides}, {output.get(), output_strides}, shape, window_size, threads);
                case BORDER_ZERO:
                    return medfilt1_<T, BORDER_ZERO>(
                            {input.get(), input_strides}, {output.get(), output_strides}, shape, window_size, threads);
                default:
                    NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                              BORDER_ZERO, BORDER_REFLECT, border_mode);
            }
        });
    }

    template<typename T, typename>
    void median2(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides,
                 dim4_t shape, BorderMode border_mode, dim_t window_size, Stream& stream) {
        NOA_ASSERT(input != output && all(shape > 0));
        if (window_size == 1)
            return memory::copy(input, input_strides, output, output_strides, shape, stream);

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            NOA_ASSERT(window_size % 2);

            const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
            if (any(order_2d != dim2_t{0, 1})) {
                std::swap(input_strides[2], input_strides[3]);
                std::swap(output_strides[2], output_strides[3]);
                std::swap(shape[2], shape[3]);
            }

            switch (border_mode) {
                case BORDER_REFLECT:
                    NOA_ASSERT(window_size / 2 + 1 <= shape[2]);
                    NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                    return medfilt2_<T, BORDER_REFLECT>(
                            {input.get(), input_strides}, {output.get(), output_strides}, shape, window_size, threads);
                case BORDER_ZERO:
                    return medfilt2_<T, BORDER_ZERO>(
                            {input.get(), input_strides}, {output.get(), output_strides}, shape, window_size, threads);
                default:
                    NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                              BORDER_ZERO, BORDER_REFLECT, border_mode);
            }
        });
    }

    template<typename T, typename>
    void median3(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides,
                 dim4_t shape, BorderMode border_mode, dim_t window_size, Stream& stream) {
        NOA_ASSERT(input != output && all(shape > 0));
        if (window_size == 1)
            return memory::copy(input, input_strides, output, output_strides, shape, stream);

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            NOA_ASSERT(window_size % 2);

            const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1))) + 1;
            if (any(order_3d != dim3_t{1, 2, 3})) {
                const dim4_t order{0, order_3d[0], order_3d[1], order_3d[2]};
                input_strides = indexing::reorder(input_strides, order);
                output_strides = indexing::reorder(output_strides, order);
                shape = indexing::reorder(shape, order);
            }

            switch (border_mode) {
                case BORDER_REFLECT:
                    NOA_ASSERT(window_size / 2 + 1 <= shape[1]);
                    NOA_ASSERT(window_size / 2 + 1 <= shape[2]);
                    NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                    return medfilt3_<T, BORDER_REFLECT>(
                            {input.get(), input_strides}, {output.get(), output_strides}, shape, window_size, threads);
                case BORDER_ZERO:
                    return medfilt3_<T, BORDER_ZERO>(
                            {input.get(), input_strides}, {output.get(), output_strides}, shape, window_size, threads);
                default:
                    NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                              BORDER_ZERO, BORDER_REFLECT, border_mode);
            }
        });
    }

    #define NOA_INSTANTIATE_MEDFILT_(T)                                                                                             \
    template void median1<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, BorderMode, dim_t, Stream&); \
    template void median2<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, BorderMode, dim_t, Stream&); \
    template void median3<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, BorderMode, dim_t, Stream&)

    NOA_INSTANTIATE_MEDFILT_(half_t);
    NOA_INSTANTIATE_MEDFILT_(float);
    NOA_INSTANTIATE_MEDFILT_(double);
    NOA_INSTANTIATE_MEDFILT_(int32_t);
    NOA_INSTANTIATE_MEDFILT_(int64_t);
    NOA_INSTANTIATE_MEDFILT_(uint32_t);
    NOA_INSTANTIATE_MEDFILT_(uint64_t);
}
