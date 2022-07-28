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
    int getMirrorIdx_(int idx, int shape) {
        if (idx < 0)
            idx *= -1;
        else if (idx >= shape)
            idx = 2 * (shape - 1) - idx;
        return idx;
    }

    template<typename T, BorderMode MODE>
    void medfilt1_(const T* input, size4_t input_strides, T* output, size4_t output_strides,
                   size4_t shape, size_t window, size_t threads) {
        const int4_t int_shape(shape);
        const int int_window = static_cast<int>(window);
        const size_t WINDOW_HALF = window / 2;
        const int HALO = int_window / 2;

        // If half precision, do the sort in single-precision.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer(window * threads);
        Comp* buffer_ptr = buffer.get();

        #pragma omp parallel for default(none) num_threads(threads) collapse(4) \
        shared(input, input_strides, output, output_strides, int_shape, int_window, buffer_ptr, WINDOW_HALF, HALO)

        for (int i = 0; i < int_shape[0]; ++i) {
            for (int j = 0; j < int_shape[1]; ++j) {
                for (int k = 0; k < int_shape[2]; ++k) {
                    for (int l = 0; l < int_shape[3]; ++l) {
                        using namespace noa::indexing;

                        #if NOA_ENABLE_OPENMP
                        Comp* tmp = buffer_ptr + omp_get_thread_num() * int_window;
                        #else
                        Comp* tmp = buffer_ptr;
                        #endif

                        // Gather the window.
                        if constexpr (MODE == BORDER_REFLECT) {
                            for (int wl = 0; wl < int_window; ++wl) {
                                int il = getMirrorIdx_(l - HALO + wl, int_shape[3]);
                                tmp[wl] = static_cast<Comp>(input[at(i, j, k, il, input_strides)]);
                            }
                        } else { // BORDER_ZERO
                            for (int wl = 0; wl < int_window; ++wl) {
                                int il = l - HALO + wl;
                                if (il < 0 || il >= int_shape[3])
                                    tmp[wl] = static_cast<Comp>(0);
                                else
                                    tmp[wl] = static_cast<Comp>(input[at(i, j, k, il, input_strides)]);
                            }
                        }

                        // Sort the elements in the window to get the median.
                        std::nth_element(tmp, tmp + WINDOW_HALF, tmp + int_window);
                        output[at(i, j, k, l, output_strides)] = static_cast<T>(tmp[WINDOW_HALF]);
                    }
                }
            }
        }
    }

    template<typename T, BorderMode MODE>
    void medfilt2_(const T* input, size4_t input_strides, T* output, size4_t output_strides,
                   size4_t shape, size_t window, size_t threads) {
        const int4_t int_shape(shape);
        const int int_window = static_cast<int>(window);
        const size_t WINDOW_SIZE = window * window;
        const size_t WINDOW_HALF = WINDOW_SIZE / 2;
        const int HALO = int_window / 2;

        // If half precision, do the sort in single-precision.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer(WINDOW_SIZE * threads);
        Comp* buffer_ptr = buffer.get();

        #pragma omp parallel for default(none) num_threads(threads) collapse(4)     \
        shared(input, input_strides, output, output_strides, int_shape, int_window, \
               buffer_ptr, WINDOW_SIZE, WINDOW_HALF, HALO)

        for (int i = 0; i < int_shape[0]; ++i) {
            for (int j = 0; j < int_shape[1]; ++j) {
                for (int k = 0; k < int_shape[2]; ++k) {
                    for (int l = 0; l < int_shape[3]; ++l) {
                        using namespace noa::indexing;

                        #if NOA_ENABLE_OPENMP
                        Comp* tmp = buffer_ptr + static_cast<size_t>(omp_get_thread_num()) * WINDOW_SIZE;
                        #else
                        Comp* tmp = buffer_ptr;
                        #endif

                        if constexpr (MODE == BORDER_REFLECT) {
                            for (int wk = 0; wk < int_window; ++wk) {
                                int ik = getMirrorIdx_(k - HALO + wk, int_shape[2]);
                                for (int wl = 0; wl < int_window; ++wl) {
                                    int il = getMirrorIdx_(l - HALO + wl, int_shape[3]);
                                    tmp[wk * int_window + wl] =
                                            static_cast<Comp>(input[at(i, j, ik, il, input_strides)]);
                                }
                            }
                        } else { // BORDER_ZERO
                            for (int wk = 0; wk < int_window; ++wk) {
                                int ik = k - HALO + wk;
                                for (int wl = 0; wl < int_window; ++wl) {
                                    int il = l - HALO + wl;
                                    if (ik < 0 || ik >= int_shape[2] || il < 0 || il >= int_shape[3]) {
                                        tmp[wk * int_window + wl] = static_cast<Comp>(0);
                                    } else {
                                        tmp[wk * int_window + wl] =
                                                static_cast<Comp>(input[at(i, j, ik, il, input_strides)]);
                                    }
                                }
                            }
                        }

                        // Sort the elements in the window to get the median.
                        std::nth_element(tmp, tmp + WINDOW_HALF, tmp + WINDOW_SIZE);
                        output[at(i, j, k, l, output_strides)] = static_cast<T>(tmp[WINDOW_HALF]);
                    }
                }
            }
        }
    }

    template<typename T, BorderMode MODE>
    void medfilt3_(const T* input, size4_t input_strides, T* output, size4_t output_strides,
                   size4_t shape, size_t window, size_t threads) {
        const int4_t int_shape(shape);
        const int int_window = static_cast<int>(window);
        const size_t WINDOW = window;
        const size_t WINDOW_SIZE = WINDOW * WINDOW * WINDOW;
        const size_t WINDOW_HALF = WINDOW_SIZE / 2;
        const int HALO = int_window / 2;

        // If half precision, do the sort in single-precision.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer(WINDOW_SIZE * threads);
        Comp* buffer_ptr = buffer.get();

        #pragma omp parallel for default(none) num_threads(threads) collapse(4)     \
        shared(input, input_strides, output, output_strides, int_shape, int_window, \
               buffer_ptr, WINDOW_SIZE, WINDOW_HALF, HALO)

        for (int i = 0; i < int_shape[0]; ++i) {
            for (int j = 0; j < int_shape[1]; ++j) {
                for (int k = 0; k < int_shape[2]; ++k) {
                    for (int l = 0; l < int_shape[3]; ++l) {
                        using namespace noa::indexing;

                        #if NOA_ENABLE_OPENMP
                        Comp* tmp = buffer_ptr + static_cast<size_t>(omp_get_thread_num()) * WINDOW_SIZE;
                        #else
                        Comp* tmp = buffer_ptr;
                        #endif

                        if constexpr (MODE == BORDER_REFLECT) {
                            for (int wj = 0; wj < int_window; ++wj) {
                                int ij = getMirrorIdx_(j - HALO + wj, int_shape[1]);
                                for (int wk = 0; wk < int_window; ++wk) {
                                    int ik = getMirrorIdx_(k - HALO + wk, int_shape[2]);
                                    for (int wl = 0; wl < int_window; ++wl) {
                                        int il = getMirrorIdx_(l - HALO + wl, int_shape[3]);
                                        tmp[(wj * int_window + wk) * int_window + wl] =
                                                static_cast<Comp>(input[at(i, ij, ik, il, input_strides)]);
                                    }
                                }
                            }
                        } else { // BORDER_ZERO
                            for (int wj = 0; wj < int_window; ++wj) {
                                int ij = j - HALO + wj;
                                for (int wk = 0; wk < int_window; ++wk) {
                                    int ik = k - HALO + wk;
                                    for (int wl = 0; wl < int_window; ++wl) {
                                        int il = l - HALO + wl;
                                        int idx = (wj * int_window + wk) * int_window + wl;
                                        if (ij < 0 || ij >= int_shape[1] ||
                                            ik < 0 || ik >= int_shape[2] ||
                                            il < 0 || il >= int_shape[3]) {
                                            tmp[idx] = static_cast<Comp>(0);
                                        } else {
                                            tmp[idx] = static_cast<Comp>(input[at(i, ij, ik, il, input_strides)]);
                                        }
                                    }
                                }
                            }
                        }

                        // Sort the elements in the window to get the median.
                        std::nth_element(tmp, tmp + WINDOW_HALF, tmp + WINDOW_SIZE);
                        output[at(i, j, k, l, output_strides)] = static_cast<T>(tmp[WINDOW_HALF]);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::signal {
    template<typename T, typename>
    void median1(const shared_t<T[]>& input, size4_t input_strides,
                 const shared_t<T[]>& output, size4_t output_strides,
                 size4_t shape, BorderMode border_mode, size_t window_size, Stream& stream) {
        NOA_ASSERT(input != output);
        if (window_size == 1)
            return memory::copy(input, input_strides, output, output_strides, shape, stream);

        const size_t threads = stream.threads();
        stream.enqueue([=](){
            NOA_ASSERT(window_size % 2);

            switch (border_mode) {
                case BORDER_REFLECT:
                    NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                    return medfilt1_<T, BORDER_REFLECT>(
                            input.get(), input_strides, output.get(), output_strides, shape, window_size, threads);
                case BORDER_ZERO:
                    return medfilt1_<T, BORDER_ZERO>(
                            input.get(), input_strides, output.get(), output_strides, shape, window_size, threads);
                default:
                    NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                              BORDER_ZERO, BORDER_REFLECT, border_mode);
            }
        });
    }

    template<typename T, typename>
    void median2(const shared_t<T[]>& input, size4_t input_strides,
                 const shared_t<T[]>& output, size4_t output_strides,
                 size4_t shape, BorderMode border_mode, size_t window_size, Stream& stream) {
        NOA_ASSERT(input != output);
        if (window_size == 1)
            return memory::copy(input, input_strides, output, output_strides, shape, stream);

        const size_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            NOA_ASSERT(window_size % 2);

            const size2_t order_2d = indexing::order(size2_t(output_strides.get(2)), size2_t(shape.get(2)));
            if (any(order_2d != size2_t{0, 1})) {
                std::swap(input_strides[2], input_strides[3]);
                std::swap(output_strides[2], output_strides[3]);
                std::swap(shape[2], shape[3]);
            }

            switch (border_mode) {
                case BORDER_REFLECT:
                    NOA_ASSERT(window_size / 2 + 1 <= shape[2]);
                    NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                    return medfilt2_<T, BORDER_REFLECT>(
                            input.get(), input_strides, output.get(), output_strides, shape, window_size, threads);
                case BORDER_ZERO:
                    return medfilt2_<T, BORDER_ZERO>(
                            input.get(), input_strides, output.get(), output_strides, shape, window_size, threads);
                default:
                    NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                              BORDER_ZERO, BORDER_REFLECT, border_mode);
            }
        });
    }

    template<typename T, typename>
    void median3(const shared_t<T[]>& input, size4_t input_strides,
                 const shared_t<T[]>& output, size4_t output_strides,
                 size4_t shape, BorderMode border_mode, size_t window_size, Stream& stream) {
        NOA_ASSERT(input != output);
        if (window_size == 1)
            return memory::copy(input, input_strides, output, output_strides, shape, stream);

        const size_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            NOA_ASSERT(window_size % 2);

            const size3_t order_3d = indexing::order(size3_t(output_strides.get(1)), size3_t(shape.get(1))) + 1;
            if (any(order_3d != size3_t{1, 2, 3})) {
                const size4_t order{0, order_3d[0], order_3d[1], order_3d[2]};
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
                            input.get(), input_strides, output.get(), output_strides, shape, window_size, threads);
                case BORDER_ZERO:
                    return medfilt3_<T, BORDER_ZERO>(
                            input.get(), input_strides, output.get(), output_strides, shape, window_size, threads);
                default:
                    NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                              BORDER_ZERO, BORDER_REFLECT, border_mode);
            }
        });
    }

    #define NOA_INSTANTIATE_MEDFILT_(T)                                                                                                 \
    template void median1<T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, BorderMode, size_t, Stream&); \
    template void median2<T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, BorderMode, size_t, Stream&); \
    template void median3<T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, BorderMode, size_t, Stream&)

    NOA_INSTANTIATE_MEDFILT_(half_t);
    NOA_INSTANTIATE_MEDFILT_(float);
    NOA_INSTANTIATE_MEDFILT_(double);
    NOA_INSTANTIATE_MEDFILT_(int32_t);
    NOA_INSTANTIATE_MEDFILT_(int64_t);
    NOA_INSTANTIATE_MEDFILT_(uint32_t);
    NOA_INSTANTIATE_MEDFILT_(uint64_t);
}
