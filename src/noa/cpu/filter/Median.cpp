#include <algorithm> // std::nth_element
#include <omp.h>

#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/filter/Median.h"

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
    void medfilt1_(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                   size4_t shape, size_t window, size_t threads) {
        const int4_t int_shape(shape);
        const int int_window = static_cast<int>(window);
        const size_t WINDOW_HALF = window / 2;
        const int HALO = int_window / 2;

        // If half precision, do the sort in single-precision.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer(window * threads);
        Comp* buffer_ptr = buffer.get();

        #pragma omp parallel for default(none) num_threads(threads) collapse(3) \
        shared(input, input_stride, output, output_stride, int_shape, int_window, buffer_ptr, WINDOW_HALF, HALO)

        for (int i = 0; i < int_shape[0]; ++i) {
            for (int j = 0; j < int_shape[1]; ++j) {
                for (int k = 0; k < int_shape[2]; ++k) {
                    for (int l = 0; l < int_shape[3]; ++l) {

                        #if NOA_ENABLE_OPENMP
                        Comp* tmp = buffer_ptr + omp_get_thread_num() * int_window;
                        #else
                        Comp* tmp = buffer_ptr;
                        #endif

                        // Gather the window.
                        if constexpr (MODE == BORDER_REFLECT) {
                            for (int wl = 0; wl < int_window; ++wl) {
                                int il = getMirrorIdx_(l - HALO + wl, int_shape[3]);
                                tmp[wl] = static_cast<Comp>(input[at(i, j, k, il, input_stride)]);
                            }
                        } else { // BORDER_ZERO
                            for (int wl = 0; wl < int_window; ++wl) {
                                int il = l - HALO + wl;
                                if (il < 0 || il >= int_shape[3])
                                    tmp[wl] = static_cast<Comp>(0);
                                else
                                    tmp[wl] = static_cast<Comp>(input[at(i, j, k, il, input_stride)]);
                            }
                        }

                        // Sort the elements in the window to get the median.
                        std::nth_element(tmp, tmp + WINDOW_HALF, tmp + int_window);
                        output[at(i, j, k, l, output_stride)] = static_cast<T>(tmp[WINDOW_HALF]);
                    }
                }
            }
        }
    }

    template<typename T, BorderMode MODE>
    void medfilt2_(const T* input, size4_t input_stride, T* output, size4_t output_stride,
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

        #pragma omp parallel for default(none) num_threads(threads) collapse(3) \
        shared(input, input_stride, output, output_stride, int_shape, int_window, \
               buffer_ptr, WINDOW_SIZE, WINDOW_HALF, HALO)

        for (int i = 0; i < int_shape[0]; ++i) {
            for (int j = 0; j < int_shape[1]; ++j) {
                for (int k = 0; k < int_shape[2]; ++k) {
                    for (int l = 0; l < int_shape[3]; ++l) {

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
                                            static_cast<Comp>(input[at(i, j, ik, il, input_stride)]);
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
                                                static_cast<Comp>(input[at(i, j, ik, il, input_stride)]);
                                    }
                                }
                            }
                        }

                        // Sort the elements in the window to get the median.
                        std::nth_element(tmp, tmp + WINDOW_HALF, tmp + WINDOW_SIZE);
                        output[at(i, j, k, l, output_stride)] = static_cast<T>(tmp[WINDOW_HALF]);
                    }
                }
            }
        }
    }

    template<typename T, BorderMode MODE>
    void medfilt3_(const T* input, size4_t input_stride, T* output, size4_t output_stride,
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

        #pragma omp parallel for default(none) num_threads(threads) collapse(3) \
        shared(input, input_stride, output, output_stride, int_shape, int_window, \
               buffer_ptr, WINDOW_SIZE, WINDOW_HALF, HALO)

        for (int i = 0; i < int_shape[0]; ++i) {
            for (int j = 0; j < int_shape[1]; ++j) {
                for (int k = 0; k < int_shape[2]; ++k) {
                    for (int l = 0; l < int_shape[3]; ++l) {

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
                                                static_cast<Comp>(input[at(i, ij, ik, il, input_stride)]);
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
                                            tmp[idx] = static_cast<Comp>(input[at(i, ij, ik, il, input_stride)]);
                                        }
                                    }
                                }
                            }
                        }

                        // Sort the elements in the window to get the median.
                        std::nth_element(tmp, tmp + WINDOW_HALF, tmp + WINDOW_SIZE);
                        output[at(i, j, k, l, output_stride)] = static_cast<T>(tmp[WINDOW_HALF]);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::filter {
    template<typename T>
    void median1(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                 BorderMode border_mode, size_t window_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);

        if (window_size == 1)
            return memory::copy(input, input_stride, output, output_stride, shape, stream);

        NOA_ASSERT(window_size % 2);
        switch (border_mode) {
            case BORDER_REFLECT:
                NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                return stream.enqueue(medfilt1_<T, BORDER_REFLECT>, input, input_stride, output, output_stride,
                                      shape, window_size, stream.threads());
            case BORDER_ZERO:
                return stream.enqueue(medfilt1_<T, BORDER_ZERO>, input, input_stride, output, output_stride,
                                      shape, window_size, stream.threads());
            default:
                NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                          BORDER_ZERO, BORDER_REFLECT, border_mode);
        }
    }

    template<typename T>
    void median2(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                 BorderMode border_mode, size_t window_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);

        if (window_size == 1)
            return memory::copy(input, input_stride, output, output_stride, shape, stream);

        NOA_ASSERT(window_size % 2);
        switch (border_mode) {
            case BORDER_REFLECT:
                NOA_ASSERT(window_size / 2 + 1 <= shape[2]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                return stream.enqueue(medfilt2_<T, BORDER_REFLECT>, input, input_stride, output, output_stride,
                                      shape, window_size, stream.threads());
            case BORDER_ZERO:
                return stream.enqueue(medfilt2_<T, BORDER_ZERO>, input, input_stride, output, output_stride,
                                      shape, window_size, stream.threads());
            default:
                NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                          BORDER_ZERO, BORDER_REFLECT, border_mode);
        }
    }

    template<typename T>
    void median3(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                 BorderMode border_mode, size_t window_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);

        if (window_size == 1)
            return memory::copy(input, input_stride, output, output_stride, shape, stream);

        NOA_ASSERT(window_size % 2);
        switch (border_mode) {
            case BORDER_REFLECT:
                NOA_ASSERT(window_size / 2 + 1 <= shape[1]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[2]);
                NOA_ASSERT(window_size / 2 + 1 <= shape[3]);
                return stream.enqueue(medfilt3_<T, BORDER_REFLECT>, input, input_stride, output, output_stride,
                                      shape, window_size, stream.threads());
            case BORDER_ZERO:
                return stream.enqueue(medfilt3_<T, BORDER_ZERO>, input, input_stride, output, output_stride,
                                      shape, window_size, stream.threads());
            default:
                NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                          BORDER_ZERO, BORDER_REFLECT, border_mode);
        }
    }

    #define NOA_INSTANTIATE_MEDFILT_(T)                                                             \
    template void median1<T>(const T*, size4_t, T*, size4_t, size4_t, BorderMode, size_t, Stream&); \
    template void median2<T>(const T*, size4_t, T*, size4_t, size4_t, BorderMode, size_t, Stream&); \
    template void median3<T>(const T*, size4_t, T*, size4_t, size4_t, BorderMode, size_t, Stream&)

    NOA_INSTANTIATE_MEDFILT_(half_t);
    NOA_INSTANTIATE_MEDFILT_(float);
    NOA_INSTANTIATE_MEDFILT_(double);
    NOA_INSTANTIATE_MEDFILT_(int);
    NOA_INSTANTIATE_MEDFILT_(long);
    NOA_INSTANTIATE_MEDFILT_(long long);
    NOA_INSTANTIATE_MEDFILT_(unsigned int);
    NOA_INSTANTIATE_MEDFILT_(unsigned long);
    NOA_INSTANTIATE_MEDFILT_(unsigned long long);
}
