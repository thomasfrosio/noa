#include <algorithm> // std::nth_element
#include <omp.h>

#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Set.h"
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
    void medfilt1_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                   size3_t shape, size_t batches, size_t window, size_t threads) {
        const int3_t int_shape(shape);
        const int int_window = static_cast<int>(window);
        const size_t WINDOW_HALF = window / 2;
        const int HALO = int_window / 2;

        // If half precision, do the sort in single-precision.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer(window * threads);
        Comp* buffer_ptr = buffer.get();

        #pragma omp parallel for default(none) num_threads(threads) collapse(4) \
        shared(inputs, input_pitch, outputs, output_pitch, batches, int_shape, int_window, buffer_ptr, WINDOW_HALF, HALO)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (int z = 0; z < int_shape.z; ++z) {
                for (int y = 0; y < int_shape.y; ++y) {
                    for (int x = 0; x < int_shape.x; ++x) {

                        const T* input = inputs + batch * elements(input_pitch);
                        T* output = outputs + batch * elements(output_pitch);
                        output += index(x, y, z, output_pitch);

                        #if NOA_ENABLE_OPENMP
                        Comp* tmp = buffer_ptr + omp_get_thread_num() * int_window;
                        #else
                        Comp* tmp = buffer_ptr;
                        #endif

                        // Gather the window.
                        if constexpr (MODE == BORDER_REFLECT) {
                            for (int w_x = 0; w_x < int_window; ++w_x) {
                                int idx = getMirrorIdx_(x - HALO + w_x, int_shape.x);
                                tmp[w_x] = static_cast<Comp>(input[index(idx, y, z, input_pitch)]);
                            }
                        } else { // BORDER_ZERO
                            for (int w_x = 0; w_x < int_window; ++w_x) {
                                int idx = x - HALO + w_x;
                                if (idx < 0 || idx >= int_shape.x)
                                    tmp[w_x] = static_cast<Comp>(0);
                                else
                                    tmp[w_x] = static_cast<Comp>(input[index(idx, y, z, input_pitch)]);
                            }
                        }

                        // Sort the elements in the window to get the median.
                        std::nth_element(tmp, tmp + WINDOW_HALF, tmp + int_window);
                        *output = static_cast<T>(tmp[WINDOW_HALF]);
                    }
                }
            }
        }
    }

    template<typename T, BorderMode MODE>
    void medfilt2_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                   size3_t shape, size_t batches, size_t window, size_t threads) {
        const int3_t int_shape(shape);
        const int int_window = static_cast<int>(window);
        const size_t WINDOW_SIZE = window * window;
        const size_t WINDOW_HALF = WINDOW_SIZE / 2;
        const int HALO = int_window / 2;

        // If half precision, do the sort in single-precision.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer(WINDOW_SIZE * threads);
        Comp* buffer_ptr = buffer.get();

        #pragma omp parallel for default(none) num_threads(threads) collapse(4) \
        shared(inputs, input_pitch, outputs, output_pitch, batches, int_shape, int_window, \
               buffer_ptr, WINDOW_SIZE, WINDOW_HALF, HALO)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (int z = 0; z < int_shape.z; ++z) {
                for (int y = 0; y < int_shape.y; ++y) {
                    for (int x = 0; x < int_shape.x; ++x) {

                        const T* input = inputs + batch * elements(input_pitch);
                        T* output = outputs + batch * elements(output_pitch);
                        output += index(x, y, z, output_pitch);

                        #if NOA_ENABLE_OPENMP
                        Comp* tmp = buffer_ptr + static_cast<size_t>(omp_get_thread_num()) * WINDOW_SIZE;
                        #else
                        Comp* tmp = buffer_ptr;
                        #endif

                        if constexpr (MODE == BORDER_REFLECT) {
                            for (int w_y = 0; w_y < int_window; ++w_y) {
                                int idx_y = getMirrorIdx_(y - HALO + w_y, int_shape.y);
                                for (int w_x = 0; w_x < int_window; ++w_x) {
                                    int idx_x = getMirrorIdx_(x - HALO + w_x, int_shape.x);
                                    size_t idx = index(idx_x, idx_y, z, input_pitch);
                                    tmp[w_y * int_window + w_x] = static_cast<Comp>(input[idx]);
                                }
                            }
                        } else { // BORDER_ZERO
                            for (int w_y = 0; w_y < int_window; ++w_y) {
                                int idx_y = y - HALO + w_y;
                                if (idx_y < 0 || idx_y >= int_shape.y) {
                                    int idx = w_y * int_window;
                                    cpu::memory::set(tmp + idx, tmp + idx + int_window, static_cast<Comp>(0));
                                    continue;
                                }
                                for (int w_x = 0; w_x < int_window; ++w_x) {
                                    int idx_x = x - HALO + w_x;
                                    if (idx_x < 0 || idx_x >= int_shape.x) {
                                        tmp[w_y * int_window + w_x] = static_cast<Comp>(0);
                                    } else {
                                        size_t idx = index(idx_x, idx_y, z, input_pitch);
                                        tmp[w_y * int_window + w_x] = static_cast<Comp>(input[idx]);
                                    }
                                }
                            }
                        }

                        // Sort the elements in the window to get the median.
                        std::nth_element(tmp, tmp + WINDOW_HALF, tmp + WINDOW_SIZE);
                        *output = static_cast<T>(tmp[WINDOW_HALF]);
                    }
                }
            }
        }
    }

    template<typename T, BorderMode MODE>
    void medfilt3_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                   size3_t shape, size_t batches, size_t window, size_t threads) {
        const int3_t int_shape(shape);
        const int int_window = static_cast<int>(window);
        const size_t WINDOW = window;
        const size_t WINDOW_SIZE = WINDOW * WINDOW * WINDOW;
        const size_t WINDOW_HALF = WINDOW_SIZE / 2;
        const int HALO = int_window / 2;

        // If half precision, do the sort in single-precision.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer(WINDOW_SIZE * threads);
        Comp* buffer_ptr = buffer.get();

        #pragma omp parallel for default(none) num_threads(threads) collapse(4) \
        shared(inputs, input_pitch, outputs, output_pitch, batches, int_shape, int_window, \
               buffer_ptr, WINDOW_SIZE, WINDOW_HALF, HALO)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (int z = 0; z < int_shape.z; ++z) {
                for (int y = 0; y < int_shape.y; ++y) {
                    for (int x = 0; x < int_shape.x; ++x) {

                        const T* input = inputs + batch * elements(input_pitch);
                        T* output = outputs + batch * elements(output_pitch);
                        output += index(x, y, z, output_pitch);

                        #if NOA_ENABLE_OPENMP
                        Comp* tmp = buffer_ptr + static_cast<size_t>(omp_get_thread_num()) * WINDOW_SIZE;
                        #else
                        Comp* tmp = buffer_ptr;
                        #endif

                        if constexpr (MODE == BORDER_REFLECT) {
                            for (int w_z = 0; w_z < int_window; ++w_z) {
                                int idx_z = getMirrorIdx_(z - HALO + w_z, int_shape.z);
                                for (int w_y = 0; w_y < int_window; ++w_y) {
                                    int idx_y = getMirrorIdx_(y - HALO + w_y, int_shape.y);
                                    for (int w_x = 0; w_x < int_window; ++w_x) {
                                        int idx_x = getMirrorIdx_(x - HALO + w_x, int_shape.x);
                                        tmp[(w_z * int_window + w_y) * int_window + w_x] =
                                                static_cast<Comp>(input[index(idx_x, idx_y, idx_z, input_pitch)]);
                                    }
                                }
                            }
                        } else { // BORDER_ZERO
                            for (int w_z = 0; w_z < int_window; ++w_z) {
                                int idx_z = z - HALO + w_z;
                                if (idx_z < 0 || idx_z >= int_shape.z) {
                                    int idx = w_z * int_window * int_window;
                                    cpu::memory::set(tmp + idx,
                                                     tmp + idx + int_window * int_window,
                                                     static_cast<Comp>(0));
                                    continue;
                                }
                                for (int w_y = 0; w_y < int_window; ++w_y) {
                                    int idx_y = y - HALO + w_y;
                                    if (idx_y < 0 || idx_y >= int_shape.y) {
                                        int idx = (w_z * int_window + w_y) * int_window;
                                        cpu::memory::set(tmp + idx, tmp + idx + int_window, static_cast<Comp>(0));
                                        continue;
                                    }
                                    for (int w_x = 0; w_x < int_window; ++w_x) {
                                        int idx_x = x - HALO + w_x;
                                        int tmp_idx = (w_z * int_window + w_y) * int_window + w_x;
                                        if (idx_x < 0 || idx_x >= int_shape.x) {
                                            tmp[tmp_idx] = static_cast<Comp>(0);
                                        } else {
                                            size_t idx = index(idx_x, idx_y, idx_z, input_pitch);
                                            tmp[tmp_idx] = static_cast<Comp>(input[idx]);
                                        }
                                    }
                                }
                            }
                        }

                        // Sort the elements in the window to get the median.
                        std::nth_element(tmp, tmp + WINDOW_HALF, tmp + WINDOW_SIZE);
                        *output = static_cast<T>(tmp[WINDOW_HALF]);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::filter {
    template<typename T>
    void median1(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape, size_t batches,
                 BorderMode border_mode, size_t window_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        if (window_size == 1)
            return memory::copy(inputs, input_pitch, outputs, output_pitch, shape, batches, stream);

        NOA_ASSERT(window_size % 2);
        switch (border_mode) {
            case BORDER_REFLECT:
                NOA_ASSERT(window_size / 2 + 1 <= shape.x);
                return stream.enqueue(medfilt1_<T, BORDER_REFLECT>, inputs, input_pitch, outputs, output_pitch,
                                      shape, batches, window_size, stream.threads());
            case BORDER_ZERO:
                return stream.enqueue(medfilt1_<T, BORDER_ZERO>, inputs, input_pitch, outputs, output_pitch,
                                      shape, batches, window_size, stream.threads());
            default:
                NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                          BORDER_ZERO, BORDER_REFLECT, border_mode);
        }
    }

    template<typename T>
    void median2(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape, size_t batches,
                 BorderMode border_mode, size_t window_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        if (window_size == 1)
            return memory::copy(inputs, input_pitch, outputs, output_pitch, shape, batches, stream);

        NOA_ASSERT(window_size % 2);
        switch (border_mode) {
            case BORDER_REFLECT:
                NOA_ASSERT(window_size / 2 + 1 <= shape.x);
                NOA_ASSERT(window_size / 2 + 1 <= shape.y);
                return stream.enqueue(medfilt2_<T, BORDER_REFLECT>, inputs, input_pitch, outputs, output_pitch,
                                      shape, batches, window_size, stream.threads());
            case BORDER_ZERO:
                return stream.enqueue(medfilt2_<T, BORDER_ZERO>, inputs, input_pitch, outputs, output_pitch,
                                      shape, batches, window_size, stream.threads());
            default:
                NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                          BORDER_ZERO, BORDER_REFLECT, border_mode);
        }
    }

    template<typename T>
    void median3(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape, size_t batches,
                 BorderMode border_mode, size_t window_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        if (window_size == 1)
            return memory::copy(inputs, input_pitch, outputs, output_pitch, shape, batches, stream);

        NOA_ASSERT(window_size % 2);
        switch (border_mode) {
            case BORDER_REFLECT:
                NOA_ASSERT(all(window_size / 2 + 1 <= shape));
                return stream.enqueue(medfilt3_<T, BORDER_REFLECT>, inputs, input_pitch, outputs, output_pitch,
                                      shape, batches, window_size, stream.threads());
            case BORDER_ZERO:
                return stream.enqueue(medfilt3_<T, BORDER_ZERO>, inputs, input_pitch, outputs, output_pitch,
                                      shape, batches, window_size, stream.threads());
            default:
                NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                          BORDER_ZERO, BORDER_REFLECT, border_mode);
        }
    }

    #define NOA_INSTANTIATE_MEDFILT_(T)                                                                     \
    template void median1<T>(const T*, size3_t, T*, size3_t, size3_t, size_t, BorderMode, size_t, Stream&); \
    template void median2<T>(const T*, size3_t, T*, size3_t, size3_t, size_t, BorderMode, size_t, Stream&); \
    template void median3<T>(const T*, size3_t, T*, size3_t, size3_t, size_t, BorderMode, size_t, Stream&)

    NOA_INSTANTIATE_MEDFILT_(half_t);
    NOA_INSTANTIATE_MEDFILT_(float);
    NOA_INSTANTIATE_MEDFILT_(double);
    NOA_INSTANTIATE_MEDFILT_(int);
    NOA_INSTANTIATE_MEDFILT_(uint);
}
