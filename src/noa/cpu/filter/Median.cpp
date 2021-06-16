#include <algorithm> // std::nth_element

#include "noa/Exception.h"
#include "noa/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Set.h"
#include "noa/cpu/memory/PtrHost.h"
#include "Median.h"

namespace {
    using namespace Noa;

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
    void medfilt1_(const T* in, T* out, int3_t shape, int window) {
        const auto WINDOW_HALF = static_cast<size_t>(window / 2);
        const int HALO = window / 2;
        Memory::PtrHost<T> buffer(static_cast<size_t>(window));

        for (int z = 0; z < shape.z; ++z) {
            for (int y = 0; y < shape.y; ++y) {
                for (int x = 0; x < shape.x; ++x, ++out) {

                    // Gather the window.
                    T* tmp = buffer.get();
                    if constexpr (MODE == BORDER_MIRROR) {
                        for (int w_x = 0; w_x < window; ++w_x, ++tmp)
                            *tmp = in[(z * shape.y + y) * shape.x + getMirrorIdx_(x - HALO + w_x, shape.x)];
                    } else { // BORDER_ZERO
                        for (int w_x = 0; w_x < window; ++w_x, ++tmp) {
                            int idx = x - HALO + w_x;
                            if (idx < 0 || idx >= shape.x)
                                *tmp = static_cast<T>(0);
                            else
                                *tmp = in[(z * shape.y + y) * shape.x + idx];
                        }
                    }

                    // Sort the elements in the window to get the median.
                    std::nth_element(buffer.begin(), buffer.begin() + WINDOW_HALF, buffer.end());
                    *out = buffer[WINDOW_HALF];
                }
            }
        }
    }

    template<typename T, BorderMode MODE>
    void medfilt2_(const T* in, T* out, int3_t shape, int window) {
        const size_t WINDOW_SIZE = static_cast<size_t>(window) * static_cast<size_t>(window);
        const size_t WINDOW_HALF = WINDOW_SIZE / 2;
        const int HALO = window / 2;
        Memory::PtrHost<T> buffer(WINDOW_SIZE);

        for (int z = 0; z < shape.z; ++z) {
            for (int y = 0; y < shape.y; ++y) {
                for (int x = 0; x < shape.x; ++x, ++out) {

                    // Gather the window.
                    T* tmp = buffer.get();
                    if constexpr (MODE == BORDER_MIRROR) {
                        for (int w_y = 0; w_y < window; ++w_y) {
                            int idx_y = getMirrorIdx_(y - HALO + w_y, shape.y);
                            for (int w_x = 0; w_x < window; ++w_x, ++tmp) {
                                int idx_x = getMirrorIdx_(x - HALO + w_x, shape.x);
                                *tmp = in[(z * shape.y + idx_y) * shape.x + idx_x];
                            }
                        }
                    } else { // BORDER_ZERO
                        for (int w_y = 0; w_y < window; ++w_y) {
                            int idx_y = y - HALO + w_y;
                            if (idx_y < 0 || idx_y >= shape.y) {
                                Memory::set(tmp, tmp + window, static_cast<T>(0));
                                tmp += window;
                                continue;
                            }
                            for (int w_x = 0; w_x < window; ++w_x, ++tmp) {
                                int idx_x = x - HALO + w_x;
                                if (idx_x < 0 || idx_x >= shape.x)
                                    *tmp = static_cast<T>(0);
                                else
                                    *tmp = in[(z * shape.y + idx_y) * shape.x + idx_x];
                            }
                        }
                    }

                    // Sort the elements in the window to get the median.
                    std::nth_element(buffer.begin(), buffer.begin() + WINDOW_HALF, buffer.end());
                    *out = buffer[WINDOW_HALF];
                }
            }
        }
    }

    template<typename T, BorderMode MODE>
    void medfilt3_(const T* in, T* out, int3_t shape, int window) {
        const auto WINDOW = static_cast<size_t>(window);
        const size_t WINDOW_SIZE = WINDOW * WINDOW * WINDOW;
        const size_t WINDOW_HALF = WINDOW_SIZE / 2;
        const int HALO = window / 2;
        Memory::PtrHost<T> buffer(WINDOW_SIZE);

        for (int z = 0; z < shape.z; ++z) {
            for (int y = 0; y < shape.y; ++y) {
                for (int x = 0; x < shape.x; ++x, ++out) {

                    // Gather the window.
                    T* tmp = buffer.get();
                    if constexpr (MODE == BORDER_MIRROR) {
                        for (int w_z = 0; w_z < window; ++w_z) {
                            int idx_z = getMirrorIdx_(z - HALO + w_z, shape.z);
                            for (int w_y = 0; w_y < window; ++w_y) {
                                int idx_y = getMirrorIdx_(y - HALO + w_y, shape.y);
                                for (int w_x = 0; w_x < window; ++w_x, ++tmp) {
                                    int idx_x = getMirrorIdx_(x - HALO + w_x, shape.x);
                                    *tmp = in[(idx_z * shape.y + idx_y) * shape.x + idx_x];
                                }
                            }
                        }
                    } else { // BORDER_ZERO
                        for (int w_z = 0; w_z < window; ++w_z) {
                            int idx_z = z - HALO + w_z;
                            if (idx_z < 0 || idx_z >= shape.z) {
                                Memory::set(tmp, tmp + window * window, static_cast<T>(0));
                                tmp += window * window;
                                continue;
                            }
                            for (int w_y = 0; w_y < window; ++w_y) {
                                int idx_y = y - HALO + w_y;
                                if (idx_y < 0 || idx_y >= shape.y) {
                                    Memory::set(tmp, tmp + window, static_cast<T>(0));
                                    tmp += window;
                                    continue;
                                }
                                for (int w_x = 0; w_x < window; ++w_x, ++tmp) {
                                    int idx_x = x - HALO + w_x;
                                    if (idx_x < 0 || idx_x >= shape.x)
                                        *tmp = static_cast<T>(0);
                                    else
                                        *tmp = in[(idx_z * shape.y + idx_y) * shape.x + idx_x];
                                }
                            }
                        }
                    }

                    // Sort the elements in the window to get the median.
                    std::nth_element(buffer.begin(), buffer.begin() + WINDOW_HALF, buffer.end());
                    *out = buffer[WINDOW_HALF];
                }
            }
        }
    }
}

namespace Noa::Filter {

    template<typename T>
    void median1(const T* in, T* out, size3_t shape, uint batches, BorderMode border_mode, uint window) {
        NOA_PROFILE_FUNCTION();

        size_t elements = getElements(shape);
        if (window == 1) {
            Memory::copy(in, out, elements * batches);
            return;
        }

        int3_t int_shape(shape);
        int int_window = static_cast<int>(window);
        switch (border_mode) {
            case BORDER_MIRROR: {
                for (uint batch = 0; batch < batches; ++batch)
                    medfilt1_<T, BORDER_MIRROR>(in + batch * elements, out + batch * elements, int_shape, int_window);
                break;
            }
            case BORDER_ZERO: {
                for (uint batch = 0; batch < batches; ++batch)
                    medfilt1_<T, BORDER_ZERO>(in + batch * elements, out + batch * elements, int_shape, int_window);
                break;
            }
            default:
                NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                          BORDER_ZERO, BORDER_MIRROR, border_mode);
        }
    }

    template<typename T>
    void median2(const T* in, T* out, size3_t shape, uint batches, BorderMode border_mode, uint window) {
        NOA_PROFILE_FUNCTION();

        size_t elements = getElements(shape);
        if (window == 1) {
            Memory::copy(in, out, elements * batches);
            return;
        }

        int3_t int_shape(shape);
        int int_window = static_cast<int>(window);
        switch (border_mode) {
            case BORDER_MIRROR: {
                for (uint batch = 0; batch < batches; ++batch)
                    medfilt2_<T, BORDER_MIRROR>(in + batch * elements, out + batch * elements, int_shape, int_window);
                break;
            }
            case BORDER_ZERO: {
                for (uint batch = 0; batch < batches; ++batch)
                    medfilt2_<T, BORDER_ZERO>(in + batch * elements, out + batch * elements, int_shape, int_window);
                break;
            }
            default:
                NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                          BORDER_ZERO, BORDER_MIRROR, border_mode);
        }
    }

    template<typename T>
    void median3(const T* in, T* out, size3_t shape, uint batches, BorderMode border_mode, uint window) {
        NOA_PROFILE_FUNCTION();

        size_t elements = getElements(shape);
        if (window == 1) {
            Memory::copy(in, out, elements * batches);
            return;
        }

        int3_t int_shape(shape);
        int int_window = static_cast<int>(window);
        switch (border_mode) {
            case BORDER_MIRROR: {
                for (uint batch = 0; batch < batches; ++batch)
                    medfilt3_<T, BORDER_MIRROR>(in + batch * elements, out + batch * elements, int_shape, int_window);
                break;
            }
            case BORDER_ZERO: {
                for (uint batch = 0; batch < batches; ++batch)
                    medfilt3_<T, BORDER_ZERO>(in + batch * elements, out + batch * elements, int_shape, int_window);
                break;
            }
            default:
                NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                          BORDER_ZERO, BORDER_MIRROR, border_mode);
        }
    }

    #define INSTANTIATE_MEDFILT(T)                                              \
    template void median1<T>(const T*, T*, size3_t, uint, BorderMode, uint);    \
    template void median2<T>(const T*, T*, size3_t, uint, BorderMode, uint);    \
    template void median3<T>(const T*, T*, size3_t, uint, BorderMode, uint)

    INSTANTIATE_MEDFILT(float);
    INSTANTIATE_MEDFILT(double);
    INSTANTIATE_MEDFILT(int);
    INSTANTIATE_MEDFILT(uint);
}
