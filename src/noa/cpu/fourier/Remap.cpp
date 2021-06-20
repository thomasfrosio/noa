#include "noa/Math.h"
#include "noa/Profiler.h"
#include "noa/cpu/fourier/Remap.h"
#include "noa/cpu/memory/Copy.h"

namespace noa::fourier {
    template<typename T>
    void HC2H(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        size_t half_x = (shape.x / 2 + 1);
        size_t base_z, base_y;
        for (size_t z = 0; z < shape.z; ++z) {
            base_z = math::iFFTShift(z, shape.z);
            for (size_t y = 0; y < shape.y; ++y) {
                base_y = math::iFFTShift(y, shape.y);
                memory::copy(in + (z * shape.y + y) * half_x,
                             out + (base_z * shape.y + base_y) * half_x,
                             half_x);
            }
        }
    }

    template<typename T>
    void H2HC(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        size_t half_x = (shape.x / 2 + 1);
        size_t base_z, base_y;
        for (size_t z = 0; z < shape.z; ++z) {
            base_z = math::FFTShift(z, shape.z);
            for (size_t y = 0; y < shape.y; ++y) {
                base_y = math::FFTShift(y, shape.y);
                memory::copy(in + (z * shape.y + y) * half_x,
                             out + (base_z * shape.y + base_y) * half_x,
                             half_x);
            }
        }
    }

    // TODO Replace the x loop with 2 memcpy to remove iFFTShift.
    template<typename T>
    void FC2F(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        size3_t base;
        for (size_t z = 0; z < shape.z; ++z) {
            base.z = math::iFFTShift(z, shape.z);
            for (size_t y = 0; y < shape.y; ++y) {
                base.y = math::iFFTShift(y, shape.y);
                for (size_t x = 0; x < shape.x; ++x) {
                    base.x = math::iFFTShift(x, shape.x);
                    out[(base.z * shape.y + base.y) * shape.x + base.x] = in[(z * shape.y + y) * shape.x + x];
                }
            }
        }
    }

    // TODO Replace the x loop with 2 memcpy to get remove iFFTShift.
    template<typename T>
    void F2FC(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        size3_t base;
        for (size_t z = 0; z < shape.z; ++z) {
            base.z = math::FFTShift(z, shape.z);
            for (size_t y = 0; y < shape.y; ++y) {
                base.y = math::FFTShift(y, shape.y);
                for (size_t x = 0; x < shape.x; ++x) {
                    base.x = math::FFTShift(x, shape.x);
                    out[(base.z * shape.y + base.y) * shape.x + base.x] = in[(z * shape.y + y) * shape.x + x];
                }
            }
        }
    }

    template<typename T>
    void H2F(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        size_t half_x = shape.x / 2 + 1;

        // Copy first non-redundant half.
        for (size_t z = 0; z < shape.z; ++z)
            for (size_t y = 0; y < shape.y; ++y)
                memory::copy(in + (z * shape.y + y) * half_x, out + (z * shape.y + y) * shape.x, half_x);

        // Compute the redundant elements.
        for (size_t z = 0; z < shape.z; ++z) {
            size_t in_z = z ? shape.z - z : 0;
            for (size_t y = 0; y < shape.y; ++y) {
                size_t in_y = y ? shape.y - y : 0;
                for (size_t x = half_x; x < shape.x; ++x) {
                    T value = in[(in_z * shape.y + in_y) * half_x + shape.x - x];
                    if constexpr (traits::is_complex_v<T>)
                        out[(z * shape.y + y) * shape.x + x] = math::conj(value);
                    else
                        out[(z * shape.y + y) * shape.x + x] = value;
                }
            }
        }
    }

    template<typename T>
    void F2H(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        size_t half_x = shape.x / 2 + 1;
        for (size_t z = 0; z < shape.z; ++z)
            for (size_t y = 0; y < shape.y; ++y)
                memory::copy(in + (z * shape.y + y) * shape.x, out + (z * shape.y + y) * half_x, half_x);
    }

    template<typename T>
    void FC2H(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        size_t half = shape.x / 2 + 1;
        size_t base_z, base_y;
        for (size_t z = 0; z < shape.z; ++z) {
            base_z = math::iFFTShift(z, shape.z);
            for (size_t y = 0; y < shape.y; ++y) {
                base_y = math::iFFTShift(y, shape.y);
                for (size_t x = 0; x < half; ++x) {
                    out[(base_z * shape.y + base_y) * half + x] =
                            in[(z * shape.y + y) * shape.x + math::FFTShift(x, shape.x)];
                }
            }
        }
    }

    #define INSTANTIATE_RESIZE(T)                   \
    template void HC2H<T>(const T*, T*, size3_t);   \
    template void H2HC<T>(const T*, T*, size3_t);   \
    template void FC2F<T>(const T*, T*, size3_t);   \
    template void F2FC<T>(const T*, T*, size3_t);   \
    template void H2F<T>(const T*, T*, size3_t);    \
    template void F2H<T>(const T*, T*, size3_t);    \
    template void FC2H<T>(const T*, T*, size3_t)

    INSTANTIATE_RESIZE(float);
    INSTANTIATE_RESIZE(double);
    INSTANTIATE_RESIZE(cfloat_t);
    INSTANTIATE_RESIZE(cdouble_t);
}
