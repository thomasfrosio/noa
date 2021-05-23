#include "noa/Profiler.h"
#include "noa/cpu/fourier/Resize.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Set.h"

namespace Noa::Fourier {
    template<typename T>
    void crop(const T* in, size3_t shape_in, T* out, size3_t shape_out) {
        NOA_PROFILE_FUNCTION();
        if (shape_in == shape_out) {
            Memory::copy(in, out, getElementsFFT(shape_in));
            return;
        }

        size_t in_z, in_y;
        for (size_t out_z{0}; out_z < shape_out.z; ++out_z) {
            in_z = out_z < (shape_out.z + 1) / 2 ? out_z : out_z + shape_in.z - shape_out.z;
            for (size_t out_y{0}; out_y < shape_out.y; ++out_y) {
                in_y = out_y < (shape_out.y + 1) / 2 ? out_y : out_y + shape_in.y - shape_out.y;

                Memory::copy(in + (in_z * shape_in.y + in_y) * (shape_in.x / 2 + 1),
                             out + (out_z * shape_out.y + out_y) * (shape_out.x / 2 + 1),
                             shape_out.x / 2 + 1);
            }
        }
    }

    template<typename T>
    void cropFull(const T* in, size3_t shape_in, T* out, size3_t shape_out) {
        NOA_PROFILE_FUNCTION();
        if (shape_in == shape_out) {
            Memory::copy(in, out, getElements(shape_in));
            return;
        }

        size3_t offset = shape_in - shape_out;
        size3_t start_2nd_half = (shape_out + 1ul) / 2ul;

        size_t in_z, in_y;
        for (size_t out_z{0}; out_z < shape_out.z; ++out_z) {
            in_z = out_z < start_2nd_half.z ? out_z : out_z + offset.z;
            for (size_t out_y{0}; out_y < shape_out.y; ++out_y) {
                in_y = out_y < start_2nd_half.y ? out_y : out_y + offset.y;

                Memory::copy(in + (in_z * shape_in.y + in_y) * shape_in.x,
                             out + (out_z * shape_out.y + out_y) * shape_out.x,
                             start_2nd_half.x);

                Memory::copy(in + (in_z * shape_in.y + in_y) * shape_in.x + start_2nd_half.x + offset.x,
                             out + (out_z * shape_out.y + out_y) * shape_out.x + start_2nd_half.x,
                             shape_out.x / 2);
            }
        }
    }

    template<typename T>
    void pad(const T* in, size3_t shape_in, T* out, size3_t shape_out) {
        NOA_PROFILE_FUNCTION();
        if (shape_in == shape_out) {
            Memory::copy(in, out, getElementsFFT(shape_in));
            return;
        }

        Memory::set(out, getElementsFFT(shape_out), T{0});
        size_t out_z, out_y;
        for (size_t in_z{0}; in_z < shape_in.z; ++in_z) {
            out_z = in_z < (shape_in.z + 1) / 2 ? in_z : in_z + shape_out.z - shape_in.z;
            for (size_t in_y{0}; in_y < shape_in.y; ++in_y) {
                out_y = in_y < (shape_in.y + 1) / 2 ? in_y : in_y + shape_out.y - shape_in.y;
                Memory::copy(in + (in_z * shape_in.y + in_y) * (shape_in.x / 2 + 1),
                             out + (out_z * shape_out.y + out_y) * (shape_out.x / 2 + 1),
                             shape_in.x / 2 + 1);
            }
        }
    }

    template<typename T>
    void padFull(const T* in, size3_t shape_in, T* out, size3_t shape_out) {
        NOA_PROFILE_FUNCTION();
        if (shape_in == shape_out) {
            Memory::copy(in, out, getElements(shape_in));
            return;
        }

        Memory::set(out, getElements(shape_out), T{0});
        size3_t offset = shape_out - shape_in;
        size3_t start_2nd_half = (shape_in + 1ul) / 2ul;

        size_t out_z, out_y;
        for (size_t in_z{0}; in_z < shape_in.z; ++in_z) {
            out_z = in_z < start_2nd_half.z ? in_z : in_z + offset.z;
            for (size_t in_y{0}; in_y < shape_in.y; ++in_y) {
                out_y = in_y < start_2nd_half.y ? in_y : in_y + offset.y;

                Memory::copy(in + (in_z * shape_in.y + in_y) * shape_in.x,
                             out + (out_z * shape_out.y + out_y) * shape_out.x,
                             start_2nd_half.x);
                Memory::copy(in + (in_z * shape_in.y + in_y) * shape_in.x + start_2nd_half.x,
                             out + (out_z * shape_out.y + out_y) * shape_out.x + start_2nd_half.x + offset.x,
                             shape_in.x / 2);
            }
        }
    }

    #define INSTANTIATE_RESIZE(T)                               \
    template void crop<T>(const T*, size3_t, T*, size3_t);      \
    template void cropFull<T>(const T*, size3_t, T*, size3_t);  \
    template void pad<T>(const T*, size3_t, T*, size3_t);       \
    template void padFull<T>(const T*, size3_t, T*, size3_t)

    INSTANTIATE_RESIZE(float);
    INSTANTIATE_RESIZE(double);
    INSTANTIATE_RESIZE(cfloat_t);
    INSTANTIATE_RESIZE(cdouble_t);
}
