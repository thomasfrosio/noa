#include "noa/common/Profiler.h"
#include "noa/cpu/fourier/Resize.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Set.h"

namespace noa::cpu::fourier {
    template<typename T>
    void crop(const T* inputs, size3_t inputs_shape, T* outputs, size3_t outputs_shape) {
        NOA_PROFILE_FUNCTION();
        if (all(inputs_shape == outputs_shape)) {
            memory::copy(inputs, outputs, getElementsFFT(inputs_shape));
            return;
        }

        size_t in_z, in_y;
        for (size_t out_z{0}; out_z < outputs_shape.z; ++out_z) {
            in_z = out_z < (outputs_shape.z + 1) / 2 ? out_z : out_z + inputs_shape.z - outputs_shape.z;
            for (size_t out_y{0}; out_y < outputs_shape.y; ++out_y) {
                in_y = out_y < (outputs_shape.y + 1) / 2 ? out_y : out_y + inputs_shape.y - outputs_shape.y;

                memory::copy(inputs + (in_z * inputs_shape.y + in_y) * (inputs_shape.x / 2 + 1),
                             outputs + (out_z * outputs_shape.y + out_y) * (outputs_shape.x / 2 + 1),
                             outputs_shape.x / 2 + 1);
            }
        }
    }

    template<typename T>
    void cropFull(const T* inputs, size3_t inputs_shape, T* outputs, size3_t outputs_shape) {
        NOA_PROFILE_FUNCTION();
        if (all(inputs_shape == outputs_shape)) {
            memory::copy(inputs, outputs, getElements(inputs_shape));
            return;
        }

        size3_t offset = inputs_shape - outputs_shape;
        size3_t start_2nd_half = (outputs_shape + 1ul) / 2ul;

        size_t in_z, in_y;
        for (size_t out_z{0}; out_z < outputs_shape.z; ++out_z) {
            in_z = out_z < start_2nd_half.z ? out_z : out_z + offset.z;
            for (size_t out_y{0}; out_y < outputs_shape.y; ++out_y) {
                in_y = out_y < start_2nd_half.y ? out_y : out_y + offset.y;

                memory::copy(inputs + (in_z * inputs_shape.y + in_y) * inputs_shape.x,
                             outputs + (out_z * outputs_shape.y + out_y) * outputs_shape.x,
                             start_2nd_half.x);

                memory::copy(inputs + (in_z * inputs_shape.y + in_y) * inputs_shape.x + start_2nd_half.x + offset.x,
                             outputs + (out_z * outputs_shape.y + out_y) * outputs_shape.x + start_2nd_half.x,
                             outputs_shape.x / 2);
            }
        }
    }

    template<typename T>
    void pad(const T* inputs, size3_t inputs_shape, T* outputs, size3_t outputs_shape) {
        NOA_PROFILE_FUNCTION();
        if (all(inputs_shape == outputs_shape)) {
            memory::copy(inputs, outputs, getElementsFFT(inputs_shape));
            return;
        }

        memory::set(outputs, getElementsFFT(outputs_shape), T{0});
        size_t out_z, out_y;
        for (size_t in_z{0}; in_z < inputs_shape.z; ++in_z) {
            out_z = in_z < (inputs_shape.z + 1) / 2 ? in_z : in_z + outputs_shape.z - inputs_shape.z;
            for (size_t in_y{0}; in_y < inputs_shape.y; ++in_y) {
                out_y = in_y < (inputs_shape.y + 1) / 2 ? in_y : in_y + outputs_shape.y - inputs_shape.y;
                memory::copy(inputs + (in_z * inputs_shape.y + in_y) * (inputs_shape.x / 2 + 1),
                             outputs + (out_z * outputs_shape.y + out_y) * (outputs_shape.x / 2 + 1),
                             inputs_shape.x / 2 + 1);
            }
        }
    }

    template<typename T>
    void padFull(const T* inputs, size3_t inputs_shape, T* outputs, size3_t outputs_shape) {
        NOA_PROFILE_FUNCTION();
        if (all(inputs_shape == outputs_shape)) {
            memory::copy(inputs, outputs, getElements(inputs_shape));
            return;
        }

        memory::set(outputs, getElements(outputs_shape), T{0});
        size3_t offset = outputs_shape - inputs_shape;
        size3_t start_2nd_half = (inputs_shape + 1ul) / 2ul;

        size_t out_z, out_y;
        for (size_t in_z{0}; in_z < inputs_shape.z; ++in_z) {
            out_z = in_z < start_2nd_half.z ? in_z : in_z + offset.z;
            for (size_t in_y{0}; in_y < inputs_shape.y; ++in_y) {
                out_y = in_y < start_2nd_half.y ? in_y : in_y + offset.y;

                memory::copy(inputs + (in_z * inputs_shape.y + in_y) * inputs_shape.x,
                             outputs + (out_z * outputs_shape.y + out_y) * outputs_shape.x,
                             start_2nd_half.x);
                memory::copy(inputs + (in_z * inputs_shape.y + in_y) * inputs_shape.x + start_2nd_half.x,
                             outputs + (out_z * outputs_shape.y + out_y) * outputs_shape.x + start_2nd_half.x + offset.x,
                             inputs_shape.x / 2);
            }
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T)                          \
    template void crop<T>(const T*, size3_t, T*, size3_t);      \
    template void cropFull<T>(const T*, size3_t, T*, size3_t);  \
    template void pad<T>(const T*, size3_t, T*, size3_t);       \
    template void padFull<T>(const T*, size3_t, T*, size3_t)

    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(cfloat_t);
    NOA_INSTANTIATE_RESIZE_(cdouble_t);
}
