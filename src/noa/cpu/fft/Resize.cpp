#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Set.h"
#include "noa/cpu/fft/Resize.h"

namespace noa::cpu::fft {
    template<typename T>
    void crop(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        if (all(input_shape == output_shape))
            return memory::copy(inputs, outputs, elementsFFT(input_shape) * batches);

        size_t input_elements = elementsFFT(input_shape);
        size_t output_elements = elementsFFT(output_shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + input_elements * batch;
            T* output = outputs + output_elements * batch;

            for (size_t out_z = 0; out_z < output_shape.z; ++out_z) {
                size_t in_z = out_z < (output_shape.z + 1) / 2 ?
                              out_z : out_z + input_shape.z - output_shape.z;

                for (size_t out_y = 0; out_y < output_shape.y; ++out_y) {
                    size_t in_y = out_y < (output_shape.y + 1) / 2 ?
                                  out_y : out_y + input_shape.y - output_shape.y;

                    memory::copy(input + (in_z * input_shape.y + in_y) * (input_shape.x / 2 + 1),
                                 output + (out_z * output_shape.y + out_y) * (output_shape.x / 2 + 1),
                                 output_shape.x / 2 + 1);
                }
            }
        }
    }

    template<typename T>
    void cropFull(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        if (all(input_shape == output_shape))
            return memory::copy(inputs, outputs, elements(input_shape) * batches);

        size3_t offset = input_shape - output_shape;
        size3_t start_2nd_half = (output_shape + 1ul) / 2ul;

        size_t input_elements = elements(input_shape);
        size_t output_elements = elements(output_shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + input_elements * batch;
            T* output = outputs + output_elements * batch;

            for (size_t out_z = 0; out_z < output_shape.z; ++out_z) {
                size_t in_z = out_z < start_2nd_half.z ?
                              out_z : out_z + offset.z;
                for (size_t out_y = 0; out_y < output_shape.y; ++out_y) {
                    size_t in_y = out_y < start_2nd_half.y ?
                                  out_y : out_y + offset.y;

                    memory::copy(input + (in_z * input_shape.y + in_y) * input_shape.x,
                                 output + (out_z * output_shape.y + out_y) * output_shape.x,
                                 start_2nd_half.x);

                    memory::copy(input + (in_z * input_shape.y + in_y) * input_shape.x + start_2nd_half.x + offset.x,
                                 output + (out_z * output_shape.y + out_y) * output_shape.x + start_2nd_half.x,
                                 output_shape.x / 2);
                }
            }
        }
    }

    template<typename T>
    void pad(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        if (all(input_shape == output_shape))
            return memory::copy(inputs, outputs, elementsFFT(input_shape) * batches);

        size_t input_elements = elementsFFT(input_shape);
        size_t output_elements = elementsFFT(output_shape);
        memory::set(outputs, output_elements * batches, T{0});

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + input_elements * batch;
            T* output = outputs + output_elements * batch;

            for (size_t in_z = 0; in_z < input_shape.z; ++in_z) {
                size_t out_z = in_z < (input_shape.z + 1) / 2 ?
                               in_z : in_z + output_shape.z - input_shape.z;
                for (size_t in_y = 0; in_y < input_shape.y; ++in_y) {
                    size_t out_y = in_y < (input_shape.y + 1) / 2 ?
                                   in_y : in_y + output_shape.y - input_shape.y;
                    memory::copy(input + (in_z * input_shape.y + in_y) * (input_shape.x / 2 + 1),
                                 output + (out_z * output_shape.y + out_y) * (output_shape.x / 2 + 1),
                                 input_shape.x / 2 + 1);
                }
            }
        }
    }

    template<typename T>
    void padFull(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        if (all(input_shape == output_shape))
            return memory::copy(inputs, outputs, elements(input_shape) * batches);

        size_t input_elements = elements(input_shape);
        size_t output_elements = elements(output_shape);
        memory::set(outputs, output_elements * batches, T{0});

        size3_t offset = output_shape - input_shape;
        size3_t start_half = (input_shape + 1ul) / 2ul;

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + input_elements * batch;
            T* output = outputs + output_elements * batch;

            for (size_t in_z{0}; in_z < input_shape.z; ++in_z) {
                size_t out_z = in_z < start_half.z ?
                               in_z : in_z + offset.z;
                for (size_t in_y{0}; in_y < input_shape.y; ++in_y) {
                    size_t out_y = in_y < start_half.y ?
                                   in_y : in_y + offset.y;

                    memory::copy(input + (in_z * input_shape.y + in_y) * input_shape.x,
                                 output + (out_z * output_shape.y + out_y) * output_shape.x,
                                 start_half.x);
                    memory::copy(input + (in_z * input_shape.y + in_y) * input_shape.x + start_half.x,
                                 output + (out_z * output_shape.y + out_y) * output_shape.x + start_half.x + offset.x,
                                 input_shape.x / 2);
                }
            }
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T)                                  \
    template void crop<T>(const T*, size3_t, T*, size3_t, size_t);      \
    template void cropFull<T>(const T*, size3_t, T*, size3_t, size_t);  \
    template void pad<T>(const T*, size3_t, T*, size3_t, size_t);       \
    template void padFull<T>(const T*, size3_t, T*, size3_t, size_t)

    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(cfloat_t);
    NOA_INSTANTIATE_RESIZE_(cdouble_t);
}
