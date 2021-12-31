#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Set.h"
#include "noa/cpu/fft/Resize.h"

namespace noa::cpu::fft::details {
    template<typename T>
    void cropH2H_(const T* inputs, size3_t input_pitch, size3_t input_shape,
                  T* outputs, size3_t output_pitch, size3_t output_shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        NOA_ASSERT(all(input_shape >= output_shape));
        NOA_ASSERT(all(input_pitch >= input_shape / 2 + 1) && all(output_pitch >= output_shape / 2 + 1));
        if (all(input_shape == output_shape))
            return cpu::memory::copy(inputs, input_pitch, outputs, output_pitch, shapeFFT(input_shape), batches);

        size3_t limit = {output_shape.x / 2 + 1,
                         (output_shape.y + 1) / 2,
                         (output_shape.z + 1) / 2};
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            for (size_t out_z = 0; out_z < output_shape.z; ++out_z) {
                size_t in_z = out_z < limit.z ? out_z : out_z + input_shape.z - output_shape.z;
                for (size_t out_y = 0; out_y < output_shape.y; ++out_y) {
                    size_t in_y = out_y < limit.y ? out_y : out_y + input_shape.y - output_shape.y;
                    cpu::memory::copy(input + index(in_y, in_z, input_pitch.x, input_pitch.y),
                                      output + index(out_y, out_z, output_pitch.x, output_pitch.y),
                                      limit.x);
                }
            }
        }
    }

    template<typename T>
    void cropF2F_(const T* inputs, size3_t input_pitch, size3_t input_shape,
                  T* outputs, size3_t output_pitch, size3_t output_shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        NOA_ASSERT(all(input_shape >= output_shape));
        NOA_ASSERT(all(input_pitch >= input_shape) && all(output_pitch >= output_shape));
        if (all(input_shape == output_shape))
            return cpu::memory::copy(inputs, input_pitch, outputs, output_pitch, input_shape, batches);

        size3_t offset = input_shape - output_shape;
        size3_t limit = (output_shape + 1) / 2;

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            for (size_t out_z = 0; out_z < output_shape.z; ++out_z) {
                size_t in_z = out_z < limit.z ? out_z : out_z + offset.z;
                for (size_t out_y = 0; out_y < output_shape.y; ++out_y) {
                    size_t in_y = out_y < limit.y ? out_y : out_y + offset.y;

                    size_t ibase = index(in_y, in_z, input_pitch.x, input_pitch.y);
                    size_t obase = index(out_y, out_z, output_pitch.x, output_pitch.y);
                    cpu::memory::copy(input + ibase, output + obase, limit.x);
                    cpu::memory::copy(input + ibase + limit.x + offset.x, output + obase + limit.x, output_shape.x / 2);
                }
            }
        }
    }

    template<typename T>
    void padH2H_(const T* inputs, size3_t input_pitch, size3_t input_shape,
                 T* outputs, size3_t output_pitch, size3_t output_shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        NOA_ASSERT(all(input_shape <= output_shape));
        NOA_ASSERT(all(input_pitch >= input_shape / 2 + 1) && all(output_pitch >= output_shape / 2 + 1));
        if (all(input_shape == output_shape))
            return cpu::memory::copy(inputs, input_pitch, outputs, output_pitch, shapeFFT(input_shape), batches);

        cpu::memory::set(outputs, output_pitch, shapeFFT(output_shape), batches, T{0});

        size3_t limit = {input_shape.x / 2 + 1,
                         (input_shape.y + 1) / 2,
                         (input_shape.z + 1) / 2};
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            for (size_t in_z = 0; in_z < input_shape.z; ++in_z) {
                size_t out_z = in_z < limit.z ? in_z : in_z + output_shape.z - input_shape.z;
                for (size_t in_y = 0; in_y < input_shape.y; ++in_y) {
                    size_t out_y = in_y < limit.y ? in_y : in_y + output_shape.y - input_shape.y;
                    cpu::memory::copy(input + index(in_y, in_z, input_pitch.x, input_pitch.y),
                                      output + index(out_y, out_z, output_pitch.x, output_pitch.y),
                                      limit.x);
                }
            }
        }
    }

    template<typename T>
    void padF2F_(const T* inputs, size3_t input_pitch, size3_t input_shape,
                 T* outputs, size3_t output_pitch, size3_t output_shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        NOA_ASSERT(all(input_shape <= output_shape));
        NOA_ASSERT(all(input_pitch >= input_shape) && all(output_pitch >= output_shape));
        if (all(input_shape == output_shape))
            return cpu::memory::copy(inputs, input_pitch, outputs, output_pitch, input_shape, batches);

        cpu::memory::set(outputs, output_pitch, output_shape, batches, T{0});

        size3_t offset = output_shape - input_shape;
        size3_t limit = (input_shape + 1) / 2;

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            for (size_t in_z = 0; in_z < input_shape.z; ++in_z) {
                size_t out_z = in_z < limit.z ? in_z : in_z + offset.z;
                for (size_t in_y = 0; in_y < input_shape.y; ++in_y) {
                    size_t out_y = in_y < limit.y ? in_y : in_y + offset.y;

                    size_t ibase = index(in_y, in_z, input_pitch.x, input_pitch.y);
                    size_t obase = index(out_y, out_z, output_pitch.x, output_pitch.y);

                    cpu::memory::copy(input + ibase, output + obase, limit.x);
                    cpu::memory::copy(input + ibase + limit.x, output + obase + limit.x + offset.x, input_shape.x / 2);
                }
            }
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T)                                                      \
    template void cropH2H_<T>(const T*, size3_t, size3_t, T*, size3_t, size3_t, size_t);    \
    template void cropF2F_<T>(const T*, size3_t, size3_t, T*, size3_t, size3_t, size_t);    \
    template void padH2H_<T>(const T*, size3_t, size3_t, T*, size3_t, size3_t, size_t);     \
    template void padF2F_<T>(const T*, size3_t, size3_t, T*, size3_t, size3_t, size_t)

    NOA_INSTANTIATE_RESIZE_(half_t);
    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(chalf_t);
    NOA_INSTANTIATE_RESIZE_(cfloat_t);
    NOA_INSTANTIATE_RESIZE_(cdouble_t);
}
