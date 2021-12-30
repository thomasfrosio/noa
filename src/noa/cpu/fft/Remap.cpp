#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/fft/Remap.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"

namespace noa::cpu::fft::details {
    template<typename T>
    void hc2h(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            for (size_t z = 0; z < shape.z; ++z) {
                size_t base_z = math::iFFTShift(z, shape.z);
                for (size_t y = 0; y < shape.y; ++y) {
                    size_t base_y = math::iFFTShift(y, shape.y);
                    memory::copy(input + index(y, z, input_pitch.x, input_pitch.y),
                                 output + index(base_y, base_z, output_pitch.x, output_pitch.y),
                                 shape.x / 2 + 1);
                }
            }
        }
    }

    template<typename T>
    void h2hc(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();

        if (inputs == outputs) {
            if ((shape.y != 1 && shape.y % 2) || (shape.z != 1 && shape.z % 2)) {
                NOA_THROW("In-place {} is only available when y and z have an even number of elements", Remap::H2HC);
            } else {
                // E.g. from h = [0,1,2,3,-4,-3,-2,-1] to hc = [-4,-3,-2,-1,0,1,2,3]
                // Simple swap is OK.
                memory::PtrHost<T> buffer(shape.x / 2 + 1);
                for (size_t batch = 0; batch < batches; ++batch) {
                    T* output = outputs + elements(output_pitch) * batch;

                    for (size_t z = 0; z < shape.z; ++z) {
                        size_t base_z = math::FFTShift(z, shape.z);
                        for (size_t y = 0; y < noa::math::max(shape.y / 2, size_t{1}); ++y) { // if 1D, loop once
                            size_t base_y = math::FFTShift(y, shape.y);

                            T* i_in = output + index(y, z, output_pitch.x, output_pitch.y);
                            T* i_out = output + index(base_y, base_z, output_pitch.x, output_pitch.y);

                            memory::copy(i_out, buffer.get(), buffer.size());
                            memory::copy(i_in, i_out, buffer.size());
                            memory::copy(buffer.get(), i_in, buffer.size());
                        }
                    }
                }
            }
        } else {
            for (size_t batch = 0; batch < batches; ++batch) {
                const T* input = inputs + elements(input_pitch) * batch;
                T* output = outputs + elements(output_pitch) * batch;

                for (size_t z = 0; z < shape.z; ++z) {
                    size_t base_z = math::FFTShift(z, shape.z);
                    for (size_t y = 0; y < shape.y; ++y) {
                        size_t base_y = math::FFTShift(y, shape.y);
                        memory::copy(input + index(y, z, input_pitch.x, input_pitch.y),
                                     output + index(base_y, base_z, output_pitch.x, output_pitch.y),
                                     shape.x / 2 + 1);
                    }
                }
            }
        }
    }

    template<typename T>
    void fc2f(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            size3_t base;
            for (size_t z = 0; z < shape.z; ++z) {
                base.z = math::iFFTShift(z, shape.z);
                for (size_t y = 0; y < shape.y; ++y) {
                    base.y = math::iFFTShift(y, shape.y);
                    for (size_t x = 0; x < shape.x; ++x) {
                        base.x = math::iFFTShift(x, shape.x);
                        output[index(base, output_pitch)] = input[index(x, y, z, input_pitch.x, input_pitch.y)];
                    }
                }
            }
        }
    }

    template<typename T>
    void f2fc(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            size3_t base;
            for (size_t z = 0; z < shape.z; ++z) {
                base.z = math::FFTShift(z, shape.z);
                for (size_t y = 0; y < shape.y; ++y) {
                    base.y = math::FFTShift(y, shape.y);
                    for (size_t x = 0; x < shape.x; ++x) {
                        base.x = math::FFTShift(x, shape.x);
                        output[index(base, output_pitch)] = input[index(x, y, z, input_pitch.x, input_pitch.y)];
                    }
                }
            }
        }
    }

    template<typename T>
    void h2f(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        size_t half_x = shape.x / 2 + 1;
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            for (size_t z = 0; z < shape.z; ++z) {
                size_t in_z = z ? shape.z - z : 0;
                for (size_t y = 0; y < shape.y; ++y) {
                    size_t in_y = y ? shape.y - y : 0;

                    // Copy first non-redundant half.
                    memory::copy(input + index(y, z, input_pitch.x, input_pitch.y),
                                 output + index(y, z, output_pitch.x, output_pitch.y),
                                 half_x);

                    // Compute the redundant elements.
                    for (size_t x = half_x; x < shape.x; ++x) {
                        T value = input[index(shape.x - x, in_y, in_z, input_pitch.x, input_pitch.y)];
                        if constexpr (traits::is_complex_v<T>)
                            output[index(x, y, z, output_pitch.x, output_pitch.y)] = math::conj(value);
                        else
                            output[index(x, y, z, output_pitch.x, output_pitch.y)] = value;
                    }
                }
            }
        }
    }

    template<typename T>
    void f2h(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            for (size_t z = 0; z < shape.z; ++z)
                for (size_t y = 0; y < shape.y; ++y)
                    memory::copy(input + index(y, z, input_pitch.x, input_pitch.y),
                                 output + index(y, z, output_pitch.x, output_pitch.y),
                                 shape.x / 2 + 1);
        }
    }

    template<typename T>
    void hc2f(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        size_t half_x = shape.x / 2 + 1;
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            for (size_t o_z = 0; o_z < shape.z; ++o_z) {
                size_t i_z = math::FFTShift(o_z, shape.z);
                size_t in_z = math::FFTShift(o_z ? shape.z - o_z : o_z, shape.z);

                for (size_t o_y = 0; o_y < shape.y; ++o_y) {
                    size_t i_y = math::FFTShift(o_y, shape.y);
                    size_t in_y = math::FFTShift(o_y ? shape.y - o_y : o_y, shape.y);

                    // Copy first non-redundant half.
                    memory::copy(input + index(i_y, i_z, input_pitch.x, input_pitch.y),
                                 output + index(o_y, o_z, output_pitch.x, output_pitch.y),
                                 half_x);

                    // Compute the redundant elements.
                    for (size_t x = half_x; x < shape.x; ++x) {
                        T value = input[index(shape.x - x, in_y, in_z, input_pitch.x, input_pitch.y)];
                        if constexpr (traits::is_complex_v<T>)
                            output[index(x, o_y, o_z, output_pitch.x, output_pitch.y)] = math::conj(value);
                        else
                            output[index(x, o_y, o_z, output_pitch.x, output_pitch.y)] = value;
                    }
                }
            }
        }
    }

    template<typename T>
    void f2hc(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            for (size_t z = 0; z < shape.z; ++z) {
                size_t i_offset_z = z * input_pitch.y * input_pitch.x;
                size_t o_offset_z = math::FFTShift(z, shape.z) * output_pitch.y * output_pitch.x;
                for (size_t y = 0; y < shape.y; ++y) {
                    size_t i_offset = i_offset_z + y * input_pitch.x;
                    size_t o_offset = o_offset_z + math::FFTShift(y, shape.y) * output_pitch.x;
                    memory::copy(input + i_offset, output + o_offset, shape.x / 2 + 1);
                }
            }
        }
    }

    template<typename T>
    void fc2h(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            for (size_t z = 0; z < shape.z; ++z) {
                size_t base_z = math::iFFTShift(z, shape.z);
                for (size_t y = 0; y < shape.y; ++y) {
                    size_t base_y = math::iFFTShift(y, shape.y);
                    for (size_t x = 0; x < shape.x / 2 + 1; ++x) {
                        output[index(x, base_y, base_z, output_pitch.x, output_pitch.y)] =
                                input[index(math::FFTShift(x, shape.x), y, z, input_pitch.x, input_pitch.y)];
                    }
                }
            }
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T)                                        \
    template void hc2h<T>(const T*, size3_t, T*, size3_t, size3_t, size_t);   \
    template void h2hc<T>(const T*, size3_t, T*, size3_t, size3_t, size_t);   \
    template void fc2f<T>(const T*, size3_t, T*, size3_t, size3_t, size_t);   \
    template void f2fc<T>(const T*, size3_t, T*, size3_t, size3_t, size_t);   \
    template void h2f<T>(const T*, size3_t, T*, size3_t, size3_t, size_t);    \
    template void f2h<T>(const T*, size3_t, T*, size3_t, size3_t, size_t);    \
    template void hc2f<T>(const T*, size3_t, T*, size3_t, size3_t, size_t);   \
    template void f2hc<T>(const T*, size3_t, T*, size3_t, size3_t, size_t);   \
    template void fc2h<T>(const T*, size3_t, T*, size3_t, size3_t, size_t)

    NOA_INSTANTIATE_RESIZE_(half_t);
    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(chalf_t);
    NOA_INSTANTIATE_RESIZE_(cfloat_t);
    NOA_INSTANTIATE_RESIZE_(cdouble_t);
}
