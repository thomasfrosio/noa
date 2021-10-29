#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/fft/Remap.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"

namespace noa::cpu::fft::details {
    template<typename T>
    void hc2h(const T* inputs, T* outputs, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        size_t half_x = (shape.x / 2 + 1);
        size_t base_z, base_y;
        size_t elements = noa::elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements * batch;

            for (size_t z = 0; z < shape.z; ++z) {
                base_z = math::iFFTShift(z, shape.z);
                for (size_t y = 0; y < shape.y; ++y) {
                    base_y = math::iFFTShift(y, shape.y);
                    memory::copy(input + (z * shape.y + y) * half_x,
                                 output + (base_z * shape.y + base_y) * half_x,
                                 half_x);
                }
            }
        }
    }

    template<typename T>
    void h2hc(const T* inputs, T* outputs, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        size_t size_row = (shape.x / 2 + 1);
        size_t base_z, base_y;
        size_t elements = noa::elementsFFT(shape);

        if (inputs == outputs) {
            memory::PtrHost<T> buffer;
            if ((shape.y != 1 && shape.y % 2) || (shape.z != 1 && shape.z % 2)) {
                NOA_THROW("In-place {} is only available when y and z have an even number of elements", Remap::H2HC);
            } else {
                // E.g. from h = [0,1,2,3,-4,-3,-2,-1] to hc = [-4,-3,-2,-1,0,1,2,3]
                // Simple swap is OK.
                buffer.reset(size_row);
                for (size_t batch = 0; batch < batches; ++batch) {
                    T* output = outputs + elements * batch;

                    for (size_t z = 0; z < shape.z; ++z) {
                        base_z = math::FFTShift(z, shape.z);
                        for (size_t y = 0; y < noa::math::max(shape.y / 2, size_t{1}); ++y) { // if 1D, loop once
                            base_y = math::FFTShift(y, shape.y);

                            T* i_in = output + (z * shape.y + y) * size_row;
                            T* i_out = output + (base_z * shape.y + base_y) * size_row;

                            memory::copy(i_out, buffer.get(), size_row);
                            memory::copy(i_in, i_out, size_row);
                            memory::copy(buffer.get(), i_in, size_row);
                        }
                    }
                }
            }
        } else {
            for (size_t batch = 0; batch < batches; ++batch) {
                const T* input = inputs + elements * batch;
                T* output = outputs + elements * batch;

                for (size_t z = 0; z < shape.z; ++z) {
                    base_z = math::FFTShift(z, shape.z);
                    for (size_t y = 0; y < shape.y; ++y) {
                        base_y = math::FFTShift(y, shape.y);
                        memory::copy(input + (z * shape.y + y) * size_row,
                                     output + (base_z * shape.y + base_y) * size_row,
                                     size_row);
                    }
                }
            }
        }
    }

    template<typename T>
    void fc2f(const T* inputs, T* outputs, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        size3_t base;
        size_t elements = noa::elements(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements * batch;

            for (size_t z = 0; z < shape.z; ++z) {
                base.z = math::iFFTShift(z, shape.z);
                for (size_t y = 0; y < shape.y; ++y) {
                    base.y = math::iFFTShift(y, shape.y);
                    for (size_t x = 0; x < shape.x; ++x) {
                        base.x = math::iFFTShift(x, shape.x);
                        output[(base.z * shape.y + base.y) * shape.x + base.x] =
                                input[(z * shape.y + y) * shape.x + x];
                    }
                }
            }
        }
    }

    template<typename T>
    void f2fc(const T* inputs, T* outputs, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        size3_t base;
        size_t elements = noa::elements(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements * batch;

            for (size_t z = 0; z < shape.z; ++z) {
                base.z = math::FFTShift(z, shape.z);
                for (size_t y = 0; y < shape.y; ++y) {
                    base.y = math::FFTShift(y, shape.y);
                    for (size_t x = 0; x < shape.x; ++x) {
                        base.x = math::FFTShift(x, shape.x);
                        output[(base.z * shape.y + base.y) * shape.x + base.x] =
                                input[(z * shape.y + y) * shape.x + x];
                    }
                }
            }
        }
    }

    template<typename T>
    void h2f(const T* inputs, T* outputs, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        size_t half_x = shape.x / 2 + 1;

        size_t elements = noa::elements(shape);
        size_t elements_fft = noa::elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements_fft * batch;
            T* output = outputs + elements * batch;

            for (size_t z = 0; z < shape.z; ++z) {
                size_t in_z = z ? shape.z - z : 0;
                for (size_t y = 0; y < shape.y; ++y) {
                    size_t in_y = y ? shape.y - y : 0;

                    // Copy first non-redundant half.
                    memory::copy(input + (z * shape.y + y) * half_x,
                                 output + (z * shape.y + y) * shape.x,
                                 half_x);

                    // Compute the redundant elements.
                    for (size_t x = half_x; x < shape.x; ++x) {
                        T value = input[(in_z * shape.y + in_y) * half_x + shape.x - x];
                        if constexpr (traits::is_complex_v<T>)
                            output[(z * shape.y + y) * shape.x + x] = math::conj(value);
                        else
                            output[(z * shape.y + y) * shape.x + x] = value;
                    }
                }
            }
        }
    }

    template<typename T>
    void f2h(const T* inputs, T* outputs, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        size_t half_x = shape.x / 2 + 1;

        size_t elements = noa::elements(shape);
        size_t elements_fft = noa::elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements_fft * batch;

            for (size_t z = 0; z < shape.z; ++z)
                for (size_t y = 0; y < shape.y; ++y)
                    memory::copy(input + (z * shape.y + y) * shape.x,
                                 output + (z * shape.y + y) * half_x,
                                 half_x);
        }
    }

    template<typename T>
    void hc2f(const T* inputs, T* outputs, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        size_t half_x = shape.x / 2 + 1;

        size_t elements = noa::elements(shape);
        size_t elements_fft = noa::elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements_fft * batch;
            T* output = outputs + elements * batch;

            for (size_t o_z = 0; o_z < shape.z; ++o_z) {
                size_t i_z = math::FFTShift(o_z, shape.z);
                size_t in_z = math::FFTShift(o_z ? shape.z - o_z : o_z, shape.z);

                for (size_t o_y = 0; o_y < shape.y; ++o_y) {
                    size_t i_y = math::FFTShift(o_y, shape.y);
                    size_t in_y = math::FFTShift(o_y ? shape.y - o_y : o_y, shape.y);

                    // Copy first non-redundant half.
                    memory::copy(input + (i_z * shape.y + i_y) * half_x,
                                 output + (o_z * shape.y + o_y) * shape.x,
                                 half_x);

                    // Compute the redundant elements.
                    for (size_t x = half_x; x < shape.x; ++x) {
                        T value = input[(in_z * shape.y + in_y) * half_x + shape.x - x];
                        if constexpr (traits::is_complex_v<T>)
                            output[(o_z * shape.y + o_y) * shape.x + x] = math::conj(value);
                        else
                            output[(o_z * shape.y + o_y) * shape.x + x] = value;
                    }
                }
            }
        }
    }

    template<typename T>
    void f2hc(const T* inputs, T* outputs, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        size_t half_x = shape.x / 2 + 1;

        size_t elements = noa::elements(shape);
        size_t elements_fft = noa::elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements_fft * batch;

            for (size_t z = 0; z < shape.z; ++z) {
                size_t i_offset_z = z * shape.y * shape.x;
                size_t o_offset_z = math::FFTShift(z, shape.z) * shape.y * half_x;
                for (size_t y = 0; y < shape.y; ++y) {
                    size_t i_offset = i_offset_z + y * shape.x;
                    size_t o_offset = o_offset_z + math::FFTShift(y, shape.y) * half_x;
                    memory::copy(input + i_offset, output + o_offset, half_x);
                }
            }
        }
    }

    template<typename T>
    void fc2h(const T* inputs, T* outputs, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        size_t half = shape.x / 2 + 1;
        size_t base_z, base_y;

        size_t elements = noa::elements(shape);
        size_t elements_fft = noa::elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements_fft * batch;

            for (size_t z = 0; z < shape.z; ++z) {
                base_z = math::iFFTShift(z, shape.z);
                for (size_t y = 0; y < shape.y; ++y) {
                    base_y = math::iFFTShift(y, shape.y);
                    for (size_t x = 0; x < half; ++x) {
                        output[(base_z * shape.y + base_y) * half + x] =
                                input[(z * shape.y + y) * shape.x + math::FFTShift(x, shape.x)];
                    }
                }
            }
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T)                      \
    template void hc2h<T>(const T*, T*, size3_t, size_t);   \
    template void h2hc<T>(const T*, T*, size3_t, size_t);   \
    template void fc2f<T>(const T*, T*, size3_t, size_t);   \
    template void f2fc<T>(const T*, T*, size3_t, size_t);   \
    template void h2f<T>(const T*, T*, size3_t, size_t);    \
    template void f2h<T>(const T*, T*, size3_t, size_t);    \
    template void hc2f<T>(const T*, T*, size3_t, size_t);   \
    template void f2hc<T>(const T*, T*, size3_t, size_t);   \
    template void fc2h<T>(const T*, T*, size3_t, size_t)

    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(cfloat_t);
    NOA_INSTANTIATE_RESIZE_(cdouble_t);
}
