#include "noa/common/Assert.h"
#include "noa/common/Math.h"

#include "noa/cpu/fft/Remap.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"

namespace noa::cpu::fft::details {
    template<typename T>
    void hc2h(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape) {
        NOA_ASSERT(input != output);

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3] / 2 + 1; ++l) {
                        const size_t oj = math::iFFTShift(j, shape[1]);
                        const size_t ok = math::iFFTShift(k, shape[2]);
                        output[at(i, oj, ok, l, output_stride)] = input[at(i, j, k, l, input_stride)];
                    }
                }
            }
        }
    }

    template<typename T>
    void h2hc(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape) {
        if (input == output) {
            if ((shape[2] != 1 && shape[2] % 2) || (shape[1] != 1 && shape[1] % 2)) {
                NOA_THROW("In-place remapping is only available when dim 1 and 2 have an even number of elements");
            } else {
                // E.g. from h = [0,1,2,3,-4,-3,-2,-1] to hc = [-4,-3,-2,-1,0,1,2,3]
                // Simple swap is OK.
                for (size_t i = 0; i < shape[0]; ++i) {
                    for (size_t j = 0; j < shape[1]; ++j) {
                        for (size_t k = 0; k < noa::math::max(shape[2] / 2, size_t{1}); ++k) { // if 1D, loop once

                            const size_t base_j = math::FFTShift(j, shape[1]);
                            const size_t base_k = math::FFTShift(k, shape[2]);
                            T* i_in = output + at(i, j, k, output_stride);
                            T* i_out = output + at(i, base_j, base_k, output_stride);

                            for (size_t l = 0; l < shape[3] / 2 + 1; ++l) {
                                T tmp = i_out[l * output_stride[3]];
                                i_out[l * output_stride[3]] = i_in[l * output_stride[3]];
                                i_in[l * output_stride[3]] = tmp;
                            }
                        }
                    }
                }
            }
        } else {
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    for (size_t k = 0; k < shape[2]; ++k) {
                        for (size_t l = 0; l < shape[3] / 2 + 1; ++l) {
                            const size_t oj = math::FFTShift(j, shape[1]);
                            const size_t ok = math::FFTShift(k, shape[2]);
                            output[at(i, oj, ok, l, output_stride)] = input[at(i, j, k, l, input_stride)];
                        }
                    }
                }
            }
        }
    }

    template<typename T>
    void fc2f(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape) {
        NOA_ASSERT(input != output);

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const size_t oj = math::iFFTShift(j, shape[1]);
                        const size_t ok = math::iFFTShift(k, shape[2]);
                        const size_t ol = math::iFFTShift(l, shape[3]);
                        output[at(i, oj, ok, ol, output_stride)] = input[at(i, j, k, l, input_stride)];
                    }
                }
            }
        }
    }

    template<typename T>
    void f2fc(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape) {
        NOA_ASSERT(input != output);

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const size_t oj = math::FFTShift(j, shape[1]);
                        const size_t ok = math::FFTShift(k, shape[2]);
                        const size_t ol = math::FFTShift(l, shape[3]);
                        output[at(i, oj, ok, ol, output_stride)] = input[at(i, j, k, l, input_stride)];
                    }
                }
            }
        }
    }

    template<typename T>
    void h2f(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape) {
        NOA_ASSERT(input != output);

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {

                    const size_t in_j = j ? shape[1] - j : 0;
                    const size_t in_k = k ? shape[2] - k : 0;

                    // Copy first non-redundant half.
                    for (size_t l = 0; l < shape[3] / 2 + 1; ++l)
                        output[at(i, j, k, l, output_stride)] = input[at(i, j, k, l, input_stride)];

                    // Compute the redundant elements.
                    for (size_t l = shape[3] / 2 + 1; l < shape[3]; ++l) {
                        T value = input[at(i, in_j, in_k, shape[3] - l, input_stride)];
                        if constexpr (traits::is_complex_v<T>)
                            output[at(i, j, k, l, output_stride)] = math::conj(value);
                        else
                            output[at(i, j, k, l, output_stride)] = value;
                    }
                }
            }
        }
    }

    template<typename T>
    void f2h(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape) {
        NOA_ASSERT(input != output);
        cpu::memory::copy(input, input_stride, output, output_stride, shape.fft());
    }

    template<typename T>
    void hc2f(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape) {
        NOA_ASSERT(input != output);

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t oj = 0; oj < shape[1]; ++oj) {
                for (size_t ok = 0; ok < shape[2]; ++ok) {

                    const size_t ij = math::FFTShift(oj, shape[1]);
                    const size_t inj = math::FFTShift(oj ? shape[1] - oj : oj, shape[1]);
                    const size_t ik = math::FFTShift(ok, shape[2]);
                    const size_t ink = math::FFTShift(ok ? shape[2] - ok : ok, shape[2]);

                    // Copy first non-redundant half.
                    for (size_t l = 0; l < shape[3] / 2 + 1; ++l)
                        output[at(i, oj, ok, l, output_stride)] = input[at(i, ij, ik, l, input_stride)];

                    // Compute the redundant elements.
                    for (size_t l = shape[3] / 2 + 1; l < shape[3]; ++l) {
                        T value = input[at(i, inj, ink, shape[3] - l, input_stride)];
                        if constexpr (traits::is_complex_v<T>)
                            output[at(i, oj, ok, l, output_stride)] = math::conj(value);
                        else
                            output[at(i, oj, ok, l, output_stride)] = value;
                    }
                }
            }
        }
    }

    template<typename T>
    void f2hc(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape) {
        NOA_ASSERT(input != output);

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3] / 2 + 1; ++l) {
                        const size_t oj = math::FFTShift(j, shape[1]);
                        const size_t ok = math::FFTShift(k, shape[2]);
                        output[at(i, oj, ok, l, output_stride)] = input[at(i, j, k, l, input_stride)];
                    }
                }
            }
        }
    }

    template<typename T>
    void fc2h(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape) {
        NOA_ASSERT(input != output);

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t ol = 0; ol < shape[3] / 2 + 1; ++ol) {
                        const size_t oj = math::iFFTShift(j, shape[1]);
                        const size_t ok = math::iFFTShift(k, shape[2]);
                        const size_t il = math::FFTShift(ol, shape[3]);
                        output[at(i, oj, ok, ol, output_stride)] = input[at(i, j, k, il, input_stride)];
                    }
                }
            }
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T)                              \
    template void hc2h<T>(const T*, size4_t, T*, size4_t, size4_t); \
    template void h2hc<T>(const T*, size4_t, T*, size4_t, size4_t); \
    template void fc2f<T>(const T*, size4_t, T*, size4_t, size4_t); \
    template void f2fc<T>(const T*, size4_t, T*, size4_t, size4_t); \
    template void h2f<T>(const T*, size4_t, T*, size4_t, size4_t);  \
    template void f2h<T>(const T*, size4_t, T*, size4_t, size4_t);  \
    template void hc2f<T>(const T*, size4_t, T*, size4_t, size4_t); \
    template void f2hc<T>(const T*, size4_t, T*, size4_t, size4_t); \
    template void fc2h<T>(const T*, size4_t, T*, size4_t, size4_t)

    NOA_INSTANTIATE_RESIZE_(half_t);
    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(chalf_t);
    NOA_INSTANTIATE_RESIZE_(cfloat_t);
    NOA_INSTANTIATE_RESIZE_(cdouble_t);
}
