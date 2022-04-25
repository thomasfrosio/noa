#include "noa/common/Assert.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Set.h"
#include "noa/cpu/fft/Resize.h"

namespace noa::cpu::fft::details {
    template<typename T>
    void cropH2H(const T* input, size4_t input_stride, size4_t input_shape,
                 T* output, size4_t output_stride, size4_t output_shape) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(all(input_shape >= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return cpu::memory::copy(input, input_stride, output, output_stride, input_shape.fft());

        for (size_t i = 0; i < output_shape[0]; ++i) {
            for (size_t oj = 0; oj < output_shape[1]; ++oj) {
                for (size_t ok = 0; ok < output_shape[2]; ++ok) {
                    for (size_t l = 0; l < output_shape[3] / 2 + 1; ++l) {
                        const size_t ij = oj < (output_shape[1] + 1) / 2 ? oj : oj + input_shape[1] - output_shape[1];
                        const size_t ik = ok < (output_shape[2] + 1) / 2 ? ok : ok + input_shape[2] - output_shape[2];
                        output[indexing::at(i, oj, ok, l, output_stride)] =
                                input[indexing::at(i, ij, ik, l, input_stride)];
                    }
                }
            }
        }
    }

    template<typename T>
    void cropF2F(const T* input, size4_t input_stride, size4_t input_shape,
                 T* output, size4_t output_stride, size4_t output_shape) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(all(input_shape >= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return cpu::memory::copy(input, input_stride, output, output_stride, input_shape);

        const size4_t offset(input_shape - output_shape);
        const size4_t limit((output_shape + 1) / 2);

        for (size_t i = 0; i < output_shape[0]; ++i) {
            for (size_t oj = 0; oj < output_shape[1]; ++oj) {
                for (size_t ok = 0; ok < output_shape[2]; ++ok) {
                    const size_t ij = oj < limit[1] ? oj : oj + offset[1];
                    const size_t ik = ok < limit[2] ? ok : ok + offset[2];


                    for (size_t l = 0; l < limit[3]; ++l)
                        output[indexing::at(i, oj, ok, l, output_stride)] =
                                input[indexing::at(i, ij, ik, l, input_stride)];
                    for (size_t l = 0; l < output_shape[3] / 2; ++l)
                        output[indexing::at(i, oj, ok, l, output_stride) + limit[3]] =
                                input[indexing::at(i, ij, ik, l, input_stride) + limit[3] + offset[3]];
                }
            }
        }
    }

    template<typename T>
    void padH2H(const T* input, size4_t input_stride, size4_t input_shape,
                T* output, size4_t output_stride, size4_t output_shape) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(all(input_shape <= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);
        if (all(input_shape == output_shape))
            return cpu::memory::copy(input, input_stride, output, output_stride, input_shape.fft());

        cpu::memory::set(output, output_stride, output_shape.fft(), T{0});

        for (size_t i = 0; i < input_shape[0]; ++i) {
            for (size_t ij = 0; ij < input_shape[1]; ++ij) {
                for (size_t ik = 0; ik < input_shape[2]; ++ik) {
                    for (size_t l = 0; l < input_shape[3] / 2 + 1; ++l) {
                        const size_t oj = ij < (input_shape[1] + 1) / 2 ? ij : ij + output_shape[1] - input_shape[1];
                        const size_t ok = ik < (input_shape[2] + 1) / 2 ? ik : ik + output_shape[2] - input_shape[2];
                        output[indexing::at(i, oj, ok, l, output_stride)] =
                                input[indexing::at(i, ij, ik, l, input_stride)];
                    }
                }
            }
        }
    }

    template<typename T>
    void padF2F(const T* input, size4_t input_stride, size4_t input_shape,
                T* output, size4_t output_stride, size4_t output_shape) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(all(input_shape <= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return cpu::memory::copy(input, input_stride, output, output_stride, input_shape);

        cpu::memory::set(output, output_stride, output_shape, T{0});

        const size4_t offset(output_shape - input_shape);
        const size4_t limit((input_shape + 1) / 2);

        for (size_t i = 0; i < input_shape[0]; ++i) {
            for (size_t ij = 0; ij < input_shape[1]; ++ij) {
                for (size_t ik = 0; ik < input_shape[2]; ++ik) {
                    const size_t oj = ij < limit[1] ? ij : ij + offset[1];
                    const size_t ok = ik < limit[2] ? ik : ik + offset[2];

                    for (size_t l = 0; l < limit[3]; ++l)
                        output[indexing::at(i, oj, ok, l, output_stride)] =
                                input[indexing::at(i, ij, ik, l, input_stride)];
                    for (size_t l = 0; l < input_shape[3] / 2; ++l)
                        output[indexing::at(i, oj, ok, l, output_stride) + limit[3] + offset[3]] =
                                input[indexing::at(i, ij, ik, l, input_stride) + limit[3]];
                }
            }
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T)                                          \
    template void cropH2H<T>(const T*, size4_t, size4_t, T*, size4_t, size4_t); \
    template void cropF2F<T>(const T*, size4_t, size4_t, T*, size4_t, size4_t); \
    template void padH2H<T>(const T*, size4_t, size4_t, T*, size4_t, size4_t);  \
    template void padF2F<T>(const T*, size4_t, size4_t, T*, size4_t, size4_t)

    NOA_INSTANTIATE_RESIZE_(half_t);
    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(chalf_t);
    NOA_INSTANTIATE_RESIZE_(cfloat_t);
    NOA_INSTANTIATE_RESIZE_(cdouble_t);
}
