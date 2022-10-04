#include "noa/common/Assert.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Set.h"
#include "noa/cpu/fft/Resize.h"

namespace noa::cpu::fft::details {
    template<typename T>
    void cropH2H(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                 AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape) {
        NOA_ASSERT(all(input_shape >= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return cpu::memory::copy(input.get(), dim4_t(input.strides()),
                                     output.get(), dim4_t(output.strides()),
                                     input_shape.fft());

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t oj = 0; oj < output_shape[1]; ++oj) {
                for (dim_t ok = 0; ok < output_shape[2]; ++ok) {
                    for (dim_t l = 0; l < output_shape[3] / 2 + 1; ++l) {
                        const dim_t ij = oj < (output_shape[1] + 1) / 2 ? oj : oj + input_shape[1] - output_shape[1];
                        const dim_t ik = ok < (output_shape[2] + 1) / 2 ? ok : ok + input_shape[2] - output_shape[2];
                        output(i, oj, ok, l) = input(i, ij, ik, l);
                    }
                }
            }
        }
    }

    template<typename T>
    void cropF2F(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                 AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape) {
        NOA_ASSERT(all(input_shape >= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return cpu::memory::copy(input.get(), dim4_t(input.strides()),
                                     output.get(), dim4_t(output.strides()),
                                     input_shape);

        const dim4_t offset(input_shape - output_shape);
        const dim4_t limit((output_shape + 1) / 2);

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t oj = 0; oj < output_shape[1]; ++oj) {
                for (dim_t ok = 0; ok < output_shape[2]; ++ok) {
                    const dim_t ij = oj < limit[1] ? oj : oj + offset[1];
                    const dim_t ik = ok < limit[2] ? ok : ok + offset[2];

                    const auto input_row = input[i][ij][ik];
                    const auto output_row = output[i][oj][ok];

                    for (dim_t l = 0; l < limit[3]; ++l)
                        output_row[l] = input_row[l];

                    for (dim_t l = 0; l < output_shape[3] / 2; ++l)
                        output_row[l + limit[3]] = input_row[l + limit[3] + offset[3]];
                }
            }
        }
    }

    template<typename T>
    void padH2H(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape) {
        NOA_ASSERT(all(input_shape <= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return cpu::memory::copy(input.get(), dim4_t(input.strides()),
                                     output.get(), dim4_t(output.strides()),
                                     input_shape.fft());

        cpu::memory::set(output.get(), dim4_t(output.strides()), output_shape.fft(), T{0});

        for (dim_t i = 0; i < input_shape[0]; ++i) {
            for (dim_t ij = 0; ij < input_shape[1]; ++ij) {
                for (dim_t ik = 0; ik < input_shape[2]; ++ik) {
                    for (dim_t l = 0; l < input_shape[3] / 2 + 1; ++l) {
                        const dim_t oj = ij < (input_shape[1] + 1) / 2 ? ij : ij + output_shape[1] - input_shape[1];
                        const dim_t ok = ik < (input_shape[2] + 1) / 2 ? ik : ik + output_shape[2] - input_shape[2];
                        output(i, oj, ok, l) = input(i, ij, ik, l);
                    }
                }
            }
        }
    }

    template<typename T>
    void padF2F(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape) {
        NOA_ASSERT(all(input_shape <= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return cpu::memory::copy(input.get(), dim4_t(input.strides()),
                                     output.get(), dim4_t(output.strides()),
                                     input_shape);

        cpu::memory::set(output.get(), dim4_t(output.strides()), output_shape, T{0});

        const dim4_t offset(output_shape - input_shape);
        const dim4_t limit((input_shape + 1) / 2);

        for (dim_t i = 0; i < input_shape[0]; ++i) {
            for (dim_t ij = 0; ij < input_shape[1]; ++ij) {
                for (dim_t ik = 0; ik < input_shape[2]; ++ik) {
                    const dim_t oj = ij < limit[1] ? ij : ij + offset[1];
                    const dim_t ok = ik < limit[2] ? ik : ik + offset[2];

                    const auto input_row = input[i][ij][ik];
                    const auto output_row = output[i][oj][ok];

                    for (dim_t l = 0; l < limit[3]; ++l)
                        output_row[l] = input_row[l];

                    for (dim_t l = 0; l < input_shape[3] / 2; ++l)
                        output_row[l + limit[3] + offset[3]] = input_row[l + limit[3]];
                }
            }
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T)                                                                              \
    template void cropH2H<T>(AccessorRestrict<const T, 4, dim_t>, dim4_t, AccessorRestrict<T, 4, dim_t>, dim4_t);   \
    template void cropF2F<T>(AccessorRestrict<const T, 4, dim_t>, dim4_t, AccessorRestrict<T, 4, dim_t>, dim4_t);   \
    template void padH2H<T>(AccessorRestrict<const T, 4, dim_t>, dim4_t, AccessorRestrict<T, 4, dim_t>, dim4_t);    \
    template void padF2F<T>(AccessorRestrict<const T, 4, dim_t>, dim4_t, AccessorRestrict<T, 4, dim_t>, dim4_t)

    NOA_INSTANTIATE_RESIZE_(half_t);
    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(chalf_t);
    NOA_INSTANTIATE_RESIZE_(cfloat_t);
    NOA_INSTANTIATE_RESIZE_(cdouble_t);
}
