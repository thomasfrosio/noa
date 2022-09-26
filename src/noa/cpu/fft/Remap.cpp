#include "noa/common/Assert.h"
#include "noa/common/Math.h"

#include "noa/cpu/fft/Remap.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"

namespace noa::cpu::fft::details {
    template<typename T>
    void hc2h(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output,
              dim4_t shape) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), dim4_t(input.strides()),
                                        output.get(), dim4_t(output.strides()), shape.fft()));

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t l = 0; l < shape[3] / 2 + 1; ++l) {
                        const dim_t oj = math::iFFTShift(j, shape[1]);
                        const dim_t ok = math::iFFTShift(k, shape[2]);
                        output(i, oj, ok, l) = input(i, j, k, l);
                    }
                }
            }
        }
    }

    template<typename T>
    void h2hc(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output,
              dim4_t shape) {
        if (input.get() == output.get()) {
            if ((shape[2] != 1 && shape[2] % 2) || (shape[1] != 1 && shape[1] % 2)) {
                NOA_THROW("In-place remapping is only available when the depth and height dimensions "
                          "have an even number of elements, but got shape {}", shape);
            } else {
                // E.g. from h = [0,1,2,3,-4,-3,-2,-1] to hc = [-4,-3,-2,-1,0,1,2,3]
                // Simple swap is OK.
                NOA_ASSERT(all(dim4_t(input.strides()) == dim4_t(output.strides())));

                for (dim_t i = 0; i < shape[0]; ++i) {
                    for (dim_t j = 0; j < shape[1]; ++j) {
                        for (dim_t k = 0; k < noa::math::max(shape[2] / 2, dim_t{1}); ++k) { // if 1D, loop once

                            const dim_t base_j = math::FFTShift(j, shape[1]);
                            const dim_t base_k = math::FFTShift(k, shape[2]);
                            const auto i_in = output[i][j][k];
                            const auto i_out = output[i][base_j][base_k];

                            for (dim_t l = 0; l < shape[3] / 2 + 1; ++l) {
                                T tmp = i_out[l];
                                i_out[l] = i_in[l];
                                i_in[l] = tmp;
                            }
                        }
                    }
                }
            }
        } else {
            NOA_ASSERT(!indexing::isOverlap(input.get(), dim4_t(input.strides()),
                                            output.get(), dim4_t(output.strides()), shape.fft()));

            for (dim_t i = 0; i < shape[0]; ++i) {
                for (dim_t j = 0; j < shape[1]; ++j) {
                    for (dim_t k = 0; k < shape[2]; ++k) {
                        for (dim_t l = 0; l < shape[3] / 2 + 1; ++l) {
                            const dim_t oj = math::FFTShift(j, shape[1]);
                            const dim_t ok = math::FFTShift(k, shape[2]);
                            output(i, oj, ok, l) = input(i, j, k, l);
                        }
                    }
                }
            }
        }
    }

    template<typename T>
    void fc2f(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output,
              dim4_t shape) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), dim4_t(input.strides()),
                                        output.get(), dim4_t(output.strides()), shape));

        if (indexing::isColMajor(dim4_t(input.strides())) && indexing::isColMajor(dim4_t(output.strides()))) {
            std::swap(shape[2], shape[3]);
            std::swap(input.stride(2), input.stride(3));
            std::swap(output.stride(2), output.stride(3));
        }

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t l = 0; l < shape[3]; ++l) {
                        const dim_t oj = math::iFFTShift(j, shape[1]);
                        const dim_t ok = math::iFFTShift(k, shape[2]);
                        const dim_t ol = math::iFFTShift(l, shape[3]);
                        output(i, oj, ok, ol) = input(i, j, k, l);
                    }
                }
            }
        }
    }

    template<typename T>
    void f2fc(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output,
              dim4_t shape) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), dim4_t(input.strides()),
                                        output.get(), dim4_t(output.strides()), shape));

        if (indexing::isColMajor(dim4_t(input.strides())) && indexing::isColMajor(dim4_t(output.strides()))) {
            std::swap(shape[2], shape[3]);
            std::swap(input.stride(2), input.stride(3));
            std::swap(output.stride(2), output.stride(3));
        }

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t l = 0; l < shape[3]; ++l) {
                        const dim_t oj = math::FFTShift(j, shape[1]);
                        const dim_t ok = math::FFTShift(k, shape[2]);
                        const dim_t ol = math::FFTShift(l, shape[3]);
                        output(i, oj, ok, ol) = input(i, j, k, l);
                    }
                }
            }
        }
    }

    template<typename T>
    void h2f(AccessorRestrict<const T, 4, dim_t> input,
             AccessorRestrict<T, 4, dim_t> output,
             dim4_t shape) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), dim4_t(input.strides()), shape.fft(),
                                        output.get(), dim4_t(output.strides()), shape));

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {

                    const dim_t in_j = j ? shape[1] - j : 0;
                    const dim_t in_k = k ? shape[2] - k : 0;

                    // Copy first non-redundant half.
                    for (dim_t l = 0; l < shape[3] / 2 + 1; ++l)
                        output(i, j, k, l) = input(i, j, k, l);

                    // Compute the redundant elements.
                    for (dim_t l = shape[3] / 2 + 1; l < shape[3]; ++l) {
                        const T value = input(i, in_j, in_k, shape[3] - l);
                        if constexpr (traits::is_complex_v<T>)
                            output(i, j, k, l) = math::conj(value);
                        else
                            output(i, j, k, l) = value;
                    }
                }
            }
        }
    }

    template<typename T>
    void f2h(AccessorRestrict<const T, 4, dim_t> input,
             AccessorRestrict<T, 4, dim_t> output,
             dim4_t shape) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), dim4_t(input.strides()), shape,
                                        output.get(), dim4_t(output.strides()), shape.fft()));
        cpu::memory::copy(input.get(), dim4_t(input.strides()), output.get(), dim4_t(output.strides()), shape.fft());
    }

    template<typename T>
    void hc2f(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output,
              dim4_t shape) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), dim4_t(input.strides()), shape.fft(),
                                        output.get(), dim4_t(output.strides()), shape));

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t oj = 0; oj < shape[1]; ++oj) {
                for (dim_t ok = 0; ok < shape[2]; ++ok) {

                    const dim_t ij = math::FFTShift(oj, shape[1]);
                    const dim_t inj = math::FFTShift(oj ? shape[1] - oj : oj, shape[1]);
                    const dim_t ik = math::FFTShift(ok, shape[2]);
                    const dim_t ink = math::FFTShift(ok ? shape[2] - ok : ok, shape[2]);

                    // Copy first non-redundant half.
                    for (dim_t l = 0; l < shape[3] / 2 + 1; ++l)
                        output(i, oj, ok, l) = input(i, ij, ik, l);

                    // Compute the redundant elements.
                    for (dim_t l = shape[3] / 2 + 1; l < shape[3]; ++l) {
                        const T value = input(i, inj, ink, shape[3] - l);
                        if constexpr (traits::is_complex_v<T>)
                            output(i, oj, ok, l) = math::conj(value);
                        else
                            output(i, oj, ok, l) = value;
                    }
                }
            }
        }
    }

    template<typename T>
    void f2hc(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output,
              dim4_t shape) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), dim4_t(input.strides()), shape,
                                        output.get(), dim4_t(output.strides()), shape.fft()));

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t l = 0; l < shape[3] / 2 + 1; ++l) {
                        const dim_t oj = math::FFTShift(j, shape[1]);
                        const dim_t ok = math::FFTShift(k, shape[2]);
                        output(i, oj, ok, l) = input(i, j, k, l);
                    }
                }
            }
        }
    }

    template<typename T>
    void fc2h(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output,
              dim4_t shape) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), dim4_t(input.strides()), shape,
                                        output.get(), dim4_t(output.strides()), shape.fft()));

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t ol = 0; ol < shape[3] / 2 + 1; ++ol) {
                        const dim_t oj = math::iFFTShift(j, shape[1]);
                        const dim_t ok = math::iFFTShift(k, shape[2]);
                        const dim_t il = math::FFTShift(ol, shape[3]);
                        output(i, oj, ok, ol) = input(i, j, k, il);
                    }
                }
            }
        }
    }

    template<typename T>
    void fc2hc(AccessorRestrict<const T, 4, dim_t> input,
               AccessorRestrict<T, 4, dim_t> output,
               dim4_t shape) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), dim4_t(input.strides()),
                                        output.get(), dim4_t(output.strides()), shape));

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t ol = 0; ol < shape[3] / 2 + 1; ++ol) {
                        const dim_t il = math::FFTShift(ol, shape[3]);
                        output(i, j, k, ol) = input(i, j, k, il);
                    }
                }
            }
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T)                                                                  \
    template void hc2h<T>(AccessorRestrict<const T, 4, dim_t>, AccessorRestrict<T, 4, dim_t>, dim4_t);  \
    template void h2hc<T>(AccessorRestrict<const T, 4, dim_t>, AccessorRestrict<T, 4, dim_t>, dim4_t);  \
    template void fc2f<T>(AccessorRestrict<const T, 4, dim_t>, AccessorRestrict<T, 4, dim_t>, dim4_t);  \
    template void f2fc<T>(AccessorRestrict<const T, 4, dim_t>, AccessorRestrict<T, 4, dim_t>, dim4_t);  \
    template void h2f<T>(AccessorRestrict<const T, 4, dim_t>, AccessorRestrict<T, 4, dim_t>, dim4_t);   \
    template void f2h<T>(AccessorRestrict<const T, 4, dim_t>, AccessorRestrict<T, 4, dim_t>, dim4_t);   \
    template void hc2f<T>(AccessorRestrict<const T, 4, dim_t>, AccessorRestrict<T, 4, dim_t>, dim4_t);  \
    template void f2hc<T>(AccessorRestrict<const T, 4, dim_t>, AccessorRestrict<T, 4, dim_t>, dim4_t);  \
    template void fc2hc<T>(AccessorRestrict<const T, 4, dim_t>, AccessorRestrict<T, 4, dim_t>, dim4_t); \
    template void fc2h<T>(AccessorRestrict<const T, 4, dim_t>, AccessorRestrict<T, 4, dim_t>, dim4_t)

    NOA_INSTANTIATE_RESIZE_(half_t);
    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(chalf_t);
    NOA_INSTANTIATE_RESIZE_(cfloat_t);
    NOA_INSTANTIATE_RESIZE_(cdouble_t);
}
