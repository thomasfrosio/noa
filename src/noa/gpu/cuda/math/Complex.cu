#include "noa/gpu/cuda/math/Complex.h"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/EwiseBinary.cuh"

namespace noa::cuda::math {
    template<typename T, typename _>
    void decompose(const Shared<Complex<T>[]>& input, Strides4<i64> input_strides,
                   const Shared<T[]>& real, Strides4<i64> real_strides,
                   const Shared<T[]>& imag, Strides4<i64> imag_strides,
                   Shape4<i64> shape, Stream& stream) {
        NOA_ASSERT(real.get() != imag.get() && noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(real.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(imag.get(), stream.device());

        if (noa::all(input_strides > 0)) {
            const auto order = indexing::order(input_strides, shape);
            if (noa::any(order != Vec4<i64>{0, 1, 2, 3})) {
                input_strides = indexing::reorder(input_strides, order);
                real_strides = indexing::reorder(real_strides, order);
                imag_strides = indexing::reorder(imag_strides, order);
                shape = indexing::reorder(shape, order);
            }
        }

        const auto is_contiguous =
                noa::indexing::is_contiguous(input_strides, shape) &&
                noa::indexing::is_contiguous(real_strides, shape) &&
                noa::indexing::is_contiguous(imag_strides, shape);
        if (noa::all(is_contiguous)) {
            const AccessorRestrictContiguous<const Complex<T>, 1, i64> input_accessor(input.get());
            const AccessorRestrictContiguous<T, 1, i64> real_accessor(real.get());
            const AccessorRestrictContiguous<T, 1, i64> imag_accessor(imag.get());
            const auto kernel = [=]__device__(i64 i) {
                const auto complex = input_accessor[i];
                real_accessor[i] = complex.real;
                imag_accessor[i] = complex.imag;
            };
            noa::cuda::utils::iwise_1d(
                    "math::decompose", shape.elements(), kernel, stream);
        } else {
            const AccessorRestrict<const Complex<T>, 4, i64> input_accessor(input.get(), input_strides);
            const AccessorRestrict<T, 4, i64> real_accessor(real.get(), real_strides);
            const AccessorRestrict<T, 4, i64> imag_accessor(imag.get(), imag_strides);
            const auto kernel = [=]__device__(i64 i, i64 j, i64 k, i64 l) {
                const auto complex = input_accessor(i, j, k, l);
                real_accessor(i, j, k, l) = complex.real;
                imag_accessor(i, j, k, l) = complex.imag;
            };
            noa::cuda::utils::iwise_4d(
                    "math::decompose", shape, kernel, stream);
        }
        stream.attach(input, real, imag);
    }

    template<typename T, typename _>
    void complex(const Shared<T[]>& real, const Strides4<i64>& real_strides,
                 const Shared<T[]>& imag, const Strides4<i64>& imag_strides,
                 const Shared<Complex<T>[]>& output, const Strides4<i64>& output_strides,
                 const Shape4<i64>& shape, Stream& stream) {
        noa::cuda::utils::ewise_binary(
                "memory::complex",
                real.get(), real_strides,
                imag.get(), imag_strides,
                output.get(), output_strides,
                shape, stream, []__device__(T r, T i) { return Complex<T>{r, i}; });
        stream.attach(real, imag, output);
    }

    #define NOA_INSTANTIATE_COMPLEX_(T)                     \
    template void decompose<T, void>(                       \
        const Shared<Complex<T>[]>&, Strides4<i64>,         \
        const Shared<T[]>&, Strides4<i64>,                  \
        const Shared<T[]>&, Strides4<i64>,                  \
        Shape4<i64>, Stream&);                              \
    template void complex<T, void>(                         \
        const Shared<T[]>&, const Strides4<i64>&,           \
        const Shared<T[]>&, const Strides4<i64>&,           \
        const Shared<Complex<T>[]>&, const Strides4<i64>&,  \
        const Shape4<i64>&, Stream&)

    NOA_INSTANTIATE_COMPLEX_(f16);
    NOA_INSTANTIATE_COMPLEX_(f32);
    NOA_INSTANTIATE_COMPLEX_(f64);
}
