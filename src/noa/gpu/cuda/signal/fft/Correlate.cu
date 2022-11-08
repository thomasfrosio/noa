#include "noa/gpu/cuda/fft/Transforms.h"
#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"
#include "noa/gpu/cuda/signal/fft/Correlate.h"
#include "noa/gpu/cuda/signal/fft/Shift.h"

#include "noa/gpu/cuda/utils/EwiseBinary.cuh"
#include "noa/gpu/cuda/utils/ReduceUnary.cuh"
#include "noa/gpu/cuda/utils/ReduceBinary.cuh"

namespace noa::cuda::signal::fft {
    template<Remap REMAP, typename T, typename U>
    void xmap(const shared_t<Complex<T>[]>& lhs, dim4_t lhs_strides,
              const shared_t<Complex<T>[]>& rhs, dim4_t rhs_strides,
              const shared_t<T[]>& output, dim4_t output_strides,
              dim4_t shape, bool normalize, Norm norm, Stream& stream,
              const shared_t<Complex<T>[]>& tmp, dim4_t tmp_strides) {

        const shared_t<Complex<T>[]>& buffer = tmp ? tmp : rhs;
        const dim4_t& buffer_strides = tmp ? tmp_strides : rhs_strides;
        NOA_ASSERT(all(buffer_strides > 0));

        if (normalize) {
            cuda::utils::ewise::binary(
                    "signal::fft::xmap",
                    lhs.get(), lhs_strides, rhs.get(), rhs_strides,
                    buffer.get(), buffer_strides,
                    shape.fft(), true, stream,
                    []__device__(Complex<T> l, Complex<T> r) {
                        const Complex<T> product = l * noa::math::conj(r);
                        const T magnitude = noa::math::abs(product);
                        return product / (magnitude + static_cast<T>(1e-13));
                        // The epsilon could be scaled by the max(abs(rhs)), but this seems to be useful only
                        // for input values close to zero (less than 1e-10). In most cases, this is fine.
                        // Note that the normalization can sharpen the peak considerably.
                    });
        } else {
            cuda::math::ewise(lhs, lhs_strides, rhs, rhs_strides, buffer, buffer_strides,
                              shape.fft(), noa::math::multiply_conj_t{}, stream);
        }

        if constexpr (REMAP == Remap::H2FC) {
            const dim3_t shape_3d(shape.get(1));
            if (shape_3d.ndim() == 3) {
                cuda::signal::fft::shift3D<Remap::H2H>(buffer, buffer_strides, buffer, buffer_strides, shape,
                                                       float3_t(shape_3d / 2), 1, stream);
            } else {
                cuda::signal::fft::shift2D<Remap::H2H>(buffer, buffer_strides, buffer, buffer_strides, shape,
                                                       float2_t{shape_3d[1] / 2, shape_3d[2] / 2}, 1, stream);
            }
        }

        cuda::fft::c2r(buffer, buffer_strides, output, output_strides, shape, norm, stream);
    }


    #define INSTANTIATE_XMAP(R, T)              \
    template void xmap<R, T, void>(             \
        const shared_t<Complex<T>[]>&, dim4_t,  \
        const shared_t<Complex<T>[]>&, dim4_t,  \
        const shared_t<T[]>&, dim4_t, dim4_t,   \
        bool, Norm, Stream&, const shared_t<Complex<T>[]>&, dim4_t)

    #define INSTANTIATE_XMAP_ALL(T)     \
    INSTANTIATE_XMAP(Remap::H2F, T);    \
    INSTANTIATE_XMAP(Remap::H2FC, T)

    INSTANTIATE_XMAP_ALL(float);
    INSTANTIATE_XMAP_ALL(double);
}

namespace noa::cuda::signal::fft::details {
    template<typename T>
    void xcorr(const shared_t<Complex<T>[]>& lhs, dim4_t lhs_stride,
               const shared_t<Complex<T>[]>& rhs, dim4_t rhs_stride,
               dim4_t shape, const shared_t<T[]>& coefficients,
               Stream& stream, bool is_half) {
        const dim_t batches = shape[0];
        const dim4_t shape_fft = is_half ? shape.fft() : shape;

        cuda::memory::PtrPinned<T> buffer(batches * 3);
        auto denominator_lhs = buffer.get() + batches;
        auto denominator_rhs = buffer.get() + batches * 2;

        T* null{};
        cuda::utils::reduce(
                "signal::fft::xcorr", lhs.get(), lhs_stride, shape_fft,
                noa::math::abs_squared_t{}, noa::math::plus_t{}, T{0},
                denominator_lhs, 1, noa::math::copy_t{},
                null, 0, noa::math::copy_t{},
                false, true, stream);
        cuda::utils::reduce(
                "signal::fft::xcorr", rhs.get(), rhs_stride, shape_fft,
                noa::math::abs_squared_t{}, noa::math::plus_t{}, T{0},
                denominator_rhs, 1, noa::math::copy_t{},
                null, 0, noa::math::copy_t{},
                false, true, stream);

        auto combine_op = []__device__(Complex<T> l, Complex<T> r) { return noa::math::real(l * r); };
        cuda::utils::reduce<false>(
                "signal::fft::xcorr",
                lhs.get(), lhs_stride, rhs.get(), rhs_stride, shape_fft,
                noa::math::copy_t{}, noa::math::conj_t{}, combine_op, noa::math::plus_t{}, T{0},
                buffer.get(), 1, noa::math::copy_t{}, null, 1, noa::math::copy_t{}, false, stream);

        stream.synchronize(); // FIXME Add callback
        for (dim_t batch = 0; batch < batches; ++batch) {
            coefficients.get()[batch] =
                    buffer[batch] / noa::math::sqrt(denominator_lhs[batch] * denominator_rhs[batch]);
        }
    }

    template<typename T>
    T xcorr(const shared_t<Complex<T>[]>& lhs, dim4_t lhs_stride,
            const shared_t<Complex<T>[]>& rhs, dim4_t rhs_stride,
            dim4_t shape, Stream& stream, bool is_half) {
        NOA_ASSERT(shape[0] == 1);
        const dim4_t shape_fft = is_half ? shape.fft() : shape;

        T numerator{}, denominator_lhs{}, denominator_rhs{};
        T* null{};
        cuda::utils::reduce(
                "signal::fft::xcorr", lhs.get(), lhs_stride, shape_fft,
                noa::math::abs_squared_t{}, noa::math::plus_t{}, T{0},
                &denominator_lhs, 1, noa::math::copy_t{},
                null, 0, noa::math::copy_t{},
                true, true, stream);
        cuda::utils::reduce(
                "signal::fft::xcorr", rhs.get(), rhs_stride, shape_fft,
                noa::math::abs_squared_t{}, noa::math::plus_t{}, T{0},
                &denominator_rhs, 1, noa::math::copy_t{},
                null, 0, noa::math::copy_t{},
                true, true, stream);

        auto combine_op = []__device__(Complex<T> l, Complex<T> r) { return noa::math::real(l * r); };
        cuda::utils::reduce<false>(
                "signal::fft::xcorr",
                lhs.get(), lhs_stride, rhs.get(), rhs_stride, shape_fft,
                noa::math::copy_t{}, noa::math::conj_t{}, combine_op, noa::math::plus_t{}, T{0},
                &numerator, 1, noa::math::copy_t{}, null, 1, noa::math::copy_t{}, false, stream);

        stream.synchronize();
        const T denominator = noa::math::sqrt(denominator_lhs * denominator_rhs);
        return numerator / denominator;
    }

    #define INSTANTIATE_XCORR(T) \
    template void xcorr<T>(const shared_t<Complex<T>[]>&, dim4_t, const shared_t<Complex<T>[]>&, dim4_t, dim4_t, const shared_t<T[]>&, Stream&, bool); \
    template T xcorr<T>(const shared_t<Complex<T>[]>&, dim4_t, const shared_t<Complex<T>[]>&, dim4_t, dim4_t, Stream&, bool)

    INSTANTIATE_XCORR(float);
    INSTANTIATE_XCORR(double);
}
