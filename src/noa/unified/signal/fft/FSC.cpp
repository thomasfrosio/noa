#include "noa/unified/signal/fft/FSC.h"
#include "noa/unified/memory/Factory.h"

#include "noa/cpu/signal/fft/FSC.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/FSC.h"
#endif

namespace noa::signal::fft {
    template<noa::fft::Remap REMAP, typename Real, typename>
    void isotropicFSC(const Array<Complex<Real>>& lhs,
                      const Array<Complex<Real>>& rhs,
                      const Array<Real>& fsc,
                      dim4_t shape) {
        NOA_CHECK(!lhs.empty() && !rhs.empty() && !fsc.empty(), "Empty array detected");
        NOA_CHECK(lhs.get() != rhs.get(), "Computing the FSC on the same array is not allowed");

        NOA_CHECK(all(rhs.shape() == shape.fft()),
                  "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
                  shape, shape.fft(), rhs.shape());
        NOA_CHECK(all(lhs.shape() == rhs.shape()),
                  "The two input arrays should have the same shape. Got lhs:{} and rhs:{}",
                  lhs.shape(), rhs.shape());

        const Device device = fsc.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{}, fsc:{}",
                  lhs.device(), rhs.device(), device);

        const auto shell_count = noa::math::min(lhs.shape()) / 2 + 1;
        const auto expected_shape = dim4_t{shape[0], 1, 1, shell_count};
        NOA_CHECK(all(fsc.shape() == expected_shape) && fsc.contiguous(),
                  "The FSC, specified as a contiguous (batched) row vector, does not have the correct shape. "
                  "Given an input shape {}, the expected shape is {}, but got {}",
                  shape, expected_shape, fsc.shape());

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::isotropicFSC<REMAP>(
                    lhs.share(), lhs.strides(),
                    rhs.share(), rhs.strides(),
                    fsc.share(), shape,
                    stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::isotropicFSC<REMAP>(
                    lhs.share(), lhs.strides(),
                    rhs.share(), rhs.strides(),
                    fsc.share(), shape,
                    stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<noa::fft::Remap REMAP, typename Real, typename>
    Array<Real> isotropicFSC(const Array<Complex<Real>>& lhs,
                             const Array<Complex<Real>>& rhs,
                             dim4_t shape) {
        const auto shell_count = noa::math::min(lhs.shape()) / 2 + 1;
        const auto expected_shape = dim4_t{shape[0], 1, 1, shell_count};
        auto fsc = noa::memory::empty<Real>(expected_shape, rhs.options());
        isotropicFSC<REMAP>(lhs, rhs, fsc, shape);
        return fsc;
    }

    template<noa::fft::Remap REMAP, typename Real, typename>
    void anisotropicFSC(const Array<Complex<Real>>& lhs,
                        const Array<Complex<Real>>& rhs,
                        const Array<Real>& fsc, dim4_t shape,
                        const Array<float3_t>& cone_directions, float aperture) {
        NOA_CHECK(!lhs.empty() && !rhs.empty() && !fsc.empty() && !cone_directions.empty(), "Empty array detected");
        NOA_CHECK(lhs.get() != rhs.get(), "Computing the FSC on the same array is not allowed");

        NOA_CHECK(all(rhs.shape() == shape.fft()),
                  "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
                  shape, shape.fft(), rhs.shape());
        NOA_CHECK(all(lhs.shape() == rhs.shape()),
                  "The two input arrays should have the same shape. Got lhs:{} and rhs:{}",
                  lhs.shape(), rhs.shape());

        const Device device = fsc.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device() && device == cone_directions.device(),
                  "The input and output arrays must be on the same device, "
                  "but got lhs:{}, rhs:{}, fsc:{}, cones:{}",
                  lhs.device(), rhs.device(), device, cone_directions.device());

        NOA_CHECK(noa::indexing::isVector(cone_directions.shape()) && cone_directions.contiguous(),
                  "The cone directions should be specified as a contiguous vector, but got shape:{}, strides:{}",
                  cone_directions.shape(), cone_directions.strides());
        const auto shell_count = noa::math::min(lhs.shape()) / 2 + 1;
        const auto expected_shape = dim4_t{shape[0], 1, cone_directions.size(), shell_count};
        NOA_CHECK(all(fsc.shape() == expected_shape) && fsc.contiguous(),
                  "The FSC, specified as a contiguous (batched) row vector, does not have the correct shape. "
                  "Given an input shape {}, the expected shape is {}, but got {}",
                  shape, expected_shape, fsc.shape());

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::anisotropicFSC<REMAP>(
                    lhs.share(), lhs.strides(),
                    rhs.share(), rhs.strides(),
                    fsc.share(), shape,
                    cone_directions.share(), expected_shape[2], aperture,
                    stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::anisotropicFSC<REMAP>(
                    lhs.share(), lhs.strides(),
                    rhs.share(), rhs.strides(),
                    fsc.share(), shape,
                    cone_directions.share(), expected_shape[2], aperture,
                    stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<noa::fft::Remap REMAP, typename Real, typename>
    Array<Real> anisotropicFSC(const Array<Complex<Real>>& lhs,
                               const Array<Complex<Real>>& rhs, dim4_t shape,
                               const Array<float3_t>& cone_directions, float cone_aperture) {
        const auto shell_count = noa::math::min(lhs.shape()) / 2 + 1;
        const auto expected_shape = dim4_t{shape[0], 1, cone_directions.size(), shell_count};
        auto fsc = noa::memory::empty<Real>(expected_shape, rhs.options());
        anisotropicFSC<REMAP>(lhs, rhs, fsc, shape, cone_directions, cone_aperture);
        return fsc;
    }

    #define NOA_INSTANTIATE_FSC(R, T)               \
    template void isotropicFSC<R, T, void>(         \
        const Array<Complex<T>>&,                   \
        const Array<Complex<T>>&,                   \
        const Array<T>&, dim4_t);                   \
    template Array<T> isotropicFSC<R, T, void>(     \
        const Array<Complex<T>>&,                   \
        const Array<Complex<T>>&, dim4_t);          \
    template void anisotropicFSC<R, T, void>(       \
        const Array<Complex<T>>&,                   \
        const Array<Complex<T>>&,                   \
        const Array<T>&, dim4_t,                    \
        const Array<float3_t>&, float);             \
    template Array<T> anisotropicFSC<R, T, void>(   \
        const Array<Complex<T>>&,                   \
        const Array<Complex<T>>&,  dim4_t,          \
        const Array<float3_t>&, float)

    #define NOA_INSTANTIATE_FSC_ALL(T)      \
    NOA_INSTANTIATE_FSC(noa::fft::H2H, T);  \
    NOA_INSTANTIATE_FSC(noa::fft::HC2HC, T)

    NOA_INSTANTIATE_FSC_ALL(float);
    NOA_INSTANTIATE_FSC_ALL(double);
}
