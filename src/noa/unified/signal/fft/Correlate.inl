#pragma once

#ifndef NOA_UNIFIED_FFT_CORRELATE
#error "This is a private header"
#endif

#include "noa/cpu/signal/fft/Correlate.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Correlate.h"
#endif

namespace noa::signal::fft {
    template<Remap REMAP, typename T, typename>
    void xmap(const Array<Complex<T>>& lhs, const Array<Complex<T>>& rhs,
              const Array<T>& output, size4_t shape,
              bool normalize, Norm norm, const Array<Complex<T>>& tmp) {
        const size4_t expected_shape = shape.fft();
        size4_t lhs_stride;
        if (!indexing::broadcast(lhs.shape(), lhs_stride, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), expected_shape);
        }
        size4_t rhs_stride;
        if (!indexing::broadcast(rhs.shape(), rhs_stride, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), expected_shape);
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The lhs, rhs and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
                  lhs.device(), rhs.device(), device);
        NOA_CHECK(all(output.shape() == shape),
                  "The output cross-correlation map is expected to have a shape of {}, but got {}",
                  shape, output.shape());

        if (tmp.empty()) {
            NOA_CHECK(all(rhs_stride >= 0),
                      "Since no temporary buffer is passed, the rhs input will be overwritten and should not have any "
                      "stride equal to 0, but got {}", rhs_stride);
        } else {
            NOA_CHECK(device == tmp.device(),
                      "The temporary and output arrays must be on the same device, tmp:{} and output:{}",
                      tmp.device(), device);
            NOA_CHECK(all(tmp.shape() >= expected_shape) && all(tmp.stride() >= 0),
                      "The temporary buffer should be able to fit an array of shape {}, but got effective shape of {}",
                      expected_shape, indexing::effectiveShape(tmp.shape(), tmp.stride()));
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::xmap<REMAP>(
                    lhs.share(), lhs_stride, rhs.share(), rhs_stride,
                    output.share(), output.stride(),
                    shape, normalize, norm, stream.cpu(), tmp, tmp.stride());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::xmap<REMAP>(
                    lhs.share(), lhs_stride, rhs.share(), rhs_stride,
                    output.share(), output.stride(),
                    shape, normalize, norm, stream.cuda(), tmp, tmp.stride());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void xpeak2D(const Array<T>& xmap, const Array<float2_t>& peaks) {
        NOA_CHECK(peaks.shape()[3] == xmap.shape()[0] &&
                  peaks.shape().ndim() == 1 && all(peaks.contiguous()),
                  "The number of peaks, specified as a contiguous row vector, should be equal to the number "
                  "of batches in the cross-correlation map, but got {} peaks and {} output batches",
                  peaks.shape()[3], xmap.shape()[0]);
        NOA_CHECK(xmap.stride()[2] > 0, "The cross-correlation map should not have its second-most stride set to 0");

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            NOA_CHECK(peaks.dereferencable(), "The peak coordinates should be accessible to the CPU");
            if (peaks.device().gpu())
                Stream::current(peaks.device()).synchronize();
            cpu::signal::fft::xpeak2D<REMAP>(xmap.share(), xmap.stride(), xmap.shape(), peaks.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if (peaks.device().cpu())
                Stream::current(Device{}).synchronize();
            cuda::signal::fft::xpeak2D<REMAP>(xmap.share(), xmap.stride(), xmap.shape(), peaks.share(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    float2_t xpeak2D(const Array<T>& xmap) {
        NOA_CHECK(xmap.stride()[2] > 0, "The cross-correlation map should not have its second-most stride set to 0");
        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            return cpu::signal::fft::xpeak2D<REMAP>(xmap.share(), xmap.stride(), xmap.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::signal::fft::xpeak2D<REMAP>(xmap.share(), xmap.stride(), xmap.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void xpeak3D(const Array<T>& xmap, const Array<float3_t>& peaks) {
        NOA_CHECK(peaks.shape()[3] == xmap.shape()[0] &&
                  peaks.shape().ndim() == 1 && peaks.contiguous(),
                  "The number of peaks, specified as a contiguous row vector, should be equal to the number "
                  "of batches in the cross-correlation map, but got {} peaks and {} output batches",
                  peaks.shape()[3], xmap.shape()[0]);
        NOA_CHECK(xmap.stride()[1] > 0 && xmap.stride()[2] > 0,
                  "The cross-correlation map should not have its second and third-most stride set to 0");

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            NOA_CHECK(peaks.dereferencable(), "The peak coordinates should be accessible to the CPU");
            if (peaks.device().gpu())
                Stream::current(peaks.device()).synchronize();
            cpu::signal::fft::xpeak3D<REMAP>(xmap.share(), xmap.stride(), xmap.shape(), peaks.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if (peaks.device().cpu())
                Stream::current(Device{}).synchronize();
            cuda::signal::fft::xpeak3D<REMAP>(xmap.share(), xmap.stride(), xmap.shape(), peaks.share(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    float3_t xpeak3D(const Array<T>& xmap) {
        NOA_CHECK(xmap.stride()[1] > 0 && xmap.stride()[2] > 0,
                  "The cross-correlation map should not have its second and third-most stride set to 0");
        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            return cpu::signal::fft::xpeak3D<REMAP>(xmap.share(), xmap.stride(), xmap.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::signal::fft::xpeak3D<REMAP>(xmap.share(), xmap.stride(), xmap.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void xcorr(const Array<Complex<T>>& lhs, const Array<Complex<T>>& rhs, size4_t shape,
               const Array<T>& coeffs, const Array<T>& tmp) {
        NOA_CHECK(coeffs.shape()[3] == shape[0] && coeffs.shape().ndim() == 1 && coeffs.contiguous(),
                  "The number of coeffs, specified as a contiguous row vector, should be equal to the number "
                  "of batches, but got {} coeffs and {} output batches", coeffs.shape()[3], shape[0]);

        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        const size4_t expected_shape = SRC_IS_HALF ? shape.fft() : shape;
        size4_t lhs_stride;
        if (!indexing::broadcast(lhs.shape(), lhs_stride, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), expected_shape);
        }
        size4_t rhs_stride;
        if (!indexing::broadcast(rhs.shape(), rhs_stride, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), expected_shape);
        }

        NOA_CHECK(lhs.device() == rhs.device(),
                  "The lhs and rhs input arrays should be on the same device, but got lhs:{} and rhs:{}",
                  lhs.device(), rhs.device());
        NOA_CHECK(coeffs.dereferencable(), "The coeffs should be accessible to the CPU");
        if (coeffs.device() != lhs.device())
            Stream::current(coeffs.device()).synchronize();

        Stream& stream = Stream::current(lhs.device());
        if (stream.device().cpu()) {
            cpu::signal::fft::xcorr<REMAP>(lhs.share(), lhs_stride, rhs.share(), rhs_stride,
                                           shape, coeffs.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            NOA_CHECK(tmp.empty() ||
                      (tmp.device() == stream.device() && all(tmp.shape() > expected_shape) && tmp.contiguous()),
                      "The temporary array should be contiguous and with a shape of at least {}, but got effective "
                      "shape {}", expected_shape, indexing::effectiveShape(tmp.shape(), tmp.stride()));
            cuda::signal::fft::xcorr<REMAP>(lhs.share(), lhs_stride, rhs.share(), rhs_stride,
                                            shape, coeffs.share(), stream.cpu(), tmp.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    T xcorr(const Array<Complex<T>>& lhs, const Array<Complex<T>>& rhs, size4_t shape, const Array<T>& tmp) {
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        const size4_t expected_shape = SRC_IS_HALF ? shape.fft() : shape;
        size4_t lhs_stride;
        if (!indexing::broadcast(lhs.shape(), lhs_stride, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), expected_shape);
        }
        size4_t rhs_stride;
        if (!indexing::broadcast(rhs.shape(), rhs_stride, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), expected_shape);
        }

        NOA_CHECK(lhs.device() == rhs.device(),
                  "The lhs and rhs input arrays should be on the same device, but got lhs:{} and rhs:{}",
                  lhs.device(), rhs.device());

        Stream& stream = Stream::current(lhs.device());
        if (stream.device().cpu()) {
            return cpu::signal::fft::xcorr<REMAP>(
                    lhs.share(), lhs_stride, rhs.share(), rhs_stride, shape, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            NOA_CHECK(tmp.empty() ||
                      (tmp.device() == stream.device() && all(tmp.shape() > expected_shape) && tmp.contiguous()),
                      "The temporary array should be contiguous and with a shape of at least {}, but got effective "
                      "shape {}", expected_shape, indexing::effectiveShape(tmp.shape(), tmp.stride()));
            return cuda::signal::fft::xcorr<REMAP>(
                    lhs.share(), lhs_stride, rhs.share(), rhs_stride, shape, stream.cpu(), tmp.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
