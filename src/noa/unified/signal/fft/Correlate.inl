#pragma once

#ifndef NOA_UNIFIED_FFT_CORRELATE
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/signal/fft/Correlate.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Correlate.h"
#endif

namespace noa::signal::fft {
    template<Remap REMAP, typename T, typename>
    void xmap(const Array<Complex<T>>& lhs, const Array<Complex<T>>& rhs, const Array<T>& output,
              bool normalize, Norm norm, const Array<Complex<T>>& tmp) {
        NOA_CHECK(!lhs.empty() && !rhs.empty() && !output.empty(), "Empty array detected");

        const dim4_t expected_shape = output.shape().fft();
        dim4_t lhs_strides = lhs.strides();
        if (!indexing::broadcast(lhs.shape(), lhs_strides, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), expected_shape);
        }
        dim4_t rhs_strides = rhs.strides();
        if (!indexing::broadcast(rhs.shape(), rhs_strides, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), expected_shape);
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The lhs, rhs and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
                  lhs.device(), rhs.device(), device);

        if (tmp.empty()) {
            NOA_CHECK(all(rhs_strides >= 0),
                      "Since no temporary buffer is passed, the rhs input will be overwritten and should not have any "
                      "strides equal to 0, but got {}", rhs_strides);
        } else {
            NOA_CHECK(device == tmp.device(),
                      "The temporary and output arrays must be on the same device, tmp:{} and output:{}",
                      tmp.device(), device);
            NOA_CHECK(all(tmp.shape() >= expected_shape) && all(tmp.strides() >= 0),
                      "The temporary buffer should be able to fit an array of shape {}, but got effective shape of {}",
                      expected_shape, indexing::effectiveShape(tmp.shape(), tmp.strides()));
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::xmap<REMAP>(
                    lhs.share(), lhs_strides, rhs.share(), rhs_strides,
                    output.share(), output.strides(), output.shape(),
                    normalize, norm, stream.cpu(), tmp.share(), tmp.strides());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::xmap<REMAP>(
                    lhs.share(), lhs_strides, rhs.share(), rhs_strides,
                    output.share(), output.strides(), output.shape(),
                    normalize, norm, stream.cuda(), tmp.share(), tmp.strides());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void xpeak1D(const Array<T>& xmap, const Array<float>& peaks, float ellipse_radius, int64_t registration_radius) {
        NOA_CHECK(!xmap.empty() && !peaks.empty(), "Empty array detected");
        NOA_CHECK(peaks.elements() == xmap.shape()[0] &&
                  indexing::isVector(peaks.shape()) && peaks.contiguous(),
                  "The number of peaks, specified as a contiguous vector, should be equal to the number "
                  "of batches in the cross-correlation map. Got {} peaks and {} output batches",
                  peaks.elements(), xmap.shape()[0]);

        [[maybe_unused]] const bool is_column = xmap.shape()[3] == 1;
        NOA_CHECK(xmap.strides()[3 - is_column] > 0,
                  "The 1D cross-correlation map should not have its stride set to 0");
        NOA_CHECK(indexing::isVector(xmap.shape(), true) && xmap.shape()[1] == 1 && xmap.shape()[2 + is_column] == 1,
                  "The 1D cross-correlation map(s) should be a (batch of) row or column vector(s), but got shape {}",
                  xmap.shape());
        NOA_CHECK(registration_radius > 0 && registration_radius <= 256,
                  "The registration radius should be a small positive value (less than 256), but got {}",
                  registration_radius);

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            NOA_CHECK(peaks.dereferenceable(), "The peak coordinates should be accessible to the CPU");
            if (peaks.device().gpu())
                Stream::current(peaks.device()).synchronize();
            cpu::signal::fft::xpeak1D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), peaks.share(),
                    ellipse_radius, registration_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if (peaks.device().cpu())
                Stream::current(Device{}).synchronize();
            cuda::signal::fft::xpeak1D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), peaks.share(),
                    ellipse_radius, registration_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    float xpeak1D(const Array<T>& xmap, float ellipse_radius, int64_t registration_radius) {
        NOA_CHECK(!xmap.empty(), "Empty array detected");
        [[maybe_unused]] const bool is_column = xmap.shape()[3] == 1;
        NOA_CHECK(xmap.strides()[3 - is_column] > 0,
                  "The 1D cross-correlation map should not have its stride set to 0");
        NOA_CHECK(xmap.shape().ndim() == 1,
                  "The 1D cross-correlation map should be a row or column vector, but got shape {}", xmap.shape());
        NOA_CHECK(registration_radius > 0 && registration_radius <= 256,
                  "The registration radius should be a small positive value (less than 256), but got {}",
                  registration_radius);

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            return cpu::signal::fft::xpeak1D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(),
                    ellipse_radius, registration_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::signal::fft::xpeak1D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(),
                    ellipse_radius, registration_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void xpeak2D(const Array<T>& xmap, const Array<float2_t>& peaks,
                 float2_t ellipse_radius, long2_t registration_radius) {
        NOA_CHECK(!xmap.empty() && !peaks.empty(), "Empty array detected");
        NOA_CHECK(peaks.elements() == xmap.shape()[0] &&
                  indexing::isVector(peaks.shape()) && peaks.contiguous(),
                  "The number of peaks, specified as a contiguous vector, should be equal to the number "
                  "of batches in the cross-correlation map. Got {} peaks and {} output batches",
                  peaks.elements(), xmap.shape()[0]);

        NOA_CHECK(all(xmap.strides() > 0), "The cross-correlation map should not be broadcast");
        NOA_CHECK(xmap.shape()[1] == 1,
                  "The cross-correlation map(s) should be a (batch of) 2D array(s), but got shape {}",
                  xmap.shape());
        NOA_CHECK(all(registration_radius > 0 && registration_radius <= 256),
                  "The registration radius should be a small positive value (less than 256), but got {}",
                  registration_radius);

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            NOA_CHECK(peaks.dereferenceable(), "The peak coordinates should be accessible to the CPU");
            if (peaks.device().gpu())
                Stream::current(peaks.device()).synchronize();
            cpu::signal::fft::xpeak2D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(),
                    peaks.share(), ellipse_radius, registration_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if (peaks.device().cpu())
                Stream::current(Device{}).synchronize();
            cuda::signal::fft::xpeak2D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(),
                    peaks.share(), ellipse_radius, registration_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    float2_t xpeak2D(const Array<T>& xmap, float2_t ellipse_radius, long2_t registration_radius) {
        NOA_CHECK(!xmap.empty(), "Empty array detected");
        NOA_CHECK(all(xmap.strides() > 0), "The cross-correlation map should not be broadcast");
        NOA_CHECK(xmap.shape().ndim() == 2,
                  "The cross-correlation map should be a single 2D array, but got shape {}", xmap.shape());
        NOA_CHECK(all(registration_radius > 0 && registration_radius <= 256),
                  "The registration radius should be a small positive value (less than 256), but got {}",
                  registration_radius);

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            return cpu::signal::fft::xpeak2D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(),
                    ellipse_radius, registration_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::signal::fft::xpeak2D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(),
                    ellipse_radius, registration_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void xpeak3D(const Array<T>& xmap, const Array<float3_t>& peaks,
                 float3_t ellipse_radius, long3_t registration_radius) {
        NOA_CHECK(!xmap.empty() && !peaks.empty(), "Empty array detected");
        NOA_CHECK(peaks.elements() == xmap.shape()[0] &&
                  indexing::isVector(peaks.shape()) && peaks.contiguous(),
                  "The number of peaks, specified as a contiguous vector, should be equal to the number "
                  "of batches in the cross-correlation map. Got {} peaks and {} output batches",
                  peaks.elements(), xmap.shape()[0]);

        NOA_CHECK(all(xmap.strides() > 0), "The cross-correlation map should not be broadcast");
        NOA_CHECK(all(registration_radius > 0 && registration_radius <= 256),
                  "The registration radius should be a small positive value (less than 256), but got {}",
                  registration_radius);

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            NOA_CHECK(peaks.dereferenceable(), "The peak coordinates should be accessible to the CPU");
            if (peaks.device().gpu())
                Stream::current(peaks.device()).synchronize();
            cpu::signal::fft::xpeak3D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), peaks.share(),
                    ellipse_radius, registration_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if (peaks.device().cpu())
                Stream::current(Device{}).synchronize();
            cuda::signal::fft::xpeak3D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), peaks.share(),
                    ellipse_radius, registration_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    float3_t xpeak3D(const Array<T>& xmap, float3_t ellipse_radius, long3_t registration_radius) {
        NOA_CHECK(!xmap.empty(), "Empty array detected");
        NOA_CHECK(all(xmap.strides() > 0), "The cross-correlation map should not be broadcast");
        NOA_CHECK(xmap.shape().ndim() == 3,
                  "The cross-correlation map should be a single 3D array, but got shape {}", xmap.shape());
        NOA_CHECK(all(registration_radius > 0 && registration_radius <= 256),
                  "The registration radius should be a small positive value (less than 256), but got {}",
                  registration_radius);

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            return cpu::signal::fft::xpeak3D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(),
                    ellipse_radius, registration_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::signal::fft::xpeak3D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(),
                    ellipse_radius, registration_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void xcorr(const Array<Complex<T>>& lhs,
               const Array<Complex<T>>& rhs, dim4_t shape,
               const Array<T>& coeffs) {
        NOA_CHECK(!lhs.empty() && !rhs.empty() && !coeffs.empty(), "Empty array detected");
        NOA_CHECK(coeffs.elements() == shape[0] && indexing::isVector(coeffs.shape()) && coeffs.contiguous(),
                  "The number of coeffs, specified as a contiguous vector, should be equal to the number "
                  "of batches. Got {} coeffs and {} output batches", coeffs.elements(), shape[0]);

        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        const dim4_t expected_shape = SRC_IS_HALF ? shape.fft() : shape;
        dim4_t lhs_strides = lhs.strides();
        if (!indexing::broadcast(lhs.shape(), lhs_strides, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), expected_shape);
        }
        dim4_t rhs_strides = rhs.strides();
        if (!indexing::broadcast(rhs.shape(), rhs_strides, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), expected_shape);
        }

        NOA_CHECK(lhs.device() == rhs.device(),
                  "The lhs and rhs input arrays should be on the same device, but got lhs:{} and rhs:{}",
                  lhs.device(), rhs.device());
        NOA_CHECK(coeffs.dereferenceable(), "The coeffs should be accessible to the CPU");
        if (coeffs.device() != lhs.device())
            Stream::current(coeffs.device()).synchronize();

        Stream& stream = Stream::current(lhs.device());
        if (stream.device().cpu()) {
            cpu::signal::fft::xcorr<REMAP>(lhs.share(), lhs_strides, rhs.share(), rhs_strides,
                                           shape, coeffs.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::xcorr<REMAP>(lhs.share(), lhs_strides, rhs.share(), rhs_strides,
                                            shape, coeffs.share(), stream.cpu());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    T xcorr(const Array<Complex<T>>& lhs, const Array<Complex<T>>& rhs, dim4_t shape) {
        NOA_CHECK(!lhs.empty() && !rhs.empty(), "Empty array detected");
        NOA_CHECK(shape.ndim() <= 3, "The shape should have a batch of 1, got {}", shape[0]);

        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        const dim4_t expected_shape = SRC_IS_HALF ? shape.fft() : shape;
        dim4_t lhs_strides = lhs.strides();
        if (!indexing::broadcast(lhs.shape(), lhs_strides, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), expected_shape);
        }
        dim4_t rhs_strides = rhs.strides();
        if (!indexing::broadcast(rhs.shape(), rhs_strides, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), expected_shape);
        }

        NOA_CHECK(lhs.device() == rhs.device(),
                  "The lhs and rhs input arrays should be on the same device, but got lhs:{} and rhs:{}",
                  lhs.device(), rhs.device());

        Stream& stream = Stream::current(lhs.device());
        if (stream.device().cpu()) {
            return cpu::signal::fft::xcorr<REMAP>(
                    lhs.share(), lhs_strides, rhs.share(), rhs_strides, shape, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::signal::fft::xcorr<REMAP>(
                    lhs.share(), lhs_strides, rhs.share(), rhs_strides, shape, stream.cpu());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
