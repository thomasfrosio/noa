#include "noa/unified/signal/fft/Correlate.h"

#include "noa/cpu/signal/fft/Correlate.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Correlate.h"
#endif

namespace noa::signal::fft {
    template<Remap REMAP, typename Real, typename>
    void xmap(const Array<Complex<Real>>& lhs,
              const Array<Complex<Real>>& rhs,
              const Array<Real>& output,
              CorrelationMode correlation_mode, Norm norm,
              const Array<Complex<Real>>& buffer) {
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

        if (buffer.empty()) {
            NOA_CHECK(all(rhs_strides >= 0),
                      "Since no temporary buffer is passed, the rhs input will be overwritten and "
                      "should not have any strides equal to 0, but got {}", rhs_strides);
        } else {
            NOA_CHECK(device == buffer.device(),
                      "The temporary and output arrays must be on the same device, buffer:{} and output:{}",
                      buffer.device(), device);
            NOA_CHECK(all(buffer.shape() >= expected_shape) && all(buffer.strides() >= 0),
                      "The temporary buffer should be able to fit an array of shape {}, but got effective shape of {}",
                      expected_shape, indexing::effectiveShape(buffer.shape(), buffer.strides()));
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::xmap<REMAP>(
                    lhs.share(), lhs_strides, rhs.share(), rhs_strides,
                    output.share(), output.strides(), output.shape(),
                    correlation_mode, norm, stream.cpu(),
                    buffer.share(), buffer.strides());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::xmap<REMAP>(
                    lhs.share(), lhs_strides, rhs.share(), rhs_strides,
                    output.share(), output.strides(), output.shape(),
                    correlation_mode, norm, stream.cuda(),
                    buffer.share(), buffer.strides());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    #define NOA_INSTANTIATE_XMAP_(R, T)                                         \
    template void xmap<R, T, void>(                                             \
        const Array<Complex<T>>&, const Array<Complex<T>>&, const Array<T>&,    \
        CorrelationMode, Norm norm, const Array<Complex<T>>&);

    #define NOA_INSTANTIATE_XMAP_ALL(T)     \
    NOA_INSTANTIATE_XMAP_(Remap::H2F, T);   \
    NOA_INSTANTIATE_XMAP_(Remap::H2FC, T)

    NOA_INSTANTIATE_XMAP_ALL(float);
    NOA_INSTANTIATE_XMAP_ALL(double);
}

namespace {
    using namespace ::noa;

    template<int32_t NDIM, typename Real, typename Coord>
    void checkPeakND(const Array<Real>& xmap,
                     const Array<Coord>& peak_coordinates,
                     const Array<Real>& peak_values) {
        const char* FUNC_NAME = NDIM == 1 ? "xpeak1D" : NDIM == 2 ? "xpeak2D" : "xpeak3D";
        NOA_CHECK_FUNC(FUNC_NAME, !xmap.empty(), "Empty array detected");
        NOA_CHECK_FUNC(FUNC_NAME, all(xmap.strides() > 0), "The cross-correlation map should not be broadcast");

        if (!peak_coordinates.empty()) {
            NOA_CHECK_FUNC(FUNC_NAME,
                           peak_coordinates.elements() == xmap.shape()[0] &&
                           indexing::isVector(peak_coordinates.shape()) && peak_coordinates.contiguous(),
                           "The number of peak coordinates, specified as a contiguous vector, should be equal to "
                           "the number of batches in the cross-correlation map. Got {} peak coordinates and {} output "
                           "batches", peak_coordinates.elements(), xmap.shape()[0]);
            NOA_CHECK_FUNC(FUNC_NAME, xmap.device() == peak_coordinates.device(),
                           "The cross-correlation map and output peak coordinates must be on the same device, "
                           "but got xmap:{} and peak_coordinates:{}", xmap.device(), peak_coordinates.device());
        }

        if (!peak_values.empty()) {
            NOA_CHECK_FUNC(FUNC_NAME,
                           peak_values.elements() == xmap.shape()[0] &&
                           indexing::isVector(peak_values.shape()) && peak_values.contiguous(),
                           "The number of peak values, specified as a contiguous vector, should be equal to "
                           "the number of batches in the cross-correlation map. Got {} peak values and {} output "
                           "batches", peak_values.elements(), xmap.shape()[0]);
            NOA_CHECK_FUNC(FUNC_NAME, xmap.device() == peak_values.device(),
                           "The cross-correlation map and output peak values must be on the same device, "
                           "but got xmap:{} and peak_values:{}", xmap.device(), peak_values.device());
        }
    }
}

namespace noa::signal::fft {
    template<Remap REMAP, typename Real, typename>
    void xpeak1D(const Array<Real>& xmap,
                 const Array<float>& peak_coordinates,
                 const Array<Real>& peak_values,
                 float xmap_radius, PeakMode peak_mode, int64_t peak_radius) {
        checkPeakND<1>(xmap, peak_coordinates, peak_values);

        [[maybe_unused]] const bool is_column = xmap.shape()[3] == 1;
        NOA_CHECK(xmap.strides()[3 - is_column] > 0,
                  "The 1D cross-correlation map should not have its stride set to 0");
        NOA_CHECK(indexing::isVector(xmap.shape(), true) && xmap.shape()[1] == 1 && xmap.shape()[2 + is_column] == 1,
                  "The 1D cross-correlation map(s) should be a (batch of) row or column vector(s), but got shape {}",
                  xmap.shape());
        NOA_CHECK(peak_radius > 0 && peak_radius <= 64,
                  "The registration radius should be a small positive value (less than 64), but got {}",
                  peak_radius);

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            cpu::signal::fft::xpeak1D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), xmap_radius,
                    peak_coordinates.share(), peak_values.share(),
                    peak_mode, peak_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::xpeak1D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), xmap_radius,
                    peak_coordinates.share(), peak_values.share(),
                    peak_mode, peak_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Real, typename>
    std::pair<float, Real> xpeak1D(const Array<Real>& xmap, float xmap_radius,
                                   PeakMode peak_mode, int64_t peak_radius) {
        NOA_CHECK(!xmap.empty(), "Empty array detected");
        [[maybe_unused]] const bool is_column = xmap.shape()[3] == 1;
        NOA_CHECK(xmap.strides()[3 - is_column] > 0,
                  "The 1D cross-correlation map should not have its stride set to 0");
        NOA_CHECK(xmap.shape().ndim() == 1,
                  "The 1D cross-correlation map should be a row or column vector, but got shape {}", xmap.shape());
        NOA_CHECK(peak_radius > 0 && peak_radius <= 64,
                  "The registration radius should be a small positive value (less than 64), but got {}",
                  peak_radius);

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            return cpu::signal::fft::xpeak1D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), xmap_radius,
                    peak_mode, peak_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::signal::fft::xpeak1D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), xmap_radius,
                    peak_mode, peak_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Real, typename>
    void xpeak2D(const Array<Real>& xmap,
                 const Array<float2_t>& peak_coordinates,
                 const Array<Real>& peak_values,
                 float2_t xmap_radius, PeakMode peak_mode, long2_t peak_radius) {
        checkPeakND<2>(xmap, peak_coordinates, peak_values);

        NOA_CHECK(xmap.shape()[1] == 1,
                  "The cross-correlation map(s) should be a (batch of) 2D array(s), but got shape {}",
                  xmap.shape());
        const int64_t peak_radius_limit = xmap.device().gpu() && peak_mode == PEAK_COM ? 8 : 64;
        NOA_CHECK(all(peak_radius > 0 && peak_radius <= peak_radius_limit),
                  "The registration radius should be a small positive value (less than {}), but got {}",
                  peak_radius_limit, peak_radius);

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            cpu::signal::fft::xpeak2D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), xmap_radius,
                    peak_coordinates.share(), peak_values.share(),
                    peak_mode, peak_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::xpeak2D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), xmap_radius,
                    peak_coordinates.share(), peak_values.share(),
                    peak_mode, peak_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Real, typename>
    std::pair<float2_t, Real> xpeak2D(const Array<Real>& xmap, float2_t xmap_radius,
                                      PeakMode peak_mode, long2_t peak_radius) {
        NOA_CHECK(!xmap.empty(), "Empty array detected");
        NOA_CHECK(all(xmap.strides() > 0), "The cross-correlation map should not be broadcast");
        NOA_CHECK(xmap.shape().ndim() == 2,
                  "The cross-correlation map should be a single 2D array, but got shape {}", xmap.shape());
        const int64_t peak_radius_limit = xmap.device().gpu() && peak_mode == PEAK_COM ? 8 : 64;
        NOA_CHECK(all(peak_radius > 0 && peak_radius <= peak_radius_limit),
                  "The registration radius should be a small positive value (less than {}), but got {}",
                  peak_radius_limit, peak_radius);

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            return cpu::signal::fft::xpeak2D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), xmap_radius,
                    peak_mode, peak_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::signal::fft::xpeak2D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), xmap_radius,
                    peak_mode, peak_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Real, typename>
    void xpeak3D(const Array<Real>& xmap,
                 const Array<float3_t>& peak_coordinates,
                 const Array<Real>& peak_values,
                 float3_t xmap_radius, PeakMode peak_mode, long3_t peak_radius) {
        checkPeakND<3>(xmap, peak_coordinates, peak_values);

        const int64_t peak_radius_limit = xmap.device().gpu() && peak_mode == PEAK_COM ? 2 : 64;
        NOA_CHECK(all(peak_radius > 0 && peak_radius <= peak_radius_limit),
                  "The registration radius should be a small positive value (less than {}), but got {}",
                  peak_radius_limit, peak_radius);

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            cpu::signal::fft::xpeak3D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), xmap_radius,
                    peak_coordinates.share(), peak_values.share(),
                    peak_mode, peak_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::xpeak3D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), xmap_radius,
                    peak_coordinates.share(), peak_values.share(),
                    peak_mode, peak_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Real, typename>
    std::pair<float3_t, Real> xpeak3D(const Array<Real>& xmap, float3_t xmap_radius,
                                      PeakMode peak_mode, long3_t peak_radius) {
        NOA_CHECK(!xmap.empty(), "Empty array detected");
        NOA_CHECK(all(xmap.strides() > 0), "The cross-correlation map should not be broadcast");
        NOA_CHECK(xmap.shape().ndim() == 3,
                  "The cross-correlation map should be a single 3D array, but got shape {}", xmap.shape());
        const int64_t peak_radius_limit = xmap.device().gpu() && peak_mode == PEAK_COM ? 2 : 64;
        NOA_CHECK(all(peak_radius > 0 && peak_radius <= peak_radius_limit),
                  "The registration radius should be a small positive value (less than {}), but got {}",
                  peak_radius_limit, peak_radius);

        Stream& stream = Stream::current(xmap.device());
        if (stream.device().cpu()) {
            return cpu::signal::fft::xpeak3D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), xmap_radius,
                    peak_mode, peak_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::signal::fft::xpeak3D<REMAP>(
                    xmap.share(), xmap.strides(), xmap.shape(), xmap_radius,
                    peak_mode, peak_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    #define NOA_INSTANTIATE_XPEAK_(R, T)\
    template void xpeak1D<R, T, void>(const Array<T>&, const Array<float>&,    const Array<T>&, float,    PeakMode, int64_t );   \
    template void xpeak2D<R, T, void>(const Array<T>&, const Array<float2_t>&, const Array<T>&, float2_t, PeakMode, long2_t);    \
    template void xpeak3D<R, T, void>(const Array<T>&, const Array<float3_t>&, const Array<T>&, float3_t, PeakMode, long3_t );   \
    template std::pair<float, T>    xpeak1D<R, T, void>(const Array<T>&, float,    PeakMode, int64_t);    \
    template std::pair<float2_t, T> xpeak2D<R, T, void>(const Array<T>&, float2_t, PeakMode, long2_t );   \
    template std::pair<float3_t, T> xpeak3D<R, T, void>(const Array<T>&, float3_t, PeakMode, long3_t )

    #define NOA_INSTANTIATE_XPEAK_ALL(T)    \
    NOA_INSTANTIATE_XPEAK_(Remap::F2F, T);   \
    NOA_INSTANTIATE_XPEAK_(Remap::FC2FC, T)

    NOA_INSTANTIATE_XPEAK_ALL(float);
    NOA_INSTANTIATE_XPEAK_ALL(double);
}

namespace noa::signal::fft {
    template<Remap REMAP, typename Real, typename>
    void xcorr(const Array<Complex<Real>>& lhs,
               const Array<Complex<Real>>& rhs, dim4_t shape,
               const Array<Real>& coefficients) {
        NOA_CHECK(!lhs.empty() && !rhs.empty() && !coefficients.empty(), "Empty array detected");
        NOA_CHECK(coefficients.elements() == shape[0] &&
                  indexing::isVector(coefficients.shape()) &&
                  coefficients.contiguous(),
                  "The number of coefficients, specified as a contiguous vector, should be equal to the number "
                  "of batches. Got {} coefficients and {} output batches", coefficients.elements(), shape[0]);

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
        NOA_CHECK(coefficients.dereferenceable(), "The coefficients should be accessible to the CPU");
        if (coefficients.device() != lhs.device())
            Stream::current(coefficients.device()).synchronize();

        Stream& stream = Stream::current(lhs.device());
        if (stream.device().cpu()) {
            cpu::signal::fft::xcorr<REMAP>(
                    lhs.share(), lhs_strides, rhs.share(), rhs_strides,
                    shape, coefficients.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::xcorr<REMAP>(
                    lhs.share(), lhs_strides, rhs.share(), rhs_strides,
                    shape, coefficients.share(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Real, typename>
    Real xcorr(const Array<Complex<Real>>& lhs, const Array<Complex<Real>>& rhs, dim4_t shape) {
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
                    lhs.share(), lhs_strides, rhs.share(), rhs_strides, shape, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    #define NOA_INSTANTIATE_XCORR(R, T) \
    template void xcorr<R, T, void>(const Array<Complex<T>>&, const Array<Complex<T>>&, dim4_t, const Array<T>&); \
    template T xcorr<R, T, void>(const Array<Complex<T>>&, const Array<Complex<T>>&, dim4_t)

    #define NOA_INSTANTIATE_XCORR_ALL(T)    \
    NOA_INSTANTIATE_XCORR(Remap::F2F, T);   \
    NOA_INSTANTIATE_XCORR(Remap::FC2FC, T); \
    NOA_INSTANTIATE_XCORR(Remap::H2H, T);   \
    NOA_INSTANTIATE_XCORR(Remap::HC2HC, T)

    NOA_INSTANTIATE_XCORR_ALL(float);
    NOA_INSTANTIATE_XCORR_ALL(double);
}
