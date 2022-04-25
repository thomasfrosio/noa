#pragma once

#include "noa/cpu/geometry/Prefilter.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Prefilter.h"
#endif

#include "noa/unified/Array.h"

namespace noa::geometry::bspline {
    /// Applies a prefilter to \a input so that the cubic B-spline values will pass through the sample data.
    /// \details From Danny Ruijters:
    ///          "When the approach described above is directly applied, it will result in smoothened images.
    ///          This is caused by the fact that the cubic B-spline filtering yields a function that does not
    ///          pass through its coefficients (i.e. texture values). In order to wind up with a cubic B-spline
    ///          interpolated image that passes through the original samples, we need to pre-filter the texture".
    /// \see http://www2.cs.uregina.ca/~anima/408/Notes/Interpolation/UniformBSpline.htm
    /// \see http://www.dannyruijters.nl/cubicinterpolation/ for more details.
    ///
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] input        Input array.
    /// \param[out] output      Output array. Can be equal to \a input.
    template<typename T>
    void prefilter(const Array<T>& input, const Array<T>& output) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::bspline::prefilter(
                    input.share(), input_stride,
                    output.share(), output.stride(),
                    output.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::bspline::prefilter(
                    input.share(), input_stride,
                    output.share(), output.stride(),
                    output.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
