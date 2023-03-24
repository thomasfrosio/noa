#pragma once

#include "noa/cpu/geometry/Prefilter.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Prefilter.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::geometry {
    /// Applies a prefilter to \p input so that the cubic B-spline values will pass through the sample data.
    /// \details Without prefiltering, cubic B-spline filtering results in smoothened images. This is caused by the
    ///          fact that the cubic B-spline filtering yields a function that does not pass through its coefficients.
    ///          In order to wind up with a cubic B-spline interpolated image that passes through the original samples,
    ///          we need to pre-filter the input.
    /// \param[in] input    Input array of f32, f64, c32, or c64.
    /// \param[out] output  Output array. Can be equal to \p input.
    ///
    /// \see http://www2.cs.uregina.ca/~anima/408/Notes/Interpolation/UniformBSpline.htm
    /// \see http://www.dannyruijters.nl/cubicinterpolation/ for more details.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void cubic_bspline_prefilter(const Input& input, const Output& output) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");

        auto input_strides = input.strides();
        if (!noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::geometry::cubic_bspline_prefilter(
                        input.get(), input_strides,
                        output.get(), output.strides(),
                        output.shape(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::cubic_bspline_prefilter(
                    input.get(), input_strides,
                    output.get(), output.strides(),
                    output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
