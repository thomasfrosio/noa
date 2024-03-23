#pragma once

#include "noa/cpu/geometry/Prefilter.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Prefilter.cuh"
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
    template<typename Input, typename Output>
    requires (nt::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> and
              nt::is_varray_of_any_v<Output, f32, f64, c32, c64> and
              nt::are_almost_same_value_type_v<Input, Output>)
    void cubic_bspline_prefilter(const Input& input, const Output& output) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");

        auto input_strides = input.strides();
        check(ni::broadcast(input.shape(), input_strides, output.shape()),
              "Cannot broadcast an array of shape {} into an array of shape {}",
              input.shape(), output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::geometry::cubic_bspline_prefilter(
                        input.get(), input_strides,
                        output.get(), output.strides(),
                        output.shape(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::geometry::cubic_bspline_prefilter(
                    input.get(), input_strides,
                    output.get(), output.strides(),
                    output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input, output);
            #else
            panic("No GPU backend detected");
            #endif
        }
    }
}
