#pragma once

#include "noa/cpu/geometry/fft/Polar.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Polar.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/Texture.hpp"

namespace noa::geometry::fft::details {
    template<typename Input, typename Value>
    void cartesian2polar_check_parameters(const Input& input,
                                          const Shape4<i64>& input_shape,
                                          const Array<Value>& output) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input array ({}) is not compatible with the number of "
                  "batches in the output array ({})", input.shape()[0], output.shape()[0]);
        NOA_CHECK(input.shape()[3] == input_shape[3] / 2 + 1 &&
                  input.shape()[2] == input_shape[2] &&
                  input.shape()[1] == input_shape[1],
                  "The non-redundant FFT with shape {} doesn't match the logical shape {}",
                  input.shape(), input_shape);
        NOA_CHECK(input.shape()[1] == 1 && output.shape()[1] == 1,
                  "3D arrays are not supported");

        if constexpr (noa::traits::is_array_or_view_v<Input>) {
            NOA_CHECK(output.device() == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), output.device());
            NOA_CHECK(!noa::indexing::are_overlapped(input, output),
                      "The input and output arrays should not overlap");
            NOA_CHECK(noa::indexing::are_elements_unique(output.strides(), output.shape()),
                      "The elements in the output should not overlap in memory, "
                      "otherwise a data-race might occur. Got output strides:{} and shape:{}",
                      output.strides(), output.shape());
        } else {
            NOA_CHECK(input.device() == output.device(),
                      "The input texture and output array must be on the same device, "
                      "but got input:{} and output:{}", input.device(), output.device());
        }
    }
}

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Transforms 2D FFT(s) to (log-)polar coordinates.
    /// \tparam REMAP           Only HC2FC is currently supported. The output is denoted as "FC" (full-centered)
    ///                         to emphasize that it has a full shape (equals to \p polar_shape) and can map the
    ///                         entire angular range (e.g. 0 to 2PI).
    /// \param[in] cartesian    Non-redundant centered 2D FFT to interpolate onto the new coordinate system.
    /// \param cartesian_shape  BDHW logical shape of \p cartesian.
    /// \param[out] polar       Transformed 2D array on the (log-)polar grid.
    ///                         The width dimension is the radius rho, from and to \p radius_range.
    ///                         The height dimension is the angle phi, from and to \p angle_range.
    /// \param frequency_range  Frequency [start,end] range of the bounding shells to transform, in cycle/pixels.
    ///                         While Nyquist is at 0.5, higher values can be specified.
    /// \param angle_range      Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    ///                         While the range naturally included in the non-redundant centered FFT is [-pi/2, pi/2],
    ///                         this range can include the entire unit circle, e.g. [-pi, pi].
    /// \param log              Whether log-polar coordinates should be computed instead.
    /// \param interp_mode      Interpolation method used to interpolate the values onto the new grid.
    ///                         Cubic interpolations are not supported.
    template<Remap REMAP, typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             REMAP == Remap::HC2FC>>
    void cartesian2polar(const Input& cartesian, const Shape4<i64>& cartesian_shape, const Output& polar,
                         const Vec2<f32>& frequency_range, const Vec2<f32>& angle_range,
                         bool log = false, InterpMode interp_mode = InterpMode::LINEAR) {
        details::cartesian2polar_check_parameters(cartesian, cartesian_shape, polar);

        const Device device = polar.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::cartesian2polar<REMAP>(
                        cartesian.get(), cartesian.strides(), cartesian_shape,
                        polar.get(), polar.strides(), polar.shape(),
                        frequency_range, angle_range, log, interp_mode, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::cartesian2polar<REMAP>(
                    cartesian.get(), cartesian.strides(), cartesian_shape,
                    polar.get(), polar.strides(), polar.shape(),
                    frequency_range, angle_range, log, interp_mode, cuda_stream);
            cuda_stream.enqueue_attach(cartesian.share(), polar.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Transforms 2D FFT(s) to (log-)polar coordinates.
    /// This functions has the same features and limitations as the overload taking arrays.
    template<Remap REMAP, typename Value, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_any_v<Output, Value> && REMAP == Remap::HC2FC>>
    void cartesian2polar(const Texture<Value>& cartesian, const Shape4<i64>& cartesian_shape, const Output& polar,
                         const Vec2<f32>& frequency_range, const Vec2<f32>& angle_range,
                         bool log = false) {
        const Device device = polar.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            const cpu::Texture<Value>& texture = cartesian.cpu();
            const Array<Value> cartesian_array(texture.ptr, cartesian.shape(), texture.strides, cartesian.options());
            cartesian2polar<REMAP>(cartesian_array, cartesian_shape, polar,
                                   frequency_range, angle_range, log, cartesian.interp_mode());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!noa::traits::is_any_v<Value, f32, c32>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported by this function");
            } else {
                details::cartesian2polar_check_parameters(cartesian, cartesian_shape, polar);
                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = cartesian.cuda();
                cuda::geometry::fft::cartesian2polar(
                        texture.array.get(), *texture.texture, cartesian.interp_mode(), cartesian_shape,
                        polar.get(), polar.strides(), polar.shape(),
                        frequency_range, angle_range, log, cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, polar.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
