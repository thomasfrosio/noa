#pragma once

#include "noa/cpu/geometry/Polar.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Polar.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/Texture.hpp"

namespace noa::geometry::details {
    template<typename Input, typename Output>
    void polar_check_parameters(const Input& input, const Output& output) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input array ({}) is not compatible with the number of "
                  "batches in the output array ({})", input.shape()[0], output.shape()[0]);
        NOA_CHECK(input.shape()[1] == 1 && output.shape()[1] == 1, "3D arrays are not supported");

        NOA_CHECK(input.device() == output.device(),
                  "The input and output arrays must be on the same device, "
                  "but got input:{} and output:{}", input.device(), output.device());

        if constexpr (noa::traits::is_array_or_view_v<Input>) {
            NOA_CHECK(!noa::indexing::are_overlapped(input, output),
                      "Input and output arrays should not overlap");
            NOA_CHECK(noa::indexing::are_elements_unique(output.strides(), output.shape()),
                      "The elements in the output should not overlap in memory, "
                      "otherwise a data-race might occur. Got output strides:{} and shape:{}",
                      output.strides(), output.shape());
        }
    }
}

namespace noa::geometry {
    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// \param[in] cartesian        Input 2D cartesian array to interpolate onto the new coordinate system.
    /// \param[out] polar           Transformed 2D array on the (log-)polar grid.
    ///                             The width dimension is the radius rho, from and to \p radius_range.
    ///                             The height dimension is the angle phi, from and to \p angle_range.
    /// \param cartesian_center     HW transformation center.
    /// \param radius_range         Radius [start,end] range of the bounding circle to transform, in pixels.
    /// \param angle_range          Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log                  Whether log-polar coordinates should be computed instead.
    /// \param interpolation_mode   Interpolation method used to interpolate the values onto the new grid.
    ///                             Out-of-bounds elements are set to zero.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void cartesian2polar(const Input& cartesian, const Output& polar,
                         const Vec2<f32>& cartesian_center,
                         const Vec2<f32>& radius_range,
                         const Vec2<f32>& angle_range,
                         bool log = false,
                         InterpMode interpolation_mode = InterpMode::LINEAR) {
        details::polar_check_parameters(cartesian, polar);

        const Device device = polar.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::geometry::cartesian2polar(
                        cartesian.get(), cartesian.strides(), cartesian.shape(),
                        polar.get(), polar.strides(), polar.shape(),
                        cartesian_center, radius_range, angle_range, log,
                        interpolation_mode, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::cartesian2polar(
                    cartesian.get(), cartesian.strides(), cartesian.shape(),
                    polar.get(), polar.strides(), polar.shape(),
                    cartesian_center, radius_range, angle_range, log,
                    interpolation_mode, cuda_stream);
            cuda_stream.enqueue_attach(cartesian.share(), polar.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// This functions has the same features and limitations as the overload taking arrays.
    template<typename Value, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_any_v<Output, Value>>>
    void cartesian2polar(const Texture<Value>& cartesian, const Output& polar,
                         const Vec2<f32>& cartesian_center,
                         const Vec2<f32>& radius_range,
                         const Vec2<f32>& angle_range,
                         bool log = false) {
        const Device device = cartesian.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            const cpu::Texture<Value>& texture = cartesian.cpu();
            const Array<Value> cartesian_array(texture.ptr, cartesian.shape(), texture.strides, cartesian.options());
            cartesian2polar(cartesian_array, polar,
                            cartesian_center, radius_range, angle_range,
                            log, cartesian.interp_mode());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!noa::traits::is_any_v<Value, f32, c32>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                details::polar_check_parameters(cartesian, polar);
                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = cartesian.cuda();
                cuda::geometry::cartesian2polar(
                        texture.array.get(), *texture.texture, cartesian.interp_mode(), cartesian.shape(),
                        polar.get(), polar.strides(), polar.shape(),
                        cartesian_center, radius_range, angle_range,
                        log, cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, polar.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Transforms 2D array(s) from (log-)polar to cartesian coordinates.
    /// \param[in] polar        Input 2D polar array to interpolate onto the new coordinate system.
    ///                         The width is the radius rho, from and to \p radius_range.
    ///                         The height is the angle phi, from and to \p angle_range.
    /// \param[out] cartesian   Transformed 2D array on the cartesian grid.
    /// \param cartesian_center HW transformation center.
    /// \param radius_range     Radius [start,end] range of the bounding circle, in pixels.
    /// \param angle_range      Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log              Whether this is a log-polar coordinates system.
    /// \param interp           Interpolation method used to interpolate the values onto the new grid.
    ///                         Out-of-bounds elements are set to zero.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void polar2cartesian(const Input& polar, const Output& cartesian,
                         const Vec2<f32>& cartesian_center,
                         const Vec2<f32>& radius_range,
                         const Vec2<f32>& angle_range,
                         bool log = false,
                         InterpMode interpolation_mode = InterpMode::LINEAR) {
        details::polar_check_parameters(polar, cartesian);

        const Device device = cartesian.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::geometry::polar2cartesian(
                        polar.get(), polar.strides(), polar.shape(),
                        cartesian.get(), cartesian.strides(), cartesian.shape(),
                        cartesian_center, radius_range, angle_range, log,
                        interpolation_mode, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::polar2cartesian(
                    polar.get(), polar.strides(), polar.shape(),
                    cartesian.get(), cartesian.strides(), cartesian.shape(),
                    cartesian_center, radius_range, angle_range, log,
                    interpolation_mode, cuda_stream);
            cuda_stream.enqueue_attach(polar.share(), cartesian.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Transforms 2D array(s) from (log-)polar to cartesian coordinates.
    /// This functions has the same features and limitations as the overload taking arrays.
    template<typename Value, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_any_v<Output, Value>>>
    void polar2cartesian(const Texture<Value>& polar, const Output& cartesian,
                         const Vec2<f32>& cartesian_center,
                         const Vec2<f32>& radius_range,
                         const Vec2<f32>& angle_range,
                         bool log = false) {
        const Device device = cartesian.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            const cpu::Texture<Value>& texture = polar.cpu();
            const Array<Value> polar_array(texture.ptr, polar.shape(), texture.strides, polar.options());
            polar2cartesian(polar_array, cartesian,
                            cartesian_center, radius_range, angle_range,
                            log, cartesian.interp_mode());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!noa::traits::is_any_v<Value, f32, c32>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                details::polar_check_parameters(polar, cartesian);
                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = polar.cuda();
                cuda::geometry::polar2cartesian(
                        texture.array.get(), *texture.texture, polar.interp_mode(), polar.shape(),
                        cartesian.get(), cartesian.strides(), cartesian.shape(),
                        cartesian_center, radius_range, angle_range,
                        log, cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, cartesian.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
