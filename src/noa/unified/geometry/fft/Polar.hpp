#pragma once

#include "noa/cpu/geometry/fft/Polar.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Polar.hpp"
#endif

#include "noa/core/fft/Frequency.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Texture.hpp"

// TODO Add polar2cartesian
// TODO Add rotation_average() for 2d only with frequency and angle range.
//      This should be able to take multiple angle ranges for the same input,
//      to "extract" multiple wedges efficiently.

namespace noa::geometry::fft::details {
    using Remap = noa::fft::Remap;
    namespace nt = noa::traits;

    template<Remap REMAP, typename Input, typename Output, typename Weight>
    struct is_valid_rotational_average {
        using input_value_type = nt::value_type_t<Input>;
        using output_value_type = nt::value_type_t<Output>;
        using weight_value_type = nt::value_type_t<Weight>;
        static constexpr bool is_valid_remap =
                REMAP == Remap::H2H || REMAP == Remap::HC2H ||
                REMAP == Remap::F2H || REMAP == Remap::FC2H;
        static constexpr bool value =
                is_valid_remap &&
                (nt::are_same_value_type_v<input_value_type, output_value_type, weight_value_type> &&
                 ((nt::are_all_same_v<input_value_type, output_value_type> &&
                   nt::are_real_or_complex_v<input_value_type, output_value_type>) ||
                  (nt::is_complex_v<input_value_type> &&
                   nt::is_real_v<output_value_type>)));
    };

    template<Remap REMAP, typename Input, typename Output, typename Weight>
    constexpr bool is_valid_rotational_average_v = is_valid_rotational_average<REMAP, Input, Output, Weight>::value;

    template<typename Input, typename Value>
    void cartesian2polar_check_parameters(
            const Input& input,
            const Shape4<i64>& input_shape,
            const Array<Value>& output
    ) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input array ({}) is not compatible with the number of "
                  "batches in the output array ({})", input.shape()[0], output.shape()[0]);
        NOA_CHECK(noa::all(input.shape().pop_front() == input_shape.pop_front().rfft()),
                  "The rfft with shape {} doesn't match the logical shape {}",
                  input.shape(), input_shape);
        NOA_CHECK(input.shape()[1] == 1 && output.shape()[1] == 1,
                  "3d arrays are not supported");

        if constexpr (noa::traits::is_array_or_view_v<Input>) {
            NOA_CHECK(output.device() == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input={} and output={}", input.device(), output.device());
            NOA_CHECK(!noa::indexing::are_overlapped(input, output),
                      "The input and output arrays should not overlap");
            NOA_CHECK(noa::indexing::are_elements_unique(output.strides(), output.shape()),
                      "The elements in the output should not overlap in memory, "
                      "otherwise a data-race might occur. Got output strides={} and shape={}",
                      output.strides(), output.shape());
        } else {
            NOA_CHECK(input.device() == output.device(),
                      "The input texture and output array must be on the same device, "
                      "but got input={} and output={}", input.device(), output.device());
        }
    }

    inline void set_polar_window_range_to_default(
            const Shape4<i64>& cartesian_shape,
            Vec2<f32>& frequency_range,
            Vec2<f32>& angle_range
    ) {
        if (noa::all(frequency_range == Vec2<f32>{})) {
            const auto size = noa::math::min(cartesian_shape.filter(2, 3));
            frequency_range = {0, noa::fft::highest_normalized_frequency(size)};
        }
        if (noa::all(angle_range == Vec2<f32>{}))
            angle_range = {0.f, noa::math::Constant<f32>::PI};
    }

    template<typename Input, typename Output, typename Weight>
    void rotational_average_check_parameters(
            const Input& input,
            const Output& output,
            const Weight& weights,
            const Shape4<i64>& shape
    ) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        const bool weights_is_empty = weights.is_empty();

        NOA_CHECK(shape[0] == input.shape()[0] &&
                  shape[0] == output.shape()[0] &&
                  (weights_is_empty || shape[0] == weights.shape()[0]),
                  "The numbers of batches do not match. "
                  "Got expected={}, input={}, output={}",
                  shape[0], input.shape()[0], output.shape()[0],
                  weights_is_empty ? "" : noa::string::format(" and weights={}", weights.shape()[0]));

        NOA_CHECK(noa::indexing::is_contiguous_vector_batched(output),
                  "The output must be a contiguous (batched) vector, but got shape={} and strides={}",
                  output.shape(), output.strides());

        const i64 input_shell_count = noa::math::min(shape.filter(2, 3)) / 2 + 1;
        const i64 output_shell_count = output.shape().pop_front().elements();
        NOA_CHECK(input_shell_count == output_shell_count,
                  "The number of shells does not match the input shape. "
                  "Got output shells={} and input (shape={}, shells={})",
                  output_shell_count, shape, input_shell_count);

        if (!weights_is_empty) {
            NOA_CHECK(noa::indexing::is_contiguous_vector_batched(weights),
                      "The weights must be a contiguous (batched) vector, but got shape={} and strides={}",
                      weights.shape(), weights.strides());

            const i64 weights_shell_count = weights.shape().pop_front().elements();
            NOA_CHECK(output_shell_count == weights_shell_count,
                      "The number of shells does not match the input shape. "
                      "Got output shells={} and weight shells={}",
                      output_shell_count, weights_shell_count);
        }

        NOA_CHECK(input.device() == output.device() &&
                  (weights_is_empty || weights.device() == output.device()),
                  "The arrays must be on the same device, but got input={}, output={}",
                  input.device(), output.device(),
                  weights_is_empty ? "" : noa::string::format(" and weights={}", weights.device()));
    }
}

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Transforms 2d dft(s) to polar coordinates.
    /// \tparam REMAP               Only HC2FC is currently supported. The output is denoted as "FC" (full-centered)
    ///                             to emphasize that it has a full shape (equals to \p polar_shape) and can map the
    ///                             entire angular range (e.g. 0 to 2PI).
    /// \param[in] cartesian        Centered 2d rfft to interpolate onto the new coordinate system.
    /// \param cartesian_shape      BDHW logical shape of \p cartesian.
    /// \param[out] polar           Transformed 2d array on the polar grid.
    ///                             The width dimension is the radius rho, from and to \p radius_range.
    ///                             The height dimension is the angle phi, from and to \p angle_range.
    ///                             If real and \p cartesian is complex, `abs(cartesian)` is first computed.
    /// \param frequency_range      Frequency [start,end] range of the bounding shells to transform, in cycle/pixels.
    ///                             Defaults to the [0, v], where v is the highest normalized frequency of min(height,width).
    /// \param frequency_endpoint   Whether the \p frequency_range 's end should be included in the range.
    /// \param angle_range          Angle [start,end) range increasing in the counterclockwise orientation, in radians.
    ///                             While the range naturally included in the non-redundant centered FFT is [-pi/2, pi/2],
    ///                             this range can include the entire unit circle, e.g. [-pi, pi]. Defaults to [0, pi).
    /// \param angle_endpoint       Whether the \p angle_range 's end should be included in the range.
    /// \param interp_mode          Interpolation method used to interpolate the values onto the new grid.
    ///                             Cubic interpolations are not supported.
    template<Remap REMAP, typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
             (noa::traits::are_almost_same_value_type_v<Input, Output> ||
              noa::traits::are_almost_same_value_type_v<Input, noa::traits::value_type_t<Output>>) &&
             REMAP == Remap::HC2FC>>
    void cartesian2polar(const Input& cartesian, const Shape4<i64>& cartesian_shape, const Output& polar,
                         Vec2<f32> frequency_range = {},
                         bool frequency_endpoint = true,
                         Vec2<f32> angle_range = {},
                         bool angle_endpoint = false,
                         InterpMode interp_mode = InterpMode::LINEAR) {
        details::cartesian2polar_check_parameters(cartesian, cartesian_shape, polar);
        details::set_polar_window_range_to_default(cartesian_shape, frequency_range, angle_range);

        const Device device = polar.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::cartesian2polar<REMAP>(
                        cartesian.get(), cartesian.strides(), cartesian_shape,
                        polar.get(), polar.strides(), polar.shape(),
                        frequency_range, frequency_endpoint, angle_range, angle_endpoint,
                        interp_mode, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::cartesian2polar<REMAP>(
                    cartesian.get(), cartesian.strides(), cartesian_shape,
                    polar.get(), polar.strides(), polar.shape(),
                    frequency_range, frequency_endpoint, angle_range, angle_endpoint,
                    interp_mode, cuda_stream);
            cuda_stream.enqueue_attach(cartesian.share(), polar.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Transforms 2d dft(s) to polar coordinates.
    /// This functions has the same features and limitations as the overload taking arrays.
    template<Remap REMAP, typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
             (noa::traits::are_almost_same_value_type_v<Texture<Input>, Output> ||
              noa::traits::are_almost_same_value_type_v<Texture<Input>, noa::traits::value_type_t<Output>>) &&
             REMAP == Remap::HC2FC>>
    void cartesian2polar(const Texture<Input>& cartesian, const Shape4<i64>& cartesian_shape, const Output& polar,
                         Vec2<f32> frequency_range = {},
                         bool frequency_endpoint = false,
                         Vec2<f32> angle_range = {},
                         bool angle_endpoint = false) {
        const Device device = polar.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            const cpu::Texture<Input>& texture = cartesian.cpu();
            const Array<Input> cartesian_array(texture.ptr, cartesian.shape(), texture.strides, cartesian.options());
            cartesian2polar<REMAP>(
                    cartesian_array, cartesian_shape, polar,
                    frequency_range, frequency_endpoint,
                    angle_range, angle_endpoint,
                    cartesian.interp_mode());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!noa::traits::is_any_v<Input, f32, c32>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported by this function");
            } else {
                details::cartesian2polar_check_parameters(cartesian, cartesian_shape, polar);
                details::set_polar_window_range_to_default(cartesian_shape, frequency_range, angle_range);

                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Input>& texture = cartesian.cuda();
                cuda::geometry::fft::cartesian2polar<Input>(
                        texture.array.get(), *texture.texture, cartesian.interp_mode(), cartesian_shape,
                        polar.get(), polar.strides(), polar.shape(),
                        frequency_range, frequency_endpoint, angle_range, angle_endpoint, cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, polar.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the rotational sum/average of a 2d or 3d dft.
    /// \tparam REMAP       Should be either H2H, HC2H, F2H or FC2H. The output layout is "H" to emphasize that
    ///                     the output shape is the half dimension size.
    /// \param[in] input    Input dft to reduce. Can be real or complex.
    /// \param[out] output  Rotational sum/average. Should be a (batched) contiguous vector of size min(shape) // 2 + 1.
    ///                     If real and \p input is complex, `abs(input)^2` is first computed.
    /// \param[out] weights Rotational weights. Can be empty. Otherwise, the weights are directly added to this array.
    /// \param shape        BDHW logical shape.
    /// \param average      Whether the rotational average should be computed instead of the rotational sum.
    ///
    /// \note If \p weights is empty and \p average is true, a temporary vector like \p output is allocated.
    template<noa::fft::Remap REMAP, typename Input, typename Output,
             typename Weight = View<noa::traits::value_type_t<Input>>, typename = std::enable_if_t<
             noa::traits::are_array_or_view_of_real_or_complex_v<Input, Output, Weight> &&
             details::is_valid_rotational_average_v<REMAP, Input, Output, Weight>>>
    void rotational_average(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const Weight& weights = {},
            bool average = true
    ) {
        details::rotational_average_check_parameters(input, output, weights, shape);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::rotational_average<REMAP>(
                        input.get(), input.strides(), shape,
                        output.get(), weights.get(), average, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::rotational_average<REMAP>(
                    input.get(), input.strides(), shape,
                    output.get(), weights.get(), average, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share(), weights.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
