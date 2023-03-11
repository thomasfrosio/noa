#pragma once

#include "noa/cpu/geometry/Shape.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Shape.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::geometry::details {
    using namespace ::noa::fft;
    template<i32 NDIM, typename Value, typename Matrix, typename Functor, typename CValue>
    constexpr bool is_valid_shape_v =
            noa::traits::is_any_v<Value, f32, f64, c32, c64> &&
            std::is_same_v<CValue, noa::traits::value_type_t<Value>> &&
            noa::traits::is_any_v<Functor, noa::multiply_t, noa::plus_t> &&
            (NDIM == 2 && noa::traits::is_any_v<Matrix, Float22, Float23, Float33> ||
             NDIM == 3 && noa::traits::is_any_v<Matrix, Float33, Float34, Float44>);

    template<i32 NDIM, typename Matrix>
    constexpr auto extract_square_or_truncated_matrix(const Matrix& matrix) noexcept {
        if constexpr (NDIM == 2 && noa::traits::is_mat33_v<Matrix>)
            return Float23(matrix);
        else if constexpr (NDIM == 3 && noa::traits::is_mat44_v<Matrix>)
            return Float34(matrix);
        else
            return matrix;
    }
}

namespace noa::geometry {
    /// Returns or applies an elliptical mask.
    /// \tparam Matrix      2D case: Float22, Float23, or Float33.
    ///                     3D case: Float33, Float34, or Float44.
    /// \tparam Functor     noa::multiply_t, noa::plus_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input.
    /// \param center       (D)HW center of the ellipse.
    /// \param radius       (D)HW radius, in elements, of the ellipse.
    /// \param edge_size    Width, in elements, of the cosine edge, including the zero.
    /// \param inv_matrix   Inverse (D)HW (affine) matrix to apply on the ellipse.
    ///                     For non-affine matrices, the rotation center is located at \p center.
    /// \param functor      Operator defining how to apply the mask onto \p input.
    ///                     This is ignored if \p input is empty.
    /// \param cvalue       Real value of the mask. Elements outside the mask are set to 0.
    /// \param invert       Whether the mask should be inverted, i.e. elements inside the mask are set to 0,
    ///                     and elements outside the mask are set to \p cvalue.
    template<typename Output,
             typename Input = View<noa::traits::value_type_t<Output>>,
             typename Matrix = Float33, typename Functor = noa::multiply_t,
             typename CValue = noa::traits::value_type_t<noa::traits::value_type_t<Output>>, size_t N,
             typename = std::enable_if_t<
                     noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
                     noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
                     noa::traits::are_almost_same_value_type_v<Input, Output> &&
                     details::is_valid_shape_v<N, noa::traits::value_type_t<Output>, Matrix, Functor, CValue>>>
    void ellipse(const Input& input, const Output& output,
                 const Vec<f32, N>& center, const Vec<f32, N>& radius, f32 edge_size,
                 Matrix inv_matrix = {}, Functor functor = {},
                 CValue cvalue = CValue{1}, bool invert = false) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        NOA_CHECK(output.shape().ndim() <= N,
                  "3D arrays are not supported with 2D ellipses. Use 3D ellipses to support 2D and 3D arrays");
        const bool is_empty = input.is_empty();

        auto input_strides = input.strides();
        if (!is_empty && !noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(is_empty || device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=](){
                cpu::geometry::ellipse(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        center, radius, edge_size,
                        details::extract_square_or_truncated_matrix<N>(inv_matrix),
                        functor, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::ellipse(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::extract_square_or_truncated_matrix<N>(inv_matrix),
                    functor, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns or applies a spherical mask.
    /// \tparam Matrix      2D case: Float22, Float23, or Float33.
    ///                     3D case: Float33, Float34, or Float44.
    /// \tparam Functor     noa::multiply_t, noa::plus_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input.
    /// \param center       (D)HW center of the sphere.
    /// \param radius       Radius, in elements, of the sphere.
    /// \param edge_size    Width, in elements, of the cosine edge, including the zero.
    /// \param inv_matrix   Inverse (D)HW (affine) matrix to apply on the ellipse.
    ///                     For non-affine matrices, the rotation center is located at \p center.
    /// \param functor      Operator defining how to apply the mask onto \p input. This is ignored if \p input is empty.
    /// \param cvalue       Value of the mask. Elements outside the mask are set to 0.
    /// \param invert       Whether the mask should be inverted, i.e. elements inside the mask are set to 0,
    ///                     and elements outside the mask are set to \p cvalue.
    template<typename Output,
             typename Input = View<noa::traits::value_type_t<Output>>,
             typename Matrix = Float33, typename Functor = noa::multiply_t,
             typename CValue = noa::traits::value_type_t<noa::traits::value_type_t<Output>>, size_t N,
             typename = std::enable_if_t<
                     noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
                     noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
                     noa::traits::are_almost_same_value_type_v<Input, Output> &&
                     details::is_valid_shape_v<N, noa::traits::value_type_t<Output>, Matrix, Functor, CValue>>>
    void sphere(const Input& input, const Output& output,
                const Vec<f32, N>& center, f32 radius, f32 edge_size,
                Matrix inv_matrix = {}, Functor functor = {},
                CValue cvalue = CValue{1}, bool invert = false) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        NOA_CHECK(output.shape().ndim() <= N,
                  "3D arrays are not supported with 2D spheres. Use 3D spheres to support 2D and 3D arrays");
        const bool is_empty = input.is_empty();

        auto input_strides = input.strides();
        if (!is_empty && !noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(is_empty || device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=](){
                cpu::geometry::sphere(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        center, radius, edge_size,
                        details::extract_square_or_truncated_matrix<N>(inv_matrix),
                        functor, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::sphere(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::extract_square_or_truncated_matrix<N>(inv_matrix),
                    functor, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns or applies a rectangular mask.
    /// \tparam Matrix      2D case: Float22, Float23, or Float33.
    ///                     3D case: Float33, Float34, or Float44.
    /// \tparam Functor     noa::multiply_t, noa::plus_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input.
    /// \param center       (D)HW center of the rectangle.
    /// \param radius       (D)HW radius, in elements, of the rectangle.
    /// \param edge_size    Width, in elements, of the cosine edge, including the zero.
    /// \param inv_matrix   Inverse (D)HW (affine) matrix to apply on the ellipse.
    ///                     For non-affine matrices, the rotation center is located at \p center.
    /// \param functor      Operator defining how to apply the mask onto \p input. This is ignored if \p input is empty.
    /// \param cvalue       Value of the mask. Elements outside the mask are set to 0.
    /// \param invert       Whether the mask should be inverted, i.e. elements inside the mask are set to 0,
    ///                     and elements outside the mask are set to \p cvalue.
    template<typename Output,
             typename Input = View<noa::traits::value_type_t<Output>>,
             typename Matrix = Float33, typename Functor = noa::multiply_t,
             typename CValue = noa::traits::value_type_t<noa::traits::value_type_t<Output>>, size_t N,
             typename = std::enable_if_t<
                     noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
                     noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
                     noa::traits::are_almost_same_value_type_v<Input, Output> &&
                     details::is_valid_shape_v<N, noa::traits::value_type_t<Output>, Matrix, Functor, CValue>>>
    void rectangle(const Input& input, const Output& output,
                   const Vec<f32, N>& center, const Vec<f32, N>& radius, f32 edge_size,
                   Matrix inv_matrix = {}, Functor functor = {},
                   CValue cvalue = CValue{1}, bool invert = false) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        NOA_CHECK(output.shape().ndim() <= N,
                  "3D arrays are not supported with 2D rectangles. Use 3D rectangles to support 2D and 3D arrays");
        const bool is_empty = input.is_empty();

        auto input_strides = input.strides();
        if (!is_empty && !noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(is_empty || device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=](){
                cpu::geometry::rectangle(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        center, radius, edge_size,
                        details::extract_square_or_truncated_matrix<N>(inv_matrix),
                        functor, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::rectangle(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::extract_square_or_truncated_matrix<N>(inv_matrix),
                    functor, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns or applies a cylindrical mask.
    /// \tparam Matrix      Float33, Float34, or Float44.
    /// \tparam Functor     noa::multiply_t, noa::plus_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input.
    /// \param center       DHW center of the cylinder, in \p T elements.
    /// \param radius       Radius of the cylinder.
    /// \param length       Length of the cylinder along the depth dimension.
    /// \param edge_size    Width, in elements, of the cosine edge, including the zero.
    /// \param inv_matrix   Inverse DHW (affine) matrix to apply on the ellipse.
    ///                     For non-affine matrices, the rotation center is located at \p center.
    /// \param functor      Operator defining how to apply the mask onto \p input. This is ignored if \p input is empty.
    /// \param cvalue       Value of the mask. Elements outside the mask are set to 0.
    /// \param invert       Whether the mask should be inverted, i.e. elements inside the mask are set to 0,
    ///                     and elements outside the mask are set to \p cvalue.
    template<typename Output,
             typename Input = View<noa::traits::value_type_t<Output>>,
             typename Matrix = Float33, typename Functor = noa::multiply_t,
             typename CValue = noa::traits::value_type_t<noa::traits::value_type_t<Output>>, size_t N,
             typename = std::enable_if_t<
                     noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
                     noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
                     noa::traits::are_almost_same_value_type_v<Input, Output> &&
                     details::is_valid_shape_v<3, noa::traits::value_type_t<Output>, Matrix, Functor, CValue>>>
    void cylinder(const Input& input, const Output& output,
                  const Vec3<f32>& center, f32 radius, f32 length, f32 edge_size,
                  Matrix inv_matrix = {}, Functor functor = {},
                  CValue cvalue = CValue{1}, bool invert = false) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        const bool is_empty = input.is_empty();

        auto input_strides = input.strides();
        if (!is_empty && !noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(is_empty || device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=](){
                cpu::geometry::cylinder(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        center, radius, length, edge_size,
                        details::extract_square_or_truncated_matrix<3>(inv_matrix),
                        functor, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::cylinder(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    center, radius, length, edge_size,
                    details::extract_square_or_truncated_matrix<3>(inv_matrix),
                    functor, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
