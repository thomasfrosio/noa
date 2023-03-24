#pragma once

#include "noa/cpu/geometry/fft/Shape.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Shape.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::geometry::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, size_t N, typename Matrix>
    constexpr bool is_valid_shape_v =
            ((N == 2 && std::is_same_v<Matrix, Float22>) || (N == 3 && std::is_same_v<Matrix, Float33>)) &&
            (REMAP == F2F || REMAP == FC2FC || REMAP == F2FC || REMAP == FC2F);
}

namespace noa::geometry::fft {
    using Remap = ::noa::fft::Remap;

    /// Returns or applies an elliptical mask.
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F, FC2FC, F2FC and FC2F are supported.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       (D)HW center of the ellipse.
    /// \param radius       (D)HW radius, in elements, of the ellipse.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse (D)HW matrix to apply on the ellipse. The rotation center is located at \p center.
    /// \param cvalue       Value of the mask. Elements outside the mask are set to 0.
    /// \param invert       Whether the mask should be inverted, i.e. elements inside the mask are set to 0,
    ///                     and elements outside the mask are set to \p cvalue.
    template<Remap REMAP, typename Output, typename Matrix, size_t N,
            typename Input = View<const noa::traits::value_type_t<Output>>,
            typename CValue = noa::traits::value_type_t<Output>, typename = std::enable_if_t<
                    noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
                    noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
                    noa::traits::are_almost_same_value_type_v<Input, Output, CValue> &&
                    details::is_valid_shape_v<REMAP, N, Matrix>>>
    void ellipse(const Input& input, const Output& output,
                 const Vec<f32, N>& center, const Vec<f32, N>& radius, f32 edge_size,
                 const Matrix& inv_matrix = {}, CValue cvalue = CValue{1}, bool invert = false) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !noa::indexing::are_overlapped(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);
        NOA_CHECK(output.shape().ndim() <= static_cast<i64>(N),
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
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::ellipse<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        center, radius, edge_size, inv_matrix,
                        noa::multiply_t{}, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::ellipse<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix,
                    noa::multiply_t{}, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns or applies a spherical mask.
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F, FC2FC, F2FC and FC2F are supported.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       (D)HW center of the sphere.
    /// \param radius       Radius, in elements, of the sphere.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse (D)HW matrix to apply on the sphere. The rotation center is located at \p center.
    /// \param cvalue       Value of the mask. Elements outside the mask are set to 0.
    /// \param invert       Whether the mask should be inverted, i.e. elements inside the mask are set to 0,
    ///                     and elements outside the mask are set to \p cvalue.
    template<Remap REMAP, typename Output, typename Matrix, size_t N,
             typename Input = View<const noa::traits::value_type_t<Output>>,
             typename CValue = noa::traits::value_type_t<Output>, typename = std::enable_if_t<
                    noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
                    noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
                    noa::traits::are_almost_same_value_type_v<Input, Output, CValue> &&
                    details::is_valid_shape_v<REMAP, N, Matrix>>>
    void sphere(const Input& input, const Output& output,
                const Vec<f32, N>& center, f32 radius, f32 edge_size,
                const Matrix& inv_matrix = {}, CValue cvalue = CValue{1}, bool invert = false) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !noa::indexing::are_overlapped(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);
        NOA_CHECK(output.shape().ndim() <= static_cast<i64>(N),
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
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::sphere<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        center, radius, edge_size, inv_matrix,
                        noa::multiply_t{}, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::sphere<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix,
                    noa::multiply_t{}, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns or applies a rectangular mask.
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F, FC2FC, F2FC and FC2F are supported.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       (D)HW center of the rectangle.
    /// \param radius       (D)HW radius, in elements, of the rectangle.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse (D)HW matrix to apply on the rectangle. The rotation center is located at \p center.
    /// \param cvalue       Value of the mask. Elements outside the mask are set to 0.
    /// \param invert       Whether the mask should be inverted, i.e. elements inside the mask are set to 0,
    ///                     and elements outside the mask are set to \p cvalue.
    template<Remap REMAP, typename Output, typename Matrix, size_t N,
             typename Input = View<const noa::traits::value_type_t<Output>>,
             typename CValue = noa::traits::value_type_t<Output>, typename = std::enable_if_t<
                    noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
                    noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
                    noa::traits::are_almost_same_value_type_v<Input, Output, CValue> &&
                    details::is_valid_shape_v<REMAP, N, Matrix>>>
    void rectangle(const Input& input, const Output& output,
                   const Vec<f32, N>& center, const Vec<f32, N>& radius, f32 edge_size,
                   const Matrix& inv_matrix = {}, CValue cvalue = CValue{1}, bool invert = false) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !noa::indexing::are_overlapped(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);
        NOA_CHECK(output.shape().ndim() <= static_cast<i64>(N),
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
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::rectangle<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        center, radius, edge_size, inv_matrix,
                        noa::multiply_t{}, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::rectangle<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix,
                    noa::multiply_t{}, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns or applies a cylindrical mask.
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F, FC2FC, F2FC and FC2F are supported.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       DHW center of the cylinder, in \p T elements.
    /// \param radius       Radius of the cylinder.
    /// \param length       Length of the cylinder along the depth dimension.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse DHW matrix to apply on the cylinder. The rotation center is located at \p center.
    /// \param cvalue       Value of the mask. Elements outside the mask are set to 0.
    /// \param invert       Whether the mask should be inverted, i.e. elements inside the mask are set to 0,
    ///                     and elements outside the mask are set to \p cvalue.
    template<Remap REMAP, typename Output,
             typename Input = View<const noa::traits::value_type_t<Output>>,
             typename CValue = traits::value_type_t<Output>, typename = std::enable_if_t<
                    noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
                    noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
                    noa::traits::are_almost_same_value_type_v<Input, Output, CValue> &&
                    details::is_valid_shape_v<REMAP, 3, Float33>>>
    void cylinder(const Input& input, const Output& output,
                  const Vec3<f32>& center, f32 radius, f32 length, f32 edge_size,
                  const Float33 inv_matrix = {}, CValue cvalue = CValue{1}, bool invert = false) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !noa::indexing::are_overlapped(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);

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
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::cylinder<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        center, radius, length, edge_size, inv_matrix,
                        noa::multiply_t{}, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::cylinder<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    center, radius, length, edge_size, inv_matrix,
                    noa::multiply_t{}, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
