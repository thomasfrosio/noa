#pragma once

#include "noa/cpu/geometry/fft/Shape.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Shape.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/geometry/Shape.hpp"

namespace noa::geometry::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, size_t N, typename Matrix>
    constexpr bool is_valid_shape_v =
            ((N == 2 && (noa::traits::is_any_v<Matrix, Empty, Float22> ||
                         noa::traits::is_array_or_view_of_almost_any_v<Matrix, Float22>)) ||
             (N == 3 && (noa::traits::is_any_v<Matrix, Empty, Float33> ||
                         noa::traits::is_array_or_view_of_almost_any_v<Matrix, Float33>))) &&
            (REMAP == F2F || REMAP == FC2FC);

    template<size_t N, typename Matrix>
    constexpr auto extract_matrix(const Matrix& matrix) noexcept {
        if constexpr (noa::traits::is_array_or_view_of_almost_any_v<Matrix, Float22, Float33>) {
            using const_ptr_t = const typename Matrix::mutable_value_type*;
            return const_ptr_t(matrix.get());
        } else if constexpr (std::is_empty_v<Matrix>) {
            using matrix_t = std::conditional_t<N == 2, const Float22*, const Float33*>;
            return matrix_t{};
        } else {
            return matrix;
        }
    }
}

namespace noa::geometry::fft {
    using Remap = ::noa::fft::Remap;

    /// Returns or applies an elliptical mask.
    /// \details The mask can be directly saved in \p output or applied (\p see functor) to \p input and
    ///          save in \p output. The same transformation can be applied to every batch or there can be
    ///          one transformation per batch (\p see inv_matrix). Additionally, if \p input and \p output are
    ///          not batched, multiple matrices can still be passed to generate multiple geometric shapes within
    ///          the same array. In this case, multiple "masks" are computed, reduced to a single "mask",
    ///          which is then applied to \p input and/or saved in \p output. These "masks" are sum-reduced
    ///          if \p invert is false or multiplied together if \p invert is true.
    ///
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F and FC2FC is supported.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       (D)HW center of the ellipse.
    /// \param radius       (D)HW radius, in elements, of the ellipse.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse (D)HW matrix to apply on the ellipse. The rotation center is located at \p center.
    /// \param cvalue       Value of the mask. Elements outside the mask are set to 0.
    /// \param invert       Whether the mask should be inverted, i.e. elements inside the mask are set to 0,
    ///                     and elements outside the mask are set to \p cvalue.
    template<Remap REMAP, typename Output, size_t N, typename Matrix = Empty,
            typename Input = View<const noa::traits::value_type_t<Output>>,
            typename CValue = noa::traits::value_type_t<Output>, typename = std::enable_if_t<
                    noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
                    noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
                    noa::traits::are_almost_same_value_type_v<Input, Output, CValue> &&
                    details::is_valid_shape_v<REMAP, N, Matrix>>>
    void ellipse(const Input& input, const Output& output,
                 const Vec<f32, N>& center, const Vec<f32, N>& radius, f32 edge_size,
                 const Matrix& inv_matrix = {}, CValue cvalue = CValue{1}, bool invert = false) {

        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !noa::indexing::are_overlapped(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);

        // FIXME Use std::tie because clangd? doesn't like auto[] elements captured in lambdas. Compiler is fine...
        Strides4<i64> input_strides, output_strides;
        Shape4<i64> final_shape;
        std::tie(input_strides, output_strides, final_shape) =
                noa::geometry::details::check_shape_parameters<N>(input, output, inv_matrix);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::ellipse<REMAP>(
                        input.get(), input_strides,
                        output.get(), output_strides, final_shape,
                        center, radius, edge_size, details::extract_matrix<N>(inv_matrix),
                        noa::multiply_t{}, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::ellipse<REMAP>(
                    input.get(), input_strides,
                    output.get(), output_strides, final_shape,
                    center, radius, edge_size, details::extract_matrix<N>(inv_matrix),
                    noa::multiply_t{}, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns or applies a spherical mask.
    /// \details The mask can be directly saved in \p output or applied (\p see functor) to \p input and
    ///          save in \p output. The same transformation can be applied to every batch or there can be
    ///          one transformation per batch (\p see inv_matrix). Additionally, if \p input and \p output are
    ///          not batched, multiple matrices can still be passed to generate multiple geometric shapes within
    ///          the same array. In this case, multiple "masks" are computed, reduced to a single "mask",
    ///          which is then applied to \p input and/or saved in \p output. These "masks" are sum-reduced
    ///          if \p invert is false or multiplied together if \p invert is true.
    ///
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F and FC2FC is supported.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       (D)HW center of the sphere.
    /// \param radius       Radius, in elements, of the sphere.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse (D)HW matrix to apply on the sphere. The rotation center is located at \p center.
    /// \param cvalue       Value of the mask. Elements outside the mask are set to 0.
    /// \param invert       Whether the mask should be inverted, i.e. elements inside the mask are set to 0,
    ///                     and elements outside the mask are set to \p cvalue.
    template<Remap REMAP, typename Output, size_t N, typename Matrix = Empty,
             typename Input = View<const noa::traits::value_type_t<Output>>,
             typename CValue = noa::traits::value_type_t<Output>, typename = std::enable_if_t<
                    noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
                    noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
                    noa::traits::are_almost_same_value_type_v<Input, Output, CValue> &&
                    details::is_valid_shape_v<REMAP, N, Matrix>>>
    void sphere(const Input& input, const Output& output,
                const Vec<f32, N>& center, f32 radius, f32 edge_size,
                const Matrix& inv_matrix = {}, CValue cvalue = CValue{1}, bool invert = false) {

        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !noa::indexing::are_overlapped(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);

        // FIXME Use std::tie because clangd? doesn't like auto[] elements captured in lambdas. Compiler is fine...
        Strides4<i64> input_strides, output_strides;
        Shape4<i64> final_shape;
        std::tie(input_strides, output_strides, final_shape) =
                noa::geometry::details::check_shape_parameters<N>(input, output, inv_matrix);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::sphere<REMAP>(
                        input.get(), input_strides,
                        output.get(), output_strides, final_shape,
                        center, radius, edge_size, details::extract_matrix<N>(inv_matrix),
                        noa::multiply_t{}, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::sphere<REMAP>(
                    input.get(), input_strides,
                    output.get(), output_strides, final_shape,
                    center, radius, edge_size, details::extract_matrix<N>(inv_matrix),
                    noa::multiply_t{}, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns or applies a rectangular mask.
    /// \details The mask can be directly saved in \p output or applied (\p see functor) to \p input and
    ///          save in \p output. The same transformation can be applied to every batch or there can be
    ///          one transformation per batch (\p see inv_matrix). Additionally, if \p input and \p output are
    ///          not batched, multiple matrices can still be passed to generate multiple geometric shapes within
    ///          the same array. In this case, multiple "masks" are computed, reduced to a single "mask",
    ///          which is then applied to \p input and/or saved in \p output. These "masks" are sum-reduced
    ///          if \p invert is false or multiplied together if \p invert is true.
    ///
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F and FC2FC is supported.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       (D)HW center of the rectangle.
    /// \param radius       (D)HW radius, in elements, of the rectangle.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse (D)HW matrix to apply on the rectangle. The rotation center is located at \p center.
    /// \param cvalue       Value of the mask. Elements outside the mask are set to 0.
    /// \param invert       Whether the mask should be inverted, i.e. elements inside the mask are set to 0,
    ///                     and elements outside the mask are set to \p cvalue.
    template<Remap REMAP, typename Output, size_t N, typename Matrix = Empty,
             typename Input = View<const noa::traits::value_type_t<Output>>,
             typename CValue = noa::traits::value_type_t<Output>, typename = std::enable_if_t<
                    noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
                    noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
                    noa::traits::are_almost_same_value_type_v<Input, Output, CValue> &&
                    details::is_valid_shape_v<REMAP, N, Matrix>>>
    void rectangle(const Input& input, const Output& output,
                   const Vec<f32, N>& center, const Vec<f32, N>& radius, f32 edge_size,
                   const Matrix& inv_matrix = {}, CValue cvalue = CValue{1}, bool invert = false) {

        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !noa::indexing::are_overlapped(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);

        // FIXME Use std::tie because clangd? doesn't like auto[] elements captured in lambdas. Compiler is fine...
        Strides4<i64> input_strides, output_strides;
        Shape4<i64> final_shape;
        std::tie(input_strides, output_strides, final_shape) =
                noa::geometry::details::check_shape_parameters<N>(input, output, inv_matrix);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::rectangle<REMAP>(
                        input.get(), input_strides,
                        output.get(), output_strides, final_shape,
                        center, radius, edge_size, details::extract_matrix<N>(inv_matrix),
                        noa::multiply_t{}, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::rectangle<REMAP>(
                    input.get(), input_strides,
                    output.get(), output_strides, final_shape,
                    center, radius, edge_size, details::extract_matrix<N>(inv_matrix),
                    noa::multiply_t{}, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns or applies a cylindrical mask.
    /// \details The mask can be directly saved in \p output or applied (\p see functor) to \p input and
    ///          save in \p output. The same transformation can be applied to every batch or there can be
    ///          one transformation per batch (\p see inv_matrix). Additionally, if \p input and \p output are
    ///          not batched, multiple matrices can still be passed to generate multiple geometric shapes within
    ///          the same array. In this case, multiple "masks" are computed, reduced to a single "mask",
    ///          which is then applied to \p input and/or saved in \p output. These "masks" are sum-reduced
    ///          if \p invert is false or multiplied together if \p invert is true.
    ///
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F and FC2FC is supported.
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
    template<Remap REMAP, typename Output, typename Matrix = Empty,
             typename Input = View<const noa::traits::value_type_t<Output>>,
             typename CValue = traits::value_type_t<Output>, typename = std::enable_if_t<
                    noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
                    noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
                    noa::traits::are_almost_same_value_type_v<Input, Output, CValue> &&
                    details::is_valid_shape_v<REMAP, 3, Float33>>>
    void cylinder(const Input& input, const Output& output,
                  const Vec3<f32>& center, f32 radius, f32 length, f32 edge_size,
                  const Matrix inv_matrix = {}, CValue cvalue = CValue{1}, bool invert = false) {

        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !noa::indexing::are_overlapped(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);

        // FIXME Use std::tie because clangd? doesn't like auto[] elements captured in lambdas. Compiler is fine...
        Strides4<i64> input_strides, output_strides;
        Shape4<i64> final_shape;
        std::tie(input_strides, output_strides, final_shape) =
                noa::geometry::details::check_shape_parameters<3>(input, output, inv_matrix);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::cylinder<REMAP>(
                        input.get(), input_strides,
                        output.get(), output_strides, final_shape,
                        center, radius, length, edge_size, details::extract_matrix<3>(inv_matrix),
                        noa::multiply_t{}, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::cylinder<REMAP>(
                    input.get(), input_strides,
                    output.get(), output_strides, final_shape,
                    center, radius, length, edge_size, details::extract_matrix<3>(inv_matrix),
                    noa::multiply_t{}, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
