#pragma once

#include "noa/core/geometry/Transform.hpp"

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
            (NDIM == 2 &&
             (noa::traits::is_any_v<Matrix, Float22, Float23, Float33> ||
              noa::traits::is_varray_of_almost_any_v<Matrix, Float22, Float23>) ||
             NDIM == 3 &&
             (noa::traits::is_any_v<Matrix, Float33, Float34, Float44> ||
              noa::traits::is_varray_of_almost_any_v<Matrix, Float33, Float34>));

    template<i32 NDIM, typename Matrix>
    constexpr auto extract_linear_or_truncated_matrix(const Matrix& matrix) noexcept {
        if constexpr ((NDIM == 2 && noa::traits::is_mat33_v<Matrix>) ||
                      (NDIM == 3 && noa::traits::is_mat44_v<Matrix>)) {
            return noa::geometry::affine2truncated(matrix);
        } else if constexpr (noa::traits::is_varray_of_almost_any_v<Matrix, Float22, Float23, Float33, Float34>) {
            using const_ptr_t = const typename Matrix::mutable_value_type*;
            return const_ptr_t(matrix.get());
        } else {
            return matrix;
        }
    }

    template<i32 NDIM, typename Input, typename Output, typename Matrix>
    auto check_shape_parameters(
            const Input& input,
            const Output& output,
            const Matrix inv_matrices
    ) -> std::tuple<Strides4<i64>, Strides4<i64>, Shape4<i64>> {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        NOA_CHECK(output.shape().ndim() <= static_cast<i64>(NDIM),
                  "3D arrays are not supported with 2D geometric shapes. "
                  "Use 3D geometric shapes to support 2D and 3D arrays");
        const bool is_empty = input.is_empty();
        const bool is_output_batched = output.shape().is_batched();

        // Input is valid:
        //  - 1) both are batched -> broadcast input, and check matrix size is equal to the number of batches.
        //  - 2) both are not batched (reduce) -> broadcast input/output, matrix defines the number of batches.
        //  - 3) input is batched -> error.
        //  - 4) output is batched -> same as both are batched.
        // Otherwise:
        //  - 5) output is batched -> it defines the number of batches.
        //  - 6) output is not batched (reduce) -> matrix defines the number of batches.
        auto final_shape = output.shape();
        auto input_strides = input.strides();
        auto output_strides = output.strides();

        if (!is_empty) {
            // case 3 fails here.
            NOA_CHECK(noa::indexing::broadcast(input.shape(), input_strides, output.shape()),
                      "Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }
        if constexpr (noa::traits::is_varray_v<Matrix>) {
            if (!is_output_batched) { // case 2, 6
                final_shape[0] = inv_matrices.elements();
                input_strides[0] = 0;
                output_strides[0] = 0;
            }
        }

        const Device device = output.device();
        NOA_CHECK(is_empty || device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        if constexpr (noa::traits::is_varray_v<Matrix>) {
            NOA_CHECK(!inv_matrices.is_empty(), "Empty array detected");
            NOA_CHECK(inv_matrices.device() == device,
                      "The input and output arrays must be on the same device, but got inv_matrices:{}, output:{}",
                      inv_matrices.device(), device);
            NOA_CHECK(noa::indexing::is_contiguous_vector(inv_matrices) && inv_matrices.elements() == final_shape[0],
                      "The matrices should be specified as a contiguous vector of {} elements, "
                      "but got shape={} and strides={}",
                      final_shape[0], inv_matrices.shape(), inv_matrices.strides());
        }

        return {input_strides, output_strides, final_shape};
    }
}

namespace noa::geometry {
    /// Returns or applies an elliptical mask.
    /// \details The mask can be directly saved in \p output or applied (\p see functor) to \p input and
    ///          save in \p output. The same transformation can be applied to every batch or there can be
    ///          one transformation per batch (\p see inv_matrix). Additionally, if \p input and \p output are
    ///          not batched, multiple matrices can still be passed to generate multiple geometric shapes within
    ///          the same array. In this case, multiple "masks" are computed, reduced to a single "mask",
    ///          which is then applied to \p input and/or saved in \p output. These "masks" are sum-reduced
    ///          if \p invert is false or multiplied together if \p invert is true.
    ///
    /// \tparam Matrix      2D case: Float22, Float23, Float33 or an array/view of Float22 or Float23.
    ///                     3D case: Float33, Float34, Float44 or an array/view of Float33 or Float34.
    /// \tparam Functor     noa::multiply_t, noa::plus_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s).
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
             typename CValue = noa::traits::value_type_twice_t<Output>, size_t N,
             typename = std::enable_if_t<
                     noa::traits::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
                     noa::traits::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
                     noa::traits::are_almost_same_value_type_v<Input, Output> &&
                     details::is_valid_shape_v<N, noa::traits::value_type_t<Output>, Matrix, Functor, CValue>>>
    void ellipse(const Input& input, const Output& output,
                 const Vec<f32, N>& center, const Vec<f32, N>& radius, f32 edge_size,
                 const Matrix& inv_matrix = {}, Functor functor = {},
                 CValue cvalue = CValue{1}, bool invert = false) {

        // FIXME Use std::tie because clangd? doesn't like auto[] elements captured in lambdas. Compiler is fine...
        Strides4<i64> input_strides, output_strides;
        Shape4<i64> final_shape;
        std::tie(input_strides, output_strides, final_shape) =
                details::check_shape_parameters<N>(input, output, inv_matrix);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=](){
                cpu::geometry::ellipse(
                        input.get(), input_strides,
                        output.get(), output_strides, final_shape,
                        center, radius, edge_size,
                        details::extract_linear_or_truncated_matrix<N>(inv_matrix),
                        functor, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::ellipse(
                    input.get(), input_strides,
                    output.get(), output_strides, final_shape,
                    center, radius, edge_size,
                    details::extract_linear_or_truncated_matrix<N>(inv_matrix),
                    functor, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            if constexpr (noa::traits::is_varray_v<Matrix>)
                cuda_stream.enqueue_attach(inv_matrix.share());
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
    /// \tparam Matrix      2D case: Float22, Float23, Float33 or an array/view of Float22 or Float23.
    ///                     3D case: Float33, Float34, Float44 or an array/view of Float33 or Float34.
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
             typename CValue = noa::traits::value_type_twice_t<Output>, size_t N,
             typename = std::enable_if_t<
                     noa::traits::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
                     noa::traits::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
                     noa::traits::are_almost_same_value_type_v<Input, Output> &&
                     details::is_valid_shape_v<N, noa::traits::value_type_t<Output>, Matrix, Functor, CValue>>>
    void sphere(const Input& input, const Output& output,
                const Vec<f32, N>& center, f32 radius, f32 edge_size,
                const Matrix& inv_matrix = {}, Functor functor = {},
                CValue cvalue = CValue{1}, bool invert = false) {

        // FIXME Use std::tie because clangd? doesn't like auto[] elements captured in lambdas. Compiler is fine...
        Strides4<i64> input_strides, output_strides;
        Shape4<i64> final_shape;
        std::tie(input_strides, output_strides, final_shape) =
                details::check_shape_parameters<N>(input, output, inv_matrix);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=](){
                cpu::geometry::sphere(
                        input.get(), input_strides,
                        output.get(), output_strides, final_shape,
                        center, radius, edge_size,
                        details::extract_linear_or_truncated_matrix<N>(inv_matrix),
                        functor, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::sphere(
                    input.get(), input_strides,
                    output.get(), output_strides, final_shape,
                    center, radius, edge_size,
                    details::extract_linear_or_truncated_matrix<N>(inv_matrix),
                    functor, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            if constexpr (noa::traits::is_varray_v<Matrix>)
                cuda_stream.enqueue_attach(inv_matrix.share());
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
    /// \tparam Matrix      2D case: Float22, Float23, Float33 or an array/view of Float22 or Float23.
    ///                     3D case: Float33, Float34, Float44 or an array/view of Float33 or Float34.
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
             typename CValue = noa::traits::value_type_twice_t<Output>, size_t N,
             typename = std::enable_if_t<
                     noa::traits::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
                     noa::traits::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
                     noa::traits::are_almost_same_value_type_v<Input, Output> &&
                     details::is_valid_shape_v<N, noa::traits::value_type_t<Output>, Matrix, Functor, CValue>>>
    void rectangle(const Input& input, const Output& output,
                   const Vec<f32, N>& center, const Vec<f32, N>& radius, f32 edge_size,
                   const Matrix& inv_matrix = {}, Functor functor = {},
                   CValue cvalue = CValue{1}, bool invert = false) {

        // FIXME Use std::tie because clangd? doesn't like auto[] elements captured in lambdas. Compiler is fine...
        Strides4<i64> input_strides, output_strides;
        Shape4<i64> final_shape;
        std::tie(input_strides, output_strides, final_shape) =
                details::check_shape_parameters<N>(input, output, inv_matrix);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=](){
                cpu::geometry::rectangle(
                        input.get(), input_strides,
                        output.get(), output_strides, final_shape,
                        center, radius, edge_size,
                        details::extract_linear_or_truncated_matrix<N>(inv_matrix),
                        functor, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::rectangle(
                    input.get(), input_strides,
                    output.get(), output_strides, final_shape,
                    center, radius, edge_size,
                    details::extract_linear_or_truncated_matrix<N>(inv_matrix),
                    functor, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            if constexpr (noa::traits::is_varray_v<Matrix>)
                cuda_stream.enqueue_attach(inv_matrix.share());
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
    /// \tparam Matrix      Float33, Float34, Float44 or an array/view of Float33 or Float34.
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
             typename CValue = noa::traits::value_type_twice_t<Output>,
             typename = std::enable_if_t<
                     noa::traits::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
                     noa::traits::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
                     noa::traits::are_almost_same_value_type_v<Input, Output> &&
                     details::is_valid_shape_v<3, noa::traits::value_type_t<Output>, Matrix, Functor, CValue>>>
    void cylinder(const Input& input, const Output& output,
                  const Vec3<f32>& center, f32 radius, f32 length, f32 edge_size,
                  const Matrix& inv_matrix = {}, Functor functor = {},
                  CValue cvalue = CValue{1}, bool invert = false) {

        // FIXME Use std::tie because clangd? doesn't like auto[] elements captured in lambdas. Compiler is fine...
        Strides4<i64> input_strides, output_strides;
        Shape4<i64> final_shape;
        std::tie(input_strides, output_strides, final_shape) =
                details::check_shape_parameters<3>(input, output, inv_matrix);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=](){
                cpu::geometry::cylinder(
                        input.get(), input_strides,
                        output.get(), output_strides, final_shape,
                        center, radius, length, edge_size,
                        details::extract_linear_or_truncated_matrix<3>(inv_matrix),
                        functor, cvalue, invert, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::cylinder(
                    input.get(), input_strides,
                    output.get(), output_strides, final_shape,
                    center, radius, length, edge_size,
                    details::extract_linear_or_truncated_matrix<3>(inv_matrix),
                    functor, cvalue, invert, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            if constexpr (noa::traits::is_varray_v<Matrix>)
                cuda_stream.enqueue_attach(inv_matrix.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
