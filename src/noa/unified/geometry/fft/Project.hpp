#pragma once

#include "noa/core/geometry/Euler.hpp"
#include "noa/core/geometry/Transform.hpp"

#include "noa/cpu/geometry/fft/Project.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Project.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/Texture.hpp"

namespace noa::geometry::fft::details {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_rasterize_v =
            nt::is_any_v<Value, f32, f64, c32, c64> &&
            (nt::is_any_v<Scale, Float22> || nt::is_varray_of_almost_any_v<Scale, Float22>) &&
            (nt::is_any_v<Rotate, Float33> || nt::is_varray_of_almost_any_v<Rotate, Float33>) &&
            (REMAP == Remap::H2H || REMAP == Remap::H2HC || REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_interpolate_v =
            nt::is_any_v<Value, f32, f64, c32, c64> &&
            (nt::is_any_v<Scale, Float22> || nt::is_varray_of_almost_any_v<Scale, Float22>) &&
            (nt::is_any_v<Rotate, Float33> || nt::is_varray_of_almost_any_v<Rotate, Float33>) &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_extract_v =
            nt::is_any_v<Value, f32, f64, c32, c64> &&
            (nt::is_any_v<Scale, Float22> || nt::is_varray_of_almost_any_v<Scale, Float22>) &&
            (nt::is_any_v<Rotate, Float33> || nt::is_varray_of_almost_any_v<Rotate, Float33>) &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale0, typename Rotate0, typename Scale1, typename Rotate1>
    constexpr bool is_valid_insert_insert_extract_v =
            nt::is_any_v<Value, f32, f64, c32, c64> &&
            (nt::is_any_v<Scale0, Float22> || nt::is_varray_of_almost_any_v<Scale0, Float22>) &&
            (nt::is_any_v<Rotate0, Float33> || nt::is_varray_of_almost_any_v<Rotate0, Float33>) &&
            (nt::is_any_v<Scale1, Float22> || nt::is_varray_of_almost_any_v<Scale1, Float22>) &&
            (nt::is_any_v<Rotate1, Float33> || nt::is_varray_of_almost_any_v<Rotate1, Float33>) &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<bool OPTIONAL, typename Matrix>
    void project_check_matrix(const Matrix& matrix, i64 required_size, Device compute_device) {
        if constexpr (OPTIONAL) {
            if (matrix.is_empty())
                return;
        } else {
            NOA_CHECK(!matrix.is_empty(), "The matrices should not be empty");
        }

        NOA_CHECK(noa::indexing::is_contiguous_vector(matrix) && matrix.elements() == required_size,
                  "The number of matrices, specified as a contiguous vector, should be equal to the number of slices, "
                  "but got matrix shape:{}, strides:{} and {} slices",
                  matrix.shape(), matrix.strides(), required_size);

        NOA_CHECK(matrix.device() == compute_device,
                  "The transformation parameters should be on the compute device");
    }

    enum class ProjectionType { INSERT_RASTERIZE, INSERT_INTERPOLATE, EXTRACT, INSERT_EXTRACT };

    template<ProjectionType DIRECTION, typename Input, typename Output,
             typename Scale0, typename Rotate0,
             typename Scale1 = Float22, typename Rotate1 = Float33>
    void projection_check_parameters(const Input& input, const Shape4<i64>& input_shape,
                                     const Output& output, const Shape4<i64>& output_shape,
                                     const Shape4<i64>& target_shape,
                                     const Scale0& input_scaling_matrix,
                                     const Rotate0& input_rotation_matrix,
                                     const Scale1& output_scaling_matrix = {},
                                     const Rotate1& output_rotation_matrix = {}) {
        const Device output_device = output.device();
        if constexpr (!nt::is_numeric_v<Input>) {
            const Device input_device = input.device();

            NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
            if constexpr (nt::is_varray_v<Input>)
                NOA_CHECK(!noa::indexing::are_overlapped(input, output), "Input and output arrays should not overlap");

            NOA_CHECK(input_device == output_device,
                      "The input and output should be on the same device but got input:{} and output:{}",
                      input_device, output_device);

            NOA_CHECK(noa::all(input.shape() == input_shape.rfft()),
                      "The shape of the non-redundant input does not match the expected shape. "
                      "Got {} and expected {}", input.shape(), input_shape.rfft());
        }

        NOA_CHECK(noa::all(output.shape() == output_shape.rfft()),
                  "The shape of the non-redundant output does not match the expected shape. Got {} and expected {}",
                  output.shape(), output_shape.rfft());

        if constexpr (DIRECTION == ProjectionType::INSERT_RASTERIZE ||
                      DIRECTION == ProjectionType::INSERT_INTERPOLATE) {
            NOA_CHECK(input_shape[1] == 1,
                      "2D input slices are expected but got shape {}", input_shape);
            if (noa::any(target_shape == 0)) {
                NOA_CHECK(output_shape[0] == 1 && !output_shape.is_batched(),
                          "A single 3D output is expected but got shape {}", output_shape);
            } else {
                NOA_CHECK(output_shape[0] == 1 && target_shape[0] == 1 && !target_shape.is_batched(),
                          "A single grid is expected, with a target shape describing a single 3D volume, "
                          "but got output shape {} and target shape {}", output_shape, target_shape);
            }
        } else if constexpr (DIRECTION == ProjectionType::EXTRACT) {
            NOA_CHECK(output_shape[1] == 1,
                      "2D input slices are expected but got shape {}", output_shape);
            if (noa::any(target_shape == 0)) {
                NOA_CHECK(input_shape[0] == 1 && !input_shape.is_batched(),
                          "A single 3D input is expected but got shape {}", input_shape);
            } else {
                NOA_CHECK(input_shape[0] == 1 && target_shape[0] == 1 && !target_shape.is_batched(),
                          "A single grid is expected, with a target shape describing a single 3D volume, "
                          "but got input shape {} and target shape {}", input_shape, target_shape);
            }
        } else { // INSERT_EXTRACT
            NOA_CHECK(input_shape[1] == 1 && output_shape[1] == 1,
                      "2D slices are expected but got shape input:{} and output:{}",
                      input_shape, output_shape);
        }

        const auto required_matrix_count = DIRECTION == ProjectionType::EXTRACT ? output_shape[0] : input_shape[0];
        if constexpr (!nt::is_mat22_v<Scale0>)
            project_check_matrix<true>(input_scaling_matrix, required_matrix_count, output_device);
        if constexpr (!nt::is_mat33_v<Rotate0>)
            project_check_matrix<false>(input_rotation_matrix, required_matrix_count, output_device);

        // Only for INSERT_EXTRACT.
        if constexpr (!nt::is_mat22_v<Scale1>)
            project_check_matrix<true>(output_scaling_matrix, output_shape[0], output_device);
        if constexpr (!nt::is_mat33_v<Rotate1>)
            project_check_matrix<false>(output_rotation_matrix, output_shape[0], output_device);
    }

    template<typename Matrix>
    auto extract_matrix(const Matrix& matrix) {
        if constexpr (traits::is_matXX_v<Matrix>) {
            return matrix;
        } else {
            using ptr_t = const typename Matrix::value_type*;
            return ptr_t(matrix.get());
        }
    }
}

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Inserts 2D Fourier central slice(s) into a 3D Fourier volume, using tri-linear rasterization.
    /// \details The slices are scaled and the EWS curvature is applied. Then, they are rotated and added to the
    ///          3D cartesian Fourier volume using tri-linear rasterization. This method, often referred to as
    ///          direct Fourier insertion, explicitly sets the "thickness" of the central slices as the width of
    ///          the rasterization window (referred to as gridding kernel), which in this case is 1 voxel.
    ///          In practice, a density correction (i.e. normalization) is often required after this operation.
    ///          This can easily be achieved by inserting the per-slice weights into another volume to keep track
    ///          of what was inserted and where. Gridding correction can also be beneficial as post-processing
    ///          one the real-space output (see gridding_correction() below).
    ///
    /// \tparam REMAP                   Remapping from the slice to the grid layout.
    ///                                 Should be H2H, H2HC, HC2H or HC2HC.
    /// \tparam Scale                   Float22 or an array/view of this type.
    /// \tparam Rotate                  Float33 or an array/view of this type.
    /// \param[in] slice                Non-redundant 2D slice(s) to insert.
    /// \param slice_shape              BDHW logical shape of \p slice.
    /// \param[out] grid                Non-redundant 3D grid inside which the slices are inserted.
    /// \param grid_shape               BDHW logical shape of \p grid.
    /// \param[in] inv_scaling_matrix   2x2 HW \e inverse real-space scaling matrix to apply to the slices
    ///                                 before the rotation. If an array is passed, it can be empty or have
    ///                                 one matrix per slice. Otherwise the same scaling matrix is applied
    ///                                 to every slice.
    /// \param[in] fwd_rotation_matrix  3x3 DHW \e forward rotation matrices to apply to the slices.
    ///                                 If an array is passed, it should have one matrix per slice.
    ///                                 Otherwise the same rotation matrix is applied to every slice.
    /// \param cutoff                   Frequency cutoff in \p grid, in cycle/pix.
    /// \param target_shape             Actual BDHW logical shape of the 3D volume.
    /// \param ews_radius               HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                 If negative, the negative curve is computed.
    ///                                 If {0,0}, the slices are projections.
    ///
    /// \note This function normalizes the slice and grid dimensions, and works with normalized frequencies,
    ///       from -0.5 to 0.5 cycle/pix. By default (empty \p target_shape or \p target_shape == \p grid_shape),
    ///       the slice frequencies are mapped into the grid frequencies. If the grid is larger than the slices,
    ///       the slices are implicitly stretched (over-sampling case). If the grid is smaller than the slices,
    ///       the slices are shrank (under-sampling case).
    ///       However, if \p target_shape is specified, the slice frequencies are instead mapped into the frequencies
    ///       of a 3D FFT volume of shape \p target_shape. In this case, \p grid is just the region to "render" within
    ///       the volume defined by \p target_shape, which can be of any shape, e.g. a subregion of \p target_shape.
    /// \note In order to have both left and right beams assigned to different values, this function only computes one
    ///       "side" of the EWS, as specified by \p ews_radius. To insert the other side, one would have to
    ///       call this function a second time with \p ews_radius * -1.
    /// \note The scaling and the rotation matrices are kept separated from one another in order to properly compute the
    ///       curve of the Ewald sphere. Indeed, the scaling is applied first to correct for magnification, so that the
    ///       EWS is computed using the original frequencies (from the scattering) and is therefore spherical even
    ///       under anisotropic magnification. If \p ews_radius is 0, the scaling factors can be merged to the
    ///       rotations.
    /// \note The redundant line at x=0 is entirely inserted into the volume. If the projection has an in-plane
    ///       rotation, this results into having this line inserted twice. This emphasizes the need of normalizing
    ///       the output grid, or extracted slice(s), with the corresponding inserted weights, or extracted weights.
    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate, typename = std::enable_if_t<
             nt::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
             nt::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
             nt::are_almost_same_value_type_v<Input, Output> &&
             details::is_valid_insert_rasterize_v<REMAP, nt::value_type_t<Output>, Scale, Rotate>>>
    void insert_rasterize_3d(const Input& slice, const Shape4<i64> slice_shape,
                             const Output& grid, const Shape4<i64> grid_shape,
                             const Scale& inv_scaling_matrix,
                             const Rotate& fwd_rotation_matrix,
                             f32 cutoff = 0.5f,
                             const Shape4<i64> target_shape = {},
                             const Vec2<f32>& ews_radius = {}) {
        details::projection_check_parameters<details::ProjectionType::INSERT_RASTERIZE>(
                slice, slice_shape, grid, grid_shape, target_shape,
                inv_scaling_matrix, fwd_rotation_matrix);

        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=](){
                cpu::geometry::fft::insert_rasterize_3d<REMAP>(
                        slice.get(), slice.strides(), slice_shape,
                        grid.get(), grid.strides(), grid_shape,
                        details::extract_matrix(inv_scaling_matrix),
                        details::extract_matrix(fwd_rotation_matrix),
                        cutoff, target_shape, ews_radius, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::insert_rasterize_3d<REMAP>(
                    slice.get(), slice.strides(), slice_shape,
                    grid.get(), grid.strides(), grid_shape,
                    details::extract_matrix(inv_scaling_matrix),
                    details::extract_matrix(fwd_rotation_matrix),
                    cutoff, target_shape, ews_radius, cuda_stream);
            cuda_stream.enqueue_attach(
                    slice, grid, inv_scaling_matrix, fwd_rotation_matrix);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Inserts 2D Fourier central slice(s) into a 3D Fourier volume, using tri-linear rasterization.
    /// \details This function has the same features and limitations as the overload taking arrays,
    ///          but the slice is represented by a single constant value. This can be useful
    ///          to keep track of the multiplicity of the Fourier insertion.
    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate, typename = std::enable_if_t<
             nt::is_varray_of_any_v<Output, Input> &&
             details::is_valid_insert_rasterize_v<REMAP, Input, Scale, Rotate>>>
    void insert_rasterize_3d(Input slice, const Shape4<i64>& slice_shape,
                             const Output& grid, const Shape4<i64>& grid_shape,
                             const Scale& inv_scaling_matrix,
                             const Rotate& fwd_rotation_matrix,
                             f32 cutoff = 0.5f,
                             const Shape4<i64>& target_shape = {},
                             const Vec2<f32>& ews_radius = {}) {
        details::projection_check_parameters<details::ProjectionType::INSERT_RASTERIZE>(
                slice, slice_shape, grid, grid_shape, target_shape,
                inv_scaling_matrix, fwd_rotation_matrix);

        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=](){
                cpu::geometry::fft::insert_rasterize_3d<REMAP>(
                        slice, slice_shape,
                        grid.get(), grid.strides(), grid_shape,
                        details::extract_matrix(inv_scaling_matrix),
                        details::extract_matrix(fwd_rotation_matrix),
                        cutoff, target_shape, ews_radius, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::insert_rasterize_3d<REMAP>(
                    slice, slice_shape,
                    grid.get(), grid.strides(), grid_shape,
                    details::extract_matrix(inv_scaling_matrix),
                    details::extract_matrix(fwd_rotation_matrix),
                    cutoff, target_shape, ews_radius, cuda_stream);
            cuda_stream.enqueue_attach(grid, inv_scaling_matrix, fwd_rotation_matrix);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Settings for the central slice thickness, modelled as a windowed-sinc.\n
    /// \b Fourier-insertion: Central slices are inserted in a (virtual) volume. This parameter defines
    /// thickness of the slice(s), or in other words, the windowed-sinc that is convolved along the
    /// normal of the perfectly thin slice(s) to insert.\n
    /// \b Fourier-extraction: Central slices are extracted from a (virtual) volume. This parameter defines the
    /// thickness of the reconstructed object along z, or in other words, the windowed-sinc that is convolved,
    /// along the z of the reconstruction, with the perfectly thin slice(s) to extract. This is used to
    /// effectively apply an horizontal rectangular mask centered on the object _before_ the forward projection.
    /// The current API doesn't allow to change the orientation of this sinc (it is always along z) since its
    /// only purpose was originally to improve tomogram's projections with horizontal samples by removing
    /// the noise from above and below the sample.
    struct CentralSliceSincWindow {
        /// Frequency, in cycle/pix, of the first zero of the sinc.
        /// This is clamped to ensure a minimum of 1 pixel diameter,
        /// which is usually want we want for Fourier insertion.
        f64 fftfreq_sinc_cutoff{-1};

        /// Frequency, in cycle/pix, where the blackman window stops.
        /// This parameter is here to control the accuracy/performance ratio.
        /// Usually this value should be a multiple of the sinc-cutoff. The larger this multiple,
        /// the sharper the real-space window, but the slower it is to compute the slice.
        /// This is clamped to ensure the window stops at least to the first sinc-cutoff.
        /// So if both frequencies are left to their default value (-1), a 1 pixel thick slice
        /// is generated, which is usually want we want for Fourier insertion.
        f64 fftfreq_blackman_cutoff{-1};
    };

    /// Inserts 2D Fourier central slice(s) into a 3D Fourier volume, using bi-linear interpolation and sinc-weighting.
    /// \details This function computes the inverse transformation compared to the overload above using rasterization,
    ///          effectively transforming the 3D grid onto the input slice(s). Briefly, for each input slice, each
    ///          voxel is assigned to a transformed frequency (w,v,u) corresponding to the reference frame of the
    ///          current slice to insert. 1) Given the frequency w, which is the distance of the voxel along the
    ///          normal of the slice, and \p slice_z_radius, it computes a sinc-weight from 1 (on the slice) to 0
    ///          (outside the slice). 2) Then, if the slice does contribute to the voxel, i.e. the sinc-weight is
    ///          non-zero, a bi-linear interpolation is done using the (v,u) frequency component of the voxel.
    ///          The interpolated value is then sinc-weighted and added to the voxel.
    ///
    /// \tparam REMAP                   Remapping from the slice to the grid layout.
    ///                                 Should be HC2H or HC2HC.
    /// \tparam Scale                   Float22 or an array/view of this type.
    /// \tparam Rotate                  Float33 or an array/view of this type.
    /// \param[in] slice                Non-redundant 2D slice(s) to insert.
    /// \param slice_shape              BDHW logical shape of \p slice.
    /// \param[out] grid                Non-redundant 3D grid inside which the slices are inserted.
    /// \param grid_shape               BDHW logical shape of \p grid.
    /// \param[in] fwd_scaling_matrix   2x2 HW \e forward real-space scaling matrix to apply to the slices
    ///                                 before the rotation. If an array is passed, it can be empty or have
    ///                                 one matrix per slice. Otherwise the same scaling matrix is applied
    ///                                 to every slice.
    /// \param[in] inv_rotation_matrix  3x3 DHW \e inverse rotation matrices to apply to the slices.
    ///                                 If an array is passed, it should have one matrix per slice.
    ///                                 Otherwise the same rotation matrix is applied to every slice.
    /// \param slice_z_radius           Radius along the normal of the central slices, in cycle/pix.
    ///                                 This is clamped to ensure a minimum of 1 pixel diameter.
    /// \param cutoff                   Frequency cutoff in \p grid, in cycle/pix.
    /// \param target_shape             Actual BDHW logical shape of the 3D volume.
    /// \param ews_radius               HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                 If negative, the negative curve is computed.
    ///                                 If {0,0}, the slices are projections.
    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate, typename = std::enable_if_t<
             nt::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
             nt::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
             nt::are_almost_same_value_type_v<Input, Output> &&
             details::is_valid_insert_interpolate_v<REMAP, nt::value_type_t<Output>, Scale, Rotate>>>
    void insert_interpolate_3d(const Input& slice, const Shape4<i64>& slice_shape,
                               const Output& grid, const Shape4<i64>& grid_shape,
                               const Scale& fwd_scaling_matrix,
                               const Rotate& inv_rotation_matrix,
                               f32 slice_z_radius = 0.f,
                               f32 cutoff = 0.5f,
                               const Shape4<i64>& target_shape = {},
                               const Vec2<f32>& ews_radius = {}) {
        details::projection_check_parameters<details::ProjectionType::INSERT_INTERPOLATE>(
                slice, slice_shape, grid, grid_shape, target_shape,
                fwd_scaling_matrix, inv_rotation_matrix);

        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::insert_interpolate_3d<REMAP>(
                        slice.get(), slice.strides(), slice_shape,
                        grid.get(), grid.strides(), grid_shape,
                        details::extract_matrix(fwd_scaling_matrix),
                        details::extract_matrix(inv_rotation_matrix),
                        cutoff, target_shape, ews_radius,
                        slice_z_radius, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::insert_interpolate_3d<REMAP>(
                    slice.get(), slice.strides(), slice_shape,
                    grid.get(), grid.strides(), grid_shape,
                    details::extract_matrix(fwd_scaling_matrix),
                    details::extract_matrix(inv_rotation_matrix),
                    cutoff, target_shape, ews_radius,
                    slice_z_radius, cuda_stream);
            cuda_stream.enqueue_attach(
                    slice, grid, fwd_scaling_matrix, inv_rotation_matrix);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Inserts 2D Fourier central slice(s) into a 3D Fourier volume, using bi-linear interpolation and sinc-weighting.
    /// \details This function has the same features and limitations as the overload taking arrays, but uses textures.
    template<Remap REMAP, typename Value, typename Output, typename Scale, typename Rotate, typename = std::enable_if_t<
             nt::is_varray_of_any_v<Output, Value> &&
             details::is_valid_insert_interpolate_v<REMAP, Value, Scale, Rotate>>>
    void insert_interpolate_3d(const Texture<Value>& slice, const Shape4<i64>& slice_shape,
                               const Output& grid, const Shape4<i64>& grid_shape,
                               const Scale& fwd_scaling_matrix,
                               const Rotate& inv_rotation_matrix,
                               f32 slice_z_radius = 0.f,
                               f32 cutoff = 0.5f,
                               const Shape4<i64>& target_shape = {},
                               const Vec2<f32>& ews_radius = {}) {
        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            const cpu::Texture<Value>& texture = slice.cpu();
            const Array<Value> slice_array(texture.ptr, slice.shape(), texture.strides, slice.options());
            insert_interpolate_3d<REMAP>(slice_array, slice_shape, grid, grid_shape,
                                         fwd_scaling_matrix, inv_rotation_matrix,
                                         slice_z_radius, cutoff, target_shape, ews_radius);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (nt::is_any_v<Value, f64, c64>) {
                NOA_THROW("Double-precision floating-points are not supported by CUDA textures");
            } else {
                details::projection_check_parameters<details::ProjectionType::INSERT_INTERPOLATE>(
                        slice, slice_shape, grid, grid_shape, target_shape,
                        fwd_scaling_matrix, inv_rotation_matrix);

                const cuda::Texture<Value>& texture = slice.cuda();
                auto& cuda_stream = stream.cuda();
                cuda::geometry::fft::insert_interpolate_3d<REMAP>(
                        texture.array.get(), *texture.texture, slice.interp_mode(), slice_shape,
                        grid.get(), grid.strides(), grid_shape,
                        details::extract_matrix(fwd_scaling_matrix),
                        details::extract_matrix(inv_rotation_matrix),
                        cutoff, target_shape, ews_radius,
                        slice_z_radius, cuda_stream);
                cuda_stream.enqueue_attach(
                        texture.array, texture.texture, grid,
                        fwd_scaling_matrix, inv_rotation_matrix);
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// \details This function has the same features and limitations as the overload taking arrays,
    ///          but the slice is represented by a single constant value. This is for example useful
    ///          to keep track of the multiplicity of the Fourier insertion.
    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate, typename = std::enable_if_t<
             nt::is_varray_of_any_v<Output, Input> &&
             details::is_valid_insert_interpolate_v<REMAP, Input, Scale, Rotate>>>
    void insert_interpolate_3d(Input slice, const Shape4<i64>& slice_shape,
                               const Output& grid, const Shape4<i64>& grid_shape,
                               const Scale& fwd_scaling_matrix,
                               const Rotate& inv_rotation_matrix,
                               f32 slice_z_radius = 0.f,
                               f32 cutoff = 0.5f,
                               const Shape4<i64>& target_shape = {},
                               const Vec2<f32>& ews_radius = {}) {
        details::projection_check_parameters<details::ProjectionType::INSERT_INTERPOLATE>(
                slice, slice_shape, grid, grid_shape, target_shape,
                fwd_scaling_matrix, inv_rotation_matrix);

        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::insert_interpolate_3d<REMAP>(
                        slice, slice_shape,
                        grid.get(), grid.strides(), grid_shape,
                        details::extract_matrix(fwd_scaling_matrix),
                        details::extract_matrix(inv_rotation_matrix),
                        cutoff, target_shape, ews_radius,
                        slice_z_radius, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::insert_interpolate_3d<REMAP>(
                    slice, slice_shape,
                    grid.get(), grid.strides(), grid_shape,
                    details::extract_matrix(fwd_scaling_matrix),
                    details::extract_matrix(inv_rotation_matrix),
                    cutoff, target_shape, ews_radius,
                    slice_z_radius, cuda_stream);
            cuda_stream.enqueue_attach(grid, fwd_scaling_matrix, inv_rotation_matrix);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    /// \details This is the reverse operation of insert3D. The transformation itself is identical to the
    ///          transformation of insert3D using rasterization, so the same parameters can be used here.
    ///
    /// \tparam REMAP                   Remapping from the slice to the grid layout. Should be HC2H or HC2HC.
    /// \tparam Scale                   Float22 or an array/view of this type.
    /// \tparam Rotate                  Float33 or an array/view of this type.
    /// \param[out] grid                Non-redundant centered 3D grid from which to extract the slices.
    /// \param grid_shape               BDHW logical shape of \p grid.
    /// \param[in] slice                Non-redundant 2D extracted slice(s).
    /// \param slice_shape              BDHW logical shape of \p slice.
    /// \param[in] inv_scaling_matrix   2x2 HW \e inverse real-space scaling to apply to the slices before the rotation.
    ///                                 If an array is passed, it can be empty or have one matrix per slice.
    ///                                 Otherwise the same scaling matrix is applied to every slice.
    /// \param[in] fwd_rotation_matrix  3x3 DHW \e forward rotation matrices to apply to the slices.
    ///                                 If an array is passed, it should have one matrix per slice.
    ///                                 Otherwise the same rotation matrix is applied to every slice.
    /// \param cutoff                   Frequency cutoff in \p grid, in cycle/pix.
    /// \param target_shape             Actual BDHW logical shape of the 3D volume.
    /// \param ews_radius               HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                 If negative, the negative curve is computed.
    ///                                 If {0,0}, the slices are projections.
    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate, typename = std::enable_if_t<
             nt::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
             nt::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
             nt::are_almost_same_value_type_v<Input, Output> &&
             details::is_valid_extract_v<REMAP, nt::value_type_t<Output>, Scale, Rotate>>>
    void extract_3d(const Input& grid, const Shape4<i64>& grid_shape,
                    const Output& slice, const Shape4<i64>& slice_shape,
                    const Scale& inv_scaling_matrix,
                    const Rotate& fwd_rotation_matrix,
                    f32 cutoff = 0.5f,
                    const Shape4<i64>& target_shape = {},
                    const Vec2<f32>& ews_radius = {}) {

        details::projection_check_parameters<details::ProjectionType::EXTRACT>(
                grid, grid_shape, slice, slice_shape, target_shape,
                inv_scaling_matrix, fwd_rotation_matrix);

        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::extract_3d<REMAP>(
                        grid.get(), grid.strides(), grid_shape,
                        slice.get(), slice.strides(), slice_shape,
                        details::extract_matrix(inv_scaling_matrix),
                        details::extract_matrix(fwd_rotation_matrix),
                        cutoff, target_shape, ews_radius, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::extract_3d<REMAP>(
                    grid.get(), grid.strides(), grid_shape,
                    slice.get(), slice.strides(), slice_shape,
                    details::extract_matrix(inv_scaling_matrix),
                    details::extract_matrix(fwd_rotation_matrix),
                    cutoff, target_shape, ews_radius, cuda_stream);
            cuda_stream.enqueue_attach(
                    grid, slice, inv_scaling_matrix, fwd_rotation_matrix);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    /// \details This function has the same features and limitations as the overload taking arrays, but uses textures.
    template<Remap REMAP, typename Value, typename Output, typename Scale, typename Rotate, typename = std::enable_if_t<
             nt::is_varray_of_any_v<Output, Value> &&
             details::is_valid_insert_interpolate_v<REMAP, Value, Scale, Rotate>>>
    void extract_3d(const Texture<Value>& grid, const Shape4<i64>& grid_shape,
                    const Output& slice, const Shape4<i64>& slice_shape,
                    const Scale& inv_scaling_matrix,
                    const Rotate& fwd_rotation_matrix,
                    f32 cutoff = 0.5f,
                    const Shape4<i64>& target_shape = {},
                    const Vec2<f32>& ews_radius = {}) {
        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            const cpu::Texture<Value>& texture = grid.cpu();
            const Array<Value> grid_array(texture.ptr, grid.shape(), texture.strides, grid.options());
            extract_3d<REMAP>(grid_array, grid_shape, slice, slice_shape,
                              inv_scaling_matrix, fwd_rotation_matrix,
                              cutoff, target_shape, ews_radius);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (nt::is_any_v<Value, f64, c64>) {
                NOA_THROW("Double-precision floating-points are not supported by CUDA textures");
            } else {
                details::projection_check_parameters<details::ProjectionType::EXTRACT>(
                        grid, grid_shape, slice, slice_shape, target_shape,
                        inv_scaling_matrix, fwd_rotation_matrix);

                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = grid.cuda();
                cuda::geometry::fft::extract_3d<REMAP>(
                        texture.array.get(), *texture.texture, grid.interp_mode(), grid_shape,
                        slice.get(), slice.strides(), slice_shape,
                        details::extract_matrix(inv_scaling_matrix),
                        details::extract_matrix(fwd_rotation_matrix),
                        cutoff, target_shape, ews_radius, cuda_stream);
                cuda_stream.enqueue_attach(
                        texture.array, texture.texture, slice,
                        inv_scaling_matrix, fwd_rotation_matrix);
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Extracts 2D Fourier slice(s) from a virtual volume filled by other slices, using linear interpolation.
    /// \details This function effectively combines the insertion by interpolation and the extraction, but only
    ///          renders the frequencies that are going to be used for the extraction. This function is useful if
    ///          the 3D Fourier volume, where the slices are inserted, is used for extracting slice(s) immediately
    ///          after the insertion. It is much faster than calling insert_interpolate_3d and extract_3d,
    ///          uses less memory (the 3D Fourier volume is entirely skipped), and skips a layer of interpolation.
    ///
    /// \tparam REMAP                           Remapping from the slice to the grid layout. Should be HC2H or HC2HC.
    /// \tparam InputScale                      Float22 or an array/view of this type.
    /// \tparam InputRotate                     Float33 or an array/view of this type.
    /// \tparam OutputScale                     Float22 or an array/view of this type.
    /// \tparam OutputRotate                    Float33 or an array/view of this type.
    /// \param[in] input_slice                  Non-redundant 2D slice(s) to insert.
    /// \param input_slice_shape                BDHW logical shape of \p input_slice.
    /// \param[in,out] output_slice             Non-redundant 2D extracted slice(s). See \p add_to_output.
    /// \param output_slice_shape               BDHW logical shape of \p output_slice.
    /// \param[in] input_fwd_scaling_matrix     2x2 HW \e forward real-space scaling matrix to apply to the input
    ///                                         slices before the rotation. If an array is passed, it can be empty
    ///                                         or have one matrix per slice. Otherwise the same scaling matrix
    ///                                         is applied to every slice.
    /// \param[in] input_inv_rotation_matrix    3x3 DHW \e inverse rotation matrices to apply to the input slices.
    ///                                         If an array is passed, it should have one matrix per slice.
    ///                                         Otherwise the same rotation matrix is applied to every slice.
    /// \param[in] output_inv_scaling_matrix    2x2 HW \e inverse real-space scaling matrix to apply to the output
    ///                                         slices before the rotation. If an array is passed, it can be empty
    ///                                         or have one matrix per slice. Otherwise the same scaling matrix
    ///                                         is applied to every slice.
    /// \param[in] output_fwd_rotation_matrix   3x3 DHW \e forward rotation matrices to apply to the output slices.
    ///                                         If an array is passed, it should have one matrix per slice.
    ///                                         Otherwise the same rotation matrix is applied to every slice.
    /// \param slice_z_radius                   Radius along the normal of the central slices, in cycle/pix.
    ///                                         This is clamped to ensure a minimum of 1 pixel diameter.
    /// \param add_to_output                    Whether the contribution of the input slices should be added to the
    ///                                         output. By default, the function sets \p output_slice. With this option
    ///                                         enabled, it instead adds the contribution of \p input_slice to the
    ///                                         signal already in \p output_slice, allowing to reuse and progressively
    ///                                         build the output signal.
    /// \param cutoff                           Frequency cutoff of the virtual 3D Fourier volume, in cycle/pix.
    /// \param ews_radius                       HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                         If negative, the negative curve is computed.
    ///                                         If {0,0}, the slices are projections.
    template<Remap REMAP, typename Input, typename Output, typename InputScale, typename InputRotate,
             typename OutputScale, typename OutputRotate, typename = std::enable_if_t<
                     nt::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
                     nt::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
                     nt::are_almost_same_value_type_v<Input, Output> &&
                     details::is_valid_insert_insert_extract_v<
                     REMAP, nt::value_type_t<Output>, InputScale, InputRotate, OutputScale, OutputRotate>>>
    void insert_interpolate_and_extract_3d(
            const Input& input_slice, const Shape4<i64>& input_slice_shape,
            const Output& output_slice, const Shape4<i64>& output_slice_shape,
            const InputScale& input_fwd_scaling_matrix, const InputRotate& input_inv_rotation_matrix,
            const OutputScale& output_inv_scaling_matrix, const OutputRotate& output_fwd_rotation_matrix,
            f32 slice_z_radius = 0.f, bool add_to_output = false,
            f32 cutoff = 0.5f, const Vec2<f32>& ews_radius = {}) {

        details::projection_check_parameters<details::ProjectionType::INSERT_EXTRACT>(
                input_slice, input_slice_shape, output_slice, output_slice_shape, {},
                input_fwd_scaling_matrix, input_inv_rotation_matrix,
                output_inv_scaling_matrix, output_fwd_rotation_matrix);

        const Device device = output_slice.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::insert_interpolate_and_extract_3d<REMAP>(
                        input_slice.get(), input_slice.strides(), input_slice_shape,
                        output_slice.get(), output_slice.strides(), output_slice_shape,
                        details::extract_matrix(input_fwd_scaling_matrix),
                        details::extract_matrix(input_inv_rotation_matrix),
                        details::extract_matrix(output_inv_scaling_matrix),
                        details::extract_matrix(output_fwd_rotation_matrix),
                        cutoff, ews_radius, slice_z_radius, add_to_output, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::insert_interpolate_and_extract_3d<REMAP>(
                    input_slice.get(), input_slice.strides(), input_slice_shape,
                    output_slice.get(), output_slice.strides(), output_slice_shape,
                    details::extract_matrix(input_fwd_scaling_matrix),
                    details::extract_matrix(input_inv_rotation_matrix),
                    details::extract_matrix(output_inv_scaling_matrix),
                    details::extract_matrix(output_fwd_rotation_matrix),
                    cutoff, ews_radius, slice_z_radius, add_to_output, cuda_stream);
            cuda_stream.enqueue_attach(
                    input_slice, output_slice,
                    input_fwd_scaling_matrix, input_inv_rotation_matrix,
                    output_inv_scaling_matrix, output_fwd_rotation_matrix);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Extracts 2D Fourier slice(s) from a virtual volume filled by other slices, using linear interpolation.
    /// \details This function has the same features and limitations as the overload taking arrays, but uses textures.
    template<Remap REMAP, typename Value, typename Output, typename InputScale, typename InputRotate,
             typename OutputScale, typename OutputRotate, typename = std::enable_if_t<
                    nt::is_varray_of_any_v<Output, Value> &&
                    details::is_valid_insert_insert_extract_v<
                    REMAP, Value, InputScale, InputRotate, OutputScale, OutputRotate>>>
    void insert_interpolate_and_extract_3d(
            const Texture<Value>& input_slice, const Shape4<i64>& input_slice_shape,
            const Output& output_slice, const Shape4<i64>& output_slice_shape,
            const InputScale& input_fwd_scaling_matrix, const InputRotate& input_inv_rotation_matrix,
            const OutputScale& output_inv_scaling_matrix, const OutputRotate& output_fwd_rotation_matrix,
            f32 slice_z_radius = 0.f, bool add_to_output = false,
            f32 cutoff = 0.5f, const Vec2<f32>& ews_radius = {}) {

        const Device device = output_slice.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            const cpu::Texture<Value>& texture = input_slice.cpu();
            const Array<Value> input_slice_array(
                    texture.ptr, input_slice.shape(), texture.strides, input_slice.options());
            insert_interpolate_and_extract_3d<REMAP>(
                    input_slice_array, input_slice_shape, output_slice, output_slice_shape,
                    input_fwd_scaling_matrix, input_inv_rotation_matrix,
                    output_inv_scaling_matrix, output_fwd_rotation_matrix,
                    slice_z_radius, add_to_output, cutoff, ews_radius);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (nt::is_any_v<Value, f64, c64>) {
                NOA_THROW("Double-precision floating-points are not supported by CUDA textures");
            } else {
                details::projection_check_parameters<details::ProjectionType::INSERT_EXTRACT>(
                        input_slice, input_slice_shape, output_slice, output_slice_shape, {},
                        input_fwd_scaling_matrix, input_inv_rotation_matrix,
                        output_inv_scaling_matrix, output_fwd_rotation_matrix);

                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = input_slice.cuda();
                cuda::geometry::fft::insert_interpolate_and_extract_3d<REMAP>(
                        texture.array.get(), *texture.texture, input_slice.interp_mode(), input_slice_shape,
                        output_slice.get(), output_slice.strides(), output_slice_shape,
                        details::extract_matrix(input_fwd_scaling_matrix),
                        details::extract_matrix(input_inv_rotation_matrix),
                        details::extract_matrix(output_inv_scaling_matrix),
                        details::extract_matrix(output_fwd_rotation_matrix),
                        cutoff, ews_radius, slice_z_radius, add_to_output, cuda_stream);
                cuda_stream.enqueue_attach(
                        texture.array, texture.texture, output_slice,
                        input_fwd_scaling_matrix, input_inv_rotation_matrix,
                        output_inv_scaling_matrix, output_fwd_rotation_matrix);
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Extracts 2D Fourier slice(s) from a virtual volume filled by other slices, using linear interpolation.
    /// \details This function has the same features and limitations as the overload taking arrays,
    ///          but the slice is represented by a single constant value. This is for example useful
    ///          to keep track of the multiplicity of the Fourier insertion.
    template<Remap REMAP, typename Input, typename Output, typename InputScale, typename InputRotate,
             typename OutputScale, typename OutputRotate, typename = std::enable_if_t<
                    nt::is_varray_of_any_v<Output, Input> &&
                    details::is_valid_insert_insert_extract_v<
                    REMAP, Input, InputScale, InputRotate, OutputScale,OutputRotate>>>
    void insert_interpolate_and_extract_3d(
            Input input_slice, const Shape4<i64>& input_slice_shape,
            const Output& output_slice, const Shape4<i64>& output_slice_shape,
            const InputScale& input_fwd_scaling_matrix, const InputRotate& input_inv_rotation_matrix,
            const OutputScale& output_inv_scaling_matrix, const OutputRotate& output_fwd_rotation_matrix,
            f32 slice_z_radius = 0.f, bool add_to_output = false,
            f32 cutoff = 0.5f, const Vec2<f32>& ews_radius = {}) {

        details::projection_check_parameters<details::ProjectionType::INSERT_EXTRACT>(
                input_slice, input_slice_shape, output_slice, output_slice_shape, {},
                input_fwd_scaling_matrix, input_inv_rotation_matrix,
                output_inv_scaling_matrix, output_fwd_rotation_matrix);

        const Device device = output_slice.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::insert_interpolate_and_extract_3d<REMAP>(
                        input_slice, input_slice_shape,
                        output_slice.get(), output_slice.strides(), output_slice_shape,
                        details::extract_matrix(input_fwd_scaling_matrix),
                        details::extract_matrix(input_inv_rotation_matrix),
                        details::extract_matrix(output_inv_scaling_matrix),
                        details::extract_matrix(output_fwd_rotation_matrix),
                        cutoff, ews_radius, slice_z_radius, add_to_output, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::insert_interpolate_and_extract_3d<REMAP>(
                    input_slice, input_slice_shape,
                    output_slice.get(), output_slice.strides(), output_slice_shape,
                    details::extract_matrix(input_fwd_scaling_matrix),
                    details::extract_matrix(input_inv_rotation_matrix),
                    details::extract_matrix(output_inv_scaling_matrix),
                    details::extract_matrix(output_fwd_rotation_matrix),
                    cutoff, ews_radius, slice_z_radius, add_to_output, cuda_stream);
            cuda_stream.enqueue_attach(
                    output_slice,
                    input_fwd_scaling_matrix, input_inv_rotation_matrix,
                    output_inv_scaling_matrix, output_fwd_rotation_matrix);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Corrects for the gridding, assuming tri-linear interpolation was used during the Fourier insertion.
    /// \details During direct Fourier insertion of slices S into a volume B, two problems arises:
    ///          1) The insertion is not uniform (e.g. inherently more dense at low frequencies). This can be
    ///             easily corrected by inserting the data as well as its associated weights and normalizing the
    ///             inserted data with the inserted weights. This is often referred to as density correction.
    ///             This function is not about that.
    ///          2) The data-points are inserted in Fourier space by interpolation, a process called gridding,
    ///             which is essentially a convolution between the data points and the interpolation filter
    ///             (e.g. triangle pulse for linear interpolation). The interpolation filter is often referred to as
    ///             the gridding kernel. Since convolution in frequency space corresponds to a multiplication in
    ///             real-space, the resulting inverse Fourier transform of the volume B is the product of the final
    ///             wanted reconstruction and the apodization function. The apodization function is the Fourier
    ///             transform of the gridding kernel (e.g. sinc^2 for linear interpolation). This function is there
    ///             to correct for this gridding artefact, assuming tri-linear interpolation.
    /// \param[in] input        Inverse Fourier transform of the 3D grid used for direct Fourier insertion.
    /// \param[out] output      Gridding-corrected output. Can be equal to \p input.
    /// \param post_correction  Whether the correction is the post- or pre-correction.
    ///                         Post correction is meant to be applied on the volume that was just back-projected,
    ///                         whereas pre-correction is meant to be applied on the volume that is about to be
    ///                         forward projected.
    template<typename Input, typename Output, typename = std::enable_if_t<
             nt::is_varray_of_almost_any_v<Input, f32, f64> &&
             nt::is_varray_of_any_v<Output, f32, f64> &&
             nt::are_almost_same_value_type_v<Input, Output>>>
    void gridding_correction(const Input& input, const Output& output, bool post_correction) {
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
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::gridding_correction(
                        input.get(), input_strides,
                        output.get(), output.strides(),
                        output.shape(), post_correction, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::gridding_correction(
                    input.get(), input_strides,
                    output.get(), output.strides(),
                    output.shape(), post_correction, stream.cuda());
            cuda_stream.enqueue_attach(input, output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
