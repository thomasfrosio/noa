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

    template<typename Input, typename Output>
    constexpr bool is_valid_projection_input_output_v =
            nt::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
            (nt::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
             nt::are_almost_same_value_type_v<Input, Output>) ||
            (nt::is_almost_any_v<Input, f32, f64, c32, c64> &&
             nt::is_varray_of_any_v<Output, Input>);

    template<typename Scale>
    constexpr bool is_valid_projection_scale_v =
            nt::is_any_v<Scale, Float22> || nt::is_varray_of_almost_any_v<Scale, Float22>;

    template<typename Rotation>
    constexpr bool is_valid_projection_rotation_v =
            nt::is_any_v<Rotation, Float33> || nt::is_varray_of_almost_any_v<Rotation, Float33>;

    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_rasterize_v =
            is_valid_projection_input_output_v<Input, Output> &&
            is_valid_projection_scale_v<Scale> &&
            is_valid_projection_rotation_v<Rotate> &&
            (REMAP == Remap::H2H || REMAP == Remap::H2HC || REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_interpolate_v =
            is_valid_projection_input_output_v<Input, Output> &&
            is_valid_projection_scale_v<Scale> &&
            is_valid_projection_rotation_v<Rotate> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate>
    constexpr bool is_valid_extract_v =
            nt::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
            nt::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
            nt::are_almost_same_value_type_v<Input, Output> &&
            is_valid_projection_scale_v<Scale> &&
            is_valid_projection_rotation_v<Rotate> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Input, typename Output,
             typename InputScale, typename InputRotate,
             typename OutputScale, typename OutputRotate>
    constexpr bool is_valid_insert_extract_v =
            is_valid_projection_input_output_v<Input, Output> &&
            is_valid_projection_scale_v<InputScale> &&
            is_valid_projection_rotation_v<InputRotate> &&
            is_valid_projection_scale_v<OutputScale> &&
            is_valid_projection_rotation_v<OutputRotate> &&
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

    template<ProjectionType DIRECTION,
             typename Input, typename Output,
             typename Scale0, typename Rotate0,
             typename Scale1 = Float22, typename Rotate1 = Float33>
    void projection_check_parameters(
            const Input& input, const Shape4<i64>& input_shape,
            const Output& output, const Shape4<i64>& output_shape,
            const Shape4<i64>& target_shape,
            const Scale0& input_scaling_matrix,
            const Rotate0& input_rotation_matrix,
            const Scale1& output_scaling_matrix = {},
            const Rotate1& output_rotation_matrix = {}
    ) {
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
        if constexpr (nt::is_matXX_v<Matrix>) {
            return matrix;
        } else {
            using ptr_t = const typename Matrix::value_type*;
            return ptr_t(matrix.get());
        }
    }
}

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Inserts 2d Fourier central-slice(s) into a 3d Fourier volume, using tri-linear rasterization.
    /// \details Fourier-insertion using rasterization/gridding to insert central-slices in a volume.
    ///          This method is mostly used for cases with a lot of central-slices (where errors are averaged-out)
    ///          and is likely the most efficient way of implementing backward projection. Note however that this
    ///          method is not the most accurate (central-slices are modeled using a simple triliear-pulse for the
    ///          rasterization). A density correction (i.e. normalization) is then required. This can easily be
    ///          achieved by inserting the per-slice weights into another volume to keep track of what was inserted
    ///          and where. Gridding correction can also be beneficial as post-processing one the real-space output
    ///          (see gridding_correction() below).
    ///
    /// \tparam REMAP                   Remapping from the slice to the volume layout.
    ///                                 Should be H2H, H2HC, HC2H or HC2HC.
    /// \tparam Input                   (const) f32, f64, c32, c64, or a varray of this type.
    /// \tparam Output                  VArray of type f32, f64, c32, or c64.
    /// \tparam Scale                   Float22 or a varray of this type.
    /// \tparam Rotate                  Float33 or a varray of this type.
    /// \param[in] slice                2d-rfft central-slice(s) to insert. A single value can also be passed,
    ///                                 which is equivalent to a slice filled with a constant value.
    /// \param slice_shape              BDHW logical shape of \p slice.
    /// \param[out] volume              3d-rfft volume inside which the slices are inserted.
    /// \param volume_shape             BDHW logical shape of \p volume.
    /// \param[in] inv_scaling_matrix   2x2 HW \e inverse real-space scaling matrix to apply to the slices
    ///                                 before the rotation. If an array is passed, it can be empty or have
    ///                                 one matrix per slice. Otherwise the same scaling matrix is applied
    ///                                 to every slice.
    /// \param[in] fwd_rotation_matrix  3x3 DHW \e forward rotation matrices to apply to the slices.
    ///                                 If an array is passed, it should have one matrix per slice.
    ///                                 Otherwise the same rotation matrix is applied to every slice.
    /// \param fftfreq_cutoff           Frequency cutoff in \p volume, in cycle/pix.
    /// \param target_shape             Actual BDHW logical shape of the 3d volume (see note below).
    /// \param ews_radius               HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                 If negative, the negative curve is computed.
    ///                                 If {0,0}, the slices are projections.
    ///
    /// \note This function normalizes the slice and volume dimensions, and works with normalized frequencies.
    ///       By default (empty \p target_shape or \p target_shape == \p volume_shape), the slice frequencies
    ///       are mapped onto the volume frequencies. If the volume is larger than the slices, the slices are
    ///       implicitly stretched (over-sampling case). If the volume is smaller than the slices, the slices
    ///       are shrank (under-sampling case). However, if \p target_shape is specified, the slice frequencies
    ///       are instead mapped onto the frequencies of a 3d FFT volume of shape \p target_shape. In this case,
    ///       \p volume is the region to "render" within the volume, defined by \p target_shape, centered on the DC.
    ///       This can be useful for instance to only render a subregion of \p target_shape.
    /// \note In order to have both left and right beams assigned to different values, this function only computes one
    ///       "side" of the EWS, as specified by \p ews_radius. To insert the other side, one would have to
    ///       call this function a second time with \p ews_radius * -1.
    /// \note The scaling and the rotation matrices are kept separated from one another in order to properly compute the
    ///       curve of the Ewald sphere. Indeed, the scaling is applied first to correct for magnification, so that the
    ///       EWS is computed using the original frequencies (from the scattering) and is therefore spherical even
    ///       under anisotropic magnification. If \p ews_radius is 0, the scaling factors can be merged with the
    ///       rotations.
    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_rasterize_v<REMAP, Input, Output, Scale, Rotate>>>
    void insert_rasterize_3d(
            const Input& slice, const Shape4<i64> slice_shape,
            const Output& volume, const Shape4<i64> volume_shape,
            const Scale& inv_scaling_matrix,
            const Rotate& fwd_rotation_matrix,
            f32 fftfreq_cutoff = 0.5f,
            const Shape4<i64> target_shape = {},
            const Vec2<f32>& ews_radius = {}
    ) {
        details::projection_check_parameters<details::ProjectionType::INSERT_RASTERIZE>(
                slice, slice_shape, volume, volume_shape, target_shape,
                inv_scaling_matrix, fwd_rotation_matrix);

        const Device device = volume.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                if constexpr (nt::is_varray_v<Input>) {
                    noa::cpu::geometry::fft::insert_rasterize_3d<REMAP>(
                            slice.get(), slice.strides(), slice_shape,
                            volume.get(), volume.strides(), volume_shape,
                            details::extract_matrix(inv_scaling_matrix),
                            details::extract_matrix(fwd_rotation_matrix),
                            fftfreq_cutoff, target_shape, ews_radius, threads);
                } else if constexpr (nt::is_real_or_complex_v<Input>) {
                    noa::cpu::geometry::fft::insert_rasterize_3d<REMAP>(
                            slice, slice_shape,
                            volume.get(), volume.strides(), volume_shape,
                            details::extract_matrix(inv_scaling_matrix),
                            details::extract_matrix(fwd_rotation_matrix),
                            fftfreq_cutoff, target_shape, ews_radius, threads);
                } else {
                    static_assert(nt::always_false_v<Input>);
                }
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            if constexpr (nt::is_varray_v<Input>) {
                noa::cuda::geometry::fft::insert_rasterize_3d<REMAP>(
                        slice.get(), slice.strides(), slice_shape,
                        volume.get(), volume.strides(), volume_shape,
                        details::extract_matrix(inv_scaling_matrix),
                        details::extract_matrix(fwd_rotation_matrix),
                        fftfreq_cutoff, target_shape, ews_radius, cuda_stream);
            } else if constexpr (nt::is_real_or_complex_v<Input>) {
                noa::cuda::geometry::fft::insert_rasterize_3d<REMAP>(
                        slice, slice_shape,
                        volume.get(), volume.strides(), volume_shape,
                        details::extract_matrix(inv_scaling_matrix),
                        details::extract_matrix(fwd_rotation_matrix),
                        fftfreq_cutoff, target_shape, ews_radius, cuda_stream);
            } else {
                static_assert(nt::always_false_v<Input>);
            }
            cuda_stream.enqueue_attach(slice, volume, inv_scaling_matrix, fwd_rotation_matrix);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Settings for the windowed-sinc convolution of the central-slice.\n
    /// \b Fourier-insertion: Central slices are inserted in a (virtual) volume. This parameter defines
    /// the windowed-sinc that is convolved along the normal of the perfectly thin slice(s) to insert.\n
    /// \b Fourier-extraction: Central slices are extracted from a (virtual) volume. This parameter defines the
    /// windowed-sinc that is convolved, along the z of the reconstruction, with the perfectly thin slice(s) to
    /// extract. This is used to effectively apply an horizontal (smooth) rectangular mask centered on the object
    /// _before_ the forward projection. The current API doesn't allow to change the orientation of this sinc
    /// (it is always along z) since its only purpose was originally to improve projections from tomograms by
    /// masking out the noise from above and below the sample.
    struct WindowedSinc {
        /// Frequency, in cycle/pix, of the first zero of the sinc.
        /// This is clamped to ensure a minimum of 1 pixel diameter,
        /// which is usually want we want for Fourier insertion.
        f32 fftfreq_sinc{-1};

        /// Frequency, in cycle/pix, where the blackman window stops (weight is 0 at this frequency).
        /// This parameter is usually here to control the accuracy/performance ratio, but it can also be used
        /// to control the smoothness of the corresponding real-space mask. Usually this value should be a
        /// multiple of the sinc-cutoff. The larger this multiple, the sharper the step window, but the slower
        /// it is to compute the slice.
        /// This is clamped to ensure the window stops at least to the first sinc-cutoff.
        /// So if both frequencies are left to their default value (-1), a 1 pixel thick slice
        /// is generated, which is usually want we want for Fourier insertion.
        f32 fftfreq_blackman{-1};
    };

    /// Fourier-insertion using 2d-interpolation to insert central-slices in the volume.
    /// \details This function computes the inverse transformation compared to the overload above using rasterization.
    ///          This method is the most accurate one but is certainly slower than rasterization. Here, instead of
    ///          calling every pixel in the central-slices for rasterization, every voxel in the volume is sampled,
    ///          where, for each voxel, the contribution of every central-slice is computed. The advantage is that
    ///          it allows to use a more accurate model for the central-slices, i.e., a windowed-sinc. Indeed, slices
    ///          are now effectively convolved with a windowed-sinc (both the sinc frequency and window size can be
    ///          controlled) along their normal before the insertion. Note that this (windowed) sinc translates to
    ///          a (smooth) rectangular mask in real-space, along the normal of the slice (an interesting property
    ///          that can be useful for some applications).
    ///
    /// \tparam REMAP                   Remapping from the slice to the volume layout.
    ///                                 Should be HC2H or HC2HC.
    /// \tparam Input                   (const) f32, f64, c32, c64, or a varray of this type.
    /// \tparam Output                  VArray of type f32, f64, c32, or c64.
    /// \tparam Scale                   Float22 or an varray of this type.
    /// \tparam Rotate                  Float33 or an varray of this type.
    /// \param[in] slice                2d-rfft central-slice(s) to insert. A single value can also be passed,
    ///                                 which is equivalent to a slice filled with a constant value.
    /// \param slice_shape              BDHW logical shape of \p slice.
    /// \param[out] volume              3d-rfft volume inside which the slices are to be inserted.
    /// \param volume_shape             BDHW logical shape of \p volume.
    /// \param[in] fwd_scaling_matrix   2x2 HW \e forward real-space scaling matrix to apply to the slices
    ///                                 before the rotation. If an array is passed, it can be empty or have
    ///                                 one matrix per slice. Otherwise the same scaling matrix is applied
    ///                                 to every slice.
    /// \param[in] inv_rotation_matrix  3x3 DHW \e inverse rotation matrices to apply to the slices.
    ///                                 If an array is passed, it should have one matrix per slice.
    ///                                 Otherwise the same rotation matrix is applied to every slice.
    /// \param windowed_sinc            Windowed-sinc along the normal of the slice(s).
    /// \param fftfreq_cutoff           Frequency cutoff in \p volume, in cycle/pix.
    /// \param target_shape             Actual BDHW logical shape of the 3D volume.
    /// \param ews_radius               HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                 If negative, the negative curve is computed.
    ///                                 If {0,0}, the slices are projections.
    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_interpolate_v<REMAP, Input, Output, Scale, Rotate>>>
    void insert_interpolate_3d(
            const Input& slice, const Shape4<i64>& slice_shape,
            const Output& volume, const Shape4<i64>& volume_shape,
            const Scale& fwd_scaling_matrix,
            const Rotate& inv_rotation_matrix,
            const WindowedSinc& windowed_sinc = {},
            f32 fftfreq_cutoff = 0.5f,
            const Shape4<i64>& target_shape = {},
            const Vec2<f32>& ews_radius = {}
    ) {
        details::projection_check_parameters<details::ProjectionType::INSERT_INTERPOLATE>(
                slice, slice_shape, volume, volume_shape, target_shape,
                fwd_scaling_matrix, inv_rotation_matrix);

        const Device device = volume.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                if constexpr (nt::is_varray_v<Input>) {
                    noa::cpu::geometry::fft::insert_interpolate_3d<REMAP>(
                            slice.get(), slice.strides(), slice_shape,
                            volume.get(), volume.strides(), volume_shape,
                            details::extract_matrix(fwd_scaling_matrix),
                            details::extract_matrix(inv_rotation_matrix),
                            fftfreq_cutoff, windowed_sinc.fftfreq_sinc, windowed_sinc.fftfreq_blackman,
                            target_shape, ews_radius, threads);
                } else if constexpr (nt::is_real_or_complex_v<Input>) {
                    noa::cpu::geometry::fft::insert_interpolate_3d<REMAP>(
                            slice, slice_shape,
                            volume.get(), volume.strides(), volume_shape,
                            details::extract_matrix(fwd_scaling_matrix),
                            details::extract_matrix(inv_rotation_matrix),
                            fftfreq_cutoff, windowed_sinc.fftfreq_sinc, windowed_sinc.fftfreq_blackman,
                            target_shape, ews_radius, threads);
                } else {
                    static_assert(nt::always_false_v<Input>);
                }
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            if constexpr (nt::is_varray_v<Input>) {
                noa::cuda::geometry::fft::insert_interpolate_3d<REMAP>(
                        slice.get(), slice.strides(), slice_shape,
                        volume.get(), volume.strides(), volume_shape,
                        details::extract_matrix(fwd_scaling_matrix),
                        details::extract_matrix(inv_rotation_matrix),
                        fftfreq_cutoff, windowed_sinc.fftfreq_sinc, windowed_sinc.fftfreq_blackman,
                        target_shape, ews_radius, cuda_stream);
            } else if constexpr (nt::is_real_or_complex_v<Input>) {
                noa::cuda::geometry::fft::insert_interpolate_3d<REMAP>(
                        slice, slice_shape,
                        volume.get(), volume.strides(), volume_shape,
                        details::extract_matrix(fwd_scaling_matrix),
                        details::extract_matrix(inv_rotation_matrix),
                        fftfreq_cutoff, windowed_sinc.fftfreq_sinc, windowed_sinc.fftfreq_blackman,
                        target_shape, ews_radius, cuda_stream);
            } else {
                static_assert(nt::always_false_v<Input>);
            }
            cuda_stream.enqueue_attach(slice, volume, fwd_scaling_matrix, inv_rotation_matrix);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Fourier-insertion using 2d-interpolation to insert central-slices in the volume.
    /// \details This function has the same features and limitations as the overload taking arrays, but uses textures.
    template<Remap REMAP, typename Value, typename Output, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_interpolate_v<REMAP, Value, Output, Scale, Rotate>>>
    void insert_interpolate_3d(
            const Texture<Value>& slice, const Shape4<i64>& slice_shape,
            const Output& volume, const Shape4<i64>& volume_shape,
            const Scale& fwd_scaling_matrix,
            const Rotate& inv_rotation_matrix,
            const WindowedSinc& windowed_sinc = {},
            f32 fftfreq_cutoff = 0.5f,
            const Shape4<i64>& target_shape = {},
            const Vec2<f32>& ews_radius = {}
    ) {
        const Device device = volume.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            const noa::cpu::Texture<Value>& texture = slice.cpu();
            const Array<Value> slice_array(texture.ptr, slice.shape(), texture.strides, slice.options());
            insert_interpolate_3d<REMAP>(
                    slice_array, slice_shape, volume, volume_shape,
                    fwd_scaling_matrix, inv_rotation_matrix,
                    windowed_sinc, fftfreq_cutoff, target_shape, ews_radius);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (nt::is_any_v<Value, f64, c64>) {
                NOA_THROW("Double-precision floating-points are not supported by CUDA textures");
            } else {
                details::projection_check_parameters<details::ProjectionType::INSERT_INTERPOLATE>(
                        slice, slice_shape, volume, volume_shape, target_shape,
                        fwd_scaling_matrix, inv_rotation_matrix);

                const noa::cuda::Texture<Value>& texture = slice.cuda();
                auto& cuda_stream = stream.cuda();
                noa::cuda::geometry::fft::insert_interpolate_3d<REMAP>(
                        texture.array.get(), *texture.texture, slice.interp_mode(), slice_shape,
                        volume.get(), volume.strides(), volume_shape,
                        details::extract_matrix(fwd_scaling_matrix),
                        details::extract_matrix(inv_rotation_matrix),
                        fftfreq_cutoff, windowed_sinc.fftfreq_sinc, windowed_sinc.fftfreq_blackman,
                        target_shape, ews_radius, cuda_stream);
                cuda_stream.enqueue_attach(
                        texture.array, texture.texture, volume,
                        fwd_scaling_matrix, inv_rotation_matrix);
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Extracts 2d central-slice(s) from a volume.
    /// \details This is the reverse operation of the Fourier insertion. There are two main behaviors (both
    ///          controlled by the \p z_windowed_sinc parameter): 1. (default) A simple and fast slice extraction,
    ///          where every pixel of the slice(s) are sampled from the volume using 3d-interpolation.
    ///          2. A z-windowed-sinc slice extraction. This is similar, but instead of simply extracting the slice
    ///          from the volume, it convolves the volume with a 1d windowed-sinc along the z-axis of the volume.
    ///          Note that the convolution is simplified to a simple per-slice weighted-mean along the z-axis of the
    ///          volume. This windowed-sinc convolution translates to a (smooth) rectangular mask along the z-axis
    ///          and centered on the ifft of the volume. As such, if such masking is required, this method can replace
    ///          the real-space masking, which could be advantageous in scenarios where going back to real-space
    ///          is expensive.
    ///
    /// \tparam REMAP                   Remapping from the slice to the volume layout. Should be HC2H or HC2HC.
    /// \tparam Input                   VArray of type f32, f64, c32, or c64.
    /// \tparam Output                  VArray of type f32, f64, c32, or c64.
    /// \tparam Scale                   Float22 or an varray of this type.
    /// \tparam Rotate                  Float33 or an varray of this type.
    /// \param[in] volume               3d-centered-rfft volume from which to extract the slices.
    /// \param volume_shape             BDHW logical shape of \p volume.
    /// \param[out] slice               2d-rfft central-slice(s) to extract.
    /// \param slice_shape              BDHW logical shape of \p slice.
    /// \param[in] inv_scaling_matrix   2x2 HW \e inverse real-space scaling to apply to the slices before the rotation.
    ///                                 If an array is passed, it can be empty or have one matrix per slice.
    ///                                 Otherwise the same scaling matrix is applied to every slice.
    /// \param[in] fwd_rotation_matrix  3x3 DHW \e forward rotation matrices to apply to the slices.
    ///                                 If an array is passed, it should have one matrix per slice.
    ///                                 Otherwise the same rotation matrix is applied to every slice.
    /// \param z_windowed_sinc          Windowed-sinc along the z of \p volume.
    /// \param fftfreq_cutoff           Frequency cutoff in \p volume, in cycle/pix.
    /// \param target_shape             Actual BDHW logical shape of the 3D volume.
    /// \param ews_radius               HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                 If negative, the negative curve is computed.
    ///                                 If {0,0}, the slices are projections.
    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_extract_v<REMAP, Input, Output, Scale, Rotate>>>
    void extract_3d(
            const Input& volume, const Shape4<i64>& volume_shape,
            const Output& slice, const Shape4<i64>& slice_shape,
            const Scale& inv_scaling_matrix,
            const Rotate& fwd_rotation_matrix,
            const WindowedSinc& z_windowed_sinc = {},
            f32 fftfreq_cutoff = 0.5f,
            const Shape4<i64>& target_shape = {},
            const Vec2<f32>& ews_radius = {}
    ) {
        details::projection_check_parameters<details::ProjectionType::EXTRACT>(
                volume, volume_shape, slice, slice_shape, target_shape,
                inv_scaling_matrix, fwd_rotation_matrix);

        const Device device = volume.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::geometry::fft::extract_3d<REMAP>(
                        volume.get(), volume.strides(), volume_shape,
                        slice.get(), slice.strides(), slice_shape,
                        details::extract_matrix(inv_scaling_matrix),
                        details::extract_matrix(fwd_rotation_matrix),
                        fftfreq_cutoff, z_windowed_sinc.fftfreq_sinc, z_windowed_sinc.fftfreq_blackman,
                        target_shape, ews_radius, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::geometry::fft::extract_3d<REMAP>(
                    volume.get(), volume.strides(), volume_shape,
                    slice.get(), slice.strides(), slice_shape,
                    details::extract_matrix(inv_scaling_matrix),
                    details::extract_matrix(fwd_rotation_matrix),
                    fftfreq_cutoff, z_windowed_sinc.fftfreq_sinc, z_windowed_sinc.fftfreq_blackman,
                    target_shape, ews_radius, cuda_stream);
            cuda_stream.enqueue_attach(
                    volume, slice, inv_scaling_matrix, fwd_rotation_matrix);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Extracts 2d central-slice(s) from a volume.
    /// \details This function has the same features and limitations as the overload taking arrays, but uses textures.
    template<Remap REMAP, typename Value, typename Output, typename Scale, typename Rotate, typename = std::enable_if_t<
             details::is_valid_insert_interpolate_v<REMAP, Value, Output, Scale, Rotate>>>
    void extract_3d(
            const Texture<Value>& volume, const Shape4<i64>& volume_shape,
            const Output& slice, const Shape4<i64>& slice_shape,
            const Scale& inv_scaling_matrix,
            const Rotate& fwd_rotation_matrix,
            const WindowedSinc& z_windowed_sinc = {},
            f32 fftfreq_cutoff = 0.5f,
            const Shape4<i64>& target_shape = {},
            const Vec2<f32>& ews_radius = {}
    ) {
        const Device device = volume.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            const noa::cpu::Texture<Value>& texture = volume.cpu();
            const Array<Value> volume_array(texture.ptr, volume.shape(), texture.strides, volume.options());
            extract_3d<REMAP>(volume_array, volume_shape, slice, slice_shape,
                              inv_scaling_matrix, fwd_rotation_matrix,
                              z_windowed_sinc, fftfreq_cutoff, target_shape, ews_radius);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (nt::is_any_v<Value, f64, c64>) {
                NOA_THROW("Double-precision floating-points are not supported by CUDA textures");
            } else {
                details::projection_check_parameters<details::ProjectionType::EXTRACT>(
                        volume, volume_shape, slice, slice_shape, target_shape,
                        inv_scaling_matrix, fwd_rotation_matrix);

                auto& cuda_stream = stream.cuda();
                const noa::cuda::Texture<Value>& texture = volume.cuda();
                noa::cuda::geometry::fft::extract_3d<REMAP>(
                        texture.array.get(), *texture.texture, volume.interp_mode(), volume_shape,
                        slice.get(), slice.strides(), slice_shape,
                        details::extract_matrix(inv_scaling_matrix),
                        details::extract_matrix(fwd_rotation_matrix),
                        fftfreq_cutoff, z_windowed_sinc.fftfreq_sinc, z_windowed_sinc.fftfreq_blackman,
                        target_shape, ews_radius, cuda_stream);
                cuda_stream.enqueue_attach(
                        texture.array, texture.texture, slice,
                        inv_scaling_matrix, fwd_rotation_matrix);
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Extracts 2d central-slice(s) from a virtual volume filled by other central-slices.
    /// \details This function effectively combines the insertion and extraction, but instead of actually inserting
    ///          slices into a volume, it directly inserts them in the extracted slices. In other words, it builds a
    ///          virtual volume, made of central-slices, and this virtual volume is then sampled at (exactly) the
    ///          frequency of the central-slices to extract. This has massive performance benefits, because it only
    ///          samples the frequency of the output slices and never allocates/reconstructs the volume. It is also
    ///          more accurate since the volume is never actually discretized (thus skipping a layer of interpolation).
    ///          Note that these performance benefits are expected to disappear if thousands (possibly hundreds?) of
    ///          slices are extracted. Indeed, for every output slice, the operator needs to sample the volume by
    ///          collecting the signal of every input slice using 2d-interpolation. This is as opposed to the other
    ///          extract method, where the volume is already sampled, making the extraction much cheaper (and constant
    ///          cost: it's a simple 3d-interpolation).
    ///
    /// \tparam REMAP                           Remapping from the slice to the volume layout. Should be HC2H or HC2HC.
    /// \tparam Input                           VArray or value of type (const) f32, f64, c32, or c64.
    /// \tparam Output                          VArray of type f32, f64, c32, or c64.
    /// \tparam InputScale                      Float22 or a varray of this type.
    /// \tparam InputRotate                     Float33 or a varray of this type.
    /// \tparam OutputScale                     Float22 or a varray of this type.
    /// \tparam OutputRotate                    Float33 or a varray of this type.
    ///
    /// \param[in] input_slice                  2d-rfft central-slice(s) to insert. A single value can also be passed,
    ///                                         which is equivalent to a slice filled with a constant value.
    /// \param input_slice_shape                BDHW logical shape of \p input_slice.
    /// \param[in,out] output_slice             2d-rfft central-slice(s) to extract. See \p add_to_output.
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
    /// \param input_windowed_sinc              Windowed-sinc along the normal of the input slice(s).
    /// \param z_windowed_sinc                  Windowed-sinc along the z of the virtual volume.
    /// \param add_to_output                    Whether the contribution of the input slices should be added to the
    ///                                         output. By default, the function sets \p output_slice. With this option
    ///                                         enabled, it instead adds the contribution of \p input_slice to the
    ///                                         signal already in \p output_slice, allowing to progressively
    ///                                         build the output signal.
    /// \param correct_multiplicity             Correct for the multiplicity. By default, the virtual volume contains
    ///                                         the sum of the inserted values (weighted by \p input_windowed_sinc).
    ///                                         If true, the multiplicity is corrected by dividing the virtual volume
    ///                                         with the total inserted weights, only if the weight is larger than 1.
    ///                                         Indeed, if the weight for a frequency is less than 1, the frequency is
    ///                                         unchanged. The total insertion weights can be computed by filling the
    ///                                         input slices with ones. Note that this parameter is likely to only
    ///                                         make sense if \p add_to_output is false.
    /// \param fftfreq_cutoff                   Frequency cutoff of the virtual 3d volume, in cycle/pix.
    /// \param ews_radius                       HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                         If negative, the negative curve is computed.
    ///                                         If {0,0}, the slices are projections.
    template<Remap REMAP, typename Input, typename Output,
             typename InputScale, typename InputRotate,
             typename OutputScale, typename OutputRotate,
             typename = std::enable_if_t<details::is_valid_insert_extract_v<
             REMAP, Input, Output, InputScale, InputRotate, OutputScale, OutputRotate>>>
    void insert_interpolate_and_extract_3d(
            const Input& input_slice, const Shape4<i64>& input_slice_shape,
            const Output& output_slice, const Shape4<i64>& output_slice_shape,
            const InputScale& input_fwd_scaling_matrix, const InputRotate& input_inv_rotation_matrix,
            const OutputScale& output_inv_scaling_matrix, const OutputRotate& output_fwd_rotation_matrix,
            const WindowedSinc& input_windowed_sinc = {},
            const WindowedSinc& z_windowed_sinc = {},
            bool add_to_output = false,
            bool correct_multiplicity = false,
            f32 fftfreq_cutoff = 0.5f,
            const Vec2<f32>& ews_radius = {}
    ) {
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
                if constexpr (nt::is_varray_v<Input>) {
                    noa::cpu::geometry::fft::insert_interpolate_and_extract_3d<REMAP>(
                            input_slice.get(), input_slice.strides(), input_slice_shape,
                            output_slice.get(), output_slice.strides(), output_slice_shape,
                            details::extract_matrix(input_fwd_scaling_matrix),
                            details::extract_matrix(input_inv_rotation_matrix),
                            details::extract_matrix(output_inv_scaling_matrix),
                            details::extract_matrix(output_fwd_rotation_matrix),
                            fftfreq_cutoff, input_windowed_sinc.fftfreq_sinc, input_windowed_sinc.fftfreq_blackman,
                            z_windowed_sinc.fftfreq_sinc, z_windowed_sinc.fftfreq_blackman,
                            add_to_output, correct_multiplicity, ews_radius, threads);
                } else if constexpr (nt::is_real_or_complex_v<Input>) {
                    noa::cpu::geometry::fft::insert_interpolate_and_extract_3d<REMAP>(
                            input_slice, input_slice_shape,
                            output_slice.get(), output_slice.strides(), output_slice_shape,
                            details::extract_matrix(input_fwd_scaling_matrix),
                            details::extract_matrix(input_inv_rotation_matrix),
                            details::extract_matrix(output_inv_scaling_matrix),
                            details::extract_matrix(output_fwd_rotation_matrix),
                            fftfreq_cutoff, input_windowed_sinc.fftfreq_sinc, input_windowed_sinc.fftfreq_blackman,
                            z_windowed_sinc.fftfreq_sinc, z_windowed_sinc.fftfreq_blackman,
                            add_to_output, correct_multiplicity, ews_radius, threads);
                } else {
                    static_assert(nt::always_false_v<Input>);
                }
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            if constexpr (nt::is_varray_v<Input>) {
                noa::cuda::geometry::fft::insert_interpolate_and_extract_3d<REMAP>(
                        input_slice.get(), input_slice.strides(), input_slice_shape,
                        output_slice.get(), output_slice.strides(), output_slice_shape,
                        details::extract_matrix(input_fwd_scaling_matrix),
                        details::extract_matrix(input_inv_rotation_matrix),
                        details::extract_matrix(output_inv_scaling_matrix),
                        details::extract_matrix(output_fwd_rotation_matrix),
                        fftfreq_cutoff, input_windowed_sinc.fftfreq_sinc, input_windowed_sinc.fftfreq_blackman,
                        z_windowed_sinc.fftfreq_sinc, z_windowed_sinc.fftfreq_blackman,
                        add_to_output, correct_multiplicity, ews_radius, cuda_stream);
            } else if constexpr (nt::is_real_or_complex_v<Input>) {
                noa::cuda::geometry::fft::insert_interpolate_and_extract_3d<REMAP>(
                        input_slice, input_slice_shape,
                        output_slice.get(), output_slice.strides(), output_slice_shape,
                        details::extract_matrix(input_fwd_scaling_matrix),
                        details::extract_matrix(input_inv_rotation_matrix),
                        details::extract_matrix(output_inv_scaling_matrix),
                        details::extract_matrix(output_fwd_rotation_matrix),
                        fftfreq_cutoff, input_windowed_sinc.fftfreq_sinc, input_windowed_sinc.fftfreq_blackman,
                        z_windowed_sinc.fftfreq_sinc, z_windowed_sinc.fftfreq_blackman,
                        add_to_output, correct_multiplicity, ews_radius, cuda_stream);
            } else {
                static_assert(nt::always_false_v<Input>);
            }
            cuda_stream.enqueue_attach(
                    input_slice, output_slice,
                    input_fwd_scaling_matrix, input_inv_rotation_matrix,
                    output_inv_scaling_matrix, output_fwd_rotation_matrix);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Extracts 2d central-slice(s) from a virtual volume filled by other central-slices.
    /// \details This function has the same features and limitations as the overload taking arrays, but uses textures.
    template<Remap REMAP, typename Value, typename Output,
             typename InputScale, typename InputRotate,
             typename OutputScale, typename OutputRotate,
             typename = std::enable_if_t<details::is_valid_insert_extract_v<
                     REMAP, Value, Output, InputScale, InputRotate, OutputScale, OutputRotate>>>
    void insert_interpolate_and_extract_3d(
            const Texture<Value>& input_slice, const Shape4<i64>& input_slice_shape,
            const Output& output_slice, const Shape4<i64>& output_slice_shape,
            const InputScale& input_fwd_scaling_matrix, const InputRotate& input_inv_rotation_matrix,
            const OutputScale& output_inv_scaling_matrix, const OutputRotate& output_fwd_rotation_matrix,
            const WindowedSinc& input_windowed_sinc = {},
            const WindowedSinc& z_windowed_sinc = {},
            bool add_to_output = false,
            bool correct_multiplicity = false,
            f32 fftfreq_cutoff = 0.5f,
            const Vec2<f32>& ews_radius = {}
    ) {
        const Device device = output_slice.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            const noa::cpu::Texture<Value>& texture = input_slice.cpu();
            const Array<Value> input_slice_array(
                    texture.ptr, input_slice.shape(), texture.strides, input_slice.options());
            insert_interpolate_and_extract_3d<REMAP>(
                    input_slice_array, input_slice_shape, output_slice, output_slice_shape,
                    input_fwd_scaling_matrix, input_inv_rotation_matrix,
                    output_inv_scaling_matrix, output_fwd_rotation_matrix,
                    input_windowed_sinc, z_windowed_sinc,
                    add_to_output, correct_multiplicity, fftfreq_cutoff, ews_radius);
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
                const noa::cuda::Texture<Value>& texture = input_slice.cuda();
                noa::cuda::geometry::fft::insert_interpolate_and_extract_3d<REMAP>(
                        texture.array.get(), *texture.texture, input_slice.interp_mode(), input_slice_shape,
                        output_slice.get(), output_slice.strides(), output_slice_shape,
                        details::extract_matrix(input_fwd_scaling_matrix),
                        details::extract_matrix(input_inv_rotation_matrix),
                        details::extract_matrix(output_inv_scaling_matrix),
                        details::extract_matrix(output_fwd_rotation_matrix),
                        fftfreq_cutoff, input_windowed_sinc.fftfreq_sinc, input_windowed_sinc.fftfreq_blackman,
                        z_windowed_sinc.fftfreq_sinc, z_windowed_sinc.fftfreq_blackman,
                        add_to_output, correct_multiplicity, ews_radius, cuda_stream);
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
    /// \param[in] input        Inverse Fourier transform of the 3d volume used for direct Fourier insertion.
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
                noa::cpu::geometry::fft::gridding_correction(
                        input.get(), input_strides,
                        output.get(), output.strides(),
                        output.shape(), post_correction, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::geometry::fft::gridding_correction(
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
