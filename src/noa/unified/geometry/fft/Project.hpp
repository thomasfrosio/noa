#pragma once

#include "noa/core/geometry/Euler.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/core/geometry/Quaternion.hpp"

#include "noa/cpu/geometry/fft/Project.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Project.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/Texture.hpp"

namespace noa::geometry::fft::details {
    using Remap = noa::fft::Remap;

    template<typename Input, typename Output, bool ALLOW_TEXTURE, bool ALLOW_VALUE>
    constexpr bool is_valid_projection_input_output_v =
            nt::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
            ((nt::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
              nt::are_almost_same_value_type_v<Input, Output>) ||
             (ALLOW_TEXTURE && nt::is_texture_of_almost_any_v<Input, f32, f64, c32, c64> &&
              nt::are_almost_same_value_type_v<Input, Output>) ||
             (ALLOW_VALUE && nt::is_almost_any_v<Input, f32, f64, c32, c64> &&
              nt::is_varray_of_any_v<Output, Input>));

    template<typename Scale>
    constexpr bool is_valid_projection_scale_v =
            nt::is_any_v<Scale, Float22> || nt::is_varray_of_almost_any_v<Scale, Float22>;

    template<typename Rotation>
    constexpr bool is_valid_projection_rotation_v =
            nt::is_any_v<Rotation, Float33, Quaternion<f32>> ||
            nt::is_varray_of_almost_any_v<Rotation, Float33, Quaternion<f32>>;

    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_rasterize_v =
            is_valid_projection_input_output_v<Input, Output, false, true> &&
            is_valid_projection_scale_v<Scale> &&
            is_valid_projection_rotation_v<Rotate> &&
            (REMAP == Remap::H2H || REMAP == Remap::H2HC || REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_interpolate_v =
            is_valid_projection_input_output_v<Input, Output, true, true> &&
            is_valid_projection_scale_v<Scale> &&
            is_valid_projection_rotation_v<Rotate> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Input, typename Output, typename Scale, typename Rotate>
    constexpr bool is_valid_extract_v =
            is_valid_projection_input_output_v<Input, Output, true, false> &&
            is_valid_projection_scale_v<Scale> &&
            is_valid_projection_rotation_v<Rotate> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Input, typename Output,
             typename InputScale, typename InputRotate,
             typename OutputScale, typename OutputRotate>
    constexpr bool is_valid_insert_extract_v =
            is_valid_projection_input_output_v<Input, Output, true, true> &&
            is_valid_projection_scale_v<InputScale> &&
            is_valid_projection_rotation_v<InputRotate> &&
            is_valid_projection_scale_v<OutputScale> &&
            is_valid_projection_rotation_v<OutputRotate> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<bool OPTIONAL, typename Transform>
    void project_check_transform(const Transform& transform, i64 required_size, Device compute_device) {
        if constexpr (OPTIONAL) {
            if (transform.is_empty())
                return;
        } else {
            NOA_CHECK(!transform.is_empty(), "The transform should not be empty");
        }

        NOA_CHECK(noa::indexing::is_contiguous_vector(transform) && transform.elements() == required_size,
                  "The number of transforms, specified as a contiguous vector, should be equal to "
                  "the number of slices, but got transform shape={}, strides={}, with {} slices",
                  transform.shape(), transform.strides(), required_size);

        NOA_CHECK(transform.device() == compute_device,
                  "The transformation parameters should be on the compute device");
    }

    enum class ProjectionType { INSERT_RASTERIZE, INSERT_INTERPOLATE, EXTRACT, INSERT_EXTRACT };

    template<ProjectionType DIRECTION,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename InputScale, typename InputRotate,
             typename OutputScale = Float22, typename OutputRotate = Float33>
    void projection_check_parameters(
            const Input& input, const InputWeight& input_weight, const Shape4<i64>& input_shape,
            const Output& output, const OutputWeight& output_weight, const Shape4<i64>& output_shape,
            const Shape4<i64>& target_shape,
            const InputScale& input_scaling,
            const InputRotate& input_rotation,
            const OutputScale& output_scaling = {},
            const OutputRotate& output_rotation = {}
    ) {
        // Output:
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        const Device output_device = output.device();
        NOA_CHECK(noa::all(output.shape() == output_shape.rfft()),
                  "The shape of the rfft output does not match the expected shape. Got {} and expected {}",
                  output.shape(), output_shape.rfft());

        // Input:
        if constexpr (!nt::is_numeric_v<Input>) {
            NOA_CHECK(!input.is_empty(), "Empty array detected");
            if constexpr (nt::is_varray_v<Input>) {
                NOA_CHECK(!noa::indexing::are_overlapped(input, output),
                          "Input and output arrays should not overlap");
            }
            if constexpr (nt::is_texture_v<Input>) {
                NOA_CHECK(input.interp_mode() == InterpMode::LINEAR ||
                          input.interp_mode() == InterpMode::LINEAR_FAST,
                          "The texture interpolation mode should be {} or {}, but got {}",
                          InterpMode::LINEAR, InterpMode::LINEAR_FAST, input.interp_mode());
                NOA_CHECK(input.border_mode() == BorderMode::ZERO,
                          "The texture border mode should be {}, but got {}",
                          BorderMode::ZERO, input.border_mode());
            }
            const Device device = input.device();
            NOA_CHECK(device == output_device,
                      "The input and output should be on the same device but got input={} and output={}",
                      device, output_device);
            NOA_CHECK(noa::all(input.shape() == input_shape.rfft()),
                      "The shape of the rfft input does not match the expected shape. "
                      "Got {} and expected {}", input.shape(), input_shape.rfft());
        }

        // InputWeight:
        if constexpr (!nt::is_numeric_v<InputWeight> && !std::is_empty_v<InputWeight>) {
            NOA_CHECK(!input_weight.is_empty(), "Empty array detected");
            if constexpr (nt::is_varray_v<InputWeight>) {
                NOA_CHECK(!noa::indexing::are_overlapped(input_weight, input),
                          "Input and output arrays should not overlap");
            }
            if constexpr (nt::is_texture_v<InputWeight>) {
                NOA_CHECK(input_weight.interp_mode() == InterpMode::LINEAR ||
                          input_weight.interp_mode() == InterpMode::LINEAR_FAST,
                          "The texture interpolation mode should be {} or {}, but got {}",
                          InterpMode::LINEAR, InterpMode::LINEAR_FAST, input_weight.interp_mode());
                NOA_CHECK(input_weight.border_mode() == BorderMode::ZERO,
                          "The texture border mode should be {}, but got {}",
                          BorderMode::ZERO, input_weight.border_mode());
            }
            const Device device = input_weight.device();
            NOA_CHECK(device == output_device,
                      "The input weight and output should be on the same device "
                      "but got input_weight={} and output={}",
                      device, output_device);
            NOA_CHECK(noa::all(input_weight.shape() == input_shape.rfft()),
                      "The shape of the rfft input weight does not match the expected shape. "
                      "Got {} and expected {}", input_weight.shape(), input_shape.rfft());
        }

        if constexpr (!std::is_empty_v<OutputWeight>) {
            NOA_CHECK(!output_weight.is_empty(), "Empty array detected");
            NOA_CHECK(!noa::indexing::are_overlapped(output_weight, output),
                      "Output arrays should not overlap");
            const Device device = output_weight.device();
            NOA_CHECK(device == output_device,
                      "The output weight and output should be on the same device "
                      "but got output_weight={} and output={}",
                      device, output_device);
            NOA_CHECK(noa::all(output_weight.shape() == output_shape.rfft()),
                      "The shape of the rfft output weight does not match the expected shape. "
                      "Got {} and expected {}",
                      output_weight.shape(), output_shape.rfft());
        }

        if constexpr (DIRECTION == ProjectionType::INSERT_RASTERIZE ||
                      DIRECTION == ProjectionType::INSERT_INTERPOLATE) {
            NOA_CHECK(input_shape[1] == 1,
                      "2d input slices are expected but got shape {}", input_shape);
            if (noa::any(target_shape == 0)) {
                NOA_CHECK(output_shape[0] == 1 && !output_shape.is_batched(),
                          "A single 3d output is expected but got shape {}", output_shape);
            } else {
                NOA_CHECK(output_shape[0] == 1 && target_shape[0] == 1 && !target_shape.is_batched(),
                          "A single grid is expected, with a target shape describing a single 3d volume, "
                          "but got output shape {} and target shape {}", output_shape, target_shape);
            }
        } else if constexpr (DIRECTION == ProjectionType::EXTRACT) {
            NOA_CHECK(output_shape[1] == 1,
                      "2d input slices are expected but got shape {}", output_shape);
            if (noa::any(target_shape == 0)) {
                NOA_CHECK(input_shape[0] == 1 && !input_shape.is_batched(),
                          "A single 3d input is expected but got shape {}", input_shape);
            } else {
                NOA_CHECK(input_shape[0] == 1 && target_shape[0] == 1 && !target_shape.is_batched(),
                          "A single grid is expected, with a target shape describing a single 3d volume, "
                          "but got input shape {} and target shape {}", input_shape, target_shape);
            }
        } else { // INSERT_EXTRACT
            NOA_CHECK(input_shape[1] == 1 && output_shape[1] == 1,
                      "2d slices are expected but got shape input:{} and output:{}",
                      input_shape, output_shape);
        }

        const auto required_count = DIRECTION == ProjectionType::EXTRACT ? output_shape[0] : input_shape[0];
        if constexpr (nt::is_varray_v<InputScale>)
            project_check_transform<true>(input_scaling, required_count, output_device);
        if constexpr (nt::is_varray_v<InputRotate>)
            project_check_transform<false>(input_rotation, required_count, output_device);

        // Only for INSERT_EXTRACT.
        if constexpr (nt::is_varray_v<OutputScale>)
            project_check_transform<true>(output_scaling, output_shape[0], output_device);
        if constexpr (nt::is_varray_v<OutputRotate>)
            project_check_transform<false>(output_rotation, output_shape[0], output_device);
    }

    template<bool CONST = false, bool CPU = false, typename T>
    auto extract_pointer(const T& thing) {
        using mutable_t = nt::mutable_value_type_t<T>;
        if constexpr (nt::is_varray_v<T>) {
            using ptr_t = std::conditional_t<CONST, const mutable_t*, mutable_t*>;
            return ptr_t(thing.get());

        } else if constexpr (nt::is_texture_v<T> && CPU) {
            const noa::cpu::Texture<mutable_t>& texture = thing.cpu();
            using ptr_t = std::conditional_t<CONST, const mutable_t*, mutable_t*>;
            return ptr_t(texture.ptr.get());

        } else if constexpr (nt::is_texture_v<T> && !CPU) {
            const noa::gpu::Texture<mutable_t>& gpu_texture = thing.gpu();
            using texture_object_t = noa::gpu::TextureObject<mutable_t>;
            return texture_object_t{gpu_texture.array.get(), *gpu_texture.texture, thing.interp_mode()};

        } else { // real or complex, or Empty
            return thing;
        }
    }

    template<typename T>
    auto extract_strides(const T& thing) {
        if constexpr (nt::is_varray_v<T> || nt::is_texture_v<T>) {
            return thing.strides();
        } else { // real or complex, or Empty
            return Strides4<i64>{};
        }
    }

    template<typename Transform>
    auto extract_transform(const Transform& transform) {
        if constexpr (nt::is_varray_v<Transform>) {
            using ptr_t = const nt::value_type_t<Transform>*;
            return ptr_t(transform.get());
        } else {
            return transform;
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
    ///          rasterization). Gridding correction can be beneficial as post-processing one the real-space output.
    ///          A density correction (i.e. normalization) is required. This can easily be achieved by inserting the
    ///          per-slice weights into another volume to keep track of what was inserted and where.
    ///
    /// \tparam REMAP               Remapping from the slice to the volume layout.
    ///                             Should be H2H, H2HC, HC2H or HC2HC.
    /// \tparam Input               A single (const) f32|f64|c32|c64, or a varray of this type.
    /// \tparam InputWeight         A single (const) f32|f64|c32|c64, or a varray of this type, or empty.
    /// \tparam Output              VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight        VArray of type f32|f64|c32|c64, or empty.
    /// \tparam Scale               Mat22 or a varray of this type.
    /// \tparam Rotate              Mat33, Quaternion, or a varray of this type.
    ///
    /// \param[in] slice            2d-rfft central-slice(s) to insert (can be a constant value).
    /// \param[in] slice_weight     Another optional varray|value associated with \p slice.
    /// \param slice_shape          BDHW logical shape of \p slice.
    /// \param[out] volume          3d-rfft volume inside which the slices are inserted.
    /// \param[out] volume_weight   Another optional varray|value associated with \p volume.
    /// \param volume_shape         BDHW logical shape of \p volume.
    /// \param[in] inv_scaling      2x2 HW \e inverse real-space scaling matrix to apply to the slices
    ///                             before the rotation. If an array is passed, it can be empty or have
    ///                             one matrix per slice. Otherwise the same scaling matrix is applied
    ///                             to every slice.
    /// \param[in] fwd_rotation     3x3 DHW \e forward rotation-matrices or quaternions to apply to the slices.
    ///                             If an array is passed, it should have one element per slice.
    ///                             Otherwise the same rotation is applied to every slice.
    /// \param fftfreq_cutoff       Frequency cutoff in \p volume, in cycle/pix.
    /// \param target_shape         Actual BDHW logical shape of the 3d volume (see note below).
    /// \param ews_radius           HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                             If negative, the negative curve is computed.
    ///                             If {0,0}, the slices are projections.
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
    /// \note The scaling and the rotation are kept separated from one another in order to properly compute the
    ///       curve of the Ewald sphere. Indeed, the scaling is applied first to correct for magnification, so that the
    ///       EWS is computed using the original frequencies (from the scattering) and is therefore spherical even
    ///       under anisotropic magnification. If \p ews_radius is 0, the scaling factors can be merged with the
    ///       rotations (if a rotation-matrix is passed).
    template<Remap REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_rasterize_v<REMAP, Input, Output, Scale, Rotate>>>
    void insert_rasterize_3d(
            const Input& slice, const InputWeight& slice_weight, const Shape4<i64> slice_shape,
            const Output& volume, const OutputWeight& volume_weight, const Shape4<i64> volume_shape,
            const Scale& inv_scaling, const Rotate& fwd_rotation,
            f32 fftfreq_cutoff = 0.5f,
            const Shape4<i64> target_shape = {},
            const Vec2<f32>& ews_radius = {}
    ) {
        details::projection_check_parameters<details::ProjectionType::INSERT_RASTERIZE>(
                slice, slice_weight, slice_shape, volume, volume_weight, volume_shape,
                target_shape, inv_scaling, fwd_rotation);

        const Device device = volume.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::geometry::fft::insert_rasterize_3d<REMAP>(
                        details::extract_pointer<true>(slice), details::extract_strides(slice),
                        details::extract_pointer<true>(slice_weight), details::extract_strides(slice_weight),
                        slice_shape,
                        details::extract_pointer(volume), details::extract_strides(volume),
                        details::extract_pointer(volume_weight), details::extract_strides(volume_weight),
                        volume_shape,
                        details::extract_transform(inv_scaling),
                        details::extract_transform(fwd_rotation),
                        fftfreq_cutoff, target_shape, ews_radius, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::geometry::fft::insert_rasterize_3d<REMAP>(
                    details::extract_pointer<true>(slice), details::extract_strides(slice),
                    details::extract_pointer<true>(slice_weight), details::extract_strides(slice_weight),
                    slice_shape,
                    details::extract_pointer(volume), details::extract_strides(volume),
                    details::extract_pointer(volume_weight), details::extract_strides(volume_weight),
                    volume_shape,
                    details::extract_transform(inv_scaling),
                    details::extract_transform(fwd_rotation),
                    fftfreq_cutoff, target_shape, ews_radius, cuda_stream);
            cuda_stream.enqueue_attach(slice, slice_weight, volume, volume_weight, inv_scaling, fwd_rotation);
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
    /// \details This method is the most accurate one but is certainly slower than rasterization. Here, instead of
    ///          calling every pixel in the central-slices for rasterization, every voxel in the volume is sampled
    ///          by collecting the contribution of every central-slice for each output voxel. The advantage is that
    ///          it allows to use a more accurate model for the central-slices, i.e., a windowed-sinc. Indeed, slices
    ///          are now effectively convolved with a windowed-sinc (both the sinc frequency and window size can be
    ///          controlled) along their normal before the insertion. Note that this (windowed) sinc translates to
    ///          a (smooth) rectangular mask in real-space, along the normal of the slice (an interesting property
    ///          that can be useful for some applications).
    ///
    /// \tparam REMAP               Remapping from the slice to the volume layout. Should be HC2H or HC2HC.
    /// \tparam Input               A single (const) f32|f64|c32|c64, or a varray|texture of this type.
    /// \tparam InputWeight         A single (const) f32|f64|c32|c64, or a varray|texture of this type, or empty.
    /// \tparam Output              VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight        VArray of type f32|f64|c32|c64, or empty.
    /// \tparam Scale               Mat22 or a varray of this type.
    /// \tparam Rotate              Mat33, Quaternion, or a varray of this type.
    ///
    /// \param[in] slice            2d-rfft central-slice(s) to insert (can be a constant value).
    /// \param[in] slice_weight     Another optional varray|texture|value associated with \p slice.
    /// \param slice_shape          BDHW logical shape of \p slice.
    /// \param[out] volume          3d-rfft volume inside which the slices are to be inserted.
    /// \param[out] volume_weight   Another optional varray|value associated with \p volume.
    /// \param volume_shape         BDHW logical shape of \p volume.
    /// \param[in] fwd_scaling      2x2 HW \e forward real-space scaling matrix to apply to the slices
    ///                             before the rotation. If an array is passed, it can be empty or have
    ///                             one matrix per slice. Otherwise the same scaling matrix is applied
    ///                             to every slice.
    /// \param[in] inv_rotation     3x3 DHW \e inverse rotation-matrices or quaternions to apply to the slices.
    ///                             If an array is passed, it should have one rotation per slice.
    ///                             Otherwise the same rotation is applied to every slice.
    /// \param windowed_sinc        Windowed-sinc along the normal of the slice(s).
    /// \param fftfreq_cutoff       Frequency cutoff in \p volume, in cycle/pix.
    /// \param target_shape         Actual BDHW logical shape of the 3d volume.
    /// \param ews_radius           HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                             If negative, the negative curve is computed.
    ///                             If {0,0}, the slices are projections.
    /// \warning This function computes the inverse transformation compared to the overload above using rasterization.
    template<Remap REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_interpolate_v<REMAP, Input, Output, Scale, Rotate>>>
    void insert_interpolate_3d(
            const Input& slice, const InputWeight& slice_weight, const Shape4<i64> slice_shape,
            const Output& volume, const OutputWeight& volume_weight, const Shape4<i64> volume_shape,
            const Scale& fwd_scaling,
            const Rotate& inv_rotation,
            const WindowedSinc& windowed_sinc = {},
            f32 fftfreq_cutoff = 0.5f,
            const Shape4<i64>& target_shape = {},
            const Vec2<f32>& ews_radius = {}
    ) {
        details::projection_check_parameters<details::ProjectionType::INSERT_INTERPOLATE>(
                slice, slice_weight, slice_shape, volume, volume_weight, volume_shape,
                target_shape, fwd_scaling, inv_rotation);

        const Device device = volume.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::geometry::fft::insert_interpolate_3d<REMAP>(
                        details::extract_pointer<true, true>(slice), details::extract_strides(slice),
                        details::extract_pointer<true, true>(slice_weight), details::extract_strides(slice_weight),
                        slice_shape,
                        details::extract_pointer(volume), details::extract_strides(volume),
                        details::extract_pointer(volume_weight), details::extract_strides(volume_weight),
                        volume_shape,
                        details::extract_transform(fwd_scaling),
                        details::extract_transform(inv_rotation),
                        windowed_sinc.fftfreq_sinc, windowed_sinc.fftfreq_blackman, fftfreq_cutoff,
                        target_shape, ews_radius, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr ((nt::is_texture_v<Input> && nt::is_any_v<nt::value_type_t<Input>, f64, c64>) ||
                          (nt::is_texture_v<InputWeight> && nt::is_any_v<nt::value_type_t<InputWeight>, f64, c64>)) {
                NOA_THROW("f64 and c64 are not supported by CUDA textures");
            } else {
                auto& cuda_stream = stream.cuda();
                noa::cuda::geometry::fft::insert_interpolate_3d<REMAP>(
                        details::extract_pointer<true, false>(slice), details::extract_strides(slice),
                        details::extract_pointer<true, false>(slice_weight), details::extract_strides(slice_weight),
                        slice_shape,
                        details::extract_pointer(volume), details::extract_strides(volume),
                        details::extract_pointer(volume_weight), details::extract_strides(volume_weight),
                        volume_shape,
                        details::extract_transform(fwd_scaling),
                        details::extract_transform(inv_rotation),
                        windowed_sinc.fftfreq_sinc, windowed_sinc.fftfreq_blackman, fftfreq_cutoff,
                        target_shape, ews_radius, cuda_stream);
                cuda_stream.enqueue_attach(slice, slice_weight, volume, volume_weight, fwd_scaling, inv_rotation);
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
    /// \tparam REMAP               Remapping from the volume to the slice layout. Should be HC2H or HC2HC.
    /// \tparam Input               A varray|texture of (const) f32|f64|c32|c64.
    /// \tparam InputWeight         A varray|texture of (const) f32|f64|c32|c64, or empty.
    /// \tparam Output              VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight        VArray of type f32|f64|c32|c64, or empty.
    /// \tparam Scale               Mat22 or a varray of this type.
    /// \tparam Rotate              Mat33|Quaternion, or a varray of this type.
    ///
    /// \param[in] volume           3d-centered-rfft volume from which to extract the slices.
    /// \param[in] volume_weight    Another optional varray|texture associated with \p volume.
    /// \param volume_shape         BDHW logical shape of \p volume.
    /// \param[out] slice           2d-rfft central-slice(s) to extract.
    /// \param[out] slice_weight    Another optional varray associated with \p slice.
    /// \param slice_shape          BDHW logical shape of \p slice.
    /// \param[in] inv_scaling      2x2 HW \e inverse real-space scaling to apply to the slices before the rotation.
    ///                             If an array is passed, it can be empty or have one matrix per slice.
    ///                             Otherwise the same scaling matrix is applied to every slice.
    /// \param[in] fwd_rotation     3x3 DHW \e forward rotation-matrices or quaternions to apply to the slices.
    ///                             If an array is passed, it should have one rotation per slice.
    ///                             Otherwise the same rotation is applied to every slice.
    /// \param z_windowed_sinc      Windowed-sinc along the z of \p volume.
    /// \param fftfreq_cutoff       Frequency cutoff in \p volume, in cycle/pix.
    /// \param target_shape         Actual BDHW logical shape of the 3d volume.
    /// \param ews_radius           HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                             If negative, the negative curve is computed.
    ///                             If {0,0}, the slices are projections.
    template<Remap REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_extract_v<REMAP, Input, Output, Scale, Rotate>>>
    void extract_3d(
            const Input& volume, const InputWeight& volume_weight, const Shape4<i64> volume_shape,
            const Output& slice, const OutputWeight& slice_weight, const Shape4<i64> slice_shape,
            const Scale& inv_scaling, const Rotate& fwd_rotation,
            const WindowedSinc& z_windowed_sinc = {},
            f32 fftfreq_cutoff = 0.5f,
            const Shape4<i64>& target_shape = {},
            const Vec2<f32>& ews_radius = {}
    ) {
        details::projection_check_parameters<details::ProjectionType::EXTRACT>(
                volume, volume_weight, volume_shape, slice, slice_weight, slice_shape,
                target_shape, inv_scaling, fwd_rotation);

        const Device device = volume.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::geometry::fft::extract_3d<REMAP>(
                        details::extract_pointer<true, true>(volume), details::extract_strides(volume),
                        details::extract_pointer<true, true>(volume_weight), details::extract_strides(volume_weight),
                        volume_shape,
                        details::extract_pointer(slice), details::extract_strides(slice),
                        details::extract_pointer(slice_weight), details::extract_strides(slice_weight),
                        slice_shape,
                        details::extract_transform(inv_scaling),
                        details::extract_transform(fwd_rotation),
                        z_windowed_sinc.fftfreq_sinc, z_windowed_sinc.fftfreq_blackman, fftfreq_cutoff,
                        target_shape, ews_radius, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr ((nt::is_texture_v<Input> && nt::is_any_v<nt::value_type_t<Input>, f64, c64>) ||
                          (nt::is_texture_v<InputWeight> && nt::is_any_v<nt::value_type_t<InputWeight>, f64, c64>)) {
                NOA_THROW("f64 and c64 are not supported by CUDA textures");
            } else {
                auto& cuda_stream = stream.cuda();
                noa::cuda::geometry::fft::extract_3d<REMAP>(
                        details::extract_pointer<true, false>(volume), details::extract_strides(volume),
                        details::extract_pointer<true, false>(volume_weight), details::extract_strides(volume_weight),
                        volume_shape,
                        details::extract_pointer(slice), details::extract_strides(slice),
                        details::extract_pointer(slice_weight), details::extract_strides(slice_weight),
                        slice_shape,
                        details::extract_transform(inv_scaling),
                        details::extract_transform(fwd_rotation),
                        z_windowed_sinc.fftfreq_sinc, z_windowed_sinc.fftfreq_blackman, fftfreq_cutoff,
                        target_shape, ews_radius, cuda_stream);
                cuda_stream.enqueue_attach(
                        volume, volume_weight, slice, slice_weight, inv_scaling, fwd_rotation);
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
    /// \tparam REMAP                   Remapping from the input slice to the output slice layout. Should be HC2H or HC2HC.
    /// \tparam Input                   A varray|texture of (const) f32|f64|c32|c64.
    /// \tparam InputWeight             A varray|texture of (const) f32|f64|c32|c64, or empty.
    /// \tparam Output                  VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight            VArray of type f32|f64|c32|c64, or empty.
    /// \tparam InputScale              Mat22 or a varray of this type.
    /// \tparam InputRotate             Mat33|Quaternion, or a varray of this type.
    /// \tparam OutputScale             Mat22 or a varray of this type.
    /// \tparam OutputRotate            Mat33|Quaternion, or a varray of this type.
    ///
    /// \param[in] input_slice          2d-rfft central-slice(s) to insert (can be a constant value).
    /// \param[in] input_weight         Another optional varray|texture|value associated with \p input_slice.
    /// \param input_slice_shape        BDHW logical shape of \p input_slice.
    /// \param[in,out] output_slice     2d-rfft central-slice(s) to extract. See \p add_to_output.
    /// \param[in,out] output_weight    Another optional varray|value associated with \p output_slice.
    /// \param output_slice_shape       BDHW logical shape of \p output_slice.
    /// \param[in] input_fwd_scaling    2x2 HW \e forward real-space scaling matrix to apply to the input slices
    ///                                 before the rotation. If an array is passed, it can be empty or have one matrix
    ///                                 per slice. Otherwise the same scaling matrix is applied to every slice.
    /// \param[in] input_inv_rotation   3x3 DHW \e inverse rotation-matrices or quaternions to apply to the input slices.
    ///                                 If an array is passed, it should have one rotation per slice.
    ///                                 Otherwise the same rotation is applied to every slice.
    /// \param[in] output_inv_scaling   2x2 HW \e inverse real-space scaling matrix to apply to the output slices
    ///                                 before the rotation. If an array is passed, it can be empty or have one matrix
    ///                                 per slice. Otherwise the same scaling matrix is applied to every slice.
    /// \param[in] output_fwd_rotation  3x3 DHW \e forward rotation-matrices or quaternions to apply to the output slices.
    ///                                 If an array is passed, it should have one rotation per slice.
    ///                                 Otherwise the same rotation is applied to every slice.
    /// \param input_windowed_sinc      Windowed-sinc along the normal of the input slice(s).
    /// \param z_windowed_sinc          Windowed-sinc along the z of the virtual volume.
    /// \param add_to_output            Whether the contribution of the input slices should be added to the output.
    ///                                 By default, the function sets \p output_slice. With this option enabled,
    ///                                 it instead adds the contribution of \p input_slice to the signal already in
    ///                                 \p output_slice, allowing to progressively build the output signal.
    /// \param correct_multiplicity     Correct for the multiplicity. By default, the virtual volume contains the sum
    ///                                 of the inserted values (weighted by \p input_windowed_sinc). If true, the
    ///                                 multiplicity is corrected by dividing the virtual volume with the total inserted
    ///                                 weights, only if the weight is larger than 1. Indeed, if the weight for a
    ///                                 frequency is less than 1, the frequency is unchanged. The total insertion
    ///                                 weights can be computed by filling the input slices with ones. Note that this
    ///                                 parameter is likely to only make sense if \p add_to_output is false.
    /// \param fftfreq_cutoff           Frequency cutoff of the virtual 3d volume, in cycle/pix.
    /// \param ews_radius               HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                 If negative, the negative curve is computed.
    ///                                 If {0,0}, the slices are projections.
    template<Remap REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename InputScale, typename InputRotate,
             typename OutputScale, typename OutputRotate,
             typename = std::enable_if_t<details::is_valid_insert_extract_v<
             REMAP, Input, Output, InputScale, InputRotate, OutputScale, OutputRotate>>>
    void insert_interpolate_and_extract_3d(
            const Input& input_slice, const InputWeight& input_weight, const Shape4<i64>& input_slice_shape,
            const Output& output_slice, const OutputWeight& output_weight, const Shape4<i64>& output_slice_shape,
            const InputScale& input_fwd_scaling, const InputRotate& input_inv_rotation,
            const OutputScale& output_inv_scaling, const OutputRotate& output_fwd_rotation,
            const WindowedSinc& input_windowed_sinc = {},
            const WindowedSinc& z_windowed_sinc = {},
            bool add_to_output = false,
            bool correct_multiplicity = false,
            f32 fftfreq_cutoff = 0.5f,
            const Vec2<f32>& ews_radius = {}
    ) {
        details::projection_check_parameters<details::ProjectionType::INSERT_EXTRACT>(
                input_slice, input_weight, input_slice_shape, output_slice, output_weight, output_slice_shape,
                {}, input_fwd_scaling, input_inv_rotation, output_inv_scaling, output_fwd_rotation);

        const Device device = output_slice.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::geometry::fft::insert_interpolate_and_extract_3d<REMAP>(
                        details::extract_pointer<true, true>(input_slice), details::extract_strides(input_slice),
                        details::extract_pointer<true, true>(input_weight), details::extract_strides(input_weight),
                        input_slice_shape,
                        details::extract_pointer(output_slice), details::extract_strides(output_slice),
                        details::extract_pointer(output_weight), details::extract_strides(output_weight),
                        output_slice_shape,
                        details::extract_transform(input_fwd_scaling),
                        details::extract_transform(input_inv_rotation),
                        details::extract_transform(output_inv_scaling),
                        details::extract_transform(output_fwd_rotation),
                        input_windowed_sinc.fftfreq_sinc, input_windowed_sinc.fftfreq_blackman,
                        z_windowed_sinc.fftfreq_sinc, z_windowed_sinc.fftfreq_blackman,
                        fftfreq_cutoff, add_to_output, correct_multiplicity, ews_radius, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr ((nt::is_texture_v<Input> && nt::is_any_v<nt::value_type_t<Input>, f64, c64>) ||
                          (nt::is_texture_v<InputWeight> && nt::is_any_v<nt::value_type_t<InputWeight>, f64, c64>)) {
                NOA_THROW("f64 and c64 are not supported by CUDA textures");
            } else {
                auto& cuda_stream = stream.cuda();
                noa::cuda::geometry::fft::insert_interpolate_and_extract_3d<REMAP>(
                        details::extract_pointer<true, false>(input_slice), details::extract_strides(input_slice),
                        details::extract_pointer<true, false>(input_weight), details::extract_strides(input_weight),
                        input_slice_shape,
                        details::extract_pointer(output_slice), details::extract_strides(output_slice),
                        details::extract_pointer(output_weight), details::extract_strides(output_weight),
                        output_slice_shape,
                        details::extract_transform(input_fwd_scaling),
                        details::extract_transform(input_inv_rotation),
                        details::extract_transform(output_inv_scaling),
                        details::extract_transform(output_fwd_rotation),
                        input_windowed_sinc.fftfreq_sinc, input_windowed_sinc.fftfreq_blackman,
                        z_windowed_sinc.fftfreq_sinc, z_windowed_sinc.fftfreq_blackman,
                        fftfreq_cutoff, add_to_output, correct_multiplicity, ews_radius, cuda_stream);
                cuda_stream.enqueue_attach(
                        input_slice, input_weight, output_slice, output_weight,
                        input_fwd_scaling, input_inv_rotation,
                        output_inv_scaling, output_fwd_rotation);
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Corrects for the gridding, assuming the Fourier insertion was done using tri-linear rasterization.
    /// \details During direct Fourier insertion of central-slices S into a volume B, two problems arises:
    ///          1) The insertion is not uniform (e.g. inherently more dense at low frequencies). This can be
    ///             easily corrected by inserting the data as well as its associated weights and normalizing the
    ///             inserted data with the inserted weights. This is often referred to as density or multiplicity
    ///             correction. This function is not about that.
    ///          2) The data-points can be inserted in Fourier space by rasterization, a process also called gridding,
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
