#include "noa/unified/geometry/fft/Project.h"

#include "noa/cpu/geometry/fft/Project.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Project.h"
#endif

namespace {
    using namespace ::noa;

    enum Direction { INSERT, INSERT_THICK, EXTRACT, INSERT_EXTRACT };

    // For the GPU, some matrices should be accessible so to the CPU.
    template<bool OPTIONAL, typename Matrix>
    void checkMatrix_(const char* func_name, const Matrix& matrix,
                      dim_t required_size, Device compute_device) {
        if constexpr (OPTIONAL) {
            if (matrix.empty())
                return;
        } else {
            NOA_CHECK_FUNC(func_name, !matrix.empty(), "The matrices should not be empty");
        }

        NOA_CHECK_FUNC(func_name,
                       indexing::isVector(matrix.shape()) && matrix.elements() == required_size && matrix.contiguous(),
                       "The number of matrices, specified as a contiguous vector, "
                       "should be equal to the number of slices, but got {} matrices and {} slices",
                       matrix.elements(), required_size);

        NOA_CHECK_FUNC(func_name, matrix.device() == compute_device,
                       "The transformation parameters should be on the compute device");
    }

    template<Direction DIRECTION, typename Value, typename ArrayOrTextureOrValue,
             typename Scale0, typename Rotate0,
             typename Scale1 = float22_t, typename Rotate1 = float33_t>
    void insertOrExtractCheckParameters_(const ArrayOrTextureOrValue& input, const dim4_t& input_shape,
                                         const Array<Value>& output, const dim4_t& output_shape,
                                         dim4_t target_shape,
                                         const Scale0& input_scaling_matrix,
                                         const Rotate0& input_rotation_matrix,
                                         const Scale1& output_scaling_matrix = {},
                                         const Rotate1& output_rotation_matrix = {}) {
        const char* func_name = DIRECTION >= Direction::EXTRACT ? "extract3D" : "insert3D";
        const Device output_device = output.device();

        if constexpr (traits::is_any_v<ArrayOrTextureOrValue, Array<Value>, Texture<Value>>) {
            const Device input_device = input.device();

            NOA_CHECK_FUNC(func_name, !input.empty() && !output.empty(), "Empty array detected");
            if constexpr (std::is_same_v<ArrayOrTextureOrValue, Array<Value>>)
                NOA_CHECK(!indexing::isOverlap(input, output), "Input and output arrays should not overlap");

            NOA_CHECK_FUNC(func_name, input_device == output_device,
                           "The input and output should be on the same device but got input:{} and output:{}",
                           input_device, output_device);

            NOA_CHECK_FUNC(func_name, all(input.shape() == input_shape.fft()),
                           "The shape of the non-redundant input do not match the expected shape. "
                           "Got {} and expected {}", input.shape(), input_shape.fft());
        }

        NOA_CHECK_FUNC(func_name, all(output.shape() == output_shape.fft()),
                       "The shape of the non-redundant output do not match the expected shape. Got {} and expected {}",
                       output.shape(), output_shape.fft());

        if constexpr (DIRECTION == Direction::INSERT || DIRECTION == Direction::INSERT_THICK) {
            NOA_CHECK_FUNC(func_name, input_shape[1] == 1,
                           "2D input slices are expected but got shape {}", input_shape);
            if (any(target_shape == 0)) {
                NOA_CHECK_FUNC(func_name, output_shape[0] == 1 && output_shape.ndim() == 3,
                               "A single 3D output is expected but got shape {}", output_shape);
            } else {
                NOA_CHECK_FUNC(func_name, output_shape[0] == 1 && target_shape[0] == 1 && target_shape.ndim() == 3,
                               "A single grid is expected, with a target shape describing a single 3D volume, "
                               "but got output shape {} and target shape {}", output_shape, target_shape);
            }
        } else if constexpr (DIRECTION == Direction::EXTRACT) {
            NOA_CHECK_FUNC(func_name, output_shape[1] == 1,
                           "2D input slices are expected but got shape {}", output_shape);
            if (any(target_shape == 0)) {
                NOA_CHECK_FUNC(func_name, input_shape[0] == 1 && input_shape.ndim() == 3,
                               "A single 3D input is expected but got shape {}", input_shape);
            } else {
                NOA_CHECK_FUNC(func_name, input_shape[0] == 1 && target_shape[0] == 1 && target_shape.ndim() == 3,
                               "A single grid is expected, with a target shape describing a single 3D volume, "
                               "but got input shape {} and target shape {}", input_shape, target_shape);
            }
        } else { // INSERT_EXTRACT
            NOA_CHECK_FUNC(func_name, input_shape[1] == 1 && output_shape[1] == 1,
                           "2D slices are expected but got shape input:{} and output:{}",
                           input_shape, output_shape);
        }

        const dim_t required_matrix_count = DIRECTION == Direction::EXTRACT ? output_shape[0] : input_shape[0];
        if constexpr (!traits::is_float22_v<Scale0>)
            checkMatrix_<true>(func_name, input_scaling_matrix, required_matrix_count, output_device);
        if constexpr (!traits::is_float33_v<Rotate0>)
            checkMatrix_<false>(func_name, input_rotation_matrix, required_matrix_count, output_device);

        // Only for INSERT_EXTRACT.
        if constexpr (!traits::is_float22_v<Scale1>)
            checkMatrix_<true>(func_name, output_scaling_matrix, output_shape[0], output_device);
        if constexpr (!traits::is_float33_v<Rotate1>)
            checkMatrix_<false>(func_name, output_rotation_matrix, output_shape[0], output_device);
    }

    template<typename Matrix>
    auto extractMatrix_(const Matrix& matrix) {
        using shared_matrix_t = const traits::shared_type_t<Matrix>&;
        if constexpr (traits::is_floatXX_v<Matrix>)
            return shared_matrix_t(matrix);
        else
            return shared_matrix_t(matrix.share());
    }
}

namespace noa::geometry::fft {
    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert3D(const Array<Value>& slice, dim4_t slice_shape,
                  const Array<Value>& grid, dim4_t grid_shape,
                  const Scale& inv_scaling_matrix,
                  const Rotate& fwd_rotation_matrix,
                  float cutoff,
                  dim4_t target_shape,
                  float2_t ews_radius) {
        insertOrExtractCheckParameters_<Direction::INSERT>(
                slice, slice_shape, grid, grid_shape, target_shape,
                inv_scaling_matrix, fwd_rotation_matrix);

        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::insert3D<REMAP>(
                    slice.share(), slice.strides(), slice_shape,
                    grid.share(), grid.strides(), grid_shape,
                    extractMatrix_(inv_scaling_matrix),
                    extractMatrix_(fwd_rotation_matrix),
                    cutoff, target_shape, ews_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::fft::insert3D<REMAP>(
                    slice.share(), slice.strides(), slice_shape,
                    grid.share(), grid.strides(), grid_shape,
                    extractMatrix_(inv_scaling_matrix),
                    extractMatrix_(fwd_rotation_matrix),
                    cutoff, target_shape, ews_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert3D(Value slice, dim4_t slice_shape,
                  const Array<Value>& grid, dim4_t grid_shape,
                  const Scale& inv_scaling_matrix,
                  const Rotate& fwd_rotation_matrix,
                  float cutoff,
                  dim4_t target_shape,
                  float2_t ews_radius) {
        insertOrExtractCheckParameters_<Direction::INSERT>(
                slice, slice_shape, grid, grid_shape, target_shape,
                inv_scaling_matrix, fwd_rotation_matrix);

        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::insert3D<REMAP>(
                    slice, slice_shape,
                    grid.share(), grid.strides(), grid_shape,
                    extractMatrix_(inv_scaling_matrix),
                    extractMatrix_(fwd_rotation_matrix),
                    cutoff, target_shape, ews_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::fft::insert3D<REMAP>(
                    slice, slice_shape,
                    grid.share(), grid.strides(), grid_shape,
                    extractMatrix_(inv_scaling_matrix),
                    extractMatrix_(fwd_rotation_matrix),
                    cutoff, target_shape, ews_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert3D(const Array<Value>& slice, dim4_t slice_shape,
                  const Array<Value>& grid, dim4_t grid_shape,
                  const Scale& fwd_scaling_matrix,
                  const Rotate& inv_rotation_matrix,
                  float slice_z_radius,
                  float cutoff,
                  dim4_t target_shape,
                  float2_t ews_radius) {
        insertOrExtractCheckParameters_<Direction::INSERT_THICK>(
                slice, slice_shape, grid, grid_shape, target_shape,
                fwd_scaling_matrix, inv_rotation_matrix);

        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::insert3D<REMAP>(
                    slice.share(), slice.strides(), slice_shape,
                    grid.share(), grid.strides(), grid_shape,
                    extractMatrix_(fwd_scaling_matrix),
                    extractMatrix_(inv_rotation_matrix),
                    cutoff, target_shape, ews_radius,
                    slice_z_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::fft::insert3D<REMAP>(
                    slice.share(), slice.strides(), slice_shape,
                    grid.share(), grid.strides(), grid_shape,
                    extractMatrix_(fwd_scaling_matrix),
                    extractMatrix_(inv_rotation_matrix),
                    cutoff, target_shape, ews_radius,
                    slice_z_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert3D(Value slice, dim4_t slice_shape,
                  const Array<Value>& grid, dim4_t grid_shape,
                  const Scale& fwd_scaling_matrix,
                  const Rotate& inv_rotation_matrix,
                  float slice_z_radius,
                  float cutoff,
                  dim4_t target_shape,
                  float2_t ews_radius) {
        insertOrExtractCheckParameters_<Direction::INSERT_THICK>(
                slice, slice_shape, grid, grid_shape, target_shape,
                fwd_scaling_matrix, inv_rotation_matrix);

        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::insert3D<REMAP>(
                    slice, slice_shape,
                    grid.share(), grid.strides(), grid_shape,
                    extractMatrix_(fwd_scaling_matrix),
                    extractMatrix_(inv_rotation_matrix),
                    cutoff, target_shape, ews_radius,
                    slice_z_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::fft::insert3D<REMAP>(
                    slice, slice_shape,
                    grid.share(), grid.strides(), grid_shape,
                    extractMatrix_(fwd_scaling_matrix),
                    extractMatrix_(inv_rotation_matrix),
                    cutoff, target_shape, ews_radius,
                    slice_z_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert3D(const Texture<Value>& slice, dim4_t slice_shape,
                  const Array<Value>& grid, dim4_t grid_shape,
                  const Scale& fwd_scaling_matrix,
                  const Rotate& inv_rotation_matrix,
                  float slice_z_radius,
                  float cutoff,
                  dim4_t target_shape,
                  float2_t ews_radius) {
        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = slice.cpu();
            const Array<Value> slice_array(texture.ptr, slice.shape(), texture.strides, slice.options());
            insert3D<REMAP>(slice_array, slice_shape, grid, grid_shape,
                            fwd_scaling_matrix, inv_rotation_matrix,
                            slice_z_radius, cutoff, target_shape, ews_radius);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (traits::is_any_v<Value, double, cdouble_t>) {
                NOA_THROW("Double-precision floating-points are not supported by CUDA textures");
            } else {
                insertOrExtractCheckParameters_<Direction::INSERT_THICK>(
                        slice, slice_shape, grid, grid_shape, target_shape,
                        fwd_scaling_matrix, inv_rotation_matrix);

                const cuda::Texture<Value>& texture = slice.cuda();
                cuda::geometry::fft::insert3D<REMAP>(
                        texture.array, texture.texture, slice.interp(), slice_shape,
                        grid.share(), grid.strides(), grid_shape,
                        extractMatrix_(fwd_scaling_matrix),
                        extractMatrix_(inv_rotation_matrix),
                        cutoff, target_shape, ews_radius,
                        slice_z_radius, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void extract3D(const Array<Value>& grid, dim4_t grid_shape,
                   const Array<Value>& slice, dim4_t slice_shape,
                   const Scale& inv_scaling_matrix,
                   const Rotate& fwd_rotation_matrix,
                   float cutoff,
                   dim4_t target_shape,
                   float2_t ews_radius) {
        insertOrExtractCheckParameters_<Direction::EXTRACT>(
                grid, grid_shape, slice, slice_shape, target_shape,
                inv_scaling_matrix, fwd_rotation_matrix);

        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::extract3D<REMAP>(
                    grid.share(), grid.strides(), grid_shape,
                    slice.share(), slice.strides(), slice_shape,
                    extractMatrix_(inv_scaling_matrix),
                    extractMatrix_(fwd_rotation_matrix),
                    cutoff, target_shape, ews_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::fft::extract3D<REMAP>(
                    grid.share(), grid.strides(), grid_shape,
                    slice.share(), slice.strides(), slice_shape,
                    extractMatrix_(inv_scaling_matrix),
                    extractMatrix_(fwd_rotation_matrix),
                    cutoff, target_shape, ews_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void extract3D(const Texture<Value>& grid, dim4_t grid_shape,
                   const Array<Value>& slice, dim4_t slice_shape,
                   const Scale& inv_scaling_matrix,
                   const Rotate& fwd_rotation_matrix,
                   float cutoff,
                   dim4_t target_shape,
                   float2_t ews_radius) {
        const Device device = grid.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = grid.cpu();
            const Array<Value> grid_array(texture.ptr, grid.shape(), texture.strides, grid.options());
            extract3D<REMAP>(grid_array, grid_shape, slice, slice_shape,
                             inv_scaling_matrix, fwd_rotation_matrix,
                             cutoff, target_shape, ews_radius);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (traits::is_any_v<Value, double, cdouble_t>) {
                NOA_THROW("Double-precision floating-points are not supported by CUDA textures");
            } else {
                insertOrExtractCheckParameters_<Direction::EXTRACT>(
                        grid, grid_shape, slice, slice_shape, target_shape,
                        inv_scaling_matrix, fwd_rotation_matrix);

                const cuda::Texture<Value>& texture = grid.cuda();
                cuda::geometry::fft::extract3D<REMAP>(
                        texture.array, texture.texture, grid.interp(), grid_shape,
                        slice.share(), slice.strides(), slice_shape,
                        extractMatrix_(inv_scaling_matrix),
                        extractMatrix_(fwd_rotation_matrix),
                        cutoff, target_shape, ews_radius, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename Scale0, typename Rotate0, typename Scale1, typename Rotate1, typename>
    void extract3D(const Array<Value>& input_slice, dim4_t input_slice_shape,
                   const Array<Value>& output_slice, dim4_t output_slice_shape,
                   const Scale0& input_fwd_scaling_matrix, const Rotate0& input_inv_rotation_matrix,
                   const Scale1& output_inv_scaling_matrix, const Rotate1& output_fwd_rotation_matrix,
                   float slice_z_radius,
                   float cutoff,
                   float2_t ews_radius) {
        insertOrExtractCheckParameters_<Direction::INSERT_EXTRACT>(
                input_slice, input_slice_shape, output_slice, output_slice_shape, {},
                input_fwd_scaling_matrix, input_inv_rotation_matrix,
                output_inv_scaling_matrix, output_fwd_rotation_matrix);

        const Device device = output_slice.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::extract3D<REMAP>(
                    input_slice.share(), input_slice.strides(), input_slice_shape,
                    output_slice.share(), output_slice.strides(), output_slice_shape,
                    extractMatrix_(input_fwd_scaling_matrix), extractMatrix_(input_inv_rotation_matrix),
                    extractMatrix_(output_inv_scaling_matrix), extractMatrix_(output_fwd_rotation_matrix),
                    cutoff, ews_radius, slice_z_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::fft::extract3D<REMAP>(
                    input_slice.share(), input_slice.strides(), input_slice_shape,
                    output_slice.share(), output_slice.strides(), output_slice_shape,
                    extractMatrix_(input_fwd_scaling_matrix), extractMatrix_(input_inv_rotation_matrix),
                    extractMatrix_(output_inv_scaling_matrix), extractMatrix_(output_fwd_rotation_matrix),
                    cutoff, ews_radius, slice_z_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename Scale0, typename Rotate0, typename Scale1, typename Rotate1, typename>
    void extract3D(Value input_slice, dim4_t input_slice_shape,
                   const Array<Value>& output_slice, dim4_t output_slice_shape,
                   const Scale0& input_fwd_scaling_matrix, const Rotate0& input_inv_rotation_matrix,
                   const Scale1& output_inv_scaling_matrix, const Rotate1& output_fwd_rotation_matrix,
                   float slice_z_radius,
                   float cutoff,
                   float2_t ews_radius) {
        insertOrExtractCheckParameters_<Direction::INSERT_EXTRACT>(
                input_slice, input_slice_shape, output_slice, output_slice_shape, {},
                input_fwd_scaling_matrix, input_inv_rotation_matrix,
                output_inv_scaling_matrix, output_fwd_rotation_matrix);

        const Device device = output_slice.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::extract3D<REMAP>(
                    input_slice, input_slice_shape,
                    output_slice.share(), output_slice.strides(), output_slice_shape,
                    extractMatrix_(input_fwd_scaling_matrix), extractMatrix_(input_inv_rotation_matrix),
                    extractMatrix_(output_inv_scaling_matrix), extractMatrix_(output_fwd_rotation_matrix),
                    cutoff, ews_radius, slice_z_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::fft::extract3D<REMAP>(
                    input_slice, input_slice_shape,
                    output_slice.share(), output_slice.strides(), output_slice_shape,
                    extractMatrix_(input_fwd_scaling_matrix), extractMatrix_(input_inv_rotation_matrix),
                    extractMatrix_(output_inv_scaling_matrix), extractMatrix_(output_fwd_rotation_matrix),
                    cutoff, ews_radius, slice_z_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename Scale0, typename Rotate0, typename Scale1, typename Rotate1, typename>
    void extract3D(const Texture<Value>& input_slice, dim4_t input_slice_shape,
                   const Array<Value>& output_slice, dim4_t output_slice_shape,
                   const Scale0& input_fwd_scaling_matrix, const Rotate0& input_inv_rotation_matrix,
                   const Scale1& output_inv_scaling_matrix, const Rotate1& output_fwd_rotation_matrix,
                   float slice_z_radius,
                   float cutoff,
                   float2_t ews_radius) {
        const Device device = output_slice.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = input_slice.cpu();
            const Array<Value> tmp(texture.ptr, input_slice.shape(), texture.strides, input_slice.options());
            extract3D<REMAP>(tmp, input_slice_shape, output_slice, output_slice_shape,
                             input_fwd_scaling_matrix, input_inv_rotation_matrix,
                             output_inv_scaling_matrix, output_fwd_rotation_matrix,
                             slice_z_radius, cutoff, ews_radius);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (traits::is_any_v<Value, double, cdouble_t>) {
                NOA_THROW("Double-precision floating-points are not supported by CUDA textures");
            } else {
                insertOrExtractCheckParameters_<Direction::INSERT_EXTRACT>(
                        input_slice, input_slice_shape, output_slice, output_slice_shape, {},
                        input_fwd_scaling_matrix, input_inv_rotation_matrix,
                        output_inv_scaling_matrix, output_fwd_rotation_matrix);

                const cuda::Texture<Value>& texture = input_slice.cuda();
                cuda::geometry::fft::extract3D<REMAP>(
                        texture.array, texture.texture, input_slice.interp(), input_slice_shape,
                        output_slice.share(), output_slice.strides(), output_slice_shape,
                        extractMatrix_(input_fwd_scaling_matrix), extractMatrix_(input_inv_rotation_matrix),
                        extractMatrix_(output_inv_scaling_matrix), extractMatrix_(output_fwd_rotation_matrix),
                        cutoff, ews_radius, slice_z_radius, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename>
    void griddingCorrection(const Array<Value>& input, const Array<Value>& output, bool post_correction) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");

        dim4_t input_strides = input.strides();
        if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::griddingCorrection(
                    input.share(), input_strides,
                    output.share(), output.strides(),
                    output.shape(), post_correction, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::fft::griddingCorrection(
                    input.share(), input_strides,
                    output.share(), output.strides(),
                    output.shape(), post_correction, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    #define NOA_INSTANTIATE_INSERT_(T, REMAP, S, R)         \
    template void insert3D<REMAP, T, S, R, void>(           \
        const Array<T>&, dim4_t, const Array<T>&, dim4_t,   \
        const S&, const R&, float, dim4_t, float2_t);       \
    template void insert3D<REMAP, T, S, R, void>(           \
        T, dim4_t, const Array<T>&, dim4_t,                 \
        const S&, const R&, float, dim4_t, float2_t)

    #define NOA_INSTANTIATE_INSERT_THICK_(T, REMAP, S, R)   \
    template void insert3D<REMAP, T, S, R, void>(           \
        const Array<T>&, dim4_t, const Array<T>&, dim4_t,   \
        const S&, const R&, float, float, dim4_t, float2_t);\
    template void insert3D<REMAP, T, S, R, void>(           \
        T, dim4_t, const Array<T>&, dim4_t,                 \
        const S&, const R&, float, float, dim4_t, float2_t);\
    template void insert3D<REMAP, T, S, R, void>(           \
        const Texture<T>&, dim4_t, const Array<T>&, dim4_t, \
        const S&, const R&, float, float, dim4_t, float2_t)

    #define NOA_INSTANTIATE_EXTRACT_(T, REMAP, S, R)        \
    template void extract3D<REMAP, T, S, R, void>(          \
        const Array<T>&, dim4_t, const Array<T>&, dim4_t,   \
        const S&, const R&, float, dim4_t, float2_t);       \
    template void extract3D<REMAP, T, S, R, void>(          \
        const Texture<T>&, dim4_t, const Array<T>&, dim4_t, \
        const S&, const R&, float, dim4_t, float2_t)

    #define NOA_INSTANTIATE_INSERT_EXTRACT_(T, REMAP, S0, S1, R0, R1)   \
    template void extract3D<REMAP, T, S0, R0, S1, R1, void>(            \
        const Array<T>&, dim4_t, const Array<T>&, dim4_t,               \
        const S0&, const R0&, const S1&, const R1&,                     \
        float, float, float2_t);                                        \
    template void extract3D<REMAP, T, S0, R0, S1, R1, void>(            \
        T, dim4_t, const Array<T>&, dim4_t,                             \
        const S0&, const R0&, const S1&, const R1&,                     \
        float, float, float2_t);                                        \
    template void extract3D<REMAP, T, S0, R0, S1, R1, void>(            \
        const Texture<T>&, dim4_t, const Array<T>&, dim4_t,             \
        const S0&, const R0&, const S1&, const R1&,                     \
        float, float, float2_t)

    #define NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, S, R)      \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2H, S, R);           \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2HC, S, R);          \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2H, S, R);          \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2HC, S, R);         \
    NOA_INSTANTIATE_INSERT_THICK_(T, Remap::HC2H, S, R);    \
    NOA_INSTANTIATE_INSERT_THICK_(T, Remap::HC2HC, S, R);   \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2H, S, R);         \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2HC, S, R)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, S0, S1, R0, R1)  \
    NOA_INSTANTIATE_INSERT_EXTRACT_(T, Remap::HC2H, S0, S1, R0, R1);    \
    NOA_INSTANTIATE_INSERT_EXTRACT_(T, Remap::HC2HC, S0, S1, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, R0, R1)                              \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, float22_t, float22_t, R0, R1);               \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, Array<float22_t>, float22_t, R0, R1);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, float22_t, Array<float22_t>, R0, R1);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, Array<float22_t>, Array<float22_t>, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(T)                             \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, float33_t, float33_t);               \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, Array<float33_t>, float33_t);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, float33_t, Array<float33_t>);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, Array<float33_t>, Array<float33_t>)

    #define NOA_INSTANTIATE_PROJECT_ALL(T)                                  \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, float22_t, float33_t);             \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, Array<float22_t>, float33_t); \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, float22_t, Array<float33_t>); \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, Array<float22_t>, Array<float33_t>);\
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(T)

    NOA_INSTANTIATE_PROJECT_ALL(float);
    NOA_INSTANTIATE_PROJECT_ALL(cfloat_t);
    NOA_INSTANTIATE_PROJECT_ALL(double);
    NOA_INSTANTIATE_PROJECT_ALL(cdouble_t);

    template void griddingCorrection<float, void>(const Array<float>&, const Array<float>&, bool);
    template void griddingCorrection<double, void>(const Array<double>&, const Array<double>&, bool);
}
