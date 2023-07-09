#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/geometry/Symmetry.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::geometry::fft::details {
    using namespace ::noa::fft;

    template<i32 NDIM, typename Matrix, typename Shift>
    constexpr bool is_valid_matrix_and_shift_v =
            (NDIM == 2 &&
             noa::traits::is_any_v<Matrix, Float22, const Float22*> &&
             noa::traits::is_any_v<Shift, Vec2<f32>, const Vec2<f32>*>) ||
            (NDIM == 3 &&
             noa::traits::is_any_v<Matrix, Float33, const Float33*> &&
             noa::traits::is_any_v<Shift, Vec3<f32>, const Vec3<f32>*>);

    template<i32 NDIM, Remap REMAP, typename Value, typename Matrix, typename Shift>
    constexpr bool is_valid_transform_v =
            noa::traits::is_any_v<Value, f32, f64, c32, c64> &&
            (REMAP == HC2HC || REMAP == HC2H) &&
            is_valid_matrix_and_shift_v<NDIM, Matrix, Shift>;

    template<i32 NDIM, Remap REMAP, typename Value, typename Matrix, typename Shift>
    constexpr bool is_valid_transform_texture_v =
            noa::traits::is_any_v<Value, f32, c32> &&
            (REMAP == HC2HC || REMAP == HC2H) &&
            is_valid_matrix_and_shift_v<NDIM, Matrix, Shift>;

    template<Remap REMAP, typename Value>
    constexpr bool is_valid_transform_sym_v =
            noa::traits::is_any_v<Value, f32, f64, c32, c64> &&
            (REMAP == HC2HC || REMAP == HC2H);

    template<Remap REMAP, typename Value>
    constexpr bool is_valid_transform_sym_texture_v =
            noa::traits::is_any_v<Value, f32, c32> &&
            (REMAP == HC2HC || REMAP == HC2H);
}

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_v<2, REMAP, Value, Matrix, Shift>>>
    void transform_2d(const Value* input, const Strides4<i64>& input_strides,
                      Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                      const Matrix& inv_matrices, const Shift& shifts,
                      f32 cutoff, InterpMode interp_mode, Stream& stream);

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_v<3, REMAP, Value, Matrix, Shift>>>
    void transform_3d(const Value* input, const Strides4<i64>& input_strides,
                      Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                      const Matrix& inv_matrices, const Shift& shifts,
                      f32 cutoff, InterpMode interp_mode, Stream& stream);

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_texture_v<2, REMAP, Value, Matrix, Shift>>>
    void transform_2d(cudaArray* array, cudaTextureObject_t texture, InterpMode texture_interp_mode,
                      Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                      const Matrix& inv_matrices, const Shift& shifts, f32 cutoff, Stream& stream);

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_texture_v<3, REMAP, Value, Matrix, Shift>>>
    void transform_3d(cudaArray* array, cudaTextureObject_t texture, InterpMode texture_interp_mode,
                      Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                      const Matrix& inv_matrices, const Shift& shifts, f32 cutoff, Stream& stream);
}

namespace noa::cuda::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    template<noa::fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void transform_and_symmetrize_2d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float22& inv_matrix, const Symmetry& symmetry, const Vec2<f32>& shift,
            f32 cutoff, InterpMode interp_mode, bool normalize, Stream& stream);

    template<noa::fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void transform_and_symmetrize_3d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float33& inv_matrix, const Symmetry& symmetry, const Vec3<f32>& shift,
            f32 cutoff, InterpMode interp_mode, bool normalize, Stream& stream);

    template<noa::fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_texture_v<REMAP, Value>>>
    void transform_and_symmetrize_2d(
            cudaArray* array, cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float22& inv_matrix, const Symmetry& symmetry, const Vec2<f32>& shift,
            f32 cutoff, bool normalize, Stream& stream);

    template<noa::fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_texture_v<REMAP, Value>>>
    void transform_and_symmetrize_3d(
            cudaArray* array, cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float33& inv_matrix, const Symmetry& symmetry, const Vec3<f32>& shift,
            f32 cutoff, bool normalize, Stream& stream);

    template<noa::fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void symmetrize_2d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Symmetry& symmetry, const Vec2<f32>& shift,
            f32 cutoff, InterpMode interp_mode, bool normalize, Stream& stream
    ) {
        transform_2d<REMAP>(input, input_strides, output, output_strides, shape, Float22{}, symmetry,
                            shift, cutoff, interp_mode, normalize, stream);
    }

    template<noa::fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void symmetrize_3d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Symmetry& symmetry, const Vec3<f32>& shift,
            f32 cutoff, InterpMode interp_mode, bool normalize, Stream& stream
    ) {
        transform_3d<REMAP>(input, input_strides, output, output_strides, shape, Float33{}, symmetry,
                            shift, cutoff, interp_mode, normalize, stream);
    }
}
