#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/geometry/Symmetry.hpp"

namespace noa::cpu::geometry::fft::details {
    template<i32 NDIM, noa::fft::Remap REMAP, typename Value, typename Matrix, typename Shift>
    constexpr bool is_valid_transform_v =
            noa::traits::is_any_v<Value, f32, f64, c32, c64> &&
            (REMAP == noa::fft::HC2HC || REMAP == noa::fft::HC2H) &&
            ((NDIM == 2 &&
              noa::traits::is_any_v<Matrix, Float22, const Float22*> &&
              noa::traits::is_any_v<Shift, Vec2<f32>, const Vec2<f32>*>) ||
             (NDIM == 3 &&
              noa::traits::is_any_v<Matrix, Float33, const Float33*> &&
              noa::traits::is_any_v<Shift, Vec3<f32>, const Vec3<f32>*>));

    template<noa::fft::Remap REMAP, typename Value>
    constexpr bool is_valid_transform_sym_v =
            noa::traits::is_any_v<Value, f32, f64, c32, c64> &&
            (REMAP == noa::fft::HC2HC || REMAP == noa::fft::HC2H);
}

namespace noa::cpu::geometry::fft {
    // Rotates/scales a non-redundant 2D (batched) FFT.
    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_v<2, REMAP, Value, Matrix, Shift>>>
    void transform_2d(const Value* input, const Strides4<i64>& input_strides,
                      Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                      const Matrix& inv_matrices, const Shift& shifts,
                      f32 cutoff, InterpMode interp_mode, i64 threads);

    // Rotates/scales a non-redundant 3D (batched) FFT.
    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_v<3, REMAP, Value, Matrix, Shift>>>
    void transform_3d(const Value* input, const Strides4<i64>& input_strides,
                      Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                      const Matrix& inv_matrices, const Shift& shifts,
                      f32 cutoff, InterpMode interp_mode, i64 threads);
}

namespace noa::cpu::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    // Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    // TODO ADD TESTS!
    template<noa::fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void transform_and_symmetrize_2d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float22& inv_matrix, const Symmetry& symmetry, const Vec2<f32>& shift,
            f32 cutoff, InterpMode interp_mode, bool normalize, i64 threads);

    // Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    // TODO ADD TESTS!
    template<noa::fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void transform_and_symmetrize_3d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float33& inv_matrix, const Symmetry& symmetry, const Vec3<f32>& shift,
            f32 cutoff, InterpMode interp_mode, bool normalize, i64 threads);

    // Symmetrizes a non-redundant 2D (batched) FFT.
    // TODO ADD TESTS!
    template<noa::fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void symmetrize_2d(const Value* input, const Strides4<i64>& input_strides,
                       Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                       const Symmetry& symmetry, const Vec2<f32>& shift,
                       f32 cutoff, InterpMode interp_mode, bool normalize, i64 threads) {
        transform_2d<REMAP>(input, input_strides, output, output_strides, shape, Float22{}, symmetry,
                            shift, cutoff, interp_mode, normalize, threads);
    }

    // Symmetrizes a non-redundant 3D (batched) FFT.
    // TODO ADD TESTS!
    template<noa::fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void symmetrize_3d(const Value* input, const Strides4<i64>& input_strides,
                       Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                       const Symmetry& symmetry, const Vec3<f32>& shift,
                       f32 cutoff, InterpMode interp_mode, bool normalize, i64 threads) {
        transform_3d<REMAP>(input, input_strides, output, output_strides, shape, Float33{}, symmetry,
                            shift, cutoff, interp_mode, normalize, threads);
    }
}
