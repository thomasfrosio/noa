#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/geometry/Symmetry.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::geometry::fft::details {
    using namespace ::noa::fft;

    template<int32_t NDIM, typename Matrix, typename Shift>
    constexpr bool is_valid_matrix_and_shift_v =
            (NDIM == 2 &&
             traits::is_any_v<Matrix, float22_t, shared_t<float22_t[]>> &&
             traits::is_any_v<Shift, float2_t, shared_t<float2_t[]>>) ||
            (NDIM == 3 &&
             traits::is_any_v<Matrix, float33_t, shared_t<float33_t[]>> &&
             traits::is_any_v<Shift, float3_t, shared_t<float3_t[]>>);

    template<int NDIM, Remap REMAP, typename Value, typename Matrix, typename Shift>
    constexpr bool is_valid_transform_v =
            traits::is_any_v<Value, float, cfloat_t, double, cdouble_t> &&
            (REMAP == HC2HC || REMAP == HC2H) &&
            is_valid_matrix_and_shift_v<NDIM, Matrix, Shift>;

    template<int NDIM, Remap REMAP, typename Value, typename Matrix, typename Shift>
    constexpr bool is_valid_transform_texture_v =
            traits::is_any_v<Value, float, cfloat_t> &&
            (REMAP == HC2HC || REMAP == HC2H) &&
            is_valid_matrix_and_shift_v<NDIM, Matrix, Shift>;

    template<Remap REMAP, typename Value>
    constexpr bool is_valid_transform_sym_v =
            traits::is_any_v<Value, float, cfloat_t, double, cdouble_t> &&
            (REMAP == HC2HC || REMAP == HC2H);

    template<Remap REMAP, typename Value>
    constexpr bool is_valid_transform_sym_texture_v =
            traits::is_any_v<Value, float, cfloat_t> &&
            (REMAP == HC2HC || REMAP == HC2H);
}

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    // Rotates/scales a non-redundant 2D (batched) FFT.
    template<Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_v<2, REMAP, Value, Matrix, Shift>>>
    void transform2D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream);

    // Rotates/scales a non-redundant 3D (batched) FFT.
    template<Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_v<3, REMAP, Value, Matrix, Shift>>>
    void transform3D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream);

    // Rotates/scales a non-redundant 2D (batched) FFT.
    // Input texture should be bound to a CUDA array, with un-normalized coordinates.
    template<Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_texture_v<2, REMAP, Value, Matrix, Shift>>>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& shifts, float cutoff, Stream& stream);

    // Rotates/scales a non-redundant 3D (batched) FFT.
    // Input texture should be bound to a CUDA array, with un-normalized coordinates.
    template<Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_texture_v<3, REMAP, Value, Matrix, Shift>>>
    void transform3D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& shifts, float cutoff, Stream& stream);
}

namespace noa::cuda::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    // Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    template<Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void transform2D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     float22_t inv_matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream);

    // Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    template<Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void transform3D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     float33_t inv_matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream);

    // Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    // Input texture should be bound to a CUDA array, with un-normalized coordinates.
    template<Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_texture_v<REMAP, Value>>>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     float22_t inv_matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, bool normalize, Stream& stream);

    // Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    // Input texture should be bound to a CUDA array, with un-normalized coordinates.
    template<Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_texture_v<REMAP, Value>>>
    void transform3D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     float33_t inv_matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, bool normalize, Stream& stream);

    // Symmetrizes a non-redundant 2D (batched) FFT.
    template<Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void symmetrize2D(const shared_t<Value[]>& input, dim4_t input_strides,
                      const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                      const Symmetry& symmetry, float2_t shift,
                      float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        transform2D<REMAP>(input, input_strides, output, output_strides, shape, float22_t{}, symmetry,
                           shift, cutoff, interp_mode, normalize, stream);
    }

    // Symmetrizes a non-redundant 3D (batched) FFT.
    template<Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void symmetrize3D(const shared_t<Value[]>& input, dim4_t input_strides,
                      const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                      const Symmetry& symmetry, float3_t shift,
                      float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        transform3D<REMAP>(input, input_strides, output, output_strides, shape, float33_t{}, symmetry,
                           shift, cutoff, interp_mode, normalize, stream);
    }
}
