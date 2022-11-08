#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/geometry/Symmetry.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::geometry::fft::details {
    using namespace ::noa::fft;

    template<int NDIM, Remap REMAP, typename T, typename M, typename S>
    constexpr bool is_valid_transform_v =
            traits::is_any_v<T, float, cfloat_t> && (REMAP == HC2HC || REMAP == HC2H) &&
            ((NDIM == 2 && traits::is_any_v<M, float22_t, shared_t<float22_t[]>> && traits::is_any_v<S, float2_t, shared_t<float2_t[]>>) ||
             (NDIM == 3 && traits::is_any_v<M, float33_t, shared_t<float33_t[]>> && traits::is_any_v<S, float3_t, shared_t<float3_t[]>>));

    template<Remap REMAP, typename T>
    constexpr bool is_valid_transform_sym_v =
            traits::is_any_v<T, float, cfloat_t> && (REMAP == HC2HC || REMAP == HC2H);
}

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    // Rotates/scales a non-redundant 2D (batched) FFT.
    template<Remap REMAP, typename T, typename M, typename S,
             typename = std::enable_if_t<details::is_valid_transform_v<2, REMAP, T, M, S>>>
    void transform2D(const shared_t<T[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream);

    // Rotates/scales a non-redundant 2D (batched) FFT.
    // Input texture should be bound to a CUDA array, with un-normalized coordinates.
    template<Remap REMAP, typename T, typename M, typename S,
             typename = std::enable_if_t<details::is_valid_transform_v<2, REMAP, T, M, S>>>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     const M& matrices, const S& shifts, float cutoff, Stream& stream);

    // Rotates/scales a non-redundant 3D (batched) FFT.
    template<Remap REMAP, typename T, typename M, typename S,
             typename = std::enable_if_t<details::is_valid_transform_v<3, REMAP, T, M, S>>>
    void transform3D(const shared_t<T[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream);

    // Rotates/scales a non-redundant 3D (batched) FFT.
    // Input texture should be bound to a CUDA array, with un-normalized coordinates.
    template<Remap REMAP, typename T, typename M, typename S,
             typename = std::enable_if_t<details::is_valid_transform_v<3, REMAP, T, M, S>>>
    void transform3D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     const M& matrices, const S& shifts, float cutoff, Stream& stream);
}

namespace noa::cuda::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    // Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, T>>>
    void transform2D(const shared_t<T[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream);

    // Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    // Input texture should be bound to a CUDA array, with un-normalized coordinates.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, T>>>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, bool normalize, Stream& stream);

    // Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, T>>>
    void transform3D(const shared_t<T[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream);

    // Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    // Input texture should be bound to a CUDA array, with un-normalized coordinates.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, T>>>
    void transform3D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, bool normalize, Stream& stream);

    // Symmetrizes a non-redundant 2D (batched) FFT.
    // TODO ADD TESTS!
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, T>>>
    void symmetrize2D(const shared_t<T[]>& input, dim4_t input_strides,
                      const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                      const Symmetry& symmetry, float2_t shift,
                      float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        transform2D<REMAP>(input, input_strides, output, output_strides, shape, float22_t{}, symmetry,
                           shift, cutoff, interp_mode, normalize, stream);
    }

    // Symmetrizes a non-redundant 3D (batched) FFT.
    // TODO ADD TESTS!
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, T>>>
    void symmetrize3D(const shared_t<T[]>& input, dim4_t input_strides,
                      const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                      const Symmetry& symmetry, float3_t shift,
                      float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        transform3D<REMAP>(input, input_strides, output, output_strides, shape, float33_t{}, symmetry,
                           shift, cutoff, interp_mode, normalize, stream);
    }
}
