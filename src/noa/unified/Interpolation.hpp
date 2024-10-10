#pragma once

#include "noa/core/Interpolation.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Shape.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Texture.hpp"
#endif

namespace noa::guts {
    /// Returns the input type for the Interpolator.
    /// - Supports View, Array and Texture.
    ///   For Texture, use IS_GPU to extract the GPU texture; otherwise it uses the underlying CPU array.
    /// - Automatically broadcasts the batch dimension if the input batch is 1.
    template<size_t N, typename Index, typename Coord, Interp INTERP, Border BORDER, bool IS_GPU, nt::varray_or_texture T>
    constexpr auto to_interpolator_input(const T& input) {
        if constexpr (nt::texture<T> and IS_GPU) {
            #ifdef NOA_ENABLE_CUDA
            using texture_t = noa::cuda::interpolation_to_texture_t<N, INTERP, BORDER, nt::value_type_t<T>, Coord>;
            return texture_t(input.gpu()->texture, input.shape().template filter_nd<N>());
            #else
            std::terminate(); // unreachable
            #endif
        } else if constexpr (nt::varray_or_texture<T>) {
            auto strides = input.strides().template filter_nd<N>();
            if (input.shape()[0] == 1)
                strides[0] = 0;
            return Accessor<nt::const_value_type_t<T>, N + 1, Index>(input.get(), strides.template as_safe<Index>());
        } else {
            static_assert(nt::always_false<T>);
        }
    }

    /// Creates an Interpolator.
    template<size_t N, Interp INTERP, Border BORDER, typename Index, typename Coord, bool IS_GPU, nt::varray_or_texture T>
    constexpr auto to_interpolator(const T& input, nt::mutable_value_type_t<T> cvalue = {}) {
        constexpr Interp INTERP_ = not IS_GPU ? INTERP.erase_fast() : INTERP;

        auto interp_input = ng::to_interpolator_input<N, Index, Coord, INTERP_, BORDER, IS_GPU>(input);
        return Interpolator<N, INTERP_, BORDER, decltype(interp_input)>(
            interp_input, input.shape().template filter_nd<N>().pop_front().template as<Index>(), cvalue);
    }

    /// Creates an InterpolatorSpectrum.
    template<size_t N, Remap REMAP, Interp INTERP, typename Coord, bool IS_GPU, nt::varray_or_texture T, typename Index>
    constexpr auto to_interpolator_spectrum(const T& input, const Shape<Index, 4>& logical_shape) {
        constexpr Interp INTERP_ = not IS_GPU ? INTERP.erase_fast() : INTERP;

        auto interp_input = ng::to_interpolator_input<N, Index, Coord, INTERP_, Border::ZERO, IS_GPU>(input);
        return InterpolatorSpectrum<N, REMAP, INTERP_, decltype(interp_input)>(
            interp_input, logical_shape.template filter_nd<N>().pop_front().template as<Index>());
    }
}
