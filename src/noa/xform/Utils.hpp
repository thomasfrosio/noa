#pragma once

#include "noa/xform/core/Interpolation.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/xform/cuda/Allocators.hpp"
#include "noa/xform/cuda/Texture.cuh"
#endif

namespace noa::xform::details {
    /// Returns the input type for the Interpolator given an array or texture.
    /// For GPU textures, EXTRACT_GPU_TEXTURE must be true, otherwise a runtime error is raised.
    /// Automatically broadcasts the batch dimension if the input batch is 1.
    template<usize N, typename Index, typename Coord, Interp INTERP, Border BORDER, bool EXTRACT_GPU_TEXTURE, nt::array_or_texture T>
    constexpr auto to_interpolator_input(const T& input) {
        if constexpr (nt::texture<T> and EXTRACT_GPU_TEXTURE) {
            check(input.device().is_gpu(), "Trying to construct a GPU texture from is a CPU texture");
            #ifdef NOA_ENABLE_CUDA
            using texture_t = noa::xform::cuda::AllocatorTexture::texture_type<N, INTERP, BORDER, nt::value_type_t<T>, Coord, Index>;
            return texture_t(*input.gpu().handle->texture, input.shape().template filter_nd<N>());
            #else
            panic_no_gpu_backend();
            #endif
        } else if constexpr (nt::array_or_texture<T>) {
            if constexpr (nt::texture<T>)
                check(input.device().is_cpu(), "Trying to construct a CPU accessor from a GPU texture");
            auto view = input.view().template as_nd<N + 1>();
            auto strides = view.strides().template as_safe<Index>();
            if (view.shape()[0] == 1)
                strides[0] = 0;
            return Accessor<nt::const_value_type_t<T>, N + 1, Index>(input.get(), strides);
        } else {
            static_assert(nt::always_false<T>);
        }
    }

    /// The "fast" interpolation methods are only used in certain conditions.
    /// We could leave it as such, the interpolator would generate the correct code, but we would
    /// instantiate more kernels than necessary, so erase the "fast" when it's not relevant.
    template<Interp INTERP, Border BORDER, bool EXTRACT_GPU_TEXTURE, nt::array_or_texture T>
    consteval auto to_interpolator_interp() {
        return EXTRACT_GPU_TEXTURE and nt::texture<T> and
            BORDER.is_any(Border::MIRROR, Border::PERIODIC, Border::ZERO, Border::CLAMP) and
            INTERP.is_any(Interp::NEAREST_FAST, Interp::LINEAR_FAST, Interp::CUBIC_BSPLINE_FAST)
            ? INTERP : INTERP.erase_fast();
    }

    /// Creates an Interpolator.
    /// Should be N=3 (2 batched), or N=4 (3d batched).
    template<usize N, Interp INTERP, Border BORDER, typename Index, typename Coord, bool EXTRACT_GPU_TEXTURE, nt::array_or_texture T>
    constexpr auto to_interpolator(const T& input, nt::mutable_value_type_t<T> cvalue = {}) {
        namespace nxd = ::noa::xform::details;
        constexpr Interp INTERP_ = nxd::to_interpolator_interp<INTERP, BORDER, EXTRACT_GPU_TEXTURE, T>();
        auto interp_input = nxd::to_interpolator_input<N, Index, Coord, INTERP_, BORDER, EXTRACT_GPU_TEXTURE>(input);
        return Interpolator<N, INTERP_, BORDER, decltype(interp_input)>(
            interp_input, input.shape().template filter_nd<N>().pop_front().template as<Index>(), cvalue);
    }

    /// Creates an InterpolatorSpectrum.
    template<usize N, nf::Layout REMAP, Interp INTERP, typename Coord, bool EXTRACT_GPU_TEXTURE, nt::array_or_texture T, typename Index>
    constexpr auto to_interpolator_spectrum(const T& input, const Shape<Index, 4>& logical_shape) {
        namespace nxd = ::noa::xform::details;
        constexpr Interp INTERP_ = nxd::to_interpolator_interp<INTERP, Border::ZERO, EXTRACT_GPU_TEXTURE, T>();
        auto interp_input = nxd::to_interpolator_input<N, Index, Coord, INTERP_, Border::ZERO, EXTRACT_GPU_TEXTURE>(input);
        return InterpolatorSpectrum<N, REMAP, INTERP_, decltype(interp_input)>(
            interp_input, logical_shape.template filter_nd<N>().pop_front().template as<Index>());
    }

    // nvcc struggles with C++20 template parameters in lambda, so use a worse C++17 syntax and create this type...
    template<Interp VALUE>
    struct WrapInterp {
        consteval auto operator()() const -> Interp { return VALUE; }
    };
    template<Border VALUE>
    struct WrapBorder {
        consteval auto operator()() const -> Border { return VALUE; }
    };
}
