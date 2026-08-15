#pragma once

#include "noa/xform/core/Interpolation.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/xform/cuda/Allocators.hpp"
#include "noa/xform/cuda/Texture.cuh"
#endif

namespace noa::xform::details {
    /// Returns the input type for interpolation given an array or texture.
    /// For GPU textures, EXTRACT_GPU_TEXTURE must be true, otherwise a runtime error is raised.
    /// Automatically broadcasts the batch axes if the corresponding input axes are 1.
    template<usize R, typename Index, typename Coord, Interp INTERP, Border BORDER, bool EXTRACT_GPU_TEXTURE, nt::array_or_texture T>
    constexpr auto to_interpolator_input(const T& input) {
        constexpr usize N = T::SIZE;
        constexpr usize B = N > R ? N - R : 0;
        auto [shape_b, shape_r] = input.shape().template as_safe<Index>().template split<B>();
        auto [strides_b, strides_r] = input.strides().template as_safe<Index>().template split<B>();

        // Automatically broadcast the batch axes.
        if constexpr (B > 0) {
            for (usize i = 0; i < B; ++i) {
                if (shape_b[i] == 1)
                    strides_b[i] = 0;
            }
        }

        using value_t = nt::const_value_type_t<T>;
        if constexpr (nt::texture<T> and EXTRACT_GPU_TEXTURE) {
            #ifdef NOA_ENABLE_CUDA
            check(input.device().is_gpu(), "Trying to construct a GPU texture from is a CPU texture");
            if constexpr (B > 0) {
                NOA_ASSERT(strides_r == 0);
            }
            using texture_t = noa::xform::cuda::AllocatorTexture::accessor_texture_type<INTERP, BORDER, value_t, Coord, Index, B, R>;
            return texture_t(input.gpu().data()->texture, shape_r, strides_b);
            #else
            panic_no_gpu_backend();
            #endif
        } else if constexpr (nt::array_or_texture<T>) {
            value_t* ptr;
            if constexpr (nt::texture<T>) {
                check(input.device().is_cpu(), "Trying to construct a CPU accessor from a GPU texture");
                ptr = input.cpu().data();
            } else {
                ptr = input.data();
            }
            auto strides = strides_b.push_back(strides_r.vec);
            return Accessor<value_t, N, Index>(ptr, strides);
        } else {
            static_assert(nt::always_false<T>);
        }
    }

    /// The "fast" interpolation methods are only used in certain conditions.
    /// We could leave it as such, the interpolator would generate the correct code, but we would
    /// instantiate more kernels than necessary, so erase the "fast" when it's not relevant.
    template<Interp INTERP, Border BORDER, bool IS_TEXTURE, bool EXTRACT_GPU_TEXTURE>
    consteval auto to_interpolator_interp() {
        return EXTRACT_GPU_TEXTURE and IS_TEXTURE and
            BORDER.is_any(Border::MIRROR, Border::PERIODIC, Border::ZERO, Border::CLAMP) and
            INTERP.is_any(Interp::NEAREST_FAST, Interp::LINEAR_FAST, Interp::CUBIC_BSPLINE_FAST)
            ? INTERP : INTERP.erase_fast();
    }

    template<typename Interpolator, typename InputAccessor>
    struct ToInterpolatorResult {
        using accessor_type = InputAccessor;
        using interpolator_type = Interpolator;

        accessor_type accessor;
        interpolator_type interpolator;
    };

    template<usize R, Interp INTERP, Border BORDER, bool IS_GPU, typename Index, typename Coord, bool NO_SHAPE, typename T>
    constexpr auto prepare_interpolation_inputs(const T& input, nt::mutable_value_type_t<T> cvalue = {}) {
        constexpr Interp INTERP_ = to_interpolator_interp<INTERP, BORDER, nt::texture_decay<T>, IS_GPU>();
        constexpr usize N = nt::array_size_v<T>;
        constexpr usize B = N > R ? N - R : 0;
        using accessor_t = decltype(to_interpolator_input<R, Index, Coord, INTERP_, BORDER, IS_GPU>(input));
        using accessor_r_t = std::decay_t<decltype(std::declval<const accessor_t&>()[Vec<i32, B>{}])>;
        using const_value_t = nt::const_value_type_t<T>;
        constexpr bool TEXTURE_ONLY = nt::textureable_nd<accessor_r_t, BORDER, R>;
        constexpr bool NO_SHAPE_ = NO_SHAPE ? true : TEXTURE_ONLY;

        return ToInterpolatorResult{
            .accessor = to_interpolator_input<R, Index, Coord, INTERP_, BORDER, IS_GPU>(input),
            .interpolator = Interpolator<INTERP_, BORDER, const_value_t, R, Index, NO_SHAPE_, TEXTURE_ONLY>(
                input.shape().template pop_front<B>().template as<Index>(), cvalue),
        };
    }

    template<usize R, nf::Layout REMAP, Interp INTERP, bool IS_GPU, typename Coord, bool NO_SHAPE, typename T, typename Index>
    constexpr auto prepare_interpolation_spectrum_inputs(const T& input, const Shape<Index, R>& logical_shape) {
        constexpr Interp INTERP_ = to_interpolator_interp<INTERP, Border::ZERO, nt::texture_decay<T>, IS_GPU>();
        constexpr usize N = nt::array_size_v<T>;
        constexpr usize B = N > R ? N - R : 0;
        using accessor_t = decltype(to_interpolator_input<R, Index, Coord, INTERP_, Border::ZERO, IS_GPU>(input));
        using accessor_r_t = std::decay_t<decltype(std::declval<const accessor_t&>()[Vec<i32, B>{}])>;
        using const_value_t = nt::const_value_type_t<T>;
        constexpr bool TEXTURE_ONLY = nt::textureable_nd<accessor_r_t, Border::ZERO, R>;

        return ToInterpolatorResult{
            .accessor = to_interpolator_input<R, Index, Coord, INTERP_, Border::ZERO, IS_GPU>(input),
            .interpolator = InterpolatorSpectrum<REMAP, INTERP_, const_value_t, R, Index, NO_SHAPE, TEXTURE_ONLY>(logical_shape),
        };
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
