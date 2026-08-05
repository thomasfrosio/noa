#pragma once

#include "noa/base/Tuple.hpp"
#include "noa/runtime/cuda/Error.hpp"
#include "noa/runtime/core/Border.hpp"
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/xform/core/Interp.hpp"

namespace noa::xform::cuda {
    using noa::cuda::check; // import CUDA error checking to this namespace

    template<Interp INTERP, Border BORDER,
             typename Value, typename Coord, typename Index, usize B, usize R,
             bool NORMALIZED, bool LAYERED>
    class AccessorTexture;

    class AllocatorTexture {
    public:
        using allocate_type = noa::cuda::AllocatorArray::allocate_with_texture_type;

    public:
        template<usize N>
        static auto array_shape(Shape<isize, N> shape, usize rank) -> Shape<isize, 3> {
            check(rank == 1 or rank == 2 or rank == 3, "The rank isn't set, got rank={}", rank);
            check(N >= rank, "The rank is too large for the number of axes, got N={}, rank={}", N, rank);
            if constexpr (N >= 3) {
                if (rank == 3) {
                    for (usize i{}; i < N - rank; ++i)
                        if (shape[i] != 1)
                            panic("Non-empty batch axes with 3D CUDA textures are not supported. Got shape={}", shape);
                }
            }
            return shape.template as_nd<3>().ranked(rank);
        }

        template<usize N>
        static auto array_shape_and_strides(Shape<isize, N> shape, usize rank) -> Pair<Shape<isize, 3>, Strides<isize, N>> {
            auto shape_3d = array_shape(shape, rank);
            for (usize i{N - rank}; i < N; ++i)
                shape[i] = 1;
            auto strides = shape.strides();
            for (usize i{N - rank}; i < N; ++i)
                strides[i] = 0;
            return {shape_3d, strides};
        }

        template<nt::any_of<i8, i16, i32, u8, u16, u32, f16, f32, c16, c32> T>
        static auto allocate(
            const Shape3& shape,
            usize rank,
            noa::cuda::Device device,
            Interp interp,
            Border border
        ) -> allocate_type {
            check(rank == 1 or rank == 2 or rank == 3);
            const bool is_layered = rank == 2;
            const auto [filter, address, read_mode, normalized_coords] = convert_to_description(interp, border);
            return noa::cuda::AllocatorArray::allocate_with_texture<T>(
                shape, is_layered, device, filter, address, read_mode, normalized_coords);
        }

    public: // static texture utilities
        static constexpr auto convert_to_texture(
            Interp interp,
            Border border
        ) -> Pair<Interp, Border> {
            bool is_addressable = border.is_any(Border::ZERO, Border::CLAMP, Border::MIRROR, Border::PERIODIC);
            Border border_tex = is_addressable ? border : Border{Border::ZERO};
            Interp interp_tex =
                is_addressable and interp.is_any(Interp::LINEAR_FAST, Interp::CUBIC_BSPLINE_FAST) ?
                Interp::LINEAR_FAST : Interp::NEAREST_FAST;
            return {interp_tex, border_tex};
        }

        static auto convert_to_description(
            Interp interp,
            Border border
        ) -> Tuple<cudaTextureFilterMode, cudaTextureAddressMode, cudaTextureReadMode, bool> {
            auto [interp_tex, border_tex] = convert_to_texture(interp, border);

            cudaTextureFilterMode filter_mode =
                interp_tex == Interp::LINEAR_FAST ? cudaFilterModeLinear : cudaFilterModePoint;

            cudaTextureAddressMode address_mode{};
            bool normalized_coordinates{false};
            switch (border_tex) {
                case Border::PERIODIC: {
                    address_mode = cudaAddressModeWrap;
                    normalized_coordinates = true;
                    break;
                }
                case Border::MIRROR: {
                    address_mode = cudaAddressModeMirror;
                    normalized_coordinates = true;
                    break;
                }
                case Border::CLAMP: {
                    address_mode = cudaAddressModeClamp;
                    break;
                }
                case Border::ZERO: {
                    address_mode = cudaAddressModeBorder;
                    break;
                }
                default: panic();
            }
            return noa::make_tuple(filter_mode, address_mode, cudaReadModeElementType, normalized_coordinates);
        }

        template<Interp INTERP, Border BORDER, typename Value, typename Coord, typename Index, usize B, usize R>
        struct accessor_texture_type_ {
            static constexpr bool LAYERED = R == 2;
            static constexpr bool NORMALIZED = BORDER == Border::MIRROR or BORDER == Border::PERIODIC;
            static constexpr auto TEX = convert_to_texture(INTERP, BORDER);
            using type = AccessorTexture<TEX.first, TEX.second, Value, Coord, Index, B, R, NORMALIZED, LAYERED>;
        };

        /// The corresponding AccessorTexture type created by the allocator.
        template<Interp INTERP, Border BORDER, typename Value, typename Coord, typename Index, usize B, usize R>
        using accessor_texture_type = accessor_texture_type_<INTERP, BORDER, Value, Coord, Index, B, R>::type;
    };
}
