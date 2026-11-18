#pragma once
#include "noa/runtime/gpu/IncludeGuard.hpp"

#include "noa/base/Operators.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/gpu/Allocators.hpp"
#include "noa/runtime/gpu/Ewise.hpp"
#include "noa/runtime/gpu/Error.hpp"
#include "noa/runtime/gpu/Iwise.hpp"
#include "noa/runtime/gpu/Pointers.hpp"
#include "noa/runtime/gpu/Runtime.hpp"
#include "noa/runtime/gpu/Stream.hpp"

namespace noa::gpu::details {
    // Naive single threaded copy.
    // Rarely, if ever, called.
    template<typename T, typename U, usize N>
    void ewise_copy_host(T src, U dst, Shape<isize, N> shape) {
        if constexpr (N == 1) {
            for (isize i = 0; i < shape[0]; ++i)
                dst(i) = src(i);
        } else if constexpr (N == 2) {
            for (isize i = 0; i < shape[0]; ++i)
                for (isize j = 0; j < shape[1]; ++j)
                    dst(i, j) = src(i, j);
        } else if constexpr (N == 3) {
            for (isize i = 0; i < shape[0]; ++i)
                for (isize j = 0; j < shape[1]; ++j)
                    for (isize k = 0; k < shape[2]; ++k)
                        dst(i, j, k) = src(i, j, k);
        } else if constexpr (N == 4) {
            for (isize i = 0; i < shape[0]; ++i)
                for (isize j = 0; j < shape[1]; ++j)
                    for (isize k = 0; k < shape[2]; ++k)
                        for (isize l = 0; l < shape[3]; ++l)
                            dst(i, j, k, l) = src(i, j, k, l);
        } else if constexpr (N == 5) {
            for (isize i = 0; i < shape[0]; ++i)
                for (isize j = 0; j < shape[1]; ++j)
                    for (isize k = 0; k < shape[2]; ++k)
                        for (isize l = 0; l < shape[3]; ++l)
                            for (isize m = 0; m < shape[4]; ++m)
                                dst(i, j, k, l, m) = src(i, j, k, l, m);
        } else if constexpr (N == 6) {
            for (isize i = 0; i < shape[0]; ++i)
                for (isize j = 0; j < shape[1]; ++j)
                    for (isize k = 0; k < shape[2]; ++k)
                        for (isize l = 0; l < shape[3]; ++l)
                            for (isize m = 0; m < shape[4]; ++m)
                                for (isize n = 0; n < shape[5]; ++n)
                                    dst(i, j, k, l, m, n) = src(i, j, k, l, m, n);
        } else {
            static_assert(nt::always_false<T>);
        }
    }

    template<bool Contiguous = false, typename Config = EwiseConfig<>, typename T, usize N>
    void ewise_copy(
        const T* src, Strides<isize, N> src_strides,
        T* dst, Strides<isize, N> dst_strides,
        Shape<isize, N> shape, Stream& stream
    ) {
        constexpr auto STRIDES_TRAIT = Contiguous ? StridesTraits::CONTIGUOUS : StridesTraits::STRIDED;
        using src_accessor = AccessorRestrict<const T, N, isize, STRIDES_TRAIT>;
        using dst_accessor = AccessorRestrict<T, N, isize, STRIDES_TRAIT>;

        const auto src_info = PointerInfo::from_pointer(src);
        const auto dst_info = PointerInfo::from_pointer(dst);
        const auto device_id = stream.device().id();

        if (src_info.is_accessible_from_device_id(device_id) and dst_info.is_accessible_from_device_id(device_id)) {
            // Device-only, pinned, and unified.
            // In CUDA, unified with stream-only visibility may raise an error if the stream doesn't match. This is
            // only relevant for old devices with cudaDevAttrConcurrentManagedAccess == 0. Unfortunately, we can't
            // recover the stream used for the allocation from the raw pointer.
            // TODO prefetch managed memory. For large strides this could be problematic as we may ask to prefetch
            //      large regions that are actually not accessed during the copy. Letting page faults resolve might
            //      be better?
            auto input = noa::make_tuple(src_accessor(static_cast<const T*>(src_info.device_ptr), src_strides));
            auto output = noa::make_tuple(dst_accessor(static_cast<T*>(dst_info.device_ptr), dst_strides));
            ewise<Config>(shape, Copy{}, std::move(input), std::move(output), stream);
        } else if (src_info.is_accessible_from_host() and dst_info.is_accessible_from_host()) {
            stream.synchronize(); // TODO Use a callback instead?
            ewise_copy_host(src_accessor(src, src_strides), dst_accessor(dst, dst_strides), shape);
        } else {
            // Unregistered host memory involved. While we could enqueue one memcpy per row (if trivially copyable),
            // this could be quite inefficient. Instead of hiding this problematic case, prefer to throw an error to
            // alert the user to either use a contiguous region (by possibly copying the entire contiguous region) or
            // by using unified memory.
            panic(
                "Cannot copy src ({}, contiguous={}, device:id={}) to dst ({}, contiguous={}, device:id={}) (trivially_copyable={})",
                src_info.kind, src_strides.is_contiguous(shape), src_info.device_id,
                dst_info.kind, dst_strides.is_contiguous(shape), dst_info.device_id,
                std::is_trivially_copyable_v<T>
            );
        }
    }

    template<typename Config = EwiseConfig<>, typename T, usize N>
    void ewise_copy_contiguous(
        const T* src, const Strides<isize, N>& ss,
        T* dst, const Strides<isize, N>& ds,
        const Shape<isize, N>& shape,
        Stream& stream
    ) {
        ewise_copy<true, Config>(src, ss, dst, ds, shape, stream);
    }

#ifndef NOA_ENABLE_SYCL
#   if defined(NOA_ENABLE_CUDA)
    using Memcpy3DParams = cudaMemcpy3DParms;
    using GpuArray = cudaArray;
    inline constexpr u32 ARRAY_LAYERED = cudaArrayLayered;
    inline void memcpy3d(const Memcpy3DParams& p) { check(cudaMemcpy3D(&p)); }
    inline void memcpy3d_async(const Memcpy3DParams& p, Stream& s) { check(cudaMemcpy3DAsync(&p, s.id())); }
#   else defined(NOA_ENABLE_HIP)
    using Memcpy3DParams = hipMemcpy3DParms;
    using GpuArray = hipArray;
    inline constexpr u32 ARRAY_LAYERED = hipArrayLayered;
    inline void memcpy3d(const Memcpy3DParams& p) { check(hipMemcpy3D(&p)); }
    inline void memcpy3d_async(const Memcpy3DParams& p, Stream& s) { check(hipMemcpy3DAsync(&p, s.id())); }
#   endif

    template<typename T>
    auto to_copy_parameters(const T* src, isize src_pitch, T* dst, isize dst_pitch, const Shape3& shape) -> Memcpy3DParams {
        const auto s_shape = shape.as_safe<usize>();
        Memcpy3DParams params{};
        params.srcPtr = {const_cast<T*>(src), static_cast<usize>(src_pitch) * sizeof(T), s_shape[2], s_shape[1]};
        params.dstPtr = {dst, static_cast<usize>(dst_pitch) * sizeof(T), s_shape[2], s_shape[1]};
        params.extent = {s_shape[2] * sizeof(T), s_shape[1], s_shape[0]};
#   if defined(NOA_ENABLE_CUDA)
        params.kind = cudaMemcpyDefault;
#   else defined(NOA_ENABLE_HIP)
        params.kind = hipMemcpyDefault;
#   endif
        return params;
    }
#endif

    template<typename T> requires std::is_trivially_copyable_v<T>
    void memcpy(const T* src, T* dst, isize n_elements, Stream& stream) {
        const auto n_bytes = static_cast<usize>(n_elements) * sizeof(T);
        #if defined(NOA_ENABLE_CUDA)
        check(cudaMemcpyAsync(dst, src, n_bytes, cudaMemcpyDefault, stream.id()));
        #elif defined(NOA_ENABLE_HIP)
        check(hipMemcpyAsync(dst, src, n_bytes, hipMemcpyDefault, stream.id()));
        #elif defined(NOA_ENABLE_SYCL)
        // USM -> USM (same context) or from/to plain host memory.
        // Copies between pointers of different contexts are not valid in SYCL, but let SYCL report the error.
        if (n_bytes > 0)
            stream.enqueue_sycl_memcpy(dst, src, n_bytes);
        #endif
    }

    template<typename T> requires std::is_trivially_copyable_v<T>
    void memcpy(const T* src, isize src_pitch, T* dst, isize dst_pitch, const Shape3& shape, Stream& stream) {
        #ifndef NOA_ENABLE_SYCL
        const auto params = to_copy_parameters(src, src_pitch, dst, dst_pitch, shape);
        memcpy3d_async(params, stream);
        #else
        // TODO check DPC++ ext_oneapi_memcpy2d
        if (src_pitch == shape[2] and dst_pitch == shape[2])
            return memcpy(src, dst, shape.n_elements(), stream);
        // No pitched memcpy in SYCL. The main consequence is that it cannot copy
        // pitched layouts between device and unregistered host, whereas CUDA/HIP can.
        ewise_copy_contiguous(
            src, Strides3{src_pitch * shape[1], src_pitch, 1},
            dst, Strides3{dst_pitch * shape[1], dst_pitch, 1},
            shape, stream
        );
        #endif
    }
}

namespace noa::gpu {
    /// Copy a single element, asynchronously.
    template<typename T>
    void copy(const T* src, T* dst, Stream& stream) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            details::memcpy(src, dst, 1, stream);
        } else {
            const auto shape = Shape4::from_value(1);
            const auto strides = Strides4::from_value(1);
            using config = EwiseConfig<false, false, 1, 1>; // 1 thread
            details::ewise_copy_contiguous<config>(src, strides, dst, strides, shape, stream);
        }
    }

    /// Copy a contiguous range, asynchronously.
    template<typename T>
    void copy(const T* src, T* dst, isize elements, Stream& stream) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            details::memcpy(src, dst, elements, stream);
        } else {
            const auto shape = Shape4{1, 1, 1, elements};
            const auto strides = Strides4::from_value(1);
            details::ewise_copy_contiguous(src, strides, dst, strides, shape, stream);
        }
    }

    /// Copy a pitched range, asynchronously.
    template<typename T>
    void copy(const T* src, isize src_pitch, T* dst, isize dst_pitch, const Shape3& shape, Stream& stream) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            details::memcpy(src, src_pitch, dst, dst_pitch, shape, stream);
        } else {
            const auto src_strides = Strides3{src_pitch * shape[1], src_pitch, 1};
            const auto dst_strides = Strides3{dst_pitch * shape[1], dst_pitch, 1};
            details::ewise_copy_contiguous(src, src_strides, dst, dst_strides, shape, stream);
        }
    }

    /// Copies a strided range, asynchronously.
    template<typename T, usize N>
    void copy(
        const T* src, Strides<isize, N> src_strides,
        T* dst, Strides<isize, N> dst_strides,
        Shape<isize, N> shape, Stream& stream
    ) {
        Vec<bool, N> is_contiguous;
        for (i32 test = 0; test <= 1; ++test) {
            // Rearrange to the rightmost order. Empty and broadcast dimensions in the output are moved to the left.
            // The input can be broadcast onto the output shape. While it is not valid for the output to broadcast
            // a non-empty dimension in the input, here, broadcast dimensions in the output are treated as empty,
            // so the corresponding input dimension isn't used and everything is fine.
            shape = dst_strides.effective_shape(shape);
            nd::permute_all_to_rightmost_order(dst_strides, shape, shape, src_strides, dst_strides);

            is_contiguous = src_strides.contiguity(shape) and dst_strides.contiguity(shape);
            if (is_contiguous == true)
                return copy(src, dst, shape.n_elements(), stream); // contiguous

            if constexpr (N >= 2) {
                if (is_contiguous.template set<(N - 2)>(true) == true and
                    src_strides[N - 2] >= shape[N - 1] and
                    dst_strides[N - 2] >= shape[N - 1]) {
                    auto shape_3d = Shape<isize, 3>{1, shape[N - 2], shape[N - 1]};
                    if constexpr (N >= 3)
                        shape_3d[0] = shape.template pop_back<2>().n_elements();
                    return copy(src, src_strides[N - 2], dst, dst_strides[N - 2], shape_3d, stream); // pitched
                }
            }

            if (test == 0) { // try once
                // Before trying to call our own kernels, which cannot copy between devices/host-unregistered,
                // collapse the contiguous dimensions together, and check again. This can reveal 2d pitched layouts.
                const auto contiguity = src_strides.contiguity(shape) and dst_strides.contiguity(shape);
                const auto broadcasting = src_strides.broadcasting(shape) or dst_strides.broadcasting(shape);
                auto collapsed_shape = noa::collapse(shape, contiguity, broadcasting);
                collapsed_shape = collapsed_shape.permute(noa::squeeze_empty_dimensions_left(collapsed_shape));

                // We have a new shape, so compute the new strides.
                Strides<isize, N> new_src_strides;
                Strides<isize, N> new_dst_strides;
                if (noa::reshape(shape, src_strides, collapsed_shape, new_src_strides) and
                    noa::reshape(shape, dst_strides, collapsed_shape, new_dst_strides)) {
                    // Update and try again.
                    shape = collapsed_shape;
                    src_strides = new_src_strides;
                    dst_strides = new_dst_strides;
                } else {
                    panic("INTERNAL: reshape failed, please report this issue, shape:{}, src_strides:{}, dst_strides:{}",
                          shape, src_strides, dst_strides);
                }
            }
        }
        details::ewise_copy(src, src_strides, dst, dst_strides, shape, stream);
    }

    /// Copies \p src to the constant memory at \p dst.
    template<typename T> requires std::is_trivially_copyable_v<T>
    void copy_to_constant_memory(const T* src, const void* dst, isize elements, isize offset, Stream& stream) {
        const auto n_bytes = sizeof(T) * static_cast<usize>(elements);
        const auto byte_offset = sizeof(T) * static_cast<usize>(offset);
        #if defined(NOA_ENABLE_CUDA)
        check(cudaMemcpyToSymbolAsync(dst, const_cast<T*>(src), n_bytes, byte_offset, cudaMemcpyDefault, stream.id()));
        #elif defined(NOA_ENABLE_HIP)
        check(hipMemcpyToSymbolAsync(dst, const_cast<T*>(src), n_bytes, byte_offset, hipMemcpyDefault, stream.id()));
        #elif defined(NOA_ENABLE_SYCL)
        // No constant memory in SYCL: dst must be a simple USM device pointer.
        if (n_bytes > 0) {
            details::memcpy(
                static_cast<const std::byte*>(src),
                static_cast<std::byte*>(const_cast<void*>(dst)) + byte_offset,
                n_bytes, stream
            );
        }
        #endif
    }

    template<typename T> requires std::is_trivially_copyable_v<T>
    void fill_with_zeroes(T* ptr, isize n_elements, Stream& stream) {
        const auto count = static_cast<usize>(n_elements) * sizeof(T);
        #if defined(NOA_ENABLE_CUDA)
        check(cudaMemsetAsync(ptr, 0, count, stream.id()));
        #elif defined(NOA_ENABLE_HIP)
        check(hipMemsetAsync(ptr, 0, count, stream.id()));
        #elif defined(NOA_ENABLE_SYCL)
        if (count > 0)
            stream.enqueue_sycl_memset(ptr, 0, count);
        #endif
    }
}

#ifndef NOA_ENABLE_SYCL
namespace noa::gpu::details {
    template<typename T>
    auto to_copy_parameters(
        const gpuArray* src,
        T* dst, isize dst_pitch,
        const Shape3& shape
    ) -> cudaMemcpy3DParms {
        const auto s_shape = shape.as_safe<usize>();
        cudaMemcpy3DParms params{};
        params.srcArray = const_cast<gpuArray*>(src);
        params.dstPtr = {dst, static_cast<usize>(dst_pitch) * sizeof(T), s_shape[2], s_shape[1]};
        params.extent = {s_shape[2], s_shape[1], s_shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    auto to_copy_parameters(
        const T* src, isize src_pitch,
        gpuArray* dst,
        const Shape3& shape
    ) -> cudaMemcpy3DParms {
        const auto s_shape = shape.as_safe<usize>();
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), static_cast<usize>(src_pitch) * sizeof(T), s_shape[2], s_shape[1]};
        params.dstArray = dst;
        params.extent = {s_shape[2], s_shape[1], s_shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }
}

namespace noa::gpu {
    template<typename T>
    void copy(const T* src, isize src_pitch, gpuArray* dst, const Shape3& shape) {
        auto params = details::to_copy_parameters(src, src_pitch, dst, shape);
        details::memcpy3d(&params);
    }

    template<typename T>
    void copy(const T* src, isize src_pitch, gpuArray* dst, const Shape3& shape, Stream& stream) {
        auto params = details::to_copy_parameters(src, src_pitch, dst, shape);
        details::memcpy3d_async(&params, stream.id());
    }

    template<typename T>
    void copy(const gpuArray* src, T* dst, isize dst_pitch, const Shape3& shape) {
        auto params = details::to_copy_parameters(src, dst, dst_pitch, shape);
        details::memcpy3d(&params);
    }

    template<typename T>
    void copy(const gpuArray* src, T* dst, isize dst_pitch, const Shape3& shape, Stream& stream) {
        auto params = details::to_copy_parameters(src, dst, dst_pitch, shape);
        details::memcpy3d_async(&params, stream.id());
    }

    /// Copy an array into a GPU array.
    /// The source can be on any device or on the host.
    template<typename T>
    void copy(
        const T* src, const Strides3& src_strides, gpuArray* dst,
        const Shape3& shape, Stream& stream
    ) {
        const auto[desc_, actual_extent, flags] = AllocatorArray::array_info(dst);
        const bool is_layered = flags & ARRAY_LAYERED;
        const cudaExtent expected_extent = AllocatorArray::shape2extent(shape, is_layered);

        check(expected_extent.depth == actual_extent.depth and
              expected_extent.height == actual_extent.height and
              expected_extent.width == actual_extent.width,
              "The input shape is not compatible with the output GPU array shape");

        // cudaExtent for GPU array has empty dimensions equal to 0.
        // However, for cudaMemcpy3D, dimensions equal to 0 are invalid.
        auto shape_3d = Shape3::from_values(expected_extent.depth, expected_extent.height, expected_extent.width);
        shape_3d += Shape3::from_vec(shape_3d.cmp_eq(0));

        const bool is_rightmost = src_strides.is_rightmost();
        const bool has_valid_pitch = src_strides[1] >= shape_3d[2];
        const bool is_contiguous_2 = src_strides[2] == 1;
        const bool is_contiguous_0 = src_strides[0] == src_strides[1] * shape_3d[1];
        check(is_rightmost and has_valid_pitch and is_contiguous_0 and is_contiguous_2,
              "Input layout cannot be copied into a GPU array. The input should be in the rightmost order, and its leftmost and rightmost axes should be contiguous, but got shape={} and strides={}",
              shape, src_strides);

        copy(src, src_strides[1], dst, shape_3d, stream);
    }

    template<typename T>
    void copy(gpuArray* src, T* dst, const Strides3& dst_strides, const Shape3& shape, Stream& stream) {
        const auto[_, actual_extent, flags] = AllocatorArray::array_info(src);
        const bool is_layered = flags & ARRAY_LAYERED;
        const cudaExtent expected_extent = AllocatorArray::shape2extent(shape, is_layered);

        check(expected_extent.depth == actual_extent.depth and
              expected_extent.height == actual_extent.height and
              expected_extent.width == actual_extent.width,
              "The output shape is not compatible with the input GPU array shape");

        // cudaExtent for GPU array has empty dimensions equal to 0.
        // However, for cudaMemcpy3D, dimensions equal to 0 are invalid.
        auto shape_3d = Shape3::from_values(expected_extent.depth, expected_extent.height, expected_extent.width);
        shape_3d += Shape3::from_vec(shape_3d.cmp_eq(0));

        const bool is_rightmost = dst_strides.is_rightmost();
        const bool has_valid_pitch = dst_strides[1] >= shape_3d[2];
        const bool is_contiguous_2 = dst_strides[2] == 1;
        const bool is_contiguous_0 = dst_strides[0] == dst_strides[1] * shape_3d[1];
        check(is_rightmost and has_valid_pitch and is_contiguous_0 and is_contiguous_2,
              "Input layout cannot be copied into a GPU array. The input should be in the rightmost order, and its leftmost and rightmost axes should be contiguous, but got shape={} and strides={}",
              shape, dst_strides);

        copy(src, dst, dst_strides[1], shape_3d, stream);
    }
}
#endif
