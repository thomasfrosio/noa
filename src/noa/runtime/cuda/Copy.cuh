#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/base/Operators.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Ewise.cuh"
#include "noa/runtime/cuda/Error.hpp"
#include "noa/runtime/cuda/Iwise.cuh"
#include "noa/runtime/cuda/Pointers.hpp"
#include "noa/runtime/cuda/Runtime.hpp"
#include "noa/runtime/cuda/Stream.hpp"

// Since we assume Compute Capability >= 2.0, all devices support the Unified Virtual Address Space, so
// the CUDA driver can determine, for each pointer, where the data is located, and one does not have to
// specify the cudaMemcpyKind. In the documentation they don't explicitly say that cudaMemcpyDefault allows
// for concurrent transfers between host and device if the host is pinned, but why would it make a difference?

namespace noa::cuda::details {
    template<typename T>
    auto to_copy_parameters(
        const T* src, isize src_pitch,
        T* dst, isize dst_pitch,
        const Shape3& shape
    ) -> cudaMemcpy3DParms {
        const auto s_shape = shape.as_safe<size_t>();
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), static_cast<size_t>(src_pitch) * sizeof(T), s_shape[2], s_shape[1]};
        params.dstPtr = {dst, static_cast<size_t>(dst_pitch) * sizeof(T), s_shape[2], s_shape[1]};
        params.extent = {s_shape[2] * sizeof(T), s_shape[1], s_shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    auto to_copy_parameters(
        const cudaArray* src,
        T* dst, isize dst_pitch,
        const Shape3& shape
    ) -> cudaMemcpy3DParms {
        const auto s_shape = shape.as_safe<size_t>();
        cudaMemcpy3DParms params{};
        params.srcArray = const_cast<cudaArray*>(src);
        params.dstPtr = {dst, static_cast<size_t>(dst_pitch) * sizeof(T), s_shape[2], s_shape[1]};
        params.extent = {s_shape[2], s_shape[1], s_shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    auto to_copy_parameters(
        const T* src, isize src_pitch,
        cudaArray* dst,
        const Shape3& shape
    ) -> cudaMemcpy3DParms {
        const auto s_shape = shape.as_safe<size_t>();
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), static_cast<size_t>(src_pitch) * sizeof(T), s_shape[2], s_shape[1]};
        params.dstArray = dst;
        params.extent = {s_shape[2], s_shape[1], s_shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    void memcpy(const T* src, T* dst, isize elements, Stream& stream) {
        const auto count = static_cast<size_t>(elements) * sizeof(T);
        check(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream.id()));
    }

    template<typename T>
    void memcpy(
        const T* src, isize src_pitch,
        T* dst, isize dst_pitch,
        const Shape3& shape, Stream& stream
    ) {
        const auto params = details::to_copy_parameters(src, src_pitch, dst, dst_pitch, shape);
        check(cudaMemcpy3DAsync(&params, stream.id()));
    }

    template<typename T, typename U, usize N>
    void copy_accessors_nd(T src, U dst, Shape<isize, N> shape) {
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

    template<typename Config = EwiseConfig<>, typename T, usize N> requires (not std::is_trivially_copyable_v<T>)
    void copy_non_trivial_contiguous(
        const T* src, Strides<isize, N> src_strides,
        T* dst, Strides<isize, N> dst_strides,
        Shape<isize, N> shape, Stream& stream
    ) {
        static_assert(cudaMemoryTypeUnregistered == 0);
        static_assert(cudaMemoryTypeHost == 1);
        static_assert(cudaMemoryTypeDevice == 2);
        static_assert(cudaMemoryTypeManaged == 3);

        const cudaPointerAttributes src_attr = pointer_attributes(src);
        const cudaPointerAttributes dst_attr = pointer_attributes(dst);

        if (src_attr.type == 2 and dst_attr.type == 2) {
            check(src_attr.device == dst_attr.device,
                  "Copying elements of a non-trivially copyable type between device memory of two different devices (device:src={}, device:dst={}) is not supported",
                  src_attr.device, dst_attr.device);

            auto input = noa::make_tuple(AccessorRestrictContiguous<const T, N, isize>(src, src_strides));
            auto output = noa::make_tuple(AccessorRestrictContiguous<T, N, isize>(dst, dst_strides));
            ewise<Config>(shape, Copy{}, std::move(input), std::move(output), stream);

        } else if (src_attr.type >= 1 and dst_attr.type >= 1) {
            // Both can be accessed on the device, so do the copy on the device.
            // For device memory, make sure the stream's device is correct. For pinned memory,
            // it seems that "portable-memory" is not a thing anymore since the documentation
            // says that any pinned allocation, regardless of the cudaHostAllocPortable flag,
            // can be accessed on any device. For managed memory, if the null stream was used
            // for the allocation or if the cudaMemAttachGlobal flag was used, it can be on
            // any device. As such, only enforce the device for device pointers here,
            // and let the driver check for pinned and managed if needed.
            check((src_attr.type != 2 or src_attr.device == stream.device().id()) and
                  (dst_attr.type != 2 or dst_attr.device == stream.device().id()),
                  "Copying elements of a non-trivially copyable type from or to a device that is not the stream's device is not supported");

            // TODO For managed pointers, use cudaMemPrefetchAsync()?
            auto src_ptr = static_cast<const T*>(src_attr.devicePointer);
            auto dst_ptr = static_cast<T*>(dst_attr.devicePointer);
            auto input = noa::make_tuple(AccessorRestrictContiguous<const T, N, isize>(src_ptr, src_strides));
            auto output = noa::make_tuple(AccessorRestrictContiguous<T, N, isize>(dst_ptr, dst_strides));
            ewise<Config>(shape, Copy{}, std::move(input), std::move(output), stream);

        } else if ((src_attr.type <= 1 or src_attr.type == 3) and
                   (dst_attr.type <= 1 or dst_attr.type == 3)) {
            // Both can be accessed on the host. Realistically, this never happens.
            const auto src_accessor = AccessorRestrictContiguous<const T, N, isize>(src, src_strides);
            const auto dst_accessor = AccessorRestrictContiguous<T, N, isize>(dst, dst_strides);
            stream.synchronize(); // TODO Use a callback instead?
            copy_accessors_nd(src_accessor, dst_accessor, shape);

        } else {
            panic("Copying elements of a non-trivially copyable type between an unregistered host region and a device is not supported");
        }
    }
}

namespace noa::cuda {
    /// Copy a single element, asynchronously.
    template<typename T>
    void copy(const T* src, T* dst, Stream& stream) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            details::memcpy(src, dst, 1, stream);
        } else {
            const auto shape = Shape4::from_value(1);
            const auto strides = Strides4::from_value(1);
            using config = EwiseConfig<false, false, 1, 1>; // 1 thread
            details::copy_non_trivial_contiguous<config>(src, strides, dst, strides, shape, stream);
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
            details::copy_non_trivial_contiguous(src, strides, dst, strides, shape, stream);
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
            details::copy_non_trivial_contiguous(src, src_strides, dst, dst_strides, shape, stream);
        }
    }

    /// Copies a strided range, asynchronously.
    template<typename T, usize N>
    void copy(
        const T* src, Strides<isize, N> src_strides,
        T* dst, Strides<isize, N> dst_strides,
        Shape<isize, N> shape, Stream& stream
    ) {
        // If contiguous or with a pitch, then we can rely on the CUDA runtime.
        // Given that we reorder to rightmost order and collapse the contiguous dimensions together,
        // this ends up being 99% of cases.
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
                // Before trying to call our own kernels, which cannot copy between devices/host,
                // collapse the contiguous dimensions together, and check again. This can reveal
                // 2d pitched layouts.
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

        // Otherwise:
        const cudaPointerAttributes src_attr = pointer_attributes(src);
        const cudaPointerAttributes dst_attr = pointer_attributes(dst);

        if (src_attr.type == 2 and dst_attr.type == 2) { // within device memory
            check(src_attr.device == dst_attr.device,
                  "Copying strided regions with an uncollapsible layout between different devices is currently not supported. Trying to copy an array of shape {} from (device={}, strides={}) to (device={}, strides={}) ",
                  shape, src_attr.device, src_strides, dst_attr.device, dst_strides);

            auto input = noa::make_tuple(AccessorRestrict<const T, N, isize>(src, src_strides));
            auto output = noa::make_tuple(AccessorRestrict<T, N, isize>(dst, dst_strides));
            ewise(shape, Copy{}, std::move(input), std::move(output), stream);

        } else if (src_attr.type >= 1 and dst_attr.type >= 1) { // between pinned/device/managed memory
            check((src_attr.type != 2 or src_attr.device == stream.device().id()) and
                  (dst_attr.type == 2 or dst_attr.device == stream.device().id()),
                  "Copying strided regions with an uncollapsible layout from or to a device that is not the stream's device is not supported");

            // TODO For managed pointers, use cudaMemPrefetchAsync()?
            auto src_ptr = static_cast<const T*>(src_attr.devicePointer);
            auto dst_ptr = static_cast<T*>(dst_attr.devicePointer);
            auto input = noa::make_tuple(AccessorRestrict<const T, N, isize>(src_ptr, src_strides));
            auto output = noa::make_tuple(AccessorRestrict<T, N, isize>(dst_ptr, dst_strides));
            ewise(shape, Copy{}, std::move(input), std::move(output), stream);

        } else if ((src_attr.type <= 1 or src_attr.type == 3) and
                   (dst_attr.type <= 1 or dst_attr.type == 3)) { // between unregistered-host and managed memory
            const auto src_accessor = AccessorRestrict<const T, N, isize>(src, src_strides);
            const auto dst_accessor = AccessorRestrict<T, N, isize>(dst, dst_strides);
            stream.synchronize(); // FIXME Use a callback instead?
            details::copy_accessors_nd(src_accessor, dst_accessor, shape);

        } else {
            panic("Copying strided regions with an uncollapsible layout between an unregistered host region and a device is not supported");
        }
    }
}

namespace noa::cuda {
    template<typename T>
    void copy(const T* src, isize src_pitch, cudaArray* dst, const Shape3& shape) {
        cudaMemcpy3DParms params = details::to_copy_parameters(src, src_pitch, dst, shape);
        check(cudaMemcpy3D(&params));
    }

    template<typename T>
    void copy(const T* src, isize src_pitch, cudaArray* dst, const Shape3& shape, Stream& stream) {
        cudaMemcpy3DParms params = details::to_copy_parameters(src, src_pitch, dst, shape);
        check(cudaMemcpy3DAsync(&params, stream.id()));
    }

    template<typename T>
    void copy(const cudaArray* src, T* dst, isize dst_pitch, const Shape3& shape) {
        cudaMemcpy3DParms params = details::to_copy_parameters(src, dst, dst_pitch, shape);
        check(cudaMemcpy3D(&params));
    }

    template<typename T>
    void copy(const cudaArray* src, T* dst, isize dst_pitch, const Shape3& shape, Stream& stream) {
        cudaMemcpy3DParms params = details::to_copy_parameters(src, dst, dst_pitch, shape);
        check(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /// Copy an array into a CUDA array.
    /// The source can be on any device or on the host.
    template<typename T>
    void copy(
        const T* src, const Strides3& src_strides, cudaArray* dst,
        const Shape3& shape, Stream& stream
    ) {
        const auto[desc_, actual_extent, flags] = AllocatorArray::array_info(dst);
        const bool is_layered = flags & cudaArrayLayered;
        const cudaExtent expected_extent = AllocatorArray::shape2extent(shape, is_layered);

        check(expected_extent.depth == actual_extent.depth and
              expected_extent.height == actual_extent.height and
              expected_extent.width == actual_extent.width,
              "The input shape is not compatible with the output CUDA array shape");

        // cudaExtent for CUDA array has empty dimensions equal to 0.
        // However, for cudaMemcpy3D, dimensions equal to 0 are invalid.
        auto shape_3d = Shape3::from_values(expected_extent.depth, expected_extent.height, expected_extent.width);
        shape_3d += Shape3::from_vec(shape_3d.cmp_eq(0));

        const bool is_rightmost = src_strides.is_rightmost();
        const bool has_valid_pitch = src_strides[1] >= shape_3d[2];
        const bool is_contiguous_2 = src_strides[2] == 1;
        const bool is_contiguous_0 = src_strides[0] == src_strides[1] * shape_3d[1];
        check(is_rightmost and has_valid_pitch and is_contiguous_0 and is_contiguous_2,
              "Input layout cannot be copied into a CUDA array. The input should be in the rightmost order, and its leftmost and rightmost axes should be contiguous, but got shape={} and strides={}",
              shape, src_strides);

        copy(src, src_strides[1], dst, shape_3d, stream);
    }

    template<typename T>
    void copy(cudaArray* src, T* dst, const Strides3& dst_strides, const Shape3& shape, Stream& stream) {
        const auto[_, actual_extent, flags] = AllocatorArray::array_info(src);
        const bool is_layered = flags & cudaArrayLayered;
        const cudaExtent expected_extent = AllocatorArray::shape2extent(shape, is_layered);

        check(expected_extent.depth == actual_extent.depth and
              expected_extent.height == actual_extent.height and
              expected_extent.width == actual_extent.width,
              "The output shape is not compatible with the input CUDA array shape");

        // cudaExtent for CUDA array has empty dimensions equal to 0.
        // However, for cudaMemcpy3D, dimensions equal to 0 are invalid.
        auto shape_3d = Shape3::from_values(expected_extent.depth, expected_extent.height, expected_extent.width);
        shape_3d += Shape3::from_vec(shape_3d.cmp_eq(0));

        const bool is_rightmost = dst_strides.is_rightmost();
        const bool has_valid_pitch = dst_strides[1] >= shape_3d[2];
        const bool is_contiguous_2 = dst_strides[2] == 1;
        const bool is_contiguous_0 = dst_strides[0] == dst_strides[1] * shape_3d[1];
        check(is_rightmost and has_valid_pitch and is_contiguous_0 and is_contiguous_2,
              "Input layout cannot be copied into a CUDA array. The input should be in the rightmost order, and its leftmost and rightmost axes should be contiguous, but got shape={} and strides={}",
              shape, dst_strides);

        copy(src, dst, dst_strides[1], shape_3d, stream);
    }

    /// Copies \p src to the constant memory at \p dst.
    template<typename T> requires std::is_trivially_copyable_v<T>
    void copy_to_constant_memory(const T* src, const void* dst, isize elements, isize offset, Stream& stream) {
        check(cudaMemcpyToSymbolAsync(
            /*symbol=*/ dst,
            /*src=*/ const_cast<T*>(src),
            /*count=*/ sizeof(T) * static_cast<size_t>(elements),
            /*offset=*/ sizeof(T) * static_cast<size_t>(offset),
            /*kind=*/ cudaMemcpyDefault,
            /*stream=*/ stream.id()));
    }

    template<typename T> requires std::is_trivially_copyable_v<T>
    void fill_with_zeroes(T* ptr, isize n_elements, Stream& stream) {
        check(cudaMemsetAsync(ptr, 0, static_cast<size_t>(n_elements) * sizeof(T), stream.id()));
    }
}
