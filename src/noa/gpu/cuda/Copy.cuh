#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/Operators.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/Pointers.hpp"
#include "noa/gpu/cuda/AllocatorArray.hpp"
#include "noa/gpu/cuda/Ewise.cuh"
#include "noa/gpu/cuda/Iwise.cuh"

// TODO Add nvrtc to support any type.

// Since we assume Compute Capability >= 2.0, all devices support the Unified Virtual Address Space, so
// the CUDA driver can determine, for each pointer, where the data is located, and one does not have to
// specify the cudaMemcpyKind. In the documentation they don't explicitly say that cudaMemcpyDefault allows
// for concurrent transfers between host and device if the host is pinned, but why would it make a difference?

namespace noa::cuda::guts {
    template<typename T>
    cudaMemcpy3DParms to_copy_parameters(
            const T* src, i64 src_pitch,
            T* dst, i64 dst_pitch,
            const Shape4<i64>& shape
    ) {
        const auto s_shape = shape.as_safe<size_t>();
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), static_cast<size_t>(src_pitch) * sizeof(T), s_shape[3], s_shape[2]};
        params.dstPtr = {dst, static_cast<size_t>(dst_pitch) * sizeof(T), s_shape[3], s_shape[2]};
        params.extent = {s_shape[3] * sizeof(T), s_shape[2], s_shape[1] * s_shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    inline cudaMemcpy3DParms to_copy_parameters(
            const cudaArray* src,
            T* dst, i64 dst_pitch,
            const Shape3<i64>& shape
    ) {
        const auto s_shape = shape.as_safe<size_t>();
        cudaMemcpy3DParms params{};
        params.srcArray = const_cast<cudaArray*>(src);
        params.dstPtr = {dst, static_cast<size_t>(dst_pitch) * sizeof(T), s_shape[2], s_shape[1]};
        params.extent = {s_shape[2], s_shape[1], s_shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    inline cudaMemcpy3DParms to_copy_parameters(
            const T* src, i64 src_pitch,
            cudaArray* dst,
            const Shape3<i64>& shape
    ) {
        const auto s_shape = shape.as_safe<size_t>();
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), static_cast<size_t>(src_pitch) * sizeof(T), s_shape[2], s_shape[1]};
        params.dstArray = dst;
        params.extent = {s_shape[2], s_shape[1], s_shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    inline void memcpy(const T* src, T* dst, i64 elements, Stream& stream) {
        const auto count = static_cast<size_t>(elements) * sizeof(T);
        check(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream.id()));
    }

    template<typename T>
    inline void memcpy(
            const T* src, i64 src_pitch,
            T* dst, i64 dst_pitch,
            const Shape4<i64>& shape, Stream& stream
    ) {
        const auto params = guts::to_copy_parameters(src, src_pitch, dst, dst_pitch, shape);
        check(cudaMemcpy3DAsync(&params, stream.id()));
    }

    template<typename Config = EwiseConfig<>, typename T> requires (not std::is_trivially_copyable_v<T>)
    void copy_non_trivial_contiguous(
            const T* src, Strides4<i64> src_strides,
            T* dst, Strides4<i64> dst_strides,
            Shape4<i64> shape, Stream& stream
    ) {
        static_assert(cudaMemoryTypeUnregistered == 0);
        static_assert(cudaMemoryTypeHost == 1);
        static_assert(cudaMemoryTypeDevice == 2);
        static_assert(cudaMemoryTypeManaged == 3);

        const cudaPointerAttributes src_attr = pointer_attributes(src);
        const cudaPointerAttributes dst_attr = pointer_attributes(dst);

        if (src_attr.type == 2 and dst_attr.type == 2) {
            check(src_attr.device == dst_attr.device,
                  "Copying elements of a non-trivially copyable type between device memory of two different devices "
                  "(device:src={}, device:dst={}) is not supported", src_attr.device, dst_attr.device);

            auto input = make_tuple(AccessorRestrictContiguousI64<const T, 4>(src, src_strides));
            auto output = make_tuple(AccessorRestrictContiguousI64<T, 4>(dst, dst_strides));
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
                  "Copying elements of a non-trivially copyable type from or to a device that "
                  "is not the stream's device is not supported");

            // FIXME For managed pointers, use cudaMemPrefetchAsync()?
            auto src_ptr = static_cast<const T*>(src_attr.devicePointer);
            auto dst_ptr = static_cast<T*>(dst_attr.devicePointer);
            auto input = make_tuple(AccessorRestrictContiguousI64<const T, 4>(src_ptr, src_strides));
            auto output = make_tuple(AccessorRestrictContiguousI64<T, 4>(dst_ptr, dst_strides));
            ewise<Config>(shape, Copy{}, std::move(input), std::move(output), stream);

        } else if ((src_attr.type <= 1 or src_attr.type == 3) and
                   (dst_attr.type <= 1 or dst_attr.type == 3)) {
            // Both can be accessed on the host. Realistically, this never happens.
            const auto src_accessor = AccessorRestrictContiguous<const T, 4, i64>(src, src_strides);
            const auto dst_accessor = AccessorRestrictContiguous<T, 4, i64>(dst, dst_strides);
            stream.synchronize(); // FIXME Use a callback instead?
            for (i64 i = 0; i < shape[0]; ++i)
                for (i64 j = 0; j < shape[1]; ++j)
                    for (i64 k = 0; k < shape[2]; ++k)
                        for (i64 l = 0; l < shape[3]; ++l)
                            dst_accessor(i, j, k, l) = src_accessor(i, j, k, l);

        } else {
            panic("Copying elements of a non-trivially copyable type between an unregistered "
                  "host region and a device is not supported");
        }
    }
}

namespace noa::cuda {
    /// Copy a single element, asynchronously.
    template<typename T>
    inline void copy(const T* src, T* dst, Stream& stream) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            guts::memcpy(src, dst, 1, stream);
        } else {
            const auto shape = Shape4<i64>::from_value(1);
            const auto strides = Strides4<i64>::from_value(1);
            using config = EwiseConfig<false, false, 1, 1>; // 1 thread
            guts::copy_non_trivial_contiguous<config>(src, strides, dst, strides, shape, stream);
        }
    }

    /// Copy a contiguous range, asynchronously.
    template<typename T>
    inline void copy(const T* src, T* dst, i64 elements, Stream& stream) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            guts::memcpy(src, dst, elements, stream);
        } else {
            const auto shape = Shape4<i64>{1, 1, 1, elements};
            const auto strides = Strides4<i64>::from_value(1);
            guts::copy_non_trivial_contiguous(src, strides, dst, strides, shape, stream);
        }
    }

    /// Copy a pitched range, asynchronously.
    template<typename T>
    inline void copy(const T* src, i64 src_pitch, T* dst, i64 dst_pitch, const Shape4<i64>& shape, Stream& stream) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            guts::memcpy(src, src_pitch, dst, dst_pitch, shape, stream);
        } else {
            const auto src_strides = Strides4<i64>{src_pitch, src_pitch, src_pitch, 1};
            const auto dst_strides = Strides4<i64>{dst_pitch, dst_pitch, dst_pitch, 1};
            guts::copy_non_trivial_contiguous(src, src_strides, dst, dst_strides, shape, stream);
        }
    }

    /// Copies a strided range, asynchronously.
    template<typename T>
    void copy(
            const T* src, Strides4<i64> src_strides,
            T* dst, Strides4<i64> dst_strides,
            Shape4<i64> shape, Stream& stream
    ) {
        // If contiguous or with a pitch, then we can rely on the CUDA runtime.
        // Given that we reorder to rightmost order and collapse the contiguous dimensions together,
        // this ends up being 99% of cases.
        Vec4<bool> is_contiguous;
        for (i32 test = 0; test <= 1; ++test) {
            // Rearrange to the rightmost order. Empty and broadcast dimensions in the output are moved to the left.
            // The input can be broadcast onto the output shape. While it is not valid for the output to broadcast
            // a non-empty dimension in the input, here, broadcast dimensions in the output are treated as empty,
            // so the corresponding input dimension isn't used and everything is fine.
            shape = ni::effective_shape(shape, dst_strides);
            const auto order = ni::order(dst_strides, shape);
            if (noa::any(order != Vec4<i64>{0, 1, 2, 3})) {
                shape = ni::reorder(shape, order);
                src_strides = ni::reorder(src_strides, order);
                dst_strides = ni::reorder(dst_strides, order);
            }

            is_contiguous = ni::is_contiguous(src_strides, shape) and
                            ni::is_contiguous(dst_strides, shape);
            if (is_contiguous[0] and is_contiguous[1] and is_contiguous[3]) {
                if (is_contiguous[2]) { // contiguous
                    return copy(src, dst, shape.elements(), stream);
                } else if (src_strides[2] >= shape[3] and dst_strides[2] >= shape[3]) { // 2d pitched
                    return copy(src, src_strides[2], dst, dst_strides[2], shape, stream);
                }
            }

            if (test == 0) { // try once
                // Before trying to call our own kernels, which cannot copy between devices/host,
                // collapse the contiguous dimensions together, and check again. This can reveal
                // 2d pitched layouts.
                auto collapsed_shape = shape;
                for (i64 i = 0; i < 3; ++i) {
                    if (is_contiguous[i] and is_contiguous[i + 1]) {
                        // Starting from the outermost dim, if the current dim and the next dim
                        // are contiguous, move the current dim to the next one.
                        collapsed_shape[i + 1] *= collapsed_shape[i];
                        collapsed_shape[i] = 1;
                    }
                }
                // We have a new shape, so compute the new strides.
                Strides4<i64> new_src_strides;
                Strides4<i64> new_dst_strides;
                if (ni::reshape(shape, src_strides, collapsed_shape, new_src_strides) and
                    ni::reshape(shape, dst_strides, collapsed_shape, new_dst_strides)) {
                    // Update and try again.
                    shape = collapsed_shape;
                    src_strides = new_src_strides;
                    dst_strides = new_dst_strides;
                } else {
                    panic("Copy failed. This should not have happened. Please report this issue. "
                          "shape:{}, src_strides:{}, dst_strides:{}",
                          shape, src_strides, dst_strides);
                }
            }
        }

        // Otherwise:
        const cudaPointerAttributes src_attr = pointer_attributes(src);
        const cudaPointerAttributes dst_attr = pointer_attributes(dst);

        if (src_attr.type == 2 and dst_attr.type == 2) { // within device memory
            check(src_attr.device == dst_attr.device,
                  "Copying strided regions, other than in the height dimension, between different devices "
                  "is currently not supported. Trying to copy an array of shape {} from (device:{}, strides:{}) "
                  "to (device:{}, strides:{}) ",
                  shape, src_attr.device, src_strides, dst_attr.device, dst_strides);

            auto input = make_tuple(AccessorRestrictI64<const T, 4>(src, src_strides));
            auto output = make_tuple(AccessorRestrictI64<T, 4>(dst, dst_strides));
            ewise(shape, Copy{}, std::move(input), std::move(output), stream);

        } else if (src_attr.type >= 1 and dst_attr.type >= 1) { // between pinned/device/managed memory
            check((src_attr.type != 2 or src_attr.device == stream.device().id()) and
                  (dst_attr.type == 2 or dst_attr.device == stream.device().id()),
                  "Copying strided regions, other than in the height dimension, "
                  "from or to a device that is not the stream's device is not supported");

            // FIXME For managed pointers, use cudaMemPrefetchAsync()?
            auto src_ptr = static_cast<const T*>(src_attr.devicePointer);
            auto dst_ptr = static_cast<T*>(dst_attr.devicePointer);
            auto input = make_tuple(AccessorRestrictI64<const T, 4>(src_ptr, src_strides));
            auto output = make_tuple(AccessorRestrictI64<T, 4>(dst_ptr, dst_strides));
            ewise(shape, Copy{}, std::move(input), std::move(output), stream);

        } else if ((src_attr.type <= 1 or src_attr.type == 3) and
                   (dst_attr.type <= 1 or dst_attr.type == 3)) { // between unregistered-host and managed memory
            const auto src_accessor = AccessorRestrict<const T, 4, i64>(src, src_strides);
            const auto dst_accessor = AccessorRestrict<T, 4, i64>(dst, dst_strides);
            stream.synchronize(); // FIXME Use a callback instead?
            for (i64 i = 0; i < shape[0]; ++i)
                for (i64 j = 0; j < shape[1]; ++j)
                    for (i64 k = 0; k < shape[2]; ++k)
                        for (i64 l = 0; l < shape[3]; ++l)
                            dst_accessor(i, j, k, l) = src_accessor(i, j, k, l);

        } else if (all(is_contiguous.pop_back())) {
            // Last resort for strided row-vector(s). Since 3 first dimensions are contiguous, collapse them.
            // Non-contiguous row vector can be reshaped to a 2D pitch array so that it can be passed to the CUDA API.
            // This works for column vectors as well, since we've swapped everything to the rightmost order.
            // Note: This is the last resort because it should be less efficient than our custom copy
            // (on host or device), so this is only if the copy is between unregister host and device, and
            // has a stride in the innermost dimension.
            const auto shape_2d_pitched = Shape4<i64>{1, shape[0] * shape[1] * shape[2], shape[3], 1};
            return copy(src, src_strides[3], dst, dst_strides[3], shape_2d_pitched, stream);

        } else {
            panic("Copying strided regions, other than in the height dimension, "
                  "between an unregistered host region and a device is not supported");
        }
    }
}

namespace noa::cuda {
    template<typename T>
    inline void copy(const T* src, i64 src_pitch, cudaArray* dst, const Shape3<i64>& shape) {
        cudaMemcpy3DParms params = guts::to_copy_parameters(src, src_pitch, dst, shape);
        check(cudaMemcpy3D(&params));
    }

    template<typename T>
    inline void copy(const T* src, i64 src_pitch, cudaArray* dst, const Shape3<i64>& shape, Stream& stream) {
        cudaMemcpy3DParms params = guts::to_copy_parameters(src, src_pitch, dst, shape);
        check(cudaMemcpy3DAsync(&params, stream.id()));
    }

    template<typename T>
    inline void copy(const cudaArray* src, T* dst, i64 dst_pitch, const Shape3<i64>& shape) {
        cudaMemcpy3DParms params = guts::to_copy_parameters(src, dst, dst_pitch, shape);
        check(cudaMemcpy3D(&params));
    }

    template<typename T>
    inline void copy(const cudaArray* src, T* dst, i64 dst_pitch, const Shape3<i64>& shape, Stream& stream) {
        cudaMemcpy3DParms params = guts::to_copy_parameters(src, dst, dst_pitch, shape);
        check(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /// Copy an array into a CUDA array.
    /// The source can be on any device or on the host.
    /// The source BDHW shape should match the shape of the CUDA array such that:
    ///  - If the CUDA array is layered, its shape should match the BHW dimensions of the source.
    ///  - If the CUDA array is NOT layered, its shape should match the DHW dimensions of the source.
    template<typename T>
    void copy(
            const T* src, const Strides4<i64>& src_strides, cudaArray* dst,
            const Shape4<i64>& shape, Stream& stream
    ) {
        const auto[desc_, actual_extent, flags] = AllocatorArray<T>::info(dst);
        const bool is_layered = flags & cudaArrayLayered;
        const cudaExtent expected_extent = AllocatorArray<T>::shape2extent(shape, is_layered);

        check(expected_extent.depth == actual_extent.depth and
              expected_extent.height == actual_extent.height and
              expected_extent.width == actual_extent.width,
              "The input shape is not compatible with the output CUDA array shape");

        // cudaExtent for CUDA array has empty dimensions equal to 0.
        // However, for cudaMemcpy3D, dimensions equal to 0 are invalid.
        auto shape_3d = Shape3<i64>::from_values(expected_extent.depth, expected_extent.height, expected_extent.width);
        shape_3d += Shape3<i64>::from_vec(shape_3d == 0);

        const bool is_column = shape[2] >= 1 and shape[3] == 1;
        const auto src_strides_3d = Strides3<i64>{
                src_strides[!is_layered],
                src_strides[2 + is_column],
                src_strides[3 - is_column]};
        const bool is_rightmost = ni::is_rightmost(src_strides_3d);
        const bool has_valid_pitch = src_strides_3d[1] >= shape_3d[2];
        const bool is_contiguous_2 = src_strides_3d[2] == 1;
        const bool is_contiguous_0 = src_strides_3d[0] == src_strides_3d[1] * shape_3d[1];
        check(is_rightmost and has_valid_pitch and is_contiguous_0 and is_contiguous_2,
              "Input layout cannot be copied into a CUDA array. The input should be in the rightmost order, "
              "and its {} and width dimension should be contiguous, but got shape {} and strides {}",
              is_layered ? "batch" : "depth", shape, src_strides);

        copy(src, src_strides[2], dst, shape_3d, stream);
    }

    template<typename T>
    void copy(cudaArray* src, T* dst, const Strides4<i64>& dst_strides, const Shape4<i64>& shape, Stream& stream) {
        const auto[desc_, actual_extent, flags] = AllocatorArray<T>::info(src);
        const bool is_layered = flags & cudaArrayLayered;
        const cudaExtent expected_extent = AllocatorArray<T>::shape2extent(shape, is_layered);

        check(expected_extent.depth == actual_extent.depth and
              expected_extent.height == actual_extent.height and
              expected_extent.width == actual_extent.width,
              "The output shape is not compatible with the input CUDA array shape");

        // cudaExtent for CUDA array has empty dimensions equal to 0.
        // However, for cudaMemcpy3D, dimensions equal to 0 are invalid.
        auto shape_3d = Shape3<i64>::from_values(expected_extent.depth, expected_extent.height, expected_extent.width);
        shape_3d += Shape3<i64>::from_vec(shape_3d == 0);

        const bool is_column = shape[2] >= 1 and shape[3] == 1;
        const auto dst_strides_3d = Strides3<i64>{
                dst_strides[!is_layered],
                dst_strides[2 + is_column],
                dst_strides[3 - is_column]};
        const bool is_rightmost = ni::is_rightmost(dst_strides_3d);
        const bool has_valid_pitch = dst_strides_3d[1] >= shape_3d[2];
        const bool is_contiguous_2 = dst_strides_3d[2] == 1;
        const bool is_contiguous_0 = dst_strides_3d[0] == dst_strides_3d[1] * shape_3d[1];
        check(is_rightmost and has_valid_pitch and is_contiguous_0 and is_contiguous_2,
              "Input layout cannot be copied into a CUDA array. The input should be in the rightmost order, "
              "and its {} and width dimension should be contiguous, but got shape {} and strides {}",
              is_layered ? "batch" : "depth", shape, dst_strides);

        copy(src, dst, dst_strides[2], shape_3d, stream);
    }

    /// Copies \p src to the constant memory at \p dst.
    template<typename T> requires std::is_trivially_copyable_v<T>
    void copy_to_constant_memory(const T* src, const void* dst, i64 elements, i64 offset, Stream& stream) {
        check(cudaMemcpyToSymbolAsync(
                /*symbol=*/ dst,
                /*src=*/ const_cast<T*>(src),
                /*count=*/ sizeof(T) * static_cast<size_t>(elements),
                /*offset=*/ sizeof(T) * static_cast<size_t>(offset),
                /*kind=*/ cudaMemcpyDefault,
                /*stream=*/ stream.id()));
    }
}
#endif
