#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/memory/AllocatorArray.hpp"

// TODO Add nvrtc to support any type.

// Since we assume Compute Capability >= 2.0, all devices support the Unified Virtual Address Space, so
// the CUDA driver can determine, for each pointer, where the data is located, and one does not have to
// specify the cudaMemcpyKind. In the documentation they don't explicitly say that cudaMemcpyDefault allows
// for concurrent transfers between host and device if the host is pinned, but why would it make a difference?

namespace noa::cuda::memory::details {
    template<typename T>
    cudaMemcpy3DParms to_copy_parameters(
            const T* src, i64 src_pitch,
            T* dst, i64 dst_pitch,
            const Shape4<i64>& shape) {
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
            const Shape3<i64>& shape) {
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
            const Shape3<i64>& shape) {
        const auto s_shape = shape.as_safe<size_t>();
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), static_cast<size_t>(src_pitch) * sizeof(T), s_shape[2], s_shape[1]};
        params.dstArray = dst;
        params.extent = {s_shape[2], s_shape[1], s_shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    // Copy strided data between two pointers accessible by the stream's device.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_numeric_v<T> || std::is_same_v<T, bool>>>
    void copy(const T* src, const Strides4<i64>& src_strides,
              T* dst, const Strides4<i64>& dst_strides,
              const Shape4<i64>& shape, Stream& stream);
}

namespace noa::cuda::memory {
    // Copies contiguous memory from one region to another. These can point to host or device memory.
    // If a stream is passed, copy is enqueued to the stream. In this case, passing raw pointers one must make
    // sure the memory stay valid until completion. Copies between host and device can execute concurrently only
    // if the copy involves pinned memory.

    template<typename T>
    inline void copy(const T* src, T* dst, i64 elements) {
        const auto count = static_cast<size_t>(elements) * sizeof(T);
        NOA_THROW_IF(cudaMemcpy(dst, src, count, cudaMemcpyDefault));
    }

    template<typename T>
    inline void copy(const T* src, T* dst, i64 elements, Stream& stream) {
        const auto count = static_cast<size_t>(elements) * sizeof(T);
        NOA_THROW_IF(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream.id()));
    }

    // Copies asynchronously pitched memory from one region to another. These can point to host or device memory.
    // If a stream is passed, copy is enqueued to the stream. In this case, passing raw pointers one must make
    // sure the memory stay valid until completion. Copies between host and device can execute concurrently only
    // if the copy involves pinned memory.

    template<typename T>
    inline void copy(const T* src, i64 src_pitch,
                     T* dst, i64 dst_pitch,
                     const Shape4<i64>& shape, Stream& stream) {
        const auto params = details::to_copy_parameters(src, src_pitch, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    // Copies asynchronously regions of (strided/padded) memory.
    // Contiguous regions of memory have no copy restrictions, as well as (batched) row/column vectors,
    // regions with padding on the right side of the innermost dimension (referred to as "pitched" in CUDA),
    // or layouts that can be reordered/reshaped to these aforementioned cases.
    // However, if the contiguity is broken in any other dimension, an error will be thrown if:
    // 1) the source and destination are on different devices,
    // 2) the copy is between unregistered host memory and a device,
    // 3) the copy involves a device that is not the stream's device.
    // If the copy involves an unregistered memory regions, the stream will be synchronized when the function
    // returns. Otherwise, this function can be asynchronous relative to the host and may return before completion.
    template<typename T>
    void copy(const T* src, Strides4<i64> src_strides,
              T* dst, Strides4<i64> dst_strides,
              Shape4<i64> shape, Stream& stream) {
        NOA_ASSERT(noa::all(shape > 0));

        // If contiguous or with a pitch, then we can rely on the CUDA runtime.
        // Given that we reorder to rightmost order and collapse the contiguous dimensions together,
        // this ends up being 99% of cases.
        Vec4<bool> is_contiguous;
        for (i32 test = 0; test <= 1; ++test) {
            // Rearrange to the rightmost order. Empty and broadcast dimensions in the output are moved to the left.
            // The input can be broadcast onto the output shape. While it is not valid for the output to broadcast
            // a non-empty dimension in the input, here, broadcast dimensions in the output are treated as empty,
            // so the corresponding input dimension isn't used and everything is fine.
            shape = noa::indexing::effective_shape(shape, dst_strides);
            const auto order = noa::indexing::order(dst_strides, shape);
            if (noa::any(order != Vec4<i64>{0, 1, 2, 3})) {
                shape = noa::indexing::reorder(shape, order);
                src_strides = noa::indexing::reorder(src_strides, order);
                dst_strides = noa::indexing::reorder(dst_strides, order);
            }

            is_contiguous = noa::indexing::is_contiguous(src_strides, shape) &&
                            noa::indexing::is_contiguous(dst_strides, shape);
            if (is_contiguous[0] && is_contiguous[1] && is_contiguous[3]) {
                if (is_contiguous[2]) { // contiguous
                    return copy(src, dst, shape.elements(), stream);
                } else if (src_strides[2] >= shape[3] && dst_strides[2] >= shape[3]) { // 2d pitched
                    return copy(src, src_strides[2], dst, dst_strides[2], shape, stream);
                }
            }

            if (test == 0) { // try once
                // Before trying to call our own kernels, which cannot copy between devices/host,
                // collapse the contiguous dimensions together, and check again. This can reveal
                // 2d pitched layouts.
                auto collapsed_shape = shape;
                for (i64 i = 0; i < 3; ++i) {
                    if (is_contiguous[i] && is_contiguous[i + 1]) {
                        // Starting from the outermost dim, if the current dim and the next dim
                        // are contiguous, move the current dim to the next one.
                        collapsed_shape[i + 1] *= collapsed_shape[i];
                        collapsed_shape[i] = 1;
                    }
                }
                // We have a new shape, so compute the new strides.
                Strides4<i64> new_src_strides;
                Strides4<i64> new_dst_strides;
                if (noa::indexing::reshape(shape, src_strides, collapsed_shape, new_src_strides) &&
                    noa::indexing::reshape(shape, dst_strides, collapsed_shape, new_dst_strides)) {
                    // Update and try again.
                    shape = collapsed_shape;
                    src_strides = new_src_strides;
                    dst_strides = new_dst_strides;
                } else {
                    NOA_THROW("Copy failed. This should not have happened. Please report this issue. "
                              "shape:{}, src_strides:{}, dst_strides:{}",
                              shape, src_strides, dst_strides);
                }
            }
        }

        // Otherwise:
        static_assert(cudaMemoryTypeUnregistered == 0);
        static_assert(cudaMemoryTypeHost == 1);
        static_assert(cudaMemoryTypeDevice == 2);
        static_assert(cudaMemoryTypeManaged == 3);
        const cudaPointerAttributes src_attr = noa::cuda::utils::pointer_attributes(src);
        const cudaPointerAttributes dst_attr = noa::cuda::utils::pointer_attributes(dst);

        if (src_attr.type == 2 && dst_attr.type == 2) {
            // Both regions are on the same device, we can therefore launch our copy kernel.
            if (src_attr.device == dst_attr.device) {
                if constexpr (nt::is_restricted_numeric_v<T> || std::is_same_v<T, bool>)
                    details::copy(src, src_strides, dst, dst_strides, shape, stream);
                else
                    NOA_THROW("Copying strided regions, other than in the height dimension, "
                              "is not supported for type {}", string::human<T>());
            } else {
                NOA_THROW("Copying strided regions, other than in the height dimension, "
                          "between different devices is currently not supported. Trying to copy an array of "
                          "shape {} from (device:{}, strides:{}) to (device:{}, strides:{}) ",
                          shape, src_attr.device, src_strides, dst_attr.device, dst_strides);
            }

        } else if (src_attr.type >= 1 && dst_attr.type >= 1) {
            // Both can be accessed on the device, so do the copy on the device.
            // For device memory, make sure the stream's device is correct. For pinned memory,
            // it seems that "portable-memory" is not a thing anymore since the documentation
            // says that any pinned allocation, regardless of the cudaHostAllocPortable flag,
            // can be accessed on any device. For managed memory, if the null stream was used
            // for the allocation or if the cudaMemAttachGlobal flag was used, it can be on
            // any device. As such, only enforce the device for device pointers here,
            // and let the driver check for pinned and managed if needed.
            if ((src_attr.type == 2 && src_attr.device != stream.device().id()) ||
                (dst_attr.type == 2 && dst_attr.device != stream.device().id()))
                NOA_THROW("Copying strided regions, other than in the height dimension, "
                          "from or to a device that is not the stream's device is not supported");

            // FIXME For managed pointers, use cudaMemPrefetchAsync()?
            if constexpr (nt::is_restricted_numeric_v<T> || std::is_same_v<T, bool>)
                details::copy(reinterpret_cast<const T*>(src_attr.devicePointer), src_strides,
                              reinterpret_cast<T*>(dst_attr.devicePointer), dst_strides,
                              shape, stream);
            else
                NOA_THROW("Copying strided regions, other than in the height dimension, "
                          "is not supported for type {}", noa::string::human<T>());

        } else if ((src_attr.type <= 1 || src_attr.type == 3) &&
                   (dst_attr.type <= 1 || dst_attr.type == 3)) {
            // Both can be accessed on the host. Realistically, this never happens.
            NOA_ASSERT(!noa::indexing::are_overlapped(src, src_strides, dst, dst_strides, shape));
            const auto src_accessor = AccessorRestrict<const T, 4, i64>(src, src_strides);
            const auto dst_accessor = AccessorRestrict<T, 4, i64>(dst, dst_strides);
            stream.synchronize(); // FIXME Use a callback instead?
            for (i64 i = 0; i < shape[0]; ++i)
                for (i64 j = 0; j < shape[1]; ++j)
                    for (i64 k = 0; k < shape[2]; ++k)
                        for (i64 l = 0; l < shape[3]; ++l)
                            dst_accessor(i, j, k, l) = src_accessor(i, j, k, l);

        } else if (is_contiguous[0] && is_contiguous[1] && is_contiguous[2]) {
            // Last resort for strided row-vector(s). Since 3 first dimensions are contiguous, collapse them.
            // Non-contiguous row vector can be reshaped to a 2D pitch array so that it can be passed to the CUDA API.
            // This works for column vectors as well, since we've swapped everything to the rightmost order.
            // Note: This is the last resort because it should be less efficient than our custom copy
            // (on host or device), so this is only if the copy is between unregister host and device, and
            // has a stride in the innermost dimension.
            const auto shape_2d_pitched = Shape4<i64>{1, shape[0] * shape[1] * shape[2], shape[3], 1};
            return copy(src, src_strides[3], dst, dst_strides[3], shape_2d_pitched, stream);

        } else {
            NOA_THROW("Copying strided regions, other than in the height dimension, between an unregistered "
                      "host region and a device is not supported, yet. Hint: copy the strided data to a temporary "
                      "contiguous buffer");
        }
    }
}

// -- CUDA arrays -- //
namespace noa::cuda::memory {
    template<typename T>
    inline void copy(const T* src, i64 src_pitch, cudaArray* dst, const Shape3<i64>& shape) {
        cudaMemcpy3DParms params = details::to_copy_parameters(src, src_pitch, dst, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    template<typename T>
    inline void copy(const T* src, i64 src_pitch, cudaArray* dst, const Shape3<i64>& shape, Stream& stream) {
        cudaMemcpy3DParms params = details::to_copy_parameters(src, src_pitch, dst, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    template<typename T>
    inline void copy(const cudaArray* src, T* dst, i64 dst_pitch, const Shape3<i64>& shape) {
        cudaMemcpy3DParms params = details::to_copy_parameters(src, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    template<typename T>
    inline void copy(const cudaArray* src, T* dst, i64 dst_pitch, const Shape3<i64>& shape, Stream& stream) {
        cudaMemcpy3DParms params = details::to_copy_parameters(src, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    // Copy an array into a CUDA array.
    // The source can be on any device or on the host.
    // The source BDHW shape should match the shape of the CUDA array such that:
    //  - If the CUDA array is layered, its shape should match the BHW dimensions of the source.
    //  - If the CUDA array is NOT layered, its shape should match the DHW dimensions of the source.
    template<typename T>
    void copy(const T* src, const Strides4<i64>& src_strides, cudaArray* dst,
              const Shape4<i64>& shape, Stream& stream) {
        const auto[desc_, actual_extent, flags] = AllocatorArray<T>::info(dst);
        const bool is_layered = flags & cudaArrayLayered;
        const cudaExtent expected_extent = AllocatorArray<T>::shape2extent(shape, is_layered);

        NOA_CHECK(expected_extent.depth == actual_extent.depth &&
                  expected_extent.height == actual_extent.height &&
                  expected_extent.width == actual_extent.width,
                  "The input shape is not compatible with the output CUDA array shape");

        // cudaExtent for CUDA array has empty dimensions equal to 0.
        // However, for cudaMemcpy3D, dimensions equal to 0 are invalid.
        auto shape_3d = Shape3<i64>{expected_extent.depth, expected_extent.height, expected_extent.width};
        shape_3d += Shape3<i64>(shape_3d == 0);

        const bool is_column = shape[2] >= 1 && shape[3] == 1;
        const auto src_strides_3d = Strides3<i64>{
                src_strides[!is_layered],
                src_strides[2 + is_column],
                src_strides[3 - is_column]};
        const bool is_rightmost = noa::indexing::is_rightmost(src_strides_3d);
        const bool has_valid_pitch = src_strides_3d[1] >= shape_3d[2];
        const bool is_contiguous_2 = src_strides_3d[2] == 1;
        const bool is_contiguous_0 = src_strides_3d[0] == src_strides_3d[1] * shape_3d[1];
        NOA_CHECK(is_rightmost && has_valid_pitch && is_contiguous_0 && is_contiguous_2,
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

        NOA_CHECK(expected_extent.depth == actual_extent.depth &&
                  expected_extent.height == actual_extent.height &&
                  expected_extent.width == actual_extent.width,
                  "The output shape is not compatible with the input CUDA array shape");

        // cudaExtent for CUDA array has empty dimensions equal to 0.
        // However, for cudaMemcpy3D, dimensions equal to 0 are invalid.
        auto shape_3d = Shape3<i64>{expected_extent.depth, expected_extent.height, expected_extent.width};
        shape_3d += Shape3<i64>(shape_3d == 0);

        const bool is_column = shape[2] >= 1 && shape[3] == 1;
        const auto dst_strides_3d = Strides3<i64>{
                dst_strides[!is_layered],
                dst_strides[2 + is_column],
                dst_strides[3 - is_column]};
        const bool is_rightmost = noa::indexing::is_rightmost(dst_strides_3d);
        const bool has_valid_pitch = dst_strides_3d[1] >= shape_3d[2];
        const bool is_contiguous_2 = dst_strides_3d[2] == 1;
        const bool is_contiguous_0 = dst_strides_3d[0] == dst_strides_3d[1] * shape_3d[1];
        NOA_CHECK(is_rightmost && has_valid_pitch && is_contiguous_0 && is_contiguous_2,
                  "Input layout cannot be copied into a CUDA array. The input should be in the rightmost order, "
                  "and its {} and width dimension should be contiguous, but got shape {} and strides {}",
                  is_layered ? "batch" : "depth", shape, dst_strides);

        copy(src, dst, dst_strides[2], shape_3d, stream);
    }
}
