#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/memory/PtrArray.h"

// TODO Add nvrtc to support any type.

// Since we assume Compute Capability >= 2.0, all devices support the Unified Virtual Address Space, so
// the CUDA driver can determine, for each pointer, where the data is located, and one does not have to
// specify the cudaMemcpyKind. In the documentation they don't explicitly say that cudaMemcpyDefault allows
// for concurrent transfers between host and device if the host is pinned, but why would it make a difference?

namespace noa::cuda::memory::details {
    template<typename T>
    inline cudaMemcpy3DParms toParams(const T* src, dim_t src_pitch, T* dst, dim_t dst_pitch, dim4_t shape) {
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), src_pitch * sizeof(T), shape[3], shape[2]};
        params.dstPtr = {dst, dst_pitch * sizeof(T), shape[3], shape[2]};
        params.extent = {shape[3] * sizeof(T), shape[2], shape[1] * shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    inline cudaMemcpy3DParms toParams(const cudaArray* src, T* dst, dim_t dst_pitch, dim3_t shape) {
        cudaMemcpy3DParms params{};
        params.srcArray = const_cast<cudaArray*>(src);
        params.dstPtr = {dst, dst_pitch * sizeof(T), shape[2], shape[1]};
        params.extent = {shape[2], shape[1], shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    inline cudaMemcpy3DParms toParams(const T* src, dim_t src_pitch, cudaArray* dst, dim3_t shape) {
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), src_pitch * sizeof(T), shape[2], shape[1]};
        params.dstArray = dst;
        params.extent = {shape[2], shape[1], shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    // Copy strided data between two pointers accessible by the stream's device.
    // No reordering is done here.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void copy(const T* src, dim4_t src_strides,
              T* dst, dim4_t dst_strides,
              dim4_t shape, Stream& stream);
}

namespace noa::cuda::memory {
    // Copies contiguous memory from one region to another. These can point to host or device memory.
    // If a stream is passed, copy is enqueued to the stream. In this case, passing raw pointers one must make
    // sure the memory stay valid until completion. Copies between host and device can execute concurrently only
    // if the copy involves pinned memory.

    template<typename T>
    inline void copy(const T* src, T* dst, dim_t elements) {
        const auto count = elements * sizeof(T);
        NOA_THROW_IF(cudaMemcpy(dst, src, count, cudaMemcpyDefault));
    }

    template<typename T>
    inline void copy(const T* src, T* dst, dim_t elements, Stream& stream) {
        const auto count = elements * sizeof(T);
        NOA_THROW_IF(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream.id()));
    }

    template<typename T>
    inline void copy(const shared_t<T[]>& src, const shared_t<T[]>& dst, dim_t elements, Stream& stream) {
        const auto count = elements * sizeof(T);
        NOA_THROW_IF(cudaMemcpyAsync(dst.get(), src.get(), count, cudaMemcpyDefault, stream.id()));
        stream.attach(src, dst);
    }

    // Copies asynchronously pitched memory from one region to another. These can point to host or device memory.
    // If a stream is passed, copy is enqueued to the stream. In this case, passing raw pointers one must make
    // sure the memory stay valid until completion. Copies between host and device can execute concurrently only
    // if the copy involves pinned memory.

    template<typename T>
    inline void copy(const T* src, dim_t src_pitch, T* dst, dim_t dst_pitch, dim4_t shape, Stream& stream) {
        const auto params = details::toParams(src, src_pitch, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    template<typename T>
    inline void copy(const shared_t<T[]>& src, dim_t src_pitch,
                     const shared_t<T[]>& dst, dim_t dst_pitch,
                     dim4_t shape, Stream& stream) {
        const auto params = details::toParams(src.get(), src_pitch, dst.get(), dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
        stream.attach(src, dst);
    }

    // Copies asynchronously regions of (strided/padded) memory.
    // Contiguous regions of memory have no copy restrictions, as well as (batched) row vectors (and column
    // vectors if SWAP_LAYOUT is true), and regions with padding on the right side of the innermost
    // dimension (referred to as a "pitch" in CUDA).
    // However, if the contiguity is broken in any other dimension, an error will be thrown if:
    // 1) the source and destination are on different devices,
    // 2) the copy is between unregistered host memory and a device,
    // 3) the copy involves a device that is not the stream's device.
    // If the copy involves an unregistered memory regions, the stream will be synchronized when the function
    // returns. Otherwise, this function can be asynchronous relative to the host and may return before completion.
    template<bool SWAP_LAYOUT = true, typename T>
    void copy(const T* src, dim4_t src_strides,
              T* dst, dim4_t dst_strides,
              dim4_t shape, Stream& stream) {
        if constexpr (SWAP_LAYOUT) {
            const auto order = indexing::order(dst_strides, shape);
            shape = indexing::reorder(shape, order);
            src_strides = indexing::reorder(src_strides, order);
            dst_strides = indexing::reorder(dst_strides, order);
        }

        const bool4_t is_contiguous = indexing::isContiguous(src_strides, shape) &&
                                      indexing::isContiguous(dst_strides, shape);

        // If contiguous or with a pitch (as defined in CUDA, i.e. padding the width on the right side),
        // then we can rely on the CUDA runtime. This should be 99% of cases. If SWAP_LAYOUT, F-contiguous
        // arrays are swapped to C-contiguous.
        if (is_contiguous[0] && is_contiguous[1] && is_contiguous[3]) {
            if (is_contiguous[2]) {
                return copy(src, dst, shape.elements(), stream);
            } else { // 2d pitch
                return copy(src, src_strides[2], dst, dst_strides[2], shape, stream);
            }
        }

        // Otherwise:
        static_assert(cudaMemoryTypeUnregistered == 0);
        static_assert(cudaMemoryTypeHost == 1);
        static_assert(cudaMemoryTypeDevice == 2);
        static_assert(cudaMemoryTypeManaged == 3);
        const cudaPointerAttributes src_attr = cuda::utils::getAttributes(src);
        const cudaPointerAttributes dst_attr = cuda::utils::getAttributes(dst);

        if (src_attr.type == 2 && dst_attr.type == 2) {
            // Both regions are on the same device, we can therefore launch our copy kernel.
            if (src_attr.device == dst_attr.device) {
                if constexpr (traits::is_restricted_data_v<T>)
                    details::copy(src, src_strides, dst, dst_strides, shape, stream);
                else
                    NOA_THROW("Copying strided regions, other than in the second-most dimension, "
                              "is not supported for type {}", string::human<T>());
            } else {
                NOA_THROW("Copying strided regions, other than in the second-most dimension, "
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
                NOA_THROW("Copying strided regions, other than in the second-most dimension, "
                          "from or to a device that is not the stream's device is not supported");

            // FIXME For managed pointers, use cudaMemPrefetchAsync()?
            if constexpr (traits::is_restricted_data_v<T>)
                details::copy(reinterpret_cast<const T*>(src_attr.devicePointer), src_strides,
                              reinterpret_cast<T*>(dst_attr.devicePointer), dst_strides,
                              shape, stream);
            else
                NOA_THROW("Copying strided regions, other than in the second-most dimension, "
                          "is not supported for type {}", string::human<T>());

        } else if ((src_attr.type <= 1 || src_attr.type == 3) &&
                   (dst_attr.type <= 1 || dst_attr.type == 3)) {
            // Both can be accessed on the host.
            NOA_ASSERT(!indexing::isOverlap(src, src_strides, dst, dst_strides, shape));
            const AccessorRestrict<const T, 4, dim_t> src_accessor(src, src_strides);
            const AccessorRestrict<T, 4, dim_t> dst_accessor(dst, dst_strides);
            stream.synchronize(); // FIXME Use a callback instead?
            for (dim_t i = 0; i < shape[0]; ++i)
                for (dim_t j = 0; j < shape[1]; ++j)
                    for (dim_t k = 0; k < shape[2]; ++k)
                        for (dim_t l = 0; l < shape[3]; ++l)
                            dst_accessor(i, j, k, l) = src_accessor(i, j, k, l);

        } else if (is_contiguous[0] && is_contiguous[1] && is_contiguous[2]) {
            // Last resort for strided row-vector(s). Since 3 first dimensions are contiguous, collapse them.
            // Non-contiguous row vector can be reshaped to a 2D pitch array so that it can be passed to the CUDA API.
            // If SWAP_LAYOUT, this works for column vectors as well.
            // Note: This is the last resort because it is much less efficient than our custom copy (on host or device),
            // so this is only if the copy is between unregister host and device.
            const dim4_t shape_{1, shape[0] * shape[1] * shape[2], shape[3], 1};
            return copy(src, src_strides[3], dst, dst_strides[3], shape_, stream);

        } else {
            NOA_THROW("Copying strided regions, other than in the second-most dimension, between an unregistered "
                      "host region and a device is not supported, yet. Hint: copy the strided data to a temporary "
                      "contiguous buffer");
        }
    }

    // Same as above, but making sure the memory regions stay valid until completion.
    template<bool SWAP_LAYOUT = true, typename T>
    void copy(const shared_t<T[]>& src, dim4_t src_strides,
              const shared_t<T[]>& dst, dim4_t dst_strides,
              dim4_t shape, Stream& stream) {
        copy<SWAP_LAYOUT>(src.get(), src_strides, dst.get(), dst_strides, shape, stream);
        stream.attach(src, dst);
    }
}

// -- CUDA arrays -- //
namespace noa::cuda::memory {
    template<typename T>
    inline void copy(const T* src, dim_t src_pitch, cudaArray* dst, dim3_t shape) {
        cudaMemcpy3DParms params = details::toParams(src, src_pitch, dst, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    template<typename T>
    inline void copy(const T* src, dim_t src_pitch, cudaArray* dst, dim3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = details::toParams(src, src_pitch, dst, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    template<typename T>
    inline void copy(const cudaArray* src, T* dst, dim_t dst_pitch, dim3_t shape) {
        cudaMemcpy3DParms params = details::toParams(src, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    template<typename T>
    inline void copy(const cudaArray* src, T* dst, dim_t dst_pitch, dim3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = details::toParams(src, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    template<typename T>
    inline void copy(const shared_t<T[]>& src, dim_t src_pitch,
                     const shared_t<cudaArray>& dst, dim3_t shape, Stream& stream) {
        copy(src.get(), src_pitch, dst.get(), shape, stream);
        stream.attach(src, dst);
    }

    template<typename T>
    inline void copy(const shared_t<cudaArray>& src,
                     const shared_t<T[]>& dst, dim_t dst_pitch,
                     dim3_t shape, Stream& stream) {
        copy(src.get(), dst.get(), dst_pitch, shape, stream);
        stream.attach(src, dst);
    }

    // Copy an array into a CUDA array.
    // The source can be on any device or on the host.
    // The source BDHW shape should match the shape of the CUDA array such that:
    //  - If the CUDA array is layered, its shape should match the BHW dimensions of the source.
    //  - If the CUDA array is NOT layered, its shape should match the DHW dimensions of the source.
    template<typename T>
    void copy(const T* src, dim4_t src_strides, cudaArray* dst, dim4_t shape, Stream& stream) {
        const auto[desc_, actual_extent, flags] = PtrArray<T>::info(dst);
        const bool is_layered = flags & cudaArrayLayered;
        const cudaExtent expected_extent = PtrArray<T>::shape2extent(shape, is_layered);

        NOA_CHECK(expected_extent.depth == actual_extent.depth &&
                  expected_extent.height == actual_extent.height &&
                  expected_extent.width == actual_extent.width,
                  "The input shape is not compatible with the output CUDA array shape");

        // cudaExtent for CUDA array has empty dimensions equal to 0.
        // However, for cudaMemcpy3D, dimensions equal to 0 are invalid.
        dim3_t shape_3d{expected_extent.depth, expected_extent.height, expected_extent.width};
        shape_3d += dim3_t(shape_3d == 0);

        const bool is_column = shape[2] >= 1 && shape[3] == 1;
        const dim3_t src_strides_3d{src_strides[!is_layered], src_strides[2 + is_column], src_strides[3 - is_column]};
        const bool is_rightmost = indexing::isRightmost(src_strides_3d);
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
    void copy(const shared_t<T[]>& src, dim4_t src_strides,
              const shared_t<cudaArray>& dst, dim4_t shape, Stream& stream) {
        copy(src.get(), src_strides, dst.get(), shape, stream);
        stream.attach(src, dst);
    }

    template<typename T>
    void copy(cudaArray* src, T* dst, dim4_t dst_strides, dim4_t shape, Stream& stream) {
        const auto[desc_, actual_extent, flags] = PtrArray<T>::info(src);
        const bool is_layered = flags & cudaArrayLayered;
        const cudaExtent expected_extent = PtrArray<T>::shape2extent(shape, is_layered);

        NOA_CHECK(expected_extent.depth == actual_extent.depth &&
                  expected_extent.height == actual_extent.height &&
                  expected_extent.width == actual_extent.width,
                  "The output shape is not compatible with the input CUDA array shape");

        // cudaExtent for CUDA array has empty dimensions equal to 0.
        // However, for cudaMemcpy3D, dimensions equal to 0 are invalid.
        dim3_t shape_3d{expected_extent.depth, expected_extent.height, expected_extent.width};
        shape_3d += dim3_t(shape_3d == 0);

        const bool is_column = shape[2] >= 1 && shape[3] == 1;
        const dim3_t dst_strides_3d{dst_strides[!is_layered], dst_strides[2 + is_column], dst_strides[3 - is_column]};
        const bool is_rightmost = indexing::isRightmost(dst_strides_3d);
        const bool has_valid_pitch = dst_strides_3d[1] >= shape_3d[2];
        const bool is_contiguous_2 = dst_strides_3d[2] == 1;
        const bool is_contiguous_0 = dst_strides_3d[0] == dst_strides_3d[1] * shape_3d[1];
        NOA_CHECK(is_rightmost && has_valid_pitch && is_contiguous_0 && is_contiguous_2,
                  "Input layout cannot be copied into a CUDA array. The input should be in the rightmost order, "
                  "and its {} and width dimension should be contiguous, but got shape {} and strides {}",
                  is_layered ? "batch" : "depth", shape, dst_strides);

        copy(src, dst, dst_strides[2], shape_3d, stream);
    }

    template<typename T>
    void copy(const shared_t<cudaArray>& src,
              const shared_t<T[]>& dst, dim4_t dst_strides,
              dim4_t shape, Stream& stream) {
        copy(src.get(), dst.get(), dst_strides, shape, stream);
        stream.attach(src, dst);
    }
}
