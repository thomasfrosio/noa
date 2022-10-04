#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/util/Pointers.h"

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
        params.extent = {shape[2], shape[1], shape[0]}; // an array is involved, so shape in elements.
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
        const cudaPointerAttributes src_attr = cuda::util::getAttributes(src);
        const cudaPointerAttributes dst_attr = cuda::util::getAttributes(dst);

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
            // 1) Managed memory has no restrictions.
            // 2) Device memory must belong to the stream's device.
            // 3) Pinned memory should be accessed via their device pointer.
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
    inline void copy(const T* src, dim_t src_pitch,
                     cudaArray* dst, dim3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = details::toParams(src, src_pitch, dst, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    template<typename T>
    inline void copy(const shared_t<T[]>& src, dim_t src_pitch,
                     const shared_t<cudaArray>& dst, dim3_t shape, Stream& stream) {
        copy(src.get(), src_pitch, dst.get(), shape, stream);
        stream.attach(src, dst);
    }

    template<typename T>
    inline void copy(const cudaArray* src, T* dst, dim_t dst_pitch, dim3_t shape) {
        cudaMemcpy3DParms params = details::toParams(src, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    template<typename T>
    inline void copy(const cudaArray* src,
                     T* dst, dim_t dst_pitch,
                     dim3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = details::toParams(src, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    template<typename T>
    inline void copy(const shared_t<cudaArray>& src,
                     const shared_t<T[]>& dst, dim_t dst_pitch,
                     dim3_t shape, Stream& stream) {
        copy(src.get(), dst.get(), dst_pitch, shape, stream);
        stream.attach(src, dst);
    }
}
