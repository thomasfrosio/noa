/// \file noa/gpu/cuda/memory/Copy.h
/// \brief Copy from/to device memory.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021
#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/util/Pointers.h"

// Since we assume Compute Capability >= 2.0, all devices support the Unified Virtual Address Space, so
// the CUDA driver can determine, for each pointer, where the data is located, and one does not have to
// specify the cudaMemcpyKind. In the documentation they don't explicitly say that cudaMemcpyDefault allows
// for concurrent transfers between host and device if the host is pinned, but why would it make a difference?

namespace noa::cuda::memory::details {
    template<typename T>
    NOA_IH cudaMemcpy3DParms toParams(const T* src, size_t src_pitch, T* dst, size_t dst_pitch, size4_t shape) {
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), src_pitch * sizeof(T), shape[3], shape[2]};
        params.dstPtr = {dst, dst_pitch * sizeof(T), shape[3], shape[2]};
        params.extent = {shape[3] * sizeof(T), shape[2], shape[1] * shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    NOA_IH cudaMemcpy3DParms toParams(const cudaArray* src, T* dst, size_t dst_pitch, size3_t shape) {
        cudaMemcpy3DParms params{};
        params.srcArray = const_cast<cudaArray*>(src);
        params.dstPtr = {dst, dst_pitch * sizeof(T), shape[2], shape[1]};
        params.extent = {shape[2], shape[1], shape[0]}; // an array is involved, so shape in elements.
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    NOA_IH cudaMemcpy3DParms toParams(const T* src, size_t src_pitch, cudaArray* dst, size3_t shape) {
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), src_pitch * sizeof(T), shape[2], shape[1]};
        params.dstArray = dst;
        params.extent = {shape[2], shape[1], shape[0]};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    // Copy strided data between two pointers accessible by the stream's device.
    template<typename T>
    NOA_HOST void copy(const T* src, size4_t src_stride, T* dst, size4_t dst_stride, size4_t shape, Stream& stream);
}

namespace noa::cuda::memory {
    /// Copies synchronously contiguous memory from one region to another. These can point to host or device memory.
    /// \param[in] src      Source. Contiguous memory either on the host or on the device.
    /// \param[out] dst     Destination. Contiguous memory either on the host or on the device.
    /// \param elements     Elements to copy.
    template<typename T>
    NOA_IH void copy(const T* src, T* dst, size_t elements) {
        NOA_PROFILE_FUNCTION();
        NOA_THROW_IF(cudaMemcpy(dst, src, elements * sizeof(T), cudaMemcpyDefault));
    }

    /// Copies asynchronously contiguous memory from one region to another. These can point to host or device memory.
    /// \param[in] src          Source. Contiguous memory either on the host or on the device.
    /// \param[out] dst         Destination. Contiguous memory either on the host or on the device.
    /// \param elements         Elements to copy.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function can be asynchronous relative to the host and may return before completion.
    /// \note Memory copies between host and device can execute concurrently only if \p src or \p dst are pinned.
    template<typename T>
    NOA_IH void copy(const T* src, T* dst, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_THROW_IF(cudaMemcpyAsync(dst, src, elements * sizeof(T), cudaMemcpyDefault, stream.id()));
    }

    /// Copies asynchronously regions of (strided and/or padded) memory.
    /// \details Contiguous regions of memory have no copy restrictions, as well as regions with padding at the right
    ///          side of the innermost dimension (referred to as a "pitch" in CUDA). However, if there's any padding
    ///          or stride in the other dimensions, an error will be thrown if: 1) the source and destination are on
    ///          different devices, 2) the copy is between unregistered host memory and a device, 3) the copy involves
    ///          a device that is not the stream's device.
    ///
    /// \tparam T               Any data type.
    /// \param[in] src          Source. Can be on the host or a device.
    /// \param src_stride       Rightmost strides, in elements, of \p src.
    /// \param[out] dst         Destination. Can be on the host or a device.
    /// \param dst_stride       Rightmost strides, in elements, of \p dst.
    /// \param shape            Rightmost shape to copy.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note If the copy is between unregistered and/or pinned memory regions, the copy will be done synchronously
    ///       on the host and the stream will be synchronized when the function returns. Otherwise this function can
    ///       be asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void copy(const T* src, size4_t src_stride, T* dst, size4_t dst_stride, size4_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const bool4_t is_contiguous = isContiguous(src_stride, shape) && isContiguous(dst_stride, shape);

        // If contiguous or with a pitch (as defined in CUDA, i.e. padding at the right of the innermost dimension),
        // then we can rely on the CUDA runtime. This should be 99% or cases.
        if (is_contiguous[0] && is_contiguous[1] && is_contiguous[3]) {
            if (is_contiguous[2]) { // pitch == shape[3]
                copy(src, dst, shape.elements(), stream);
            } else {
                cudaMemcpy3DParms params = details::toParams(src, src_stride.pitches()[2],
                                                             dst, dst_stride.pitches()[2], shape);
                NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
            }
        } else {
            static_assert(cudaMemoryTypeUnregistered == 0);
            static_assert(cudaMemoryTypeHost == 1);
            static_assert(cudaMemoryTypeDevice == 2);
            static_assert(cudaMemoryTypeManaged == 3);
            const cudaPointerAttributes src_attr = cuda::util::getAttributes(src);
            const cudaPointerAttributes dst_attr = cuda::util::getAttributes(dst);

            if (src_attr.type == 2 && dst_attr.type == 2) {
                // Both regions are on the same device, we can therefore launch our copy kernel.
                if (src_attr.device == dst_attr.device)
                    details::copy(src, src_stride, dst, dst_stride, shape, stream);
                else
                    NOA_THROW("Copying strided regions, or padded regions other than in the innermost dimension, "
                              "between different devices is currently not supported. Trying to copy a shape of {} "
                              "from (device:{}, stride:{}) to (device:{}, stride:{}) ",
                              shape, src_attr.device, src_stride, dst_stride, dst_attr.device);

            } else if (src_attr.type <= 1 && dst_attr.type <= 1) {
                // Both on host.
                stream.synchronize();
                for (size_t i = 0; i < shape[0]; ++i)
                    for (size_t j = 0; j < shape[1]; ++j)
                        for (size_t k = 0; k < shape[2]; ++k)
                            for (size_t l = 0; l < shape[3]; ++l)
                                dst[at(i, j, k, l, dst_stride)] = src[at(i, j, k, l, src_stride)];

            } else if (src_attr.type >= 1 && dst_attr.type >= 1) {
                // A device or managed pointer is involved: copy on the device.
                // 1) Managed memory has no restrictions.
                // 2) Device memory must belong to the stream's device.
                // 3) Pinned memory should be accessed via their device pointer.
                // FIXME Managed memory can be accessed by any device.
                if ((src_attr.type == 2 && src_attr.device != stream.device().id()) ||
                    (dst_attr.type == 2 && dst_attr.device != stream.device().id()))
                    NOA_THROW("Copying strided regions, or padded regions other than in the innermost dimension, "
                              "from or to a device that is not the stream's device");

                // FIXME For managed pointers, use cudaMemPrefetchAsync()?
                details::copy(reinterpret_cast<const T*>(src_attr.devicePointer), src_stride,
                              reinterpret_cast<T*>(dst_attr.devicePointer), dst_stride,
                              shape, stream);

            } else {
                NOA_THROW("Copying strided regions, or padded regions other than in the innermost dimension, between "
                          "an unregistered host region and a device is not supported, yet. Hint: copy the strided data "
                          "to a temporary contiguous buffer");
            }
        }
    }
}

// -- CUDA arrays -- //
namespace noa::cuda::memory {
    /// Copies memory region into a CUDA array.
    /// \param[in] src      Region to copy.
    /// \param src_pitch    Pitch, in elements, of the innermost dimension of \p src.
    /// \param[out] dst     N dimensional CUDA array. Should correspond to \p shape. All elements will be filled.
    /// \param shape        Rightmost shape of the CUDA array.
    template<typename T>
    NOA_IH void copy(const T* src, size_t src_pitch, cudaArray* dst, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        cudaMemcpy3DParms params = details::toParams(src, src_pitch, dst, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /// Copies memory region into a CUDA array.
    /// \param[in] src          Region to copy.
    /// \param src_pitch        Pitch, in elements, of the innermost dimension of \p src.
    /// \param[out] dst         N dimensional CUDA array. Should correspond to \p shape. All elements will be filled.
    /// \param shape            Rightmost shape to copy.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function can be asynchronous relative to the host and may return before completion.
    /// \note Memory copies between host and device can execute concurrently only if \p src is pinned.
    template<typename T>
    NOA_IH void copy(const T* src, size_t src_pitch, cudaArray* dst, size3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = details::toParams(src, src_pitch, dst, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /// Copies a CUDA array into a memory region.
    /// \param[in] src      N dimensional CUDA array. Should correspond to \p shape. All elements will be copied.
    /// \param[out] dst     Region to copy into.
    /// \param dst_pitch    Pitch, in elements, of the innermost dimension of \p dst.
    /// \param shape        Rightmost shape of the CUDA array.
    template<typename T>
    NOA_IH void copy(const cudaArray* src, T* dst, size_t dst_pitch, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        cudaMemcpy3DParms params = details::toParams(src, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /// Copies a CUDA array into a memory region.
    /// \param[in] src          N dimensional CUDA array. Should correspond to \p shape. All elements will be copied.
    /// \param[out] dst         Region to copy into.
    /// \param dst_pitch        Pitch, in elements, of the innermost dimension of \p dst.
    /// \param shape            Rightmost shape of the CUDA array.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function can be asynchronous relative to the host and may return before completion.
    /// \note Memory copies between host and device can execute concurrently only if \p dst is pinned.
    template<typename T>
    NOA_IH void copy(const cudaArray* src, T* dst, size_t dst_pitch, size3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = details::toParams(src, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }
}
