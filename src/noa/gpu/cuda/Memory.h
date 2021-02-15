/**
 * @file Memory.h
 * @brief Memory related function for CUDA.
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#include "noa/gpu/cuda/Stream.h"

/** Memory related functions. */
namespace Noa::CUDA::Memory {
    // Since we assume Compute Capability >= 2.0, all devices support the Unified Virtual Address Space, so
    // the CUDA driver can determine, for each pointer, where the data is located, and one does not have to
    // specify the cudaMemcpyKind.

    /**
     * Copies synchronously "linear" memory from one region to another. These can point to host or device memory.
     * @param[in] src   Source. Linear memory either on the host or on the device.
     * @param[out] dst  Destination. Linear memory either on the host or on the device.
     * @param bytes     How many bytes to copy.
     */
    NOA_IH void copy(void* dst, const void* src, size_t bytes) {
        NOA_THROW_IF(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
    }

    /**
     * Copies asynchronously "linear" memory from one region to another. These can point to host or device memory.
     * @param[in] src       Source. Linear memory either on the host or on the device.
     * @param[out] dst      Destination. Linear memory either on the host or on the device.
     * @param bytes         How many bytes to copy.
     * @param[in] stream    Stream on which to enqueue the copy.
     *
     * @note    This function will return immediately, as it is just enqueuing a memory copy to @a stream.
     * @note    Memory copies between host and device can execute concurrently only if @a dst or @a src is pinned.
     */
    NOA_IH void copy(void* dst, const void* src, size_t bytes, Stream& stream) {
        NOA_THROW(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream.id()));
    }

    /**
     * Copies asynchronously a region of @a src_shape into another region of @a dst_shape.
     * Only the data is copied, i.e. the padding is ignored.
     * These regions can point to host or device memory.
     *
     * @param[out] dst      Destination.
     * @param dst_pitch     Pitch of the @a dst, in bytes.
     * @param dst_shape     Shape of @a dst, in @a T elements. It should be greater or equal that @a src_shape.
     *
     * @param[in] src       Source.
     * @param src_pitch     Pitch of the @a src, in bytes.
     * @param src_shape     Shape of @a src, in @a T elements. It should be less or equal that @a dst_shape.
     *
     * @warning This function should not be used with cuda arrays.
     * @note    If @a dst and @a src have the same shape (and therefore the same pitch), "linear" copies can be used.
     * @note    If @a src or @a dst is "linear", their pitch should be equal to `shape.x * sizeof(T)`.
     */
    template<typename T, typename T>
    NOA_IH void copy(T* dst, size_t dst_pitch, size3_t dst_shape,
                     const T* src, size_t src_pitch, size3_t src_shape) {
        cudaMemcpy3DParms params{}; // zero initialize.
        params.srcPtr.ptr = src;
        params.srcPtr.pitch = src_pitch;
        params.srcPtr.xsize = src_shape.x; // in elements
        params.srcPtr.ysize = src_shape.y;

        params.dstPtr.ptr = dst;
        params.dstPtr.pitch = dst_pitch;
        params.dstPtr.xsize = dst_shape.x;
        params.dstPtr.ysize = dst_shape.y;

        params.extent.width = src_shape.x * sizeof(T);
        params.extent.height = src_shape.y;
        params.extent.depth = src_shape.z;
        params.kind = cudaMemcpyDefault;
        NOA_THROW(cudaMemcpy3D(&params));
    }

    template<typename T1, typename T2>
    NOA_IH void copy(T1* dst, size_t dst_pitch, size3_t dst_shape,
                     const T2* src, size_t src_pitch, size3_t src_shape, Stream& stream) {
        cudaMemcpy3DParms params{}; // zero initialize.
        params.srcPtr.ptr = src;
        params.srcPtr.pitch = src_pitch;    // bytes
        params.srcPtr.xsize = src_shape.x;  // elements
        params.srcPtr.ysize = src_shape.y;  // elements

        params.dstPtr.ptr = dst;
        params.dstPtr.pitch = dst_pitch;
        params.dstPtr.xsize = dst_shape.x;
        params.dstPtr.ysize = dst_shape.y;

        params.extent.width = src_shape.x * sizeof(T2); // bytes
        params.extent.height = src_shape.y;             // elements
        params.extent.depth = src_shape.z;              // elements
        params.kind = cudaMemcpyDefault;
        NOA_THROW(cudaMemcpy3DAsync(&params, stream.id()));
    }

    // Arrays
}
