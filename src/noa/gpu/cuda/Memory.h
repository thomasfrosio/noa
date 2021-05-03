/**
 * @file noa/gpu/cuda/Memory.h
 * @brief Memory related function.
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#include "noa/Definitions.h"
#include "noa/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Memory::Details {
    template<typename T>
    NOA_IH cudaMemcpy3DParms toParams(const T* src, size_t pitch_src, T* dst, size_t pitch_dst, size3_t shape) {
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), pitch_src, shape.x, shape.y};
        params.dstPtr = {dst, pitch_dst, shape.x, shape.y};
        params.extent = {shape.x * sizeof(T), shape.y, shape.z};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    NOA_IH cudaMemcpy3DParms toParams(const cudaArray* src, T* dst, size_t pitch_dst, size3_t shape) {
        cudaMemcpy3DParms params{};
        params.srcArray = const_cast<cudaArray*>(src);
        params.dstPtr = {dst, pitch_dst, shape.x, shape.y};
        params.extent = {shape.x, shape.y, shape.z}; // an array is involved, so shape in elements.
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    NOA_IH cudaMemcpy3DParms toParams(const T* src, size_t pitch_src, cudaArray* dst, size3_t shape) {
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), pitch_src, shape.x, shape.y};
        params.dstArray = dst;
        params.extent = {shape.x, shape.y, shape.z};
        params.kind = cudaMemcpyDefault;
        return params;
    }
}

/** Memory related functions. */
namespace Noa::CUDA::Memory {
    // Since we assume Compute Capability >= 2.0, all devices support the Unified Virtual Address Space, so
    // the CUDA driver can determine, for each pointer, where the data is located, and one does not have to
    // specify the cudaMemcpyKind. In the documentation they don't explicitly say that cudaMemcpyDefault allows
    // for concurrent transfers between host and device if the host is pinned, but why would it make a difference?

    /* ------------------------- */
    /* --- Contiguous memory --- */
    /* ------------------------- */

    /**
     * Copies synchronously contiguous memory from one region to another. These can point to host or device memory.
     * @param[in] src   Source. Contiguous memory either on the host or on the device.
     * @param[out] dst  Destination. Contiguous memory either on the host or on the device.
     * @param bytes     How many bytes to copy.
     *
     * @note This function can be used to copy padded memory if both regions have the same shape and pitch.
     *       If the padded memory is managed by PtrDevicePadded, one should use PtrDevicePadded::bytesPadded() to
     *       specify the @a bytes argument.
     */
    NOA_IH void copy(const void* src, void* dst, size_t bytes) {
        NOA_PROFILE_FUNCTION();
        NOA_THROW_IF(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
    }

    /**
     * Copies asynchronously contiguous memory from one region to another. These can point to host or device memory.
     * @note The copy is enqueued to @a stream. Therefore, this function runs asynchronously with respect to the host
     *       and may return before the copy is complete. Memory copies between host and device can execute concurrently
     *       only if @a src or @a dst is pinned.
     */
    NOA_IH void copy(const void* src, void* dst, size_t bytes, Stream& stream) {
        NOA_THROW_IF(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream.id()));
    }

    /* --------------------- */
    /* --- Padded memory --- */
    /* --------------------- */

    /**
     * Copies memory with a given physical @a shape from @a src to @a dst.
     * @param[in] src   Source.
     * @param pitch_src Pitch, in bytes, of @a src.
     * @param[out] dst  Destination.
     * @param pitch_dst Pitch, in bytes, of @a dst.
     * @param shape     Logical {fast, medium, slow} shape to copy. In total, `getElements(shape) * sizeof(T)`
     *                  bytes are copied. i.e. padded regions are of course excluded from the copy.
     *
     * @note If @a pitch_src == @a pitch_dst, then one could use the function overloads above
     *       and copies a contiguous block of memory.
     * @note The order of the last 2 dimensions of the @a shape does not matter. What matters is the number of
     *       total rows. As such, when working with batches, simply multiply one of the last 2 dimensions by the
     *       batch number.
     */
    template<typename T>
    NOA_IH void copy(const T* src, size_t pitch_src, T* dst, size_t pitch_dst, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        cudaMemcpy3DParms params = Details::toParams(src, pitch_src, dst, pitch_dst, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /**
     * Copies asynchronously memory with a given physical @a shape from @a src to @a dst.
     * @note The copy is enqueued to @a stream. Therefore, this function runs asynchronously with respect to the host
     *       and may return before the copy is complete. Memory copies between host and device can execute concurrently
     *       only if @a src or @a dst is pinned.
     */
    template<typename T>
    NOA_IH void copy(const T* src, size_t pitch_src, T* dst, size_t pitch_dst, size3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = Details::toParams(src, pitch_src, dst, pitch_dst, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /* ----------------------------------------- */
    /* --- CUDA arrays and contiguous memory --- */
    /* ----------------------------------------- */

    /**
     * Copies a CUDA array with a given physical @a shape into @a dst.
     * @param[in] src   N dimensional CUDA array. Should correspond to @a shape. All elements will be copied.
     * @param[out] dst  Contiguous memory. Should be large enough to contain @a src.
     * @param shape     Physical {fast, medium, slow} shape to copy. In total, `getElements(shape) * sizeof(T)`
     *                  bytes are copied.
     */
    template<typename T>
    NOA_IH void copy(const cudaArray* src, T* dst, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        cudaMemcpy3DParms params = Details::toParams(src, dst, shape.x * sizeof(T), shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /**
     * Copies asynchronously a CUDA array with a given physical @a shape into @a dst.
     * @note The copy is enqueued to @a stream. Therefore, this function runs asynchronously with respect to the host
     *       and may return before the copy is complete. Memory copies between host and device can execute concurrently
     *       only if @a dst is pinned.
     */
    template<typename T>
    NOA_IH void copy(const cudaArray* src, T* dst, size3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = Details::toParams(src, dst, shape.x * sizeof(T), shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /**
     * Copies memory with a given physical @a shape into the CUDA array @a dst.
     * @param[in] src   Contiguous memory. Should correspond or be larger than @a shape.
     * @param[out] dst  N dimensional CUDA array. Should correspond to @a shape. All elements will be filled.
     * @param shape     Physical {fast, medium, slow} shape to copy. In total, `getElements(shape) * sizeof(T)`
     *                  bytes are copied.
     */
    template<typename T>
    NOA_IH void copy(const T* src, cudaArray* dst, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        cudaMemcpy3DParms params = Details::toParams(src, shape.x * sizeof(T), dst, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /**
     * Copies memory with a given physical @a shape into the CUDA array @a dst.
     * @note The copy is enqueued to @a stream. Therefore, this function runs asynchronously with respect to the host
     *       and may return before the copy is complete. Memory copies between host and device can execute concurrently
     *       only if @a src is pinned.
     */
    template<typename T>
    NOA_IH void copy(const T* src, cudaArray* dst, size3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = Details::toParams(src, shape.x * sizeof(T), dst, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /* ------------------------------------- */
    /* --- CUDA arrays and padded memory --- */
    /* ------------------------------------- */

    /**
     * Copies a CUDA array with a given physical @a shape into @a dst.
     * @param[in] src   N dimensional CUDA array. Should correspond to @a shape. All elements will be copied.
     * @param[out] dst  Should be large enough to contain @a src.
     * @param pitch_dst Pitch, in bytes, of @a dst.
     * @param shape     Physical {fast, medium, slow} shape to copy. In total, `getElements(shape) * sizeof(T)`
     *                  bytes are copied.
     */
    template<typename T>
    NOA_IH void copy(const cudaArray* src, T* dst, size_t pitch_dst, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        cudaMemcpy3DParms params = Details::toParams(src, dst, pitch_dst, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /**
     * Copies asynchronously a CUDA array with a given physical @a shape into @a dst.
     * @note The copy is enqueued to @a stream. Therefore, this function runs asynchronously with respect to the host
     *       and may return before the copy is complete. Memory copies between host and device can execute concurrently
     *       only if @a dst is pinned.
     */
    template<typename T>
    NOA_IH void copy(const cudaArray* src, T* dst, size_t pitch_dst, size3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = Details::toParams(src, dst, pitch_dst, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /**
     * Copies memory with a given physical @a shape into the CUDA array @a dst.
     * @param[in] src   Should correspond or be larger than @a shape.
     * @param pitch_src Pitch, in bytes, of @a src.
     * @param[out] dst  N dimensional CUDA array. Should correspond to @a shape. All elements will be filled.
     * @param shape     Physical {fast, medium, slow} shape to copy. In total, `getElements(shape) * sizeof(T)`
     *                  bytes are copied.
     */
    template<typename T>
    NOA_IH void copy(const T* src, size_t pitch_src, cudaArray* dst, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        cudaMemcpy3DParms params = Details::toParams(src, pitch_src, dst, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /**
     * Copies memory with a given physical @a shape into the CUDA array @a dst.
     * @note The copy is enqueued to @a stream. Therefore, this function runs asynchronously with respect to the host
     *       and may return before the copy is complete. Memory copies between host and device can execute concurrently
     *       only if @a src is pinned.
     */
    template<typename T>
    NOA_IH void copy(const T* src, size_t pitch_src, cudaArray* dst, size3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = Details::toParams(src, pitch_src, dst, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }
}
