/**
 * @file noa/gpu/cuda/Memory.h
 * @brief Memory related function.
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"
#include "noa/gpu/cuda/CudaRuntime.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"
#include "noa/gpu/cuda/PtrDevicePadded.h"
#include "noa/gpu/cuda/PtrArray.h"

/** Memory related functions. */
namespace Noa::CUDA::Memory {
    // Since we assume Compute Capability >= 2.0, all devices support the Unified Virtual Address Space, so
    // the CUDA driver can determine, for each pointer, where the data is located, and one does not have to
    // specify the cudaMemcpyKind. In the documentation they don't explicitly say that cudaMemcpyDefault allows
    // for concurrent transfers between host and device if the host is pinned, but is OK, isn't it?

    /**
     * Copies synchronously "linear" memory from one region to another. These can point to host or device memory.
     * @param[in] src   Source. Linear memory either on the host or on the device.
     * @param[out] dst  Destination. Linear memory either on the host or on the device.
     * @param bytes     How many bytes to copy.
     *
     * @note This function can be used to copy padded memory (i.e. managed by PtrDevicePadded) if both regions have
     *       the same shape (and therefore pitch). In this case, one should use the PtrDevicePadded::bytesPadded()
     *       returned value to specify the @a bytes argument.
     */
    NOA_IH void copy(void* dst, const void* src, size_t bytes) {
        NOA_THROW_IF(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
    }

    /**
     * Copies asynchronously "linear" memory from one region to another. These can point to host or device memory.
     * @param[in] src       Source. Linear memory either on the host or on the device.
     * @param[out] dst      Destination. Linear memory either on the host or on the device.
     * @param bytes         How many bytes to copy.
     * @param[out] stream   Stream on which to enqueue the copy.
     *
     * @note This function can be used to copy padded memory (i.e. managed by PtrDevicePadded) if both regions have
     *       the same shape (and therefore pitch). In this case, one should use the PtrDevicePadded::bytesPadded()
     *       returned value to specify the @a bytes argument.
     * @note This function runs asynchronously with respect to the host and may return before the copy is complete.
     *       Memory copies between host and device can execute concurrently only if @a src is pinned.
     */
    NOA_IH void copy(void* dst, const void* src, size_t bytes, Stream& stream) {
        NOA_THROW_IF(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream.id()));
    }

    /* --------------------------------- */
    /* --- Host to/from DevicePadded --- */
    /* --------------------------------- */

    /**
     * Fills (i.e. copies) the multidimensional padded memory of @a dst with the linear memory of @a src.
     * @param[out] dst  Destination. All elements will be filled. Padded region are of course excluded.
     * @param[in] src   Source. It should have enough elements to fill @a dst (i.e. @c dst.elements()).
     *                  Can be on host (pageable or pinned) or device memory.
     */
    template<typename T>
    NOA_IH void copy(PtrDevicePadded<T>* dst, const T* src) {
        size3_t dst_shape = dst->shape();
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), dst_shape.x * sizeof(T), dst_shape.x, dst_shape.y};
        params.dstPtr = {dst->get(), dst->pitch(), dst_shape.x, dst_shape.y};
        params.extent = {dst_shape.x * sizeof(T), dst_shape.y, dst_shape.z};
        params.kind = cudaMemcpyDefault;
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /**
     * Fills (i.e. copies) the multidimensional padded memory of @a dst with the linear memory of @a src.
     * @param[out] dst      Destination. All elements will be filled. Padded region are of course excluded.
     * @param[in] src       Source. It should have enough elements to fill @a dst (i.e. @c dst.elements()).
     *                      Can be on host (pageable or pinned) or device memory.
     * @param[out] stream   Stream on which to enqueue the copy.
     *
     * @note This function runs asynchronously with respect to the host and may return before the copy is complete.
     *       Memory copies between host and device can execute concurrently only if @a src is pinned.
     */
    template<typename T>
    NOA_IH void copy(PtrDevicePadded<T>* dst, const T* src, Stream& stream) {
        size3_t dst_shape = dst->shape();
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), dst_shape.x * sizeof(T), dst_shape.x, dst_shape.y};
        params.dstPtr = {dst->get(), dst->pitch(), dst_shape.x, dst_shape.y};
        params.extent = {dst_shape.x * sizeof(T), dst_shape.y, dst_shape.z};
        params.kind = cudaMemcpyDefault;
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /**
     * Fills (i.e. copies) the linear memory of @a dst with the multidimensional padded memory of @a src.
     * @param[out] dst  Destination. It should have enough elements to contain @a dst (i.e. @c dst.elements()).
     *                  Can be on host (pageable or pinned) or device memory.
     * @param[in] src   Source. All elements will be copied. Padded region are of course excluded.
     */
    template<typename T>
    NOA_IH void copy(T* dst, const PtrDevicePadded<T>* src) {
        size3_t src_shape = src->shape();
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src->get()), src->pitch(), src_shape.x, src_shape.y};
        params.dstPtr = {dst, src_shape.x * sizeof(T), src_shape.x, src_shape.y};
        params.extent = {src_shape.x * sizeof(T), src_shape.y, src_shape.z};
        params.kind = cudaMemcpyDefault;
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /**
     * Fills (i.e. copies) the linear memory of @a dst with the multidimensional padded memory of @a src.
     * @param[out] dst      Destination. It should have enough elements to contain @a dst (i.e. @c dst.elements()).
     *                      Can be on host (pageable or pinned) or device memory.
     * @param[in] src       Source. All elements will be copied. Padded region are of course excluded.
     * @param[out] stream   Stream on which to enqueue the copy.
     *
     * @note This function runs asynchronously with respect to the host and may return before the copy is complete.
     *       Memory copies between host and device can execute concurrently only if @a dst is pinned.
     */
    template<typename T>
    NOA_IH void copy(T* dst, const PtrDevicePadded<T>* src, Stream& stream) {
        size3_t src_shape = src->shape();
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src->get()), src->pitch(), src_shape.x, src_shape.y};
        params.dstPtr = {dst, src_shape.x * sizeof(T), src_shape.x, src_shape.y};
        params.extent = {src_shape.x * sizeof(T), src_shape.y, src_shape.z};
        params.kind = cudaMemcpyDefault;
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /* --------------------------------- */
    /* --- Host/Device to/from Array --- */
    /* --------------------------------- */

    /**
     * Fills (i.e. copies) the @a N dimensional CUDA array @a dst with the linear memory of @a src.
     * @param[out] dst  Destination. All elements will be filled.
     * @param[in] src   Source. It should have enough elements to fill @a dst.
     *                  Can be on host (pageable or pinned) or device memory.
     */
    template<typename T, uint N>
    NOA_IH void copy(PtrArray<T, N>* dst, const T* src) {
        cudaMemcpy3DParms params{};
        size3_t shape = dst->shape();
        params.srcPtr = {const_cast<T*>(src), shape.x * sizeof(T), shape.x, shape.y};
        params.dstArray = dst->get();
        params.extent = {shape.x, shape.y, shape.z};
        params.kind = cudaMemcpyDefault;
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /**
     * Fills (i.e. copies) the @a N dimensional CUDA array @a dst with the linear memory of @a src.
     * @param[out] dst      Destination. All elements will be filled.
     * @param[in] src       Source. It should have enough elements to fill @a dst.
     *                      Can be from host (pageable or pinned) or device memory.
     * @param[out] stream   Stream on which to enqueue the copy.
     *
     * @note This function runs asynchronously with respect to the host and may return before the copy is complete.
     *       Memory copies between host and device can execute concurrently only if @a src is pinned.
     */
    template<typename T, uint N>
    NOA_IH void copy(PtrArray<T, N>* dst, const T* src, Stream& stream) {
        cudaMemcpy3DParms params{};
        size3_t shape = dst->shape();
        params.srcPtr = {const_cast<T*>(src), shape.x * sizeof(T), shape.x, shape.y};
        params.dstArray = dst->get();
        params.extent = {shape.x, shape.y, shape.z};
        params.kind = cudaMemcpyDefault;
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.get()));
    }

    /**
     * Fills (i.e. copies) the linear memory of @a dst with the @a N dimensional CUDA array.
     * @param[out] dst  Destination. It should have enough elements to contain @a src.
     *                  Can be on host (pageable or pinned) or device memory.
     * @param[in] src   Source. All elements will be copied.
     */
    template<typename T, uint N>
    NOA_IH void copy(T* dst, const PtrArray<T, N>* src) {
        cudaMemcpy3DParms params{};
        size3_t shape = src->shape();
        params.srcArray = const_cast<cudaArray*>(src->get());
        params.dstPtr = {dst, shape.x * sizeof(T), shape.x, shape.y};
        params.extent = {shape.x, shape.y, shape.z};
        params.kind = cudaMemcpyDefault;
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /**
     * Fills (i.e. copies) the linear memory of @a dst with the @a N dimensional CUDA array.
     * @param[out] dst      Destination. It should have enough elements to contain @a src.
     *                      Can be on host (pageable or pinned) or device memory.
     * @param[in] src       Source. All elements will be copied.
     * @param[out] stream   Stream on which to enqueue the copy.
     *
     * @note This function runs asynchronously with respect to the host and may return before the copy is complete.
     *       Memory copies between host and device can execute concurrently only if @a dst is pinned.
     */
    template<typename T, uint N>
    NOA_IH void copy(T* dst, const PtrArray<T, N>* src, Stream& stream) {
        cudaMemcpy3DParms params{};
        size3_t shape = src->shape();
        params.srcArray = const_cast<cudaArray*>(src->get());
        params.dstPtr = {dst, shape.x * sizeof(T), shape.x, shape.y};
        params.extent = {shape.x, shape.y, shape.z};
        params.kind = cudaMemcpyDefault;
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /* ----------------------------- */
    /* --- DevicePadded to Array --- */
    /* ----------------------------- */

    /**
     * Fills (i.e. copies) the @a N dimensional CUDA array @a dst with the multidimensional padded memory of @a src.
     * @param[out] dst  Destination. All elements will be filled.
     * @param[in] src   Source. It should have the same shape as @a dst.
     */
    template<typename T, uint N>
    NOA_IH void copy(PtrArray<T, N>* dst, const PtrDevicePadded<T>* src) {
        size3_t shape = src->shape();
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src->get()), src->pitch(), shape.x, shape.y};
        params.dstArray = dst->get();
        params.extent = {shape.x, shape.y, shape.z};
        params.kind = cudaMemcpyDefault;
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /**
     * Fills (i.e. copies) the @a N dimensional CUDA array @a dst with the multidimensional padded memory of @a src.
     * @param[out] dst      Destination. All elements will be filled.
     * @param[in] src       Source. It should have the same shape as @a dst.
     * @param[out] stream   Stream on which to enqueue the copy.
     * @note This function runs asynchronously with respect to the host and may return before the copy is complete.
     */
    template<typename T, uint N>
    NOA_IH void copy(PtrArray<T, N>* dst, const PtrDevicePadded<T>* src, Stream& stream) {
        size3_t shape = src->shape();
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src->get()), src->pitch(), shape.x, shape.y};
        params.dstArray = dst->get();
        params.extent = {shape.x, shape.y, shape.z};
        params.kind = cudaMemcpyDefault;
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /**
     * Fills (i.e. copies) the multidimensional padded memory of @a dst with the @a N dimensional CUDA array @a src.
     * @param[out] dst  Destination. All elements will be filled.
     * @param[in] src   Source. It should have the same shape as @a dst.
     */
    template<typename T, uint N>
    NOA_IH void copy(PtrDevicePadded<T>* dst, const PtrArray<T, N>* src) {
        size3_t shape = src->shape();
        cudaMemcpy3DParms params{};
        params.srcArray = const_cast<cudaArray*>(src->get());
        params.dstPtr = {dst->get(), dst->pitch(), shape.x, shape.y};
        params.extent = {shape.x, shape.y, shape.z};
        params.kind = cudaMemcpyDefault;
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /**
     * Fills (i.e. copies) the multidimensional padded memory of @a dst with the @a N dimensional CUDA array @a src.
     * @param[out] dst      Destination. All elements will be filled.
     * @param[in] src       Source. It should have the same shape as @a dst.
     * @param[out] stream   Stream on which to enqueue the copy.
     * @note This function runs asynchronously with respect to the host and may return before the copy is complete.
     */
    template<typename T, uint N>
    NOA_IH void copy(PtrDevicePadded<T>* dst, const PtrArray<T, N>* src, Stream& stream) {
        size3_t shape = src->shape();
        cudaMemcpy3DParms params{};
        params.srcArray = const_cast<cudaArray*>(src->get());
        params.dstPtr = {dst->get(), dst->pitch(), shape.x, shape.y};
        params.extent = {shape.x, shape.y, shape.z};
        params.kind = cudaMemcpyDefault;
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }
}
