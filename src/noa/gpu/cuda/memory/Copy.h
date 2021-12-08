/// \file noa/gpu/cuda/memory/Copy.h
/// \brief Copy from/to device memory.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::memory::details {
    template<typename T>
    NOA_IH cudaMemcpy3DParms toParams(const T* src, size_t src_pitch, T* dst, size_t dst_pitch, size3_t shape) {
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), src_pitch * sizeof(T), shape.x, shape.y};
        params.dstPtr = {dst, dst_pitch * sizeof(T), shape.x, shape.y};
        params.extent = {shape.x * sizeof(T), shape.y, shape.z};
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    NOA_IH cudaMemcpy3DParms toParams(const cudaArray* src, T* dst, size_t dst_pitch, size3_t shape) {
        cudaMemcpy3DParms params{};
        params.srcArray = const_cast<cudaArray*>(src);
        params.dstPtr = {dst, dst_pitch * sizeof(T), shape.x, shape.y};
        params.extent = {shape.x, shape.y, shape.z}; // an array is involved, so shape in elements.
        params.kind = cudaMemcpyDefault;
        return params;
    }

    template<typename T>
    NOA_IH cudaMemcpy3DParms toParams(const T* src, size_t src_pitch, cudaArray* dst, size3_t shape) {
        cudaMemcpy3DParms params{};
        params.srcPtr = {const_cast<T*>(src), src_pitch * sizeof(T), shape.x, shape.y};
        params.dstArray = dst;
        params.extent = {shape.x, shape.y, shape.z};
        params.kind = cudaMemcpyDefault;
        return params;
    }
}

namespace noa::cuda::memory {
    // Since we assume Compute Capability >= 2.0, all devices support the Unified Virtual Address Space, so
    // the CUDA driver can determine, for each pointer, where the data is located, and one does not have to
    // specify the cudaMemcpyKind. In the documentation they don't explicitly say that cudaMemcpyDefault allows
    // for concurrent transfers between host and device if the host is pinned, but why would it make a difference?

    // -- Contiguous memory -- //

    /// Copies synchronously contiguous memory from one region to another. These can point to host or device memory.
    /// \param[in] src      Source. Contiguous memory either on the host or on the device.
    /// \param[out] dst     Destination. Contiguous memory either on the host or on the device.
    /// \param elements     How many elements to copy.
    template<typename T>
    NOA_IH void copy(const T* src, T* dst, size_t elements) {
        NOA_PROFILE_FUNCTION();
        NOA_THROW_IF(cudaMemcpy(dst, src, elements * sizeof(T), cudaMemcpyDefault));
    }

    /// Copies asynchronously contiguous memory from one region to another. These can point to host or device memory.
    /// \note The copy is enqueued to \p stream. Therefore, this function runs asynchronously with respect to the host
    ///       and may return before the copy is complete. Memory copies between host and device can execute concurrently
    ///       only if \p src or \p dst are pinned.
    template<typename T>
    NOA_IH void copy(const T* src, T* dst, size_t elements, Stream& stream) {
        NOA_THROW_IF(cudaMemcpyAsync(dst, src, elements * sizeof(T), cudaMemcpyDefault, stream.id()));
    }

    // -- Padded memory -- //

    /// Copies memory where data is organized in a non-contiguous (aka padded) layout.
    /// \tparam CHECK_CONTIGUOUS    Copying a contiguous block of memory is often more efficient.
    ///                             If true, the function checks the data is contiguous and if so performs one single copy.
    ///                             Otherwise, assume the data is padded.
    /// \param[in] src              Source. Can be on the host or the device.
    /// \param src_pitch            Pitch, in elements, of \p src.
    /// \param[out] dst             Destination. Can be on the host or the device.
    /// \param dst_pitch            Pitch, in elements, of \p dst.
    /// \param shape                Logical {fast, medium, slow} shape to copy. Padded regions are NOT copied.
    ///
    /// \note The order of the last 2 dimensions of the \p shape does not matter, but the number of total rows does.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void copy(const T* src, size_t src_pitch, T* dst, size_t dst_pitch, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        if constexpr (CHECK_CONTIGUOUS) {
            if (shape.x == src_pitch && shape.x == dst_pitch)
                return copy(src, dst, elements(shape));
        }
        cudaMemcpy3DParms params = details::toParams(src, src_pitch, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /// Copies memory where data is organized in a non-contiguous (aka padded) layout.
    /// \tparam CHECK_CONTIGUOUS    Copying a contiguous block of memory is often more efficient. If true, the function
    ///                             checks whether or not the data is contiguous and if so performs one contiguous copy.
    /// \param[in] src              Source. One per batch.
    /// \param src_pitch            Pitch, in elements, of \p src.
    /// \param[out] dst             Destination. One per batch.
    /// \param dst_pitch            Pitch, in elements, of \p dst.
    /// \param shape                Logical {fast, medium, slow} shape to copy (ignoring the batches).
    ///                             Padded regions are NOT copied.
    /// \param batches              Number of batches to copy.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void copy(const T* src, size_t src_pitch, T* dst, size_t dst_pitch, size3_t shape, size_t batches) {
        copy<CHECK_CONTIGUOUS>(src, src_pitch, dst, dst_pitch, size3_t(shape.x, rows(shape), batches));
    }

    /// Copies memory where data is organized in a non-contiguous (aka padded) layout.
    /// \note The copy is enqueued to \p stream. Therefore, this function runs asynchronously with respect to the host
    ///       and may return before the copy is complete. Memory copies between host and device can execute concurrently
    ///       only if \p src or \p dst is pinned.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void copy(const T* src, size_t src_pitch, T* dst, size_t dst_pitch, size3_t shape, Stream& stream) {
        if constexpr (CHECK_CONTIGUOUS) {
            if (shape.x == src_pitch && shape.x == dst_pitch)
                copy(src, dst, elements(shape), stream);
        }
        cudaMemcpy3DParms params = details::toParams(src, src_pitch, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /// Copies memory where data is organized in a non-contiguous (aka padded) layout.
    /// \note The copy is enqueued to \p stream. Therefore, this function runs asynchronously with respect to the host
    ///       and may return before the copy is complete. Memory copies between host and device can execute concurrently
    ///       only if \p src or \p dst is pinned.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void copy(const T* src, size_t src_pitch, T* dst, size_t dst_pitch,
                     size3_t shape, size_t batches, Stream& stream) {
        copy<CHECK_CONTIGUOUS>(src, src_pitch, dst, dst_pitch, size3_t(shape.x, rows(shape), batches), stream);
    }

    // -- CUDA arrays and contiguous memory -- //

    /// Copies a CUDA array with a given physical \p shape into \p dst.
    /// \param[in] src      N dimensional CUDA array. Should correspond to \p shape. All elements will be copied.
    /// \param[out] dst     Contiguous memory. Should be large enough to contain \p src.
    /// \param shape        Logical {fast, medium, slow} shape to copy.
    ///                     In total, `noa::elements(shape) * sizeof(T)` bytes are copied.
    template<typename T>
    NOA_IH void copy(const cudaArray* src, T* dst, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        cudaMemcpy3DParms params = details::toParams(src, dst, shape.x, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /// Copies asynchronously a CUDA array with a given physical \p shape into \p dst.
    /// \note The copy is enqueued to \p stream. Therefore, this function runs asynchronously with respect to the host
    ///       and may return before the copy is complete. Memory copies between host and device can execute concurrently
    ///       only if \p dst is pinned.
    template<typename T>
    NOA_IH void copy(const cudaArray* src, T* dst, size3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = details::toParams(src, dst, shape.x, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /// Copies memory with a given physical \p shape into the CUDA array \p dst.
    /// \param[in] src      Contiguous memory. Should correspond or be larger than \p shape.
    /// \param[out] dst     N dimensional CUDA array. Should correspond to \p shape. All elements will be filled.
    /// \param shape        Logical {fast, medium, slow} shape to copy.
    ///                     In total, `noa::elements(shape) * sizeof(T)` bytes are copied.
    template<typename T>
    NOA_IH void copy(const T* src, cudaArray* dst, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        cudaMemcpy3DParms params = details::toParams(src, shape.x, dst, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /// Copies memory with a given physical \p shape into the CUDA array \p dst.
    /// \note The copy is enqueued to \p stream. Therefore, this function runs asynchronously with respect to the host
    ///       and may return before the copy is complete. Memory copies between host and device can execute concurrently
    ///       only if \p src is pinned.
    template<typename T>
    NOA_IH void copy(const T* src, cudaArray* dst, size3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = details::toParams(src, shape.x, dst, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    // -- CUDA arrays and padded memory -- //

    /// Copies a CUDA array with a given physical \p shape into \p dst.
    /// \param[in] src      N dimensional CUDA array. Should correspond to \p shape. All elements will be copied.
    /// \param[out] dst     Should be large enough to contain \p src.
    /// \param dst_pitch    Pitch, in elements, of \p dst.
    /// \param shape        Logical {fast, medium, slow} shape to copy.
    ///                     In total, `noa::elements(shape) * sizeof(T)` bytes are copied.
    template<typename T>
    NOA_IH void copy(const cudaArray* src, T* dst, size_t dst_pitch, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        cudaMemcpy3DParms params = details::toParams(src, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /// Copies asynchronously a CUDA array with a given physical \p shape into \p dst.
    /// \note The copy is enqueued to \p stream. Therefore, this function runs asynchronously with respect to the host
    ///       and may return before the copy is complete. Memory copies between host and device can execute concurrently
    ///       only if \p dst is pinned.
    template<typename T>
    NOA_IH void copy(const cudaArray* src, T* dst, size_t dst_pitch, size3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = details::toParams(src, dst, dst_pitch, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }

    /// Copies memory with a given physical \p shape into the CUDA array \p dst.
    /// \param[in] src      Should correspond or be larger than \p shape.
    /// \param src_pitch    Pitch, in elements, of \p src.
    /// \param[out] dst     N dimensional CUDA array. Should correspond to \p shape. All elements will be filled.
    /// \param shape        Logical {fast, medium, slow} shape to copy.
    ///                     In total, `noa::elements(shape) * sizeof(T)` bytes are copied.
    template<typename T>
    NOA_IH void copy(const T* src, size_t src_pitch, cudaArray* dst, size3_t shape) {
        NOA_PROFILE_FUNCTION();
        cudaMemcpy3DParms params = details::toParams(src, src_pitch, dst, shape);
        NOA_THROW_IF(cudaMemcpy3D(&params));
    }

    /// Copies memory with a given physical \p shape into the CUDA array \p dst.
    /// \note The copy is enqueued to \p stream. Therefore, this function runs asynchronously with respect to the host
    ///       and may return before the copy is complete. Memory copies between host and device can execute concurrently
    ///       only if \p src is pinned.
    template<typename T>
    NOA_IH void copy(const T* src, size_t src_pitch, cudaArray* dst, size3_t shape, Stream& stream) {
        cudaMemcpy3DParms params = details::toParams(src, src_pitch, dst, shape);
        NOA_THROW_IF(cudaMemcpy3DAsync(&params, stream.id()));
    }
}
