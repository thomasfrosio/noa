/// \file noa/cpu/memory/Copy.h
/// \brief Copy memory regions.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <algorithm>

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    /// Copies all elements in the range [\p first, \p last) starting from \p first and proceeding to \p last - 1.
    /// The behavior is undefined if \p dst_first is within the range [\p first, \p last).
    /// \tparam T               Any type with a copy assignment operator.
    /// \param[in] first        On the \b host. The beginning of range to copy.
    /// \param[in] last         On the \b host. The end of range to copy.
    /// \param[out] dst_first   On the \b host. The beginning of the destination range.
    template<typename T>
    NOA_IH void copy(const T* first, const T* last, T* dst_first) {
        std::copy(first, last, dst_first);
    }

    /// Copies \p src into \p dst.
    /// \tparam T               Any type with a copy assignment operator.
    /// \param[in] src          On the \b host. The beginning of the range to copy.
    /// \param[out] dst         On the \b host. The beginning of the destination range.
    /// \param elements         Number of elements to copy.
    template<typename T>
    NOA_IH void copy(const T* src, T* dst, size_t elements) {
        copy(src, src + elements, dst);
    }

    /// Copies \p src into \p dst.
    /// \tparam T               Any type with a copy assignment operator.
    /// \param[in] src          On the \b host. The beginning of the range to copy.
    /// \param[out] dst         On the \b host. The beginning of the destination range.
    /// \param elements         Number of elements to copy.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void copy(const shared_t<T[]>& src, const shared_t<T[]>& dst, size_t elements, Stream& stream) {
        stream.enqueue([=](){
            copy(src.get(), dst.get(), elements);
        });
    }

    /// Copies all logical elements from \p src to \p dst.
    /// \tparam CHECK_CONTIGUOUS    Copying a contiguous block of memory is often more efficient.
    ///                             If true, the function checks if a contiguous copy can be done instead.
    /// \tparam T                   Any type with a copy assignment operator.
    /// \param[in] src              On the \b host. Input array to copy.
    /// \param src_stride           Rightmost strides, in elements, of \p src.
    /// \param[out] dst             On the \b host. Output array.
    /// \param dst_stride           Rightmost strides, in elements, of \p dst.
    /// \param shape                Rightmost shape of \p src and \p dst.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void copy(const T* src, size4_t src_stride, T* dst, size4_t dst_stride, size4_t shape) {
        if constexpr (CHECK_CONTIGUOUS) {
            if (all(indexing::isContiguous(src_stride, shape)) && all(indexing::isContiguous(dst_stride, shape)))
                return copy(src, src + shape.elements(), dst);

            // Loop through the destination in the most cache-friendly way:
//            const size4_t order = indexing::order(indexing::effectiveStride(shape, dst_stride));
//            dst_stride = indexing::reorder(dst_stride, order);
//            src_stride = indexing::reorder(src_stride, order);
//            shape = indexing::reorder(shape, order);
        }

        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                for (size_t k = 0; k < shape[2]; ++k)
                    for (size_t l = 0; l < shape[3]; ++l)
                        dst[indexing::at(i, j, k, l, dst_stride)] = src[indexing::at(i, j, k, l, src_stride)];
    }

    /// Copies all logical elements from \p src to \p dst.
    /// \tparam CHECK_CONTIGUOUS    Copying a contiguous block of memory is often more efficient.
    ///                             If true, the function checks if a contiguous copy can be done instead.
    /// \tparam T                   Any type with a copy assignment operator.
    /// \param[in] src              On the \b host. Input array to copy.
    /// \param src_stride           Rightmost strides, in elements, of \p src.
    /// \param[out] dst             On the \b host. Output array.
    /// \param dst_stride           Rightmost strides, in elements, of \p dst.
    /// \param shape                Rightmost shape of \p src and \p dst.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void copy(const shared_t<T[]>& src, size4_t src_stride,
                     const shared_t<T[]>& dst, size4_t dst_stride, size4_t shape, Stream& stream) {
        stream.enqueue([=]() {
            return copy<CHECK_CONTIGUOUS>(src.get(), src_stride, dst.get(), dst_stride, shape);
        });
    }
}
