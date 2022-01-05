/// \file noa/cpu/memory/Copy.h
/// \brief Copy memory regions.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <algorithm>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    /// Copies all elements in the range [\p first, \p last) starting from \p first and proceeding to \p last - 1.
    /// The behavior is undefined if \p dst_first is within the range [\p first, \p last).
    /// \tparam T               Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
    /// \param[in] first        On the \b host. The beginning of range to copy.
    /// \param[in] last         On the \b host. The end of range to copy.
    /// \param[out] dst_first   On the \b host. The beginning of the destination range.
    template<typename T>
    NOA_IH void copy(const T* first, const T* last, T* dst_first) {
        NOA_PROFILE_FUNCTION();
        std::copy(first, last, dst_first);
    }

    /// Copies all elements in the range [\p first, \p last) starting from \p first and proceeding to \p last - 1.
    /// The behavior is undefined if \p dst_first is within the range [\p first, \p last).
    /// \tparam T               Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
    /// \param[in] first        On the \b host. The beginning of range to copy.
    /// \param[in] last         On the \b host. The end of range to copy.
    /// \param[out] dst_first   On the \b host. The beginning of the destination range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void copy(const T* first, const T* last, T* dst_first, Stream& stream) {
        stream.enqueue([first, last, dst_first]() {
            NOA_PROFILE_FUNCTION();
            return std::copy(first, last, dst_first);
        });
    }

    /// Copies all elements in the range [\p src, \p src + \p elements) starting from \p src and proceeding to
    /// \p src + \p elements - 1. The behavior is undefined if \p dst is within the source range.
    /// \tparam T               Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
    /// \param[in] src          On the \b host. The beginning of the range to copy.
    /// \param[out] dst         On the \b host. The beginning of the destination range.
    /// \param elements         Number of \p T elements to copy.
    template<typename T>
    NOA_IH void copy(const T* src, T* dst, size_t elements) {
        copy(src, src + elements, dst);
    }

    /// Copies all elements in the range [\p src, \p src + \p elements) starting from \p src and proceeding to
    /// \p src + \p elements - 1. The behavior is undefined if \p dst is within the source range.
    /// \tparam T               Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
    /// \param[in] src          On the \b host. The beginning of the range to copy.
    /// \param[out] dst         On the \b host. The beginning of the destination range.
    /// \param elements         Number of \p T elements to copy.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void copy(const T* src, T* dst, size_t elements, Stream& stream) {
        copy(src, src + elements, dst, stream);
    }

    /// Copies all logical elements from \p src to \p dst.
    /// \tparam CHECK_CONTIGUOUS    Copying a contiguous block of memory is often more efficient.
    ///                             If true, the function checks the data is contiguous and if so performs one single copy.
    ///                             Otherwise, assume the data is padded.
    /// \tparam T                   Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
    /// \param[in] src              On the \b host. Input array to copy.
    /// \param src_pitch            Pitch, in elements, of \p src.
    /// \param[out] dst             On the \b host. Output array.
    /// \param dst_pitch            Pitch, in elements, of \p dst.
    /// \param shape                Logical shape of \p src and \p dst.
    /// \param batches              Number of batches to copy.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void copy(const T* src, size3_t src_pitch, T* dst, size3_t dst_pitch,
                     size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        if constexpr (CHECK_CONTIGUOUS) {
            if (all(shape == src_pitch) && all(shape == dst_pitch))
                return copy(src, src + elements(shape) * batches, dst);
        }

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* i_src = src + batch * elements(src_pitch);
            T* i_dst = dst + batch * elements(dst_pitch);
            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    std::copy(i_src + index(y, z, src_pitch.x, src_pitch.y),
                              i_src + index(y, z, src_pitch.x, src_pitch.y) + shape.x,
                              i_dst + index(y, z, dst_pitch.x, dst_pitch.y));
                }
            }
        }
    }

    /// Copies all logical elements from \p src to \p dst.
    /// \tparam CHECK_CONTIGUOUS    Copying a contiguous block of memory is often more efficient.
    ///                             If true, the function checks the data is contiguous and if so performs one single copy.
    ///                             Otherwise, assume the data is padded.
    /// \tparam T                   Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
    /// \param[in] src              On the \b host. Input array to copy.
    /// \param src_pitch            Pitch, in elements, of \p src.
    /// \param[out] dst             On the \b host. Output array.
    /// \param dst_pitch            Pitch, in elements, of \p dst.
    /// \param shape                Logical shape of \p src and \p dst.
    /// \param batches              Number of batches to copy.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void copy(const T* src, size3_t src_pitch, T* dst, size3_t dst_pitch,
                     size3_t shape, size_t batches, Stream& stream) {
        stream.enqueue([=]() {
            return copy<CHECK_CONTIGUOUS>(src, src_pitch, dst, dst_pitch, shape, batches);
        });
    }
}
