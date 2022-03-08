/// \file noa/cpu/memory/Set.h
/// \brief Set to value.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    /// Sets an array to a given value.
    /// \tparam T           Any type with a copy assignment operator.
    /// \param[out] first   On the \b host. The beginning of range to set.
    /// \param[out] last    On the \b host. The end of range to set.
    /// \param value        The value to assign.
    template<typename T>
    NOA_IH void set(T* first, T* last, T value) {
        NOA_PROFILE_FUNCTION();
        // calling memset, https://godbolt.org/z/1zEzTnoTK
        // the cast is not necessary for basic types, but for Complex<>, IntX<> or FloatX<>, it could help...
        if constexpr (noa::traits::is_data_v<T> || noa::traits::is_intX_v<T> || noa::traits::is_floatX_v<T>)
            if (value == static_cast<T>(0))
                return std::fill(reinterpret_cast<char*>(first), reinterpret_cast<char*>(last), 0);
        return std::fill(first, last, value);
    }

    /// Sets an array to a given value.
    /// \tparam T               Any type with a copy assignment operator.
    /// \param[out] first       On the \b host. The beginning of range to set.
    /// \param[out] last        On the \b host. The end of range to set.
    /// \param value            The value to assign.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void set(T* first, T* last, T value, Stream& stream) {
        stream.enqueue([=]() { return set(first, last, value); });
    }

    /// Sets an array to a given value.
    /// \tparam T       Any type with a copy assignment operator.
    /// \param[out] src On the \b host. The beginning of range to set.
    /// \param elements Number of elements to set.
    /// \param value    The value to assign.
    template<typename T>
    NOA_IH void set(T* src, size_t elements, T value) {
        set(src, src + elements, value);
    }

    /// Sets an array to a given value.
    /// \tparam T               Any type with a copy assignment operator.
    /// \param[out] src         On the \b host. The beginning of range to set.
    /// \param elements         Number of elements to set.
    /// \param value            The value to assign.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void set(T* src, size_t elements, T value, Stream& stream) {
        return set(src, src + elements, value, stream);
    }

    /// Sets an array to a given value.
    /// \tparam CHECK_CONTIGUOUS    Writing to a contiguous block of memory can be often more efficient.
    ///                             If true, the function checks if the data can be accessed contiguously.
    /// \tparam T                   Any type with a copy assignment operator.
    //// \param[out] src            On the \b host. The beginning of range to set.
    /// \param stride               Rightmost strides, in elements, of \p src.
    /// \param shape                Rightmost shape to set.
    /// \param value                The value to assign.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void set(T* src, size4_t stride, size4_t shape, T value) {
        NOA_PROFILE_FUNCTION();
        if constexpr (CHECK_CONTIGUOUS) {
            if (all(isContiguous(stride, shape)))
                return set(src, shape.elements(), value);
        }
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                for (size_t k = 0; k < shape[2]; ++k)
                    for (size_t l = 0; l < shape[3]; ++l)
                        src[at(i, j, k, l, stride)] = value;
    }

    /// Sets an array to a given value.
    /// \tparam CHECK_CONTIGUOUS    Writing to a contiguous block of memory can be often more efficient.
    ///                             If true, the function checks if the data can be accessed contiguously.
    /// \tparam T                   Any type with a copy assignment operator.
    /// \param[out] src             On the \b host. The beginning of range to set.
    /// \param stride               Rightmost strides, in elements, of \p src.
    /// \param shape                Rightmost shape to set.
    /// \param value                The value to assign.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void set(T* src, size4_t stride, size4_t shape, T value, Stream& stream) {
        stream.enqueue([=]() { return set<CHECK_CONTIGUOUS>(src, stride, shape, value); });
    }
}
