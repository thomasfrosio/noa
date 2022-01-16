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
    /// \tparam T           Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
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
    /// \tparam T       Most types are supported.
    /// \param[out] src On the \b host. The beginning of range to set.
    /// \param elements Number of elements to set.
    /// \param value    The value to assign.
    template<typename T>
    NOA_IH void set(T* src, size_t elements, T value) {
        set(src, src + elements, value);
    }

    /// Sets an array to a given value.
    /// \tparam T       Most types are supported.
    /// \param src      On the \b host. The beginning of range to set.
    /// \param pitch    Pitch, in elements, of \p src.
    /// \param shape    Logical {fast, medium, slow} shape to set.
    /// \param batches  Number of batches to set.
    /// \param value    The value to assign.
    template<typename T>
    NOA_IH void set(T* src, size3_t pitch, size3_t shape, size_t batches, T value) {
        NOA_PROFILE_FUNCTION();
        for (size_t batch = 0; batch < batches; ++batch) {
            T* i_src = src + batch * elements(pitch);
            for (size_t z = 0; z < shape.z; ++z)
                for (size_t y = 0; y < shape.y; ++y)
                    set(i_src + index(y, z, pitch), shape.x, value);
        }
    }

    /// Sets an array to a given value.
    /// \tparam T               Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
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
    /// \tparam T               Most types are supported.
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
    /// \tparam T               Most types are supported.
    /// \param src              On the \b host. The beginning of range to set.
    /// \param pitch            Pitch, in elements, of \p src.
    /// \param shape            Logical {fast, medium, slow} shape to set.
    /// \param batches          Number of batches to set.
    /// \param value            The value to assign.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void set(T* src, size3_t pitch, size3_t shape, size_t batches, T value, Stream& stream) {
        stream.enqueue([=]() { return set(src, pitch, shape, batches, value); });
    }
}
