#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    /// Casts one array to another type.
    /// \tparam T               Any type that can be explicitly converted to \p U.
    /// \tparam U               Any type.
    /// \param[in] input        On the \b host. Array to convert.
    /// \param[out] output      On the \b host. Converted array.
    /// \param elements         Number of elements to convert.
    /// \param clamp            Whether the values should be clamp within the \p U range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U>
    void cast(const shared_t<T[]>& input, const shared_t<U[]>& output,
              size_t elements, bool clamp, Stream& stream) {
        stream.enqueue([=]() {
            const T* input_ = input.get();
            U* output_ = output.get();
            if (clamp) {
                for (size_t i = 0; i < elements; ++i, ++input_, ++output_)
                    *output_ = clamp_cast<U>(*input_);
            } else {
                for (size_t i = 0; i < elements; ++i, ++input_, ++output_)
                    *output_ = static_cast<U>(*input_);
            }
        });
    }

    /// Casts one array to another type.
    /// \tparam T               Any type that can be explicitly converted to \p U.
    /// \tparam U               Any type.
    /// \param[in] input        On the \b host. Array to convert.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Converted array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param clamp            Whether the values should be clamp within the \p U range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U>
    void cast(const shared_t<T[]>& input, size4_t input_stride,
              const shared_t<U[]>& output, size4_t output_stride,
              size4_t shape, bool clamp, Stream& stream) {
        stream.enqueue([=]() {
            const T* input_ = input.get();
            U* output_ = output.get();
            if (clamp) {
                for (size_t i = 0; i < shape[0]; ++i)
                    for (size_t j = 0; j < shape[1]; ++j)
                        for (size_t k = 0; k < shape[2]; ++k)
                            for (size_t l = 0; l < shape[3]; ++l)
                                output_[indexing::at(i, j, k, l, output_stride)] =
                                        clamp_cast<U>(input_[indexing::at(i, j, k, l, input_stride)]);
            } else {
                for (size_t i = 0; i < shape[0]; ++i)
                    for (size_t j = 0; j < shape[1]; ++j)
                        for (size_t k = 0; k < shape[2]; ++k)
                            for (size_t l = 0; l < shape[3]; ++l)
                                output_[indexing::at(i, j, k, l, output_stride)] =
                                        static_cast<U>(input_[indexing::at(i, j, k, l, input_stride)]);
            }
        });
    }
}
