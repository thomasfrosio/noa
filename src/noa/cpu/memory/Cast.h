#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    /// Casts one array to another type.
    /// \tparam T               Any type that can be explicitly converted to \p U.
    /// \tparam U               Any type.
    /// \param[in] inputs       On the \b host. Array(s) to convert.
    /// \param[out] outputs     On the \b host. Converted array(s).
    /// \param elements         Number of elements to convert.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U>
    void cast(const T* inputs, U* outputs, size_t elements, bool clamp, Stream& stream) {
        stream.enqueue([=]() mutable {
            if (clamp) {
                for (size_t i = 0; i < elements; ++i, ++inputs, ++outputs)
                    *outputs = clamp_cast<U>(*inputs);
            } else {
                for (size_t i = 0; i < elements; ++i, ++inputs, ++outputs)
                    *outputs = static_cast<U>(*inputs);
            }
        });
    }

    /// Casts one array to another type.
    /// \tparam T               Any type that can be explicitly converted to \p U.
    /// \tparam U               Any type.
    /// \param[in] inputs       On the \b host. Array(s) to convert.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Converted array(s).
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of batches to convert.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U>
    void cast(const T* inputs, size3_t input_pitch, U* outputs, size3_t output_pitch,
              size3_t shape, size_t batches, bool clamp, Stream& stream) {
        stream.enqueue([=]() mutable {
            const size_t iffset = elements(input_pitch);
            const size_t offset = elements(output_pitch);
            if (clamp) {
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t z = 0; z < shape.z; ++z)
                        for (size_t y = 0; y < shape.y; ++y)
                            for (size_t x = 0; x < shape.x; ++x)
                                outputs[batch * offset + index(x, y, z, output_pitch)] =
                                        clamp_cast<U>(inputs[batch * iffset + index(x, y, z, input_pitch)]);
            } else {
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t z = 0; z < shape.z; ++z)
                        for (size_t y = 0; y < shape.y; ++y)
                            for (size_t x = 0; x < shape.x; ++x)
                                outputs[batch * offset + index(x, y, z, output_pitch)] =
                                        static_cast<U>(inputs[batch * iffset + index(x, y, z, input_pitch)]);
            }
        });
    }
}
