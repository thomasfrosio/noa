/// \file noa/cpu/memory/Complex.h
/// \brief Decompose complex numbers into real and imaginary components.
/// \author Thomas - ffyr2w
/// \date 12 Jan 2022

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"

#include "noa/cpu/Stream.h"
#include "noa/cpu/math/Ewise.h"

namespace noa::cpu::memory {
    /// Extracts the real and imaginary part of complex numbers.
    /// \tparam T               half_t, float, double.
    /// \param[in] inputs       On the \b host. Complex array to decompose. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] real        On the \b host. Real elements. One array per batch.
    /// \param real_pitch       Pitch, in elements, of \p real.
    /// \param[out] imag        On the \b host. Imaginary elements. One array per batch.
    /// \param imag_pitch       Pitch, in elements, of \p imag.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs, \p real and \p imag.
    /// \param batches          Number of batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void decompose(const noa::Complex<T>* inputs, size3_t input_pitch,
                            T* real, size3_t real_pitch,
                            T* imag, size3_t imag_pitch,
                            size3_t shape, size_t batches, Stream& stream) {
        stream.enqueue([=]() {
            NOA_PROFILE_FUNCTION();
            const size_t iffset = elements(input_pitch);
            const size_t real_offset = elements(real_pitch);
            const size_t imag_offset = elements(imag_pitch);
            for (size_t batch = 0; batch < batches; ++batch) {
                for (size_t z = 0; z < shape.z; ++z) {
                    for (size_t y = 0; y < shape.y; ++y) {
                        for (size_t x = 0; x < shape.x; ++x) {
                            const auto& tmp = inputs[batch * iffset + index(x, y, z, input_pitch.x, input_pitch.y)];
                            real[batch * real_offset + index(x, y, z, real_pitch.x, real_pitch.y)] = tmp.real;
                            imag[batch * imag_offset + index(x, y, z, imag_pitch.x, imag_pitch.y)] = tmp.imag;
                        }
                    }
                }
            }
        });
    }

    /// Fuses the real and imaginary components.
    /// \tparam T               half_t, float, double.
    /// \param[in] real         On the \b host. Real elements to interleave. One array per batch.
    /// \param real_pitch       Pitch, in elements, of \p real.
    /// \param[in] imag         On the \b host. Imaginary elements to interleave. One array per batch.
    /// \param imag_pitch       Pitch, in elements, of \p imag.
    /// \param outputs          On the \b host. Complex array. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p real, \p imag and \p outputs.
    /// \param batches          Number of batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void complex(const T* real, size3_t real_pitch, const T* imag, size3_t imag_pitch,
                        noa::Complex<T>* outputs, size3_t output_pitch, size3_t shape, size_t batches, Stream& stream) {
        return ewise(real, real_pitch, imag, imag_pitch, outputs, output_pitch, shape, batches,
                     [](const T& r, const T& i) { return noa::Complex<T>(r, i); }, stream);
    }
}
