/// \file noa/gpu/cuda/memory/Transpose.h
/// \brief Reverse or permute the axes of an array.
/// \author Thomas - ffyr2w
/// \date 29 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"
#include "noa/gpu/cuda/memory/Copy.h"

namespace noa::cuda::memory::details {
    template<typename T>
    void transpose021(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream);
    template<typename T>
    void transpose102(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream);
    template<typename T>
    void transpose120(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream);
    template<typename T>
    void transpose201(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream);
    template<typename T>
    void transpose210(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream);
}

namespace noa::cuda::memory::details::inplace {
    template<typename T>
    void transpose021(T* outputs, size_t outputs_pitch, size3_t shape, uint batches, Stream& stream);
    template<typename T>
    void transpose102(T* outputs, size_t outputs_pitch, size3_t shape, uint batches, Stream& stream);
    template<typename T>
    void transpose210(T* outputs, size_t outputs_pitch, size3_t shape, uint batches, Stream& stream);
}

namespace noa::cuda::memory {
    /// Reverse or permute the axes of an array.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] inputs       Input arrays to permute. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \a inputs.
    /// \param shape            Physical {fast, medium, slow} shape of \a inputs, ignoring the batches.
    /// \param[out] outputs     Output permuted arrays. Should have at least the same elements as \a inputs.
    /// \param outputs_pitch    Pitch, in elements, of \a outputs.
    /// \param permutation      Specifies the particular transpose to be performed. Values should be 0, 1 and 2, which
    ///                         represent the fast, medium and slow axes as entered in \a shape.
    ///                         For 3D arrays, all 5 permutations are supported: 012, 021, 102, 120, 201, 210.
    ///                         For 2D arrays, only 012 and 102 are supported.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \throw For in-place permutations, only 012, 021, 102 and 210 are supported. Anything else throws an error.
    /// \throw The in-place 021 permutation requires the axis 1 and 2 to have the same size.
    /// \throw The in-place 102 permutation requires the axis 0 and 1 to have the same size.
    /// \throw The in-place 210 permutation requires the axis 0 and 2 to have the same size.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void transpose(const T* inputs, size_t inputs_pitch, size3_t shape, T* outputs, size_t outputs_pitch,
                            uint3_t permutation, uint batches, Stream& stream) {
        if (any(permutation > 2U))
            NOA_THROW("Permutation {} is not valid", permutation);

        const uint idx = permutation.x * 100 + permutation.y * 10 + permutation.z;
        if (inputs == outputs) {
            switch (idx) {
                case 12U:
                    break;
                case 21U:
                    details::inplace::transpose021(outputs, outputs_pitch, shape, batches, stream);
                    break;
                case 102U:
                    details::inplace::transpose102(outputs, outputs_pitch, shape, batches, stream);
                    break;
                case 210U:
                    details::inplace::transpose210(outputs, outputs_pitch, shape, batches, stream);
                    break;
                case 120U:
                case 201U:
                    NOA_THROW("The in-place permutation {} is not yet supported. Use the out of place version instead",
                              permutation);
                default:
                    NOA_THROW("Permutation {} is not valid", permutation);
            }
        } else {
            switch (idx) {
                case 12U:
                    copy(inputs, inputs_pitch, outputs, outputs_pitch, {shape.x, getRows(shape), batches}, stream);
                    break;
                case 21U:
                    details::transpose021(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, stream);
                    break;
                case 102U:
                    details::transpose102(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, stream);
                    break;
                case 120U:
                    details::transpose120(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, stream);
                    break;
                case 201U:
                    details::transpose201(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, stream);
                    break;
                case 210U:
                    details::transpose210(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, stream);
                    break;
                default:
                    NOA_THROW("Permutation {} is not valid", permutation);
            }
        }
    }

    /// Reverse or permute the axes of an array. Version for contiguous layouts.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void transpose(const T* inputs, size3_t shape, T* outputs, uint3_t permutation,
                          uint batches, Stream& stream) {
        // The pitch of the output is the first axis of the transposed shape, which is given by permutation[0]
        transpose(inputs, shape.x, shape, outputs, shape[permutation[0]], permutation, batches, stream);
    }
}