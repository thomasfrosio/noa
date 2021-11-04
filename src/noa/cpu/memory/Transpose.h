/// \file noa/cpu/memory/Transpose.h
/// \brief Reverse or permute the axes of an array.
/// \author Thomas - ffyr2w
/// \date 29 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/cpu/memory/Copy.h"

namespace noa::cpu::memory::details {
    template<typename T> void transpose021(const T* inputs, T* outputs, size3_t shape, size_t batches);
    template<typename T> void transpose102(const T* inputs, T* outputs, size3_t shape, size_t batches);
    template<typename T> void transpose120(const T* inputs, T* outputs, size3_t shape, size_t batches);
    template<typename T> void transpose201(const T* inputs, T* outputs, size3_t shape, size_t batches);
    template<typename T> void transpose210(const T* inputs, T* outputs, size3_t shape, size_t batches);
}

namespace noa::cpu::memory::details::inplace {
    template<typename T> void transpose021(T* outputs, size3_t shape, size_t batches);
    template<typename T> void transpose102(T* outputs, size3_t shape, size_t batches);
    template<typename T> void transpose210(T* outputs, size3_t shape, size_t batches);
}

namespace noa::cpu::memory {
    /// Returns the transposed shape.
    constexpr NOA_IH size3_t transpose(size3_t shape, uint3_t permutation) {
        const uint idx = permutation.x * 100 + permutation.y * 10 + permutation.z;
        switch (idx) {
            case 12U:
                return shape;
            case 21U:
                return {shape.x, shape.z, shape.y};
            case 102U:
                return {shape.y, shape.x, shape.z};
            case 120U:
                return {shape.y, shape.z, shape.x};
            case 201U:
                return {shape.z, shape.x, shape.y};
            case 210U:
                return {shape.z, shape.y, shape.x};
            default:
                NOA_THROW("Permutation {} is not valid", permutation);
        }
    }

    /// Reverses or permutes the axes of an array.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] inputs       On the \b host. Input arrays to permute. One per batch.
    /// \param shape            Physical {fast, medium, slow} shape of \a inputs, ignoring the batches.
    /// \param[out] outputs     On the \b host. Output permuted arrays. One per batch.
    /// \param permutation      Specifies the particular transposition to be performed. Values should be 0, 1 and 2,
    ///                         which represent the fast, medium and slow axes as entered in \a shape.
    ///                         For 3D arrays, all 6 permutations are supported: 012, 021, 102, 120, 201, 210.
    ///                         For 2D arrays, only 012 and 102 are supported.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    ///
    /// \throw For in-place permutations, only 012, 021, 102 and 210 are supported. Anything else throws an error.
    /// \throw The in-place 021 permutation requires the axis 1 and 2 to have the same size.
    /// \throw The in-place 102 permutation requires the axis 0 and 1 to have the same size.
    /// \throw The in-place 210 permutation requires the axis 0 and 2 to have the same size.
    template<typename T>
    NOA_HOST void transpose(const T* inputs, size3_t shape, T* outputs, uint3_t permutation, size_t batches) {
        NOA_PROFILE_FUNCTION();
        if (any(permutation > 2U))
            NOA_THROW("Permutation {} is not valid", permutation);

        const uint idx = permutation.x * 100 + permutation.y * 10 + permutation.z;
        if (inputs == outputs) {
            switch (idx) {
                case 12U:
                    break;
                case 21U:
                    details::inplace::transpose021(outputs, shape, batches);
                    break;
                case 102U:
                    details::inplace::transpose102(outputs, shape, batches);
                    break;
                case 210U:
                    details::inplace::transpose210(outputs, shape, batches);
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
                    copy(inputs, outputs, elements(shape) * batches);
                    break;
                case 21U:
                    details::transpose021(inputs, outputs, shape, batches);
                    break;
                case 102U:
                    details::transpose102(inputs, outputs, shape, batches);
                    break;
                case 120U:
                    details::transpose120(inputs, outputs, shape, batches);
                    break;
                case 201U:
                    details::transpose201(inputs, outputs, shape, batches);
                    break;
                case 210U:
                    details::transpose210(inputs, outputs, shape, batches);
                    break;
                default:
                    NOA_THROW("Permutation {} is not valid", permutation);
            }
        }
    }
}
