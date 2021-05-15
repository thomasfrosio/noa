#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Math {
    /**
     * Computes output[x] = threshold < input[x], for every x from 0 to @a elements.
     * @see This is the CUDA version of Noa::Math::isLess.
     *      The full documentation is described on the CPU version.
     *      The same features and restrictions apply to this function.
     *
     * @tparam T            float, double or any integral type.
     * @tparam U            Any type allowed for @a T or bool.
     * @param[out] stream   Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T, typename U>
    NOA_HOST void isLess(T* input, T threshold, U* output, size_t elements, Stream& stream);

    /**
     * Computes output[x] = threshold < input[x], for every x in @a shape.
     * @see This version is for padded memory. See the overload for contiguous memory for more details.
     *
     * @tparam T            float, double or any integral type.
     * @tparam U            Any type allowed for @a T or bool.
     * @param pitch_input   Pitch, in elements, of @a input.
     * @param pitch_output  Pitch, in elements, of @a output.
     * @param shape         Logical {fast, medium, slow} shape of @a input and @a output.
     * @param stream        Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T, typename U>
    NOA_HOST void isLess(T* input, size_t pitch_input, T threshold, U* output, size_t pitch_output,
                         size3_t shape, Stream& stream);

    /**
     * Computes output[x] = input[x] < threshold, for every x from 0 to @a elements.
     * @see This is the CUDA version of Noa::Math::isGreater.
     *      The full documentation is described on the CPU version and on the Noa::CUDA::Math::isLess functions.
     *      The same features and restrictions apply to this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T, typename U>
    NOA_HOST void isGreater(T* input, T threshold, U* output, size_t elements, Stream& stream);

    /// @see This version is for padded memory. See the overload for contiguous memory for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isGreater(T* input, size_t pitch_input, T threshold, U* output, size_t pitch_output,
                            size3_t shape, Stream& stream);

    /**
     * Computes output[x] = threshold < input[x] && input[x] < threshold, for every x from 0 to @a elements.
     * @see This is the CUDA version of Noa::Math::isWithin.
     *      The full documentation is described on the CPU version.
     *      The same features and restrictions apply to this function.
     *
     * @tparam T            float, double or any integral type.
     * @tparam U            Can be equal to @a T, or any type allowed for @a T or bool.
     * @param[out] stream   Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T, typename U>
    NOA_HOST void isWithin(T* input, T low, T high, U* output, size_t elements, Stream& stream);

    /// @see This version is for padded memory. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isWithin(T* input, size_t pitch_input, T low, T high, U* output, size_t pitch_output,
                           size3_t shape, Stream& stream);

    /**
     * Computes the logical NOT, i.e. output[x] = !input[x], for every x from 0 to @a elements.
     * @see This is the CUDA version of Noa::Math::isWithin.
     *      The full documentation is described on the CPU version.
     *      The same features and restrictions apply to this function.
     *
     * @tparam T            float, double or any integral type.
     * @param[out] stream   Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T, typename U>
    NOA_HOST void logicNOT(T* input, U* output, size_t elements, Stream& stream);

    /// @see This version is for padded memory. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void logicNOT(T* input, size_t pitch_input, U* output, size_t pitch_output, size3_t shape, Stream& stream);
}
