#pragma once

#include "noa/Definitions.h"
#include "noa/Math.h"
#include "noa/Types.h"
#include "noa/Profiler.h"

// Implementation details:
namespace Noa::Math::Details {
    enum : int { ADD, SUBTRACT, MULTIPLY, DIVIDE, DIVIDE_SAFE };

    template<int OPERATION, typename T, typename U>
    NOA_HOST void applyValue(T* arrays, U* values, T* output, size_t elements, uint batches);

    template<int OPERATION, typename T, typename U>
    NOA_HOST void applyArray(T* arrays, U* weights, T* output, size_t elements, uint batches);
}

namespace Noa::Math {
    /* ---------------- */
    /* --- Multiply --- */
    /* ---------------- */

    /**
     * Multiplies the input array by a single value: output[x] = input[x] * value, for every x from 0 to @a elements.
     * @tparam T            Any type with `T operator*(T, U)` defined.
     * @tparam U            Any type with `T operator*(T, U)` defined. Can be equal to @a T.
     * @param[in] input     Input array to multiply.
     * @param value         Multiplier.
     * @param[out] outputs  Output array. Can be equal to @a inputs.
     * @param elements      Number of elements to compute.
     *
     * @note @a inputs and @a outputs should be at least @a elements elements.
     * @note The functions divide*, add* and subtract* operate similarly using the operators, /, + and -, respectively.
     */
    template<typename T, typename U>
    NOA_IH void multiplyByValue(T* input, U value, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        Details::applyValue<Details::MULTIPLY>(input, &value, output, elements, 1);
    }

    /**
     * For each batch, multiplies one input array by a single value:
     * outputs[b][x] = inputs[b][x] * values[b], for every x from 0 to @a elements and for every b from 0 to @a batches.
     *
     * @tparam T            Any type with `T operator*(T, U)` defined.
     * @tparam U            Any type with `T operator*(T, U)` defined. Can be equal to @a T.
     * @param[in] inputs    Input arrays to multiply. One array of @a elements elements per batch.
     * @param values        Multipliers. One value per batch.
     * @param[out] outputs  Output arrays. One array of @a elements elements per batch. Can be equal to @a inputs.
     * @param elements      Number of elements per batch.
     * @param batches       Number of batches to compute.
     *
     * @note @a inputs and @a outputs should be at least @a elements * @a batches elements.
     * @note The functions divide*, add* and subtract* operate similarly using the operators, /, + and -, respectively.
     */
    template<typename T, typename U>
    NOA_IH void multiplyByValue(T* inputs, U* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        Details::applyValue<Details::MULTIPLY>(inputs, values, outputs, elements, batches);
    }

    /**
     * For each batch, computes the element-wise multiplication between one input array and the second array:
     * outputs[b][x] = inputs[b][x] * array[x], for every x from 0 to @a elements and for every b from 0 to @a batches.
     *
     * @tparam T            Any type with `T operator*(T, U)` defined.
     * @tparam U            Any type with `T operator*(T, U)` defined. Can be equal to @a T.
     * @param[in] inputs    Input arrays to multiply. One array of @a elements elements per batch.
     * @param array         Multipliers. The same array is applied to every batch.
     * @param[out] outputs  Output arrays. One array of @a elements elements per batch. Can be equal to @a inputs.
     * @param elements      Number of elements per batch.
     * @param batches       Number of batches to compute.
     *
     * @note @a inputs and @a outputs should be at least @a elements * @a batches elements, whereas @a array should be
     *       at least @a elements elements.
     * @note The functions divide*, add* and subtract* operate similarly using the operators, /, + and -, respectively.
     */
    template<typename T, typename U>
    NOA_IH void multiplyByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        Details::applyArray<Details::MULTIPLY>(inputs, array, outputs, elements, batches);
    }

    /* -------------- */
    /* --- Divide --- */
    /* -------------- */

    /// Divides the input array by a single value.
    /// @see This function supports same features and restrictions than Noa::Math::multiplyByValue.
    template<typename T, typename U>
    NOA_IH void divideByValue(T* input, U value, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        Details::applyValue<Details::DIVIDE>(input, &value, output, elements, 1);
    }

    /// For each batch, divides one input array by a single value.
    /// @see This function supports same features and restrictions than Noa::Math::multiplyByValue.
    template<typename T, typename U>
    NOA_IH void divideByValue(T* inputs, U* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        Details::applyValue<Details::DIVIDE>(inputs, values, outputs, elements, batches);
    }

    /// For each batch, computes the element-wise division between one array and the weights.
    /// @see This function supports same features and restrictions than Noa::Math::multiplyByArray.
    template<typename T, typename U>
    NOA_IH void divideByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        Details::applyArray<Details::DIVIDE>(inputs, array, outputs, elements, batches);
    }

    /// For each batch, computes the element-wise safe division (division by 0 returns 0) between one array and the array.
    /// @see This function supports same features and restrictions than Noa::Math::multiplyByArray, with the additional
    ///      restriction that @a U cannot be complex (cfloat_t or cdouble_t).
    template<typename T, typename U, typename = std::enable_if_t<!Noa::Traits::is_complex_v<U>>>
    NOA_IH void divideSafeByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        Details::applyArray<Details::DIVIDE_SAFE>(inputs, array, outputs, elements, batches);
    }

    /* ----------- */
    /* --- Add --- */
    /* ----------- */

    /// Adds a single value to the input array.
    /// @see This function supports same features and restrictions than Noa::Math::multiplyByValue.
    template<typename T, typename U>
    NOA_IH void addValue(T* input, U value, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        Details::applyValue<Details::ADD>(input, &value, output, elements, 1);
    }

    /// For each batch, adds a single value to an input array.
    /// @see This function supports same features and restrictions than Noa::Math::multiplyByValue.
    template<typename T, typename U>
    NOA_IH void addValue(T* inputs, U* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        Details::applyValue<Details::ADD>(inputs, values, outputs, elements, batches);
    }

    /// For each batch, adds one array (@a addends) and the all input @a inputs.
    /// @see This function supports same features and restrictions than Noa::Math::multiplyByArray.
    template<typename T, typename U>
    NOA_IH void addArray(T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        Details::applyArray<Details::ADD>(inputs, array, outputs, elements, batches);
    }

    /* ---------------- */
    /* --- Subtract --- */
    /* ---------------- */

    /// Subtracts a single value to the input array.
    /// @see This function supports same features and restrictions than Noa::Math::multiplyByValue.
    template<typename T, typename U>
    NOA_IH void subtractValue(T* input, U value, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        Details::applyValue<Details::SUBTRACT>(input, &value, output, elements, 1);
    }

    /// For each batch, subtracts a single value to an input array.
    /// @see This function supports same features and restrictions than Noa::Math::multiplyByValue.
    template<typename T, typename U>
    NOA_IH void subtractValue(T* inputs, U* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        Details::applyValue<Details::SUBTRACT>(inputs, values, outputs, elements, batches);
    }

    /// For each batch, subtracts one array (@a subtrahends) and the all input @a inputs.
    /// @see This function supports same features and restrictions than Noa::Math::multiplyByArray.
    template<typename T, typename U>
    NOA_IH void subtractArray(T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        Details::applyArray<Details::SUBTRACT>(inputs, array, outputs, elements, batches);
    }
}
