/// \file noa/cpu/math/Arithmetics.h
/// \brief Basic math arithmetics for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"

namespace noa::math::details {
    enum : int { ADD, SUBTRACT, MULTIPLY, DIVIDE, DIVIDE_SAFE };

    template<int OPERATION, typename T, typename U>
    void applyValue(const T* arrays, const U* values, T* outputs, size_t elements, uint batches);

    template<int OPERATION, typename T, typename U>
    void applyArray(const T* arrays, const U* weights, T* outputs, size_t elements, uint batches);
}

namespace noa::math {
    // -- Multiply -- //

    /// Multiplies the input array by a single value: output[x] = input[x] * value, for every x from 0 to \a elements.
    /// \tparam T           int, uint, float, double, cfloat_t, cdouble_t.
    /// \tparam U           Should be \a T, except if \a T is cfloat_t/cdouble_t, then \a U can also be float/double.
    /// \param[in] input    Input array to multiply.
    /// \param value        Multiplier.
    /// \param[out] outputs Output array. Can be equal to \a inputs.
    /// \param elements     Number of elements to compute.
    ///
    /// \note \a inputs and \a outputs should be at least \a elements elements.
    /// \note The divide*, add* and subtract* functions operate similarly, but uses the operators, /, + and -, respectively.
    template<typename T, typename U>
    NOA_IH void multiplyByValue(const T* input, U value, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::MULTIPLY>(input, &value, output, elements, 1);
    }

    /// For each batch, multiplies one input array by a single value:
    /// outputs[b][x] = inputs[b][x] * values[b], for every x from 0 to \a elements and for every b from 0 to \a batches.
    ///
    /// \tparam T           int, uint, float, double, cfloat_t, cdouble_t.
    /// \tparam U           Should be \a T, except if \a T is cfloat_t/cdouble_t, then \a U can also be float/double.
    /// \param[in] inputs   Input arrays to multiply. One array of \a elements elements per batch.
    /// \param values       Multipliers. One value per batch.
    /// \param[out] outputs Output arrays. One array of \a elements elements per batch. Can be equal to \a inputs.
    /// \param elements     Number of elements per batch.
    /// \param batches      Number of batches to compute.
    ///
    /// \note \a inputs and \a outputs should be at least \a elements * \a batches elements.
    /// \note The divide*, add* and subtract* functions operate similarly, but uses the operators, /, + and -, respectively.
    template<typename T, typename U>
    NOA_IH void multiplyByValue(const T* inputs, U* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::MULTIPLY>(inputs, values, outputs, elements, batches);
    }

    /// For each batch, computes the element-wise multiplication between one input array and the second array:
    /// outputs[b][x] = inputs[b][x] * array[x], for every x from 0 to \a elements and for every b from 0 to \a batches.
    ///
    /// \tparam T           int, uint, float, double, cfloat_t, cdouble_t.
    /// \tparam U           Should be \a T, except if \a T is cfloat_t/cdouble_t, then \a U should be float/double.
    /// \param[in] inputs   Input arrays to multiply. One array of \a elements elements per batch.
    /// \param array        Multipliers. The same array is applied to every batch.
    /// \param[out] outputs Output arrays. One array of \a elements elements per batch. Can be equal to \a inputs.
    /// \param elements     Number of elements per batch.
    /// \param batches      Number of batches to compute.
    ///
    /// \note \a inputs and \a outputs should be at least \a elements * \a batches elements,
    ///       whereas \a array should be at least \a elements elements.
    /// \note The divide*, add* and subtract* functions operate similarly, but uses the operators, /, + and -, respectively.
    template<typename T, typename U>
    NOA_IH void multiplyByArray(const T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyArray<details::MULTIPLY>(inputs, array, outputs, elements, batches);
    }

    // -- Divide -- //

    /// Divides the input array by a single value.
    /// \see This function supports same features and restrictions than noa::math::multiplyByValue.
    template<typename T, typename U>
    NOA_IH void divideByValue(const T* input, U value, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::DIVIDE>(input, &value, output, elements, 1);
    }

    /// For each batch, divides one input array by a single value.
    /// \see This function supports same features and restrictions than noa::math::multiplyByValue.
    template<typename T, typename U>
    NOA_IH void divideByValue(const T* inputs, U* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::DIVIDE>(inputs, values, outputs, elements, batches);
    }

    /// For each batch, computes the element-wise division between one array and the weights.
    /// \see This function supports same features and restrictions than noa::math::multiplyByArray.
    template<typename T, typename U>
    NOA_IH void divideByArray(const T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyArray<details::DIVIDE>(inputs, array, outputs, elements, batches);
    }

    /// For each batch, computes the element-wise safe division (division by 0 returns 0) between one array and the array.
    /// \see This function supports same features and restrictions than noa::math::multiplyByArray, with the additional
    ///      restriction that \a U cannot be complex (cfloat_t or cdouble_t).
    template<typename T, typename U, typename = std::enable_if_t<!noa::traits::is_complex_v<U>>>
    NOA_IH void divideSafeByArray(const T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyArray<details::DIVIDE_SAFE>(inputs, array, outputs, elements, batches);
    }

    // -- Add -- //

    /// Adds a single value to the input array.
    /// \see This function supports same features and restrictions than noa::math::multiplyByValue.
    template<typename T, typename U>
    NOA_IH void addValue(const T* input, U value, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::ADD>(input, &value, output, elements, 1);
    }

    /// For each batch, adds a single value to an input array.
    /// \see This function supports same features and restrictions than noa::math::multiplyByValue.
    template<typename T, typename U>
    NOA_IH void addValue(const T* inputs, U* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::ADD>(inputs, values, outputs, elements, batches);
    }

    /// For each batch, adds one array (\a addends) and the all input \a inputs.
    /// \see This function supports same features and restrictions than noa::math::multiplyByArray.
    template<typename T, typename U>
    NOA_IH void addArray(const T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyArray<details::ADD>(inputs, array, outputs, elements, batches);
    }

    // -- Subtract -- //

    /// Subtracts a single value to the input array.
    /// \see This function supports same features and restrictions than noa::math::multiplyByValue.
    template<typename T, typename U>
    NOA_IH void subtractValue(const T* input, U value, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::SUBTRACT>(input, &value, output, elements, 1);
    }

    /// For each batch, subtracts a single value to an input array.
    /// \see This function supports same features and restrictions than noa::math::multiplyByValue.
    template<typename T, typename U>
    NOA_IH void subtractValue(const T* inputs, U* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::SUBTRACT>(inputs, values, outputs, elements, batches);
    }

    /// For each batch, subtracts one array (\a subtrahends) and the all input \a inputs.
    /// \see This function supports same features and restrictions than noa::math::multiplyByArray.
    template<typename T, typename U>
    NOA_IH void subtractArray(const T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyArray<details::SUBTRACT>(inputs, array, outputs, elements, batches);
    }
}
