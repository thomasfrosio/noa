/// \file noa/cpu/math/Arithmetics.h
/// \brief Basic math arithmetics for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"

namespace noa::cpu::math::details {
    enum : int { ADD, SUBTRACT, MULTIPLY, DIVIDE, DIVIDE_SAFE };

    template<int OPERATION, typename T, typename U>
    void applyValue(const T* arrays, const U* values, T* outputs, size_t elements, uint batches);

    template<int OPERATION, typename T, typename U>
    void applyArray(const T* arrays, const U* weights, T* outputs, size_t elements, uint batches);
}

namespace noa::cpu::math {
    /// Multiplies \p input by \p value.
    /// \tparam T           (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \tparam U           If \p T is complex, \p U can be the corresponding value type. Otherwise, same as \p T.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param value        Multiplier.
    /// \param[out] outputs On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements to compute.
    ///
    /// \note The divide*, add* and subtract* functions operate similarly, but uses the operators, /, + and -, respectively.
    template<typename T, typename U>
    NOA_IH void multiplyByValue(const T* input, U value, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::MULTIPLY>(input, &value, output, elements, 1);
    }

    /// For each batch, multiplies \p input by a single value.
    /// \tparam T           (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \tparam U           If \p T is complex, \p U can be the corresponding value type. Otherwise, same as \p T.
    /// \param[in] inputs   On the \b host. Contiguous input arrays. One array per batch.
    /// \param[in] values   On the \b host. Contiguous multipliers. One value per batch.
    /// \param[out] outputs On the \b host. Contiguous output arrays. One array per batch.
    ///                     Can be equal to \p inputs or \p values.
    /// \param elements     Number of elements to compute per batch.
    /// \param batches      Number of batches to compute.
    ///
    /// \note The divide*, add* and subtract* functions operate similarly, but uses the operators, /, + and -, respectively.
    template<typename T, typename U>
    NOA_IH void multiplyByValue(const T* inputs, const U* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::MULTIPLY>(inputs, values, outputs, elements, batches);
    }

    /// For each batch, computes the element-wise multiplication between \p input and \p array.
    /// \tparam T           (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \tparam U           If \p T is complex, \p U can be the corresponding value type. Otherwise, same as \p T.
    /// \param[in] inputs   On the \b host. Contiguous input arrays. One array per batch.
    /// \param[in] array    On the \b host. Contiguous multipliers. The same array is used to every batch.
    /// \param[out] outputs On the \b host. Contiguous output arrays. One array per batch.
    ///                     Can be equal to \p inputs or \p arrays.
    /// \param elements     Number of elements to compute per batch.
    /// \param batches      Number of batches to compute.
    ///
    /// \note The divide*, add* and subtract* functions operate similarly, but uses the operators, /, + and -, respectively.
    template<typename T, typename U>
    NOA_IH void multiplyByArray(const T* inputs, const U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyArray<details::MULTIPLY>(inputs, array, outputs, elements, batches);
    }
}

namespace noa::cpu::math {
    /// Divides \p input by \p value.
    /// \see This function has the same features and restrictions than noa::cpu::math::multiplyByValue().
    template<typename T, typename U>
    NOA_IH void divideByValue(const T* input, U value, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::DIVIDE>(input, &value, output, elements, 1);
    }

    /// For each batch, divides \p input by a single value.
    /// \see This function has the same features and restrictions than noa::cpu::math::multiplyByValue().
    template<typename T, typename U>
    NOA_IH void divideByValue(const T* inputs, U* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::DIVIDE>(inputs, values, outputs, elements, batches);
    }

    /// For each batch, computes the element-wise division between \p input and \p array.
    /// \see This function has the same features and restrictions than noa::cpu::math::multiplyByArray().
    template<typename T, typename U>
    NOA_IH void divideByArray(const T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyArray<details::DIVIDE>(inputs, array, outputs, elements, batches);
    }

    /// For each batch, computes the element-wise safe division (division by 0 returns 0) between \p inputs and \p array.
    /// \see This function has the same features and restrictions than noa::cpu::math::multiplyByArray(),
    ///      with the additional restriction that \p U cannot be complex (cfloat_t or cdouble_t).
    template<typename T, typename U, typename = std::enable_if_t<!noa::traits::is_complex_v<U>>>
    NOA_IH void divideSafeByArray(const T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyArray<details::DIVIDE_SAFE>(inputs, array, outputs, elements, batches);
    }
}

namespace noa::cpu::math {
    /// Adds \p value to \p input.
    /// \see This function has the same features and restrictions than noa::cpu::math::multiplyByValue().
    template<typename T, typename U>
    NOA_IH void addValue(const T* input, U value, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::ADD>(input, &value, output, elements, 1);
    }

    /// For each batch, adds a single value to \p input.
    /// \see This function has the same features and restrictions than noa::cpu::math::multiplyByValue().
    template<typename T, typename U>
    NOA_IH void addValue(const T* inputs, U* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::ADD>(inputs, values, outputs, elements, batches);
    }

    /// For each batch, adds \p array to \p inputs.
    /// \see This function has the same features and restrictions than noa::cpu::math::multiplyByArray().
    template<typename T, typename U>
    NOA_IH void addArray(const T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyArray<details::ADD>(inputs, array, outputs, elements, batches);
    }
}

namespace noa::cpu::math {
    /// Subtracts \p value to \p input.
    /// \see This function has the same features and restrictions than noa::cpu::math::multiplyByValue().
    template<typename T, typename U>
    NOA_IH void subtractValue(const T* input, U value, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::SUBTRACT>(input, &value, output, elements, 1);
    }

    /// For each batch, subtracts a single value to \p input.
    /// \see This function has the same features and restrictions than noa::cpu::math::multiplyByValue().
    template<typename T, typename U>
    NOA_IH void subtractValue(const T* inputs, U* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyValue<details::SUBTRACT>(inputs, values, outputs, elements, batches);
    }

    /// For each batch, subtracts \p array to \p input.
    /// \see This function has the same features and restrictions than noa::cpu::math::multiplyByArray().
    template<typename T, typename U>
    NOA_IH void subtractArray(const T* inputs, U* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        details::applyArray<details::SUBTRACT>(inputs, array, outputs, elements, batches);
    }
}
