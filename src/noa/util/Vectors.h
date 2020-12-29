/**
 * @file Vectors.h
 * @brief Small utility static arrays.
 * @author Thomas - ffyr2w
 * @date 25/10/2020
 */
#pragma once

#include "noa/util/VectorInt.h"
#include "noa/util/VectorFloat.h"


namespace Noa::Traits {
    template<typename T> struct NOA_API is_vector { static constexpr bool value = (is_vector_float_v<T> || is_vector_int_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_vector_v = is_vector<T>::value;
}

