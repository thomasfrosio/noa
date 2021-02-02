/**
 * @file noa/util/IntX.h
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 */
#pragma once

/**
 * Although the IntX vectors support "short" integers ((u)int8_t abd (u)int16_t), in most cases
 * there is an integral promotion performed before arithmetic operations. It then triggers
 * a narrowing conversion when the promoted integer needs to be casted back to these "short" integers.
 * See: https://stackoverflow.com/questions/24371868/why-must-a-short-be-converted-to-an-int-before-arithmetic-operations-in-c-and-c
 *
 * @warning Only int32_t, int64_t, uint32_t and uint64_t are tested!
 *
 * @note 29/12/2020 - TF: Since the compiler is allowed to pad, the structures are not necessarily
 *       contiguous. Therefore, remove member function data() and add corresponding constructors.
 */

#include "noa/util/Int2.h"
#include "noa/util/Int3.h"
#include "noa/util/Int4.h"
