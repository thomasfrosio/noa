


/* ----------------- */
/* --- One minus --- */
/* ----------------- */

/// Computes one minus. See corresponding documentation for Noa::Math::inverse.
/// @warning This function is asynchronous with respect to the host and may return before completion.
template<typename T>
NOA_HOST void oneMinus(T* vector, T* output, size_t elements, Stream& stream);

/**
 * Computes one minus, i.e. output[x] = T(1) - vector[x], for every x from 0 to @a shape.
 * @tparam T            float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
 * @param[in] vector    Right operand (i.e. subtrahends).
 * @param pitch_vector  Pitch, in elements, of @a vector.
 * @param[out] output   Results. Can be equal to @a vector (i.e. in-place).
 * @param pitch_output  Pitch, in elements, of @a output.
 * @param shape         Logical {fast, medium, slow} shape of @a vector and @a output.
 * @param[out] stream   Stream on which to enqueue this function.
 *
 * @warning This function is asynchronous with respect to the host and may return before completion.
 */
template<typename T>
NOA_HOST void oneMinus(T* vector, size_t pitch_vector, T* output, size_t pitch_output,
                       size3_t shape, Stream& stream);

/* --------------- */
/* --- Inverse --- */
/* --------------- */

/// Computes the inverse. See corresponding documentation for Noa::Math::inverse.
/// @warning This function is asynchronous with respect to the host and may return before completion.
template<typename T>
NOA_HOST void inverse(T* vector, T* output, size_t elements, Stream& stream);

/**
 * Computes the inverse, i.e. output[x] = T(1) / vector[x], for every x from 0 to @a shape.
 * @tparam T            float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
 * @param[in] vector    Input data.
 * @param pitch_vector  Pitch, in elements, of @a vector.
 * @param[out] output   Results. Can be equal to @a vector (i.e. in-place).
 * @param pitch_output  Pitch, in elements, of @a output.
 * @param shape         Logical {fast, medium, slow} shape of @a vector and @a output.
 * @param[out] stream   Stream on which to enqueue this function.
 *
 * @warning This function is asynchronous with respect to the host and may return before completion.
 */
template<typename T>
NOA_HOST void inverse(T* vector, size_t pitch_vector, T* output, size_t pitch_output,
                      size3_t shape, Stream& stream);
