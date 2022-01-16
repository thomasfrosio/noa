/// \file noa/cpu/memory/Index.h
/// \brief Remapping functions.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

// -- Using coordinates -- //
namespace noa::cpu::memory {
    /// Extracts from the input array(s) one or multiple subregions (with the same shape) at various locations.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, or any (complex) floating-point.
    /// \param[in] inputs       On the \b host. Input array(s) to use for the extraction.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param input_shape      Logical {fast, medium, slow} shape of \p inputs.
    /// \param[out] subregions  On the \b host. Output subregions.
    /// \param subregion_pitch  Pitch, in elements, of \p subregions.
    /// \param subregion_shape  Logical {fast, medium, slow} shape of subregions.
    /// \param[in] origins      On the \b host. Indexes, starting from 0, defining the origin (i.e. corner left)
    ///                         where to extract subregions. While usually within the input frame, subregions
    ///                         can be (partially) out-of-bound (OBB).
    /// \param batches          Number of subregions to extract.
    /// \param border_mode      Border mode used for OOB conditions. Can be BORDER_NOTHING, BORDER_ZERO,
    ///                         BORDER_VALUE, BORDER_CLAMP, BORDER_MIRROR or BORDER_REFLECT.
    /// \param border_value     Constant value to use for OOB. Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ////
    /// \note \p inputs and \p subregions should not overlap.
    /// \note As usual, batches can use the same input array by passing 0 to \p input_pitch.z.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void extract(const T* inputs, size3_t input_pitch, size3_t input_shape,
                          T* subregions, size3_t subregion_pitch, size3_t subregion_shape,
                          const int3_t* origins, size_t batches,
                          BorderMode border_mode, T border_value, Stream& stream);

    /// Extracts a subregion from the input array.
    template<typename T>
    NOA_IH void extract(const T* input, size2_t input_pitch, size3_t input_shape,
                        T* subregion, size2_t subregion_pitch, size3_t subregion_shape,
                        int3_t origin, BorderMode border_mode, T border_value, Stream& stream) {
        extract(input, {input_pitch, 0}, input_shape, subregion, {subregion_pitch, 0}, subregion_shape,
                &origin, 1, border_mode, border_value, stream);
    }

    /// Inserts into the output array(s) one or multiple subregions (with the same shape) at various locations.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, or any (complex) floating-point.
    /// \param[in] subregions   On the \b host. Subregion(s) to insert into \p outputs.
    /// \param subregion_pitch  Pitch, in elements, of \p subregions.
    /// \param subregion_shape  Logical {fast, medium, slow} shape of one subregion.
    /// \param[out] output      On the \b host. Output array(s).
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param output_shape     Logical {fast, medium, slow} shape of \p outputs.
    /// \param[in] origins      On the \b host. Indexes, starting from 0, defining the origin (i.e. corner left)
    ///                         where to insert subregions. While usually within the output frame, subregions
    ///                         can be (partially) out-of-bound (OBB).
    /// \param batches          Number of subregions to insert.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs and \p subregions should not overlap.
    /// \note This function assumes no overlap between subregions. There's no guarantee on the order of insertion.
    /// \note As usual, batches can use the same output array by passing 0 to \p output_pitch.z.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void insert(const T* subregions, size3_t subregion_pitch, size3_t subregion_shape,
                         T* outputs, size3_t output_pitch, size3_t output_shape,
                         const int3_t* origins, size_t batches, Stream& stream);

    /// Inserts a subregion into the output array.
    template<typename T>
    NOA_IH void insert(const T* subregion, size2_t subregion_pitch, size3_t subregion_shape,
                       T* output, size2_t output_pitch, size3_t output_shape,
                       int3_t origins, Stream& stream) {
        insert(subregion, {subregion_pitch, 0}, subregion_shape, output, {output_pitch, 0},
               output_shape, &origins, 1, stream);
    }

    /// Gets the atlas layout (shape + subregion origins).
    /// \param subregion_shape          Logical {fast,medium,slow} shape of the subregion(s).
    /// \param subregion_count          Number of subregion(s) to place into the atlas.
    /// \param[out] origins             On the \b host. Subregion origin(s), relative to the atlas shape.
    /// \return                         Atlas shape.
    ///
    /// \note The shape of the atlas is not necessary a square. For instance, with 4 subregions the atlas layout
    ///       is `2x2`, but with 5 subregions is goes to `3x2` with one empty region. Subregions are in row-major order.
    NOA_IH size3_t atlasLayout(size3_t subregion_shape, size_t subregion_count, int3_t* origins);
}

// -- Using a sequence of linear indexes -- //
namespace noa::cpu::memory {
    /// Extracts elements (and/or indexes) from the input arrays(s) based on an unary bool operator.
    /// \tparam EXTRACT         Whether elements should be extracted.
    /// \tparam I               Integral type of the indexes of the extracted elements.
    ///                         These indexes are mostly used when the extracted elements needs to be inserted
    ///                         back into the input array(s). If \p I is void, the indexes are not returned.
    /// \param[in] inputs       On the \b host. Input arrays(s) to extract from.
    /// \param pitch            Pitch, in elements, of \p inputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs.
    /// \param batches          Number of batches to compute.
    /// \param unary_op         Unary operation function object that will be used as criterion to extract elements.
    ///                         Each element is passed through that operator and if the return value
    ///                         evaluates to true, the element is extracted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \return                 1: Extracted elements. If \p EXTRACT is false, returns nullptr.
    ///                         2: Sequence of indexes. If \p I is void, returns nullptr.
    ///                         3: Number of extracted elements.
    /// \note The callee is the owner of the returned pointer. Use PtrHost::dealloc() to free them.
    template<bool EXTRACT = true, typename I = void, typename T, typename UnaryOp>
    NOA_HOST std::tuple<T*, I*, size_t> extract(const T* inputs, size3_t pitch, size3_t shape, size_t batches,
                                                UnaryOp unary_op, Stream& stream);

    /// Extracts elements (and/or indexes) from the input arrays(s) based on a binary bool operator.
    /// \tparam EXTRACT         Whether elements should be extracted.
    /// \tparam I               Integral type of the indexes of the extracted elements.
    ///                         These indexes are mostly used when the extracted elements needs to be inserted
    ///                         back into the input array(s). If \p I is void, the indexes are not returned.
    /// \param[in] inputs       On the \b host. Input arrays(s) to extract from.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs.
    /// \param values           Value(s) to use as right-hand side argument.
    ///                         If \p U is a pointer, there should be one per batch.
    ///                         Otherwise, the same value is applied to all batches.
    /// \param batches          Number of batches to compute.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element and \p values are passed through that operator and if the return value
    ///                         evaluates to true, the element is extracted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \return                 1: Extracted elements. If \p EXTRACT is false, returns nullptr.
    ///                         2: Sequence of indexes. If \p I is void, returns nullptr.
    ///                         3: Number of extracted elements.
    template<bool EXTRACT = true, typename I = void, typename T, typename U, typename BinaryOp>
    NOA_HOST std::tuple<T*, I*, size_t> extract(const T* inputs, size3_t input_pitch, size3_t shape, U values,
                                                size_t batches, BinaryOp binary_op, Stream& stream);

    /// Extracts elements (and/or indexes) from the input arrays(s) based on a binary bool operator.
    /// \tparam EXTRACT         Whether elements should be extracted.
    /// \tparam I               Integral type of the indexes of the extracted elements.
    ///                         These indexes are mostly used when the extracted elements needs to be inserted
    ///                         back into the input array(s). If \p I is void, the indexes are not returned.
    /// \param[in] inputs       On the \b host. Input arrays(s) to extract from. One or one per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[in] arrays       On the \b host. Array(s) to use as right-hand side argument. One or one per batch.
    /// \param array_pitch      Pitch, in elements, of \p arrays.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs and \p arrays.
    /// \param batches          Number of batches to compute.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of \p inputs and \p arrays are passed through that operator and if the
    ///                         return value evaluates to true, the element is extracted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \return                 1: Extracted elements. If \p EXTRACT is false, returns nullptr.
    ///                         2: Sequence of indexes. If \p I is void, returns nullptr.
    ///                         3: Number of extracted elements.
    template<bool EXTRACT = true, typename I = void, typename T, typename U, typename BinaryOp>
    NOA_HOST std::tuple<T*, I*, size_t> extract(const T* inputs, size3_t input_pitch,
                                                const U* arrays, size3_t array_pitch,
                                                size3_t shape, size_t batches, BinaryOp binary_op, Stream& stream);

    /// Inserts elements into the \p output.
    /// \param[in] sequence_values  On the \b host. Sequence of values that were extracted and need to be reinserted.
    /// \param[in] sequence_indexes On the \b host. Linear indexes in \p output where the values should be inserted.
    /// \param sequence_size        Number of elements to insert.
    /// \param[out] output          On the \b host. Output array inside which the values are going to be inserted.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename I>
    NOA_HOST void insert(const T* sequence_values, const I* sequence_indexes, size_t sequence_size,
                         T* output, Stream& stream);
}

#define NOA_INDEX_INL_
#include "noa/cpu/memory/Index.inl"
#undef NOA_INDEX_INL_
