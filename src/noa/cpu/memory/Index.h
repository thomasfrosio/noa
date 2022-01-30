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
    /// Extracts one or multiple (up to 3D) subregions at various locations in the input array.
    /// \tparam T                   Any data type.
    /// \param[in] input            On the \b host. Input array to use for the extraction.
    /// \param input_stride         Rightmost strides, in elements, of \p input.
    /// \param input_shape          Rightmost shape of \p input.
    /// \param[out] subregions      On the \b host. Output subregions.
    /// \param subregion_stride     Rightmost strides, in elements, of \p subregions.
    /// \param subregion_shape      Rightmost shape of subregions.
    /// \param[in] origins          On the \b host. One per batch.
    ///                             Rightmost indexes, defining the origin where to extract subregions into \p input.
    ///                             While usually within the input frame, subregions can be (partially) out-of-bound.
    ///                             The outermost dimension of \p subregion_shape is the batch dimension and sets the
    ///                             number of subregions to extract. Thus, subregions can be up to 3 dimensions.
    /// \param border_mode          Border mode used for out-of-bound conditions. Can be BORDER_NOTHING, BORDER_ZERO,
    ///                             BORDER_VALUE, BORDER_CLAMP, BORDER_MIRROR or BORDER_REFLECT.
    /// \param border_value         Constant value to use for out-of-bound. Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ////
    /// \note \p input and \p subregions should not overlap.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void extract(const T* input, size4_t input_stride, size4_t input_shape,
                          T* subregions, size4_t subregion_stride, size4_t subregion_shape,
                          const int4_t* origins, BorderMode border_mode, T border_value, Stream& stream);

    /// Inserts into the output array(s) one or multiple subregions (with the same shape) at various locations.
    /// \tparam T                   Any data type.
    /// \param[in] subregions       On the \b host. Subregion(s) to insert into \p outputs.
    /// \param subregion_stride     Rightmost strides, in elements, of \p subregions.
    /// \param subregion_shape      Rightmost shape of one subregion.
    /// \param[out] output          On the \b host. Output array(s).
    /// \param output_stride        Rightmost strides, in elements, of \p outputs.
    /// \param output_shape         Rightmost shape of \p outputs.
    /// \param[in] origins          On the \b host. One per batch.
    ///                             Rightmost indexes, defining the origin where to insert subregions into \p output.
    ///                             While usually within the output frame, subregions can be (partially) out-of-bound.
    ///                             The outermost dimension of \p subregion_shape is the batch dimension and sets the
    ///                             number of subregions to insert. Thus, subregions can be up to 3 dimensions.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note \p outputs and \p subregions should not overlap.
    /// \note This function assumes no overlap between subregions. There's no guarantee on the order of insertion.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void insert(const T* subregions, size4_t subregion_stride, size4_t subregion_shape,
                         T* output, size4_t output_stride, size4_t output_shape,
                         const int4_t* origins, Stream& stream);

    /// Gets the atlas layout (shape + subregion origins).
    /// \param subregion_shape          Rightmost shape of the subregion(s).
    ///                                 The outermost dimension is the number of subregion(s) to place into the atlas.
    /// \param[out] origins             On the \b host. Subregion origin(s), relative to the atlas shape.
    /// \return                         Atlas shape.
    ///
    /// \note The shape of the atlas is not necessary a square. For instance, with 4 subregions the atlas layout
    ///       is `2x2`, but with 5 subregions is goes to `3x2` with one empty region. Subregions are in row-major order.
    /// \note The origin is always 0 for the two outermost dimensions. The function is effectively un-batching the
    ///       2D/3D subregions into a 2D/3D atlas.
    NOA_IH size4_t atlasLayout(size4_t subregion_shape, int4_t* origins);
}

// -- Using a sequence of linear indexes -- //
namespace noa::cpu::memory {
    /// Extracts elements (and/or indexes) from the input array based on an unary bool operator.
    /// \tparam EXTRACT         Whether elements should be extracted.
    /// \tparam I               Integral type of the extracted elements' indexes.
    ///                         These indexes are mostly used when the extracted elements needs to be inserted
    ///                         back into the input array. If \p I is void, the indexes are not returned.
    /// \param[in] input        On the \b host. Input array to extract from.
    /// \param stride           Rightmost strides, in elements, of \p input.
    /// \param shape            Rightmost shape of \p input. The outermost dimension is the batch dimension.
    /// \param unary_op         Unary operation function object that will be used as criterion to extract elements.
    ///                         Each element is passed through that operator and if the return value evaluates
    ///                         to true, the element is extracted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \return                 1: Extracted elements. If \p EXTRACT is false, returns nullptr.
    ///                         2: Sequence of indexes. If \p I is void, returns nullptr.
    ///                         3: Number of extracted elements.
    /// \note The callee is the owner of the returned pointers. Use PtrHost::dealloc() to free them.
    template<bool EXTRACT = true, typename I = void, typename T, typename UnaryOp>
    NOA_HOST std::tuple<T*, I*, size_t> extract(const T* input, size4_t stride, size4_t shape,
                                                UnaryOp unary_op, Stream& stream);

    /// Extracts elements (and/or indexes) from the input array based on a binary bool operator.
    /// \tparam EXTRACT         Whether elements should be extracted.
    /// \tparam I               Integral type of the extracted elements' indexes.
    ///                         These indexes are mostly used when the extracted elements needs to be inserted
    ///                         back into the input array. If \p I is void, the indexes are not returned.
    /// \param[in] input        On the \b host. Input array to extract from.
    /// \param stride           Rightmost strides, in elements, of \p input.
    /// \param shape            Rightmost shape of \p input. The outermost dimension is the batch dimension.
    /// \param values           Value(s) to use as right-hand side argument.
    ///                         If \p U is a pointer, there should be one per batch.
    ///                         Otherwise, the same value is applied to all batches.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element and the current value are passed through that operator and if the
    ///                         return value evaluates to true, the element is extracted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \return                 1: Extracted elements. If \p EXTRACT is false, returns nullptr.
    ///                         2: Sequence of indexes. If \p I is void, returns nullptr.
    ///                         3: Number of extracted elements.
    template<bool EXTRACT = true, typename I = void, typename T, typename U, typename BinaryOp>
    NOA_HOST std::tuple<T*, I*, size_t> extract(const T* input, size4_t stride, size4_t shape, U values,
                                                BinaryOp binary_op, Stream& stream);

    /// Extracts elements (and/or indexes) from the input array based on a binary bool operator.
    /// \tparam EXTRACT         Whether elements should be extracted.
    /// \tparam I               Integral type of the extracted elements' indexes.
    ///                         These indexes are mostly used when the extracted elements needs to be inserted
    ///                         back into the input array. If \p I is void, the indexes are not returned.
    /// \param[in] input        On the \b host. Input array to extract from.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[in] array        On the \b host. Array to use as right-hand side argument.
    /// \param array_stride     Rightmost strides, in elements, of \p array.
    /// \param shape            Rightmost shape of \p input and \p array. The outermost dimension is the batch dimension.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of \p input and \p array are passed through that operator and if the
    ///                         return value evaluates to true, the element is extracted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \return                 1: Extracted elements. If \p EXTRACT is false, returns nullptr.
    ///                         2: Sequence of indexes. If \p I is void, returns nullptr.
    ///                         3: Number of extracted elements.
    template<bool EXTRACT = true, typename I = void, typename T, typename U, typename BinaryOp>
    NOA_HOST std::tuple<T*, I*, size_t> extract(const T* input, size4_t input_stride,
                                                const U* array, size4_t array_stride,
                                                size4_t shape, BinaryOp binary_op, Stream& stream);

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
