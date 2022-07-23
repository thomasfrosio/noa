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
    /// Extracts one or multiple ND (1 <= N <= 3) subregions at various locations in the input array.
    /// \tparam T                   Any data type.
    /// \param[in] input            On the \b host. Input array to use for the extraction.
    /// \param input_strides        BDHW strides, in elements, of \p input.
    /// \param input_shape          BDHW shape of \p input.
    /// \param[out] subregions      On the \b host. Output subregions.
    /// \param subregion_strides    BDHW strides, in elements, of \p subregions.
    /// \param subregion_shape      BDHW shape of subregions.
    /// \param[in] origins          On the \b host. One per batch.
    ///                             BDHW indexes, defining the origin where to extract subregions from \p input.
    ///                             While usually within the input frame, subregions can be (partially) out-of-bound.
    ///                             The batch dimension of \p subregion_shape sets the number of subregions to extract.
    /// \param border_mode          Border mode used for out-of-bound conditions. Can be BORDER_NOTHING, BORDER_ZERO,
    ///                             BORDER_VALUE, BORDER_CLAMP, BORDER_MIRROR or BORDER_REFLECT.
    /// \param border_value         Constant value to use for out-of-bound. Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note \p input and \p subregions should not overlap.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void extract(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                 const shared_t<T[]>& subregions, size4_t subregion_strides, size4_t subregion_shape,
                 const shared_t<int4_t[]>& origins, BorderMode border_mode, T border_value,
                 Stream& stream);

    /// Inserts into the output array one or multiple ND (1 <= N <= 3) subregions at various locations.
    /// \tparam T                   Any data type.
    /// \param[in] subregions       On the \b host. Subregion(s) to insert into \p output.
    /// \param subregion_strides    BDHW strides, in elements, of \p subregions.
    /// \param subregion_shape      BDHW shape \p subregions.
    /// \param[out] output          On the \b host. Output array.
    /// \param output_strides       BDHW strides, in elements, of \p output.
    /// \param output_shape         BDHW shape of \p output.
    /// \param[in] origins          On the \b host. One per batch.
    ///                             BDHW indexes, defining the origin where to insert subregions into \p output.
    ///                             While usually within the output frame, subregions can be (partially) out-of-bound.
    ///                             The batch dimension of \p subregion_shape sets the number of subregions to insert.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note \p outputs and \p subregions should not overlap.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \warning This function assumes no overlap between subregions. There's no guarantee on the order of insertion.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void insert(const shared_t<T[]>& subregions, size4_t subregion_strides, size4_t subregion_shape,
                const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                const shared_t<int4_t[]>& origins, Stream& stream);

    /// Gets the atlas layout (shape + subregion origins).
    /// \param subregion_shape          BDHW shape of the subregion(s).
    ///                                 The batch dimension is the number of subregion(s) to place into the atlas.
    /// \param[out] origins             On the \b host. Subregion origin(s), relative to the atlas shape.
    ///                                 The Batch and Depth dimensions are always set to 0, hence the possibility
    ///                                 to return only the Height and Width dimension.
    /// \return                         Atlas shape.
    ///
    /// \note The shape of the atlas is not necessary a square. For instance, with 4 subregions the atlas layout
    ///       is `2x2`, but with 5 subregions it goes to `3x2` with one empty region. Subregions are saved in
    ///       \p origins in row-major order.
    template<typename T, typename = std::enable_if_t<traits::is_int2_v<T> || traits::is_int4_v<T>>>
    inline size4_t atlasLayout(size4_t subregion_shape, T* origins);
}

// -- Using a sequence of linear indexes -- //
namespace noa::cpu::memory {
    template<typename T, typename I>
    struct Extracted {
        shared_t<T[]> values{};
        shared_t<I[]> offsets{};
        size_t count{};
    };

    /// Extracts elements (and/or offsets) from the input array based on an unary bool operator.
    /// \tparam value_t         Any data type.
    /// \tparam offset_t        Any integral type.
    /// \tparam T, U            Any data type.
    /// \param[in] input        On the \b host. Input array to extract from.
    ///                         Can be nullptr if \p extract_values is false.
    /// \param input_strides    Strides, in elements, of \p input.
    /// \param[in] lhs          On the \b host. Used as left-hand side argument.
    /// \param lhs_strides      Strides, in elements, of \p lhs.
    /// \param shape            Shape of \p input and \p lhs.
    /// \param unary_op         Unary operation device function object that will be used as criterion to extract elements.
    ///                         Each element of \p lhs is passed through that operator and if the return value
    ///                         evaluates to true, the corresponding element in \p input is extracted.
    ///                         Supported unary operator: noa::math::logical_not_t.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \param[in,out] stream   Stream on which to enqueue this function. The stream is synchronized once.
    /// \return                 1: On the \b host. Extracted elements.
    ///                         2: On the \b host. Sequence of offsets.
    ///                         3: Number of extracted elements.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename value_t, typename offset_t, typename T, typename U, typename UnaryOp>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, size4_t input_strides,
                                         const shared_t<U[]>& lhs, size4_t lhs_strides, size4_t shape,
                                         UnaryOp unary_op, bool extract_values, bool extract_offsets, Stream& stream);

    /// Extracts elements (and/or offsets) from the input array based on an binary bool operator.
    /// \tparam value_t         Any data type.
    /// \tparam offset_t        Any integral type.
    /// \tparam T, U, V         Any data type.
    /// \param[in] input        On the \b host. Input array to extract from.
    ///                         Can be nullptr if \p extract_values is false.
    /// \param input_strides    Strides, in elements, of \p input.
    /// \param[in] lhs          On the \b host. Used as left-hand side argument.
    /// \param lhs_strides      Strides, in elements, of \p lhs.
    /// \param rhs              Value to use as right-hand side argument.
    /// \param shape            Shape of \p input and \p lhs.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of \p lhs and \p rhs are passed through that operator and if the return
    ///                         value evaluates to true, the corresponding element in \p input is extracted.
    ///                         Supported noa::math binary operator: equal_t, not_equal_t, less_t, less_equal_t,
    ///                         greater_t, greater_equal_t.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \param[in,out] stream   Stream on which to enqueue this function. The stream is synchronized once.
    /// \return                 1: On the \b host. Extracted elements.
    ///                         2: On the \b host. Sequence of offsets.
    ///                         3: Number of extracted elements.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, size4_t input_strides,
                                         const shared_t<U[]>& lhs, size4_t lhs_strides, V rhs, size4_t shape,
                                         BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream);

    /// Extracts elements (and/or offsets) from the input array based on an binary bool operator.
    /// \tparam value_t         Any data type.
    /// \tparam offset_t        Any integral type.
    /// \tparam T, U, V         Any data type.
    /// \param[in] input        On the \b host. Input array to extract from.
    ///                         Can be nullptr if \p extract_values is false.
    /// \param input_strides    Strides, in elements, of \p input.
    /// \param lhs              Value to use as left-hand side argument.
    /// \param[in] rhs          On the \b host. Used as right-hand side argument.
    /// \param rhs_strides      Strides, in elements, of \p rhs.
    /// \param shape            Shape of \p input and \p rhs.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         \p lhs and each element of \p rhs are passed through that operator and if the return
    ///                         value evaluates to true, the corresponding element in \p input is extracted.
    ///                         Supported noa::math binary operator: equal_t, not_equal_t, less_t, less_equal_t,
    ///                         greater_t, greater_equal_t.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \param[in,out] stream   Stream on which to enqueue this function. The stream is synchronized once.
    /// \return                 1: On the \b host. Extracted elements.
    ///                         2: On the \b host. Sequence of offsets.
    ///                         3: Number of extracted elements.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, size4_t input_strides,
                                         U lhs, const shared_t<V[]>& rhs, size4_t rhs_strides, size4_t shape,
                                         BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream);

    /// Extracts elements (and/or offsets) from the input array based on an binary bool operator.
    /// \tparam value_t         Any data type.
    /// \tparam offset_t        Any integral type.
    /// \tparam T, U, V         Any data type.
    /// \param[in] input        On the \b host. Input array to extract from.
    ///                         Can be nullptr if \p extract_values is false.
    /// \param input_strides    Strides, in elements, of \p input.
    /// \param[in] lhs          On the \b host. Used as left-hand side argument.
    /// \param lhs_strides      Strides, in elements, of \p lhs.
    /// \param[in] rhs          On the \b host. Used as right-hand side argument.
    /// \param rhs_strides      Strides, in elements, of \p rhs.
    /// \param shape            Shape of \p input, \p lhs and \p rhs.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of both \p lhs and \p rhs are passed through that operator and if the
    ///                         return value evaluates to true, the corresponding element in \p input is extracted.
    ///                         Supported noa::math binary operator: equal_t, not_equal_t, less_t, less_equal_t,
    ///                         greater_t, greater_equal_t.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \param[in,out] stream   Stream on which to enqueue this function. The stream is synchronized once.
    /// \return                 1: On the \b host. Extracted elements.
    ///                         2: On the \b host. Sequence of offsets.
    ///                         3: Number of extracted elements.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, size4_t input_strides,
                                         const shared_t<U[]>& lhs, size4_t lhs_strides,
                                         const shared_t<V[]>& rhs, size4_t rhs_strides,
                                         size4_t shape, BinaryOp binary_op, bool extract_values, bool extract_offsets,
                                         Stream& stream);

    /// Extracts elements from the input array at particular offsets.
    /// \tparam T,V             Any data type.
    /// \tparam U               Any integral type.
    /// \param[in] input        On the \b host. Input array containing the elements to extract.
    /// \param[in] offsets      On the \b host. Memory offsets at which to extract the elements in \p input.
    /// \param[out] output      On the \b host. Output array containing the extracted elements.
    ///                         The elements are written contiguously in the same order as specified in \p offsets.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T, typename U, typename V>
    void extract(const shared_t<T[]>& input, const shared_t<U[]>& offsets,
                 const shared_t<V[]>& output, size_t elements, Stream& stream);

    /// Inserts elements into \p output.
    /// \tparam value_t         Any data type.
    /// \tparam offset_t        Any integral type.
    /// \tparam T               Any data type.
    /// \param[in] extracted    1: On the \b host. Sequence of values that were extracted and need to be reinserted.
    ///                         2: On the \b host. Linear offsets in \p output where the values should be inserted.
    ///                         3: Number of elements to insert.
    /// \param[out] output      On the \b host. Output array inside which the values are going to be inserted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename value_t, typename offset_t, typename T>
    void insert(const Extracted<value_t, offset_t>& extracted, const shared_t<T[]>& output, Stream& stream);
}

#define NOA_INDEX_INL_
#include "noa/cpu/memory/Index.inl"
#undef NOA_INDEX_INL_
