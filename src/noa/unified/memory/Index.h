#pragma once

#include "noa/unified/Array.h"

namespace noa::memory {
    /// Extracts one or multiple ND (1 <= N <= 3) subregions at various locations in the input array.
    /// \tparam T               Any data type.
    /// \param[in] input        Input array to extract from.
    /// \param[out] subregions  Output subregion(s).
    /// \param[in] origins      Rightmost indexes, defining the origin where to extract subregions from \p input.
    ///                         Should be a row vector with a set of 4 indexes per subregion. The outermost
    ///                         dimension of \p subregions is the batch dimension and sets the number of subregions
    ///                         to extract. While usually within the input frame, subregions can be (partially)
    ///                         out-of-bound.
    /// \param border_mode      Border mode used for out-of-bound conditions.
    ///                         Can be BORDER_{NOTHING|ZERO|VALUE|CLAMP|MIRROR|REFLECT}.
    /// \param border_value     Constant value to use for out-of-bound conditions.
    ///                         Only used if \p border_mode is BORDER_VALUE.
    /// \note \p input and \p subregions should not overlap.
    /// \note On the GPU, \p origins can be on any device, including the CPU.
    template<typename T>
    void extract(const Array<T>& input, const Array<T>& subregions, const Array<int4_t>& origins,
                 BorderMode border_mode = BORDER_ZERO, T border_value = T(0));

    /// Inserts into the output array one or multiple ND (1 <= N <= 3) subregions at various locations.
    /// \tparam T                   Any data type.
    /// \param[in] subregions       Subregion(s) to insert into \p output.
    /// \param[out] output          Output array.
    /// \param[in] origins          Rightmost indexes, defining the origin where to insert subregions into \p output.
    ///                             Should be a row vector with a set of 4 indexes per subregion. The outermost
    ///                             dimension of \p subregion_shape is the batch dimension and sets the number of
    ///                             subregions to insert. Thus, subregions can be up to 3 dimensions. While usually
    ///                             within the output frame, subregions can be (partially) out-of-bound. However,
    ///                             this function assumes no overlap between subregions. There's no guarantee on the
    ///                             order of insertion.
    template<typename T>
    void insert(const Array<T>& subregions, const Array<T>& output, const Array<int4_t>& origins);

    /// Gets the atlas layout (shape + subregion origins).
    /// \param subregion_shape          Rightmost shape of the subregion(s).
    ///                                 The outermost dimension is the number of subregion(s) to place into the atlas.
    /// \param[out] origins             Subregion origin(s), relative to the atlas shape.
    /// \return                         Atlas shape.
    ///
    /// \note The shape of the atlas is not necessary a square. For instance, with 4 subregions the atlas layout
    ///       is `2x2`, but with 5 subregions is goes to `3x2` with one empty region. Subregions are in row-major order.
    /// \note The origin is always 0 for the two outermost dimensions. The function is effectively un-batching the
    ///       2D/3D subregions into a 2D/3D atlas.
    NOA_IH size4_t atlasLayout(size4_t subregion_shape, int4_t* origins);
}

namespace noa::memory {
    template<typename T, typename I>
    struct Extracted {
        Array<T> values;
        Array<I> offsets;
    };

    /// Extracts elements (and/or offsets) from the input array based on an unary bool operator.
    /// \tparam value_t         Any data type.
    /// \tparam offset_t        Integral type of the extracted elements' offsets.
    /// \tparam T, U            Any data type.
    /// \param[in] input        Input array to extract from.
    /// \param[lhs] lhs         Left-hand side argument.
    /// \param unary_op         Unary operation device function object that will be used as criterion to extract elements.
    ///                         Each element of \p lhs is passed through that operator and if the return value evaluates
    ///                         to true, the corresponding element in \p input is extracted.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \return 1: 1D array with the extracted values. Can be empty, depending on \p extract_values.
    ///         2: 1D array with the extracted offsets. Can be empty, depending on \p extract_offsets.
    ///         Both arrays are on the same device as \p input.
    /// \note On the GPU:
    ///         - \p value_t should be (u)int32_t, (u)int64_t, or any floating-point.
    ///         - \p offset_t should be uint32_t or uint64_t.
    ///         - \p T and \p U should be equal to \p value_t.
    ///         - \p unary_op is limited to math::logical_not_t.
    template<typename value_t, typename offset_t, typename T, typename U, typename UnaryOp>
    Extracted<value_t, offset_t> extract(const Array<T>& input, const Array<U>& lhs, UnaryOp unary_op,
                                         bool extract_values = true, bool extract_offsets = true);

    /// Extracts elements (and/or offsets) from the input array based on an binary bool operator.
    /// \tparam value_t         Any data type.
    /// \tparam offset_t        Integral type of the extracted elements' offsets.
    /// \tparam T, U, V         Any data type.
    /// \param[in] input        Input array to extract from.
    /// \param[in] lhs          Left-hand side argument.
    /// \param rhs              Right-hand side argument.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of \p lhs and \p rhs are passed through that operator and if the return
    ///                         value evaluates to true, the corresponding element in \p input is extracted.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \return 1: 1D array with the extracted elements. Can be empty, depending on \p extract_values.
    ///         2: 1D array with the extracted offsets. Can be empty, depending on \p extract_offsets.
    ///         Both arrays are on the same device as \p input.
    /// \note On the GPU:
    ///         - \p value_t should be (u)int32_t, (u)int64_t, or any floating-point.
    ///         - \p offset_t should be uint32_t or uint64_t.
    ///         - \p T and \p U should be equal to \p value_t. \p V is casted to \p value_t.
    ///         - \p binary_op is limited to math::{equal|not_equal|less|less_equal|greater|greater_equal}_t.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_t> extract(const Array<T>& input, const Array<U>& lhs, V rhs, BinaryOp binary_op,
                                         bool extract_values = true, bool extract_offsets = true);

    /// Extracts elements (and/or offsets) from the input array based on an binary bool operator.
    /// \tparam value_t         Any data type.
    /// \tparam offset_t        Integral type of the extracted elements' offsets.
    /// \tparam T, U, V         Any data type.
    /// \param[in] input        Input array to extract from.
    /// \param lhs              Left-hand side argument.
    /// \param[in] rhs          Right-hand side argument.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         \p lhs and each element of \p rhs are passed through that operator and if the return
    ///                         value evaluates to true, the corresponding element in \p input is extracted.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \return 1: 1D array with the extracted elements. Can be empty, depending on \p extract_values.
    ///         2: 1D array with the extracted offsets. Can be empty, depending on \p extract_offsets.
    ///         Both arrays are on the same device as \p input.
    /// \note On the GPU:
    ///         - \p value_t should be (u)int32_t, (u)int64_t, or any floating-point.
    ///         - \p offset_t should be uint32_t or uint64_t.
    ///         - \p T and \p V should be equal to \p value_t. \p U is casted to \p value_t.
    ///         - \p binary_op is limited to math::{equal|not_equal|less|less_equal|greater|greater_equal}_t.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_t> extract(const Array<T>& input, U lhs, const Array<V>& rhs, BinaryOp binary_op,
                                         bool extract_values = true, bool extract_offsets = true);

    /// Extracts elements (and/or offsets) from the input array based on an binary bool operator.
    /// \tparam value_t         Any data type.
    /// \tparam offset_t        Integral type of the extracted elements' offsets.
    /// \tparam T, U, V         Any data type.
    /// \param[in] input        Input array to extract from.
    /// \param[in] lhs          Left-hand side argument.
    /// \param[in] rhs          Right-hand side argument.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of both \p lhs and \p rhs are passed through that operator and if the
    ///                         return value evaluates to true, the corresponding element in \p input is extracted.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    ///
    /// \return 1: 1D array with the extracted elements. Can be empty, depending on \p extract_values.
    ///         2: 1D array with the extracted offsets. Can be empty, depending on \p extract_offsets.
    ///         Both arrays are on the same device as \p input.
    /// \note On the GPU:
    ///         - \p value_t should be (u)int32_t, (u)int64_t, or any floating-point.
    ///         - \p offset_t should be uint32_t or uint64_t.
    ///         - \p T, \p U and \p V should be equal to \p value_t.
    ///         - \p binary_op is limited to math::{equal|not_equal|less|less_equal|greater|greater_equal}_t.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_t> extract(const Array<T>& input, const Array<U>& lhs, const Array<V>& rhs,
                                         BinaryOp binary_op, bool extract_values = true, bool extract_offsets = true);

    /// Inserts elements into \p output.
    /// \tparam value_t         Any data type.
    /// \tparam offset_t        Integral type of the extracted elements' offsets.
    /// \tparam T               Any data type.
    /// \param[in] extracted    1: Sequence of values that were extracted and need to be reinserted.
    ///                         2: Linear offsets in \p output where the values should be inserted.
    /// \param[out] output      Output array inside which the values are going to be inserted.
    /// \note On the GPU:
    ///         - \p value_t should be (u)int32_t, (u)int64_t, or any floating-point.
    ///         - \p offset_t should be uint32_t or uint64_t.
    ///         - \p T should be equal to \p value_t.
    template<typename value_t, typename offset_t, typename T>
    void insert(const Extracted<value_t, offset_t>& extracted, const Array<T>& output);

    /// Inserts elements into \p output.
    /// \tparam value_t     Any data type.
    /// \tparam offset_t     Integral type of the extracted elements' offsets.
    /// \tparam T           Any data type.
    /// \param[in] values   Sequence of values that were extracted and need to be reinserted.
    /// \param[in] offsets  Linear offsets in \p output where the values should be inserted.
    /// \param[out] output  Output array inside which the values are going to be inserted.
    /// \note On the GPU:
    ///         - \p value_t should be (u)int32_t, (u)int64_t, or any floating-point.
    ///         - \p offset_t should be uint32_t or uint64_t.
    ///         - \p T should be equal to \p value_t.
    template<typename value_t, typename offset_t, typename T>
    void insert(const Array<value_t>& values, const Array<offset_t>& offsets, const Array<T>& output);
}

#define NOA_UNIFIED_INDEX_
#include "noa/unified/memory/Index.inl"
#undef NOA_UNIFIED_INDEX_
