/// \file noa/gpu/cuda/memory/Index.h
/// \brief Remapping functions.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021
#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Functors.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

// -- Using coordinates -- //
namespace noa::cuda::memory {
    /// Extracts one or multiple ND (1 <= N <= 3) subregions at various locations in the input array.
    /// \tparam T                   Any data type.
    /// \param[in] input            On the \b device. Input array to use for the extraction.
    /// \param input_stride         Rightmost strides, in elements, of \p input.
    /// \param input_shape          Rightmost shape of \p input.
    /// \param[out] subregions      On the \b device. Output subregions.
    /// \param subregion_stride     Rightmost strides, in elements, of \p subregions.
    /// \param subregion_shape      Rightmost shape of \p subregions.
    /// \param[in] origins          On the \b host or device. One per batch.
    ///                             Rightmost indexes, defining the origin where to extract subregions from \p input.
    ///                             While usually within the input frame, subregions can be (partially) out-of-bound.
    ///                             The outermost dimension of \p subregion_shape is the batch dimension and sets the
    ///                             number of subregions to extract. Thus, subregions can be up to 3 dimensions.
    /// \param border_mode          Border mode used for out-of-bound conditions. Can be BORDER_NOTHING, BORDER_ZERO,
    ///                             BORDER_VALUE, BORDER_CLAMP, BORDER_MIRROR or BORDER_REFLECT.
    /// \param border_value         Constant value to use for out-of-bound conditions.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note \p input and \p subregions should not overlap.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T>
    void extract(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& subregions, size4_t subregion_stride, size4_t subregion_shape,
                 const shared_t<int4_t[]>& origins,
                 BorderMode border_mode, T border_value, Stream& stream);

    /// Inserts into the output array one or multiple ND (1 <= N <= 3) subregions at various locations.
    /// \tparam T                   Any data type.
    /// \param[in] subregions       On the \b device. Subregion(s) to insert into \p output.
    /// \param subregion_stride     Rightmost strides, in elements, of \p subregions.
    /// \param subregion_shape      Rightmost shape \p subregions.
    /// \param[out] output          On the \b device. Output array.
    /// \param output_stride        Rightmost strides, in elements, of \p output.
    /// \param output_shape         Rightmost shape of \p output.
    /// \param[in] origins          On the \b host or device. One per batch.
    ///                             Rightmost indexes, defining the origin where to insert subregions into \p output.
    ///                             While usually within the output frame, subregions can be (partially) out-of-bound.
    ///                             The outermost dimension of \p subregion_shape is the batch dimension and sets the
    ///                             number of subregions to insert. Thus, subregions can be up to 3 dimensions.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note \p outputs and \p subregions should not overlap.
    /// \note This function assumes no overlap between subregions. There's no guarantee on the order of insertion.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T>
    void insert(const shared_t<T[]>& subregions, size4_t subregion_stride, size4_t subregion_shape,
                const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                const shared_t<int4_t[]>& origins, Stream& stream);

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
    size4_t atlasLayout(size4_t subregion_shape, int4_t* origins);
}

// -- Using a sequence of linear indexes -- //
namespace noa::cuda::memory {
    template<typename T, typename I>
    struct Extracted {
        shared_t<T[]> elements{};
        shared_t<I[]> indexes{};
        size_t count{};
    };

    /// Extracts elements (and/or indexes) from the input array based on an unary bool operator.
    /// \tparam I               Integral type of the extracted elements' indexes. Either uint32_t, or uint64_t.
    ///                         These indexes are mostly used when the extracted elements needs to be inserted
    ///                         back into the input array.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b device. Input array to extract from.
    /// \param stride           Rightmost strides, in elements, of \p input.
    /// \param shape            Rightmost shape of \p input. The outermost dimension is the batch dimension.
    /// \param unary_op         Unary operation device function object that will be used as criterion to extract elements.
    ///                         Each element is passed through that operator and if the return value evaluates
    ///                         to true, the element is extracted. Supported unary operator: noa::math::logical_not_t.
    /// \param extract_elements Whether the elements should be extracted.
    /// \param extract_indexes  Whether the indexes should be extracted.
    /// \param[in,out] stream   Stream on which to enqueue this function. The stream is synchronized once.
    /// \return                 1: Extracted elements.
    ///                         2: Sequence of indexes.
    ///                         3: Number of extracted elements.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T, typename I, typename UnaryOp>
    Extracted<T, I> extract(const shared_t<T[]>& input, size4_t stride, size4_t shape,
                            UnaryOp unary_op, bool extract_elements, bool extract_indexes, Stream& stream);

    /// Extracts elements (and/or indexes) from the input array based on an binary bool operator.
    /// \tparam I               Integral type of the extracted elements' indexes. Either uint32_t, or uint64_t.
    ///                         These indexes are mostly used when the extracted elements needs to be inserted
    ///                         back into the input array.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double.
    /// \tparam U               Should be equal to \p T.
    /// \param[in] lhs          On the \b device. Input array to extract from. Used as left-hand side argument.
    /// \param lhs_stride       Rightmost strides, in elements, of \p lhs.
    /// \param lhs_shape        Rightmost shape of \p lhs.
    /// \param rhs              Value to use as right-hand side argument.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of \p lhs and \p rhs are passed through that operator and if the return
    ///                         value evaluates to true, the element in \p lhs is extracted.
    ///                         Supported noa::math binary operator: equal_t, not_equal_t, less_t, less_equal_t,
    ///                         greater_t, greater_equal_t.
    /// \param extract_elements Whether the elements should be extracted.
    /// \param extract_indexes  Whether the indexes should be extracted.
    /// \param[in,out] stream   Stream on which to enqueue this function. The stream is synchronized once.
    /// \return                 1: Extracted elements.
    ///                         2: Sequence of indexes.
    ///                         3: Number of extracted elements.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T, typename I, typename U, typename BinaryOp,
             typename = std::enable_if_t<!std::is_pointer_v<U>>>
    Extracted<T, I> extract(const shared_t<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape, U rhs,
                            BinaryOp binary_op, bool extract_elements, bool extract_indexes, Stream& stream);

    /// Extracts elements (and/or indexes) from the input array based on an binary bool operator.
    /// \tparam I               Integral type of the extracted elements' indexes. Either uint32_t, or uint64_t.
    ///                         These indexes are mostly used when the extracted elements needs to be inserted
    ///                         back into the input array.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double.
    /// \tparam U               Should be equal to \p T.
    /// \param lhs              Value to use as left-hand side argument.
    /// \param[in] rhs          On the \b device. Input array to extract from. Used as right-hand side argument.
    /// \param rhs_stride       Rightmost strides, in elements, of \p rhs.
    /// \param rhs_shape        Rightmost shape of \p rhs.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         \p lhs and each element of \p rhs are passed through that operator and if the return
    ///                         value evaluates to true, the element in \p rhs is extracted.
    ///                         Supported noa::math binary operator: equal_t, not_equal_t, less_t, less_equal_t,
    ///                         greater_t, greater_equal_t.
    /// \param extract_elements Whether the elements should be extracted.
    /// \param extract_indexes  Whether the indexes should be extracted.
    /// \param[in,out] stream   Stream on which to enqueue this function. The stream is synchronized once.
    /// \return                 1: Extracted elements.
    ///                         2: Sequence of indexes.
    ///                         3: Number of extracted elements.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T, typename I, typename U, typename BinaryOp,
             typename = std::enable_if_t<!std::is_pointer_v<T>>>
    Extracted<T, I> extract(T lhs, const shared_t<U[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
                            BinaryOp binary_op, bool extract_elements, bool extract_indexes, Stream& stream);

    /// Extracts elements (and/or indexes) from the input array based on an binary bool operator.
    /// \tparam I               Integral type of the extracted elements' indexes. Either uint32_t, or uint64_t.
    ///                         These indexes are mostly used when the extracted elements needs to be inserted
    ///                         back into the input array.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double.
    /// \tparam U               Should be equal to \p T.
    /// \param[in] lhs          On the \b device. Input array to extract from. Used as left-hand side argument.
    /// \param lhs_stride       Rightmost strides, in elements, of \p lhs.
    /// \param[in] rhs          On the \b device. Array used as right-hand side argument.
    /// \param rhs_stride       Rightmost strides, in elements, of \p rhs.
    /// \param shape            Rightmost shape of \p lhs. The outermost dimension is the batch dimension.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of both \p lhs and \p rhs are passed through that operator and if the
    ///                         return value evaluates to true, the element in \p lhs is extracted.
    ///                         Supported noa::math binary operator: equal_t, not_equal_t, less_t, less_equal_t,
    ///                         greater_t, greater_equal_t.
    /// \param extract_elements Whether the elements should be extracted.
    /// \param extract_indexes  Whether the indexes should be extracted.
    /// \param[in,out] stream   Stream on which to enqueue this function. The stream is synchronized once.
    /// \return                 1: Extracted elements.
    ///                         2: Sequence of indexes.
    ///                         3: Number of extracted elements.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T, typename I, typename U, typename BinaryOp>
    Extracted<T, I> extract(const shared_t<T[]>& lhs, size4_t lhs_stride,
                            const shared_t<U[]>& rhs, size4_t rhs_stride,
                            size4_t shape, BinaryOp binary_op, bool extract_elements, bool extract_indexes,
                            Stream& stream);

    /// Inserts elements into \p output.
    /// \tparam I               uint32_t, uint64_t.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] extracted    1: On the \b device. Sequence of values that were extracted and need to be reinserted.
    ///                         2: On the \b device. Linear indexes in \p output where the values should be inserted.
    ///                         3: Number of elements to insert.
    /// \param[out] output      On the \b device. Output array inside which the values are going to be inserted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T, typename I>
    void insert(const Extracted<T, I>& extracted, const shared_t<T[]>& output, Stream& stream);
}
