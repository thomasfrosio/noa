/// \file noa/gpu/cuda/memory/Index.h
/// \brief Remapping functions.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021
#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Functors.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory::details {
    using namespace noa::math;
    using namespace noa::traits;

    template<typename in_t, typename lhs_t, typename out_t, typename idx_t, typename op_t>
    constexpr bool is_valid_extract_unary_v =
            (is_any_v<out_t, int32_t, int64_t, uint32_t, uint64_t> || is_float_v<out_t>) &&
            are_all_same_v<in_t, lhs_t, out_t> && is_any_v<idx_t, uint32_t, uint64_t> &&
            std::is_same_v<op_t, logical_not_t>;

    template<typename in_t, typename lhs_t, typename rhs_t, typename out_t, typename idx_t, typename op_t>
    constexpr bool is_valid_extract_binary_v =
            (is_any_v<out_t, int32_t, int64_t, uint32_t, uint64_t> || is_float_v<out_t>) &&
            are_all_same_v<in_t, lhs_t, rhs_t, out_t> && is_any_v<idx_t, uint32_t, uint64_t> &&
            is_any_v<op_t, equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t>;

    template<typename in_t, typename idx_t, typename out_t>
    constexpr bool is_valid_insert_v =
            (is_any_v<out_t, int32_t, int64_t, uint32_t, uint64_t> || is_float_v<out_t>) &&
            std::is_same_v<in_t, out_t> && is_any_v<idx_t, uint32_t, uint64_t>;
}

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
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
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
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
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
        shared_t<T[]> values{};
        shared_t<I[]> offsets{};
        size_t count{};
    };

    /// Extracts elements (and/or offsets) from the input array based on an unary bool operator.
    /// \tparam value_t         (u)int32_t, (u)int64_t, or any floating-point.
    /// \tparam offset_t        Integral type of the extracted elements' offsets. Either uint32_t, or uint64_t.
    /// \tparam T, U            Should be equal to \p value_t.
    /// \param[in] input        On the \b device. Input array to extract from.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[in] lhs          On the \b device. Used as left-hand side argument.
    /// \param lhs_stride       Rightmost strides, in elements, of \p lhs.
    /// \param shape            Rightmost shape of \p input and \p lhs.
    /// \param unary_op         Unary operation device function object that will be used as criterion to extract elements.
    ///                         Each element of \p lhs is passed through that operator and if the return value
    ///                         evaluates to true, the corresponding element in \p input is extracted.
    ///                         Supported unary operator: noa::math::logical_not_t.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \param[in,out] stream   Stream on which to enqueue this function. The stream is synchronized once.
    /// \return                 1: On the \b device. Extracted elements.
    ///                         2: On the \b device. Sequence of offsets.
    ///                         3: Number of extracted elements.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename value_t, typename offset_t, typename T, typename U, typename UnaryOp,
             typename = std::enable_if_t<details::is_valid_extract_unary_v<T, U, value_t, offset_t, UnaryOp>>>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, size4_t input_stride,
                                         const shared_t<U[]>& lhs, size4_t lhs_stride, size4_t shape,
                                         UnaryOp unary_op, bool extract_values, bool extract_offsets, Stream& stream);

    /// Extracts elements (and/or offsets) from the input array based on an binary bool operator.
    /// \tparam value_t         (u)int32_t, (u)int64_t, or any floating-point.
    /// \tparam offset_t        Integral type of the extracted elements' offsets. Either uint32_t, or uint64_t.
    /// \tparam T, U, V         Should be equal to \p value_t.
    /// \param[in] input        On the \b device. Input array to extract from.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[in] lhs          On the \b device. Used as left-hand side argument.
    /// \param lhs_stride       Rightmost strides, in elements, of \p lhs.
    /// \param rhs              Value to use as right-hand side argument.
    /// \param shape            Rightmost shape of \p input and \p lhs.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of \p lhs and \p rhs are passed through that operator and if the return
    ///                         value evaluates to true, the corresponding element in \p input is extracted.
    ///                         Supported noa::math binary operator: equal_t, not_equal_t, less_t, less_equal_t,
    ///                         greater_t, greater_equal_t.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \param[in,out] stream   Stream on which to enqueue this function. The stream is synchronized once.
    /// \return                 1: On the \b device. Extracted elements.
    ///                         2: On the \b device. Sequence of offsets.
    ///                         3: Number of extracted elements.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_extract_binary_v<T, U, V, value_t, offset_t, BinaryOp>>>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, size4_t input_stride,
                                         const shared_t<U[]>& lhs, size4_t lhs_stride, V rhs, size4_t shape,
                                         BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream);

    /// Extracts elements (and/or offsets) from the input array based on an binary bool operator.
    /// \tparam value_t         (u)int32_t, (u)int64_t, or any floating-point.
    /// \tparam offset_t        Integral type of the extracted elements' offsets. Either uint32_t, or uint64_t.
    /// \tparam T, U, V         Should be equal to \p value_t.
    /// \param[in] input        On the \b device. Input array to extract from.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param lhs              Value to use as left-hand side argument.
    /// \param[in] rhs          On the \b device. Used as right-hand side argument.
    /// \param rhs_stride       Rightmost strides, in elements, of \p rhs.
    /// \param shape            Rightmost shape of \p input and \p rhs.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         \p lhs and each element of \p rhs are passed through that operator and if the return
    ///                         value evaluates to true, the corresponding element in \p input is extracted.
    ///                         Supported noa::math binary operator: equal_t, not_equal_t, less_t, less_equal_t,
    ///                         greater_t, greater_equal_t.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \param[in,out] stream   Stream on which to enqueue this function. The stream is synchronized once.
    /// \return                 1: On the \b device. Extracted elements.
    ///                         2: On the \b device. Sequence of offsets.
    ///                         3: Number of extracted elements.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_extract_binary_v<T, U, V, value_t, offset_t, BinaryOp>>>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, size4_t input_stride,
                                         U lhs, const shared_t<V[]>& rhs, size4_t rhs_stride, size4_t shape,
                                         BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream);

    /// Extracts elements (and/or offsets) from the input array based on an binary bool operator.
    /// \tparam value_t         (u)int32_t, (u)int64_t, or any floating-point.
    /// \tparam offset_t        Integral type of the extracted elements' offsets. Either uint32_t, or uint64_t.
    /// \tparam T, U, V         Should be equal to \p value_t.
    /// \param[in] input        On the \b device. Input array to extract from.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[in] lhs          On the \b device. Used as left-hand side argument.
    /// \param lhs_stride       Rightmost strides, in elements, of \p lhs.
    /// \param[in] rhs          On the \b device. Used as right-hand side argument.
    /// \param rhs_stride       Rightmost strides, in elements, of \p rhs.
    /// \param shape            Rightmost shape of \p input, \p lhs and \p rhs.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of both \p lhs and \p rhs are passed through that operator and if the
    ///                         return value evaluates to true, the corresponding element in \p input is extracted.
    ///                         Supported noa::math binary operator: equal_t, not_equal_t, less_t, less_equal_t,
    ///                         greater_t, greater_equal_t.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_offsets  Whether the offsets should be extracted. These offsets are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \param[in,out] stream   Stream on which to enqueue this function. The stream is synchronized once.
    /// \return                 1: On the \b device. Extracted elements.
    ///                         2: On the \b device. Sequence of offsets.
    ///                         3: Number of extracted elements.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_extract_binary_v<T, U, V, value_t, offset_t, BinaryOp>>>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, size4_t input_stride,
                                         const shared_t<U[]>& lhs, size4_t lhs_stride,
                                         const shared_t<V[]>& rhs, size4_t rhs_stride,
                                         size4_t shape, BinaryOp binary_op, bool extract_values, bool extract_offsets,
                                         Stream& stream);

    /// Extracts elements from the input array at particular offsets.
    /// \tparam T,V             (u)int32_t, (u)int64_t, or any floating-point.
    /// \tparam U               uint32_t, uint64_t.
    /// \param[in] input        On the \b device. Input array containing the elements to extract.
    /// \param[in] offsets      On the \b device. Memory offsets at which to extract the elements in \p input.
    /// \param[out] output      On the \b device. Output array containing the extracted elements.
    ///                         The elements are written contiguously in the same order as specified in \p offsets.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T, typename U, typename V,
            typename = std::enable_if_t<details::is_valid_insert_v<T, U, V>>>
    void extract(const shared_t<T[]>& input, const shared_t<U[]>& offsets,
                 const shared_t<V[]>& output, size_t elements, Stream& stream);

    /// Inserts elements into \p output.
    /// \tparam value_t         (u)int32_t, (u)int64_t, or any floating-point.
    /// \tparam offset_t        uint32_t, uint64_t.
    /// \tparam T               Should be equal to \p V.
    /// \param[in] extracted    1: On the \b device. Sequence of values that were extracted and need to be reinserted.
    ///                         2: On the \b device. Linear offsets in \p output where the values should be inserted.
    ///                         3: Number of elements to insert.
    /// \param[out] output      On the \b device. Output array inside which the values are going to be inserted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename value_t, typename offset_t, typename T,
             typename = std::enable_if_t<details::is_valid_insert_v<value_t, offset_t, T>>>
    void insert(const Extracted<value_t, offset_t>& extracted, const shared_t<T[]>& output, Stream& stream);
}
