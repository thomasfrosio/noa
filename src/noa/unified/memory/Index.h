#pragma once

#include "noa/cpu/memory/Index.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Index.h"
#endif

#include "noa/unified/Array.h"

// TODO Add sequences

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
                 BorderMode border_mode = BORDER_ZERO, T border_value = T(0)) {
        NOA_CHECK(subregions.device() == input.device(),
                  "The input and subregion arrays must be on the same device, but got input:{} and subregion:{}",
                  subregions.device(), input.device());
        NOA_CHECK(origins.shape().ndim() == 1 && origins.shape()[3] == subregions.shape()[0],
                  "The indexes should be specified as a row vector of shape {} but got {}",
                  int4_t{1, 1, 1, subregions.shape()[0]}, origins.shape());
        NOA_CHECK(subregions.get() != input.get(), "The subregion(s) and the output arrays should not overlap");

        const Device device(subregions.device());
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(origins.dereferencable(), "The origins should be accessible to the CPU");
            cpu::memory::extract<T>(input.share(), input.stride(), input.shape(),
                                    subregions.share(), subregions.stride(), subregions.shape(),
                                    origins.share(), border_mode, border_value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::extract<T>(input.share(), input.stride(), input.shape(),
                                     subregions.share(), subregions.stride(), subregions.shape(),
                                     origins.share(), border_mode, border_value, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

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
    void insert(const Array<T>& subregions, const Array<T>& output, const Array<int4_t>& origins) {
        NOA_CHECK(subregions.device() == output.device(),
                  "The output and subregion arrays must be on the same device, but got output:{} and subregion:{}",
                  subregions.device(), output.device());
        NOA_CHECK(origins.shape().ndim() == 1 && origins.shape()[3] == subregions.shape()[0],
                  "The indexes should be specified as a row vector of shape {} but got {}",
                  int4_t{1, 1, 1, subregions.shape()[0]}, origins.shape());
        NOA_CHECK(subregions.get() != output.get(), "The subregion(s) and the output arrays should not overlap");

        const Device device(subregions.device());
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(origins.dereferencable(), "The origins should be accessible to the CPU");
            cpu::memory::insert<T>(subregions.share(), subregions.stride(), subregions.shape(),
                                   output.share(), output.stride(), output.shape(),
                                   origins.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::insert<T>(subregions.share(), subregions.stride(), subregions.shape(),
                                    output.share(), output.stride(), output.shape(),
                                    origins.share(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

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
    NOA_IH size4_t atlasLayout(size4_t subregion_shape, int4_t* origins) {
        return cpu::memory::atlasLayout(subregion_shape, origins);
    }
}

namespace noa::memory {
    template<typename T, typename I>
    struct Extracted {
        Array<T> values{};
        Array<I> indexes{};
    };

    /// Extracts elements (and/or indexes) from the input array based on an unary bool operator.
    /// \tparam value_t         Any data type.
    /// \tparam index_t         Integral type of the extracted elements' indexes.
    /// \tparam T, U            Any data type.
    /// \param[in] input        Input array to extract from.
    /// \param[lhs] lhs         Left-hand side argument.
    /// \param unary_op         Unary operation device function object that will be used as criterion to extract elements.
    ///                         Each element of \p lhs is passed through that operator and if the return value evaluates
    ///                         to true, the corresponding element in \p input is extracted.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_indexes  Whether the indexes should be extracted. These indexes are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \return 1: 1D array with the extracted values. Can be empty, depending on \p extract_values.
    ///         2: 1D array with the extracted indexes. Can be empty, depending on \p extract_indexes.
    ///         Both arrays are on the same device as \p input.
    /// \note On the GPU:
    ///         - \p value_t should be (u)int32_t, (u)int64_t, and any (complex) floating-point.
    ///         - \p index_t should be uint32_t or uint64_t.
    ///         - \p T and \p U should be equal to \p value_t.
    ///         - \p unary_op is limited to math::logical_not_t.
    template<typename value_t, typename index_t, typename T, typename U, typename UnaryOp>
    Extracted<value_t, index_t> extract(const Array<T>& input, const Array<U>& lhs, UnaryOp unary_op,
                                        bool extract_values = true, bool extract_indexes = true) {
        NOA_CHECK(all(input.shape() == lhs.shape()),
                  "The input arrays should have the same shape, but got input:{} and lhs:{}",
                  input.shape(), lhs.shape());
        NOA_CHECK(input.device() == lhs.device(),
                  "The input arrays should be on the same device, but got input:{} and lhs:{}",
                  input.device(), lhs.device());

        Extracted<value_t, index_t> out;
        const Device device(input.device());
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            auto extracted = cpu::memory::extract<value_t, index_t>(
                    input.share(), input.stride(), lhs.share(), lhs.stride(), input.shape(),
                    unary_op, extract_values, extract_indexes, stream.cpu());
            out.values = Array<value_t>{extracted.values, extracted.count, input.options()};
            out.indexes = Array<index_t>{extracted.indexes, extracted.count, input.options()};
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_extract_unary_v<T, U, value_t, index_t, UnaryOp>) {
                auto extracted = cuda::memory::extract<value_t, index_t>(
                        input.share(), input.stride(), lhs.share(), lhs.stride(), input.shape(),
                        unary_op, extract_values, extract_indexes, stream.cuda());
                out.values = Array<value_t>{extracted.values, extracted.count, input.options()};
                out.indexes = Array<index_t>{extracted.indexes, extracted.count, input.options()};
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        return out;
    }

    /// Extracts elements (and/or indexes) from the input array based on an binary bool operator.
    /// \tparam value_t         Any data type.
    /// \tparam index_t         Integral type of the extracted elements' indexes.
    /// \tparam T, U, V         Any data type.
    /// \param[in] input        Input array to extract from.
    /// \param[in] lhs          Left-hand side argument.
    /// \param rhs              Right-hand side argument.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of \p lhs and \p rhs are passed through that operator and if the return
    ///                         value evaluates to true, the corresponding element in \p input is extracted.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_indexes  Whether the indexes should be extracted. These indexes are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \return 1: 1D array with the extracted elements. Can be empty, depending on \p extract_values.
    ///         2: 1D array with the extracted indexes. Can be empty, depending on \p extract_indexes.
    ///         Both arrays are on the same device as \p input.
    /// \note On the GPU:
    ///         - \p value_t should be (u)int32_t, (u)int64_t, and any (complex) floating-point.
    ///         - \p index_t should be uint32_t or uint64_t.
    ///         - \p T, \p U and \p V should be equal to \p value_t.
    ///         - \p binary_op is limited to math::{equal|not_equal|less|less_equal|greater|greater_equal}_t.
    template<typename value_t, typename index_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, index_t> extract(const Array<T>& input, const Array<U>& lhs, V rhs, BinaryOp binary_op,
                                        bool extract_values = true, bool extract_indexes = true) {
        NOA_CHECK(all(input.shape() == lhs.shape()),
                  "The input arrays should have the same shape, but got input:{} and lhs:{}",
                  input.shape(), lhs.shape());
        NOA_CHECK(input.device() == lhs.device(),
                  "The input arrays should be on the same device, but got input:{} and lhs:{}",
                  input.device(), lhs.device());

        Extracted<value_t, index_t> out;
        const Device device = lhs.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            auto extracted = cpu::memory::extract<value_t, index_t>(
                    input.share(), input.stride(), lhs.share(), lhs.stride(), rhs, lhs.shape(),
                    binary_op, extract_values, extract_indexes, stream.cpu());
            out.values = Array<V>{extracted.values, extracted.count, lhs.options()};
            out.indexes = Array<T>{extracted.indexes, extracted.count, lhs.options()};
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_extract_binary_v<T, U, T, value_t, index_t, BinaryOp>) {
                auto extracted = cuda::memory::extract<value_t, index_t>(
                        input.share(), input.stride(), lhs.share(), lhs.stride(), static_cast<T>(rhs), lhs.shape(),
                        binary_op, extract_values, extract_indexes, stream.cuda());
                out.values = Array<V>{extracted.values, extracted.count, lhs.options()};
                out.indexes = Array<T>{extracted.indexes, extracted.count, lhs.options()};
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        return out;
    }

    /// Extracts elements (and/or indexes) from the input array based on an binary bool operator.
    /// \tparam value_t         Any data type.
    /// \tparam index_t         Integral type of the extracted elements' indexes.
    /// \tparam T, U, V         Any data type.
    /// \param[in] input        Input array to extract from.
    /// \param[in] lhs          Left-hand side argument.
    /// \param rhs              Right-hand side argument.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         \p lhs and each element of \p rhs are passed through that operator and if the return
    ///                         value evaluates to true, the corresponding element in \p input is extracted.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_indexes  Whether the indexes should be extracted. These indexes are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    /// \return 1: 1D array with the extracted elements. Can be empty, depending on \p extract_values.
    ///         2: 1D array with the extracted indexes. Can be empty, depending on \p extract_indexes.
    ///         Both arrays are on the same device as \p input.
    /// \note On the GPU:
    ///         - \p value_t should be (u)int32_t, (u)int64_t, and any (complex) floating-point.
    ///         - \p index_t should be uint32_t or uint64_t.
    ///         - \p T, \p U and \p V should be equal to \p value_t.
    ///         - \p binary_op is limited to math::{equal|not_equal|less|less_equal|greater|greater_equal}_t.
    template<typename value_t, typename index_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, index_t> extract(const Array<T>& input, U lhs, const Array<V>& rhs, BinaryOp binary_op,
                                        bool extract_values, bool extract_indexes) {
        NOA_CHECK(all(input.shape() == rhs.shape()),
                  "The input arrays should have the same shape, but got input:{} and rhs:{}",
                  input.shape(), rhs.shape());
        NOA_CHECK(input.device() == rhs.device(),
                  "The input arrays should be on the same device, but got input:{} and rhs:{}",
                  input.device(), rhs.device());

        Extracted<value_t, index_t> out;
        const Device device = rhs.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            auto extracted = cpu::memory::extract<value_t, index_t>(
                    input.share(), input.stride(), lhs, rhs.share(), rhs.stride(), rhs.shape(),
                    binary_op, extract_values, extract_indexes, stream.cpu());
            out.values = Array<V>{extracted.values, extracted.count, lhs.options()};
            out.indexes = Array<T>{extracted.indexes, extracted.count, lhs.options()};
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_extract_binary_v<T, U, T, value_t, index_t, BinaryOp>) {
                auto extracted = cuda::memory::extract<value_t, index_t>(
                        input.share(), input.stride(), static_cast<T>(lhs), rhs.share(), rhs.stride(), rhs.shape(),
                        binary_op, extract_values, extract_indexes, stream.cuda());
                out.values = Array<V>{extracted.values, extracted.count, lhs.options()};
                out.indexes = Array<T>{extracted.indexes, extracted.count, lhs.options()};
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        return out;
    }

    /// Extracts elements (and/or indexes) from the input array based on an binary bool operator.
    /// \tparam value_t         Any data type.
    /// \tparam index_t         Integral type of the extracted elements' indexes.
    /// \tparam T, U, V         Any data type.
    /// \param[in] input        Input array to extract from.
    /// \param[in] lhs          Left-hand side argument.
    /// \param rhs              Right-hand side argument.
    /// \param binary_op        Binary operation function object that will be used as criterion to extract elements.
    ///                         Each element of both \p lhs and \p rhs are passed through that operator and if the
    ///                         return value evaluates to true, the corresponding element in \p input is extracted.
    /// \param extract_values   Whether the elements should be extracted.
    /// \param extract_indexes  Whether the indexes should be extracted. These indexes are mostly used when the
    ///                         extracted elements needs to be inserted back into the input array.
    ///
    /// \return 1: 1D array with the extracted elements. Can be empty, depending on \p extract_values.
    ///         2: 1D array with the extracted indexes. Can be empty, depending on \p extract_indexes.
    ///         Both arrays are on the same device as \p input.
    /// \note On the GPU:
    ///         - \p value_t should be (u)int32_t, (u)int64_t, and any (complex) floating-point.
    ///         - \p index_t should be uint32_t or uint64_t.
    ///         - \p T, \p U and \p V should be equal to \p value_t.
    ///         - \p binary_op is limited to math::{equal|not_equal|less|less_equal|greater|greater_equal}_t.
    template<typename value_t, typename index_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, index_t> extract(const Array<T>& input, const Array<U>& lhs, const Array<V>& rhs,
                                        BinaryOp binary_op, bool extract_values, bool extract_indexes) {
        NOA_CHECK(all(input.shape() == lhs.shape()) && all(input.shape() == rhs.shape()),
                  "The input arrays should have the same shape, but got input:{}, lhs:{} and rhs:{}",
                  input.shape(), lhs.shape(), rhs.shape());
        NOA_CHECK(input.device() == lhs.device() && input.device() == rhs.device(),
                  "The input arrays should be on the same device, but got input:{}, lhs:{} and rhs:{}",
                  input.device(), lhs.device(), rhs.device());

        Extracted<value_t, index_t> out;
        const Device device = lhs.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            auto extracted = cpu::memory::extract<value_t, index_t>(
                    input.share(), input.stride(), lhs.share(), lhs.stride(),
                    rhs.share(), rhs.stride(), rhs.shape(), binary_op,
                    extract_values, extract_indexes, stream.cpu());
            out.values = Array<V>{extracted.values, extracted.count, lhs.options()};
            out.indexes = Array<T>{extracted.indexes, extracted.count, lhs.options()};
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_extract_binary_v<T, U, V, value_t, index_t, BinaryOp>) {
                auto extracted = cuda::memory::extract<value_t, index_t>(
                        input.share(), input.stride(), lhs.share(), lhs.stride(),
                        rhs.share(), rhs.stride(), rhs.shape(), binary_op,
                        extract_values, extract_indexes, stream.cuda());
                out.values = Array<V>{extracted.values, extracted.count, lhs.options()};
                out.indexes = Array<T>{extracted.indexes, extracted.count, lhs.options()};
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        return out;
    }

    /// Inserts elements into \p output.
    /// \tparam value_t         Any data type.
    /// \tparam index_t         Integral type of the extracted elements' indexes.
    /// \tparam T               Any data type.
    /// \param[in] extracted    1: Sequence of values that were extracted and need to be reinserted.
    ///                         2: Linear indexes in \p output where the values should be inserted.
    /// \param[out] output      Output array inside which the values are going to be inserted.
    /// \note On the GPU:
    ///         - \p value_t should be (u)int32_t, (u)int64_t, and any (complex) floating-point.
    ///         - \p index_t should be uint32_t or uint64_t.
    ///         - \p T should be equal to \p value_t.
    template<typename value_t, typename index_t, typename T>
    void insert(const Extracted<value_t, index_t>& extracted, const Array<T>& output) {
        NOA_CHECK(extracted.values.device() == extracted.indexes.device() &&
                  extracted.values.device() == output.device(),
                  "The input and output arrays should be on the same device, "
                  "but got values:{}, indexes:{} and output:{}",
                  extracted.values.device(), extracted.indexes.device(), output.device());
        NOA_CHECK(all(extracted.values.shape() == extracted.indexes.shape()) &&
                  extracted.indexes.shape().ndim() == 1,
                  "The sequence of values and indexes should be two row vectors of the same size, "
                  "but got values:{} and indexes:{}", extracted.values.shape(), extracted.indexes.shape());

        const size_t elements = extracted.values.shape()[3];
        const Device device(output.device());
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::insert({extracted.values.share(), extracted.indexes.share(), elements},
                                output.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_insert_v<value_t, index_t, T>) {
                cuda::memory::insert({extracted.values.share(), extracted.indexes.share(), elements},
                                     output.share(), stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract(...) for more details");
            }

            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
