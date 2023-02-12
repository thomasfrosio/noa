#include <cub/device/device_scan.cuh>

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Index.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/utils/EwiseUnary.cuh"
#include "noa/gpu/cuda/utils/EwiseBinary.cuh"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace {
    using namespace ::noa;

    template<typename Value, typename Offset>
    struct ExtractFromMap1D {
        constexpr ExtractFromMap1D(
                const Value* input,
                const i32* binary_map,
                const i32* inclusive_sum,
                Value* sequence_values,
                Offset* sequence_offsets)
                : m_input(input),
                  m_binary_map(binary_map),
                  m_inclusive_sum(inclusive_sum),
                  m_sequence_elements(sequence_values),
                  m_sequence_offsets(sequence_offsets) {}

        __device__ constexpr void operator()(i32 gid) const noexcept {
            if (m_binary_map[gid]) {
                const auto offset = m_inclusive_sum[gid] - 1; // inclusive sum starts at 1
                if (m_sequence_elements)
                    m_sequence_elements[offset] = m_input[gid];
                if (m_sequence_offsets)
                    m_sequence_offsets[offset] = static_cast<Offset>(gid);
            }
        }

    private:
        AccessorRestrictContiguous<const Value, 1, i32> m_input;
        AccessorRestrictContiguous<const i32, 1, i32> m_binary_map;
        AccessorRestrictContiguous<const i32, 1, i32> m_inclusive_sum;
        AccessorRestrictContiguous<Value, 1, i32> m_sequence_elements;
        AccessorRestrictContiguous<Offset, 1, i32> m_sequence_offsets;
    };

    template<typename Value, typename Offset>
    struct ExtractFromMap4D {
        constexpr ExtractFromMap4D(
                const Value* input,
                const Strides4<i32>& input_strides,
                const Shape4<i32>& input_shape,
                const i32* binary_map,
                const i32* inclusive_sum,
                Value* sequence_values,
                Offset* sequence_offsets)
                : m_input(input, input_strides),
                  m_binary_map(binary_map),
                  m_inclusive_sum(inclusive_sum),
                  m_sequence_elements(sequence_values),
                  m_sequence_offsets(sequence_offsets),
                  m_contiguous_strides(input_shape.strides()) {}

        __device__ constexpr void operator()(i32 ii, i32 ij, i32 ik, i32 il) const noexcept {
            const i32 gid = noa::indexing::at(ii, ij, ik, il, m_contiguous_strides);
            if (m_binary_map[gid]) {
                const auto offset = m_inclusive_sum[gid] - 1; // inclusive sum will start at 1
                const auto input_offset = noa::indexing::at(ii, ij, ik, il, m_input.strides());
                if (m_sequence_elements)
                    m_sequence_elements[offset] = m_input.get()[input_offset];
                if (m_sequence_offsets)
                    m_sequence_offsets[offset] = static_cast<Offset>(input_offset);
            }
        }

    private:
        AccessorRestrict<const Value, 4, i32> m_input;
        AccessorRestrictContiguous<const i32, 1, i32> m_binary_map;
        AccessorRestrictContiguous<const i32, 1, i32> m_inclusive_sum;
        AccessorRestrictContiguous<Value, 1, i32> m_sequence_elements;
        AccessorRestrictContiguous<Offset, 1, i32> m_sequence_offsets;
        Strides4<i32> m_contiguous_strides;
    };

    // TODO I'm sure there's a much better way of doing this but that's the approach I came up with.
    //      I checked PyTorch, but couldn't figure out how they know how many elements/indexes need to be extracted.
    //      One easy optimization is to merge in the same kernel the inclusive scan and the unary_op transformation
    //      using cub::BlockScan... Benchmark to compare with the CPU backend, because maybe transferring back and
    //      forth to the host is faster (it will use less memory that's for sure).
    template<typename T, typename I>
    cuda::memory::Extracted<T, I> extract_elements_(
            const T* input, const Strides4<i32>& strides,
            const Shape4<i32>& shape, i32 elements, const i32* binary_map,
            bool extract_values, bool extract_offsets, cuda::Stream& stream) {

        // Inclusive scan sum:
        // 1) to get the number of elements to extract.
        // 2) to know the order of the elements to extract.
        i32 elements_to_extract{};
        const auto inclusive_sum = noa::cuda::memory::PtrDevice<i32>::alloc(elements, stream);
        {
            // Include cuda namespace for NOA_THROW_IF to properly format cudaError_t
            using namespace noa::cuda;

            // Get how much buffer it needs.
            size_t cub_buffer_size{};
            NOA_THROW_IF(cub::DeviceScan::InclusiveSum(
                    nullptr, cub_buffer_size, binary_map,
                    inclusive_sum.get(), elements, stream.id()));

            // Allocate necessary buffer.
            const auto cub_buffer = noa::cuda::memory::PtrDevice<Byte>::alloc(
                    static_cast<i64>(cub_buffer_size), stream);

            // Compute the inclusive scan sum.
            NOA_THROW_IF(cub::DeviceScan::InclusiveSum(
                    cub_buffer.get(), cub_buffer_size, binary_map,
                    inclusive_sum.get(), elements, stream.id()));

            // The last element of the inclusive scan sum contains the sum
            // of the binary_map, i.e. the number of elements to extract.
            noa::cuda::memory::copy(
                    inclusive_sum.get() + elements - 1,
                    &elements_to_extract,
                    1, stream);
        }

        // We cannot use elements_to_extract before that point.
        stream.synchronize();
        if (!elements_to_extract)
            return {};

        using output_t = noa::cuda::memory::Extracted<T, I>;
        output_t extracted{
            extract_values ? noa::cuda::memory::PtrDevice<T>::alloc(elements_to_extract, stream) : nullptr,
            extract_offsets ? noa::cuda::memory::PtrDevice<I>::alloc(elements_to_extract, stream) : nullptr,
            elements_to_extract
        };

        // Go through the input and extract the elements.
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        if (indexing::are_contiguous(strides, shape)) {
            auto kernel = ExtractFromMap1D<T, I>{
                    input, binary_map, inclusive_sum.get(),
                    extracted.values.get(), extracted.offsets.get()};
            noa::cuda::utils::iwise_1d("extract", elements, kernel, stream);
        } else {
            auto kernel = ExtractFromMap4D<T, I>{
                    input, strides, shape, binary_map, inclusive_sum.get(),
                    extracted.values.get(), extracted.offsets.get()};
            noa::cuda::utils::iwise_4d("extract", shape, kernel, stream);
        }
        return extracted;
    }

    template<typename Input, typename Output, typename Offset>
    struct ExtractFromOffsets {
        const Input* input;
        const Offset* offsets;
        Output* output;

        __device__ constexpr void operator()(i64 gid) const noexcept {
            const auto offset = offsets[gid];
            output[gid] = input[offset];
        }
    };

    template<typename Input, typename Output, typename Offset>
    struct InsertSequence {
        const Input* sequence_values;
        const Offset* sequence_offsets;
        Output* output;

        __device__ constexpr void operator()(i64 gid) const noexcept {
            const auto offset = sequence_offsets[gid];
            output[offset] = sequence_values[gid];
        }
    };

    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename UnaryOp>
    auto extract_unary_wrapper(
            const Shared<Input[]>& input, Strides4<i64> input_strides,
            const Shared<Lhs[]>& lhs, Strides4<i64> lhs_strides, Shape4<i64> shape,
            UnaryOp unary_op, bool extract_values, bool extract_offsets, cuda::Stream& stream)
    -> cuda::memory::Extracted<ExtractedValue, ExtractedOffset> {

        if (!extract_values && !extract_offsets)
            return {};

        const auto order = noa::indexing::order(input_strides, shape);
        if (noa::any(order != Vec4<i64>{0, 1, 2, 3})) {
            input_strides = noa::indexing::reorder(input_strides, order);
            lhs_strides = noa::indexing::reorder(lhs_strides, order);
            shape = noa::indexing::reorder(shape, order);
        }

        // Compute the binary map.
        const i64 elements = shape.elements();
        const auto binary_map = noa::cuda::memory::PtrDevice<i32>::alloc(elements, stream);
        noa::cuda::utils::ewise_unary<PointerTraits::RESTRICT>(
                "extract",
                lhs.get(), lhs_strides,
                binary_map.get(), shape.strides(),
                shape, stream, unary_op);

        // Extract the elements (values and/or offset).
        // cub is limited to i32, so safe_cast here to make sure the input shape isn't too large.
        auto extracted = extract_elements_<ExtractedValue, ExtractedOffset>(
                input.get(), input_strides.as_safe<i32>(),
                shape.as<i32>(), safe_cast<i32>(elements),
                binary_map.get(), extract_values, extract_offsets, stream);
        stream.attach(input, lhs, extracted.values, extracted.offsets);
        return extracted;
    }
}

// TODO If input is contiguous AND lhs/rhs are equal to input AND the offsets are not extracted,
//      cud::DeviceSelect::If could be used instead.
namespace noa::cuda::memory {
    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename UnaryOp, typename>
    auto extract_unary(
            const Shared<Input[]>& input, const Strides4<i64>& input_strides,
            const Shared<Lhs[]>& lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& shape,
            UnaryOp unary_op, bool extract_values, bool extract_offsets, Stream& stream)
    -> Extracted<ExtractedValue, ExtractedOffset> {
        return extract_unary_wrapper<ExtractedValue, ExtractedOffset>(
                input, input_strides, lhs, lhs_strides, shape, unary_op, extract_values, extract_offsets, stream
        );
    }

    #define INSTANTIATE_EXTRACT_UNARY_BASE_(T, O)                               \
    template Extracted<T, O> extract_unary<T,O,T,T,::noa::logical_not_t,void>(  \
        const Shared<T[]>&, const Strides4<i64>&,                               \
        const Shared<T[]>&, const Strides4<i64>&,                               \
        const Shape4<i64>&, ::noa::logical_not_t, bool, bool, Stream&)

    #define INSTANTIATE_EXTRACT_UNARY_(T)       \
    INSTANTIATE_EXTRACT_UNARY_BASE_(T, i32);    \
    INSTANTIATE_EXTRACT_UNARY_BASE_(T, i64);    \
    INSTANTIATE_EXTRACT_UNARY_BASE_(T, u32);    \
    INSTANTIATE_EXTRACT_UNARY_BASE_(T, u64)

    INSTANTIATE_EXTRACT_UNARY_(i32);
    INSTANTIATE_EXTRACT_UNARY_(i64);
    INSTANTIATE_EXTRACT_UNARY_(u32);
    INSTANTIATE_EXTRACT_UNARY_(u64);
    INSTANTIATE_EXTRACT_UNARY_(f16);
    INSTANTIATE_EXTRACT_UNARY_(f32);
    INSTANTIATE_EXTRACT_UNARY_(f64);

    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename Rhs, typename BinaryOp, typename>
    auto extract_binary(
            const Shared<Input[]>& input, Strides4<i64> input_strides,
            const Shared<Lhs[]>& lhs, Strides4<i64> lhs_strides,
            const Shared<Rhs[]>& rhs, Strides4<i64> rhs_strides, Shape4<i64> shape,
            BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream)
    -> Extracted<ExtractedValue, ExtractedOffset> {

        if (!extract_values && !extract_offsets)
            return {};

        const auto order = noa::indexing::order(input_strides, shape);
        if (noa::any(order != Vec4<i64>{0, 1, 2, 3})) {
            input_strides = noa::indexing::reorder(input_strides, order);
            lhs_strides = noa::indexing::reorder(lhs_strides, order);
            rhs_strides = noa::indexing::reorder(rhs_strides, order);
            shape = noa::indexing::reorder(shape, order);
        }

        const i64 elements = shape.elements();
        const auto binary_map = noa::cuda::memory::PtrDevice<i32>::alloc(elements, stream);
        noa::cuda::utils::ewise_binary<PointerTraits::RESTRICT>(
                "extract",
                lhs.get(), lhs_strides,
                rhs.get(), rhs_strides,
                binary_map.get(), shape.strides(),
                shape, stream, binary_op);

        // Extract the elements (values and/or offset).
        // cub is limited to i32, so safe_cast here to make sure the input shape isn't too large.
        auto extracted = extract_elements_<ExtractedValue, ExtractedOffset>(
                input.get(), input_strides.as_safe<i32>(),
                shape.as<i32>(), safe_cast<i32>(elements),
                binary_map.get(), extract_values, extract_offsets, stream);
        stream.attach(input, lhs, rhs, extracted.values, extracted.offsets);
        return extracted;
    }

    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename Rhs, typename BinaryOp, typename _>
    auto extract_binary(
            const Shared<Input[]>& input, const Strides4<i64>& input_strides,
            const Shared<Lhs[]>& lhs, const Strides4<i64>& lhs_strides, Rhs rhs,
            const Shape4<i64>& shape, BinaryOp binary_op,
            bool extract_values, bool extract_offsets, Stream& stream)
    -> Extracted<ExtractedValue, ExtractedOffset> {
        auto unary_op = [=]__device__(Lhs lhs_value) {
            return binary_op(lhs_value, rhs);
        };
        return extract_unary_wrapper<ExtractedValue, ExtractedOffset>(
                input, input_strides, lhs, lhs_strides, shape,
                unary_op, extract_values, extract_offsets, stream);
    }

    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename Rhs, typename BinaryOp, typename _>
    auto extract_binary(
            const Shared<Input[]>& input, const Strides4<i64>& input_strides,
            Lhs lhs, const Shared<Rhs[]>& rhs, const Strides4<i64>& rhs_strides,
            const Shape4<i64>& shape, BinaryOp binary_op,
            bool extract_values, bool extract_offsets, Stream& stream)
    -> Extracted<ExtractedValue, ExtractedOffset> {
        auto unary_op = [=]__device__(Rhs rhs_value) {
            return binary_op(lhs, rhs_value);
        };
        return extract_unary_wrapper<ExtractedValue, ExtractedOffset>(
                input, input_strides, rhs, rhs_strides, shape,
                unary_op, extract_values, extract_offsets, stream);
    }

    #define INSTANTIATE_EXTRACT_BINARY_BASE0_(T, I, BINARY)         \
    template Extracted<T,I> extract_binary<T,I,T,T,T,BINARY,void>(  \
        const Shared<T[]>&, Strides4<i64>,                          \
        const Shared<T[]>&, Strides4<i64>,                          \
        const Shared<T[]>&, Strides4<i64>,                          \
        Shape4<i64>, BINARY, bool, bool, Stream&);                  \
    template Extracted<T,I> extract_binary<T,I,T,T,T,BINARY,void>(  \
        const Shared<T[]>&, const Strides4<i64>&,                   \
        const Shared<T[]>&, const Strides4<i64>&,                   \
        T,                                                          \
        const Shape4<i64>&, BINARY, bool, bool, Stream&);           \
    template Extracted<T,I> extract_binary<T,I,T,T,T,BINARY,void>(  \
        const Shared<T[]>&, const Strides4<i64>&,                   \
        T,                                                          \
        const Shared<T[]>&, const Strides4<i64>&,                   \
        const Shape4<i64>&, BINARY, bool, bool, Stream&)

    #define INSTANTIATE_EXTRACT_BINARY_BASE2_(T, I)             \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,I,::noa::equal_t);      \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,I,::noa::not_equal_t);  \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,I,::noa::less_t);       \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,I,::noa::less_equal_t); \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,I,::noa::greater_t);    \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,I,::noa::greater_equal_t)

    #define INSTANTIATE_EXTRACT_BINARY_(T)      \
    INSTANTIATE_EXTRACT_BINARY_BASE2_(T, i32);  \
    INSTANTIATE_EXTRACT_BINARY_BASE2_(T, i64);  \
    INSTANTIATE_EXTRACT_BINARY_BASE2_(T, u32);  \
    INSTANTIATE_EXTRACT_BINARY_BASE2_(T, u64)

    INSTANTIATE_EXTRACT_BINARY_(i32);
    INSTANTIATE_EXTRACT_BINARY_(i64);
    INSTANTIATE_EXTRACT_BINARY_(u32);
    INSTANTIATE_EXTRACT_BINARY_(u64);
    INSTANTIATE_EXTRACT_BINARY_(f16);
    INSTANTIATE_EXTRACT_BINARY_(f32);
    INSTANTIATE_EXTRACT_BINARY_(f64);

    template<typename Input, typename Offset, typename Output, typename>
    void extract_elements(
            const Shared<Input[]>& input,
            const Shared<Offset[]>& offsets,
            const Shared<Output[]>& output,
            i64 elements, Stream& stream) {
        auto kernel = ExtractFromOffsets<Input, Output, Offset>{input.get(), offsets.get(), output.get()};
        noa::cuda::utils::iwise_1d("extract", elements, kernel, stream);
    }

    template<typename ExtractedValue, typename ExtractedOffset, typename Output, typename>
    void insert_elements(
            const Extracted<ExtractedValue, ExtractedOffset>& extracted,
            const Shared<Output[]>& output, Stream& stream) {
        auto kernel = InsertSequence<ExtractedValue, Output, ExtractedOffset>{
                extracted.values.get(), extracted.offsets.get(), output.get()};
        noa::cuda::utils::iwise_1d("extract", extracted.count, kernel, stream);
    }

    #define INSTANTIATE_INSERT_BASE_(T, I)                                                                                  \
    template void extract_elements<T,I,T,void>(const Shared<T[]>&, const Shared<I[]>&, const Shared<T[]>&, i64, Stream&);   \
    template void insert_elements<T,I,T,void>(const Extracted<T, I>&, const Shared<T[]>&, Stream&)

    #define INSTANTIATE_INSERT_(T)      \
    INSTANTIATE_INSERT_BASE_(T, i32);   \
    INSTANTIATE_INSERT_BASE_(T, i64);   \
    INSTANTIATE_INSERT_BASE_(T, u32);   \
    INSTANTIATE_INSERT_BASE_(T, u64)

    INSTANTIATE_INSERT_(i32);
    INSTANTIATE_INSERT_(i64);
    INSTANTIATE_INSERT_(u32);
    INSTANTIATE_INSERT_(u64);
    INSTANTIATE_INSERT_(f16);
    INSTANTIATE_INSERT_(f32);
    INSTANTIATE_INSERT_(f64);
}
