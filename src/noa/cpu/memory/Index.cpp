#include "noa/common/Assert.h"
#include "noa/common/memory/details/ExtractInsert.h"
#include "noa/cpu/memory/Index.h"
#include "noa/cpu/utils/Iwise.h"

namespace noa::cpu::memory {
    template<typename Value, typename>
    void extract_subregions(
            const shared_t<Value[]>& input,
            dim4_t input_strides,
            dim4_t input_shape,
            const shared_t<Value[]>& subregions,
            dim4_t subregion_strides,
            dim4_t subregion_shape,
            const shared_t<int4_t[]>& origins,
            BorderMode border_mode,
            Value border_value,
            Stream& stream) {

        NOA_ASSERT(all(input_shape > 0) && all(subregion_shape > 0));
        NOA_ASSERT(input && subregions && origins);
        NOA_ASSERT(input.get() != subregions.get());

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            // Reorder the DHW dimensions to the rightmost order.
            // We'll have to reorder the origins similarly later.
            const auto order_3d = noa::indexing::order(
                    dim3_t(subregion_strides.get(1)), dim3_t(subregion_shape.get(1))) + 1;
            const auto order = int4_t(0, order_3d[0], order_3d[1], order_3d[2]);
            input_strides = noa::indexing::reorder(input_strides, order);
            input_shape = noa::indexing::reorder(input_shape, order);
            subregion_strides = noa::indexing::reorder(subregion_strides, order);
            subregion_shape = noa::indexing::reorder(subregion_shape, order);

            const auto iwise_shape = safe_cast<long4_t>(subregion_shape);
            switch (border_mode) {
                case BORDER_NOTHING: {
                    const auto extract_functor = noa::memory::details::extract<BORDER_NOTHING, int64_t, int64_t>(
                            input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                            origins.get(), order);
                    noa::cpu::utils::iwise4D(iwise_shape, extract_functor, threads);
                    break;
                }
                case BORDER_ZERO: {
                    const auto extract_functor = noa::memory::details::extract<BORDER_ZERO, int64_t, int64_t>(
                            input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                            origins.get(), order);
                    noa::cpu::utils::iwise4D(iwise_shape, extract_functor, threads);
                    break;
                }
                case BORDER_VALUE: {
                    const auto extract_functor = noa::memory::details::extract<BORDER_VALUE, int64_t, int64_t>(
                            input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                            origins.get(), order, border_value);
                    noa::cpu::utils::iwise4D(iwise_shape, extract_functor, threads);
                    break;
                }
                case BORDER_CLAMP: {
                    const auto extract_functor = noa::memory::details::extract<BORDER_CLAMP, int64_t, int64_t>(
                            input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                            origins.get(), order);
                    noa::cpu::utils::iwise4D(iwise_shape, extract_functor, threads);
                    break;
                }
                case BORDER_MIRROR: {
                    const auto extract_functor = noa::memory::details::extract<BORDER_MIRROR, int64_t, int64_t>(
                            input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                            origins.get(), order);
                    noa::cpu::utils::iwise4D(iwise_shape, extract_functor, threads);
                    break;
                }
                case BORDER_PERIODIC: {
                    const auto extract_functor = noa::memory::details::extract<BORDER_PERIODIC, int64_t, int64_t>(
                            input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                            origins.get(), order);
                    noa::cpu::utils::iwise4D(iwise_shape, extract_functor, threads);
                    break;
                }
                case BORDER_REFLECT: {
                    const auto extract_functor = noa::memory::details::extract<BORDER_REFLECT, int64_t, int64_t>(
                            input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                            origins.get(), order);
                    noa::cpu::utils::iwise4D(iwise_shape, extract_functor, threads);
                    break;
                }
            }
        });
    }

    template<typename Value, typename>
    void insert_subregions(
            const shared_t<Value[]>& subregions,
            dim4_t subregion_strides,
            dim4_t subregion_shape,
            const shared_t<Value[]>& output,
            dim4_t output_strides,
            dim4_t output_shape,
            const shared_t<int4_t[]>& origins,
            Stream& stream) {

        NOA_ASSERT(all(output_shape > 0) && all(subregion_shape > 0));
        NOA_ASSERT(origins && subregions && output);
        NOA_ASSERT(output.get() != subregions.get());

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            // Reorder the DHW dimensions to the rightmost order.
            // We'll have to reorder the origins similarly later.
            const auto order_3d = indexing::order(
                    dim3_t(subregion_strides.get(1)), dim3_t(subregion_shape.get(1))) + 1;
            const auto order = int4_t(0, order_3d[0], order_3d[1], order_3d[2]);
            output_strides = indexing::reorder(output_strides, order);
            output_shape = indexing::reorder(output_shape, order);
            subregion_strides = indexing::reorder(subregion_strides, order);
            subregion_shape = indexing::reorder(subregion_shape, order);

            const auto iwise_shape = safe_cast<long4_t>(subregion_shape);
            const auto insert_functor = noa::memory::details::insert<int64_t, int64_t>(
                    subregions.get(), subregion_strides,
                    output.get(), output_strides, output_shape,
                    origins.get(), order);
            noa::cpu::utils::iwise4D(iwise_shape, insert_functor, threads);
        });
    }

    #define NOA_INSTANTIATE_EXTRACT_INSERT_(T)              \
    template void extract_subregions<T, void>(              \
        const shared_t<T[]>&, dim4_t, dim4_t,               \
        const shared_t<T[]>&, dim4_t, dim4_t,               \
        const shared_t<int4_t[]>&, BorderMode, T, Stream&); \
    template void insert_subregions<T, void>(               \
        const shared_t<T[]>&, dim4_t, dim4_t,               \
        const shared_t<T[]>&, dim4_t, dim4_t,               \
        const shared_t<int4_t[]>&, Stream&)

    NOA_INSTANTIATE_EXTRACT_INSERT_(int8_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(int16_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(int32_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(int64_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(uint8_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(uint16_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(uint32_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(uint64_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(half_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(float);
    NOA_INSTANTIATE_EXTRACT_INSERT_(double);
    NOA_INSTANTIATE_EXTRACT_INSERT_(chalf_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(cfloat_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(cdouble_t);
}
