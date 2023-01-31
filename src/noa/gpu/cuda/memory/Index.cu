#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/memory/details/ExtractInsert.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/memory/Index.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace noa::cuda::memory {
    template<typename T, typename>
    void extract_subregions(
            const shared_t<T[]>& input,
            dim4_t input_strides,
            dim4_t input_shape,
            const shared_t<T[]>& subregions,
            dim4_t subregion_strides,
            dim4_t subregion_shape,
            const shared_t<int4_t[]>& origins,
            BorderMode border_mode,
            T border_value,
            Stream& stream) {

        // Reorder the DHW dimensions to the rightmost order.
        // We'll have to reorder the origins similarly later.
        NOA_ASSERT(all(input_shape > 0) && all(subregion_shape > 0));
        const dim3_t order_3d = indexing::order(dim3_t(subregion_strides.get(1)),
                                                dim3_t(subregion_shape.get(1))) + 1;
        const int4_t order(0, order_3d[0], order_3d[1], order_3d[2]);
        input_strides = indexing::reorder(input_strides, order);
        input_shape = indexing::reorder(input_shape, order);
        subregion_strides = indexing::reorder(subregion_strides, order);
        subregion_shape = indexing::reorder(subregion_shape, order);

        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(subregions.get(), stream.device());
        const shared_t<int4_t[]> d_origins = utils::ensureDeviceAccess(origins, stream, subregion_shape[0]);

        const auto iwise_shape = safe_cast<int4_t>(subregion_shape);
        switch (border_mode) {
            case BORDER_NOTHING: {
                const auto extractor = noa::memory::details::extract<BORDER_NOTHING, int32_t, uint32_t>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        d_origins.get(), order);
                noa::cuda::utils::iwise4D("cuda::memory::extractOrNothing", iwise_shape, extractor, stream);
                break;
            }
            case BORDER_ZERO: {
                const auto extractor = noa::memory::details::extract<BORDER_ZERO, int32_t, uint32_t>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        d_origins.get(), order);
                noa::cuda::utils::iwise4D("cuda::memory::extractOrZero", iwise_shape, extractor, stream);
                break;
            }
            case BORDER_VALUE: {
                const auto extractor = noa::memory::details::extract<BORDER_VALUE, int32_t, uint32_t>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        d_origins.get(), order, border_value);
                noa::cuda::utils::iwise4D("cuda::memory::extractOrValue", iwise_shape, extractor, stream);
                break;
            }
            case BORDER_CLAMP: {
                const auto extractor = noa::memory::details::extract<BORDER_CLAMP, int32_t, uint32_t>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        d_origins.get(), order);
                noa::cuda::utils::iwise4D("cuda::memory::extractOrClamp", iwise_shape, extractor, stream);
                break;
            }
            case BORDER_MIRROR: {
                const auto extractor = noa::memory::details::extract<BORDER_MIRROR, int32_t, uint32_t>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        d_origins.get(), order);
                noa::cuda::utils::iwise4D("cuda::memory::extractOrMirror", iwise_shape, extractor, stream);
                break;
            }
            case BORDER_PERIODIC: {
                const auto extractor = noa::memory::details::extract<BORDER_PERIODIC, int32_t, uint32_t>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        d_origins.get(), order);
                noa::cuda::utils::iwise4D("cuda::memory::extractOrPeriodic", iwise_shape, extractor, stream);
                break;
            }
            case BORDER_REFLECT: {
                const auto extractor = noa::memory::details::extract<BORDER_REFLECT, int32_t, uint32_t>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        d_origins.get(), order);
                noa::cuda::utils::iwise4D("cuda::memory::extractOrReflect", iwise_shape, extractor, stream);
                break;
            }
        }
        stream.attach(input, subregions, d_origins);
    }

    template<typename T, typename>
    void insert_subregion(const shared_t<T[]>& subregions, dim4_t subregion_strides, dim4_t subregion_shape,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                const shared_t<int4_t[]>& origins, Stream& stream) {
        // Reorder the DHW dimensions to the rightmost order.
        // We'll have to reorder the origins similarly later.
        NOA_ASSERT(all(output_shape > 0) && all(subregion_shape > 0));
        const dim3_t order_3d = indexing::order(dim3_t(subregion_strides.get(1)),
                                                dim3_t(subregion_shape.get(1))) + 1;
        const int4_t order(0, order_3d[0], order_3d[1], order_3d[2]);
        output_strides = indexing::reorder(output_strides, order);
        output_shape = indexing::reorder(output_shape, order);
        subregion_strides = indexing::reorder(subregion_strides, order);
        subregion_shape = indexing::reorder(subregion_shape, order);

        NOA_ASSERT_DEVICE_PTR(subregions.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const shared_t<int4_t[]> d_origins = utils::ensureDeviceAccess(origins, stream, subregion_shape[0]);
        const auto i_shape = safe_cast<int2_t>(dim2_t(subregion_shape.get(2)));
        const uint32_t blocks_x = math::divideUp(static_cast<uint32_t>(i_shape[1]), BLOCK_WORK_SIZE_2D.x);
        const uint32_t blocks_y = math::divideUp(static_cast<uint32_t>(i_shape[0]), BLOCK_WORK_SIZE_2D.y);
        const dim3 blocks(blocks_x * blocks_y, subregion_shape[1], subregion_shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> subregions_accessor(subregions.get(), safe_cast<uint4_t>(subregion_strides));
        const AccessorRestrict<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("memory::insert", insert_<T>, {blocks, BLOCK_SIZE_2D},
                       subregions_accessor, i_shape,
                       output_accessor, safe_cast<int4_t>(output_shape), d_origins.get(),
                       order, blocks_x);


        const auto insert_functor = noa::memory::details::insert<int64_t, int64_t>(
                subregions.get(), subregion_strides,
                output.get(), output_strides, output_shape,
                origins.get(), order);
        noa::cuda::utils::iwise4D("cuda::memory::insert", iwise_shape, extractor, stream);
        stream.attach(subregions, output, d_origins);
    }

    #define INSTANTIATE_EXTRACT_INSERT_(T)                                                                                                                          \
    template void extract<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<int4_t[]>&, BorderMode, T, Stream&);  \
    template void insert<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<int4_t[]>&, Stream&)

    INSTANTIATE_EXTRACT_INSERT_(bool);
    INSTANTIATE_EXTRACT_INSERT_(int8_t);
    INSTANTIATE_EXTRACT_INSERT_(int16_t);
    INSTANTIATE_EXTRACT_INSERT_(int32_t);
    INSTANTIATE_EXTRACT_INSERT_(int64_t);
    INSTANTIATE_EXTRACT_INSERT_(uint8_t);
    INSTANTIATE_EXTRACT_INSERT_(uint16_t);
    INSTANTIATE_EXTRACT_INSERT_(uint32_t);
    INSTANTIATE_EXTRACT_INSERT_(uint64_t);
    INSTANTIATE_EXTRACT_INSERT_(half_t);
    INSTANTIATE_EXTRACT_INSERT_(float);
    INSTANTIATE_EXTRACT_INSERT_(double);
    INSTANTIATE_EXTRACT_INSERT_(chalf_t);
    INSTANTIATE_EXTRACT_INSERT_(cfloat_t);
    INSTANTIATE_EXTRACT_INSERT_(cdouble_t);
}
