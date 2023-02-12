#include "noa/gpu/cuda/memory/Subregion.hpp"
#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/algorithms/memory/ExtractInsert.hpp"

namespace noa::cuda::memory {
    template<typename Value, typename>
    void extract_subregions(
            const Shared<Value[]>& input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            const Shared<Value[]>& subregions, Strides4<i64> subregion_strides, Shape4<i64> subregion_shape,
            const Shared<Vec4<i64>[]>& origins, BorderMode border_mode, Value border_value,
            Stream& stream) {

        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(subregions.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(origins.get(), stream.device());
        NOA_ASSERT(noa::all(input_shape > 0) && noa::all(subregion_shape > 0));

        // Reorder the DHW dimensions to the rightmost order.
        // We'll have to reorder the origins similarly later.
        const auto order_3d = noa::indexing::order(subregion_strides.pop_front(), subregion_shape.pop_front()) + 1;
        const auto order = Vec4<i64>{0, order_3d[0], order_3d[1], order_3d[2]};
        if (noa::any(order_3d != Vec3<i64>{1, 2, 3})) {
            input_strides = noa::indexing::reorder(input_strides, order);
            input_shape = noa::indexing::reorder(input_shape, order);
            subregion_strides = noa::indexing::reorder(subregion_strides, order);
            subregion_shape = noa::indexing::reorder(subregion_shape, order);
        }

        switch (border_mode) {
            case BorderMode::NOTHING: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::NOTHING, i64, i64>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        origins.get(), order);
                noa::cuda::utils::iwise_4d("extract_subregion", subregion_shape, kernel, stream);
                break;
            }
            case BorderMode::ZERO: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::ZERO, i64, i64>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        origins.get(), order);
                noa::cuda::utils::iwise_4d("extract_subregion", subregion_shape, kernel, stream);
                break;
            }
            case BorderMode::VALUE: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::VALUE, i64, i64>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        origins.get(), order, border_value);
                noa::cuda::utils::iwise_4d("extract_subregion", subregion_shape, kernel, stream);
                break;
            }
            case BorderMode::CLAMP: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::CLAMP, i64, i64>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        origins.get(), order);
                noa::cuda::utils::iwise_4d("extract_subregion", subregion_shape, kernel, stream);
                break;
            }
            case BorderMode::MIRROR: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::MIRROR, i64, i64>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        origins.get(), order);
                noa::cuda::utils::iwise_4d("extract_subregion", subregion_shape, kernel, stream);
                break;
            }
            case BorderMode::PERIODIC: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::PERIODIC, i64, i64>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        origins.get(), order);
                noa::cuda::utils::iwise_4d("extract_subregion", subregion_shape, kernel, stream);
                break;
            }
            case BorderMode::REFLECT: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::REFLECT, i64, i64>(
                        input.get(), input_strides, input_shape, subregions.get(), subregion_strides,
                        origins.get(), order);
                noa::cuda::utils::iwise_4d("extract_subregion", subregion_shape, kernel, stream);
                break;
            }
        }
        stream.attach(input, subregions, origins);
    }

    template<typename Value, typename>
    void insert_subregions(
            const Shared<Value[]>& subregions, Strides4<i64> subregion_strides, Shape4<i64> subregion_shape,
            const Shared<Value[]>& output, Strides4<i64> output_strides, Shape4<i64> output_shape,
            const Shared<Vec4<i64>[]>& origins,
            Stream& stream) {

        NOA_ASSERT_DEVICE_PTR(subregions.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(origins.get(), stream.device());
        NOA_ASSERT(noa::all(output_shape > 0) && noa::all(subregion_shape > 0));

        // Reorder the DHW dimensions to the rightmost order.
        // We'll have to reorder the origins similarly later.
        const auto order_3d = noa::indexing::order(subregion_strides.pop_front(), subregion_shape.pop_front()) + 1;
        const auto order = Vec4<i64>{0, order_3d[0], order_3d[1], order_3d[2]};
        if (noa::any(order_3d != Vec3<i64>{1, 2, 3})) {
            output_strides = noa::indexing::reorder(output_strides, order);
            output_shape = noa::indexing::reorder(output_shape, order);
            subregion_strides = noa::indexing::reorder(subregion_strides, order);
            subregion_shape = noa::indexing::reorder(subregion_shape, order);
        }

        const auto kernel = noa::algorithm::memory::insert_subregion<i64, i64>(
                subregions.get(), subregion_strides,
                output.get(), output_strides, output_shape,
                origins.get(), order);
        noa::cuda::utils::iwise_4d("insert_subregions", subregion_shape, kernel, stream);
        stream.attach(subregions, output, origins);
    }

    #define NOA_INSTANTIATE_EXTRACT_INSERT_(T)                  \
    template void extract_subregions<T, void>(                  \
        const Shared<T[]>&, Strides4<i64>, Shape4<i64>,         \
        const Shared<T[]>&, Strides4<i64>, Shape4<i64>,         \
        const Shared<Vec4<i64>[]>&, BorderMode, T, Stream&);    \
    template void insert_subregions<T, void>(                   \
        const Shared<T[]>&, Strides4<i64>, Shape4<i64>,         \
        const Shared<T[]>&, Strides4<i64>, Shape4<i64>,         \
        const Shared<Vec4<i64>[]>&, Stream&)

    NOA_INSTANTIATE_EXTRACT_INSERT_(i8);
    NOA_INSTANTIATE_EXTRACT_INSERT_(i16);
    NOA_INSTANTIATE_EXTRACT_INSERT_(i32);
    NOA_INSTANTIATE_EXTRACT_INSERT_(i64);
    NOA_INSTANTIATE_EXTRACT_INSERT_(u8);
    NOA_INSTANTIATE_EXTRACT_INSERT_(u16);
    NOA_INSTANTIATE_EXTRACT_INSERT_(u32);
    NOA_INSTANTIATE_EXTRACT_INSERT_(u64);
    NOA_INSTANTIATE_EXTRACT_INSERT_(f16);
    NOA_INSTANTIATE_EXTRACT_INSERT_(f32);
    NOA_INSTANTIATE_EXTRACT_INSERT_(f64);
    NOA_INSTANTIATE_EXTRACT_INSERT_(c16);
    NOA_INSTANTIATE_EXTRACT_INSERT_(c32);
    NOA_INSTANTIATE_EXTRACT_INSERT_(c64);
}
