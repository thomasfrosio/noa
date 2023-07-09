#include "noa/gpu/cuda/memory/Subregion.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/algorithms/memory/Subregion.hpp"

namespace noa::cuda::memory {
    template<typename Value, typename Index, typename>
    void extract_subregions(
            const Value* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            Value* subregions, Strides4<i64> subregion_strides, Shape4<i64> subregion_shape,
            const Vec4<Index>* origins, BorderMode border_mode, Value border_value,
            Stream& stream
    ) {
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(subregions, stream.device());
        NOA_ASSERT_DEVICE_PTR(origins, stream.device());
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
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order);
                noa::cuda::utils::iwise_4d(subregion_shape, kernel, stream);
                break;
            }
            case BorderMode::ZERO: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::ZERO, i64, i64>(
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order);
                noa::cuda::utils::iwise_4d(subregion_shape, kernel, stream);
                break;
            }
            case BorderMode::VALUE: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::VALUE, i64, i64>(
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order, border_value);
                noa::cuda::utils::iwise_4d(subregion_shape, kernel, stream);
                break;
            }
            case BorderMode::CLAMP: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::CLAMP, i64, i64>(
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order);
                noa::cuda::utils::iwise_4d(subregion_shape, kernel, stream);
                break;
            }
            case BorderMode::MIRROR: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::MIRROR, i64, i64>(
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order);
                noa::cuda::utils::iwise_4d(subregion_shape, kernel, stream);
                break;
            }
            case BorderMode::PERIODIC: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::PERIODIC, i64, i64>(
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order);
                noa::cuda::utils::iwise_4d(subregion_shape, kernel, stream);
                break;
            }
            case BorderMode::REFLECT: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::REFLECT, i64, i64>(
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order);
                noa::cuda::utils::iwise_4d(subregion_shape, kernel, stream);
                break;
            }
        }
    }

    template<typename Value, typename Index, typename>
    void insert_subregions(
            const Value* subregions, Strides4<i64> subregion_strides, Shape4<i64> subregion_shape,
            Value* output, Strides4<i64> output_strides, Shape4<i64> output_shape,
            const Vec4<Index>* origins,
            Stream& stream
    ) {
        NOA_ASSERT_DEVICE_PTR(subregions, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT_DEVICE_PTR(origins, stream.device());
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
                subregions, subregion_strides,
                output, output_strides, output_shape,
                origins, order);
        noa::cuda::utils::iwise_4d(subregion_shape, kernel, stream);
    }

    #define NOA_INSTANTIATE_EXTRACT_INSERT_(T, I)   \
    template void extract_subregions<T, I, void>(   \
        const T*, Strides4<i64>, Shape4<i64>,       \
        T*, Strides4<i64>, Shape4<i64>,             \
        const Vec4<I>*, BorderMode, T, Stream&);    \
    template void insert_subregions<T, I, void>(    \
        const T*, Strides4<i64>, Shape4<i64>,       \
        T*, Strides4<i64>, Shape4<i64>,             \
        const Vec4<I>*, Stream&)

    #define NOA_INSTANTIATE_EXTRACT_INSERT_ALL(T)   \
    NOA_INSTANTIATE_EXTRACT_INSERT_(T, i32);        \
    NOA_INSTANTIATE_EXTRACT_INSERT_(T, i64)

    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(i8);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(i16);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(i32);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(i64);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(u8);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(u16);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(u32);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(u64);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(f16);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(f32);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(f64);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(c16);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(c32);
    NOA_INSTANTIATE_EXTRACT_INSERT_ALL(c64);
}
