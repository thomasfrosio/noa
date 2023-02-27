#include "noa/core/Assert.hpp"
#include "noa/cpu/memory/Subregion.hpp"
#include "noa/cpu/utils/Iwise.hpp"
#include "noa/algorithms/memory/Subregion.hpp"

namespace noa::cpu::memory {
    template<typename Value, typename>
    void extract_subregions(
            const Value* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            Value* subregions, Strides4<i64> subregion_strides, Shape4<i64> subregion_shape,
            const Vec4<i64>* origins, BorderMode border_mode, Value border_value, i64 threads) {

        NOA_ASSERT(noa::all(input_shape > 0) && noa::all(subregion_shape > 0));
        NOA_ASSERT(input && subregions && origins);
        NOA_ASSERT(input != subregions);

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
                return noa::cpu::utils::iwise_4d(subregion_shape, kernel, threads);
            }
            case BorderMode::ZERO: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::ZERO, i64, i64>(
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order);
                return noa::cpu::utils::iwise_4d(subregion_shape, kernel, threads);
            }
            case BorderMode::VALUE: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::VALUE, i64, i64>(
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order, border_value);
                return noa::cpu::utils::iwise_4d(subregion_shape, kernel, threads);
            }
            case BorderMode::CLAMP: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::CLAMP, i64, i64>(
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order);
                return noa::cpu::utils::iwise_4d(subregion_shape, kernel, threads);
            }
            case BorderMode::MIRROR: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::MIRROR, i64, i64>(
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order);
                return noa::cpu::utils::iwise_4d(subregion_shape, kernel, threads);
            }
            case BorderMode::PERIODIC: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::PERIODIC, i64, i64>(
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order);
                return noa::cpu::utils::iwise_4d(subregion_shape, kernel, threads);
            }
            case BorderMode::REFLECT: {
                const auto kernel = noa::algorithm::memory::extract_subregion<BorderMode::REFLECT, i64, i64>(
                        input, input_strides, input_shape, subregions, subregion_strides,
                        origins, order);
                return noa::cpu::utils::iwise_4d(subregion_shape, kernel, threads);
            }
        }
    }

    template<typename Value, typename>
    void insert_subregions(
            const Value* subregions, Strides4<i64> subregion_strides, Shape4<i64> subregion_shape,
            Value* output, Strides4<i64> output_strides, Shape4<i64> output_shape,
            const Vec4<i64>* origins, i64 threads) {

        NOA_ASSERT(all(output_shape > 0) && all(subregion_shape > 0));
        NOA_ASSERT(origins && subregions && output);
        NOA_ASSERT(output != subregions);

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
        noa::cpu::utils::iwise_4d(subregion_shape, kernel, threads);
    }

    #define NOA_INSTANTIATE_EXTRACT_INSERT_(T)  \
    template void extract_subregions<T, void>(  \
        const T*, Strides4<i64>, Shape4<i64>,   \
        T*, Strides4<i64>, Shape4<i64>,         \
        const Vec4<i64>*, BorderMode, T, i64);  \
    template void insert_subregions<T, void>(   \
        const T*, Strides4<i64>, Shape4<i64>,   \
        T*, Strides4<i64>, Shape4<i64>,         \
        const Vec4<i64>*, i64)

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
