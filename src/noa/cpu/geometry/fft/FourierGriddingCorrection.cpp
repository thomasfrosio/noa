#include "noa/cpu/geometry/fft/Project.hpp"
#include "noa/cpu/utils/Iwise.hpp"
#include "noa/algorithms/geometry/FourierGriddingCorrection.hpp"

namespace noa::cpu::geometry::fft {
    template<typename Value, typename>
    void gridding_correction(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, bool post_correction, i64 threads
    ) {
        const auto input_accessor = Accessor<const Value, 4, i64>(input, input_strides);
        const auto output_accessor = Accessor<Value, 4, i64>(output, output_strides);

        if (post_correction) {
            const auto op = noa::geometry::gridding_correction_op<true>(
                    input_accessor, output_accessor, shape);
            noa::cpu::utils::iwise_4d(shape, op, threads);
        } else {
            const auto op = noa::geometry::gridding_correction_op<false>(
                    input_accessor, output_accessor, shape);
            noa::cpu::utils::iwise_4d(shape, op, threads);
        }
    }
    template void gridding_correction<f32, void>(const f32*, const Strides4<i64>&, f32*, const Strides4<i64>&, const Shape4<i64>&, bool, i64);
    template void gridding_correction<f64, void>(const f64*, const Strides4<i64>&, f64*, const Strides4<i64>&, const Shape4<i64>&, bool, i64);
}
