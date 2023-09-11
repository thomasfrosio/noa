#include "noa/gpu/cuda/geometry/fft/Project.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/algorithms/geometry/FourierGriddingCorrection.hpp"

namespace noa::cuda::geometry::fft {
    template<typename Value, typename>
    void gridding_correction(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, bool post_correction, Stream& stream
    ) {
        const auto i_shape = shape.as_safe<u32>();
        const auto input_accessor = Accessor<const Value, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = Accessor<Value, 4, u32>(output, output_strides.as_safe<u32>());

        if (post_correction) {
            const auto op = noa::geometry::gridding_correction_op<true>(
                    input_accessor, output_accessor, i_shape);
            noa::cuda::utils::iwise_4d(i_shape, op, stream);
        } else {
            const auto op = noa::geometry::gridding_correction_op<false>(
                    input_accessor, output_accessor, i_shape);
            noa::cuda::utils::iwise_4d(i_shape, op, stream);
        }
    }
    template void gridding_correction<f32, void>(const f32*, const Strides4<i64>&, f32*, const Strides4<i64>&, const Shape4<i64>&, bool, Stream&);
    template void gridding_correction<f64, void>(const f64*, const Strides4<i64>&, f64*, const Strides4<i64>&, const Shape4<i64>&, bool, Stream&);
}
