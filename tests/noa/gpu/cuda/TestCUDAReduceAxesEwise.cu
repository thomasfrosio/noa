#include <noa/gpu/cuda/ReduceAxesEwise.cuh>
#include <noa/gpu/cuda/Allocators.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

TEST_CASE("cuda::reduce_axes_ewise") {
    using namespace noa::types;
    using noa::cuda::reduce_axes_ewise;
    using noa::cuda::ReduceEwiseConfig;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;

    Stream stream(Device::current());

    SECTION("sum over one axis") {
        std::array input_shapes{Shape4<i64>{10, 34, 65, 130}, test::random_shape_batched(4)};
        for (auto& input_shape: input_shapes) {
            if (input_shape[0] == 1)
                input_shape[0] += 1; // make sure there's something to reduce
            const auto input_n_elements = input_shape.n_elements();

            const auto input_buffer = AllocatorManaged::allocate<f32>(input_n_elements, stream);
            std::fill_n(input_buffer.get(), input_n_elements, 1);

            auto input = noa::make_tuple(AccessorI64<f32, 4>(input_buffer.get(), input_shape.strides()));
            auto init = noa::make_tuple(AccessorValue<f32>(0.));
            auto reduce_op = []__device__(f32 to_reduce, f32& reduced) { reduced += to_reduce; };

            for (auto i: noa::irange(4)) {
                INFO("axis=" << i);
                auto output_shape = input_shape;
                output_shape[i] = 1;
                const auto output_n_elements = output_shape.n_elements();
                const auto output_buffer = AllocatorManaged::allocate<f32>(output_n_elements, stream);
                auto output = noa::make_tuple(AccessorI64<f32, 4>(output_buffer.get(), output_shape.strides()));

                reduce_axes_ewise(input_shape, output_shape, reduce_op, input, init, output, stream);
                stream.synchronize();

                const f32 expected_value = static_cast<f32>(input_shape[i]);
                REQUIRE(test::allclose_abs(output_buffer.get(), expected_value, output_n_elements, 1e-5));
            }
        }
    }

    SECTION("sum per batch") {
        const std::array input_shapes{Shape4<i64>{6, 8, 13, 99}, Shape4<i64>{4, 35, 55, 130}};
        for (const auto& input_shape: input_shapes) {
            const auto output_shape = Shape4<i64>{input_shape[0], 1, 1, 1};
            const auto input_n_elements = input_shape.n_elements();
            const auto output_n_elements = output_shape.n_elements();
            const f64 expected_value = static_cast<f64>(input_shape.pop_front().n_elements());

            const auto input_buffer = AllocatorManaged::allocate<f64>(input_n_elements, stream);
            const auto output_buffer = AllocatorManaged::allocate<f64>(output_n_elements, stream);
            std::fill_n(input_buffer.get(), input_n_elements, 1);

            auto output = noa::make_tuple(AccessorI64<f64, 4>(output_buffer.get(), output_shape.strides()));
            auto init = noa::make_tuple(AccessorValue<f64>(0.));
            auto reduce_op = []__device__(f64 to_reduce, f64& reduced) { reduced += to_reduce; };

            auto input_contiguous = noa::make_tuple(AccessorI64<f64, 4>(input_buffer.get(), input_shape.strides()));
            reduce_axes_ewise(input_shape, output_shape, reduce_op, input_contiguous, init, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_value, output_n_elements, 1e-5));

            const auto strided = input_shape.strides().set<2>(0);
            auto input_strided = noa::make_tuple(AccessorI64<f64, 4>(input_buffer.get(), strided));
            reduce_axes_ewise(input_shape, output_shape, reduce_op, input_strided, init, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_value, output_n_elements, 1e-5));
        }
    }
}
