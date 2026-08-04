#include <noa/runtime/cuda/ReduceAxesEwise.cuh>
#include <noa/runtime/cuda/Allocators.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

TEST_CASE("runtime::cuda::reduce_axes_ewise") {
    using namespace noa::types;
    using noa::cuda::reduce_axes_ewise;
    using noa::cuda::ReduceEwiseConfig;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;
    using noa::Accessor;

    auto stream = Stream(Device::current());
    const auto reduce_op = []__device__(auto to_reduce, auto& reduced) { reduced += to_reduce; };

    SECTION("sum over one axis") {
        auto input_shapes = noa::make_tuple(
            test::random_shape<isize, 1>(1),
            test::random_shape<isize, 2>(2),
            test::random_shape<isize, 3>(3),
            test::random_shape<isize, 4>(3, {.batch_range = {2, 6}}),
            test::random_shape<isize, 5>(3, {.batch_range = {2, 6}}),
            test::random_shape<isize, 6>(3, {.batch_range = {2, 6}})
        );
        input_shapes.for_each([&]<usize N>(const Shape<isize, N>& input_shape) {
            INFO("input_shape=" << input_shape);
            const auto input_n_elements = input_shape.n_elements();
            const auto input_buffer = AllocatorManaged::allocate<f32>(input_n_elements, stream);
            std::fill_n(input_buffer.get(), input_n_elements, 1);

            auto input = noa::make_tuple(Accessor<f32, N>(input_buffer.get(), input_shape.strides()));
            auto init = noa::make_tuple(noa::AccessorValue<f32>(0.));

            for (auto i: noa::irange(N)) {
                INFO("axis=" << i);
                auto output_shape = input_shape;
                output_shape[i] = 1;
                const auto output_n_elements = output_shape.n_elements();
                const auto output_buffer = AllocatorManaged::allocate<f32>(output_n_elements, stream);
                auto output = noa::make_tuple(Accessor<f32, N>(output_buffer.get(), output_shape.strides()));

                reduce_axes_ewise(input_shape, output_shape, reduce_op, input, init, output, stream);
                stream.synchronize();

                const f32 expected_value = static_cast<f32>(input_shape[i]);
                REQUIRE(test::allclose_abs(output_buffer.get(), expected_value, output_n_elements, 1e-5));
            }
        });
    }

    SECTION("sum per batch") {
        auto input_shapes = noa::make_tuple(
            test::random_shape<isize, 3>(3),
            test::random_shape<isize, 4>(3, {.batch_range = {2, 6}}),
            test::random_shape<isize, 5>(3, {.batch_range = {2, 6}}),
            test::random_shape<isize, 6>(3, {.batch_range = {2, 6}})
        );
        input_shapes.for_each([&]<usize N>(const Shape<isize, N>& input_shape) {
            const auto output_shape = Shape<isize, N>::from_value(1).template set<0>(input_shape[0]);
            const auto input_n_elements = input_shape.n_elements();
            const auto output_n_elements = output_shape.n_elements();
            const f64 expected_value = static_cast<f64>(input_shape.pop_front().n_elements());

            const auto input_buffer = AllocatorManaged::allocate<f64>(input_n_elements, stream);
            const auto output_buffer = AllocatorManaged::allocate<f64>(output_n_elements, stream);
            std::fill_n(input_buffer.get(), input_n_elements, 1);

            auto output = noa::make_tuple(Accessor<f64, N>(output_buffer.get(), output_shape.strides()));
            auto init = noa::make_tuple(noa::AccessorValue<f64>(0.));

            auto input_contiguous = noa::make_tuple(Accessor<f64, N>(input_buffer.get(), input_shape.strides()));
            reduce_axes_ewise(input_shape, output_shape, reduce_op, input_contiguous, init, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_value, output_n_elements, 1e-5));
        });
    }
}
